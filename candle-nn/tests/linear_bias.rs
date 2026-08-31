#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "cuda")]
mod cublaslt {
    use anyhow::Result;
    use candle::{DType, Device, Tensor};
    use candle_nn::{Linear, Module};

    /// Deterministic xorshift values, uniform over `[-1, 1)`.
    fn inputs(dims: &[usize], seed: u64) -> Result<Tensor> {
        let n: usize = dims.iter().product();
        let mut s = seed;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0
            })
            .collect();
        Ok(Tensor::from_vec(data, dims, &Device::Cpu)?)
    }

    fn max_abs_err(a: &Tensor, b: &Tensor) -> Result<f64> {
        let a = a.flatten_all()?.to_dtype(DType::F64)?.to_vec1::<f64>()?;
        let b = b.flatten_all()?.to_dtype(DType::F64)?.to_vec1::<f64>()?;
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f64, f64::max))
    }

    /// `(fused error, unfused error)` against an f64 CPU reference, relative to
    /// the reference's largest magnitude.
    fn check(x_dims: &[usize], out: usize, dtype: DType) -> Result<(f64, f64)> {
        let cuda = Device::new_cuda(0)?;
        let k = x_dims[x_dims.len() - 1];
        let x = inputs(x_dims, 0x9E37_79B9_7F4A_7C15)?;
        let w = inputs(&[out, k], 0xDEAD_BEEF_CAFE_1234)?;
        let b = inputs(&[out], 0x0123_4567_89AB_CDEF)?;

        let reference = Linear::new(w.to_dtype(DType::F64)?, Some(b.to_dtype(DType::F64)?))
            .forward(&x.to_dtype(DType::F64)?)?;

        let fused = Linear::new(
            w.to_device(&cuda)?.to_dtype(dtype)?,
            Some(b.to_device(&cuda)?.to_dtype(dtype)?),
        )
        .forward(&x.to_device(&cuda)?.to_dtype(dtype)?)?
        .to_device(&Device::Cpu)?;
        let mut expected = x_dims.to_vec();
        expected[x_dims.len() - 1] = out;
        assert_eq!(fused.dims(), expected.as_slice());

        // The unfused path at the same dtype: matmul then broadcast_add, on the
        // same device, since the cpu has no bf16 matmul.
        let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
        let unfused = x
            .to_device(&cuda)?
            .to_dtype(dtype)?
            .reshape((rows, k))?
            .matmul(&w.to_device(&cuda)?.to_dtype(dtype)?.t()?)?
            .broadcast_add(&b.to_device(&cuda)?.to_dtype(dtype)?)?
            .reshape(expected.as_slice())?
            .to_device(&Device::Cpu)?;

        let scale = reference
            .abs()?
            .max_all()?
            .to_dtype(DType::F64)?
            .to_scalar::<f64>()?
            .max(f64::MIN_POSITIVE);
        Ok((
            max_abs_err(&fused, &reference)? / scale,
            max_abs_err(&unfused, &reference)? / scale,
        ))
    }

    fn report(x_dims: &[usize], out: usize, dtype: DType, bound: f64) -> Result<()> {
        let (fused, unfused) = check(x_dims, out, dtype)?;
        println!("{x_dims:?} -> {out} {dtype:?}: fused {fused:e} unfused {unfused:e}");
        assert!(fused <= bound, "fused error {fused:e} exceeds {bound:e}");
        assert!(
            fused <= unfused * 1.5 + 1e-12,
            "fused error {fused:e} is worse than the unfused {unfused:e}"
        );
        Ok(())
    }

    #[test]
    fn sdxl_shapes() -> Result<()> {
        // the GeGLU projection, which is where most of the bias traffic is.
        report(&[2, 1024, 1280], 10240, DType::F16, 4e-3)?;
        report(&[2, 4096, 640], 5120, DType::F16, 4e-3)?;
        // a 4-d contiguous input, and the attention output projection.
        report(&[2, 2, 1024, 1280], 1280, DType::F16, 4e-3)?;
        Ok(())
    }

    #[test]
    fn tail_shapes() -> Result<()> {
        // widths that are not multiples of any tile size.
        report(&[7, 333], 129, DType::F16, 8e-3)?;
        report(&[7, 333], 129, DType::BF16, 4e-2)?;
        // a plain 2-d input, and a single row.
        report(&[13, 64], 32, DType::F16, 8e-3)?;
        report(&[1, 1280], 1280, DType::F16, 4e-3)?;
        Ok(())
    }

    /// The unfused path must stay reachable and must still be correct.
    #[test]
    fn no_bias_is_untouched() -> Result<()> {
        let cuda = Device::new_cuda(0)?;
        let x = inputs(&[3, 5, 64], 0x1111_2222_3333_4444)?.to_device(&cuda)?;
        let w = inputs(&[32, 64], 0x5555_6666_7777_8888)?.to_device(&cuda)?;
        let l = Linear::new(w.clone(), None);
        let got = l.forward(&x)?;
        let want = x.reshape((15, 64))?.matmul(&w.t()?)?.reshape((3, 5, 32))?;
        assert!(max_abs_err(&got, &want)? == 0.0);
        Ok(())
    }

    /// A non-contiguous input takes the broadcast path, which has no fused
    /// version; it must still produce the same answer.
    #[test]
    fn non_contiguous_input() -> Result<()> {
        let cuda = Device::new_cuda(0)?;
        let x = inputs(&[3, 64, 5], 0x2222_3333_4444_5555)?
            .to_device(&cuda)?
            .to_dtype(DType::F16)?
            .transpose(1, 2)?;
        let w = inputs(&[32, 64], 0x6666_7777_8888_9999)?
            .to_device(&cuda)?
            .to_dtype(DType::F16)?;
        let b = inputs(&[32], 0xAAAA_BBBB_CCCC_DDDD)?
            .to_device(&cuda)?
            .to_dtype(DType::F16)?;
        let l = Linear::new(w.clone(), Some(b.clone()));
        let got = l.forward(&x)?;
        let want = l.forward(&x.contiguous()?)?;
        // the two paths round differently, but only at the f16 storage ulp.
        assert!(max_abs_err(&got, &want)? < 1e-2);
        Ok(())
    }
}
