#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{DType, Device, Tensor, D};

fn unfused(xs: &Tensor) -> Result<Tensor> {
    let chunks = xs.chunk(2, D::Minus1)?;
    Ok((&chunks[0] * chunks[1].gelu()?)?)
}

/// Deterministic xorshift input, uniform over `[-6, 6)` so the gate spans both
/// the saturating and the near-zero parts of the gelu. `Device::Cpu` cannot be
/// seeded, and a one-ulp accuracy comparison needs a reproducible input.
fn inputs(dims: &[usize]) -> Result<Tensor> {
    let n: usize = dims.iter().product();
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 16_777_216.0 * 12.0 - 6.0
        })
        .collect();
    Ok(Tensor::from_vec(data, dims, &Device::Cpu)?)
}

/// `(max, mean)` absolute error.
fn abs_err(a: &Tensor, b: &Tensor) -> Result<(f64, f64)> {
    let a = a.flatten_all()?.to_dtype(DType::F64)?.to_vec1::<f64>()?;
    let b = b.flatten_all()?.to_dtype(DType::F64)?.to_vec1::<f64>()?;
    let errs = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs());
    let (max, sum) = errs.fold((0f64, 0f64), |(m, s), e| (m.max(e), s + e));
    Ok((max, sum / a.len() as f64))
}

fn max_abs_err(a: &Tensor, b: &Tensor) -> Result<f64> {
    Ok(abs_err(a, b)?.0)
}

/// Errors against an f64 CPU reference, relative to the reference's largest
/// magnitude: `(fused, unfused)`, each `(max, mean)`.
fn check(dims: &[usize], dtype: DType, device: &Device) -> Result<((f64, f64), (f64, f64))> {
    let x = inputs(dims)?;
    let reference = unfused(&x.to_dtype(DType::F64)?)?;
    let unfused_at_dtype = unfused(&x.to_dtype(dtype)?)?;
    let fused =
        candle_nn::ops::geglu(&x.to_device(device)?.to_dtype(dtype)?)?.to_device(&Device::Cpu)?;

    let mut expected = dims.to_vec();
    expected[dims.len() - 1] /= 2;
    assert_eq!(fused.dims(), expected.as_slice());

    let scale = reference
        .abs()?
        .max_all()?
        .to_dtype(DType::F64)?
        .to_scalar::<f64>()?
        .max(f64::MIN_POSITIVE);
    let rel = |(m, a): (f64, f64)| (m / scale, a / scale);
    Ok((
        rel(abs_err(&fused, &reference)?),
        rel(abs_err(&unfused_at_dtype, &reference)?),
    ))
}

/// `bound` is a relative error, so it tracks the dtype rather than the shape.
/// The max is a single element and at one ulp it is a coin flip between the two
/// paths, so accuracy is ranked on the mean.
fn report(dims: &[usize], dtype: DType, device: &Device, bound: f64) -> Result<()> {
    let ((fused_max, fused_mean), (unfused_max, unfused_mean)) = check(dims, dtype, device)?;
    println!(
        "{dims:?} {dtype:?} {device:?}: fused max {fused_max:e} mean {fused_mean:e} / \
         unfused max {unfused_max:e} mean {unfused_mean:e}"
    );
    assert!(
        fused_max <= bound,
        "fused error {fused_max:e} exceeds {bound:e}"
    );
    assert!(
        fused_mean <= unfused_mean * 1.05 + 1e-12,
        "fused mean error {fused_mean:e} is worse than the unfused {unfused_mean:e}"
    );
    Ok(())
}

#[test]
fn cpu_shapes() -> Result<()> {
    let d = Device::Cpu;
    report(&[2, 7, 8], DType::F32, &d, 4e-7)?;
    // odd width, not a multiple of any block size.
    report(&[3, 333 * 2], DType::F32, &d, 4e-7)?;
    report(&[2, 1024, 2560], DType::F16, &d, 3e-3)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sdxl_shapes() -> Result<()> {
    let d = Device::new_cuda(0)?;
    // the three SDXL feed-forward widths: proj is (dim -> 8 * dim).
    report(&[2, 1024, 320 * 8], DType::F16, &d, 3e-3)?;
    report(&[2, 1024, 640 * 8], DType::F16, &d, 3e-3)?;
    report(&[2, 1024, 1280 * 8], DType::F16, &d, 3e-3)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_tail_shapes() -> Result<()> {
    let d = Device::new_cuda(0)?;
    // 333 output columns per row: not a multiple of the launch block size, and
    // the row stride is odd so no vectorised access can be assumed.
    report(&[7, 333 * 2], DType::F16, &d, 3e-3)?;
    report(&[7, 333 * 2], DType::BF16, &d, 2e-2)?;
    report(&[7, 333 * 2], DType::F32, &d, 4e-7)?;
    report(&[7, 333 * 2], DType::F64, &d, 1e-14)?;
    // a single column, and a single row.
    report(&[5, 2], DType::F32, &d, 4e-7)?;
    report(&[1, 4096], DType::F16, &d, 3e-3)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matches_cpu_fused() -> Result<()> {
    let cuda = Device::new_cuda(0)?;
    let x = Tensor::randn(0f32, 2.0f32, (4, 129 * 2), &Device::Cpu)?;
    let on_cpu = candle_nn::ops::geglu(&x)?;
    let on_cuda = candle_nn::ops::geglu(&x.to_device(&cuda)?)?.to_device(&Device::Cpu)?;
    let scale = on_cpu.abs()?.max_all()?.to_scalar::<f32>()? as f64;
    assert!(max_abs_err(&on_cpu, &on_cuda)? <= 1e-6 * scale);
    Ok(())
}

#[test]
fn odd_width_is_rejected() {
    let x = Tensor::randn(0f32, 1.0f32, (2, 5), &Device::Cpu).unwrap();
    assert!(candle_nn::ops::geglu(&x).is_err());
}
