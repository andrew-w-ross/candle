/* Equivalent PyTorch code.
import torch
from torch.nn.functional import group_norm
t = torch.tensor(
        [[[-0.3034,  0.2726, -0.9659],
          [-1.1845, -1.3236,  0.0172],
          [ 1.9507,  1.2554, -0.8625],
          [ 1.0682,  0.3604,  0.3985],
          [-0.4957, -0.4461, -0.9721],
          [ 1.5157, -0.1546, -0.5596]],

         [[-1.6698, -0.4040, -0.7927],
          [ 0.3736, -0.0975, -0.1351],
          [-0.9461,  0.5461, -0.6334],
          [-1.0919, -0.1158,  0.1213],
          [-0.9535,  0.1281,  0.4372],
          [-0.2845,  0.3488,  0.5641]]])
print(group_norm(t, num_groups=2))
print(group_norm(t, num_groups=3))
*/
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::test_utils::to_vec3_round;
use candle::{Device, Tensor};
use candle_nn::{GroupNorm, Module};

#[test]
fn group_norm() -> Result<()> {
    let device = &Device::Cpu;
    let w = Tensor::from_vec(vec![1f32; 6], 6, device)?;
    let b = Tensor::from_vec(vec![0f32; 6], 6, device)?;
    let gn2 = GroupNorm::new(w.clone(), b.clone(), 6, 2, 1e-5)?;
    let gn3 = GroupNorm::new(w, b, 6, 3, 1e-5)?;

    let input = Tensor::new(
        &[
            [
                [-0.3034f32, 0.2726, -0.9659],
                [-1.1845, -1.3236, 0.0172],
                [1.9507, 1.2554, -0.8625],
                [1.0682, 0.3604, 0.3985],
                [-0.4957, -0.4461, -0.9721],
                [1.5157, -0.1546, -0.5596],
            ],
            [
                [-1.6698, -0.4040, -0.7927],
                [0.3736, -0.0975, -0.1351],
                [-0.9461, 0.5461, -0.6334],
                [-1.0919, -0.1158, 0.1213],
                [-0.9535, 0.1281, 0.4372],
                [-0.2845, 0.3488, 0.5641],
            ],
        ],
        device,
    )?;
    assert_eq!(
        to_vec3_round(&gn2.forward(&input)?, 4)?,
        &[
            [
                [-0.1653, 0.3748, -0.7866],
                [-0.9916, -1.1220, 0.1353],
                [1.9485, 1.2965, -0.6896],
                [1.2769, 0.3628, 0.4120],
                [-0.7427, -0.6786, -1.3578],
                [1.8547, -0.3022, -0.8252]
            ],
            [
                [-1.9342, 0.0211, -0.5793],
                [1.2223, 0.4945, 0.4365],
                [-0.8163, 1.4887, -0.3333],
                [-1.7960, -0.0392, 0.3875],
                [-1.5469, 0.3998, 0.9561],
                [-0.3428, 0.7970, 1.1845]
            ]
        ]
    );
    assert_eq!(
        to_vec3_round(&gn3.forward(&input)?, 4)?,
        &[
            [
                [0.4560, 1.4014, -0.6313],
                [-0.9901, -1.2184, 0.9822],
                [1.4254, 0.6360, -1.7682],
                [0.4235, -0.3800, -0.3367],
                [-0.3890, -0.3268, -0.9862],
                [2.1325, 0.0386, -0.4691]
            ],
            [
                [-1.8797, 0.0777, -0.5234],
                [1.2802, 0.5517, 0.4935],
                [-1.0102, 1.5327, -0.4773],
                [-1.2587, 0.4047, 0.8088],
                [-1.9074, 0.1691, 0.7625],
                [-0.6230, 0.5928, 1.0061]
            ]
        ]
    );

    Ok(())
}

#[cfg(feature = "cuda")]
mod fused_cuda {
    use anyhow::Result;
    use candle::{DType, Device, Tensor};
    use candle_nn::{GroupNorm, Module};

    fn max_abs_err(a: &Tensor, b: &Tensor) -> Result<f64> {
        let a = a.flatten_all()?.to_dtype(DType::F64)?.to_vec1::<f64>()?;
        let b = b.flatten_all()?.to_dtype(DType::F64)?.to_vec1::<f64>()?;
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f64, f64::max))
    }

    fn check(dims: &[usize], num_groups: usize, dtype: DType) -> Result<(f64, f64)> {
        let cpu = Device::Cpu;
        let cuda = Device::new_cuda(0)?;
        let n_channels = dims[1];
        let eps = 1e-5;

        let x = Tensor::randn(0.5f32, 3.0f32, dims, &cpu)?;
        let w = Tensor::randn(1f32, 0.3f32, n_channels, &cpu)?;
        let b = Tensor::randn(0f32, 0.3f32, n_channels, &cpu)?;

        // f64 reference on cpu, going through the unfused implementation.
        let gn = GroupNorm::new(
            w.to_dtype(DType::F64)?,
            b.to_dtype(DType::F64)?,
            n_channels,
            num_groups,
            eps,
        )?;
        let reference = gn.forward(&x.to_dtype(DType::F64)?)?;

        // the existing unfused path, at the tested dtype, on cpu.
        let gn = GroupNorm::new(
            w.to_dtype(dtype)?,
            b.to_dtype(dtype)?,
            n_channels,
            num_groups,
            eps,
        )?;
        let unfused = gn.forward(&x.to_dtype(dtype)?)?;

        // the fused cuda kernel.
        let gn = GroupNorm::new(
            w.to_device(&cuda)?.to_dtype(dtype)?,
            b.to_device(&cuda)?.to_dtype(dtype)?,
            n_channels,
            num_groups,
            eps,
        )?;
        let fused = gn
            .forward(&x.to_device(&cuda)?.to_dtype(dtype)?)?
            .to_device(&cpu)?;

        assert_eq!(fused.dims(), dims);
        Ok((
            max_abs_err(&fused, &reference)?,
            max_abs_err(&unfused, &reference)?,
        ))
    }

    fn report(
        name: &str,
        dims: &[usize],
        num_groups: usize,
        dtype: DType,
        bound: f64,
    ) -> Result<()> {
        let (fused, unfused) = check(dims, num_groups, dtype)?;
        println!(
            "{name} {dims:?} groups={num_groups} {dtype:?}: fused {fused:e} unfused {unfused:e}"
        );
        assert!(
            fused <= bound,
            "{name}: fused error {fused:e} exceeds {bound:e}"
        );
        assert!(
            fused <= unfused * 1.5 + 1e-6,
            "{name}: fused error {fused:e} is worse than the unfused {unfused:e}"
        );
        Ok(())
    }

    #[test]
    fn sdxl_shapes() -> Result<()> {
        report("sdxl-320", &[2, 320, 128, 128], 32, DType::F16, 8e-3)?;
        report("sdxl-640", &[2, 640, 64, 64], 32, DType::F16, 8e-3)?;
        report("sdxl-1280", &[2, 1280, 32, 32], 32, DType::F16, 8e-3)?;
        Ok(())
    }

    #[test]
    fn tail_shapes() -> Result<()> {
        // hidden_size = 2178, not a multiple of the 1024-thread block, over a
        // spatial extent that is not a power of two.
        report("tail-2178", &[2, 64, 33, 33], 32, DType::F16, 8e-3)?;
        // hidden_size = 182, smaller than one block.
        report("tail-182", &[3, 64, 7, 13], 32, DType::F16, 8e-3)?;
        // rank 3, one channel per group.
        report("tail-rank3", &[2, 32, 1000], 32, DType::F16, 8e-3)?;
        Ok(())
    }

    #[test]
    fn bf16_shapes() -> Result<()> {
        report("bf16-640", &[2, 640, 64, 64], 32, DType::BF16, 7e-2)?;
        report("bf16-tail", &[2, 64, 33, 33], 32, DType::BF16, 7e-2)?;
        Ok(())
    }
}
