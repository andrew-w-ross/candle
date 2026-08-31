//! NVFP4 dequantisation.
//!
//! An NVFP4 tensor is stored as three safetensors entries, following the
//! TensorRT-ModelOpt export convention that vLLM consumes:
//!
//! ```text
//! weight          U8       [.., k/2]     two E2M1 nibbles per byte, along the input dim
//! weight_scale    F8_E4M3  [.., k/16]    one scale per 16 logical values, along the input dim
//! weight_scale_2  F32      []            per-tensor global scale = amax / (6.0 * 448.0)
//! ```
//!
//! `weight_scale_2` is stored in the dequantisation direction and is *not*
//! reciprocated, so `value = e2m1 * weight_scale * weight_scale_2`.
use crate::{DType, Result, Tensor};

pub const BLOCK_SIZE: usize = 16;

/// The 8 magnitudes of E2M1: sign in bit 3, `exp` in bits 2..1, mantissa in bit 0.
/// E2M1 has no inf or NaN encoding, so this table is total.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[inline]
fn e2m1(nibble: u8) -> f32 {
    let v = E2M1[(nibble & 0x07) as usize];
    if nibble & 0x08 != 0 {
        -v
    } else {
        v
    }
}

/// Dequantise a packed NVFP4 tensor to `dtype`.
///
/// `weight` is `U8` with the packed input dim last, `weight_scale` is `F8E4M3`
/// with matching leading dims, and `global_scale` is the `weight_scale_2` value.
pub fn dequantize(
    weight: &Tensor,
    weight_scale: &Tensor,
    global_scale: f32,
    dtype: DType,
) -> Result<Tensor> {
    if weight.dtype() != DType::U8 {
        crate::bail!("nvfp4: weight must be U8, got {:?}", weight.dtype())
    }
    if weight_scale.dtype() != DType::F8E4M3 {
        crate::bail!(
            "nvfp4: weight_scale must be F8E4M3, got {:?}",
            weight_scale.dtype()
        )
    }
    let dims = weight.dims();
    let (&packed_k, lead) = match dims.split_last() {
        Some((last, lead)) => (last, lead),
        None => crate::bail!("nvfp4: weight must have at least one dimension"),
    };
    let k = packed_k * 2;
    if k % BLOCK_SIZE != 0 {
        crate::bail!("nvfp4: logical width {k} is not a multiple of {BLOCK_SIZE}")
    }
    let n_blocks = k / BLOCK_SIZE;
    let expected_scale: Vec<usize> = lead.iter().copied().chain([n_blocks]).collect();
    if weight_scale.dims() != expected_scale.as_slice() {
        crate::bail!(
            "nvfp4: weight_scale shape {:?} does not match expected {:?} for weight {:?}",
            weight_scale.dims(),
            expected_scale,
            dims
        )
    }

    let packed = weight.flatten_all()?.to_vec1::<u8>()?;
    let scales = weight_scale.flatten_all()?.to_vec1::<float8::F8E4M3>()?;

    let mut out = vec![0f32; packed.len() * 2];
    for (block, chunk) in packed.chunks_exact(BLOCK_SIZE / 2).enumerate() {
        let scale = f32::from(scales[block]) * global_scale;
        let dst = &mut out[block * BLOCK_SIZE..(block + 1) * BLOCK_SIZE];
        for (i, &byte) in chunk.iter().enumerate() {
            // Element 2i is the low nibble, 2i+1 the high nibble.
            dst[2 * i] = e2m1(byte & 0x0f) * scale;
            dst[2 * i + 1] = e2m1(byte >> 4) * scale;
        }
    }

    let shape: Vec<usize> = lead.iter().copied().chain([k]).collect();
    Tensor::from_vec(out, shape, weight.device())?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn scales(vs: &[f32], shape: (usize, usize)) -> Result<Tensor> {
        let vs: Vec<float8::F8E4M3> = vs.iter().map(|&v| float8::F8E4M3::from_f32(v)).collect();
        Tensor::from_vec(vs, shape, &Device::Cpu)
    }

    /// One block of 16, scale 1.0, global 1.0, so the output is the raw E2M1 table.
    /// Bytes are (low, high) pairs: 0x10 -> [E2M1[0], E2M1[1]] = [0.0, 0.5].
    #[test]
    fn table_and_nibble_order() -> Result<()> {
        let bytes: Vec<u8> = vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
        let w = Tensor::from_vec(bytes, (1, 8), &Device::Cpu)?;
        let s = scales(&[1.0], (1, 1))?;
        let got = dequantize(&w, &s, 1.0, DType::F32)?.to_vec2::<f32>()?;
        assert_eq!(
            got[0],
            vec![
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                -6.0
            ]
        );
        Ok(())
    }

    /// A wrong nibble order would swap the two halves of each pair, so pin a
    /// byte whose nibbles are distinguishable: 0x27 -> low 7 = 6.0, high 2 = 1.0.
    #[test]
    fn asymmetric_byte() -> Result<()> {
        let mut bytes = vec![0u8; 8];
        bytes[0] = 0x27;
        bytes[7] = 0x8f; // low f = -6.0, high 8 = -0.0
        let w = Tensor::from_vec(bytes, (1, 8), &Device::Cpu)?;
        let s = scales(&[1.0], (1, 1))?;
        let got = dequantize(&w, &s, 1.0, DType::F32)?.to_vec2::<f32>()?;
        assert_eq!(got[0][0], 6.0);
        assert_eq!(got[0][1], 1.0);
        assert_eq!(got[0][14], -6.0);
        assert_eq!(got[0][15], -0.0);
        Ok(())
    }

    /// Two blocks with different scales, plus a non-trivial global scale.
    /// Block 0: 6.0 * 2.0 * 0.25 = 3.0. Block 1: 6.0 * 0.5 * 0.25 = 0.75.
    /// Both scales are exactly representable in F8E4M3, so this is exact.
    #[test]
    fn block_boundary_and_global_scale() -> Result<()> {
        let mut bytes = vec![0u8; 16];
        bytes[0] = 0x07;
        bytes[8] = 0x07;
        let w = Tensor::from_vec(bytes, (1, 16), &Device::Cpu)?;
        let s = scales(&[2.0, 0.5], (1, 2))?;
        let got = dequantize(&w, &s, 0.25, DType::F32)?.to_vec2::<f32>()?;
        assert_eq!(got[0][0], 3.0);
        assert_eq!(got[0][16], 0.75);
        assert_eq!(got[0][1], 0.0);
        Ok(())
    }

    /// The largest magnitude a ModelOpt checkpoint can encode: 6 * 448 * scale_2.
    /// With scale_2 = amax/2688 this reproduces amax exactly, which is the
    /// invariant that fixes the direction of the global scale.
    #[test]
    fn largest_magnitude_round_trips_amax() -> Result<()> {
        let amax = 3.5f32;
        let scale_2 = amax / (6.0 * 448.0);
        let mut bytes = vec![0u8; 8];
        bytes[0] = 0x07;
        let w = Tensor::from_vec(bytes, (1, 8), &Device::Cpu)?;
        let s = scales(&[448.0], (1, 1))?;
        let got = dequantize(&w, &s, scale_2, DType::F32)?.to_vec2::<f32>()?;
        assert!((got[0][0] - amax).abs() < 1e-6, "got {}", got[0][0]);
        Ok(())
    }

    #[test]
    fn rows_are_independent() -> Result<()> {
        let mut bytes = vec![0u8; 16];
        bytes[0] = 0x07;
        bytes[8] = 0x07;
        let w = Tensor::from_vec(bytes, (2, 8), &Device::Cpu)?;
        let s = scales(&[1.0, 3.0], (2, 1))?;
        let got = dequantize(&w, &s, 1.0, DType::BF16)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        assert_eq!(got[0][0], 6.0);
        assert_eq!(got[1][0], 18.0);
        Ok(())
    }

    /// Checks the layout against a real ModelOpt-exported checkpoint when
    /// `CANDLE_NVFP4_SAFETENSORS` points at one. The quantiser picks
    /// `weight_scale_2 = amax / (6 * 448)`, so the block holding the tensor
    /// amax gets scale exactly 448 and its peak element exactly 6: the
    /// dequantised amax must come back as `weight_scale_2 * 2688`. Reciprocating
    /// either scale, or mistaking the block size, breaks this by orders of
    /// magnitude.
    #[test]
    fn real_checkpoint_amax() -> Result<()> {
        let Ok(path) = std::env::var("CANDLE_NVFP4_SAFETENSORS") else {
            return Ok(());
        };
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(path)? };
        let names: Vec<String> = st
            .tensors()
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| n.ends_with(".weight_scale_2"))
            .collect();
        assert!(!names.is_empty(), "no nvfp4 tensors found");
        for name in names.iter().take(4) {
            let base = name.trim_end_matches(".weight_scale_2");
            let w = st.load(&format!("{base}.weight"), &Device::Cpu)?;
            let s = st.load(&format!("{base}.weight_scale"), &Device::Cpu)?;
            let gs = st
                .load(name, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?[0];
            let out = dequantize(&w, &s, gs, DType::F32)?;
            let amax = out.abs()?.max_all()?.to_scalar::<f32>()?;
            let expected = gs * 6.0 * 448.0;
            assert!(
                (amax - expected).abs() <= 1e-4 * expected,
                "{base}: amax {amax} != {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_mismatched_scale_count() -> Result<()> {
        let w = Tensor::from_vec(vec![0u8; 16], (1, 16), &Device::Cpu)?;
        let s = scales(&[1.0], (1, 1))?;
        assert!(dequantize(&w, &s, 1.0, DType::F32).is_err());
        Ok(())
    }

    #[test]
    fn rejects_unaligned_width() -> Result<()> {
        let w = Tensor::from_vec(vec![0u8; 4], (1, 4), &Device::Cpu)?;
        let s = scales(&[1.0], (1, 1))?;
        assert!(dequantize(&w, &s, 1.0, DType::F32).is_err());
        Ok(())
    }
}
