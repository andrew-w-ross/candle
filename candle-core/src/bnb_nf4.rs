//! bitsandbytes NF4 dequantisation.
//!
//! A `load_in_4bit` checkpoint stores one quantised weight as six safetensors
//! entries, all keyed off the name the model asks for:
//!
//! ```text
//! weight                                U8   [n/2, 1]  two nf4 codes per byte
//! weight.absmax                         U8   [n/blocksize]
//! weight.quant_map                      F32  [16]      the nf4 codebook
//! weight.nested_absmax                  F32  [blocks/nested_blocksize]
//! weight.nested_quant_map               F32  [256]
//! weight.quant_state.bitsandbytes__nf4  U8   [..]      a json blob
//! ```
//!
//! Double quantisation means the block scales are themselves quantised: an
//! `absmax` byte indexes `nested_quant_map`, which is scaled by `nested_absmax`
//! and then shifted by the blob's `nested_offset`. Every one of those numbers is
//! per-tensor, so they are read from the blob and never assumed — a dropped
//! `nested_offset` biases the whole tensor and still produces a picture.
use crate::{DType, Result, Shape, Tensor};

/// The `quant_state.bitsandbytes__nf4` blob, as the file writes it.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(try_from = "RawQuantState")]
pub struct QuantState {
    pub blocksize: usize,
    pub nested_blocksize: usize,
    pub nested_offset: f32,
    /// The dtype the weights held before quantisation.
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// No field carries a default: bitsandbytes writes all of them, and a checkpoint
/// that omits one is a format this has never been run against.
#[derive(serde::Deserialize)]
struct RawQuantState {
    quant_type: String,
    blocksize: usize,
    dtype: String,
    shape: Vec<usize>,
    nested_blocksize: usize,
    nested_dtype: String,
    nested_offset: f32,
}

impl TryFrom<RawQuantState> for QuantState {
    type Error = crate::Error;

    fn try_from(raw: RawQuantState) -> Result<Self> {
        if raw.quant_type != "nf4" {
            crate::bail!("bnb: quant_type {:?} is not nf4", raw.quant_type)
        }
        if raw.nested_dtype != "float32" {
            crate::bail!("bnb: nested_dtype {:?} is not float32", raw.nested_dtype)
        }
        let dtype = match raw.dtype.as_str() {
            "bfloat16" => DType::BF16,
            "float16" => DType::F16,
            "float32" => DType::F32,
            other => crate::bail!("bnb: unknown dtype {other:?}"),
        };
        if raw.blocksize == 0 || raw.blocksize % 2 != 0 {
            crate::bail!("bnb: blocksize {} is not a positive even number", raw.blocksize)
        }
        if raw.nested_blocksize == 0 {
            crate::bail!("bnb: nested_blocksize is zero")
        }
        Ok(Self {
            blocksize: raw.blocksize,
            nested_blocksize: raw.nested_blocksize,
            nested_offset: raw.nested_offset,
            dtype,
            shape: raw.shape,
        })
    }
}

impl QuantState {
    /// Parses the blob, refusing anything this dequantiser has not been run
    /// against rather than filling in a default.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| crate::Error::msg(format!("bnb: {e}")))
    }

    /// Parses the blob from the `U8` tensor safetensors stores it in.
    pub fn from_tensor(quant_state: &Tensor) -> Result<Self> {
        if quant_state.dtype() != DType::U8 {
            crate::bail!("bnb: quant_state must be U8, got {:?}", quant_state.dtype())
        }
        Self::from_json(&quant_state.flatten_all()?.to_vec1::<u8>()?)
    }
}

/// The six entries one bitsandbytes NF4 weight is stored as.
#[derive(Clone, Copy)]
pub struct Nf4Weight<'a> {
    pub weight: &'a Tensor,
    pub absmax: &'a Tensor,
    pub quant_map: &'a Tensor,
    pub nested_absmax: &'a Tensor,
    pub nested_quant_map: &'a Tensor,
    pub quant_state: &'a Tensor,
}

fn f32s(tensor: &Tensor, what: &str, len: Option<usize>) -> Result<Vec<f32>> {
    let vs = tensor.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    match len {
        Some(len) if vs.len() != len => {
            crate::bail!("bnb: {what} has {} entries, expected {len}", vs.len())
        }
        _ => Ok(vs),
    }
}

impl Nf4Weight<'_> {
    /// Dequantises to `dtype`, in the logical shape the blob records.
    pub fn dequantize(&self, dtype: DType) -> Result<Tensor> {
        let state = QuantState::from_tensor(self.quant_state)?;
        if self.weight.dtype() != DType::U8 {
            crate::bail!("bnb: weight must be U8, got {:?}", self.weight.dtype())
        }
        if self.absmax.dtype() != DType::U8 {
            crate::bail!("bnb: absmax must be U8, got {:?}", self.absmax.dtype())
        }
        let codes = f32s(self.quant_map, "quant_map", Some(16))?;
        let nested_codes = f32s(self.nested_quant_map, "nested_quant_map", Some(256))?;

        let n: usize = state.shape.iter().product();
        let blocks = n.div_ceil(state.blocksize);
        let absmax = self.absmax.flatten_all()?.to_vec1::<u8>()?;
        if absmax.len() != blocks {
            crate::bail!(
                "bnb: absmax has {} entries, expected {blocks} for shape {:?} at blocksize {}",
                absmax.len(),
                state.shape,
                state.blocksize
            )
        }
        let nested = f32s(
            self.nested_absmax,
            "nested_absmax",
            Some(blocks.div_ceil(state.nested_blocksize)),
        )?;

        let packed = self.weight.flatten_all()?.to_vec1::<u8>()?;
        let padded = blocks * state.blocksize;
        if packed.len() * 2 != padded {
            crate::bail!(
                "bnb: weight holds {} bytes, expected {} for shape {:?}",
                packed.len(),
                padded / 2,
                state.shape
            )
        }

        let scales: Vec<f32> = absmax
            .iter()
            .enumerate()
            .map(|(i, &code)| {
                nested_codes[code as usize] * nested[i / state.nested_blocksize]
                    + state.nested_offset
            })
            .collect();

        let mut out = vec![0f32; padded];
        for (block, chunk) in packed.chunks(state.blocksize / 2).enumerate() {
            let scale = scales[block];
            let dst = &mut out[block * state.blocksize..(block + 1) * state.blocksize];
            for (i, &byte) in chunk.iter().enumerate() {
                // Element 2i is the high nibble, 2i+1 the low: bitsandbytes
                // shifts the first of the pair up when it packs the byte.
                dst[2 * i] = codes[(byte >> 4) as usize] * scale;
                dst[2 * i + 1] = codes[(byte & 0x0f) as usize] * scale;
            }
        }
        out.truncate(n);

        let shape = Shape::from(state.shape);
        Tensor::from_vec(out, shape, self.weight.device())?.to_dtype(dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    /// The codebook this checkpoint family ships, read out of
    /// `noise_refiner.0.attention.to_q.weight.quant_map`.
    const NF4: [f32; 16] = [
        -1.0,
        -0.696_192_8,
        -0.525_073_05,
        -0.394_917_5,
        -0.284_441_38,
        -0.184_773_43,
        -0.091_050_036,
        0.0,
        0.079_580_3,
        0.160_930_2,
        0.246_112_3,
        0.337_915_24,
        0.440_709_83,
        0.562_617,
        0.722_956_84,
        1.0,
    ];

    fn blob(blocksize: usize, nested_blocksize: usize, offset: f32, shape: &[usize]) -> String {
        format!(
            r#"{{"quant_type": "nf4", "blocksize": {blocksize}, "dtype": "bfloat16",
                 "shape": {shape:?}, "nested_blocksize": {nested_blocksize},
                 "nested_dtype": "float32", "nested_offset": {offset}}}"#
        )
    }

    fn u8s(vs: &[u8]) -> Result<Tensor> {
        let len = vs.len();
        Tensor::from_vec(vs.to_vec(), len, &Device::Cpu)
    }

    fn f32t(vs: &[f32]) -> Result<Tensor> {
        let len = vs.len();
        Tensor::from_vec(vs.to_vec(), len, &Device::Cpu)
    }

    /// `quant_map[i] = i` and `nested_quant_map[i] = i`, so every output is the
    /// nibble it came from and the layout is all that is under test.
    fn identity_maps() -> Result<(Tensor, Tensor)> {
        let codes: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let nested: Vec<f32> = (0..256).map(|i| i as f32).collect();
        Ok((f32t(&codes)?, f32t(&nested)?))
    }

    fn dequant(
        bytes: &[u8],
        absmax: &[u8],
        nested_absmax: &[f32],
        state: &str,
        maps: (&Tensor, &Tensor),
    ) -> Result<Vec<f32>> {
        let weight = u8s(bytes)?;
        let absmax = u8s(absmax)?;
        let nested_absmax = f32t(nested_absmax)?;
        let quant_state = u8s(state.as_bytes())?;
        Nf4Weight {
            weight: &weight,
            absmax: &absmax,
            quant_map: maps.0,
            nested_absmax: &nested_absmax,
            nested_quant_map: maps.1,
            quant_state: &quant_state,
        }
        .dequantize(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
    }

    /// The first element of a pair lives in the high nibble. With the order
    /// reversed 0x12 would decode as [2, 1], which is the failure that loads,
    /// runs, and quietly degrades the image.
    #[test]
    fn high_nibble_holds_the_first_element() -> Result<()> {
        let (codes, nested) = identity_maps()?;
        let got = dequant(
            &[0x12, 0x34],
            &[1],
            &[1.0],
            &blob(4, 1, 0.0, &[1, 4]),
            (&codes, &nested),
        )?;
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    /// `nested_offset` is added after the nested codebook is scaled, so an
    /// absmax code of 0 leaves the offset alone as the block scale.
    #[test]
    fn nested_offset_is_the_scale_floor() -> Result<()> {
        let (codes, nested) = identity_maps()?;
        let got = dequant(
            &[0x12, 0x34],
            &[0],
            &[7.0],
            &blob(4, 1, 3.0, &[1, 4]),
            (&codes, &nested),
        )?;
        assert_eq!(got, vec![3.0, 6.0, 9.0, 12.0]);
        Ok(())
    }

    /// Two blocks under one nested block, then two nested blocks: the absmax
    /// code indexes, the nested absmax scales, and neither index is the other's.
    #[test]
    fn block_and_nested_block_boundaries() -> Result<()> {
        let (codes, nested) = identity_maps()?;
        let got = dequant(
            &[0xf0, 0xf0],
            &[2, 5],
            &[1.0, 10.0],
            &blob(2, 1, 0.0, &[4]),
            (&codes, &nested),
        )?;
        // Block 0 scale 2*1, block 1 scale 5*10.
        assert_eq!(got, vec![30.0, 0.0, 750.0, 0.0]);

        let got = dequant(
            &[0xf0, 0xf0],
            &[2, 5],
            &[1.0],
            &blob(2, 2, 0.0, &[4]),
            (&codes, &nested),
        )?;
        assert_eq!(got, vec![30.0, 0.0, 75.0, 0.0]);
        Ok(())
    }

    /// The real codebook's extremes are exactly +-1, so a block's largest
    /// magnitude comes back as its own dequantised absmax.
    #[test]
    fn codebook_extremes_reproduce_the_block_scale() -> Result<()> {
        let codes = f32t(&NF4)?;
        let nested: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let nested = f32t(&nested)?;
        let got = dequant(
            &[0x0f],
            &[4],
            &[0.5],
            &blob(2, 1, 0.25, &[2]),
            (&codes, &nested),
        )?;
        let scale = 4.0f32 * 0.5 + 0.25;
        assert_eq!(got, vec![-scale, scale]);
        Ok(())
    }

    #[test]
    fn last_block_may_be_padded() -> Result<()> {
        let (codes, nested) = identity_maps()?;
        let got = dequant(
            &[0x12, 0x34],
            &[1],
            &[1.0],
            &blob(4, 1, 0.0, &[3]),
            (&codes, &nested),
        )?;
        assert_eq!(got, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn rejects_a_foreign_quant_type() {
        let state = blob(4, 1, 0.0, &[1, 4]).replace("nf4", "fp4");
        assert!(QuantState::from_json(state.as_bytes()).is_err());
    }

    #[test]
    fn rejects_a_missing_field() {
        let state = br#"{"quant_type": "nf4", "blocksize": 64, "dtype": "bfloat16",
                         "shape": [4, 4], "nested_dtype": "float32"}"#;
        assert!(QuantState::from_json(state).is_err());
    }

    #[test]
    fn rejects_a_mismatched_absmax() -> Result<()> {
        let (codes, nested) = identity_maps()?;
        let got = dequant(
            &[0x12, 0x34],
            &[1, 1],
            &[1.0],
            &blob(4, 1, 0.0, &[1, 4]),
            (&codes, &nested),
        );
        assert!(got.is_err());
        Ok(())
    }

    #[test]
    fn rejects_a_weight_that_is_not_the_declared_shape() -> Result<()> {
        let (codes, nested) = identity_maps()?;
        let got = dequant(
            &[0x12],
            &[1],
            &[1.0],
            &blob(4, 1, 0.0, &[1, 4]),
            (&codes, &nested),
        );
        assert!(got.is_err());
        Ok(())
    }

    /// Pins the nibble order against the bf16 original the quantiser was run on.
    /// NF4 keeps roughly two significant bits, so the right order lands within a
    /// few percent of the reference while the swapped order decorrelates
    /// entirely — the gap is three orders of magnitude, not a judgement call.
    /// Set `CANDLE_BNB_NF4_SAFETENSORS` to a bnb-4bit shard and
    /// `CANDLE_BNB_NF4_REFERENCE` to the unquantised shard holding the same
    /// tensors.
    #[test]
    fn matches_the_unquantised_original() -> Result<()> {
        let (Ok(quantised), Ok(reference)) = (
            std::env::var("CANDLE_BNB_NF4_SAFETENSORS"),
            std::env::var("CANDLE_BNB_NF4_REFERENCE"),
        ) else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        let q = unsafe { crate::safetensors::MmapedSafetensors::new(quantised)? };
        let r = unsafe { crate::safetensors::MmapedSafetensors::new(reference)? };
        let reference_names: std::collections::HashSet<String> =
            r.tensors().into_iter().map(|(n, _)| n).collect();
        let bases: Vec<String> = q
            .tensors()
            .into_iter()
            .filter_map(|(n, _)| n.strip_suffix(".quant_state.bitsandbytes__nf4").map(String::from))
            .filter(|base| reference_names.contains(base))
            .collect();
        assert!(!bases.is_empty(), "no comparable nf4 tensors");

        for base in bases.iter().take(4) {
            let load = |suffix: &str| q.load(&format!("{base}{suffix}"), &cpu);
            let (weight, absmax, quant_map) = (load("")?, load(".absmax")?, load(".quant_map")?);
            let (nested_absmax, nested_quant_map) =
                (load(".nested_absmax")?, load(".nested_quant_map")?);
            let quant_state = load(".quant_state.bitsandbytes__nf4")?;
            let nf4 = Nf4Weight {
                weight: &weight,
                absmax: &absmax,
                quant_map: &quant_map,
                nested_absmax: &nested_absmax,
                nested_quant_map: &nested_quant_map,
                quant_state: &quant_state,
            };
            let got = nf4.dequantize(DType::F32)?;
            let want = r.load(base, &cpu)?.to_dtype(DType::F32)?;
            assert_eq!(got.shape(), want.shape(), "{base}");

            let norm = |t: &Tensor| -> Result<f32> { t.sqr()?.sum_all()?.to_scalar::<f32>() };
            let scale = norm(&want)?;
            let ours = norm(&(&got - &want)?)? / scale;

            let swapped = {
                let bytes = weight.to_dtype(DType::F32)?;
                let high = (bytes.clone() / 16.0)?.floor()?;
                let low = (bytes - (high.clone() * 16.0)?)?;
                ((low * 16.0)? + high)?.to_dtype(DType::U8)?
            };
            let flipped = Nf4Weight {
                weight: &swapped,
                ..nf4
            }
            .dequantize(DType::F32)?;
            let other = norm(&(&flipped - &want)?)? / scale;

            eprintln!("{base}: high-nibble-first {ours}, swapped {other}");
            assert!(
                ours < 0.02 && other > 100.0 * ours,
                "{base}: high-nibble-first {ours}, swapped {other}"
            );
        }
        Ok(())
    }
}
