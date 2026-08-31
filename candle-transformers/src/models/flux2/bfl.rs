//! Reading a FLUX.2 checkpoint that uses Black Forest Labs' own tensor naming.
//!
//! The gguf exports and the single-file fp8 releases both ship this convention;
//! only the diffusers repos use the names [`super::transformer`] is written
//! against. Rather than teach the model two vocabularies, this is a
//! [`VarBuilder`] backend that translates on read, the way the stable diffusion
//! checkpoint loader already does for LDM.
//!
//! Three of the differences are not renames:
//!
//! - the attention projections are one fused `qkv` tensor, chunked in q, k, v
//!   row order;
//! - `final_layer.adaLN_modulation.1` stores shift before scale, where
//!   diffusers stores scale before shift, so its halves are swapped;
//! - an fp8 tensor is stored beside a scalar `weight_scale` it must be
//!   multiplied by. Any `input_scale` is for activation quantisation and is
//!   ignored.
//!
//! All three are settled by `scripts/convert_flux2_to_diffusers.py`, not guessed.

use std::collections::HashSet;
use std::path::Path;

use candle::safetensors::MmapedSafetensors;
use candle::{DType, Device, Result, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;

/// Names that differ only in spelling.
const TOP: &[(&str, &str)] = &[
    ("x_embedder.weight", "img_in.weight"),
    ("context_embedder.weight", "txt_in.weight"),
    (
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_in.in_layer.weight",
    ),
    (
        "time_guidance_embed.timestep_embedder.linear_2.weight",
        "time_in.out_layer.weight",
    ),
    (
        "time_guidance_embed.guidance_embedder.linear_1.weight",
        "guidance_in.in_layer.weight",
    ),
    (
        "time_guidance_embed.guidance_embedder.linear_2.weight",
        "guidance_in.out_layer.weight",
    ),
    (
        "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_img.lin.weight",
    ),
    (
        "double_stream_modulation_txt.linear.weight",
        "double_stream_modulation_txt.lin.weight",
    ),
    (
        "single_stream_modulation.linear.weight",
        "single_stream_modulation.lin.weight",
    ),
    (
        "norm_out.linear.weight",
        "final_layer.adaLN_modulation.1.weight",
    ),
    ("proj_out.weight", "final_layer.linear.weight"),
];

const DOUBLE: &[(&str, &str)] = &[
    ("attn.to_out.0.weight", "img_attn.proj.weight"),
    ("attn.to_add_out.weight", "txt_attn.proj.weight"),
    ("attn.norm_q.weight", "img_attn.norm.query_norm.scale"),
    ("attn.norm_k.weight", "img_attn.norm.key_norm.scale"),
    ("attn.norm_added_q.weight", "txt_attn.norm.query_norm.scale"),
    ("attn.norm_added_k.weight", "txt_attn.norm.key_norm.scale"),
    ("ff.linear_in.weight", "img_mlp.0.weight"),
    ("ff.linear_out.weight", "img_mlp.2.weight"),
    ("ff_context.linear_in.weight", "txt_mlp.0.weight"),
    ("ff_context.linear_out.weight", "txt_mlp.2.weight"),
];

/// The six projections diffusers splits out of two fused tensors, with the third
/// of the rows each takes.
const DOUBLE_QKV: &[(&str, &str, usize)] = &[
    ("attn.to_q.weight", "img_attn.qkv.weight", 0),
    ("attn.to_k.weight", "img_attn.qkv.weight", 1),
    ("attn.to_v.weight", "img_attn.qkv.weight", 2),
    ("attn.add_q_proj.weight", "txt_attn.qkv.weight", 0),
    ("attn.add_k_proj.weight", "txt_attn.qkv.weight", 1),
    ("attn.add_v_proj.weight", "txt_attn.qkv.weight", 2),
];

const SINGLE: &[(&str, &str)] = &[
    ("attn.to_qkv_mlp_proj.weight", "linear1.weight"),
    ("attn.to_out.weight", "linear2.weight"),
    ("attn.norm_q.weight", "norm.query_norm.scale"),
    ("attn.norm_k.weight", "norm.key_norm.scale"),
];

/// Which rows of the named tensor a diffusers name actually wants.
#[derive(Debug, Clone, Copy)]
enum Take {
    Whole,
    /// One third of a fused `qkv`, counted in q, k, v order.
    Third(usize),
    /// Both halves, exchanged.
    SwapHalves,
}

fn lookup(table: &[(&str, &str)], key: &str) -> Option<String> {
    table
        .iter()
        .find(|(from, _)| *from == key)
        .map(|(_, to)| (*to).to_string())
}

fn translate(name: &str) -> Option<(String, Take)> {
    if let Some((index, rest)) = block_of("transformer_blocks.", name) {
        if let Some(to) = lookup(DOUBLE, rest) {
            return Some((format!("double_blocks.{index}.{to}"), Take::Whole));
        }
        let (_, to, third) = DOUBLE_QKV.iter().find(|(from, _, _)| *from == rest)?;
        return Some((format!("double_blocks.{index}.{to}"), Take::Third(*third)));
    }
    if let Some((index, rest)) = block_of("single_transformer_blocks.", name) {
        let to = lookup(SINGLE, rest)?;
        return Some((format!("single_blocks.{index}.{to}"), Take::Whole));
    }
    let take = if name == "norm_out.linear.weight" {
        Take::SwapHalves
    } else {
        Take::Whole
    };
    Some((lookup(TOP, name)?, take))
}

/// Splits `prefix{index}.{rest}` into its index and remainder.
fn block_of<'a>(prefix: &str, name: &'a str) -> Option<(&'a str, &'a str)> {
    name.strip_prefix(prefix)?.split_once('.')
}

/// How a stored name is reached from the name the model asks for.
type Translate = fn(&str) -> Option<(String, Take)>;

/// A checkpoint that stores its own names, adding a `weight_scale` beside each
/// quantized weight. ComfyUI's text encoder exports are this: transformers'
/// naming, some tensors in fp8, nothing renamed or reordered.
fn identity(name: &str) -> Option<(String, Take)> {
    Some((name.to_string(), Take::Whole))
}

struct Bfl {
    weights: MmapedSafetensors,
    names: HashSet<String>,
    translate: Translate,
}

impl Bfl {
    /// The stored weight, cast to `dtype` and scaled if the file quantised it.
    ///
    /// A stored weight in `dtype`, undoing whatever quantisation the file used.
    ///
    /// Three schemes appear, sometimes inside one file: plain, per-tensor scaled
    /// fp8 (a scalar multiply), and nvfp4, which a second scale beside the first
    /// identifies. Anything else still refuses by name rather than letting the
    /// shapes collide somewhere further in.
    fn weight(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        let Some(base) = name.strip_suffix(".weight") else {
            return self.weights.load(name, device)?.to_dtype(dtype);
        };
        let scale = format!("{base}.weight_scale");
        let global = format!("{base}.weight_scale_2");
        if self.names.contains(&global) {
            return self.nvfp4(name, base, &scale, &global, dtype, device);
        }
        let tensor = self.weights.load(name, device)?.to_dtype(dtype)?;
        if !self.names.contains(&scale) {
            return Ok(tensor);
        }
        let scale = self.weights.load(&scale, device)?;
        if scale.rank() != 0 {
            candle::bail!(
                "{name} has a {:?} weight_scale and no weight_scale_2: neither a per-tensor \
                 fp8 scale nor an nvfp4 block scale, so this cannot dequantize it",
                scale.shape()
            )
        }
        tensor.broadcast_mul(&scale.to_dtype(dtype)?)
    }

    /// Nvfp4: 4-bit values packed two to a byte, fp8 scales over blocks of
    /// sixteen along the input dim, and one f32 scale for the tensor. Two
    /// packers write this format with opposite nibble orders, so the file has to
    /// say which one it was.
    ///
    /// The unpack is cpu-only, so this is the one place weights are built on the
    /// host and moved across rather than loaded straight onto the device.
    fn nvfp4(
        &self,
        name: &str,
        base: &str,
        scale: &str,
        global: &str,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let cpu = Device::Cpu;
        let weight = self.weights.load(name, &cpu)?;
        // Which half of the byte holds element 2i is a property of the packer,
        // and the file records the packer: comfy labels every tensor it wrote,
        // ModelOpt exports carry no such key. Reading it is the difference
        // between a clean image and a plausible, badly degraded one.
        let weight = if self.names.contains(&format!("{base}.comfy_quant")) {
            swap_nibbles(&weight)?
        } else {
            weight
        };
        let block_scale = self.weights.load(scale, &cpu)?;
        let global = self.weights.load(global, &cpu)?.to_dtype(DType::F32)?;
        let global = global.to_scalar::<f32>()?;
        candle::nvfp4::dequantize(&weight, &block_scale, global, dtype)?.to_device(device)
    }
}

impl SimpleBackend for Bfl {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _init: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let Some((from, take)) = (self.translate)(name) else {
            candle::bail!("{name} has no counterpart in this checkpoint's naming")
        };
        let tensor = self.weight(&from, dtype, device)?;
        let tensor = match take {
            Take::Whole => tensor,
            Take::Third(index) => {
                let rows = tensor.dim(0)? / 3;
                tensor.narrow(0, index * rows, rows)?
            }
            Take::SwapHalves => {
                let half = tensor.dim(0)? / 2;
                Tensor::cat(
                    &[tensor.narrow(0, half, half)?, tensor.narrow(0, 0, half)?],
                    0,
                )?
            }
        };
        if tensor.shape() != &shape {
            candle::bail!(
                "{name} maps to {from}, which is {:?} where {shape:?} was wanted",
                tensor.shape()
            )
        }
        tensor.contiguous()
    }

    fn contains_tensor(&self, name: &str) -> bool {
        (self.translate)(name).is_some_and(|(from, _)| self.names.contains(&from))
    }

    /// Only reachable without a shape to check against, so the caller gets
    /// whatever the file holds.
    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        let Some((from, _)) = (self.translate)(name) else {
            candle::bail!("{name} has no counterpart in this checkpoint's naming")
        };
        self.weight(&from, dtype, device)
    }
}

/// Exchanges the two nibbles of every byte, which is the difference between
/// ModelOpt's packing order and ComfyUI's.
fn swap_nibbles(weight: &Tensor) -> Result<Tensor> {
    let bytes = weight.to_dtype(DType::F32)?;
    let high = (bytes.clone() / 16.0)?.floor()?;
    let low = (bytes - (high.clone() * 16.0)?)?;
    ((low * 16.0)? + high)?.to_dtype(DType::U8)
}

/// A [`VarBuilder`] reading `paths` under Black Forest Labs' naming.
///
/// # Safety
/// The files must not be written to while the mapping is live.
pub unsafe fn var_builder<'a, P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'a>> {
    unsafe { backend(paths, dtype, device, translate) }
}

/// A [`VarBuilder`] over a checkpoint that keeps the names the model already
/// asks for but stores some tensors as scaled fp8 — ComfyUI's text encoder
/// exports. Only the dequant is shared with the BFL path; nothing is renamed.
///
/// # Safety
/// The files must not be written to while the mapping is live.
pub unsafe fn scaled_var_builder<'a, P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'a>> {
    unsafe { backend(paths, dtype, device, identity) }
}

unsafe fn backend<'a, P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
    translate: Translate,
) -> Result<VarBuilder<'a>> {
    let weights = unsafe { MmapedSafetensors::multi(paths)? };
    let names = weights.tensors().into_iter().map(|(name, _)| name).collect();
    Ok(VarBuilder::from_backend(
        Box::new(Bfl {
            weights,
            names,
            translate,
        }),
        dtype,
        device.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name the model asks for resolves against a real checkpoint, and
    /// every tensor in that checkpoint is claimed by one — apart from the
    /// activation scales, which are for a quantisation scheme this does not run.
    #[test]
    #[ignore = "needs a bfl checkpoint; set FLUX2_SAFETENSORS"]
    fn covers_a_real_checkpoint() {
        let Ok(path) = std::env::var("FLUX2_SAFETENSORS") else {
            return;
        };
        let weights = unsafe { MmapedSafetensors::new(&path).unwrap() };
        let present: HashSet<String> = weights.tensors().into_iter().map(|(n, _)| n).collect();

        // Klein-9B: eight double blocks, twenty-four single, no guidance.
        let mut claimed = HashSet::new();
        let mut wanted = Vec::new();
        for index in 0..8 {
            for (name, _) in DOUBLE {
                wanted.push(format!("transformer_blocks.{index}.{name}"));
            }
            for (name, _, _) in DOUBLE_QKV {
                wanted.push(format!("transformer_blocks.{index}.{name}"));
            }
        }
        for index in 0..24 {
            for (name, _) in SINGLE {
                wanted.push(format!("single_transformer_blocks.{index}.{name}"));
            }
        }
        for (name, _) in TOP {
            if !name.contains("guidance_embedder") {
                wanted.push((*name).to_string());
            }
        }

        for name in &wanted {
            let (from, _) = translate(name).unwrap_or_else(|| panic!("no mapping for {name}"));
            assert!(present.contains(&from), "{name} maps to missing {from}");
            claimed.insert(from);
        }

        let unclaimed: Vec<_> = present
            .iter()
            .filter(|name| !claimed.contains(*name))
            .filter(|name| !name.ends_with("_scale"))
            .collect();
        assert!(unclaimed.is_empty(), "unread tensors: {unclaimed:?}");
    }

    #[test]
    fn translates_the_names_the_conversion_script_does() {
        let name = |n: &str| translate(n).unwrap().0;
        assert_eq!(
            name("transformer_blocks.3.attn.to_out.0.weight"),
            "double_blocks.3.img_attn.proj.weight"
        );
        assert_eq!(
            name("transformer_blocks.0.ff_context.linear_in.weight"),
            "double_blocks.0.txt_mlp.0.weight"
        );
        assert_eq!(
            name("transformer_blocks.7.attn.norm_added_k.weight"),
            "double_blocks.7.txt_attn.norm.key_norm.scale"
        );
        assert_eq!(
            name("single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight"),
            "single_blocks.19.linear1.weight"
        );
        assert_eq!(name("proj_out.weight"), "final_layer.linear.weight");
        assert!(translate("transformer_blocks.0.attn.nonsense.weight").is_none());

        // The three that are not renames.
        assert!(matches!(
            translate("transformer_blocks.1.attn.add_v_proj.weight"),
            Some((_, Take::Third(2)))
        ));
        assert!(matches!(
            translate("norm_out.linear.weight"),
            Some((_, Take::SwapHalves))
        ));
        assert_eq!(
            name("norm_out.linear.weight"),
            "final_layer.adaLN_modulation.1.weight"
        );
    }
}
