//! Loading a FLUX.2 transformer out of a gguf.
//!
//! The published klein ggufs store BFL's tensor names, quantize the large
//! projections and leave the modulation, embedders and final layer in bf16. Only
//! the loading differs: [`super::transformer`] runs the same forward over
//! [`Proj`] whichever way its weights arrived, so there is one implementation of
//! the arithmetic and nothing that can drift between a dense klein and a
//! quantized one.

use candle::quantized::{gguf_file::Content, GgmlDType, QMatMul, QTensor};
use candle::{DType, Device, Result, Tensor};
use candle_nn::RmsNorm;

use super::transformer::{Config, Proj};

/// FLUX.2 fixes the head dimension at 128 across every published size, so the
/// head count follows from the hidden width rather than being stored.
const HEAD_DIM: usize = 128;

/// A gguf held open, read by BFL tensor name.
pub struct Gguf<R: std::io::Read + std::io::Seek> {
    content: Content,
    reader: R,
    device: Device,
    dtype: DType,
}

impl<R: std::io::Read + std::io::Seek> Gguf<R> {
    /// `dtype` is the activation dtype; whatever the file left unquantized is
    /// cast to it.
    pub fn new(content: Content, reader: R, device: Device, dtype: DType) -> Self {
        Self {
            content,
            reader,
            device,
            dtype,
        }
    }

    fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        self.content.tensor(&mut self.reader, name, &self.device)
    }

    /// A tensor in the activation dtype, dequantizing if it is stored quantized.
    pub(crate) fn tensor(&mut self, name: &str) -> Result<Tensor> {
        self.qtensor(name)?
            .dequantize(&self.device)?
            .to_dtype(self.dtype)
    }

    /// The checkpoint decides what stays unquantized, so this reads that rather
    /// than choosing: a quantized matmul takes and returns the activation dtype,
    /// and a stored-unquantized weight is a plain tensor that must match it.
    pub(crate) fn proj(&mut self, name: &str) -> Result<Proj> {
        let weight = self.qtensor(&format!("{name}.weight"))?;
        Ok(match weight.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => Proj::Dense(candle_nn::Linear::new(
                weight
                    .dequantize(&self.device)?
                    .to_dtype(self.dtype)?
                    .contiguous()?,
                None,
            )),
            _ => Proj::Quantized(QMatMul::from_qtensor(weight)?),
        })
    }

    /// BFL names its qk norms `scale` where diffusers names them `weight`.
    pub(crate) fn qk_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        Ok(RmsNorm::new(self.tensor(&format!("{name}.scale"))?, eps))
    }

    /// The output modulation, with its halves put back the way diffusers orders
    /// them. BFL stores shift before scale; reading it in that order would swap
    /// every scale and shift in the final layer, which is a wrong image rather
    /// than an error. This must be dequantized to do, and every published klein
    /// leaves it in bf16 — a file that quantized it would need the swap moved
    /// into the forward, so refuse rather than silently mis-order it.
    pub(crate) fn norm_out(&mut self, name: &str) -> Result<Proj> {
        let weight = self.qtensor(&format!("{name}.weight"))?;
        if !matches!(
            weight.dtype(),
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        ) {
            candle::bail!("{name} is quantized; its shift/scale halves cannot be reordered")
        }
        let weight = weight.dequantize(&self.device)?.to_dtype(self.dtype)?;
        let half = weight.dim(0)? / 2;
        let swapped = Tensor::cat(
            &[weight.narrow(0, half, half)?, weight.narrow(0, 0, half)?],
            0,
        )?;
        Ok(Proj::Dense(candle_nn::Linear::new(swapped.contiguous()?, None)))
    }

    fn dims(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self
            .content
            .tensor_infos
            .get(name)
            .ok_or_else(|| candle::Error::debug(format!("{name} is not in this gguf")))?
            .shape
            .dims()
            .to_vec())
    }

    /// The configuration the weights themselves describe. A gguf carries no
    /// diffusers `config.json`, and its declared `general.architecture` is
    /// merely `flux` for every FLUX generation, so every dimension here is read
    /// off a shape and none is taken on the file's word.
    pub fn config(&self) -> Result<Config> {
        let img_in = self.dims("img_in.weight")?;
        let txt_in = self.dims("txt_in.weight")?;
        let mlp = self.dims("double_blocks.0.img_mlp.0.weight")?;
        let (inner, in_channels) = (img_in[0], img_in[1]);
        if inner % HEAD_DIM != 0 || txt_in[0] != inner {
            candle::bail!("img_in is {img_in:?} and txt_in {txt_in:?}: not a flux.2 transformer")
        }
        let count = |prefix: &str| {
            (0..)
                .take_while(|index| {
                    self.content
                        .tensor_infos
                        .contains_key(&format!("{prefix}.{index}.{}", "img_attn.qkv.weight"))
                        || self
                            .content
                            .tensor_infos
                            .contains_key(&format!("{prefix}.{index}.linear1.weight"))
                })
                .count()
        };
        Ok(Config {
            in_channels,
            out_channels: None,
            num_layers: count("double_blocks"),
            num_single_layers: count("single_blocks"),
            attention_head_dim: HEAD_DIM,
            num_attention_heads: inner / HEAD_DIM,
            joint_attention_dim: txt_in[1],
            timestep_guidance_channels: self.dims("time_in.in_layer.weight")?[1],
            // `img_mlp.0` emits the SwiGLU gate and value together, so the
            // hidden width is half its rows.
            mlp_ratio: (mlp[0] / 2) as f64 / inner as f64,
            axes_dims_rope: vec![HEAD_DIM / 4; 4],
            rope_theta: 2000.0,
            eps: 1e-6,
            guidance_embeds: self
                .content
                .tensor_infos
                .contains_key("guidance_in.in_layer.weight"),
            use_accelerated_attn: true,
        })
    }
}

/// Whether `path` is a FLUX.2 transformer, judged on shapes.
///
/// Never on `general.architecture`: the klein ggufs declare `flux`, which is
/// what FLUX.1 declares too. The weights cannot disagree with themselves.
pub fn is_flux2_gguf<P: AsRef<std::path::Path>>(path: P) -> bool {
    let Ok(mut file) = std::fs::File::open(path.as_ref()) else {
        return false;
    };
    let Ok(content) = Content::read(&mut file) else {
        return false;
    };
    let has = |name: &str| content.tensor_infos.contains_key(name);
    let joint = content
        .tensor_infos
        .get("txt_in.weight")
        .map(|info| info.shape.dims().to_vec());
    // Four axes of 32 make FLUX.2's 128-wide head, and the conditioning is three
    // stacked Qwen3 states; FLUX.1 has neither a `txt_in` of that width nor a
    // `double_stream_modulation_img`, which is FLUX.2's shared modulation.
    has("double_blocks.0.img_attn.qkv.weight")
        && has("single_blocks.0.linear1.weight")
        && has("double_stream_modulation_img.lin.weight")
        && joint.is_some_and(|dims| dims.len() == 2 && dims[1] % 3 == 0)
}
