//! FLUX.2 transformer (`Flux2Transformer2DModel`), diffusers weight naming.
//!
//! Shares FLUX.1's double-stream/single-stream shape but differs in almost every
//! detail: modulation is computed once for the whole stack rather than per block,
//! the feedforward is a SwiGLU whose gate is fused into one projection, the single
//! stream is a parallel block with attention and MLP sharing both projections,
//! RoPE runs over four axes, and nothing carries a bias.

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, RmsNorm, VarBuilder};

#[cfg(feature = "flash-attn")]
fn flash_attn(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, scale, false)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32) -> Result<Tensor> {
    candle::bail!("flash-attn feature not enabled, compile with '--features flash-attn'")
}

/// Every `serde` default here is the value diffusers' own constructor uses, not
/// klein's. They differ — those are FLUX.2-dev's numbers — but a partial config
/// has to behave the way the reference would, and `klein_4b`/`klein_9b` name the
/// presets we actually target.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default)]
    pub out_channels: Option<usize>,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_num_single_layers")]
    pub num_single_layers: usize,
    #[serde(default = "default_attention_head_dim")]
    pub attention_head_dim: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_joint_attention_dim")]
    pub joint_attention_dim: usize,
    #[serde(default = "default_timestep_guidance_channels")]
    pub timestep_guidance_channels: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "default_axes_dims_rope")]
    pub axes_dims_rope: Vec<usize>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_eps")]
    pub eps: f64,
    /// True by default because that is diffusers' own default: klein's config
    /// says `false` outright, but FLUX.2-dev omits the key entirely, and
    /// defaulting to false there would silently leave its guidance embedder
    /// unread.
    #[serde(default = "default_true")]
    pub guidance_embeds: bool,
    #[serde(default = "default_use_accelerated_attn")]
    pub use_accelerated_attn: bool,
}

fn default_in_channels() -> usize {
    128
}
fn default_num_layers() -> usize {
    8
}
fn default_num_single_layers() -> usize {
    48
}
fn default_attention_head_dim() -> usize {
    128
}
fn default_num_attention_heads() -> usize {
    48
}
fn default_joint_attention_dim() -> usize {
    15360
}
fn default_timestep_guidance_channels() -> usize {
    256
}
fn default_mlp_ratio() -> f64 {
    3.0
}
fn default_axes_dims_rope() -> Vec<usize> {
    vec![32, 32, 32, 32]
}
fn default_rope_theta() -> f64 {
    2000.0
}
fn default_eps() -> f64 {
    1e-6
}
fn default_use_accelerated_attn() -> bool {
    true
}
fn default_true() -> bool {
    true
}

impl Config {
    /// FLUX.2-klein-4B.
    pub fn klein_4b() -> Self {
        Self {
            in_channels: 128,
            out_channels: None,
            num_layers: 5,
            num_single_layers: 20,
            attention_head_dim: 128,
            num_attention_heads: 24,
            joint_attention_dim: 7680,
            timestep_guidance_channels: 256,
            mlp_ratio: 3.0,
            axes_dims_rope: vec![32, 32, 32, 32],
            rope_theta: 2000.0,
            eps: 1e-6,
            guidance_embeds: false,
            use_accelerated_attn: true,
        }
    }

    /// FLUX.2-klein-9B.
    pub fn klein_9b() -> Self {
        Self {
            joint_attention_dim: 12288,
            num_layers: 8,
            num_single_layers: 24,
            num_attention_heads: 32,
            ..Self::klein_4b()
        }
    }

    pub fn set_use_accelerated_attn(&mut self, enabled: bool) {
        self.use_accelerated_attn = enabled;
    }

    fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    fn mlp_hidden_dim(&self) -> usize {
        (self.inner_dim() as f64 * self.mlp_ratio) as usize
    }
}

// ==================== Positional embedding ====================

/// Cosines and sines for one rope axis, repeat-interleaved to `dim` so that a
/// coordinate pair shares one angle.
fn rope_axis(pos: &Tensor, dim: usize, theta: f64) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / theta.powf(2.0 * i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), pos.device())?;
    let freqs = pos.contiguous()?.reshape(((), 1))?.matmul(&inv_freq)?;
    let interleave = |xs: Tensor| -> Result<Tensor> {
        let seq = xs.dim(0)?;
        xs.unsqueeze(D::Minus1)?
            .broadcast_as((seq, half, 2))?
            .contiguous()?
            .reshape((seq, dim))
    };
    Ok((interleave(freqs.cos()?)?, interleave(freqs.sin()?)?))
}

/// Rope tables for token coordinates `ids`, shaped `(seq, axes)`. The axes are
/// concatenated along the feature dimension, so they must sum to the head dim.
///
/// The tables stay in f32: bf16 rope angles lose enough of the high-frequency
/// axes to shift fine detail across the image.
pub fn rope(ids: &Tensor, axes_dim: &[usize], theta: f64) -> Result<(Tensor, Tensor)> {
    let ids = ids.to_dtype(DType::F32)?;
    let mut cos = Vec::with_capacity(axes_dim.len());
    let mut sin = Vec::with_capacity(axes_dim.len());
    for (axis, &dim) in axes_dim.iter().enumerate() {
        let (c, s) = rope_axis(&ids.i((.., axis))?, dim, theta)?;
        cos.push(c);
        sin.push(s);
    }
    Ok((Tensor::cat(&cos, 1)?, Tensor::cat(&sin, 1)?))
}

/// Rotates `xs`, shaped `(batch, seq, heads, head_dim)`, by the rope tables.
/// Diffusers rotates in f32 whatever the model dtype, and bf16 here is visible.
fn apply_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs = xs.to_dtype(DType::F32)?;
    let (b, seq, heads, dim) = xs.dims4()?;
    let pairs = xs.reshape((b, seq, heads, dim / 2, 2))?;
    let even = pairs.narrow(D::Minus1, 0, 1)?;
    let odd = pairs.narrow(D::Minus1, 1, 1)?;
    let rotated = Tensor::cat(&[odd.neg()?, even], D::Minus1)?.reshape((b, seq, heads, dim))?;
    let cos = cos.reshape((1, seq, 1, dim))?;
    let sin = sin.reshape((1, seq, 1, dim))?;
    (xs.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?.to_dtype(dtype)
}

/// Attention over `(batch, seq, heads, head_dim)`, flattened back to
/// `(batch, seq, heads * head_dim)`. Every token attends to every other one, so
/// there is no mask to carry.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, accelerated: bool) -> Result<Tensor> {
    let (b, seq, heads, head_dim) = q.dims4()?;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let out = if accelerated && q.device().is_cuda() && cfg!(feature = "flash-attn") {
        flash_attn(&q.contiguous()?, &k.contiguous()?, &v.contiguous()?, scale as f32)?
    } else if accelerated && q.device().is_metal() {
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?.transpose(1, 2)?
    } else {
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        candle_nn::ops::softmax_last_dim(&weights)?
            .matmul(&v)?
            .transpose(1, 2)?
    };
    out.contiguous()?.reshape((b, seq, heads * head_dim))
}

// ==================== Building blocks ====================

fn layer_norm(dim: usize, eps: f64, device: &Device, dtype: DType) -> Result<LayerNorm> {
    Ok(LayerNorm::new_no_bias(
        Tensor::ones(dim, dtype, device)?,
        eps,
    ))
}

/// Every projection in the stack, without exception, is bias-free. Routing them
/// all through one constructor leaves a quantized backend one line to replace
/// rather than sixteen.
fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    candle_nn::linear_no_bias(in_dim, out_dim, vb)
}

fn rms_norm(dim: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    Ok(RmsNorm::new(vb.get(dim, "weight")?, eps))
}

/// Normalises the head dimension of a `(batch, seq, heads, head_dim)` tensor.
fn norm_heads(xs: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
    let dims = xs.dims().to_vec();
    xs.contiguous()?
        .flatten_to(D::Minus2)?
        .apply(norm)?
        .reshape(dims)
}

/// One `Flux2Modulation`: `mod_param_sets` triples of shift, scale and gate for
/// the whole stack at once, rather than FLUX.1's per-block modulation.
#[derive(Debug, Clone)]
struct Modulation {
    linear: Linear,
    sets: usize,
}

impl Modulation {
    fn new(dim: usize, sets: usize, vb: VarBuilder) -> Result<Self> {
        let linear = linear(dim, dim * 3 * sets, vb.pp("linear"))?;
        Ok(Self { linear, sets })
    }

    /// `3 * sets` tensors of shape `(batch, 1, dim)`, in shift/scale/gate order.
    fn forward(&self, temb: &Tensor) -> Result<Vec<Tensor>> {
        temb.silu()?
            .apply(&self.linear)?
            .unsqueeze(1)?
            .chunk(3 * self.sets, D::Minus1)
    }
}

fn scale_shift(xs: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    xs.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// SwiGLU feedforward whose gate projection is fused into `linear_in`, so the
/// first half of its output gates the second.
#[derive(Debug, Clone)]
struct FeedForward {
    linear_in: Linear,
    linear_out: Linear,
    inner_dim: usize,
}

impl FeedForward {
    fn new(dim: usize, inner_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_in: linear(dim, inner_dim * 2, vb.pp("linear_in"))?,
            linear_out: linear(inner_dim, dim, vb.pp("linear_out"))?,
            inner_dim,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        swiglu(&xs.apply(&self.linear_in)?, self.inner_dim)?.apply(&self.linear_out)
    }
}

fn swiglu(xs: &Tensor, half: usize) -> Result<Tensor> {
    let gate = xs.narrow(D::Minus1, 0, half)?.silu()?;
    gate.mul(&xs.narrow(D::Minus1, half, half)?)
}

/// Joint attention for a double-stream block: the text stream has its own
/// projections, the two streams are concatenated for the attention itself, then
/// split again for their separate output projections.
#[derive(Debug, Clone)]
struct JointAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    to_out: Linear,
    add_q_proj: Linear,
    add_k_proj: Linear,
    add_v_proj: Linear,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    to_add_out: Linear,
    heads: usize,
    head_dim: usize,
    accelerated: bool,
}

impl JointAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim();
        let head_dim = cfg.attention_head_dim;
        let square = |name: &str| linear(dim, dim, vb.pp(name));
        Ok(Self {
            to_q: square("to_q")?,
            to_k: square("to_k")?,
            to_v: square("to_v")?,
            norm_q: rms_norm(head_dim, cfg.eps, vb.pp("norm_q"))?,
            norm_k: rms_norm(head_dim, cfg.eps, vb.pp("norm_k"))?,
            to_out: linear(dim, dim, vb.pp("to_out").pp(0))?,
            add_q_proj: square("add_q_proj")?,
            add_k_proj: square("add_k_proj")?,
            add_v_proj: square("add_v_proj")?,
            norm_added_q: rms_norm(head_dim, cfg.eps, vb.pp("norm_added_q"))?,
            norm_added_k: rms_norm(head_dim, cfg.eps, vb.pp("norm_added_k"))?,
            to_add_out: square("to_add_out")?,
            heads: cfg.num_attention_heads,
            head_dim,
            accelerated: cfg.use_accelerated_attn,
        })
    }

    fn heads_of(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;
        xs.reshape((b, seq, self.heads, self.head_dim))
    }

    /// Returns `(image, text)` attention outputs, each already projected out.
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let q = norm_heads(&self.heads_of(&img.apply(&self.to_q)?)?, &self.norm_q)?;
        let k = norm_heads(&self.heads_of(&img.apply(&self.to_k)?)?, &self.norm_k)?;
        let v = self.heads_of(&img.apply(&self.to_v)?)?;

        let tq = norm_heads(&self.heads_of(&txt.apply(&self.add_q_proj)?)?, &self.norm_added_q)?;
        let tk = norm_heads(&self.heads_of(&txt.apply(&self.add_k_proj)?)?, &self.norm_added_k)?;
        let tv = self.heads_of(&txt.apply(&self.add_v_proj)?)?;

        let q = apply_rope(&Tensor::cat(&[&tq, &q], 1)?, cos, sin)?;
        let k = apply_rope(&Tensor::cat(&[&tk, &k], 1)?, cos, sin)?;
        let v = Tensor::cat(&[&tv, &v], 1)?;

        let out = attention(&q, &k, &v, self.accelerated)?;
        let txt_len = txt.dim(1)?;
        let img_out = out.i((.., txt_len.., ..))?.apply(&self.to_out)?;
        let txt_out = out.i((.., ..txt_len, ..))?.apply(&self.to_add_out)?;
        Ok((img_out, txt_out))
    }
}

#[derive(Debug, Clone)]
struct DoubleStreamBlock {
    norm1: LayerNorm,
    norm1_context: LayerNorm,
    attn: JointAttention,
    norm2: LayerNorm,
    ff: FeedForward,
    norm2_context: LayerNorm,
    ff_context: FeedForward,
}

impl DoubleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim();
        let inner = cfg.mlp_hidden_dim();
        let norm = || layer_norm(dim, cfg.eps, vb.device(), vb.dtype());
        Ok(Self {
            norm1: norm()?,
            norm1_context: norm()?,
            attn: JointAttention::new(cfg, vb.pp("attn"))?,
            norm2: norm()?,
            ff: FeedForward::new(dim, inner, vb.pp("ff"))?,
            norm2_context: norm()?,
            ff_context: FeedForward::new(dim, inner, vb.pp("ff_context"))?,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_mod: &[Tensor],
        txt_mod: &[Tensor],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let norm_img = scale_shift(&img.apply(&self.norm1)?, &img_mod[0], &img_mod[1])?;
        let norm_txt = scale_shift(&txt.apply(&self.norm1_context)?, &txt_mod[0], &txt_mod[1])?;

        let (img_attn, txt_attn) = self.attn.forward(&norm_img, &norm_txt, cos, sin)?;

        let img = (img + img_attn.broadcast_mul(&img_mod[2])?)?;
        let ff = scale_shift(&img.apply(&self.norm2)?, &img_mod[3], &img_mod[4])?.apply(&self.ff)?;
        let img = (&img + ff.broadcast_mul(&img_mod[5])?)?;

        let txt = (txt + txt_attn.broadcast_mul(&txt_mod[2])?)?;
        let ff = scale_shift(&txt.apply(&self.norm2_context)?, &txt_mod[3], &txt_mod[4])?
            .apply(&self.ff_context)?;
        let txt = (&txt + ff.broadcast_mul(&txt_mod[5])?)?;
        Ok((img, txt))
    }
}

/// A parallel transformer block: one projection feeds both the qkv and the MLP
/// gate, and one projection takes both the attention output and the MLP back to
/// the hidden dimension. FLUX.1 fuses only the input side.
#[derive(Debug, Clone)]
struct SingleStreamBlock {
    norm: LayerNorm,
    to_qkv_mlp_proj: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    to_out: Linear,
    heads: usize,
    head_dim: usize,
    inner_dim: usize,
    mlp_hidden_dim: usize,
    accelerated: bool,
}

impl SingleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim();
        let mlp_hidden_dim = cfg.mlp_hidden_dim();
        let vb_attn = vb.pp("attn");
        Ok(Self {
            norm: layer_norm(dim, cfg.eps, vb.device(), vb.dtype())?,
            to_qkv_mlp_proj: linear(
                dim,
                dim * 3 + mlp_hidden_dim * 2,
                vb_attn.pp("to_qkv_mlp_proj"),
            )?,
            norm_q: rms_norm(cfg.attention_head_dim, cfg.eps, vb_attn.pp("norm_q"))?,
            norm_k: rms_norm(cfg.attention_head_dim, cfg.eps, vb_attn.pp("norm_k"))?,
            to_out: linear(dim + mlp_hidden_dim, dim, vb_attn.pp("to_out"))?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
            inner_dim: dim,
            mlp_hidden_dim,
            accelerated: cfg.use_accelerated_attn,
        })
    }

    fn forward(&self, xs: &Tensor, mod_: &[Tensor], cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let norm = scale_shift(&xs.apply(&self.norm)?, &mod_[0], &mod_[1])?;
        let projected = norm.apply(&self.to_qkv_mlp_proj)?;
        let (b, seq, _) = projected.dims3()?;

        let qkv = projected
            .narrow(D::Minus1, 0, self.inner_dim * 3)?
            .reshape((b, seq, 3, self.heads, self.head_dim))?;
        let q = norm_heads(&qkv.i((.., .., 0))?, &self.norm_q)?;
        let k = norm_heads(&qkv.i((.., .., 1))?, &self.norm_k)?;
        let v = qkv.i((.., .., 2))?.contiguous()?;
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
        let attn = attention(&q, &k, &v, self.accelerated)?;

        let mlp = projected.narrow(
            D::Minus1,
            self.inner_dim * 3,
            self.mlp_hidden_dim * 2,
        )?;
        let mlp = swiglu(&mlp, self.mlp_hidden_dim)?;
        let out = Tensor::cat(&[&attn, &mlp], D::Minus1)?.apply(&self.to_out)?;
        xs + out.broadcast_mul(&mod_[2])?
    }
}

/// `AdaLayerNormContinuous`, which chunks its conditioning into scale **then**
/// shift — the opposite order to FLUX.1's final layer.
#[derive(Debug, Clone)]
struct NormOut {
    norm: LayerNorm,
    linear: Linear,
}

impl NormOut {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(dim, eps, vb.device(), vb.dtype())?,
            linear: linear(dim, dim * 2, vb.pp("linear"))?,
        })
    }

    fn forward(&self, xs: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let chunks = temb.silu()?.apply(&self.linear)?.unsqueeze(1)?.chunk(2, D::Minus1)?;
        scale_shift(&xs.apply(&self.norm)?, &chunks[1], &chunks[0])
    }
}

// ==================== Model ====================

#[derive(Debug, Clone)]
pub struct Flux2Transformer2DModel {
    timestep_linear_1: Linear,
    timestep_linear_2: Linear,
    guidance_linear_1: Option<Linear>,
    guidance_linear_2: Option<Linear>,
    double_stream_modulation_img: Modulation,
    double_stream_modulation_txt: Modulation,
    single_stream_modulation: Modulation,
    x_embedder: Linear,
    context_embedder: Linear,
    double_blocks: Vec<DoubleStreamBlock>,
    single_blocks: Vec<SingleStreamBlock>,
    norm_out: NormOut,
    proj_out: Linear,
    axes_dims_rope: Vec<usize>,
    rope_theta: f64,
    timestep_channels: usize,
}

impl Flux2Transformer2DModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim();
        let out_channels = cfg.out_channels.unwrap_or(cfg.in_channels);
        let embedder = |vb: VarBuilder| -> Result<(Linear, Linear)> {
            Ok((
                linear(cfg.timestep_guidance_channels, dim, vb.pp("linear_1"))?,
                linear(dim, dim, vb.pp("linear_2"))?,
            ))
        };
        let vb_time = vb.pp("time_guidance_embed");
        let (timestep_linear_1, timestep_linear_2) = embedder(vb_time.pp("timestep_embedder"))?;
        let (guidance_linear_1, guidance_linear_2) = if cfg.guidance_embeds {
            let (a, b) = embedder(vb_time.pp("guidance_embedder"))?;
            (Some(a), Some(b))
        } else {
            (None, None)
        };

        let mut double_blocks = Vec::with_capacity(cfg.num_layers);
        let vb_double = vb.pp("transformer_blocks");
        for idx in 0..cfg.num_layers {
            double_blocks.push(DoubleStreamBlock::new(cfg, vb_double.pp(idx))?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        let vb_single = vb.pp("single_transformer_blocks");
        for idx in 0..cfg.num_single_layers {
            single_blocks.push(SingleStreamBlock::new(cfg, vb_single.pp(idx))?);
        }

        Ok(Self {
            timestep_linear_1,
            timestep_linear_2,
            guidance_linear_1,
            guidance_linear_2,
            double_stream_modulation_img: Modulation::new(
                dim,
                2,
                vb.pp("double_stream_modulation_img"),
            )?,
            double_stream_modulation_txt: Modulation::new(
                dim,
                2,
                vb.pp("double_stream_modulation_txt"),
            )?,
            single_stream_modulation: Modulation::new(dim, 1, vb.pp("single_stream_modulation"))?,
            x_embedder: linear(cfg.in_channels, dim, vb.pp("x_embedder"))?,
            context_embedder: linear(
                cfg.joint_attention_dim,
                dim,
                vb.pp("context_embedder"),
            )?,
            double_blocks,
            single_blocks,
            norm_out: NormOut::new(dim, cfg.eps, vb.pp("norm_out"))?,
            proj_out: linear(dim, out_channels, vb.pp("proj_out"))?,
            axes_dims_rope: cfg.axes_dims_rope.clone(),
            rope_theta: cfg.rope_theta,
            timestep_channels: cfg.timestep_guidance_channels,
        })
    }

    /// Predicts the flow velocity.
    ///
    /// `img` is packed latents `(batch, image_tokens, in_channels)` and `txt`
    /// the stacked encoder hidden states `(batch, text_tokens, joint_dim)`;
    /// `img_ids` and `txt_ids` carry one four-axis coordinate per token.
    /// `timestep` is the sigma in `0..=1`, not a discrete step index.
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timestep: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dtype = img.dtype();
        let temb = sinusoidal(timestep, self.timestep_channels, dtype)?
            .apply(&self.timestep_linear_1)?
            .silu()?
            .apply(&self.timestep_linear_2)?;
        let temb = match (&self.guidance_linear_1, &self.guidance_linear_2, guidance) {
            (Some(l1), Some(l2), Some(guidance)) => (temb
                + sinusoidal(guidance, self.timestep_channels, dtype)?
                    .apply(l1)?
                    .silu()?
                    .apply(l2)?)?,
            _ => temb,
        };

        let img_mod = self.double_stream_modulation_img.forward(&temb)?;
        let txt_mod = self.double_stream_modulation_txt.forward(&temb)?;
        let single_mod = self.single_stream_modulation.forward(&temb)?;

        let ids = Tensor::cat(&[txt_ids, img_ids], 0)?;
        let (cos, sin) = rope(&ids, &self.axes_dims_rope, self.rope_theta)?;

        let mut img = img.apply(&self.x_embedder)?;
        let mut txt = txt.apply(&self.context_embedder)?;
        for block in &self.double_blocks {
            (img, txt) = block.forward(&img, &txt, &img_mod, &txt_mod, &cos, &sin)?;
        }

        let txt_len = txt.dim(1)?;
        let mut xs = Tensor::cat(&[&txt, &img], 1)?;
        for block in &self.single_blocks {
            xs = block.forward(&xs, &single_mod, &cos, &sin)?;
        }
        let xs = xs.i((.., txt_len.., ..))?;
        self.norm_out.forward(&xs, &temb)?.apply(&self.proj_out)
    }
}

/// Timestep sinusoids, cosines before sines, over a timestep scaled to
/// `0..=1000`.
fn sinusoidal(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const MAX_PERIOD: f64 = 10_000.;
    let half = dim / 2;
    let t = (t.to_dtype(DType::F32)? * 1000.)?;
    let arange = Tensor::arange(0, half as u32, t.device())?.to_dtype(DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)
}
