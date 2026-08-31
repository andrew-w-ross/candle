//! Quantized Z-Image transformer, loaded from a GGUF file.
//!
//! The GGUF checkpoints published for ComfyUI carry Z-Image's original
//! (Lumina-style) tensor names, not the diffusers ones the safetensors path
//! reads: `x_embedder` rather than `all_x_embedder.2-1`, a single fused
//! `attention.qkv` rather than `to_q`/`to_k`/`to_v`, and `q_norm`/`k_norm`
//! rather than `norm_q`/`norm_k`. They also declare
//! `general.architecture = lumina2`, which is a ComfyUI label rather than a
//! description of the weights.
//!
//! Which tensors are quantized is the checkpoint's decision, not ours: the
//! published files quantize the 30 main layers and leave the embedders,
//! refiners, norms and final projection in BF16/F32. A tensor stored
//! unquantized is dequantized once at load and used as an ordinary matmul, so
//! one code path covers both.

use candle::quantized::{gguf_file::Content, GgmlDType, QMatMul, QTensor};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::RmsNorm;

use crate::models::z_image::transformer::{
    apply_rotary_emb, attention_dispatch, create_coordinate_grid, patchify, unpatchify, Config,
    LayerNormNoParams, RopeEmbedder, FREQUENCY_EMBEDDING_SIZE, MAX_PERIOD,
};

/// Rejects a file whose first layer is not this configuration's shape.
///
/// The declared architecture is no test: these files say `lumina2`, which
/// Lumina-Image-2.0 also says at a different width. The fused qkv carries both
/// the model width and the head layout, so it is what tells them apart, and
/// checking it up front turns a wrong file into one error rather than a
/// half-built model.
fn check_shapes(content: &Content, cfg: &Config) -> Result<()> {
    const QKV: &str = "layers.0.attention.qkv.weight";
    let Some(info) = content.tensor_infos.get(QKV) else {
        candle::bail!("{QKV} is missing: not a z-image transformer in gguf form")
    };
    let (out, in_dim) = info.shape.dims2()?;
    let expected = (cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim();
    if in_dim != cfg.dim || out != expected {
        candle::bail!(
            "{QKV} is {out}x{in_dim}, expected {expected}x{}: not this z-image configuration",
            cfg.dim
        )
    }
    Ok(())
}

/// Whether `path` is a Z-Image transformer this configuration can load.
///
/// A gguf carries no diffusers `config.json`, so a listing that identifies a
/// family by the declared denoiser class has nothing to read. This is what it
/// reads instead — and it is the stronger test, since the weights cannot
/// disagree with themselves. Validate on shapes, never on the declared
/// `general.architecture`: both published z-image ggufs claim `lumina2`.
pub fn is_z_image_gguf<P: AsRef<std::path::Path>>(cfg: &Config, path: P) -> bool {
    let Ok(mut file) = std::fs::File::open(path.as_ref()) else {
        return false;
    };
    Content::read(&mut file).is_ok_and(|content| check_shapes(&content, cfg).is_ok())
}

/// A GGUF weight file held open, read by tensor name.
pub struct Gguf<R: std::io::Read + std::io::Seek> {
    content: Content,
    reader: R,
    device: Device,
    dtype: DType,
}

impl<R: std::io::Read + std::io::Seek> Gguf<R> {
    /// `dtype` is the activation dtype; unquantized weights are cast to it.
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
    fn tensor(&mut self, name: &str) -> Result<Tensor> {
        let qt = self.qtensor(name)?;
        qt.dequantize(&self.device)?.to_dtype(self.dtype)
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        Ok(RmsNorm::new(self.tensor(&format!("{name}.weight"))?, eps))
    }

    fn linear(&mut self, name: &str, bias: bool) -> Result<QLinear> {
        let weight = self.qtensor(&format!("{name}.weight"))?;
        let weight = match weight.dtype() {
            // A quantized matmul takes and returns the activation dtype; a
            // stored-unquantized one is a plain tensor and must match it.
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => QMatMul::Tensor(
                weight
                    .dequantize(&self.device)?
                    .to_dtype(self.dtype)?
                    .contiguous()?,
            ),
            _ => QMatMul::from_qtensor(weight)?,
        };
        let bias = if bias {
            Some(self.tensor(&format!("{name}.bias"))?)
        } else {
            None
        };
        Ok(QLinear { weight, bias })
    }
}

/// A linear layer whose weight may or may not be quantized.
#[derive(Debug, Clone)]
struct QLinear {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.weight.forward(xs)?;
        match &self.bias {
            Some(bias) => xs.broadcast_add(bias),
            None => Ok(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct TimestepEmbedder {
    linear1: QLinear,
    linear2: QLinear,
}

impl TimestepEmbedder {
    fn new<R: std::io::Read + std::io::Seek>(gguf: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear1: gguf.linear(&format!("{prefix}.mlp.0"), true)?,
            linear2: gguf.linear(&format!("{prefix}.mlp.2"), true)?,
        })
    }

    fn forward(&self, t: &Tensor, dtype: DType) -> Result<Tensor> {
        let half = FREQUENCY_EMBEDDING_SIZE / 2;
        let freqs = Tensor::arange(0u32, half as u32, t.device())?.to_dtype(DType::F32)?;
        let freqs = (freqs * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&freqs.unsqueeze(0)?)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)?;
        embedding.apply(&self.linear1)?.silu()?.apply(&self.linear2)
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    w1: QLinear,
    w2: QLinear,
    w3: QLinear,
}

impl FeedForward {
    fn new<R: std::io::Read + std::io::Seek>(gguf: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: gguf.linear(&format!("{prefix}.w1"), false)?,
            w2: gguf.linear(&format!("{prefix}.w2"), false)?,
            w3: gguf.linear(&format!("{prefix}.w3"), false)?,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (xs.apply(&self.w1)?.silu()? * xs.apply(&self.w3)?)?.apply(&self.w2)
    }
}

/// Attention over a fused qkv projection.
///
/// The fusion is along the output dimension, where a block-quantized weight is
/// row-independent, so the three projections are recovered by narrowing the
/// product rather than by slicing the weight — which a quantized tensor does
/// not permit. One matmul for three projections is also the faster shape.
#[derive(Debug, Clone)]
struct Attention {
    qkv: QLinear,
    out: QLinear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    use_accelerated_attn: bool,
}

impl Attention {
    fn new<R: std::io::Read + std::io::Seek>(
        gguf: &mut Gguf<R>,
        prefix: &str,
        cfg: &Config,
    ) -> Result<Self> {
        let (q_norm, k_norm) = if cfg.qk_norm {
            (
                Some(gguf.rms_norm(&format!("{prefix}.q_norm"), 1e-5)?),
                Some(gguf.rms_norm(&format!("{prefix}.k_norm"), 1e-5)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            qkv: gguf.linear(&format!("{prefix}.qkv"), false)?,
            out: gguf.linear(&format!("{prefix}.out"), false)?,
            q_norm,
            k_norm,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim(),
            use_accelerated_attn: cfg.use_accelerated_attn,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;

        let qkv = hidden_states.apply(&self.qkv)?;
        let q = qkv
            .narrow(D::Minus1, 0, q_dim)?
            .reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let k = qkv
            .narrow(D::Minus1, q_dim, kv_dim)?
            .reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;
        let v = qkv
            .narrow(D::Minus1, q_dim + kv_dim, kv_dim)?
            .reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;

        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };

        let q = apply_rotary_emb(&q, cos, sin)?.transpose(1, 2)?.contiguous()?;
        let k = apply_rotary_emb(&k, cos, sin)?.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let context = attention_dispatch(
            &q,
            &k,
            &v,
            attention_mask,
            scale,
            self.n_heads,
            self.use_accelerated_attn,
            hidden_states.device(),
        )?;

        context
            .transpose(1, 2)?
            .reshape((b, seq_len, ()))?
            .apply(&self.out)
    }
}

#[derive(Debug, Clone)]
struct Block {
    attention: Attention,
    feed_forward: FeedForward,
    attention_norm1: RmsNorm,
    attention_norm2: RmsNorm,
    ffn_norm1: RmsNorm,
    ffn_norm2: RmsNorm,
    adaln_modulation: Option<QLinear>,
}

impl Block {
    fn new<R: std::io::Read + std::io::Seek>(
        gguf: &mut Gguf<R>,
        prefix: &str,
        modulation: bool,
        cfg: &Config,
    ) -> Result<Self> {
        let adaln_modulation = if modulation {
            Some(gguf.linear(&format!("{prefix}.adaLN_modulation.0"), true)?)
        } else {
            None
        };
        Ok(Self {
            attention: Attention::new(gguf, &format!("{prefix}.attention"), cfg)?,
            feed_forward: FeedForward::new(gguf, &format!("{prefix}.feed_forward"))?,
            attention_norm1: gguf.rms_norm(&format!("{prefix}.attention_norm1"), cfg.norm_eps)?,
            attention_norm2: gguf.rms_norm(&format!("{prefix}.attention_norm2"), cfg.norm_eps)?,
            ffn_norm1: gguf.rms_norm(&format!("{prefix}.ffn_norm1"), cfg.norm_eps)?,
            ffn_norm2: gguf.rms_norm(&format!("{prefix}.ffn_norm2"), cfg.norm_eps)?,
            adaln_modulation,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        let Some(adaln) = &self.adaln_modulation else {
            let normed = self.attention_norm1.forward(x)?;
            let attn_out = self.attention.forward(&normed, attn_mask, cos, sin)?;
            let x = (x + self.attention_norm2.forward(&attn_out)?)?;

            let normed = self.ffn_norm1.forward(&x)?;
            let ffn_out = self.feed_forward.forward(&normed)?;
            let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
            return x + ffn_out;
        };

        let adaln_input = match adaln_input {
            Some(t) => t,
            None => candle::bail!("adaln_input required for a modulated block"),
        };
        let modulation = adaln_input.apply(adaln)?.unsqueeze(1)?;
        let chunks = modulation.chunk(4, D::Minus1)?;
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) =
            (&chunks[0], &chunks[1], &chunks[2], &chunks[3]);
        let gate_msa = gate_msa.tanh()?;
        let gate_mlp = gate_mlp.tanh()?;
        let scale_msa = (scale_msa + 1.0)?;
        let scale_mlp = (scale_mlp + 1.0)?;

        let normed = self.attention_norm1.forward(x)?;
        let scaled = normed.broadcast_mul(&scale_msa)?;
        let attn_out = self.attention.forward(&scaled, attn_mask, cos, sin)?;
        let attn_out = self.attention_norm2.forward(&attn_out)?;
        let x = (x + gate_msa.broadcast_mul(&attn_out)?)?;

        let normed = self.ffn_norm1.forward(&x)?;
        let scaled = normed.broadcast_mul(&scale_mlp)?;
        let ffn_out = self.feed_forward.forward(&scaled)?;
        let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
        let gated = gate_mlp.broadcast_mul(&ffn_out)?;
        x + gated
    }
}

#[derive(Debug, Clone)]
struct FinalLayer {
    norm_final: LayerNormNoParams,
    linear: QLinear,
    adaln_silu: QLinear,
}

impl FinalLayer {
    fn new<R: std::io::Read + std::io::Seek>(gguf: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm_final: LayerNormNoParams::new(1e-6),
            linear: gguf.linear(&format!("{prefix}.linear"), true)?,
            adaln_silu: gguf.linear(&format!("{prefix}.adaLN_modulation.1"), true)?,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let scale = c.silu()?.apply(&self.adaln_silu)?;
        let scale = (scale + 1.0)?.unsqueeze(1)?;
        self.norm_final
            .forward(x)?
            .broadcast_mul(&scale)?
            .apply(&self.linear)
    }
}

/// Z-Image's transformer with the weights read from a GGUF file.
pub struct QuantizedZImageTransformer2DModel {
    t_embedder: TimestepEmbedder,
    cap_embedder_norm: RmsNorm,
    cap_embedder_linear: QLinear,
    x_embedder: QLinear,
    final_layer: FinalLayer,
    noise_refiner: Vec<Block>,
    context_refiner: Vec<Block>,
    layers: Vec<Block>,
    rope_embedder: RopeEmbedder,
    cfg: Config,
}

impl QuantizedZImageTransformer2DModel {
    /// # Errors
    /// Fails when the file is not a Z-Image checkpoint in the original naming,
    /// or when a tensor's shape contradicts `cfg`.
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        cfg: &Config,
        content: Content,
        reader: R,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        check_shapes(&content, cfg)?;
        let mut gguf = Gguf::new(content, reader, device.clone(), dtype);

        let mut blocks = |prefix: &str, count: usize, modulation: bool| {
            (0..count)
                .map(|i| Block::new(&mut gguf, &format!("{prefix}.{i}"), modulation, cfg))
                .collect::<Result<Vec<_>>>()
        };
        let noise_refiner = blocks("noise_refiner", cfg.n_refiner_layers, true)?;
        let context_refiner = blocks("context_refiner", cfg.n_refiner_layers, false)?;
        let layers = blocks("layers", cfg.n_layers, true)?;

        let t_embedder = TimestepEmbedder::new(&mut gguf, "t_embedder")?;
        let cap_embedder_norm = gguf.rms_norm("cap_embedder.0", cfg.norm_eps)?;
        let cap_embedder_linear = gguf.linear("cap_embedder.1", true)?;
        let x_embedder = gguf.linear("x_embedder", true)?;
        let final_layer = FinalLayer::new(&mut gguf, "final_layer")?;

        let rope_embedder = RopeEmbedder::new(
            cfg.rope_theta,
            cfg.axes_dims.clone(),
            cfg.axes_lens.clone(),
            device,
            dtype,
        )?;

        Ok(Self {
            t_embedder,
            cap_embedder_norm,
            cap_embedder_linear,
            x_embedder,
            final_layer,
            noise_refiner,
            context_refiner,
            layers,
            rope_embedder,
            cfg: cfg.clone(),
        })
    }

    /// Reads a GGUF file from `path`.
    ///
    /// # Errors
    /// Fails when the file cannot be read as GGUF, or as this model.
    pub fn from_gguf_path<P: AsRef<std::path::Path>>(
        cfg: &Config,
        path: P,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let content = Content::read(&mut file)?;
        Self::from_gguf(cfg, content, file, device, dtype)
    }

    /// Latents `(B, C, F, H, W)`, timesteps `t` in `[0, 1]`, caption features
    /// `(B, text_len, cap_feat_dim)` and a `(B, text_len)` caption mask.
    ///
    /// # Errors
    /// Fails on a shape the configuration does not describe.
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> Result<Tensor> {
        let device = x.device();
        let (b, _c, f, h, w) = x.dims5()?;
        let patch_size = self.cfg.all_patch_size[0];
        let f_patch_size = self.cfg.all_f_patch_size[0];

        let t_scaled = (t * self.cfg.t_scale)?;
        let adaln_input = self.t_embedder.forward(&t_scaled, cap_feats.dtype())?;

        let (x_patches, orig_size) = patchify(x, patch_size, f_patch_size)?;
        let mut x = x_patches.apply(&self.x_embedder)?;
        let img_seq_len = x.dim(1)?;

        let text_len = cap_feats.dim(1)?;
        let x_pos_ids = create_coordinate_grid(
            (f / f_patch_size, h / patch_size, w / patch_size),
            (text_len + 1, 0, 0),
            device,
        )?;
        let (x_cos, x_sin) = self.rope_embedder.forward(&x_pos_ids)?;

        let cap_normed = self.cap_embedder_norm.forward(cap_feats)?;
        let mut cap = cap_normed.apply(&self.cap_embedder_linear)?;

        let cap_pos_ids = create_coordinate_grid((text_len, 1, 1), (1, 0, 0), device)?;
        let (cap_cos, cap_sin) = self.rope_embedder.forward(&cap_pos_ids)?;

        let x_attn_mask = Tensor::ones((b, img_seq_len), DType::U8, device)?;
        let cap_attn_mask = cap_mask.to_dtype(DType::U8)?;

        for layer in &self.noise_refiner {
            x = layer.forward(&x, Some(&x_attn_mask), &x_cos, &x_sin, Some(&adaln_input))?;
        }
        for layer in &self.context_refiner {
            cap = layer.forward(&cap, Some(&cap_attn_mask), &cap_cos, &cap_sin, None)?;
        }

        let mut unified = Tensor::cat(&[&x, &cap], 1)?;
        let unified_pos_ids = Tensor::cat(&[&x_pos_ids, &cap_pos_ids], 0)?;
        let (unified_cos, unified_sin) = self.rope_embedder.forward(&unified_pos_ids)?;
        let unified_attn_mask = Tensor::cat(&[&x_attn_mask, &cap_attn_mask], 1)?;

        for layer in &self.layers {
            unified = layer.forward(
                &unified,
                Some(&unified_attn_mask),
                &unified_cos,
                &unified_sin,
                Some(&adaln_input),
            )?;
        }

        let x_out = unified.narrow(1, 0, img_seq_len)?;
        let x_out = self.final_layer.forward(&x_out, &adaln_input)?;
        unpatchify(
            &x_out,
            orig_size,
            patch_size,
            f_patch_size,
            self.cfg.in_channels,
        )
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }
}
