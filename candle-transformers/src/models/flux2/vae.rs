//! FLUX.2 autoencoder (`AutoencoderKLFlux2`).
//!
//! The encoder and decoder stacks are the diffusers-format `AutoencoderKL` that
//! [`crate::models::z_image::vae`] already implements, only 32 latent channels
//! wide. What is new is on either side of them: 1x1 quant convolutions, and a
//! batch norm whose running statistics whiten the latent *after* it has been
//! folded into 2x2 patches. `encode` and `decode` apply both, so latents cross
//! this boundary already whitened and no caller should scale them again.

use candle::{DType, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

use crate::models::z_image::vae::{Decoder, Encoder, VaeConfig};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_latent_channels")]
    pub latent_channels: usize,
    #[serde(default = "default_block_out_channels")]
    pub block_out_channels: Vec<usize>,
    #[serde(default = "default_layers_per_block")]
    pub layers_per_block: usize,
    #[serde(default = "default_norm_num_groups")]
    pub norm_num_groups: usize,
    #[serde(default = "default_batch_norm_eps")]
    pub batch_norm_eps: f64,
    #[serde(default = "default_true")]
    pub use_quant_conv: bool,
    #[serde(default = "default_true")]
    pub use_post_quant_conv: bool,
}

fn default_in_channels() -> usize {
    3
}
fn default_out_channels() -> usize {
    3
}
fn default_latent_channels() -> usize {
    32
}
fn default_block_out_channels() -> Vec<usize> {
    vec![128, 256, 512, 512]
}
fn default_layers_per_block() -> usize {
    2
}
fn default_norm_num_groups() -> usize {
    32
}
fn default_batch_norm_eps() -> f64 {
    1e-4
}
fn default_true() -> bool {
    true
}

impl Default for Config {
    fn default() -> Self {
        Self::flux2()
    }
}

impl Config {
    pub fn flux2() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 32,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            batch_norm_eps: 1e-4,
            use_quant_conv: true,
            use_post_quant_conv: true,
        }
    }

    /// The encoder and decoder are built from the shared diffusers config; the
    /// scale and shift there stay neutral because flux.2 whitens with the batch
    /// norm instead, and applying both would scale the latent twice.
    fn inner(&self) -> VaeConfig {
        VaeConfig {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            latent_channels: self.latent_channels,
            block_out_channels: self.block_out_channels.clone(),
            layers_per_block: self.layers_per_block,
            scaling_factor: 1.0,
            shift_factor: 0.0,
            norm_num_groups: self.norm_num_groups,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: Option<Conv2d>,
    post_quant_conv: Option<Conv2d>,
    mean: Tensor,
    std: Tensor,
}

impl AutoEncoderKL {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner();
        let latent = cfg.latent_channels;
        let quant = |channels: usize, name: &str| {
            candle_nn::conv2d(channels, channels, 1, Conv2dConfig::default(), vb.pp(name))
        };
        let patched = latent * 4;
        let running = |name: &str| vb.get(patched, name)?.to_dtype(DType::F32);
        let std = (running("bn.running_var")? + cfg.batch_norm_eps)?.sqrt()?;
        Ok(Self {
            encoder: Encoder::new(&inner, vb.pp("encoder"))?,
            decoder: Decoder::new(&inner, vb.pp("decoder"))?,
            quant_conv: cfg
                .use_quant_conv
                .then(|| quant(2 * latent, "quant_conv"))
                .transpose()?,
            post_quant_conv: cfg
                .use_post_quant_conv
                .then(|| quant(latent, "post_quant_conv"))
                .transpose()?,
            mean: running("bn.running_mean")?
                .reshape((1, patched, 1, 1))?
                .to_dtype(vb.dtype())?,
            std: std.reshape((1, patched, 1, 1))?.to_dtype(vb.dtype())?,
        })
    }

    /// `(batch, 3, h, w)` in `[-1, 1]` to whitened patched latents
    /// `(batch, 4 * latent_channels, h / 16, w / 16)`.
    ///
    /// The distribution is taken at its mode rather than sampled: klein
    /// conditions on a reference image and a sampled latent would make the same
    /// seed and image disagree between runs.
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let moments = xs.apply(&self.encoder)?;
        let moments = match &self.quant_conv {
            Some(conv) => moments.apply(conv)?,
            None => moments,
        };
        let mean = moments.chunk(2, 1)?.swap_remove(0);
        let latents = patchify(&mean)?;
        latents
            .broadcast_sub(&self.mean)?
            .broadcast_div(&self.std)
    }

    /// The inverse: whitened patched latents back to `(batch, 3, h, w)`.
    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        let latents = xs.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let latents = unpatchify(&latents)?;
        let latents = match &self.post_quant_conv {
            Some(conv) => latents.apply(conv)?,
            None => latents,
        };
        latents.apply(&self.decoder)
    }
}

/// Folds each 2x2 latent cell into the channel dimension.
pub fn patchify(xs: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = xs.dims4()?;
    xs.reshape((b, c, h / 2, 2, w / 2, 2))?
        .permute((0, 1, 3, 5, 2, 4))?
        .contiguous()?
        .reshape((b, c * 4, h / 2, w / 2))
}

pub fn unpatchify(xs: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = xs.dims4()?;
    xs.reshape((b, c / 4, 2, 2, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .contiguous()?
        .reshape((b, c / 4, h * 2, w * 2))
}
