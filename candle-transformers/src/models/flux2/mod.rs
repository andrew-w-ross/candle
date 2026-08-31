//! FLUX.2 — text-to-image flow matching from Black Forest Labs. One module for
//! klein and dev alike: they are the same architecture at different sizes, and
//! every dimension that differs is in the config.
//!
//! - 🤗 [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
//!
//! The transformer keeps FLUX.1's double-stream then single-stream shape but
//! shares one modulation across the stack, gates its feedforward with SwiGLU,
//! runs the single stream as a parallel block and positions tokens on four rope
//! axes. Conditioning is three stacked Qwen3 hidden states rather than
//! T5 plus a pooled CLIP vector, so there is no `y` input and no pooled
//! projection.

pub mod bfl;
pub mod transformer;
pub mod vae;

pub use transformer::{Config, Flux2Transformer2DModel};
pub use vae::AutoEncoderKL;

use candle::{DType, Device, Result, Tensor};

/// Coordinates for `len` text tokens: `(0, 0, 0, position)`, so text occupies the
/// fourth rope axis alone and never collides with an image position.
pub fn text_ids(len: usize, device: &Device) -> Result<Tensor> {
    let ids: Vec<f32> = (0..len).flat_map(|i| [0., 0., 0., i as f32]).collect();
    Tensor::from_vec(ids, (len, 4), device)
}

/// Coordinates for a `height` by `width` grid of latent tokens, row-major:
/// `(0, row, column, 0)`.
pub fn latent_ids(height: usize, width: usize, device: &Device) -> Result<Tensor> {
    let ids: Vec<f32> = (0..height)
        .flat_map(|h| (0..width).flat_map(move |w| [0., h as f32, w as f32, 0.]))
        .collect();
    Tensor::from_vec(ids, (height * width, 4), device)
}

/// `(batch, channels, height, width)` to `(batch, height * width, channels)`.
pub fn pack(xs: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = xs.dims4()?;
    xs.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()
}

/// The inverse of [`pack`]; the token order is the one [`latent_ids`] assigns.
pub fn unpack(xs: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (b, _, c) = xs.dims3()?;
    xs.transpose(1, 2)?.contiguous()?.reshape((b, c, height, width))
}

/// Gaussian noise for a latent grid, in packed token form.
pub fn get_noise(
    channels: usize,
    height: usize,
    width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    pack(&Tensor::randn(0f32, 1f32, (1, channels, height, width), device)?.to_dtype(dtype)?)
}

/// The shift the flow-matching schedule is warped by. Klein fits it empirically
/// against both the token count and the step count, so unlike FLUX.1 the
/// schedule moves as the step count changes and the two cannot share one
/// interpolation.
pub fn empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    const A1: f64 = 8.73809524e-05;
    const B1: f64 = 1.89833333;
    const A2: f64 = 0.00016927;
    const B2: f64 = 0.45666666;

    let seq_len = image_seq_len as f64;
    let m_200 = A2 * seq_len + B2;
    if image_seq_len > 4300 {
        return m_200;
    }
    let m_10 = A1 * seq_len + B1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f64 + b
}

/// Sigmas for `num_steps` denoising steps, with a terminal zero appended, so a
/// euler step reads `sigmas[i]` and `sigmas[i + 1]`.
pub fn sigma_schedule(image_seq_len: usize, num_steps: usize) -> Vec<f64> {
    let mu = empirical_mu(image_seq_len, num_steps).exp();
    let steps = num_steps.max(1) as f64;
    let mut sigmas: Vec<f64> = (0..num_steps)
        .map(|i| {
            let t = 1.0 + (1.0 / steps - 1.0) * i as f64 / (steps - 1.0).max(1.0);
            mu / (mu + (1.0 / t - 1.0))
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::IndexOp;

    /// Against `Flux2KleinPipeline`'s own schedule: a `linspace(1, 1/n, n)`
    /// warped by the empirical mu, with a terminal zero appended.
    #[test]
    fn sigmas_match_the_reference_schedule() {
        let close = |a: f64, b: f64| assert!((a - b).abs() < 1e-7, "{a} vs {b}");

        close(empirical_mu(4096, 4), 2.291_179_894_1);
        close(empirical_mu(1024, 50), 1.701_956_207_3);
        // Past 4300 tokens the step count stops moving the shift.
        close(empirical_mu(4800, 10), 1.269_162_66);
        close(empirical_mu(4800, 50), 1.269_162_66);

        let sigmas = sigma_schedule(4096, 4);
        assert_eq!(sigmas.len(), 5);
        for (got, want) in sigmas
            .iter()
            .zip([1.0, 0.967_383_99, 0.908_143_92, 0.767_199_96, 0.0])
        {
            close(*got, want);
        }

        let sigmas = sigma_schedule(1024, 50);
        close(sigmas[0], 1.0);
        close(sigmas[1], 0.996_292_85);
        close(sigmas[49], 0.100_664_40);
        close(sigmas[50], 0.0);
    }

    /// Forward passes against `diffusers`, on a randomly initialised model small
    /// enough to run on the cpu. `FLUX2_REFERENCE` points at a safetensors file
    /// holding `model.*` and `vae.*` weights alongside `io.*` inputs and the
    /// outputs diffusers produced from them.
    #[test]
    #[ignore = "needs a reference dump; set FLUX2_REFERENCE"]
    fn matches_the_diffusers_reference() {
        let Ok(path) = std::env::var("FLUX2_REFERENCE") else {
            return;
        };
        let device = Device::Cpu;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device).unwrap()
        };

        let cfg = Config {
            in_channels: 16,
            num_layers: 2,
            num_single_layers: 2,
            attention_head_dim: 32,
            num_attention_heads: 2,
            joint_attention_dim: 24,
            axes_dims_rope: vec![8, 8, 8, 8],
            use_accelerated_attn: false,
            ..Config::klein_4b()
        };
        let model = Flux2Transformer2DModel::new(&cfg, vb.pp("model")).unwrap();
        let io = vb.pp("io");
        let img = io.get((1, 12, 16), "img").unwrap();
        let txt = io.get((1, 5, 24), "txt").unwrap();
        let t = io.get(1, "t").unwrap();
        let out = model
            .forward(
                &img,
                &latent_ids(3, 4, &device).unwrap(),
                &txt,
                &text_ids(5, &device).unwrap(),
                &t,
                None,
            )
            .unwrap();
        assert!(max_diff(&out, &io.get((1, 12, 16), "out").unwrap()) < 2e-4);

        let vae_cfg = vae::Config {
            latent_channels: 2,
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            norm_num_groups: 4,
            ..vae::Config::flux2()
        };
        let vae = AutoEncoderKL::new(&vae_cfg, vb.pp("vae")).unwrap();
        let image = io.get((1, 3, 32, 32), "image").unwrap();
        let enc = vae.encode(&image).unwrap();
        assert!(max_diff(&enc, &io.get((1, 8, 8, 8), "enc").unwrap()) < 2e-4);
        let dec = vae.decode(&enc).unwrap();
        assert!(max_diff(&dec, &io.get((1, 3, 32, 32), "dec").unwrap()) < 2e-4);
    }

    /// The conditioning path against `transformers`: three hidden states laid
    /// end to end per token, over a right-padded prompt whose padding is masked
    /// out of the keys.
    #[test]
    #[ignore = "needs a reference dump; set FLUX2_TE_REFERENCE"]
    fn conditioning_matches_the_transformers_reference() {
        use crate::models::z_image::{TextEncoderConfig, ZImageTextEncoder};

        let Ok(path) = std::env::var("FLUX2_TE_REFERENCE") else {
            return;
        };
        let device = Device::Cpu;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device).unwrap()
        };
        let cfg = TextEncoderConfig {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 48,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 16,
            max_position_embeddings: 128,
            ..TextEncoderConfig::z_image()
        };
        let encoder = ZImageTextEncoder::new(&cfg, vb.pp("te")).unwrap();
        let io = vb.pp("io");
        let ids = io.get((1, 12), "ids").unwrap().to_dtype(DType::U32).unwrap();
        let states = encoder.forward_hidden_states(&ids, 7, &[1, 2, 3]).unwrap();
        let cond = Tensor::cat(&states, candle::D::Minus1).unwrap();
        assert!(max_diff(&cond, &io.get((1, 12, 96), "cond").unwrap()) < 2e-5);
    }

    /// The cuda backend against the cpu one, same weights and same input. The
    /// port already matches diffusers on the cpu, so what this asks is narrower
    /// and sharper: whether a kernel, a layout assumption or a dtype promotion
    /// makes cuda disagree with the path that was verified.
    #[test]
    #[ignore = "needs a reference dump and a cuda device; set FLUX2_REFERENCE"]
    fn cpu_and_cuda_agree() {
        let Ok(path) = std::env::var("FLUX2_REFERENCE") else {
            return;
        };
        let Ok(cuda) = Device::new_cuda(0) else {
            println!("no cuda device; skipped");
            return;
        };

        let cfg = Config {
            in_channels: 16,
            num_layers: 2,
            num_single_layers: 2,
            attention_head_dim: 32,
            num_attention_heads: 2,
            joint_attention_dim: 24,
            axes_dims_rope: vec![8, 8, 8, 8],
            use_accelerated_attn: false,
            ..Config::klein_4b()
        };

        let run = |device: &Device| {
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, device)
                    .unwrap()
            };
            let model = Flux2Transformer2DModel::new(&cfg, vb.pp("model")).unwrap();
            let io = vb.pp("io");
            model
                .forward(
                    &io.get((1, 12, 16), "img").unwrap(),
                    &latent_ids(3, 4, device).unwrap(),
                    &io.get((1, 5, 24), "txt").unwrap(),
                    &text_ids(5, device).unwrap(),
                    &io.get(1, "t").unwrap(),
                    None,
                )
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
        };

        assert!(max_diff(&run(&Device::Cpu), &run(&cuda)) < 1e-4);
    }

    /// One forward pass of klein-9B read straight out of a scaled-fp8 checkpoint
    /// in BFL naming. `covers_a_real_checkpoint` proves the key map; this is the
    /// only thing that exercises the dequant arithmetic and the qkv and
    /// modulation splits on real weights.
    #[test]
    #[ignore = "needs the fp8 9B checkpoint and a cuda device; set FLUX2_FP8"]
    fn fp8_klein_9b_runs() {
        let Ok(path) = std::env::var("FLUX2_FP8") else {
            return;
        };
        let Ok(device) = Device::new_cuda(0) else {
            println!("no cuda device; skipped");
            return;
        };
        let vb = unsafe { bfl::var_builder(&[&path], DType::BF16, &device).unwrap() };
        let cfg = Config::klein_9b();
        let model = Flux2Transformer2DModel::new(&cfg, vb).unwrap();

        let (rows, cols, txt_len) = (8, 8, 16);
        let img = Tensor::randn(0f32, 1f32, (1, rows * cols, 128), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let txt = Tensor::randn(0f32, 1f32, (1, txt_len, cfg.joint_attention_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let t = Tensor::from_vec(vec![0.7f32], 1, &device).unwrap();
        let out = model
            .forward(
                &img,
                &latent_ids(rows, cols, &device).unwrap(),
                &txt,
                &text_ids(txt_len, &device).unwrap(),
                &t,
                None,
            )
            .unwrap();

        assert_eq!(out.dims(), &[1, rows * cols, 128]);
        let finite = out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let bad = finite.iter().filter(|v| !v.is_finite()).count();
        let mean = finite.iter().map(|v| v.abs()).sum::<f32>() / finite.len() as f32;
        println!("fp8 9B forward: {} values, {bad} non-finite, mean abs {mean}", finite.len());
        assert_eq!(bad, 0);
        assert!(mean > 1e-3 && mean < 1e3, "output collapsed or exploded: {mean}");
    }

    /// Prints a gguf's metadata and tensor names. Not an assertion: this is how
    /// you find out what naming a quantized flux.2 file uses before writing a
    /// loader for it.
    #[test]
    #[ignore = "prints a gguf's contents; set FLUX2_GGUF"]
    fn describe_gguf() {
        let Ok(path) = std::env::var("FLUX2_GGUF") else {
            return;
        };
        let mut file = std::fs::File::open(&path).unwrap();
        let content = candle::quantized::gguf_file::Content::read(&mut file).unwrap();
        for (key, value) in &content.metadata {
            println!("meta {key} = {value:?}");
        }
        println!("{} tensors", content.tensor_infos.len());
        let mut names: Vec<_> = content.tensor_infos.iter().collect();
        names.sort_by_key(|(name, _)| name.to_string());
        for (name, info) in names {
            println!("{name} {:?} {:?}", info.shape.dims(), info.ggml_dtype);
        }
    }

    /// The same for a safetensors file, which is how you find out whether a
    /// single-file checkpoint uses diffusers or BFL naming, and what dtype it
    /// stores.
    #[test]
    #[ignore = "prints a safetensors file's contents; set FLUX2_SAFETENSORS"]
    fn describe_safetensors() {
        let Ok(path) = std::env::var("FLUX2_SAFETENSORS") else {
            return;
        };
        let file = unsafe { candle::safetensors::MmapedSafetensors::new(&path).unwrap() };
        let mut names: Vec<_> = file.tensors().into_iter().collect();
        names.sort_by(|a, b| a.0.cmp(&b.0));
        println!("{} tensors", names.len());
        for (name, view) in names {
            println!("{name} {:?} {:?}", view.shape(), view.dtype());
        }
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        let diff = a
            .sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        println!("max abs diff {diff}");
        diff
    }

    /// Packing is the token order [`latent_ids`] numbers, so a round trip has to
    /// put every cell back where it came from.
    #[test]
    fn packing_round_trips_in_the_order_the_ids_assign() {
        let device = Device::Cpu;
        let xs = Tensor::arange(0f32, 24., &device)
            .unwrap()
            .reshape((1, 2, 3, 4))
            .unwrap();
        let packed = pack(&xs).unwrap();
        assert_eq!(packed.dims(), &[1, 12, 2]);
        let back = unpack(&packed, 3, 4).unwrap();
        assert_eq!(back.flatten_all().unwrap().to_vec1::<f32>().unwrap(), (0..24).map(|i| i as f32).collect::<Vec<_>>());

        let ids = latent_ids(3, 4, &device).unwrap();
        assert_eq!(ids.dims(), &[12, 4]);
        // Token five is row one, column one.
        assert_eq!(ids.i(5).unwrap().to_vec1::<f32>().unwrap(), vec![0., 1., 1., 0.]);
        // Text lives on the fourth axis alone.
        let text = text_ids(3, &device).unwrap();
        assert_eq!(text.i(2).unwrap().to_vec1::<f32>().unwrap(), vec![0., 0., 0., 2.]);
    }
}
