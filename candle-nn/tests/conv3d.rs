use anyhow::Result;
use candle::{Device, Tensor};
use candle_nn::{Conv3d, Conv3dConfig, Module};

#[test]
fn conv3d_module_applies_bias_per_channel() -> Result<()> {
    let dev = Device::Cpu;
    // Two output channels, a depth-2 kernel of ones over a depth-3 input of ones.
    let x = Tensor::ones((1, 1, 3, 2, 2), candle::DType::F32, &dev)?;
    let w = Tensor::ones((2, 1, 2, 2, 2), candle::DType::F32, &dev)?;
    let b = Tensor::new(&[5f32, -5.], &dev)?;
    let conv = Conv3d::new(w, Some(b), Conv3dConfig::default());
    let res = conv.forward(&x)?;
    assert_eq!(res.dims(), [1, 2, 2, 1, 1]);
    // Each output sums 2*2*2 ones, then the per-channel bias is added.
    assert_eq!(
        res.flatten_all()?.to_vec1::<f32>()?,
        [13., 13., 3., 3.].to_vec()
    );
    Ok(())
}

#[test]
fn conv3d_module_honours_config() -> Result<()> {
    let dev = Device::Cpu;
    let x = Tensor::arange(0f32, 24f32, &dev)?.reshape((1, 1, 6, 2, 2))?;
    let w = Tensor::ones((1, 1, 2, 1, 1), candle::DType::F32, &dev)?;
    let cfg = Conv3dConfig {
        stride: [2, 1, 1],
        dilation: [2, 1, 1],
        ..Default::default()
    };
    let conv = Conv3d::new(w.clone(), None, cfg);
    let res = conv.forward(&x)?;
    let direct = x.conv3d(&w, cfg.padding, cfg.stride, cfg.dilation, cfg.groups)?;
    assert_eq!(res.dims(), direct.dims());
    assert_eq!(
        res.flatten_all()?.to_vec1::<f32>()?,
        direct.flatten_all()?.to_vec1::<f32>()?
    );
    Ok(())
}
