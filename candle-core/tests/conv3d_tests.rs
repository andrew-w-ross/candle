use anyhow::Result;
use candle_core::{Device, Tensor};

macro_rules! test_conv3d_device {
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident) => {
        #[test]
        fn $test_cpu() -> Result<()> {
            $fn_name(&Device::Cpu)
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn $test_cuda() -> Result<()> {
            $fn_name(&Device::new_cuda(0)?)
        }
    };
}

/// Deterministic pseudo-random values so that the reference and the tensor under test are fed
/// exactly the same bytes on every device.
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn out_dim(i: usize, k: usize, p: usize, s: usize, d: usize) -> usize {
    (i + 2 * p - d * (k - 1) - 1) / s + 1
}

/// Naive transcription of the `torch.nn.functional.conv3d` definition:
/// `out[n, co, od, oh, ow] = sum_ci sum_kd,kh,kw w[co, ci, kd, kh, kw]
///     * x[n, ci, od*sd - pd + kd*dd, oh*sh - ph + kh*dh, ow*sw - pw + kw*dw]`
/// with out-of-range input positions contributing zero.
#[allow(clippy::too_many_arguments)]
fn conv3d_reference(
    x: &[f32],
    xd: [usize; 5],
    w: &[f32],
    wd: [usize; 5],
    padding: [usize; 3],
    stride: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
) -> (Vec<f32>, [usize; 5]) {
    let [b, c_in, i_d, i_h, i_w] = xd;
    let [c_out, c_in_g, k_d, k_h, k_w] = wd;
    assert_eq!(c_in, c_in_g * groups);
    assert_eq!(c_out % groups, 0);
    let c_out_g = c_out / groups;
    let o_d = out_dim(i_d, k_d, padding[0], stride[0], dilation[0]);
    let o_h = out_dim(i_h, k_h, padding[1], stride[1], dilation[1]);
    let o_w = out_dim(i_w, k_w, padding[2], stride[2], dilation[2]);
    let mut out = vec![0f32; b * c_out * o_d * o_h * o_w];
    for n in 0..b {
        for co in 0..c_out {
            let g = co / c_out_g;
            for od in 0..o_d {
                for oh in 0..o_h {
                    for ow in 0..o_w {
                        let mut acc = 0f32;
                        for ci in 0..c_in_g {
                            for kd in 0..k_d {
                                let sd = (od * stride[0] + kd * dilation[0]) as isize
                                    - padding[0] as isize;
                                if sd < 0 || sd >= i_d as isize {
                                    continue;
                                }
                                for kh in 0..k_h {
                                    let sh = (oh * stride[1] + kh * dilation[1]) as isize
                                        - padding[1] as isize;
                                    if sh < 0 || sh >= i_h as isize {
                                        continue;
                                    }
                                    for kw in 0..k_w {
                                        let sw = (ow * stride[2] + kw * dilation[2]) as isize
                                            - padding[2] as isize;
                                        if sw < 0 || sw >= i_w as isize {
                                            continue;
                                        }
                                        let xi = ((((n * c_in + g * c_in_g + ci) * i_d
                                            + sd as usize)
                                            * i_h
                                            + sh as usize)
                                            * i_w)
                                            + sw as usize;
                                        let wi = ((((co * c_in_g + ci) * k_d + kd) * k_h + kh)
                                            * k_w)
                                            + kw;
                                        acc += x[xi] * w[wi];
                                    }
                                }
                            }
                        }
                        out[((((n * c_out + co) * o_d + od) * o_h + oh) * o_w) + ow] = acc;
                    }
                }
            }
        }
    }
    (out, [b, c_out, o_d, o_h, o_w])
}

fn assert_close(got: &[f32], want: &[f32], eps: f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= eps,
            "index {i}: got {g}, want {w} (all: {got:?} vs {want:?})"
        );
    }
}

// out[0] = 1*10 + 2*100 = 210, out[1] = 2*10 + 3*100 = 320.
// A conv2d applied to a single temporal slice cannot produce this.
fn conv3d_temporal_by_hand(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[1f32, 2., 3.], dev)?.reshape((1, 1, 3, 1, 1))?;
    let w = Tensor::new(&[10f32, 100.], dev)?.reshape((1, 1, 2, 1, 1))?;
    let res = t.conv3d(&w, [0; 3], [1; 3], [1; 3], 1)?;
    assert_eq!(res.dims(), [1, 1, 2, 1, 1]);
    assert_close(&res.flatten_all()?.to_vec1::<f32>()?, &[210., 320.], 1e-4);
    Ok(())
}

// x = [1,2,3,4,5], k = [1,1] with dilation 2 over depth:
// out = [x0+x2, x1+x3, x2+x4] = [4, 6, 8].
fn conv3d_temporal_dilation_by_hand(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[1f32, 2., 3., 4., 5.], dev)?.reshape((1, 1, 5, 1, 1))?;
    let w = Tensor::new(&[1f32, 1.], dev)?.reshape((1, 1, 2, 1, 1))?;
    let res = t.conv3d(&w, [0; 3], [1; 3], [2, 1, 1], 1)?;
    assert_eq!(res.dims(), [1, 1, 3, 1, 1]);
    assert_close(&res.flatten_all()?.to_vec1::<f32>()?, &[4., 6., 8.], 1e-4);
    Ok(())
}

// x = [1,2,3,4], k = [1,1,1], padding 1 and stride 2 over depth:
// od=0 reads d = -1,0,1 -> 0+1+2 = 3; od=1 reads d = 1,2,3 -> 2+3+4 = 9.
fn conv3d_temporal_stride_padding_by_hand(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[1f32, 2., 3., 4.], dev)?.reshape((1, 1, 4, 1, 1))?;
    let w = Tensor::new(&[1f32, 1., 1.], dev)?.reshape((1, 1, 3, 1, 1))?;
    let res = t.conv3d(&w, [1, 0, 0], [2, 1, 1], [1; 3], 1)?;
    assert_eq!(res.dims(), [1, 1, 2, 1, 1]);
    assert_close(&res.flatten_all()?.to_vec1::<f32>()?, &[3., 9.], 1e-4);
    Ok(())
}

// Input (1, 1, 2, 2, 3) holding 1..=12 in NCDHW order, kernel (1, 1, 2, 1, 2) = [1, 2, 3, 4].
// out[oh, ow] = sum_kd,kw x[kd, oh, ow + kw] * k[kd, 0, kw]:
//   (0,0): (1*1 + 2*2) + (7*3 + 8*4) = 5 + 53 = 58
//   (0,1): (2*1 + 3*2) + (8*3 + 9*4) = 8 + 60 = 68
//   (1,0): (4*1 + 5*2) + (10*3 + 11*4) = 14 + 74 = 88
//   (1,1): (5*1 + 6*2) + (11*3 + 12*4) = 17 + 81 = 98
fn conv3d_non_cubic_kernel_by_hand(dev: &Device) -> Result<()> {
    let t = Tensor::arange(1f32, 13f32, dev)?.reshape((1, 1, 2, 2, 3))?;
    let w = Tensor::new(&[1f32, 2., 3., 4.], dev)?.reshape((1, 1, 2, 1, 2))?;
    let res = t.conv3d(&w, [0; 3], [1; 3], [1; 3], 1)?;
    assert_eq!(res.dims(), [1, 1, 1, 2, 2]);
    assert_close(
        &res.flatten_all()?.to_vec1::<f32>()?,
        &[58., 68., 88., 98.],
        1e-4,
    );
    Ok(())
}

/// A depth-1 kernel over a depth-1 input is the degenerate case a `Conv2d` reduction assumes; it
/// must agree with candle's own conv2d.
fn conv3d_depth1_matches_conv2d(dev: &Device) -> Result<()> {
    let x = pseudo_random([2, 3, 1, 7, 6].iter().product(), 11);
    let w = pseudo_random([4, 3, 1, 3, 3].iter().product(), 12);
    let t5 = Tensor::new(x.as_slice(), dev)?.reshape((2, 3, 1, 7, 6))?;
    let w5 = Tensor::new(w.as_slice(), dev)?.reshape((4, 3, 1, 3, 3))?;
    let res3 = t5.conv3d(&w5, [0, 1, 1], [1, 2, 2], [1; 3], 1)?;

    let t4 = t5.reshape((2, 3, 7, 6))?;
    let w4 = w5.reshape((4, 3, 3, 3))?;
    let res2 = t4.conv2d(&w4, 1, 2, 1, 1)?;
    assert_eq!(res3.dims(), [2, 4, 1, res2.dim(2)?, res2.dim(3)?]);
    assert_close(
        &res3.flatten_all()?.to_vec1::<f32>()?,
        &res2.flatten_all()?.to_vec1::<f32>()?,
        1e-4,
    );
    Ok(())
}

/// With `k_d == i_d` and no temporal padding the result is the sum over frames of the
/// corresponding per-frame conv2d. This pins T > 1 behaviour against candle's trusted conv2d.
fn conv3d_full_depth_matches_summed_conv2d(dev: &Device) -> Result<()> {
    let (b, c_in, c_out, i_d, i_h, i_w) = (2, 3, 4, 3, 6, 5);
    let x = pseudo_random(b * c_in * i_d * i_h * i_w, 21);
    let w = pseudo_random(c_out * c_in * i_d * 3 * 3, 22);
    let t5 = Tensor::new(x.as_slice(), dev)?.reshape((b, c_in, i_d, i_h, i_w))?;
    let w5 = Tensor::new(w.as_slice(), dev)?.reshape((c_out, c_in, i_d, 3, 3))?;
    let res3 = t5.conv3d(&w5, [0, 1, 1], [1; 3], [1; 3], 1)?;
    assert_eq!(res3.dims(), [b, c_out, 1, i_h, i_w]);

    let mut acc: Option<Tensor> = None;
    for d in 0..i_d {
        let frame = t5.narrow(2, d, 1)?.reshape((b, c_in, i_h, i_w))?;
        let k = w5.narrow(2, d, 1)?.reshape((c_out, c_in, 3, 3))?;
        let part = frame.conv2d(&k, 1, 1, 1, 1)?;
        acc = Some(match acc {
            None => part,
            Some(a) => (a + part)?,
        });
    }
    let expected = acc.unwrap();
    assert_close(
        &res3.flatten_all()?.to_vec1::<f32>()?,
        &expected.flatten_all()?.to_vec1::<f32>()?,
        1e-4,
    );
    Ok(())
}

/// Input dims, kernel dims, padding, stride, dilation, groups.
type Case = (
    [usize; 5],
    [usize; 5],
    [usize; 3],
    [usize; 3],
    [usize; 3],
    usize,
);

fn conv3d_against_reference(dev: &Device) -> Result<()> {
    #[rustfmt::skip]
    let cases: &[Case] = &[
        // T = 1: the degenerate case.
        ([2, 3, 1, 6, 5], [4, 3, 1, 3, 3], [0, 1, 1], [1; 3], [1; 3], 1),
        // T > 1, plain.
        ([2, 3, 4, 6, 5], [4, 3, 2, 3, 3], [0; 3], [1; 3], [1; 3], 1),
        // Stride > 1 on every axis.
        ([1, 2, 6, 7, 8], [3, 2, 3, 3, 3], [0; 3], [2; 3], [1; 3], 1),
        // Per-axis (asymmetric across axes) padding.
        ([1, 2, 5, 6, 7], [3, 2, 3, 2, 4], [1, 0, 2], [1; 3], [1; 3], 1),
        // Dilation > 1, mixed per axis.
        ([1, 2, 7, 7, 7], [2, 2, 2, 3, 2], [0; 3], [1; 3], [2, 1, 3], 1),
        // Non-cubic kernel with stride, padding and dilation all in play.
        ([2, 2, 6, 5, 9], [3, 2, 3, 1, 2], [2, 1, 0], [2, 1, 3], [1, 2, 2], 1),
        // Grouped, T > 1.
        ([2, 4, 5, 5, 5], [6, 2, 2, 3, 3], [1, 1, 1], [1; 3], [1; 3], 2),
        // Grouped with stride and dilation.
        ([1, 6, 5, 6, 6], [3, 2, 2, 2, 2], [0, 1, 1], [2, 1, 2], [2, 1, 1], 3),
        // 1x1x1 kernel, a pure channel mix.
        ([2, 5, 3, 4, 4], [7, 5, 1, 1, 1], [0; 3], [1; 3], [1; 3], 1),
    ];
    for (idx, (xd, wd, padding, stride, dilation, groups)) in cases.iter().enumerate() {
        let x = pseudo_random(xd.iter().product(), 100 + idx as u64);
        let w = pseudo_random(wd.iter().product(), 200 + idx as u64);
        let (expected, out_dims) =
            conv3d_reference(&x, *xd, &w, *wd, *padding, *stride, *dilation, *groups);
        let t = Tensor::new(x.as_slice(), dev)?.reshape(xd.to_vec())?;
        let k = Tensor::new(w.as_slice(), dev)?.reshape(wd.to_vec())?;
        let res = t.conv3d(&k, *padding, *stride, *dilation, *groups)?;
        assert_eq!(res.dims(), out_dims, "case {idx}: shape mismatch");
        assert_close(&res.flatten_all()?.to_vec1::<f32>()?, &expected, 1e-4);
    }
    Ok(())
}

/// The op must read through a non-contiguous input layout rather than the raw buffer order.
fn conv3d_non_contiguous_input(dev: &Device) -> Result<()> {
    let (b, c_in, c_out, i_d, i_h, i_w) = (2, 3, 4, 4, 5, 5);
    let x = pseudo_random(b * c_in * i_d * i_h * i_w, 31);
    let w = pseudo_random(c_out * c_in * 2 * 3 * 3, 32);
    // Build the tensor with H and W swapped, then transpose back into the wanted logical shape.
    let t = Tensor::new(x.as_slice(), dev)?
        .reshape((b, c_in, i_d, i_w, i_h))?
        .transpose(3, 4)?;
    let k = Tensor::new(w.as_slice(), dev)?.reshape((c_out, c_in, 2, 3, 3))?;
    let padding = [1, 1, 0];
    let stride = [1, 2, 1];
    let dilation = [1; 3];
    let res = t.conv3d(&k, padding, stride, dilation, 1)?;

    let x_c = t.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let (expected, out_dims) = conv3d_reference(
        &x_c,
        [b, c_in, i_d, i_h, i_w],
        &w,
        [c_out, c_in, 2, 3, 3],
        padding,
        stride,
        dilation,
        1,
    );
    assert_eq!(res.dims(), out_dims);
    assert_close(&res.flatten_all()?.to_vec1::<f32>()?, &expected, 1e-4);
    Ok(())
}

fn conv3d_bad_args(dev: &Device) -> Result<()> {
    let t = Tensor::zeros((1, 4, 3, 3, 3), candle_core::DType::F32, dev)?;
    let w = Tensor::zeros((2, 3, 2, 2, 2), candle_core::DType::F32, dev)?;
    assert!(t.conv3d(&w, [0; 3], [1; 3], [1; 3], 1).is_err());

    let w = Tensor::zeros((2, 4, 2, 2, 2), candle_core::DType::F32, dev)?;
    assert!(t.conv3d(&w, [0; 3], [0, 1, 1], [1; 3], 1).is_err());
    assert!(t.conv3d(&w, [0; 3], [1; 3], [0, 1, 1], 1).is_err());
    assert!(t.conv3d(&w, [0; 3], [1; 3], [1; 3], 0).is_err());
    // Kernel with dilation 4 does not fit in a depth-3 input.
    assert!(t.conv3d(&w, [0; 3], [1; 3], [4, 1, 1], 1).is_err());
    Ok(())
}

test_conv3d_device!(
    conv3d_temporal_by_hand,
    conv3d_temporal_by_hand_cpu,
    conv3d_temporal_by_hand_gpu
);
test_conv3d_device!(
    conv3d_temporal_dilation_by_hand,
    conv3d_temporal_dilation_by_hand_cpu,
    conv3d_temporal_dilation_by_hand_gpu
);
test_conv3d_device!(
    conv3d_temporal_stride_padding_by_hand,
    conv3d_temporal_stride_padding_by_hand_cpu,
    conv3d_temporal_stride_padding_by_hand_gpu
);
test_conv3d_device!(
    conv3d_non_cubic_kernel_by_hand,
    conv3d_non_cubic_kernel_by_hand_cpu,
    conv3d_non_cubic_kernel_by_hand_gpu
);
test_conv3d_device!(
    conv3d_depth1_matches_conv2d,
    conv3d_depth1_matches_conv2d_cpu,
    conv3d_depth1_matches_conv2d_gpu
);
test_conv3d_device!(
    conv3d_full_depth_matches_summed_conv2d,
    conv3d_full_depth_matches_summed_conv2d_cpu,
    conv3d_full_depth_matches_summed_conv2d_gpu
);
test_conv3d_device!(
    conv3d_against_reference,
    conv3d_against_reference_cpu,
    conv3d_against_reference_gpu
);
test_conv3d_device!(
    conv3d_non_contiguous_input,
    conv3d_non_contiguous_input_cpu,
    conv3d_non_contiguous_input_gpu
);
test_conv3d_device!(conv3d_bad_args, conv3d_bad_args_cpu, conv3d_bad_args_gpu);

/// Backward is not implemented; it must fail loudly rather than silently dropping the gradient.
#[test]
fn conv3d_backward_is_rejected() -> Result<()> {
    let dev = Device::Cpu;
    let t = candle_core::Var::from_tensor(
        &Tensor::new(&[1f32, 2., 3.], &dev)?.reshape((1, 1, 3, 1, 1))?,
    )?;
    let w = Tensor::new(&[1f32, 1.], &dev)?.reshape((1, 1, 2, 1, 1))?;
    let res = t.as_tensor().conv3d(&w, [0; 3], [1; 3], [1; 3], 1)?;
    let err = res.sum_all()?.backward().unwrap_err();
    assert!(
        err.to_string().contains("conv3d"),
        "unexpected error: {err}"
    );
    Ok(())
}
