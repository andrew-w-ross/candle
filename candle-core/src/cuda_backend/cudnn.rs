use crate::WithDType;
use cudarc;
use cudarc::cudnn::safe::{ConvForward, Cudnn};
use cudarc::driver::{CudaSlice, CudaView, DeviceRepr, ValidAsZeroBits};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// The cudnn handles are stored per thread here rather than on the CudaDevice as they are neither
// send nor sync.
thread_local! {
    static CUDNN: RefCell<HashMap<crate::cuda_backend::DeviceId, Arc<Cudnn>>> = HashMap::new().into();
}

impl From<cudarc::cudnn::CudnnError> for crate::Error {
    fn from(err: cudarc::cudnn::CudnnError) -> Self {
        crate::Error::wrap(err)
    }
}

impl From<cudarc::driver::DriverError> for crate::Error {
    fn from(err: cudarc::driver::DriverError) -> Self {
        crate::Error::wrap(err)
    }
}

/// Without this cuDNN keeps `CUDNN_DEFAULT_MATH` and the v7 heuristic picks a non-tensor-core
/// kernel (`implicit_convolve_hhgemm`) for half-precision convolutions.
fn use_tensor_op_math<T: WithDType>() -> bool {
    matches!(T::DTYPE, crate::DType::F16 | crate::DType::BF16)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConvPlanKey {
    dtype: crate::DType,
    compute: cudarc::cudnn::sys::cudnnDataType_t,
    x_shape: [i32; 4],
    x_stride: Option<[i32; 4]>,
    w_shape: [i32; 4],
    y_shape: [i32; 4],
    pad: [i32; 2],
    stride: [i32; 2],
    dilation: [i32; 2],
}

type ConvPlan = (cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t, usize);

thread_local! {
    static CONV_PLANS: RefCell<HashMap<ConvPlanKey, ConvPlan>> = HashMap::new().into();
}

/// `pick_algorithm` and `get_workspace_size` are host-side cuDNN heuristic queries that only
/// depend on the convolution geometry, so they are resolved once per distinct shape.
fn conv_plan<X, C, Y>(key: ConvPlanKey, conv: &ConvForward<'_, X, C, Y>) -> crate::Result<ConvPlan>
where
    X: cudarc::cudnn::CudnnDataType,
    C: cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
{
    if let Some(plan) = CONV_PLANS.with(|plans| plans.borrow().get(&key).copied()) {
        return Ok(plan);
    }
    let alg = conv.pick_algorithm()?;
    let plan = (alg, conv.get_workspace_size(alg)?);
    CONV_PLANS.with(|plans| plans.borrow_mut().insert(key, plan));
    Ok(plan)
}

pub(crate) fn launch_conv2d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    use crate::conv::CudnnFwdAlgo as CandleAlgo;
    use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t as A;

    let device_id = dev.id();
    let cudnn = CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?;
    let mut conv = cudnn.create_conv2d::<Y>(
        /* pad */ [params.padding as i32, params.padding as i32],
        /* stride */ [params.stride as i32, params.stride as i32],
        /* dilation */ [params.dilation as i32, params.dilation as i32],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    if use_tensor_op_math::<T>() {
        conv.set_math_type(cudarc::cudnn::sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH)?;
    }
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.i_h as i32,
        params.i_w as i32,
    ];
    let x_stride = if src_l.is_contiguous() {
        None
    } else {
        let s = src_l.stride();
        Some([s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32])
    };
    // Note that `src` already starts at the proper offset.
    let x = match x_stride {
        None => cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            x_shape,
        )?,
        Some(s) => cudnn.create_4d_tensor_ex::<T>(x_shape, s)?,
    };
    let w_shape = [
        params.c_out as i32,
        params.c_in as i32,
        params.k_h as i32,
        params.k_w as i32,
    ];
    let w = cudnn.create_4d_filter::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        w_shape,
    )?;
    let (w_out, h_out) = (params.out_w() as i32, params.out_h() as i32);
    let y_shape = [params.b_size as i32, params.c_out as i32, h_out, w_out];
    let y = cudnn.create_4d_tensor::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        y_shape,
    )?;
    let conv2d = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let key = ConvPlanKey {
        dtype: T::DTYPE,
        compute: Y::DATA_TYPE,
        x_shape,
        x_stride,
        w_shape,
        y_shape,
        pad: [params.padding as i32, params.padding as i32],
        stride: [params.stride as i32, params.stride as i32],
        dilation: [params.dilation as i32, params.dilation as i32],
    };
    let (alg, workspace_size) = match params.cudnn_fwd_algo {
        None => conv_plan(key, &conv2d)?,
        Some(forced) => {
            let alg = match forced {
                CandleAlgo::ImplicitGemm => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                CandleAlgo::ImplicitPrecompGemm => {
                    A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
                }
                CandleAlgo::Gemm => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                CandleAlgo::Direct => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
                CandleAlgo::Fft => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
                CandleAlgo::FftTiling => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
                CandleAlgo::Winograd => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                CandleAlgo::WinogradNonFused => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
                CandleAlgo::Count => A::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
            };
            (alg, conv2d.get_workspace_size(alg)?)
        }
    };
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        conv2d.launch::<CudaSlice<u8>, _, _, _>(
            alg,
            Some(&mut workspace),
            (T::one(), T::zero()),
            src,
            filter,
            dst,
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Conv3dPlanKey {
    dtype: crate::DType,
    compute: cudarc::cudnn::sys::cudnnDataType_t,
    x_shape: [i32; 5],
    x_stride: [i32; 5],
    w_shape: [i32; 5],
    y_shape: [i32; 5],
    pad: [i32; 3],
    stride: [i32; 3],
    dilation: [i32; 3],
}

thread_local! {
    static CONV3D_PLANS: RefCell<HashMap<Conv3dPlanKey, ConvPlan>> = HashMap::new().into();
}

fn conv3d_plan<X, C, Y>(
    key: Conv3dPlanKey,
    conv: &ConvForward<'_, X, C, Y>,
) -> crate::Result<ConvPlan>
where
    X: cudarc::cudnn::CudnnDataType,
    C: cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
{
    if let Some(plan) = CONV3D_PLANS.with(|plans| plans.borrow().get(&key).copied()) {
        return Ok(plan);
    }
    let alg = conv.pick_algorithm()?;
    let plan = (alg, conv.get_workspace_size(alg)?);
    CONV3D_PLANS.with(|plans| plans.borrow_mut().insert(key, plan));
    Ok(plan)
}

pub(crate) fn launch_conv3d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv3D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    use crate::conv::CudnnFwdAlgo as CandleAlgo;
    use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t as A;

    let device_id = dev.id();
    let cudnn = CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?;
    let pad = [
        params.padding[0] as i32,
        params.padding[1] as i32,
        params.padding[2] as i32,
    ];
    let stride = [
        params.stride[0] as i32,
        params.stride[1] as i32,
        params.stride[2] as i32,
    ];
    let dilation = [
        params.dilation[0] as i32,
        params.dilation[1] as i32,
        params.dilation[2] as i32,
    ];
    // The spatial rank of the convolution descriptor must match the tensor and filter ranks
    // minus the two leading (N, C) dimensions, so 3 here against 5D descriptors.
    let mut conv = cudnn.create_convnd::<Y>(
        &pad,
        &stride,
        &dilation,
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    if use_tensor_op_math::<T>() {
        conv.set_math_type(cudarc::cudnn::sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH)?;
    }
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.i_d as i32,
        params.i_h as i32,
        params.i_w as i32,
    ];
    // `create_nd_tensor` has no format-based variant, so NCDHW strides are always passed
    // explicitly; `src` already starts at the proper offset.
    let x_stride = {
        let s = src_l.stride();
        [
            s[0] as i32,
            s[1] as i32,
            s[2] as i32,
            s[3] as i32,
            s[4] as i32,
        ]
    };
    let x = cudnn.create_nd_tensor::<T>(&x_shape, &x_stride)?;
    let w_shape = [
        params.c_out as i32,
        params.c_in as i32,
        params.k_d as i32,
        params.k_h as i32,
        params.k_w as i32,
    ];
    let w = cudnn.create_nd_filter::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        &w_shape,
    )?;
    let (d_out, h_out, w_out) = (
        params.out_d() as i32,
        params.out_h() as i32,
        params.out_w() as i32,
    );
    let y_shape = [
        params.b_size as i32,
        params.c_out as i32,
        d_out,
        h_out,
        w_out,
    ];
    let y_stride = [
        params.c_out as i32 * d_out * h_out * w_out,
        d_out * h_out * w_out,
        h_out * w_out,
        w_out,
        1,
    ];
    let y = cudnn.create_nd_tensor::<T>(&y_shape, &y_stride)?;
    let conv3d = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let key = Conv3dPlanKey {
        dtype: T::DTYPE,
        compute: Y::DATA_TYPE,
        x_shape,
        x_stride,
        w_shape,
        y_shape,
        pad,
        stride,
        dilation,
    };
    let (alg, workspace_size) = match params.cudnn_fwd_algo {
        None => conv3d_plan(key, &conv3d)?,
        Some(forced) => {
            let alg = match forced {
                CandleAlgo::ImplicitGemm => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                CandleAlgo::ImplicitPrecompGemm => {
                    A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
                }
                CandleAlgo::Gemm => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                CandleAlgo::Direct => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
                CandleAlgo::Fft => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
                CandleAlgo::FftTiling => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
                CandleAlgo::Winograd => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                CandleAlgo::WinogradNonFused => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
                CandleAlgo::Count => A::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
            };
            (alg, conv3d.get_workspace_size(alg)?)
        }
    };
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        conv3d.launch::<CudaSlice<u8>, _, _, _>(
            alg,
            Some(&mut workspace),
            (T::one(), T::zero()),
            src,
            filter,
            dst,
        )?;
    }
    Ok(())
}

pub(crate) fn launch_conv1d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv1D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    use crate::conv::CudnnFwdAlgo as CandleAlgo;
    use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t as A;

    let device_id = dev.id();
    let cudnn = CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?;
    let mut conv = cudnn.create_conv2d::<Y>(
        /* pad */ [params.padding as i32, 0],
        /* stride */ [params.stride as i32, 1],
        /* dilation */ [params.dilation as i32, 1],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    if use_tensor_op_math::<T>() {
        conv.set_math_type(cudarc::cudnn::sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH)?;
    }
    // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnsettensornddescriptor
    // > Tensors are restricted to having at least 4 dimensions, and at most CUDNN_DIM_MAX
    // > dimensions (defined in cudnn.h). When working with lower dimensional data, it is
    // > recommended that the user create a 4D tensor, and set the size along unused dimensions
    // > to 1.
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.l_in as i32,
        1,
    ];
    let x_stride = if src_l.is_contiguous() {
        None
    } else {
        let s = src_l.stride();
        Some([s[0] as i32, s[1] as i32, s[2] as i32, 1i32])
    };
    // Note that `src` already starts at the proper offset.
    let x = match x_stride {
        None => cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            x_shape,
        )?,
        Some(s) => cudnn.create_4d_tensor_ex::<T>(x_shape, s)?,
    };
    let w_shape = [
        params.c_out as i32,
        params.c_in as i32,
        params.k_size as i32,
        1,
    ];
    let w = cudnn.create_4d_filter::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        w_shape,
    )?;
    let l_out = params.l_out() as i32;
    let y_shape = [params.b_size as i32, params.c_out as i32, l_out, 1];
    let y = cudnn.create_4d_tensor::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        y_shape,
    )?;
    let conv1d = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let key = ConvPlanKey {
        dtype: T::DTYPE,
        compute: Y::DATA_TYPE,
        x_shape,
        x_stride,
        w_shape,
        y_shape,
        pad: [params.padding as i32, 0],
        stride: [params.stride as i32, 1],
        dilation: [params.dilation as i32, 1],
    };
    let (alg, workspace_size) = match params.cudnn_fwd_algo {
        None => conv_plan(key, &conv1d)?,
        Some(forced) => {
            let alg = match forced {
                CandleAlgo::ImplicitGemm => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                CandleAlgo::ImplicitPrecompGemm => {
                    A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
                }
                CandleAlgo::Gemm => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                CandleAlgo::Direct => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
                CandleAlgo::Fft => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
                CandleAlgo::FftTiling => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
                CandleAlgo::Winograd => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                CandleAlgo::WinogradNonFused => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
                CandleAlgo::Count => A::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
            };
            (alg, conv1d.get_workspace_size(alg)?)
        }
    };
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        conv1d.launch::<CudaSlice<u8>, _, _, _>(
            alg,
            Some(&mut workspace),
            (T::one(), T::zero()),
            src,
            filter,
            dst,
        )?;
    }
    Ok(())
}
