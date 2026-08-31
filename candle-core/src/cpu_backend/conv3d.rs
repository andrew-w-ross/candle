use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

use crate::{conv::ParamsConv3D, cpu_backend::Map2, Layout, Result, WithDType};

pub(super) struct Conv3D<'a>(pub(super) &'a ParamsConv3D);

impl Map2 for Conv3D<'_> {
    const OP: &'static str = "conv3d";
    fn f<T: WithDType + num_traits::Num + Copy>(
        &self,
        inp: &[T],
        inp_l: &Layout,
        k: &[T],
        k_l: &Layout,
    ) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let k = &k[k_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5(inp_l.stride())?;
        let (k_s0, k_s1, k_s2, k_s3, k_s4) = dims5(k_l.stride())?;
        let (out_d, out_h, out_w) = (p.out_d(), p.out_h(), p.out_w());
        let [pad_d, pad_h, pad_w] = p.padding;
        let [str_d, str_h, str_w] = p.stride;
        let [dil_d, dil_h, dil_w] = p.dilation;

        let out_spatial = out_d * out_h * out_w;
        let mut dst = vec![T::zero(); p.b_size * p.c_out * out_spatial];
        dst.par_chunks_mut(out_spatial)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let b_idx = chunk_idx / p.c_out;
                let dst_c_idx = chunk_idx % p.c_out;
                for dst_d in 0..out_d {
                    for dst_h in 0..out_h {
                        for dst_w in 0..out_w {
                            let mut acc = T::zero();
                            for offset_d in 0..p.k_d {
                                let src_d = str_d * dst_d + offset_d * dil_d;
                                if src_d < pad_d || src_d >= p.i_d + pad_d {
                                    continue;
                                }
                                let src_d = src_d - pad_d;
                                for offset_h in 0..p.k_h {
                                    let src_h = str_h * dst_h + offset_h * dil_h;
                                    if src_h < pad_h || src_h >= p.i_h + pad_h {
                                        continue;
                                    }
                                    let src_h = src_h - pad_h;
                                    for offset_w in 0..p.k_w {
                                        let src_w = str_w * dst_w + offset_w * dil_w;
                                        if src_w < pad_w || src_w >= p.i_w + pad_w {
                                            continue;
                                        }
                                        let src_w = src_w - pad_w;
                                        let inp_base = b_idx * inp_s0
                                            + src_d * inp_s2
                                            + src_h * inp_s3
                                            + src_w * inp_s4;
                                        let k_base = dst_c_idx * k_s0
                                            + offset_d * k_s2
                                            + offset_h * k_s3
                                            + offset_w * k_s4;
                                        for c_in_idx in 0..p.c_in {
                                            acc += inp[inp_base + c_in_idx * inp_s1]
                                                * k[k_base + c_in_idx * k_s1];
                                        }
                                    }
                                }
                            }
                            dst[dst_d * out_h * out_w + dst_h * out_w + dst_w] = acc;
                        }
                    }
                }
            });
        Ok(dst)
    }
}

fn dims5(dims: &[usize]) -> Result<(usize, usize, usize, usize, usize)> {
    match dims {
        [d0, d1, d2, d3, d4] => Ok((*d0, *d1, *d2, *d3, *d4)),
        _ => crate::bail!("expected 5 dimensions, got {dims:?}"),
    }
}
