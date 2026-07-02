//! CUDA FM mini-batch gradient accumulation — milestone 3 of
//! docs/gpu_backend_plan.md (binary logistic + squared loss).
//!
//! Two-stage contract, mirroring the plan doc: the GPU computes each batch's
//! data-gradient accumulation from the frozen batch-start parameters (score,
//! loss gradient, dense `gw`/`gv`/`g_w0` buffers via `atomicAdd`); the
//! existing CPU flush (`crate::fm::FmGradAccum::flush`) then applies one
//! optimizer step per touched coordinate — SGD/AdaGrad/Adam/FTRL semantics
//! and all optimizer state stay exact and CPU-side, so early stopping and
//! `partial_fit` state hand-offs ride through unchanged.
//!
//! Transfers: the CSR arrays, `y`, `sample_weight` and `row_orders` upload
//! once per call; `w`/`V` re-upload and the dense gradient buffers download
//! every batch (the parameters change at each flush). Device-resident
//! parameters are a later milestone.
//!
//! Determinism: `atomicAdd` ordering is scheduler-dependent, so two identical
//! CUDA fits differ in float rounding run-to-run (unlike `n_jobs>1` on CPU,
//! which fixes the reduction order). Parity vs the CPU kernel is
//! tolerance-based on final predictions (tests/test_cuda_parity.py, rtol 1e-7
//! / atol 1e-8 per the plan doc). `atomicAdd(double*, double)` requires
//! compute capability >= 6.0 — the shared module is compiled for compute_60
//! (see `super::gpu`).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::data::CsrView;
use crate::fm::FmGradAccum;
use crate::optimizer::{AdamStateMut, FtrlStateMut, Loss, Optimizer};

/// Compiled once per process into the shared module (`super::gpu`).
///
/// One block per batch row, one thread per factor `f` (k <= 1024): pass 1
/// fills the shared factor cache `cache[f] = sum_i v_{i,f} x_i` and the
/// pairwise terms; thread 0 adds the linear term, forms the score from the
/// frozen parameters, applies the (numerically stable) loss gradient and the
/// sample weight, and accumulates `g_w0`/`gw`; pass 2 has thread `f` add each
/// nonzero's factor gradient `g * (x * cache[f] - v_{i,f} * x^2)` into `gv`.
/// `loss`: 0 = logistic (y in {0,1}), 1 = squared.
pub(super) const KERNEL_SRC: &str = r#"
extern "C" __global__ void fm_train_accum_csr(
    const long long* indptr,
    const long long* indices,
    const double* data,
    const double* y,
    const double* sw,
    const long long* row_orders,
    const long long batch_start,
    const double* w,
    const double* v,
    const double w0,
    const long long k,
    const long long loss,
    double* g_w0,
    double* gw,
    double* gv)
{
    long long r = row_orders[batch_start + blockIdx.x];
    long long lo = indptr[r];
    long long hi = indptr[r + 1];
    extern __shared__ double shm[];  // cache[k] | pair[k] | g
    double* cache = shm;
    double* pair = shm + k;
    double* gsh = shm + 2 * k;
    long long f = threadIdx.x;
    if (f < k) {
        double sum = 0.0;
        double sq = 0.0;
        for (long long p = lo; p < hi; ++p) {
            double vx = v[indices[p] * k + f] * data[p];
            sum += vx;
            sq += vx * vx;
        }
        cache[f] = sum;
        pair[f] = sum * sum - sq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double lin = 0.0;
        for (long long p = lo; p < hi; ++p) {
            lin += w[indices[p]] * data[p];
        }
        double pw = 0.0;
        for (long long ff = 0; ff < k; ++ff) {
            pw += pair[ff];
        }
        double s = w0 + lin + 0.5 * pw;
        double g;
        if (loss == 0) {
            double prob;
            if (s >= 0.0) {
                prob = 1.0 / (1.0 + exp(-s));
            } else {
                double e = exp(s);
                prob = e / (1.0 + e);
            }
            g = prob - y[r];
        } else {
            g = s - y[r];
        }
        g *= sw[r];
        gsh[0] = g;
        atomicAdd(g_w0, g);
        for (long long p = lo; p < hi; ++p) {
            atomicAdd(&gw[indices[p]], g * data[p]);
        }
    }
    __syncthreads();
    if (f < k) {
        double g = gsh[0];
        for (long long p = lo; p < hi; ++p) {
            long long i = indices[p];
            double x = data[p];
            atomicAdd(&gv[i * k + f], g * (x * cache[f] - v[i * k + f] * x * x));
        }
    }
}
"#;

/// Train an FM in place with CUDA batch accumulation + the CPU optimizer
/// flush. Argument contract matches `crate::fm::fit_csr` (no `n_jobs`: the
/// GPU replaces row-parallelism). Errors are stringified for the PyO3 layer.
#[allow(clippy::too_many_arguments)]
pub fn fit_csr(
    csr: &CsrView,
    y: &[f64],
    sample_weight: &[f64],
    w0: &mut f64,
    w: &mut [f64],
    v: &mut [f64],
    acc_w0: &mut f64,
    acc_w: &mut [f64],
    acc_v: &mut [f64],
    mut adam: AdamStateMut<'_>,
    mut ftrl: FtrlStateMut<'_>,
    k: usize,
    loss: Loss,
    opt: Optimizer,
    lr: f64,
    l1_linear: f64,
    l2_linear: f64,
    l1_factors: f64,
    l2_factors: f64,
    n_rows: usize,
    batch_size: usize,
    row_orders: &[i64],
) -> Result<(), String> {
    if k == 0 || k > 1024 {
        return Err(format!("CUDA FM training supports 1 <= k <= 1024, got {k}"));
    }
    fn e<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> String {
        move |err| format!("CUDA {what} failed: {err:?}")
    }
    let n = w.len();
    let loss_code: i64 = match loss {
        Loss::Logistic => 0,
        Loss::Squared => 1,
    };
    let (ctx, module) = super::gpu()?;
    let func = module
        .load_function("fm_train_accum_csr")
        .map_err(e("function load"))?;
    let stream = ctx.default_stream();
    // Static per-call uploads: CSR + targets + weights + the full multi-epoch
    // row order.
    let d_indptr = stream.clone_htod(csr.indptr).map_err(e("indptr upload"))?;
    let d_indices = stream.clone_htod(csr.indices).map_err(e("indices upload"))?;
    let d_data = stream.clone_htod(csr.data).map_err(e("data upload"))?;
    let d_y = stream.clone_htod(y).map_err(e("y upload"))?;
    let d_sw = stream.clone_htod(sample_weight).map_err(e("sample_weight upload"))?;
    let d_ro = stream.clone_htod(row_orders).map_err(e("row_orders upload"))?;
    // Per-batch buffers, allocated once and reused.
    let mut d_w = stream.alloc_zeros::<f64>(n).map_err(e("w alloc"))?;
    let mut d_v = stream.alloc_zeros::<f64>(n * k).map_err(e("V alloc"))?;
    let mut d_gw0 = stream.alloc_zeros::<f64>(1).map_err(e("g_w0 alloc"))?;
    let mut d_gw = stream.alloc_zeros::<f64>(n).map_err(e("gw alloc"))?;
    let mut d_gv = stream.alloc_zeros::<f64>(n * k).map_err(e("gv alloc"))?;
    let mut host_gw0 = vec![0.0; 1];
    let mut host_gw = vec![0.0; n];
    let mut host_gv = vec![0.0; n * k];
    let mut accum = FmGradAccum::new(n, k);
    let mut batch_start: usize = 0;
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            stream.memcpy_htod(&*w, &mut d_w).map_err(e("w upload"))?;
            stream.memcpy_htod(&*v, &mut d_v).map_err(e("V upload"))?;
            stream.memset_zeros(&mut d_gw0).map_err(e("g_w0 zero"))?;
            stream.memset_zeros(&mut d_gw).map_err(e("gw zero"))?;
            stream.memset_zeros(&mut d_gv).map_err(e("gv zero"))?;
            let cfg = LaunchConfig {
                grid_dim: (batch.len() as u32, 1, 1),
                block_dim: (k as u32, 1, 1),
                shared_mem_bytes: ((2 * k + 1) * std::mem::size_of::<f64>()) as u32,
            };
            let batch_start_i64 = batch_start as i64;
            let k_i64 = k as i64;
            let w0_val = *w0;
            let mut launch = stream.launch_builder(&func);
            launch
                .arg(&d_indptr)
                .arg(&d_indices)
                .arg(&d_data)
                .arg(&d_y)
                .arg(&d_sw)
                .arg(&d_ro)
                .arg(&batch_start_i64)
                .arg(&d_w)
                .arg(&d_v)
                .arg(&w0_val)
                .arg(&k_i64)
                .arg(&loss_code)
                .arg(&mut d_gw0)
                .arg(&mut d_gw)
                .arg(&mut d_gv);
            // Safety: the kernel reads/writes exactly the buffers bound above;
            // CSR structure, row_orders range and w/V shapes are validated by
            // the PyO3 layer before this call.
            unsafe { launch.launch(cfg) }.map_err(e("kernel launch"))?;
            stream.memcpy_dtoh(&d_gw0, &mut host_gw0).map_err(e("g_w0 download"))?;
            stream.memcpy_dtoh(&d_gw, &mut host_gw).map_err(e("gw download"))?;
            stream.memcpy_dtoh(&d_gv, &mut host_gv).map_err(e("gv download"))?;
            // Rebuild the touched-feature set on the host (order is irrelevant
            // to flush: coordinate updates are independent) and load the
            // device-accumulated gradients into the shared CPU accumulator.
            accum.g_w0 = host_gw0[0];
            for &r in batch {
                let (indices, _) = csr.row(r as usize);
                for &i in indices {
                    let i = i as usize;
                    if !accum.seen[i] {
                        accum.seen[i] = true;
                        accum.touched.push(i);
                        accum.gw[i] = host_gw[i];
                        accum.gv[i * k..(i + 1) * k].copy_from_slice(&host_gv[i * k..(i + 1) * k]);
                    }
                }
            }
            accum.flush(
                batch.len() as f64, w0, w, v, acc_w0, acc_w, acc_v, &mut adam, &mut ftrl, k, opt,
                lr, l1_linear, l2_linear, l1_factors, l2_factors,
            );
            batch_start += batch.len();
        }
    }
    Ok(())
}
