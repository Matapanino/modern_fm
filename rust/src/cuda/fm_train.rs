//! CUDA FM mini-batch gradient accumulation — milestone 3 of
//! docs/gpu_backend_plan.md (binary logistic + squared loss), with the
//! plan-doc follow-ups: sparse touched-coordinate gradient buffers and
//! device-resident parameters.
//!
//! Two-stage contract: the GPU computes each batch's data-gradient
//! accumulation from the frozen batch-start parameters (score, loss gradient,
//! `atomicAdd` into gradient buffers); the existing CPU flush
//! (`crate::fm::FmGradAccum::flush`) then applies one optimizer step per
//! touched coordinate — SGD/AdaGrad/Adam/FTRL semantics and all optimizer
//! state stay exact and CPU-side, so early stopping and `partial_fit` state
//! hand-offs ride through unchanged.
//!
//! Transfers: the CSR arrays, `y`, `sample_weight`, `row_orders` and the
//! initial `w`/`V` upload once per call; the parameters then stay
//! device-resident. Per batch, one of two modes (chosen by transfer volume,
//! `2 * batch_nnz < n_features`):
//!
//! - **compact** — the host builds the batch's touched-feature slot map (it
//!   needs the touched set for the flush anyway), uploads the per-nonzero
//!   slot ids, accumulates into compact `T`-slot buffers, downloads only
//!   those, and after the CPU flush scatters only the touched parameters
//!   back to the device (lazy L2 means untouched coordinates do not move,
//!   so the device copy stays exact).
//! - **dense** — full-size gradient buffers download and a full parameter
//!   re-upload after the flush; cheaper than slot indirection once a batch
//!   touches most features (e.g. full-batch training).
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
/// `fm_train_accum_csr`: one block per batch row, one thread per factor `f`
/// (k <= 1024): pass 1 fills the shared factor cache
/// `cache[f] = sum_i v_{i,f} x_i` and the pairwise terms; thread 0 adds the
/// linear term, forms the score from the frozen parameters, applies the
/// (numerically stable) loss gradient and the sample weight, and accumulates
/// `g_w0`/`gw`; pass 2 has thread `f` add each nonzero's factor gradient
/// `g * (x * cache[f] - v_{i,f} * x^2)` into `gv`. `loss`: 0 = logistic
/// (y in {0,1}), 1 = squared. With `use_slots != 0` the gradient buffers are
/// compact: nonzero `p` of batch row `b` writes slot
/// `slots[batch_indptr[b] + (p - lo)]` instead of feature id `indices[p]`.
///
/// `fm_scatter_params`: writes the flushed values of the `n_touched` touched
/// coordinates (w and the k factors each) back into the device-resident
/// parameter arrays at feature base `base` (0 for the binary path; class
/// offset `c * n` for the multiclass path, which stacks per-class parameters).
pub(super) const KERNEL_SRC: &str = r#"
extern "C" __global__ void fm_train_accum_csr(
    const long long* indptr,
    const long long* indices,
    const double* data,
    const double* y,
    const double* sw,
    const long long* row_orders,
    const long long batch_start,
    const long long use_slots,
    const long long* slots,
    const long long* batch_indptr,
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
    const long long* row_slots = use_slots ? slots + batch_indptr[blockIdx.x] : 0;
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
            long long slot = use_slots ? row_slots[p - lo] : indices[p];
            atomicAdd(&gw[slot], g * data[p]);
        }
    }
    __syncthreads();
    if (f < k) {
        double g = gsh[0];
        for (long long p = lo; p < hi; ++p) {
            long long i = indices[p];
            long long slot = use_slots ? row_slots[p - lo] : i;
            double x = data[p];
            atomicAdd(&gv[slot * k + f], g * (x * cache[f] - v[i * k + f] * x * x));
        }
    }
}

extern "C" __global__ void fm_scatter_params(
    const long long* touched,
    const double* w_u,
    const double* v_u,
    const long long k,
    const long long n_touched,
    const long long base,
    double* w,
    double* v)
{
    long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n_touched * (k + 1);
    if (s >= total) {
        return;
    }
    long long t = s / (k + 1);
    long long j = s % (k + 1);
    long long i = base + touched[t];
    if (j == 0) {
        w[i] = w_u[t];
    } else {
        v[i * k + (j - 1)] = v_u[t * k + (j - 1)];
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
    // Pre-pass over the batch schedule: per-batch nnz decides compact vs
    // dense (compact when 2 * batch_nnz < n — bounded transfer win) and
    // sizes the compact buffers.
    let batch_nnz = |batch: &[i64]| -> usize {
        batch
            .iter()
            .map(|&r| (csr.indptr[r as usize + 1] - csr.indptr[r as usize]) as usize)
            .sum()
    };
    let mut max_compact_nnz = 0usize;
    let mut any_dense = false;
    let mut max_batch_len = 0usize;
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            let nnz_b = batch_nnz(batch);
            max_batch_len = max_batch_len.max(batch.len());
            if 2 * nnz_b < n {
                max_compact_nnz = max_compact_nnz.max(nnz_b);
            } else {
                any_dense = true;
            }
        }
    }
    let (ctx, module) = super::gpu()?;
    let accum_func = module
        .load_function("fm_train_accum_csr")
        .map_err(e("function load"))?;
    let scatter_func = module
        .load_function("fm_scatter_params")
        .map_err(e("scatter function load"))?;
    let stream = ctx.default_stream();
    // Static per-call uploads.
    let d_indptr = stream.clone_htod(csr.indptr).map_err(e("indptr upload"))?;
    let d_indices = stream.clone_htod(csr.indices).map_err(e("indices upload"))?;
    let d_data = stream.clone_htod(csr.data).map_err(e("data upload"))?;
    let d_y = stream.clone_htod(y).map_err(e("y upload"))?;
    let d_sw = stream.clone_htod(sample_weight).map_err(e("sample_weight upload"))?;
    let d_ro = stream.clone_htod(row_orders).map_err(e("row_orders upload"))?;
    // Device-resident parameters, kept in sync with the CPU copies after
    // every flush (scatter for compact batches, full re-upload for dense).
    let mut d_w = stream.clone_htod(&*w).map_err(e("w upload"))?;
    let mut d_v = stream.clone_htod(&*v).map_err(e("V upload"))?;
    let mut d_gw0 = stream.alloc_zeros::<f64>(1).map_err(e("g_w0 alloc"))?;
    // Dense gradient buffers only if some batch takes the dense path.
    let (mut d_gw, mut d_gv) = if any_dense {
        (
            stream.alloc_zeros::<f64>(n).map_err(e("gw alloc"))?,
            stream.alloc_zeros::<f64>(n * k).map_err(e("gv alloc"))?,
        )
    } else {
        (
            stream.alloc_zeros::<f64>(1).map_err(e("gw alloc"))?,
            stream.alloc_zeros::<f64>(1).map_err(e("gv alloc"))?,
        )
    };
    // Compact buffers sized to the largest compact batch (touched count T is
    // <= batch nnz). The 1-element fallbacks keep kernel-arg binding simple.
    let cap = max_compact_nnz.max(1);
    let mut d_slots = stream.alloc_zeros::<i64>(cap).map_err(e("slots alloc"))?;
    let mut d_bindptr = stream
        .alloc_zeros::<i64>(max_batch_len + 1)
        .map_err(e("batch_indptr alloc"))?;
    let mut d_touched = stream.alloc_zeros::<i64>(cap).map_err(e("touched alloc"))?;
    let mut d_gwc = stream.alloc_zeros::<f64>(cap).map_err(e("gw_c alloc"))?;
    let mut d_gvc = stream.alloc_zeros::<f64>(cap * k).map_err(e("gv_c alloc"))?;
    let mut d_wu = stream.alloc_zeros::<f64>(cap).map_err(e("w_u alloc"))?;
    let mut d_vu = stream.alloc_zeros::<f64>(cap * k).map_err(e("v_u alloc"))?;
    // Host scratch, reused across batches.
    let mut host_gw0 = vec![0.0f64; 1];
    let (mut host_gw, mut host_gv) = if any_dense {
        (vec![0.0f64; n], vec![0.0f64; n * k])
    } else {
        (Vec::new(), Vec::new())
    };
    let mut host_gwc = vec![0.0f64; cap];
    let mut host_gvc = vec![0.0f64; cap * k];
    let mut host_wu = vec![0.0f64; cap];
    let mut host_vu = vec![0.0f64; cap * k];
    let mut slots_flat: Vec<i64> = Vec::with_capacity(cap);
    let mut bindptr: Vec<i64> = Vec::with_capacity(max_batch_len + 1);
    let mut touched_scratch: Vec<usize> = Vec::with_capacity(cap);
    let mut slot_idx: Vec<i64> = vec![0; n];
    let mut accum = FmGradAccum::new(n, k);
    let mut batch_start: usize = 0;
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            // Build the batch's touched set (needed for the flush in both
            // modes) and the per-nonzero slot ids for the compact mode.
            slots_flat.clear();
            bindptr.clear();
            bindptr.push(0);
            for &r in batch {
                let (indices, _) = csr.row(r as usize);
                for &i in indices {
                    let i = i as usize;
                    if !accum.seen[i] {
                        accum.seen[i] = true;
                        slot_idx[i] = accum.touched.len() as i64;
                        accum.touched.push(i);
                    }
                    slots_flat.push(slot_idx[i]);
                }
                bindptr.push(slots_flat.len() as i64);
            }
            let t_count = accum.touched.len();
            let nnz_b = slots_flat.len();
            let compact = 2 * nnz_b < n;
            stream.memset_zeros(&mut d_gw0).map_err(e("g_w0 zero"))?;
            if compact {
                stream.memcpy_htod(&slots_flat[..], &mut d_slots).map_err(e("slots upload"))?;
                stream.memcpy_htod(&bindptr[..], &mut d_bindptr).map_err(e("batch_indptr upload"))?;
                let mut gwc_view = d_gwc.slice_mut(..t_count);
                stream.memset_zeros(&mut gwc_view).map_err(e("gw_c zero"))?;
                let mut gvc_view = d_gvc.slice_mut(..t_count * k);
                stream.memset_zeros(&mut gvc_view).map_err(e("gv_c zero"))?;
            } else {
                stream.memset_zeros(&mut d_gw).map_err(e("gw zero"))?;
                stream.memset_zeros(&mut d_gv).map_err(e("gv zero"))?;
            }
            let cfg = LaunchConfig {
                grid_dim: (batch.len() as u32, 1, 1),
                block_dim: (k as u32, 1, 1),
                shared_mem_bytes: ((2 * k + 1) * std::mem::size_of::<f64>()) as u32,
            };
            let batch_start_i64 = batch_start as i64;
            let use_slots: i64 = if compact { 1 } else { 0 };
            let k_i64 = k as i64;
            let w0_val = *w0;
            {
                let (gw_buf, gv_buf) = if compact {
                    (&mut d_gwc, &mut d_gvc)
                } else {
                    (&mut d_gw, &mut d_gv)
                };
                let mut launch = stream.launch_builder(&accum_func);
                launch
                    .arg(&d_indptr)
                    .arg(&d_indices)
                    .arg(&d_data)
                    .arg(&d_y)
                    .arg(&d_sw)
                    .arg(&d_ro)
                    .arg(&batch_start_i64)
                    .arg(&use_slots)
                    .arg(&d_slots)
                    .arg(&d_bindptr)
                    .arg(&d_w)
                    .arg(&d_v)
                    .arg(&w0_val)
                    .arg(&k_i64)
                    .arg(&loss_code)
                    .arg(&mut d_gw0)
                    .arg(gw_buf)
                    .arg(gv_buf);
                // Safety: the kernel reads/writes exactly the buffers bound
                // above; CSR structure, row_orders range and w/V shapes are
                // validated by the PyO3 layer, and slot ids are < t_count by
                // construction.
                unsafe { launch.launch(cfg) }.map_err(e("kernel launch"))?;
            }
            stream.memcpy_dtoh(&d_gw0, &mut host_gw0).map_err(e("g_w0 download"))?;
            accum.g_w0 = host_gw0[0];
            if compact {
                stream
                    .memcpy_dtoh(&d_gwc.slice(..t_count), &mut host_gwc[..t_count])
                    .map_err(e("gw_c download"))?;
                stream
                    .memcpy_dtoh(&d_gvc.slice(..t_count * k), &mut host_gvc[..t_count * k])
                    .map_err(e("gv_c download"))?;
                for (s, &i) in accum.touched.iter().enumerate() {
                    accum.gw[i] = host_gwc[s];
                    accum.gv[i * k..(i + 1) * k].copy_from_slice(&host_gvc[s * k..(s + 1) * k]);
                }
            } else {
                stream.memcpy_dtoh(&d_gw, &mut host_gw).map_err(e("gw download"))?;
                stream.memcpy_dtoh(&d_gv, &mut host_gv).map_err(e("gv download"))?;
                for &i in &accum.touched {
                    accum.gw[i] = host_gw[i];
                    accum.gv[i * k..(i + 1) * k].copy_from_slice(&host_gv[i * k..(i + 1) * k]);
                }
            }
            // The flush clears `touched`; keep a copy for the scatter-back.
            touched_scratch.clear();
            touched_scratch.extend_from_slice(&accum.touched);
            accum.flush(
                batch.len() as f64, w0, w, v, acc_w0, acc_w, acc_v, &mut adam, &mut ftrl, k, opt,
                lr, l1_linear, l2_linear, l1_factors, l2_factors,
            );
            // Re-sync the device-resident parameters with the flushed values.
            // (t_count == 0 — an all-empty-rows batch — only moves w0, which
            // travels by value into every launch; nothing to scatter.)
            if compact && t_count > 0 {
                for (s, &i) in touched_scratch.iter().enumerate() {
                    host_wu[s] = w[i];
                    host_vu[s * k..(s + 1) * k].copy_from_slice(&v[i * k..(i + 1) * k]);
                    slots_flat[s] = i as i64; // reuse as the touched-id upload
                }
                stream
                    .memcpy_htod(&slots_flat[..t_count], &mut d_touched)
                    .map_err(e("touched upload"))?;
                stream.memcpy_htod(&host_wu[..t_count], &mut d_wu).map_err(e("w_u upload"))?;
                stream
                    .memcpy_htod(&host_vu[..t_count * k], &mut d_vu)
                    .map_err(e("v_u upload"))?;
                let total = (t_count * (k + 1)) as u32;
                let scatter_cfg = LaunchConfig {
                    grid_dim: (total.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let t_i64 = t_count as i64;
                let base: i64 = 0;
                let mut launch = stream.launch_builder(&scatter_func);
                launch
                    .arg(&d_touched)
                    .arg(&d_wu)
                    .arg(&d_vu)
                    .arg(&k_i64)
                    .arg(&t_i64)
                    .arg(&base)
                    .arg(&mut d_w)
                    .arg(&mut d_v);
                // Safety: scatter writes w[touched[t]] and the k factors of
                // each touched feature; touched ids are valid feature ids by
                // construction.
                unsafe { launch.launch(scatter_cfg) }.map_err(e("scatter launch"))?;
            } else if !compact {
                stream.memcpy_htod(&*w, &mut d_w).map_err(e("w re-upload"))?;
                stream.memcpy_htod(&*v, &mut d_v).map_err(e("V re-upload"))?;
            }
            batch_start += batch.len();
        }
    }
    Ok(())
}
