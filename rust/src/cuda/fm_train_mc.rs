//! CUDA FM multiclass (softmax) mini-batch gradient accumulation — milestone
//! 5 of docs/gpu_backend_plan.md.
//!
//! Same two-stage contract as the binary path (`cuda::fm_train`): the GPU
//! accumulates each batch's data-gradients for all classes from the frozen
//! batch-start parameters, the untouched CPU flush
//! (`crate::fm::FmGradAccum::flush` over `McState::class_views`) applies the
//! optimizer per class — SGD/AdaGrad/Adam/FTRL semantics, label smoothing
//! targets and the early-stopping / `partial_fit` state hand-offs ride through
//! unchanged from `crate::fm::fit_multiclass_csr`.
//!
//! Layout: `w0` is (C,), `w` (C, n) and `v` (C, n, k) row-major, all
//! device-resident like the binary path. The touched-feature set is shared
//! across classes (every row contributes to every class), so one slot map
//! serves all C classes and the gradient buffers are C-stacked: compact
//! `(C, T)` / `(C, T, k)` when `2 * batch_nnz < n_features`, dense
//! `(C, n)` / `(C, n, k)` otherwise. Dense mode only triggers when a batch
//! touches most of `n` (i.e. `n` is small), so the C× dense buffers stay small
//! in practice. Per-class scatter-back reuses `fm_train`'s
//! `fm_scatter_params` with feature base `c * n`.
//!
//! Determinism/parity caveats are identical to the binary path (atomicAdd →
//! nondeterministic run-to-run; tolerance-based parity on final predictions;
//! compute capability >= 6.0).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::data::CsrView;
use crate::fm::FmGradAccum;
use crate::optimizer::{McState, Optimizer};

/// Compiled once per process into the shared module (`super::gpu`).
///
/// `fm_train_mc_accum_csr`: one block per batch row, one thread per factor
/// (k <= 1024). Pass 1 loops over classes: thread `f` builds the pairwise
/// term of class `c`'s logit, thread 0 adds the linear term and stores the
/// logit in shared `probs[c]`. Thread 0 then replicates the CPU's stable
/// softmax in class order (max, exp, sum), forms each class's weighted
/// gradient `g_c = sw * (p_c - target_c)` with label-smoothed targets
/// (`1 - eps` true class, `off = eps / (C - 1)` otherwise), overwrites
/// `probs[c]` with `g_c`, and accumulates `g_w0[c]` and the linear gradients
/// `gw`. Pass 2 loops over classes again with thread `f` recomputing class
/// `c`'s factor cache in registers (shared memory holds C + k doubles, not
/// C * k) and accumulating the factor gradients into `gv`. Gradient buffers
/// are C-stacked with per-class stride `stride` (= T compact, n dense);
/// `use_slots`/`slots`/`batch_indptr` work exactly like the binary kernel.
pub(super) const KERNEL_SRC: &str = r#"
extern "C" __global__ void fm_train_mc_accum_csr(
    const long long* indptr,
    const long long* indices,
    const double* data,
    const long long* y,
    const double* sw,
    const long long* row_orders,
    const long long batch_start,
    const long long use_slots,
    const long long* slots,
    const long long* batch_indptr,
    const double* w,
    const double* v,
    const double* w0,
    const long long n,
    const long long k,
    const long long n_classes,
    const double eps,
    const double off,
    const long long stride,
    double* g_w0,
    double* gw,
    double* gv)
{
    long long r = row_orders[batch_start + blockIdx.x];
    long long lo = indptr[r];
    long long hi = indptr[r + 1];
    const long long* row_slots = use_slots ? slots + batch_indptr[blockIdx.x] : 0;
    extern __shared__ double shm[];  // probs[n_classes] | pair[k]
    double* probs = shm;
    double* pair = shm + n_classes;
    long long f = threadIdx.x;
    for (long long c = 0; c < n_classes; ++c) {
        const double* vc = v + c * n * k;
        if (f < k) {
            double sum = 0.0;
            double sq = 0.0;
            for (long long p = lo; p < hi; ++p) {
                double vx = vc[indices[p] * k + f] * data[p];
                sum += vx;
                sq += vx * vx;
            }
            pair[f] = sum * sum - sq;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const double* wc = w + c * n;
            double lin = 0.0;
            for (long long p = lo; p < hi; ++p) {
                lin += wc[indices[p]] * data[p];
            }
            double pw = 0.0;
            for (long long ff = 0; ff < k; ++ff) {
                pw += pair[ff];
            }
            probs[c] = w0[c] + lin + 0.5 * pw;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double maxl = probs[0];
        for (long long c = 1; c < n_classes; ++c) {
            if (probs[c] > maxl) {
                maxl = probs[c];
            }
        }
        double sum_ex = 0.0;
        for (long long c = 0; c < n_classes; ++c) {
            probs[c] = exp(probs[c] - maxl);
            sum_ex += probs[c];
        }
        long long yc = y[r];
        double swr = sw[r];
        for (long long c = 0; c < n_classes; ++c) {
            double p = probs[c] / sum_ex;
            double target = (c == yc) ? (1.0 - eps) : off;
            double g = swr * (p - target);
            probs[c] = g;
            atomicAdd(&g_w0[c], g);
            for (long long pnz = lo; pnz < hi; ++pnz) {
                long long slot = use_slots ? row_slots[pnz - lo] : indices[pnz];
                atomicAdd(&gw[c * stride + slot], g * data[pnz]);
            }
        }
    }
    __syncthreads();
    if (f < k) {
        for (long long c = 0; c < n_classes; ++c) {
            const double* vc = v + c * n * k;
            double g = probs[c];
            double sum = 0.0;
            for (long long p = lo; p < hi; ++p) {
                sum += vc[indices[p] * k + f] * data[p];
            }
            for (long long p = lo; p < hi; ++p) {
                long long i = indices[p];
                long long slot = use_slots ? row_slots[p - lo] : i;
                double x = data[p];
                atomicAdd(
                    &gv[(c * stride + slot) * k + f],
                    g * (x * sum - vc[i * k + f] * x * x));
            }
        }
    }
}
"#;

/// Dynamic shared memory holds `n_classes + k` doubles; 48 KB is the
/// no-opt-in per-block limit.
const SHARED_DOUBLES_MAX: usize = 48 * 1024 / std::mem::size_of::<f64>();

/// Train a multiclass (softmax) FM in place with CUDA batch accumulation +
/// the CPU per-class optimizer flush. Argument contract matches
/// `crate::fm::fit_multiclass_csr`. Errors are stringified for the PyO3 layer.
#[allow(clippy::too_many_arguments)]
pub fn fit_multiclass_csr(
    csr: &CsrView,
    y: &[i64],
    sample_weight: &[f64],
    w0: &mut [f64],
    w: &mut [f64],
    v: &mut [f64],
    mut st: McState<'_>,
    n_classes: usize,
    n_features: usize,
    k: usize,
    opt: Optimizer,
    lr: f64,
    l1_linear: f64,
    l2_linear: f64,
    l1_factors: f64,
    l2_factors: f64,
    label_smoothing: f64,
    n_rows: usize,
    batch_size: usize,
    row_orders: &[i64],
) -> Result<(), String> {
    if k == 0 || k > 1024 {
        return Err(format!("CUDA FM training supports 1 <= k <= 1024, got {k}"));
    }
    if n_classes + k > SHARED_DOUBLES_MAX {
        return Err(format!(
            "CUDA multiclass FM training needs n_classes + k <= {SHARED_DOUBLES_MAX} \
             (shared memory), got n_classes={n_classes}, k={k}"
        ));
    }
    fn e<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> String {
        move |err| format!("CUDA {what} failed: {err:?}")
    }
    let n = n_features;
    let c_n = n_classes;
    let off = if c_n > 1 {
        label_smoothing / (c_n as f64 - 1.0)
    } else {
        0.0
    };
    // Pre-pass over the batch schedule (same rule as the binary path; C
    // multiplies both the gradient transfer and the dense buffers, so the
    // compact-vs-dense threshold is unchanged).
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
        .load_function("fm_train_mc_accum_csr")
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
    // Device-resident parameters; w0 is C doubles and re-uploads per batch.
    let mut d_w = stream.clone_htod(&*w).map_err(e("w upload"))?;
    let mut d_v = stream.clone_htod(&*v).map_err(e("V upload"))?;
    let mut d_w0 = stream.clone_htod(&*w0).map_err(e("w0 upload"))?;
    let mut d_gw0 = stream.alloc_zeros::<f64>(c_n).map_err(e("g_w0 alloc"))?;
    let (mut d_gw, mut d_gv) = if any_dense {
        (
            stream.alloc_zeros::<f64>(c_n * n).map_err(e("gw alloc"))?,
            stream.alloc_zeros::<f64>(c_n * n * k).map_err(e("gv alloc"))?,
        )
    } else {
        (
            stream.alloc_zeros::<f64>(1).map_err(e("gw alloc"))?,
            stream.alloc_zeros::<f64>(1).map_err(e("gv alloc"))?,
        )
    };
    // Compact buffers sized to the largest compact batch, C-stacked for the
    // gradients; the parameter-update buffers are per-class (the scatter runs
    // once per class with feature base c * n).
    let cap = max_compact_nnz.max(1);
    let mut d_slots = stream.alloc_zeros::<i64>(cap).map_err(e("slots alloc"))?;
    let mut d_bindptr = stream
        .alloc_zeros::<i64>(max_batch_len + 1)
        .map_err(e("batch_indptr alloc"))?;
    let mut d_touched = stream.alloc_zeros::<i64>(cap).map_err(e("touched alloc"))?;
    let mut d_gwc = stream.alloc_zeros::<f64>(c_n * cap).map_err(e("gw_c alloc"))?;
    let mut d_gvc = stream.alloc_zeros::<f64>(c_n * cap * k).map_err(e("gv_c alloc"))?;
    let mut d_wu = stream.alloc_zeros::<f64>(cap).map_err(e("w_u alloc"))?;
    let mut d_vu = stream.alloc_zeros::<f64>(cap * k).map_err(e("v_u alloc"))?;
    // Host scratch, reused across batches.
    let mut host_gw0 = vec![0.0f64; c_n];
    let (mut host_gw, mut host_gv) = if any_dense {
        (vec![0.0f64; c_n * n], vec![0.0f64; c_n * n * k])
    } else {
        (Vec::new(), Vec::new())
    };
    let mut host_gwc = vec![0.0f64; c_n * cap];
    let mut host_gvc = vec![0.0f64; c_n * cap * k];
    let mut host_wu = vec![0.0f64; cap];
    let mut host_vu = vec![0.0f64; cap * k];
    let mut slots_flat: Vec<i64> = Vec::with_capacity(cap);
    let mut bindptr: Vec<i64> = Vec::with_capacity(max_batch_len + 1);
    let mut touched: Vec<usize> = Vec::with_capacity(cap);
    let mut touched_ids: Vec<i64> = Vec::with_capacity(cap);
    let mut seen: Vec<bool> = vec![false; n];
    let mut slot_idx: Vec<i64> = vec![0; n];
    // One accumulator reused for every class's flush (flush clears exactly
    // the touched entries, so the buffers are clean for the next class).
    let mut accum = FmGradAccum::new(n, k);
    let mut batch_start: usize = 0;
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            // Shared-across-classes slot map (standalone scratch: `accum` is
            // reused per class, so its own seen/touched can't hold it).
            slots_flat.clear();
            bindptr.clear();
            bindptr.push(0);
            touched.clear();
            for &r in batch {
                let (indices, _) = csr.row(r as usize);
                for &i in indices {
                    let i = i as usize;
                    if !seen[i] {
                        seen[i] = true;
                        slot_idx[i] = touched.len() as i64;
                        touched.push(i);
                    }
                    slots_flat.push(slot_idx[i]);
                }
                bindptr.push(slots_flat.len() as i64);
            }
            for &i in &touched {
                seen[i] = false;
            }
            let t_count = touched.len();
            let nnz_b = slots_flat.len();
            let compact = 2 * nnz_b < n;
            stream.memset_zeros(&mut d_gw0).map_err(e("g_w0 zero"))?;
            if compact {
                stream.memcpy_htod(&slots_flat[..], &mut d_slots).map_err(e("slots upload"))?;
                stream.memcpy_htod(&bindptr[..], &mut d_bindptr).map_err(e("batch_indptr upload"))?;
                let mut gwc_view = d_gwc.slice_mut(..c_n * t_count);
                stream.memset_zeros(&mut gwc_view).map_err(e("gw_c zero"))?;
                let mut gvc_view = d_gvc.slice_mut(..c_n * t_count * k);
                stream.memset_zeros(&mut gvc_view).map_err(e("gv_c zero"))?;
            } else {
                stream.memset_zeros(&mut d_gw).map_err(e("gw zero"))?;
                stream.memset_zeros(&mut d_gv).map_err(e("gv zero"))?;
            }
            let cfg = LaunchConfig {
                grid_dim: (batch.len() as u32, 1, 1),
                block_dim: (k as u32, 1, 1),
                shared_mem_bytes: ((c_n + k) * std::mem::size_of::<f64>()) as u32,
            };
            let batch_start_i64 = batch_start as i64;
            let use_slots: i64 = if compact { 1 } else { 0 };
            let n_i64 = n as i64;
            let k_i64 = k as i64;
            let c_i64 = c_n as i64;
            let stride: i64 = if compact { t_count as i64 } else { n as i64 };
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
                    .arg(&d_w0)
                    .arg(&n_i64)
                    .arg(&k_i64)
                    .arg(&c_i64)
                    .arg(&label_smoothing)
                    .arg(&off)
                    .arg(&stride)
                    .arg(&mut d_gw0)
                    .arg(gw_buf)
                    .arg(gv_buf);
                // Safety: the kernel reads/writes exactly the buffers bound
                // above; CSR structure, y range, row_orders range and
                // w0/w/V shapes are validated by the PyO3 layer, and slot ids
                // are < t_count by construction.
                unsafe { launch.launch(cfg) }.map_err(e("kernel launch"))?;
            }
            stream.memcpy_dtoh(&d_gw0, &mut host_gw0).map_err(e("g_w0 download"))?;
            if compact {
                stream
                    .memcpy_dtoh(&d_gwc.slice(..c_n * t_count), &mut host_gwc[..c_n * t_count])
                    .map_err(e("gw_c download"))?;
                stream
                    .memcpy_dtoh(
                        &d_gvc.slice(..c_n * t_count * k),
                        &mut host_gvc[..c_n * t_count * k],
                    )
                    .map_err(e("gv_c download"))?;
                if t_count > 0 {
                    touched_ids.clear();
                    touched_ids.extend(touched.iter().map(|&i| i as i64));
                    stream
                        .memcpy_htod(&touched_ids[..], &mut d_touched)
                        .map_err(e("touched upload"))?;
                }
            } else {
                stream.memcpy_dtoh(&d_gw, &mut host_gw).map_err(e("gw download"))?;
                stream.memcpy_dtoh(&d_gv, &mut host_gv).map_err(e("gv download"))?;
            }
            let bsz = batch.len() as f64;
            for c in 0..c_n {
                accum.g_w0 = host_gw0[c];
                accum.touched.clear();
                accum.touched.extend_from_slice(&touched);
                if compact {
                    for (s, &i) in touched.iter().enumerate() {
                        accum.gw[i] = host_gwc[c * t_count + s];
                        accum.gv[i * k..(i + 1) * k].copy_from_slice(
                            &host_gvc[(c * t_count + s) * k..(c * t_count + s + 1) * k],
                        );
                    }
                } else {
                    for &i in &touched {
                        accum.gw[i] = host_gw[c * n + i];
                        accum.gv[i * k..(i + 1) * k]
                            .copy_from_slice(&host_gv[(c * n + i) * k..(c * n + i + 1) * k]);
                    }
                }
                let (wr, vr) = (c * n..(c + 1) * n, c * n * k..(c + 1) * n * k);
                let (acc_w0_c, acc_w_c, acc_v_c, mut adam_c, mut ftrl_c) =
                    st.class_views(c, n, n * k);
                accum.flush(
                    bsz, &mut w0[c], &mut w[wr], &mut v[vr], acc_w0_c, acc_w_c, acc_v_c,
                    &mut adam_c, &mut ftrl_c, k, opt, lr, l1_linear, l2_linear, l1_factors,
                    l2_factors,
                );
                // Per-class touched scatter-back (compact); dense re-uploads
                // once after the class loop.
                if compact && t_count > 0 {
                    for (s, &i) in touched.iter().enumerate() {
                        host_wu[s] = w[c * n + i];
                        host_vu[s * k..(s + 1) * k]
                            .copy_from_slice(&v[(c * n + i) * k..(c * n + i + 1) * k]);
                    }
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
                    let base = (c * n) as i64;
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
                    // Safety: scatter writes w[c*n + touched[t]] and the k
                    // factors of each touched feature of class c; ids are
                    // valid by construction.
                    unsafe { launch.launch(scatter_cfg) }.map_err(e("scatter launch"))?;
                }
            }
            if !compact {
                stream.memcpy_htod(&*w, &mut d_w).map_err(e("w re-upload"))?;
                stream.memcpy_htod(&*v, &mut d_v).map_err(e("V re-upload"))?;
            }
            stream.memcpy_htod(&*w0, &mut d_w0).map_err(e("w0 re-upload"))?;
            batch_start += batch.len();
        }
    }
    Ok(())
}
