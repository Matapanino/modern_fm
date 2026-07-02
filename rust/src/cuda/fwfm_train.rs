//! CUDA FwFM mini-batch gradient accumulation (binary/regression +
//! multiclass) — milestone 6 of docs/gpu_backend_plan.md.
//!
//! Same two-stage contract as the FM/FFM paths: the GPU accumulates each
//! batch's data-gradients from the frozen batch-start parameters, the
//! untouched CPU flush (`crate::fwfm::FwfmGradAccum::flush`) applies the
//! optimizer — including the R group (`GroupStateMut` / `McGroupState`), so
//! SGD/AdaGrad/Adam/FTRL, R regularization, early stopping and `partial_fit`
//! state hand-offs ride through unchanged.
//!
//! FwFM gradients live on features (`gv` is FM-shaped `n * k`), so the
//! compact touched-coordinate machinery of the FM path applies — both pair
//! endpoints are row nonzeros, so the per-nonzero slot map covers every `gv`
//! write — while the kernel walks FFM-style strided pair loops (256-thread
//! blocks, no k limit). `gr` is a dense `n_fields^2` buffer (multiclass:
//! `C * n_fields^2`): tiny, memset per batch and downloaded in full, no
//! gather. `w`/`V` are device-resident with touched-only scatter-back
//! (`fm_scatter_params`, feature base 0 binary / `c * n` multiclass); `R` is
//! tiny and re-uploads in full after each flush.
//!
//! Determinism/parity caveats are identical to the FM/FFM paths (atomicAdd →
//! nondeterministic run-to-run; tolerance-based parity on final predictions;
//! compute capability >= 6.0).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::data::CsrView;
use crate::fwfm::FwfmGradAccum;
use crate::optimizer::{
    AdamStateMut, FtrlStateMut, GroupStateMut, Loss, McGroupState, McState, Optimizer,
};

/// Compiled once per process into the shared module (`super::gpu`).
///
/// `fwfm_train_accum_csr` (binary/regression): one 256-thread block per batch
/// row. Phase 1 scores the row from the frozen parameters exactly like the
/// FwFM prediction kernel (strided linear + R-weighted pair loops, tree
/// reduction); thread 0 forms the weighted loss gradient `g` (0 = logistic,
/// 1 = squared) and accumulates `g_w0`. Phase 2 re-walks the strided loops:
/// `gw[slot] += g * x` per nonzero, and per pair `coef = g * x_a * x_b`,
/// `gv[slot_a] += coef * r_ab * v[j]` (+ symmetric) and
/// `gr[pair] += coef * <v_i, v_j>`. With `use_slots != 0` the `gw`/`gv`
/// buffers are compact: the nonzero at row position `p` writes slot
/// `row_slots[p]` (both pair endpoints are row nonzeros). `gr` is always
/// dense by pair slot.
///
/// `fwfm_train_mc_accum_csr` (multiclass softmax): the FM-multiclass pattern
/// on the binary kernel — pass 1 loops classes storing logits in shared
/// `probs[c]`, thread 0 replicates the CPU's stable softmax in class order
/// with label-smoothed targets and overwrites `probs[c]` with each class's
/// weighted gradient, then per class the phase-2 loops accumulate into
/// C-stacked `gw`/`gv` (per-class stride `stride`) and dense
/// `(C, n_fields^2)` `gr`.
pub(super) const KERNEL_SRC: &str = r#"
extern "C" __global__ void fwfm_train_accum_csr(
    const long long* indptr,
    const long long* indices,
    const double* data,
    const long long* field_ids,
    const double* y,
    const double* sw,
    const long long* row_orders,
    const long long batch_start,
    const long long use_slots,
    const long long* slots,
    const long long* batch_indptr,
    const double* w,
    const double* v,
    const double* r,
    const double w0,
    const long long n_fields,
    const long long k,
    const long long loss,
    double* g_w0,
    double* gw,
    double* gv,
    double* gr)
{
    long long row = row_orders[batch_start + blockIdx.x];
    long long lo = indptr[row];
    long long hi = indptr[row + 1];
    long long z = hi - lo;
    const long long* row_slots = use_slots ? slots + batch_indptr[blockIdx.x] : 0;
    extern __shared__ double partial[];  // blockDim.x doubles
    double acc = 0.0;
    for (long long p = lo + threadIdx.x; p < hi; p += blockDim.x) {
        acc += w[indices[p]] * data[p];
    }
    for (long long a = 0; a + 1 < z; ++a) {
        long long ia = indices[lo + a];
        long long fa = field_ids[ia];
        double xa = data[lo + a];
        for (long long b = a + 1 + threadIdx.x; b < z; b += blockDim.x) {
            long long jb = indices[lo + b];
            long long fb = field_ids[jb];
            const double* va = v + ia * k;
            const double* vb = v + jb * k;
            double dot = 0.0;
            for (long long t = 0; t < k; ++t) {
                dot += va[t] * vb[t];
            }
            long long pa = fa <= fb ? fa : fb;
            long long pb = fa <= fb ? fb : fa;
            acc += r[pa * n_fields + pb] * dot * xa * data[lo + b];
        }
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double sc = w0 + partial[0];
        double g;
        if (loss == 0) {
            double prob;
            if (sc >= 0.0) {
                prob = 1.0 / (1.0 + exp(-sc));
            } else {
                double e = exp(sc);
                prob = e / (1.0 + e);
            }
            g = prob - y[row];
        } else {
            g = sc - y[row];
        }
        g *= sw[row];
        partial[0] = g;
        atomicAdd(g_w0, g);
    }
    __syncthreads();
    double g = partial[0];
    for (long long p = lo + threadIdx.x; p < hi; p += blockDim.x) {
        long long slot = use_slots ? row_slots[p - lo] : indices[p];
        atomicAdd(&gw[slot], g * data[p]);
    }
    for (long long a = 0; a + 1 < z; ++a) {
        long long ia = indices[lo + a];
        long long fa = field_ids[ia];
        double xa = data[lo + a];
        long long sa = use_slots ? row_slots[a] : ia;
        for (long long b = a + 1 + threadIdx.x; b < z; b += blockDim.x) {
            long long jb = indices[lo + b];
            long long fb = field_ids[jb];
            long long sb = use_slots ? row_slots[b] : jb;
            long long pa = fa <= fb ? fa : fb;
            long long pb = fa <= fb ? fb : fa;
            long long pair = pa * n_fields + pb;
            double coef = g * xa * data[lo + b];
            double rw = r[pair];
            const double* va = v + ia * k;
            const double* vb = v + jb * k;
            double dot = 0.0;
            for (long long t = 0; t < k; ++t) {
                dot += va[t] * vb[t];
                atomicAdd(&gv[sa * k + t], coef * rw * vb[t]);
                atomicAdd(&gv[sb * k + t], coef * rw * va[t]);
            }
            atomicAdd(&gr[pair], coef * dot);
        }
    }
}

extern "C" __global__ void fwfm_train_mc_accum_csr(
    const long long* indptr,
    const long long* indices,
    const double* data,
    const long long* field_ids,
    const long long* y,
    const double* sw,
    const long long* row_orders,
    const long long batch_start,
    const long long use_slots,
    const long long* slots,
    const long long* batch_indptr,
    const double* w,
    const double* v,
    const double* r,
    const double* w0,
    const long long n,
    const long long n_fields,
    const long long k,
    const long long n_classes,
    const double eps,
    const double off,
    const long long stride,
    double* g_w0,
    double* gw,
    double* gv,
    double* gr)
{
    long long row = row_orders[batch_start + blockIdx.x];
    long long lo = indptr[row];
    long long hi = indptr[row + 1];
    long long z = hi - lo;
    const long long* row_slots = use_slots ? slots + batch_indptr[blockIdx.x] : 0;
    extern __shared__ double shm[];  // partial[blockDim.x] | probs[n_classes]
    double* partial = shm;
    double* probs = shm + blockDim.x;
    long long rc = n_fields * n_fields;
    for (long long c = 0; c < n_classes; ++c) {
        const double* wc = w + c * n;
        const double* vc = v + c * n * k;
        const double* rcp = r + c * rc;
        double acc = 0.0;
        for (long long p = lo + threadIdx.x; p < hi; p += blockDim.x) {
            acc += wc[indices[p]] * data[p];
        }
        for (long long a = 0; a + 1 < z; ++a) {
            long long ia = indices[lo + a];
            long long fa = field_ids[ia];
            double xa = data[lo + a];
            for (long long b = a + 1 + threadIdx.x; b < z; b += blockDim.x) {
                long long jb = indices[lo + b];
                long long fb = field_ids[jb];
                const double* va = vc + ia * k;
                const double* vb = vc + jb * k;
                double dot = 0.0;
                for (long long t = 0; t < k; ++t) {
                    dot += va[t] * vb[t];
                }
                long long pa = fa <= fb ? fa : fb;
                long long pb = fa <= fb ? fb : fa;
                acc += rcp[pa * n_fields + pb] * dot * xa * data[lo + b];
            }
        }
        partial[threadIdx.x] = acc;
        __syncthreads();
        for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                partial[threadIdx.x] += partial[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            probs[c] = w0[c] + partial[0];
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
        long long yc = y[row];
        double swr = sw[row];
        for (long long c = 0; c < n_classes; ++c) {
            double p = probs[c] / sum_ex;
            double target = (c == yc) ? (1.0 - eps) : off;
            double g = swr * (p - target);
            probs[c] = g;
            atomicAdd(&g_w0[c], g);
        }
    }
    __syncthreads();
    for (long long c = 0; c < n_classes; ++c) {
        const double* vc = v + c * n * k;
        const double* rcp = r + c * rc;
        double g = probs[c];
        for (long long p = lo + threadIdx.x; p < hi; p += blockDim.x) {
            long long slot = use_slots ? row_slots[p - lo] : indices[p];
            atomicAdd(&gw[c * stride + slot], g * data[p]);
        }
        for (long long a = 0; a + 1 < z; ++a) {
            long long ia = indices[lo + a];
            long long fa = field_ids[ia];
            double xa = data[lo + a];
            long long sa = use_slots ? row_slots[a] : ia;
            for (long long b = a + 1 + threadIdx.x; b < z; b += blockDim.x) {
                long long jb = indices[lo + b];
                long long fb = field_ids[jb];
                long long sb = use_slots ? row_slots[b] : jb;
                long long pa = fa <= fb ? fa : fb;
                long long pb = fa <= fb ? fb : fa;
                long long pair = pa * n_fields + pb;
                double coef = g * xa * data[lo + b];
                double rw = rcp[pair];
                const double* va = vc + ia * k;
                const double* vb = vc + jb * k;
                double dot = 0.0;
                for (long long t = 0; t < k; ++t) {
                    dot += va[t] * vb[t];
                    atomicAdd(&gv[(c * stride + sa) * k + t], coef * rw * vb[t]);
                    atomicAdd(&gv[(c * stride + sb) * k + t], coef * rw * va[t]);
                }
                atomicAdd(&gr[c * rc + pair], coef * dot);
            }
        }
    }
}
"#;

/// Threads per block (power of two for the tree reduction).
const BLOCK: u32 = 256;

/// Dynamic shared memory limit (48 KB of doubles) for the multiclass kernel's
/// `partial[BLOCK] | probs[C]` layout.
const SHARED_DOUBLES_MAX: usize = 48 * 1024 / std::mem::size_of::<f64>();

/// Train an FwFM in place with CUDA batch accumulation + the CPU optimizer
/// flush (incl. the R group). Argument contract matches
/// `crate::fwfm::fit_csr`. Errors are stringified for the PyO3 layer.
#[allow(clippy::too_many_arguments)]
pub fn fit_csr(
    csr: &CsrView,
    y: &[f64],
    sample_weight: &[f64],
    field_ids: &[i64],
    w0: &mut f64,
    w: &mut [f64],
    v: &mut [f64],
    r: &mut [f64],
    acc_w0: &mut f64,
    acc_w: &mut [f64],
    acc_v: &mut [f64],
    mut adam: AdamStateMut<'_>,
    mut ftrl: FtrlStateMut<'_>,
    mut rst: GroupStateMut<'_>,
    n_fields: usize,
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
    if k == 0 {
        return Err("CUDA FwFM training requires k >= 1".to_string());
    }
    fn e<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> String {
        move |err| format!("CUDA {what} failed: {err:?}")
    }
    let n = w.len();
    let rc = n_fields * n_fields;
    let loss_code: i64 = match loss {
        Loss::Logistic => 0,
        Loss::Squared => 1,
    };
    // Same compact-vs-dense pre-pass as the FM path (gradients live on
    // features, FM-shaped).
    let batch_nnz = |batch: &[i64]| -> usize {
        batch
            .iter()
            .map(|&row| (csr.indptr[row as usize + 1] - csr.indptr[row as usize]) as usize)
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
        .load_function("fwfm_train_accum_csr")
        .map_err(e("function load"))?;
    let scatter_func = module
        .load_function("fm_scatter_params")
        .map_err(e("scatter function load"))?;
    let stream = ctx.default_stream();
    // Static per-call uploads.
    let d_indptr = stream.clone_htod(csr.indptr).map_err(e("indptr upload"))?;
    let d_indices = stream.clone_htod(csr.indices).map_err(e("indices upload"))?;
    let d_data = stream.clone_htod(csr.data).map_err(e("data upload"))?;
    let d_fields = stream.clone_htod(field_ids).map_err(e("field_ids upload"))?;
    let d_y = stream.clone_htod(y).map_err(e("y upload"))?;
    let d_sw = stream.clone_htod(sample_weight).map_err(e("sample_weight upload"))?;
    let d_ro = stream.clone_htod(row_orders).map_err(e("row_orders upload"))?;
    // w/V device-resident (touched scatter-back); R is tiny and re-uploads in
    // full after every flush.
    let mut d_w = stream.clone_htod(&*w).map_err(e("w upload"))?;
    let mut d_v = stream.clone_htod(&*v).map_err(e("V upload"))?;
    let mut d_r = stream.clone_htod(&*r).map_err(e("R upload"))?;
    let mut d_gw0 = stream.alloc_zeros::<f64>(1).map_err(e("g_w0 alloc"))?;
    let mut d_gr = stream.alloc_zeros::<f64>(rc).map_err(e("gr alloc"))?;
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
    let mut host_gr = vec![0.0f64; rc];
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
    let mut accum = FwfmGradAccum::new(n, n_fields, k);
    let mut batch_start: usize = 0;
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            // Touched features + per-nonzero slot map (the flush needs the
            // touched set anyway), plus the touched pair slots for the R
            // flush — the pair loop without the k-dot, like the FFM host.
            slots_flat.clear();
            bindptr.clear();
            bindptr.push(0);
            for &row in batch {
                let (indices, _) = csr.row(row as usize);
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
                for a in 0..indices.len() {
                    let fa = field_ids[indices[a] as usize] as usize;
                    for &jb in &indices[a + 1..] {
                        let fb = field_ids[jb as usize] as usize;
                        let (pa, pb) = if fa <= fb { (fa, fb) } else { (fb, fa) };
                        let pair = pa * n_fields + pb;
                        if !accum.seen_pair[pair] {
                            accum.seen_pair[pair] = true;
                            accum.touched_pair.push(pair);
                        }
                    }
                }
            }
            let t_count = accum.touched.len();
            let nnz_b = slots_flat.len();
            let compact = 2 * nnz_b < n;
            stream.memset_zeros(&mut d_gw0).map_err(e("g_w0 zero"))?;
            stream.memset_zeros(&mut d_gr).map_err(e("gr zero"))?;
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
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * std::mem::size_of::<f64>() as u32,
            };
            let batch_start_i64 = batch_start as i64;
            let use_slots: i64 = if compact { 1 } else { 0 };
            let n_fields_i64 = n_fields as i64;
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
                    .arg(&d_fields)
                    .arg(&d_y)
                    .arg(&d_sw)
                    .arg(&d_ro)
                    .arg(&batch_start_i64)
                    .arg(&use_slots)
                    .arg(&d_slots)
                    .arg(&d_bindptr)
                    .arg(&d_w)
                    .arg(&d_v)
                    .arg(&d_r)
                    .arg(&w0_val)
                    .arg(&n_fields_i64)
                    .arg(&k_i64)
                    .arg(&loss_code)
                    .arg(&mut d_gw0)
                    .arg(gw_buf)
                    .arg(gv_buf)
                    .arg(&mut d_gr);
                // Safety: the kernel reads/writes exactly the buffers bound
                // above; CSR structure, field_ids range, row_orders range and
                // w/V/R shapes are validated by the PyO3 layer, and slot ids
                // are < t_count by construction.
                unsafe { launch.launch(cfg) }.map_err(e("kernel launch"))?;
            }
            stream.memcpy_dtoh(&d_gw0, &mut host_gw0).map_err(e("g_w0 download"))?;
            stream.memcpy_dtoh(&d_gr, &mut host_gr).map_err(e("gr download"))?;
            accum.g_w0 = host_gw0[0];
            accum.gr.copy_from_slice(&host_gr);
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
                batch.len() as f64, w0, w, v, r, acc_w0, acc_w, acc_v, &mut adam, &mut ftrl,
                &mut rst, k, opt, lr, l1_linear, l2_linear, l1_factors, l2_factors,
            );
            // Re-sync device parameters: touched w/V scatter (compact) or full
            // re-upload (dense); R in full (n_fields^2 doubles).
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
                // each touched feature; ids are valid by construction.
                unsafe { launch.launch(scatter_cfg) }.map_err(e("scatter launch"))?;
            } else if !compact {
                stream.memcpy_htod(&*w, &mut d_w).map_err(e("w re-upload"))?;
                stream.memcpy_htod(&*v, &mut d_v).map_err(e("V re-upload"))?;
            }
            stream.memcpy_htod(&*r, &mut d_r).map_err(e("R re-upload"))?;
            batch_start += batch.len();
        }
    }
    Ok(())
}

/// Train a multiclass (softmax) FwFM in place with CUDA batch accumulation +
/// the CPU per-class optimizer flush (incl. the per-class R group). Argument
/// contract matches `crate::fwfm::fit_multiclass_csr`. Errors are stringified
/// for the PyO3 layer.
#[allow(clippy::too_many_arguments)]
pub fn fit_multiclass_csr(
    csr: &CsrView,
    y: &[i64],
    sample_weight: &[f64],
    field_ids: &[i64],
    w0: &mut [f64],
    w: &mut [f64],
    v: &mut [f64],
    r: &mut [f64],
    mut st: McState<'_>,
    mut rst: McGroupState<'_>,
    n_classes: usize,
    n_features: usize,
    n_fields: usize,
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
    if k == 0 {
        return Err("CUDA FwFM training requires k >= 1".to_string());
    }
    let shared_max = SHARED_DOUBLES_MAX - BLOCK as usize;
    if n_classes > shared_max {
        return Err(format!(
            "CUDA multiclass FwFM training needs n_classes <= {shared_max} (shared memory), \
             got {n_classes}"
        ));
    }
    fn e<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> String {
        move |err| format!("CUDA {what} failed: {err:?}")
    }
    let n = n_features;
    let c_n = n_classes;
    let rc = n_fields * n_fields;
    let off = if c_n > 1 {
        label_smoothing / (c_n as f64 - 1.0)
    } else {
        0.0
    };
    let batch_nnz = |batch: &[i64]| -> usize {
        batch
            .iter()
            .map(|&row| (csr.indptr[row as usize + 1] - csr.indptr[row as usize]) as usize)
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
        .load_function("fwfm_train_mc_accum_csr")
        .map_err(e("function load"))?;
    let scatter_func = module
        .load_function("fm_scatter_params")
        .map_err(e("scatter function load"))?;
    let stream = ctx.default_stream();
    let d_indptr = stream.clone_htod(csr.indptr).map_err(e("indptr upload"))?;
    let d_indices = stream.clone_htod(csr.indices).map_err(e("indices upload"))?;
    let d_data = stream.clone_htod(csr.data).map_err(e("data upload"))?;
    let d_fields = stream.clone_htod(field_ids).map_err(e("field_ids upload"))?;
    let d_y = stream.clone_htod(y).map_err(e("y upload"))?;
    let d_sw = stream.clone_htod(sample_weight).map_err(e("sample_weight upload"))?;
    let d_ro = stream.clone_htod(row_orders).map_err(e("row_orders upload"))?;
    let mut d_w = stream.clone_htod(&*w).map_err(e("w upload"))?;
    let mut d_v = stream.clone_htod(&*v).map_err(e("V upload"))?;
    let mut d_r = stream.clone_htod(&*r).map_err(e("R upload"))?;
    let mut d_w0 = stream.clone_htod(&*w0).map_err(e("w0 upload"))?;
    let mut d_gw0 = stream.alloc_zeros::<f64>(c_n).map_err(e("g_w0 alloc"))?;
    let mut d_gr = stream.alloc_zeros::<f64>(c_n * rc).map_err(e("gr alloc"))?;
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
    let mut host_gw0 = vec![0.0f64; c_n];
    let mut host_gr = vec![0.0f64; c_n * rc];
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
    let mut touched_pair: Vec<usize> = Vec::new();
    let mut seen_pair: Vec<bool> = vec![false; rc];
    // One accumulator reused for every class's flush.
    let mut accum = FwfmGradAccum::new(n, n_fields, k);
    let mut batch_start: usize = 0;
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            // Shared-across-classes touched features, slot map and touched
            // pairs (standalone scratch: `accum` is reused per class).
            slots_flat.clear();
            bindptr.clear();
            bindptr.push(0);
            touched.clear();
            touched_pair.clear();
            for &row in batch {
                let (indices, _) = csr.row(row as usize);
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
                for a in 0..indices.len() {
                    let fa = field_ids[indices[a] as usize] as usize;
                    for &jb in &indices[a + 1..] {
                        let fb = field_ids[jb as usize] as usize;
                        let (pa, pb) = if fa <= fb { (fa, fb) } else { (fb, fa) };
                        let pair = pa * n_fields + pb;
                        if !seen_pair[pair] {
                            seen_pair[pair] = true;
                            touched_pair.push(pair);
                        }
                    }
                }
            }
            for &i in &touched {
                seen[i] = false;
            }
            for &p in &touched_pair {
                seen_pair[p] = false;
            }
            let t_count = touched.len();
            let nnz_b = slots_flat.len();
            let compact = 2 * nnz_b < n;
            stream.memset_zeros(&mut d_gw0).map_err(e("g_w0 zero"))?;
            stream.memset_zeros(&mut d_gr).map_err(e("gr zero"))?;
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
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: ((BLOCK as usize + c_n) * std::mem::size_of::<f64>()) as u32,
            };
            let batch_start_i64 = batch_start as i64;
            let use_slots: i64 = if compact { 1 } else { 0 };
            let n_i64 = n as i64;
            let n_fields_i64 = n_fields as i64;
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
                    .arg(&d_fields)
                    .arg(&d_y)
                    .arg(&d_sw)
                    .arg(&d_ro)
                    .arg(&batch_start_i64)
                    .arg(&use_slots)
                    .arg(&d_slots)
                    .arg(&d_bindptr)
                    .arg(&d_w)
                    .arg(&d_v)
                    .arg(&d_r)
                    .arg(&d_w0)
                    .arg(&n_i64)
                    .arg(&n_fields_i64)
                    .arg(&k_i64)
                    .arg(&c_i64)
                    .arg(&label_smoothing)
                    .arg(&off)
                    .arg(&stride)
                    .arg(&mut d_gw0)
                    .arg(gw_buf)
                    .arg(gv_buf)
                    .arg(&mut d_gr);
                // Safety: the kernel reads/writes exactly the buffers bound
                // above; CSR structure, field_ids range, y range, row_orders
                // range and w0/w/V/R shapes are validated by the PyO3 layer,
                // and slot ids are < t_count by construction.
                unsafe { launch.launch(cfg) }.map_err(e("kernel launch"))?;
            }
            stream.memcpy_dtoh(&d_gw0, &mut host_gw0).map_err(e("g_w0 download"))?;
            stream.memcpy_dtoh(&d_gr, &mut host_gr).map_err(e("gr download"))?;
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
                accum.gr.copy_from_slice(&host_gr[c * rc..(c + 1) * rc]);
                accum.touched.clear();
                accum.touched.extend_from_slice(&touched);
                accum.touched_pair.clear();
                accum.touched_pair.extend_from_slice(&touched_pair);
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
                let (wr, vr, rr) = (
                    c * n..(c + 1) * n,
                    c * n * k..(c + 1) * n * k,
                    c * rc..(c + 1) * rc,
                );
                let (acc_w0_c, acc_w_c, acc_v_c, mut adam_c, mut ftrl_c) =
                    st.class_views(c, n, n * k);
                let mut rst_c = rst.class_views(c, rc);
                accum.flush(
                    bsz, &mut w0[c], &mut w[wr], &mut v[vr], &mut r[rr], acc_w0_c, acc_w_c,
                    acc_v_c, &mut adam_c, &mut ftrl_c, &mut rst_c, k, opt, lr, l1_linear,
                    l2_linear, l1_factors, l2_factors,
                );
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
                    // factors of each touched feature of class c.
                    unsafe { launch.launch(scatter_cfg) }.map_err(e("scatter launch"))?;
                }
            }
            if !compact {
                stream.memcpy_htod(&*w, &mut d_w).map_err(e("w re-upload"))?;
                stream.memcpy_htod(&*v, &mut d_v).map_err(e("V re-upload"))?;
            }
            stream.memcpy_htod(&*r, &mut d_r).map_err(e("R re-upload"))?;
            stream.memcpy_htod(&*w0, &mut d_w0).map_err(e("w0 re-upload"))?;
            batch_start += batch.len();
        }
    }
    Ok(())
}
