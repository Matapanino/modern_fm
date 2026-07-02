//! CUDA FFM mini-batch gradient accumulation — milestone 4 of
//! docs/gpu_backend_plan.md (binary logistic + squared loss).
//!
//! Same two-stage contract as the FM path (`cuda::fm_train`): the GPU
//! accumulates each batch's data-gradients from the frozen batch-start
//! parameters, the untouched CPU flush (`crate::ffm::FfmGradAccum::flush`)
//! applies the optimizer, so SGD/AdaGrad/Adam/FTRL, early stopping and
//! `partial_fit`/`warm_start` state hand-offs ride through unchanged.
//!
//! FFM's factor gradients live on (feature, field) slots — a dense buffer the
//! size of `V` itself — so the transfer plan differs from FM:
//!
//! - `V` uploads once and stays device-resident; `gv` accumulates into a
//!   dense device buffer by absolute slot (no per-pair slot-indirection
//!   upload, which would be O(z²) per row).
//! - The host enumerates each batch's touched (feature, field) slots (the
//!   flush needs that list anyway; it is the pair loop *without* the k-dot,
//!   so ~k× cheaper than the CPU accumulation it replaces) and only those
//!   slots move: a gather kernel packs `gv[slot]` into a compact buffer —
//!   zeroing the dense entries as it reads, so the buffer stays clean
//!   without full memsets — and after the flush a scatter kernel writes the
//!   updated `V[slot]` values back.
//! - `w` (n doubles) is small: dense gradient download + full re-upload.
//!
//! Determinism/parity caveats are identical to the FM path (atomicAdd →
//! nondeterministic run-to-run; tolerance-based parity on final predictions;
//! compute capability >= 6.0).

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::data::CsrView;
use crate::ffm::FfmGradAccum;
use crate::optimizer::{AdamStateMut, FtrlStateMut, Loss, Optimizer};

/// Compiled once per process into the shared module (`super::gpu`).
///
/// `ffm_train_accum_csr`: one 256-thread block per batch row. Phase 1 scores
/// the row from the frozen parameters exactly like the FFM prediction kernel
/// (strided linear + pair loops, shared-memory tree reduction); thread 0
/// turns the score into the weighted loss gradient `g` (0 = logistic,
/// 1 = squared) and accumulates `g_w0`. Phase 2 re-walks the strided nonzero
/// and pair loops adding `g * x` into `gw` and the symmetric pair gradients
/// `g * x_a * x_b * v[other slot]` into the dense `gv` by absolute slot.
///
/// `ffm_gather_slots`: packs `gv[slots[s] * k + t]` into `out[s * k + t]`,
/// zeroing each dense entry as it is read (keeps `gv` all-zero between
/// batches without a full memset of a V-sized buffer).
///
/// `ffm_scatter_slots`: writes the flushed `V` values of the touched slots
/// back into the device-resident `v`.
pub(super) const KERNEL_SRC: &str = r#"
extern "C" __global__ void ffm_train_accum_csr(
    const long long* indptr,
    const long long* indices,
    const double* data,
    const long long* field_ids,
    const double* y,
    const double* sw,
    const long long* row_orders,
    const long long batch_start,
    const double* w,
    const double* v,
    const double w0,
    const long long n_fields,
    const long long k,
    const long long loss,
    double* g_w0,
    double* gw,
    double* gv)
{
    long long r = row_orders[batch_start + blockIdx.x];
    long long lo = indptr[r];
    long long hi = indptr[r + 1];
    long long z = hi - lo;
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
            const double* va = v + (ia * n_fields + field_ids[jb]) * k;
            const double* vb = v + (jb * n_fields + fa) * k;
            double dot = 0.0;
            for (long long t = 0; t < k; ++t) {
                dot += va[t] * vb[t];
            }
            acc += dot * xa * data[lo + b];
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
            g = prob - y[r];
        } else {
            g = sc - y[r];
        }
        g *= sw[r];
        partial[0] = g;
        atomicAdd(g_w0, g);
    }
    __syncthreads();
    double g = partial[0];
    for (long long p = lo + threadIdx.x; p < hi; p += blockDim.x) {
        atomicAdd(&gw[indices[p]], g * data[p]);
    }
    for (long long a = 0; a + 1 < z; ++a) {
        long long ia = indices[lo + a];
        long long fa = field_ids[ia];
        double xa = data[lo + a];
        for (long long b = a + 1 + threadIdx.x; b < z; b += blockDim.x) {
            long long jb = indices[lo + b];
            long long slot_a = ia * n_fields + field_ids[jb];
            long long slot_b = jb * n_fields + fa;
            double coef = g * xa * data[lo + b];
            for (long long t = 0; t < k; ++t) {
                atomicAdd(&gv[slot_a * k + t], coef * v[slot_b * k + t]);
                atomicAdd(&gv[slot_b * k + t], coef * v[slot_a * k + t]);
            }
        }
    }
}

extern "C" __global__ void ffm_gather_slots(
    const long long* slots,
    const long long n_slots,
    const long long k,
    double* gv,
    double* out)
{
    long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots * k) {
        return;
    }
    long long idx = slots[s / k] * k + (s % k);
    out[s] = gv[idx];
    gv[idx] = 0.0;
}

extern "C" __global__ void ffm_scatter_slots(
    const long long* slots,
    const double* values,
    const long long n_slots,
    const long long k,
    double* v)
{
    long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots * k) {
        return;
    }
    v[slots[s / k] * k + (s % k)] = values[s];
}
"#;

/// Threads per block for the accumulation kernel; a power of two for the
/// tree reduction (matches the FFM prediction kernel).
const BLOCK: u32 = 256;

/// Grow-on-demand device buffer paired with a host scratch of the same length.
struct SlotBuffers {
    d_slots: CudaSlice<i64>,
    d_vals: CudaSlice<f64>,
    host_vals: Vec<f64>,
    cap: usize,
}

/// Train an FFM in place with CUDA batch accumulation + the CPU optimizer
/// flush. Argument contract matches `crate::ffm::fit_csr` (no `n_jobs`).
/// Errors are stringified for the PyO3 layer.
#[allow(clippy::too_many_arguments)]
pub fn fit_csr(
    csr: &CsrView,
    y: &[f64],
    sample_weight: &[f64],
    field_ids: &[i64],
    w0: &mut f64,
    w: &mut [f64],
    v: &mut [f64],
    acc_w0: &mut f64,
    acc_w: &mut [f64],
    acc_v: &mut [f64],
    mut adam: AdamStateMut<'_>,
    mut ftrl: FtrlStateMut<'_>,
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
        return Err("CUDA FFM training requires k >= 1".to_string());
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
    let accum_func = module
        .load_function("ffm_train_accum_csr")
        .map_err(e("function load"))?;
    let gather_func = module
        .load_function("ffm_gather_slots")
        .map_err(e("gather function load"))?;
    let scatter_func = module
        .load_function("ffm_scatter_slots")
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
    // V is device-resident (touched slots scatter back after each flush);
    // w is small and re-uploads in full.
    let mut d_w = stream.clone_htod(&*w).map_err(e("w upload"))?;
    let mut d_v = stream.clone_htod(&*v).map_err(e("V upload"))?;
    let mut d_gw0 = stream.alloc_zeros::<f64>(1).map_err(e("g_w0 alloc"))?;
    let mut d_gw = stream.alloc_zeros::<f64>(n).map_err(e("gw alloc"))?;
    // Dense slot-gradient buffer, zeroed once; the gather kernel re-zeroes
    // exactly the entries it reads, so it stays clean between batches.
    let mut d_gv = stream.alloc_zeros::<f64>(n * n_fields * k).map_err(e("gv alloc"))?;
    let mut host_gw0 = vec![0.0f64; 1];
    let mut host_gw = vec![0.0f64; n];
    let mut slot_buffers: Option<SlotBuffers> = None;
    let mut slot_list: Vec<i64> = Vec::new();
    let mut touched_slot_scratch: Vec<usize> = Vec::new();
    let mut accum = FfmGradAccum::new(n, n_fields, k);
    let mut batch_start: usize = 0;
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            // Host-side touched enumeration: the pair loop without the k-dot
            // (the flush needs these lists regardless of backend).
            for &r in batch {
                let (indices, _) = csr.row(r as usize);
                for &i in indices {
                    let i = i as usize;
                    if !accum.seen_feat[i] {
                        accum.seen_feat[i] = true;
                        accum.touched_feat.push(i);
                    }
                }
                for a in 0..indices.len() {
                    let ia = indices[a] as usize;
                    let fa = field_ids[ia] as usize;
                    for &jb in &indices[a + 1..] {
                        let jb = jb as usize;
                        let slot_a = ia * n_fields + field_ids[jb] as usize;
                        let slot_b = jb * n_fields + fa;
                        if !accum.seen_slot[slot_a] {
                            accum.seen_slot[slot_a] = true;
                            accum.touched_slot.push(slot_a);
                        }
                        if !accum.seen_slot[slot_b] {
                            accum.seen_slot[slot_b] = true;
                            accum.touched_slot.push(slot_b);
                        }
                    }
                }
            }
            let s_count = accum.touched_slot.len();
            // Grow-on-demand compact buffers (slot ids + k values per slot).
            if s_count > 0 {
                let need_grow = match &slot_buffers {
                    Some(b) => b.cap < s_count,
                    None => true,
                };
                if need_grow {
                    let cap = s_count.next_power_of_two();
                    slot_buffers = Some(SlotBuffers {
                        d_slots: stream.alloc_zeros::<i64>(cap).map_err(e("slots alloc"))?,
                        d_vals: stream.alloc_zeros::<f64>(cap * k).map_err(e("slot vals alloc"))?,
                        host_vals: vec![0.0f64; cap * k],
                        cap,
                    });
                }
            }
            stream.memset_zeros(&mut d_gw0).map_err(e("g_w0 zero"))?;
            stream.memset_zeros(&mut d_gw).map_err(e("gw zero"))?;
            let cfg = LaunchConfig {
                grid_dim: (batch.len() as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * std::mem::size_of::<f64>() as u32,
            };
            let batch_start_i64 = batch_start as i64;
            let n_fields_i64 = n_fields as i64;
            let k_i64 = k as i64;
            let w0_val = *w0;
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
                .arg(&d_w)
                .arg(&d_v)
                .arg(&w0_val)
                .arg(&n_fields_i64)
                .arg(&k_i64)
                .arg(&loss_code)
                .arg(&mut d_gw0)
                .arg(&mut d_gw)
                .arg(&mut d_gv);
            // Safety: the kernel reads/writes exactly the buffers bound above;
            // CSR structure, field_ids range, row_orders range and w/V shapes
            // are validated by the PyO3 layer.
            unsafe { launch.launch(cfg) }.map_err(e("kernel launch"))?;
            stream.memcpy_dtoh(&d_gw0, &mut host_gw0).map_err(e("g_w0 download"))?;
            stream.memcpy_dtoh(&d_gw, &mut host_gw).map_err(e("gw download"))?;
            accum.g_w0 = host_gw0[0];
            for &i in &accum.touched_feat {
                accum.gw[i] = host_gw[i];
            }
            if s_count > 0 {
                let bufs = slot_buffers.as_mut().expect("allocated above");
                slot_list.clear();
                slot_list.extend(accum.touched_slot.iter().map(|&s| s as i64));
                stream
                    .memcpy_htod(&slot_list[..], &mut bufs.d_slots)
                    .map_err(e("slots upload"))?;
                let total = (s_count * k) as u32;
                let gather_cfg = LaunchConfig {
                    grid_dim: (total.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_i64 = s_count as i64;
                let mut launch = stream.launch_builder(&gather_func);
                launch
                    .arg(&bufs.d_slots)
                    .arg(&s_i64)
                    .arg(&k_i64)
                    .arg(&mut d_gv)
                    .arg(&mut bufs.d_vals);
                // Safety: gather reads/zeroes gv at the uploaded slot ids and
                // writes s_count * k values into d_vals (cap >= s_count).
                unsafe { launch.launch(gather_cfg) }.map_err(e("gather launch"))?;
                stream
                    .memcpy_dtoh(&bufs.d_vals.slice(..s_count * k), &mut bufs.host_vals[..s_count * k])
                    .map_err(e("slot grads download"))?;
                for (s, &slot) in accum.touched_slot.iter().enumerate() {
                    accum.gv[slot * k..(slot + 1) * k]
                        .copy_from_slice(&bufs.host_vals[s * k..(s + 1) * k]);
                }
            }
            // The flush clears the touched lists; keep the slots for scatter.
            touched_slot_scratch.clear();
            touched_slot_scratch.extend_from_slice(&accum.touched_slot);
            accum.flush(
                batch.len() as f64, w0, w, v, acc_w0, acc_w, acc_v, &mut adam, &mut ftrl, k, opt,
                lr, l1_linear, l2_linear, l1_factors, l2_factors,
            );
            // Re-sync device parameters: full w re-upload (small), touched
            // slots of V scattered back (d_slots still holds this batch's
            // slot ids).
            stream.memcpy_htod(&*w, &mut d_w).map_err(e("w re-upload"))?;
            if s_count > 0 {
                let bufs = slot_buffers.as_mut().expect("allocated above");
                for (s, &slot) in touched_slot_scratch.iter().enumerate() {
                    bufs.host_vals[s * k..(s + 1) * k]
                        .copy_from_slice(&v[slot * k..(slot + 1) * k]);
                }
                stream
                    .memcpy_htod(&bufs.host_vals[..s_count * k], &mut bufs.d_vals)
                    .map_err(e("slot vals upload"))?;
                let total = (s_count * k) as u32;
                let scatter_cfg = LaunchConfig {
                    grid_dim: (total.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_i64 = s_count as i64;
                let mut launch = stream.launch_builder(&scatter_func);
                launch
                    .arg(&bufs.d_slots)
                    .arg(&bufs.d_vals)
                    .arg(&s_i64)
                    .arg(&k_i64)
                    .arg(&mut d_v);
                // Safety: scatter writes the k factors of each touched slot;
                // slot ids are valid (feature * n_fields + field) by
                // construction.
                unsafe { launch.launch(scatter_cfg) }.map_err(e("scatter launch"))?;
            }
            batch_start += batch.len();
        }
    }
    Ok(())
}
