//! CUDA FFM multiclass (softmax) mini-batch gradient accumulation — milestone
//! 5 of docs/gpu_backend_plan.md.
//!
//! Same two-stage contract as the binary path (`cuda::ffm_train`): the GPU
//! accumulates each batch's data-gradients from the frozen batch-start
//! parameters, the untouched CPU flush (`crate::ffm::FfmGradAccum::flush`
//! over `McState::class_views`) applies the optimizer per class.
//!
//! FFM's dense slot-gradient buffer is the size of one class's `V`
//! (n * n_fields * k); C-stacking it would double the already C× device
//! footprint of the resident parameters. Instead the accumulation splits into
//! two kernels so ONE class-sized `gv` buffer serves every class:
//!
//! - `ffm_mc_score_csr` scores all C logits per row from the frozen
//!   parameters, does the stable softmax (thread 0, CPU class order), stores
//!   each row's per-class weighted gradients `g` in a global
//!   `(batch_len, C)` buffer, and accumulates `g_w0` (C,) and the dense
//!   linear gradients `gw` (C, n) — both small.
//! - `ffm_mc_pair_accum_csr` launches once per class: it re-walks the pair
//!   loop reading that class's frozen `V` block and row gradient, adding the
//!   symmetric pair gradients into the shared class-local `gv`. The gather
//!   kernel (`ffm_gather_slots`, slot base 0) then packs the touched slots
//!   and zeroes them as it reads, leaving `gv` clean for the next class.
//!
//! Interleaving class c's flush before class c+1's pair kernel is safe: the
//! per-class parameter blocks are disjoint and every logit/gradient was
//! computed by the score kernel from the frozen batch-start parameters.
//! Touched features and slots are enumerated once per batch on the host and
//! shared across classes; scatter-back reuses `ffm_scatter_slots` with slot
//! base `c * n * n_fields` into the device-resident (C, n, n_fields, k) `V`.
//!
//! Determinism/parity caveats are identical to the binary path (atomicAdd →
//! nondeterministic run-to-run; tolerance-based parity on final predictions;
//! compute capability >= 6.0).

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::data::CsrView;
use crate::ffm::FfmGradAccum;
use crate::optimizer::{McState, Optimizer};

/// Compiled once per process into the shared module (`super::gpu`).
pub(super) const KERNEL_SRC: &str = r#"
extern "C" __global__ void ffm_mc_score_csr(
    const long long* indptr,
    const long long* indices,
    const double* data,
    const long long* field_ids,
    const long long* y,
    const double* sw,
    const long long* row_orders,
    const long long batch_start,
    const double* w,
    const double* v,
    const double* w0,
    const long long n,
    const long long n_fields,
    const long long k,
    const long long n_classes,
    const double eps,
    const double off,
    double* g_out,
    double* g_w0,
    double* gw)
{
    long long r = row_orders[batch_start + blockIdx.x];
    long long lo = indptr[r];
    long long hi = indptr[r + 1];
    long long z = hi - lo;
    extern __shared__ double shm[];  // partial[blockDim.x] | probs[n_classes]
    double* partial = shm;
    double* probs = shm + blockDim.x;
    long long vc_len = n * n_fields * k;
    for (long long c = 0; c < n_classes; ++c) {
        const double* wc = w + c * n;
        const double* vc = v + c * vc_len;
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
                const double* va = vc + (ia * n_fields + field_ids[jb]) * k;
                const double* vb = vc + (jb * n_fields + fa) * k;
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
        long long yc = y[r];
        double swr = sw[r];
        for (long long c = 0; c < n_classes; ++c) {
            double p = probs[c] / sum_ex;
            double target = (c == yc) ? (1.0 - eps) : off;
            double g = swr * (p - target);
            probs[c] = g;
            g_out[blockIdx.x * n_classes + c] = g;
            atomicAdd(&g_w0[c], g);
        }
    }
    __syncthreads();
    for (long long c = 0; c < n_classes; ++c) {
        double g = probs[c];
        for (long long p = lo + threadIdx.x; p < hi; p += blockDim.x) {
            atomicAdd(&gw[c * n + indices[p]], g * data[p]);
        }
    }
}

extern "C" __global__ void ffm_mc_pair_accum_csr(
    const long long* indptr,
    const long long* indices,
    const double* data,
    const long long* field_ids,
    const long long* row_orders,
    const long long batch_start,
    const double* v,
    const long long n,
    const long long n_fields,
    const long long k,
    const long long n_classes,
    const long long c,
    const double* g_in,
    double* gv)
{
    long long r = row_orders[batch_start + blockIdx.x];
    long long lo = indptr[r];
    long long hi = indptr[r + 1];
    long long z = hi - lo;
    double g = g_in[blockIdx.x * n_classes + c];
    const double* vc = v + c * n * n_fields * k;
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
                atomicAdd(&gv[slot_a * k + t], coef * vc[slot_b * k + t]);
                atomicAdd(&gv[slot_b * k + t], coef * vc[slot_a * k + t]);
            }
        }
    }
}
"#;

/// Threads per block (power of two for the tree reduction, like the binary
/// FFM kernels).
const BLOCK: u32 = 256;

/// Grow-on-demand device buffer paired with a host scratch of the same length.
struct SlotBuffers {
    d_slots: CudaSlice<i64>,
    d_vals: CudaSlice<f64>,
    host_vals: Vec<f64>,
    cap: usize,
}

/// Train a multiclass (softmax) FFM in place with CUDA batch accumulation +
/// the CPU per-class optimizer flush. Argument contract matches
/// `crate::ffm::fit_multiclass_csr`. Errors are stringified for the PyO3
/// layer.
#[allow(clippy::too_many_arguments)]
pub fn fit_multiclass_csr(
    csr: &CsrView,
    y: &[i64],
    sample_weight: &[f64],
    field_ids: &[i64],
    w0: &mut [f64],
    w: &mut [f64],
    v: &mut [f64],
    mut st: McState<'_>,
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
        return Err("CUDA FFM training requires k >= 1".to_string());
    }
    let shared_max = 48 * 1024 / std::mem::size_of::<f64>() - BLOCK as usize;
    if n_classes > shared_max {
        return Err(format!(
            "CUDA multiclass FFM training needs n_classes <= {shared_max} (shared memory), \
             got {n_classes}"
        ));
    }
    fn e<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> String {
        move |err| format!("CUDA {what} failed: {err:?}")
    }
    let n = n_features;
    let c_n = n_classes;
    let vc_len = n * n_fields * k; // V entries per class
    let off = if c_n > 1 {
        label_smoothing / (c_n as f64 - 1.0)
    } else {
        0.0
    };
    let mut max_batch_len = 0usize;
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            max_batch_len = max_batch_len.max(batch.len());
        }
    }
    let (ctx, module) = super::gpu()?;
    let score_func = module
        .load_function("ffm_mc_score_csr")
        .map_err(e("score function load"))?;
    let pair_func = module
        .load_function("ffm_mc_pair_accum_csr")
        .map_err(e("pair function load"))?;
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
    // Device-resident parameters (V is (C, n, n_fields, k); touched slots of
    // each class scatter back after its flush); w/w0 are small and re-upload
    // in full after the class loop.
    let mut d_w = stream.clone_htod(&*w).map_err(e("w upload"))?;
    let mut d_v = stream.clone_htod(&*v).map_err(e("V upload"))?;
    let mut d_w0 = stream.clone_htod(&*w0).map_err(e("w0 upload"))?;
    let mut d_gw0 = stream.alloc_zeros::<f64>(c_n).map_err(e("g_w0 alloc"))?;
    let mut d_gw = stream.alloc_zeros::<f64>(c_n * n).map_err(e("gw alloc"))?;
    // ONE class-sized dense slot-gradient buffer shared by every class; the
    // gather kernel re-zeroes exactly the entries it reads.
    let mut d_gv = stream.alloc_zeros::<f64>(vc_len).map_err(e("gv alloc"))?;
    let mut d_g = stream
        .alloc_zeros::<f64>(max_batch_len.max(1) * c_n)
        .map_err(e("g buffer alloc"))?;
    let mut host_gw0 = vec![0.0f64; c_n];
    let mut host_gw = vec![0.0f64; c_n * n];
    let mut slot_buffers: Option<SlotBuffers> = None;
    let mut slot_list: Vec<i64> = Vec::new();
    let mut touched_feat: Vec<usize> = Vec::new();
    let mut seen_feat: Vec<bool> = vec![false; n];
    let mut touched_slot: Vec<usize> = Vec::new();
    let mut seen_slot: Vec<bool> = vec![false; n * n_fields];
    // One accumulator reused for every class's flush.
    let mut accum = FfmGradAccum::new(n, n_fields, k);
    let mut batch_start: usize = 0;
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            // Host-side touched enumeration, shared across classes
            // (standalone scratch: `accum` is reused per class).
            touched_feat.clear();
            touched_slot.clear();
            for &r in batch {
                let (indices, _) = csr.row(r as usize);
                for &i in indices {
                    let i = i as usize;
                    if !seen_feat[i] {
                        seen_feat[i] = true;
                        touched_feat.push(i);
                    }
                }
                for a in 0..indices.len() {
                    let ia = indices[a] as usize;
                    let fa = field_ids[ia] as usize;
                    for &jb in &indices[a + 1..] {
                        let jb = jb as usize;
                        let slot_a = ia * n_fields + field_ids[jb] as usize;
                        let slot_b = jb * n_fields + fa;
                        if !seen_slot[slot_a] {
                            seen_slot[slot_a] = true;
                            touched_slot.push(slot_a);
                        }
                        if !seen_slot[slot_b] {
                            seen_slot[slot_b] = true;
                            touched_slot.push(slot_b);
                        }
                    }
                }
            }
            for &i in &touched_feat {
                seen_feat[i] = false;
            }
            for &s in &touched_slot {
                seen_slot[s] = false;
            }
            let s_count = touched_slot.len();
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
                let bufs = slot_buffers.as_mut().expect("allocated above");
                slot_list.clear();
                slot_list.extend(touched_slot.iter().map(|&s| s as i64));
                stream
                    .memcpy_htod(&slot_list[..], &mut bufs.d_slots)
                    .map_err(e("slots upload"))?;
            }
            stream.memset_zeros(&mut d_gw0).map_err(e("g_w0 zero"))?;
            stream.memset_zeros(&mut d_gw).map_err(e("gw zero"))?;
            let cfg = LaunchConfig {
                grid_dim: (batch.len() as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: ((BLOCK as usize + c_n) * std::mem::size_of::<f64>()) as u32,
            };
            let batch_start_i64 = batch_start as i64;
            let n_i64 = n as i64;
            let n_fields_i64 = n_fields as i64;
            let k_i64 = k as i64;
            let c_i64 = c_n as i64;
            {
                let mut launch = stream.launch_builder(&score_func);
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
                    .arg(&d_w0)
                    .arg(&n_i64)
                    .arg(&n_fields_i64)
                    .arg(&k_i64)
                    .arg(&c_i64)
                    .arg(&label_smoothing)
                    .arg(&off)
                    .arg(&mut d_g)
                    .arg(&mut d_gw0)
                    .arg(&mut d_gw);
                // Safety: the kernel reads/writes exactly the buffers bound
                // above; CSR structure, field_ids range, y range, row_orders
                // range and w0/w/V shapes are validated by the PyO3 layer.
                unsafe { launch.launch(cfg) }.map_err(e("score kernel launch"))?;
            }
            stream.memcpy_dtoh(&d_gw0, &mut host_gw0).map_err(e("g_w0 download"))?;
            stream.memcpy_dtoh(&d_gw, &mut host_gw).map_err(e("gw download"))?;
            let bsz = batch.len() as f64;
            for c in 0..c_n {
                if s_count > 0 {
                    let c_i64_arg = c as i64;
                    let pair_cfg = LaunchConfig {
                        grid_dim: (batch.len() as u32, 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launch = stream.launch_builder(&pair_func);
                    launch
                        .arg(&d_indptr)
                        .arg(&d_indices)
                        .arg(&d_data)
                        .arg(&d_fields)
                        .arg(&d_ro)
                        .arg(&batch_start_i64)
                        .arg(&d_v)
                        .arg(&n_i64)
                        .arg(&n_fields_i64)
                        .arg(&k_i64)
                        .arg(&c_i64)
                        .arg(&c_i64_arg)
                        .arg(&d_g)
                        .arg(&mut d_gv);
                    // Safety: the pair kernel reads class c's frozen V block
                    // and this batch's row gradients, writing only class-local
                    // slot entries of the shared gv buffer.
                    unsafe { launch.launch(pair_cfg) }.map_err(e("pair kernel launch"))?;
                    let bufs = slot_buffers.as_mut().expect("allocated above");
                    let total = (s_count * k) as u32;
                    let gather_cfg = LaunchConfig {
                        grid_dim: (total.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let s_i64 = s_count as i64;
                    let zero_base: i64 = 0;
                    let mut launch = stream.launch_builder(&gather_func);
                    launch
                        .arg(&bufs.d_slots)
                        .arg(&s_i64)
                        .arg(&k_i64)
                        .arg(&zero_base)
                        .arg(&mut d_gv)
                        .arg(&mut bufs.d_vals);
                    // Safety: gather reads/zeroes gv at the uploaded
                    // class-local slot ids and writes s_count * k values into
                    // d_vals (cap >= s_count).
                    unsafe { launch.launch(gather_cfg) }.map_err(e("gather launch"))?;
                    stream
                        .memcpy_dtoh(
                            &bufs.d_vals.slice(..s_count * k),
                            &mut bufs.host_vals[..s_count * k],
                        )
                        .map_err(e("slot grads download"))?;
                    for (s, &slot) in touched_slot.iter().enumerate() {
                        accum.gv[slot * k..(slot + 1) * k]
                            .copy_from_slice(&bufs.host_vals[s * k..(s + 1) * k]);
                    }
                }
                accum.g_w0 = host_gw0[c];
                accum.touched_feat.clear();
                accum.touched_feat.extend_from_slice(&touched_feat);
                accum.touched_slot.clear();
                accum.touched_slot.extend_from_slice(&touched_slot);
                for &i in &touched_feat {
                    accum.gw[i] = host_gw[c * n + i];
                }
                let (wr, vr) = (c * n..(c + 1) * n, c * vc_len..(c + 1) * vc_len);
                let (acc_w0_c, acc_w_c, acc_v_c, mut adam_c, mut ftrl_c) =
                    st.class_views(c, n, vc_len);
                accum.flush(
                    bsz, &mut w0[c], &mut w[wr], &mut v[vr], acc_w0_c, acc_w_c, acc_v_c,
                    &mut adam_c, &mut ftrl_c, k, opt, lr, l1_linear, l2_linear, l1_factors,
                    l2_factors,
                );
                // Scatter class c's touched V slots back into the resident
                // (C, n, n_fields, k) V at slot base c * n * n_fields.
                if s_count > 0 {
                    let bufs = slot_buffers.as_mut().expect("allocated above");
                    for (s, &slot) in touched_slot.iter().enumerate() {
                        bufs.host_vals[s * k..(s + 1) * k].copy_from_slice(
                            &v[c * vc_len + slot * k..c * vc_len + (slot + 1) * k],
                        );
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
                    let slot_base = (c * n * n_fields) as i64;
                    let mut launch = stream.launch_builder(&scatter_func);
                    launch
                        .arg(&bufs.d_slots)
                        .arg(&bufs.d_vals)
                        .arg(&s_i64)
                        .arg(&k_i64)
                        .arg(&slot_base)
                        .arg(&mut d_v);
                    // Safety: scatter writes the k factors of each touched
                    // slot of class c; slot ids are valid by construction.
                    unsafe { launch.launch(scatter_cfg) }.map_err(e("scatter launch"))?;
                }
            }
            stream.memcpy_htod(&*w, &mut d_w).map_err(e("w re-upload"))?;
            stream.memcpy_htod(&*w0, &mut d_w0).map_err(e("w0 re-upload"))?;
            batch_start += batch.len();
        }
    }
    Ok(())
}
