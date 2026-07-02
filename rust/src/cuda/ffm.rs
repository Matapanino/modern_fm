//! CUDA FFM CSR prediction — milestone 2 of docs/gpu_backend_plan.md.
//!
//! One block per row with a fixed 256-thread block: threads split the linear
//! term and the O(z²) pair loop by striding (outer nonzero `a` serial, inner
//! `b` strided across the block), each computing the k-dot
//! `<V[i, f_j], V[j, f_i]>` serially, then one block-wide shared-memory tree
//! reduction. No shared-memory row caching in v1: it would cap row nnz at
//! ~2000 on 48 KB parts and force a second code path, while the V loads
//! (2k doubles per pair) dominate global traffic anyway — coalesced V tiling
//! and row caching are post-profiling follow-ups, correctness-first per the
//! plan doc. No limit on row nnz or k (k is a serial per-thread loop, unlike
//! the FM kernel's block-dim bound). Tolerance-based parity vs the CPU paths
//! (rtol/atol 1e-10, tests/test_cuda_parity.py). Transfer-inclusive: every
//! call copies the CSR arrays and parameters to the device and the scores
//! back, but the context and compiled module are process-cached
//! (see `super::gpu`).

use cudarc::driver::{LaunchConfig, PushKernelArg};

/// Threads per block; a power of two for the tree reduction.
const BLOCK: u32 = 256;

/// Compiled once per process into the shared module (`super::gpu`);
/// `long long` matches the i64 CSR/field arrays. V layout is row-major
/// `(n_features, n_fields, k)`, addressed as `v[(i * n_fields + f) * k + t]`
/// exactly like the CPU kernel (`crate::ffm::ffm_score_row`).
pub(super) const KERNEL_SRC: &str = r#"
extern "C" __global__ void ffm_predict_csr(
    const long long* indptr,
    const long long* indices,
    const double* data,
    const long long* field_ids,
    const double* w,
    const double* v,
    const double w0,
    const long long n_fields,
    const long long k,
    double* out)
{
    long long row = blockIdx.x;
    long long lo = indptr[row];
    long long hi = indptr[row + 1];
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
        out[row] = w0 + partial[0];
    }
}
"#;

/// FFM CSR prediction on the first CUDA device. Errors are stringified for
/// the PyO3 layer (mapped to RuntimeError there).
#[allow(clippy::too_many_arguments)]
pub fn predict_csr(
    indptr: &[i64],
    indices: &[i64],
    data: &[f64],
    field_ids: &[i64],
    w: &[f64],
    v: &[f64],
    w0: f64,
    n_fields: usize,
    k: usize,
) -> Result<Vec<f64>, String> {
    let n_rows = indptr.len() - 1;
    if n_rows == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Err("CUDA FFM prediction requires k >= 1".to_string());
    }
    fn e<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> String {
        move |err| format!("CUDA {what} failed: {err:?}")
    }
    let (ctx, module) = super::gpu()?;
    let func = module.load_function("ffm_predict_csr").map_err(e("function load"))?;
    let stream = ctx.default_stream();
    let d_indptr = stream.clone_htod(indptr).map_err(e("indptr upload"))?;
    let d_indices = stream.clone_htod(indices).map_err(e("indices upload"))?;
    let d_data = stream.clone_htod(data).map_err(e("data upload"))?;
    let d_fields = stream.clone_htod(field_ids).map_err(e("field_ids upload"))?;
    let d_w = stream.clone_htod(w).map_err(e("w upload"))?;
    let d_v = stream.clone_htod(v).map_err(e("V upload"))?;
    let mut d_out = stream.alloc_zeros::<f64>(n_rows).map_err(e("output alloc"))?;
    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: BLOCK * std::mem::size_of::<f64>() as u32,
    };
    let n_fields_i64 = n_fields as i64;
    let k_i64 = k as i64;
    let mut launch = stream.launch_builder(&func);
    launch
        .arg(&d_indptr)
        .arg(&d_indices)
        .arg(&d_data)
        .arg(&d_fields)
        .arg(&d_w)
        .arg(&d_v)
        .arg(&w0)
        .arg(&n_fields_i64)
        .arg(&k_i64)
        .arg(&mut d_out);
    // Safety: the kernel reads/writes exactly the buffers bound above, with
    // shapes validated by the PyO3 layer (CSR structure, field_ids range,
    // w/V lengths).
    unsafe { launch.launch(cfg) }.map_err(e("kernel launch"))?;
    stream.clone_dtoh(&d_out).map_err(e("output download"))
}
