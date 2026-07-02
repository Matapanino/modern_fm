//! CUDA FwFM CSR prediction — milestone 6 of docs/gpu_backend_plan.md.
//!
//! Same geometry as the FFM prediction kernel (`cuda::ffm`): one 256-thread
//! block per row, strided pair loop, shared-memory tree reduction, no limit
//! on row nnz or k. The differences follow the FwFM math
//! (docs/math_spec_fwfm.md): `V` is FM-shaped row-major `(n_features, k)`
//! and each pair's k-dot is scaled by the field-pair weight
//! `r[min(f_i,f_j) * n_fields + max(f_i,f_j)]` (upper triangle, like
//! `crate::fwfm::pair_slot`). Tolerance-based parity vs the CPU paths
//! (rtol/atol 1e-10, tests/test_cuda_parity.py). Transfer-inclusive like the
//! FM/FFM prediction kernels.

use cudarc::driver::{LaunchConfig, PushKernelArg};

/// Threads per block; a power of two for the tree reduction.
const BLOCK: u32 = 256;

/// Compiled once per process into the shared module (`super::gpu`).
pub(super) const KERNEL_SRC: &str = r#"
extern "C" __global__ void fwfm_predict_csr(
    const long long* indptr,
    const long long* indices,
    const double* data,
    const long long* field_ids,
    const double* w,
    const double* v,
    const double* r,
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
        out[row] = w0 + partial[0];
    }
}
"#;

/// FwFM CSR prediction on the first CUDA device. Errors are stringified for
/// the PyO3 layer (mapped to RuntimeError there).
#[allow(clippy::too_many_arguments)]
pub fn predict_csr(
    indptr: &[i64],
    indices: &[i64],
    data: &[f64],
    field_ids: &[i64],
    w: &[f64],
    v: &[f64],
    r: &[f64],
    w0: f64,
    n_fields: usize,
    k: usize,
) -> Result<Vec<f64>, String> {
    let n_rows = indptr.len() - 1;
    if n_rows == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Err("CUDA FwFM prediction requires k >= 1".to_string());
    }
    fn e<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> String {
        move |err| format!("CUDA {what} failed: {err:?}")
    }
    let (ctx, module) = super::gpu()?;
    let func = module.load_function("fwfm_predict_csr").map_err(e("function load"))?;
    let stream = ctx.default_stream();
    let d_indptr = stream.clone_htod(indptr).map_err(e("indptr upload"))?;
    let d_indices = stream.clone_htod(indices).map_err(e("indices upload"))?;
    let d_data = stream.clone_htod(data).map_err(e("data upload"))?;
    let d_fields = stream.clone_htod(field_ids).map_err(e("field_ids upload"))?;
    let d_w = stream.clone_htod(w).map_err(e("w upload"))?;
    let d_v = stream.clone_htod(v).map_err(e("V upload"))?;
    let d_r = stream.clone_htod(r).map_err(e("R upload"))?;
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
        .arg(&d_r)
        .arg(&w0)
        .arg(&n_fields_i64)
        .arg(&k_i64)
        .arg(&mut d_out);
    // Safety: the kernel reads/writes exactly the buffers bound above, with
    // shapes validated by the PyO3 layer (CSR structure, field_ids range,
    // w/V/R lengths).
    unsafe { launch.launch(cfg) }.map_err(e("kernel launch"))?;
    stream.clone_dtoh(&d_out).map_err(e("output download"))
}
