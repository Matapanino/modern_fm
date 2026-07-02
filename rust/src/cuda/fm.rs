//! CUDA FM CSR prediction — milestone 1 of docs/gpu_backend_plan.md.
//!
//! One block per row, one thread per latent factor: thread `f` accumulates
//! `sum_f = Σᵢ v_{i,f} xᵢ` and `sq_f = Σᵢ (v_{i,f} xᵢ)²` over the row's
//! nonzeros and writes its pairwise contribution `sum_f² − sq_f` to shared
//! memory; thread 0 adds the linear term and reduces. Correctness-first per
//! the plan doc (coalescing/tiling only after profiling); tolerance-based
//! parity vs the CPU paths (CUDA reduction order differs — rtol/atol 1e-10,
//! see tests/test_cuda_parity.py). Transfer-inclusive: every call copies the
//! CSR arrays and parameters to the device and the scores back.

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

/// NVRTC-compiled at first use; `long long` matches the i64 CSR arrays.
const KERNEL_SRC: &str = r#"
extern "C" __global__ void fm_predict_csr(
    const long long* indptr,
    const long long* indices,
    const double* data,
    const double* w,
    const double* v,
    const double w0,
    const long long k,
    double* out)
{
    long long row = blockIdx.x;
    long long lo = indptr[row];
    long long hi = indptr[row + 1];
    extern __shared__ double pair[];  // k pairwise contributions
    long long f = threadIdx.x;
    if (f < k) {
        double sum = 0.0;
        double sq = 0.0;
        for (long long p = lo; p < hi; ++p) {
            double vx = v[indices[p] * k + f] * data[p];
            sum += vx;
            sq += vx * vx;
        }
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
        out[row] = w0 + lin + 0.5 * pw;
    }
}
"#;

/// FM CSR prediction on the first CUDA device. Errors are stringified for the
/// PyO3 layer (mapped to RuntimeError there).
#[allow(clippy::too_many_arguments)]
pub fn predict_csr(
    indptr: &[i64],
    indices: &[i64],
    data: &[f64],
    w: &[f64],
    v: &[f64],
    w0: f64,
    k: usize,
) -> Result<Vec<f64>, String> {
    let n_rows = indptr.len() - 1;
    if n_rows == 0 {
        return Ok(Vec::new());
    }
    if k == 0 || k > 1024 {
        return Err(format!("CUDA FM prediction supports 1 <= k <= 1024, got {k}"));
    }
    fn e<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> String {
        move |err| format!("CUDA {what} failed: {err:?}")
    }
    let ctx = CudaContext::new(0).map_err(e("context creation"))?;
    let ptx = compile_ptx(KERNEL_SRC).map_err(|err| format!("NVRTC compile failed: {err:?}"))?;
    let module = ctx.load_module(ptx).map_err(e("module load"))?;
    let func = module.load_function("fm_predict_csr").map_err(e("function load"))?;
    let stream = ctx.default_stream();
    let d_indptr = stream.clone_htod(indptr).map_err(e("indptr upload"))?;
    let d_indices = stream.clone_htod(indices).map_err(e("indices upload"))?;
    let d_data = stream.clone_htod(data).map_err(e("data upload"))?;
    let d_w = stream.clone_htod(w).map_err(e("w upload"))?;
    let d_v = stream.clone_htod(v).map_err(e("V upload"))?;
    let mut d_out = stream.alloc_zeros::<f64>(n_rows).map_err(e("output alloc"))?;
    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (k as u32, 1, 1),
        shared_mem_bytes: (k * std::mem::size_of::<f64>()) as u32,
    };
    let k_i64 = k as i64;
    let mut launch = stream.launch_builder(&func);
    launch
        .arg(&d_indptr)
        .arg(&d_indices)
        .arg(&d_data)
        .arg(&d_w)
        .arg(&d_v)
        .arg(&w0)
        .arg(&k_i64)
        .arg(&mut d_out);
    // Safety: the kernel reads/writes exactly the buffers bound above, with
    // shapes validated by the PyO3 layer (CSR structure, w/V lengths).
    unsafe { launch.launch(cfg) }.map_err(e("kernel launch"))?;
    stream.clone_dtoh(&d_out).map_err(e("output download"))
}
