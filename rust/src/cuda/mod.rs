//! Optional CUDA backend plumbing (docs/gpu_backend_plan.md).
//!
//! Compiled only with `--features cuda-backend` on non-macOS targets. cudarc's
//! dynamic loading defers all CUDA resolution to runtime, so builds need no
//! CUDA toolkit and importing the extension never fails on CUDA-less
//! machines. Kernels merge only after validation on a real GPU
//! (docs/cuda_validation_runbook.md).
//!
//! The device-0 context and the NVRTC-compiled module holding every
//! prediction kernel are created once per process and cached (see [`gpu`]).
//! Calls stay transfer-inclusive: only initialization is amortized, the CSR
//! arrays and parameters are still copied per call.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaModule};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

pub mod ffm;
pub mod ffm_train;
pub mod fm;
pub mod fm_train;

/// True when a CUDA driver can be loaded and at least one device exists.
/// Every failure mode (no libcuda, no device, init error) reports
/// unavailable rather than erroring. Uses `device_count()`, which does not
/// create a context — probing stays cheap and never warms the cache.
pub fn available() -> bool {
    matches!(cudarc::driver::CudaContext::device_count(), Ok(n) if n > 0)
}

struct Gpu {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
}

static GPU: Mutex<Option<Gpu>> = Mutex::new(None);

/// Cached (context, module) for the first CUDA device.
///
/// The first call creates the context and NVRTC-compiles all prediction
/// kernels into one module; later calls only clone the `Arc`s. Failures are
/// not cached — a transient init error (e.g. momentary device OOM) is retried
/// on the next call. cudarc's `CudaContext`/`CudaModule` are `Send + Sync`
/// and the safe API binds the context to the calling thread, so the cached
/// handles are safe under `py.allow_threads` from multiple Python threads
/// (all calls share the default stream, so GPU work serializes).
///
/// Caveat: a CUDA context does not survive `fork()`; a child process forked
/// after the first CUDA call inherits an unusable cached context.
pub(crate) fn gpu() -> Result<(Arc<CudaContext>, Arc<CudaModule>), String> {
    let mut slot = GPU
        .lock()
        .map_err(|_| "CUDA state mutex poisoned".to_string())?;
    if slot.is_none() {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context creation failed: {e:?}"))?;
        let src = format!(
            "{}\n{}\n{}\n{}",
            fm::KERNEL_SRC,
            ffm::KERNEL_SRC,
            fm_train::KERNEL_SRC,
            ffm_train::KERNEL_SRC
        );
        // compute_60: the training kernel's atomicAdd(double*, double) needs
        // compute capability >= 6.0 (Pascal, 2016). The PTX JIT-compiles
        // forward on every newer architecture.
        let opts = CompileOptions {
            arch: Some("compute_60"),
            ..Default::default()
        };
        let ptx =
            compile_ptx_with_opts(src, opts).map_err(|e| format!("NVRTC compile failed: {e:?}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("CUDA module load failed: {e:?}"))?;
        *slot = Some(Gpu { ctx, module });
    }
    let gpu = slot.as_ref().expect("just initialized");
    Ok((gpu.ctx.clone(), gpu.module.clone()))
}
