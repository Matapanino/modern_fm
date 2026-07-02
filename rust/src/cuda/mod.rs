//! Optional CUDA backend plumbing (docs/gpu_backend_plan.md).
//!
//! Compiled only with `--features cuda-backend` on non-macOS targets. cudarc's
//! dynamic loading defers all CUDA resolution to runtime, so builds need no
//! CUDA toolkit and importing the extension never fails on CUDA-less
//! machines. Kernels land separately (FM CSR prediction first) and merge only
//! after validation on a real GPU.

pub mod fm;

/// True when a CUDA driver can be loaded and at least one device exists.
/// Every failure mode (no libcuda, no device, init error) reports
/// unavailable rather than erroring.
pub fn available() -> bool {
    matches!(cudarc::driver::CudaContext::device_count(), Ok(n) if n > 0)
}
