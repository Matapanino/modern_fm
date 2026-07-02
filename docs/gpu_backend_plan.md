# CUDA Backend Plan

This document is an implementation brief for adding optional GPU acceleration to
`modern_fm`. It is deliberately CUDA-first: the current library is optimized
around sparse CSR data, PyO3/Rust kernels, and sklearn-style Python estimators,
and NVIDIA CUDA is the most practical path to a real speedup without replacing
the package with a deep-learning framework.

## Executive Summary

Keep the Rust CPU backend as the default and add CUDA as an optional backend.
The first CUDA milestone should accelerate prediction, then binary/regression
mini-batch gradient accumulation. Do not attempt a complete GPU rewrite first.

Recommended order:

1. FM CSR prediction. — **done (v0.5.0)**
2. FFM CSR prediction. — **done (unreleased, v0.6 work)**
3. FM binary/regression mini-batch gradient accumulation. — **done
   (unreleased, v0.6 work):** dense-buffer `atomicAdd` accumulation
   (`rust/src/cuda/fm_train.rs`), CPU optimizer flush reused verbatim, so all
   four optimizers + early stopping + `partial_fit`/`warm_start` work;
   requires compute capability >= 6.0 (double `atomicAdd`); nondeterministic
   run-to-run — parity is tolerance-based on final predictions. Sparse
   touched-coordinate gradient buffers and device-resident parameters remain
   open before claiming training performance.
4. FFM binary/regression mini-batch gradient accumulation.
5. Later: optimizer flush, multiclass, early-stopping state handoff,
   `partial_fit` / `warm_start` optimizer-state persistence.

The first target should be large sparse CTR-style batches. Small batches,
`batch_size=1`, and small dense toy inputs should stay on CPU because kernel
launch and host/device transfer costs will dominate.

## Current Backend Shape

The public estimators already expose `backend="rust_cpu"` in
`docs/api_design.md`, while private dispatch lives in `python/modern_fm/_backend.py`.
The optimized kernels are in Rust and use float64 CSR triples at the PyO3
boundary:

- FM score and training: `rust/src/fm.rs`.
- FFM score and training: `rust/src/ffm.rs`.
- PyO3 bindings and validation: `rust/src/lib.rs`.
- Optimizer state and update formulas: `rust/src/optimizer.rs`.
- CSR validation/view: `rust/src/data.rs`.

The current training contract is "parallel accumulate, serial apply":

- Rows in a mini-batch are scored against frozen batch-start parameters.
- Data gradients are accumulated per touched coordinate.
- One optimizer step is applied per touched coordinate at batch end.
- `n_jobs>1` only changes row-gradient summation order; CPU reproducibility is
  still controlled and parity-tested.

That contract is GPU-friendly for the accumulation phase, but not for bit-exact
results. CUDA reductions and atomics will change floating-point ordering, so
CUDA parity should be tolerance-based rather than bit-for-bit.

## Acceleration Targets

### FM Prediction

FM prediction is:

```text
s = w0 + sum_i w_i x_i
  + 0.5 * sum_f [(sum_i V[i,f] x_i)^2 - sum_i (V[i,f] x_i)^2]
```

For CSR input this is `O(nnz * k)`. It maps naturally to a CUDA kernel where a
row/block computes:

- linear term over row nonzeros;
- factor cache `sum_f` and `sum_sq_f`;
- final pairwise reduction over `k`.

This is the safest first kernel because it has no optimizer state and only
writes one score per row.

### FFM Prediction

FFM prediction is:

```text
s = w0 + sum_i w_i x_i
  + sum_{i<j} <V[i, field(j)], V[j, field(i)]> x_i x_j
```

For a row with `z` nonzeros this is `O(z^2 * k)`, so it has higher arithmetic
intensity than FM and should benefit more on wide one-hot field data. It also
has more irregular memory access. Start with one block per row or one warp per
feature-pair tile, then profile.

### Training Accumulation

Training is harder because many rows update the same feature coordinates. For
v1, move only the data-gradient accumulation to CUDA and copy the batch
accumulator back to CPU for the existing serial optimizer flush.

That keeps the optimizer implementation and state exact on the CPU side, while
still offloading the row-heavy scoring and gradient math.

Do not move the optimizer flush first. A sparse touched-coordinate optimizer on
GPU needs careful scatter/gather, duplicate coordination, and deterministic
state semantics across SGD/AdaGrad/Adam/FTRL.

## Recommended Architecture

### Backend Selection

Keep `rust_cpu` as the default. Add CUDA behind explicit optional plumbing:

```python
FMClassifier(..., backend="rust_cpu")  # default
FMClassifier(..., backend="cuda")      # require CUDA, fail clearly if missing
FMClassifier(..., backend="auto")      # optional later: CUDA if available else CPU
```

Initial implementation may accept only `backend="rust_cpu" | "cuda"` to avoid
ambiguous performance behavior. If `backend="cuda"` is requested and CUDA is not
available, raise a clear backend error rather than silently falling back.

### Rust Layout

Use a feature-gated CUDA module so standard wheels stay CPU-only:

```text
rust/src/cuda/mod.rs
rust/src/cuda/fm.rs
rust/src/cuda/ffm.rs
rust/src/cuda/kernels/*.cu or embedded NVRTC strings
```

Recommended crate approach:

- Use `cudarc` because it provides Rust wrappers for the CUDA Driver API, NVRTC,
  cuBLAS, cuSPARSE, and dynamic loading.
- Prefer dynamic loading so normal CPU-only builds do not require CUDA libraries
  at build time.
- Hide all CUDA-specific code behind a Cargo feature, for example
  `cuda-backend`.
- Keep PyO3 CUDA entry points private and call them only through
  `python/modern_fm/_backend.py`.

### Data Ownership

For v1, use transfer-inclusive calls:

1. Python converts input to canonical CSR using the existing `_prep_csr`.
2. Rust CUDA function copies CSR arrays and parameters to device memory.
3. CUDA kernel computes output or batch gradients.
4. Rust copies results back to host and returns existing NumPy arrays.

After prediction and accumulation are correct, add a device-resident context to
avoid repeated parameter and CSR transfers across epochs or repeated predict
calls. The CUDA Best Practices Guide specifically warns that host/device
transfers are costly and should be minimized; therefore the benchmark must show
both transfer-inclusive and device-resident timings.

Status: the context/module half of this is done — the device-0 context and the
NVRTC-compiled module are created once per process and cached
(`rust/src/cuda/mod.rs`). Model-parameter/CSR residency across calls is still
open.

## Kernel Strategy

### FM CSR Prediction v1

Implement a simple custom kernel first:

- One CUDA block per row.
- Threads cooperate across row nonzeros and latent factor dimension `k`.
- Use shared memory for per-factor `sum` and `sum_sq` when `k` is small enough.
- Write exactly one score per row.

Optimize only after profiling:

- If average `nnz` is small, group multiple rows per block.
- If `k` is large, tile `k` across warps.
- Use coalesced loads for `V[i, f]` where possible; row-major `(n_features, k)`
  makes adjacent factor loads contiguous.

Alternative library path:

- FM needs `X @ V` and `X @ (V^2)`-like sparse-dense products for prediction.
- cuSPARSE SpMM can accelerate this, but the linear term and final row-wise
  reduction still need custom kernels.
- Use cuSPARSE only after the direct kernel baseline exists; it may add layout
  and transfer complexity before the benefit is proven.

### FFM CSR Prediction v1

Implement custom kernels rather than forcing the problem into cuSPARSE:

- One block per row for moderate `z`.
- For rows with large `z`, split pair ranges across multiple blocks and reduce
  partial row scores.
- Cache row feature ids, values, and field ids in shared memory when feasible.
- Load `V[(i * n_fields + field_j) * k + t]` and the symmetric slot for each
  pair.

This path is likely the strongest GPU payoff because FFM is `O(z^2 * k)`, but it
will also be the most sensitive to sparse-row imbalance.

### Training Accumulation v1

Use a two-stage design:

1. CUDA computes per-row score, loss gradient, and data-gradient contributions.
2. Contributions are reduced to a batch accumulator.
3. CPU performs the existing optimizer flush.

For the first implementation, prioritize correctness over memory efficiency:

- Use dense gradient buffers shaped like existing CPU accumulators if the test
  problem is small enough.
- For real CTR shapes, switch to sparse triplets or sorted touched-coordinate
  buffers before claiming performance.
- Avoid GPU atomics as the only design until contention is measured; feature
  collisions in one-hot data can make atomic-heavy kernels slow and
  nondeterministic.

For FM accumulation, the row-local cache `sum_f = sum_i V[i,f] x_i` is reusable
for both score and `V` gradient. For FFM accumulation, reuse the pairwise score
loop to emit the two factor-slot gradients per `(i, j)` pair.

## API And Packaging

Do not make CUDA a hard dependency.

Recommended packaging behavior:

- CPU-only source/wheels continue to build exactly as they do now.
- CUDA support is opt-in through a Cargo feature and a Python extra only if the
  project chooses to expose one.
- Importing `modern_fm` must not fail on machines without CUDA.
- `has_rust()` should keep its current meaning. Add a separate private helper
  such as `_backend.has_cuda()` if needed.

Recommended validation behavior:

- `backend="rust_cpu"`: current behavior.
- `backend="cuda"` with unsupported model/mode: raise `NotImplementedError` or
  `ValueError` with the exact unsupported cell, for example "CUDA backend
  currently supports FM/FFM prediction and binary/regression fit only".
- `backend="cuda"` without CUDA runtime/device: raise a clear runtime backend
  error.
- `backend="auto"` if added later: fall back to CPU only when CUDA is absent,
  not when CUDA exists but a kernel fails.

## Testing And Benchmarks

### Correctness Tests

Add CUDA tests that skip when CUDA is unavailable:

- FM CSR prediction parity vs `_reference.fm_predict_fast` and Rust CPU.
- FFM CSR prediction parity vs `_reference.ffm_predict` and Rust CPU.
- Empty rows return the bias.
- Single-nonzero rows have no pairwise term.
- Dense input follows the existing dense-to-CSR training path where applicable.
- Binary logistic and squared-loss training on small deterministic datasets once
  CUDA accumulation exists.

Use tolerance-based assertions. CUDA will change floating-point reduction order;
the CUDA Best Practices Guide calls out non-associativity of floating-point
addition in parallel computation.

Suggested initial tolerances:

- Prediction float64: `rtol=1e-10`, `atol=1e-10`.
- Training float64: compare final predictions and loss trend, not raw parameter
  bit patterns; start at `rtol=1e-7`, `atol=1e-8` and tighten only if stable.
- Float32 model attributes: compare at float32-appropriate tolerances.

### Benchmark Script

Add a CUDA benchmark, for example `benchmarks/bench_cuda.py`, reporting:

- FM/FFM prediction throughput: CPU Rust vs CUDA.
- FM/FFM fit time for binary/regression once accumulation is implemented.
- Transfer-inclusive timing.
- Device-resident timing if a CUDA context/cache is added.
- Dataset dimensions: rows, features, fields, average nnz, `k`, density,
  `batch_size`.
- GPU model, CUDA version, driver version, CPU model, thread count.

Benchmark cases:

- FM: `n_rows` in `{10_000, 100_000, 1_000_000}`, average `nnz` in `{8, 32, 128}`,
  `k` in `{8, 16, 32, 64}`.
- FFM: average `nnz` in `{8, 16, 32}`, `n_fields` in `{8, 16, 32}`, `k` in
  `{4, 8, 16}`.
- Training: `batch_size` in `{256, 1024, 8192, full}`.

Do not report a CUDA speedup unless transfer-inclusive timing beats CPU on a
realistic large sparse workload.

## Risks

- Host/device transfer can erase speedups for small batches.
- FFM row lengths can be highly imbalanced, causing poor occupancy.
- Sparse gradient reduction can be dominated by atomic contention or sort/reduce
  overhead.
- cuSPARSE SpMM may not match FM/FFM's custom scoring shape well enough to
  justify integration complexity.
- CUDA reductions will not be bit-identical to CPU; tests and docs must accept
  tolerance-based parity.
- FTRL and Adam state updates are per-coordinate and lazy; moving them to GPU is
  a separate project after accumulation is proven.
- Optional CUDA packaging can complicate wheels. Keep the CPU build path
  untouched.

## Research Notes

- NVIDIA's CUDA Programming Guide is the primary CUDA programming-model
  reference and is kept current with CUDA releases:
  https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
- NVIDIA's CUDA Best Practices Guide recommends the APOD cycle: assess,
  parallelize, optimize, deploy. It also emphasizes minimizing host/device data
  transfer, using coalesced global memory access, and comparing floating-point
  results within tolerances:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- cuSPARSE expects vectors and matrices to reside in device memory, supports CSR
  descriptors, and can gracefully coexist with a CPU-only path when linked in a
  way that does not require CUDA at build time:
  https://docs.nvidia.com/cuda/cusparse/index.html
- `cudarc` is the best Rust fit found for this codebase. It provides wrappers
  for the CUDA Driver API, NVRTC, cuBLAS, cuSPARSE, and dynamic loading:
  https://docs.rs/cudarc/latest/cudarc/
- GE-SpMM is a useful design reference for CSR sparse-dense GPU kernels. Its
  main lesson for this project is to avoid assuming SpMV-style kernels will be
  efficient for SpMM-like work; coalescing, row caching, and sparse-row reuse
  matter:
  https://arxiv.org/abs/2007.03179

## Claude Code Prompt

```text
You are implementing optional CUDA acceleration for modern_fm.

Read these files before editing:
- CLAUDE.md
- docs/math_spec.md
- docs/optimization_spec.md
- docs/api_design.md
- docs/data_format.md
- docs/gpu_backend_plan.md
- python/modern_fm/_backend.py
- rust/src/lib.rs
- rust/src/fm.rs
- rust/src/ffm.rs
- rust/src/optimizer.rs

Non-negotiable constraints:
- Do not change the NumPy reference implementations for speed.
- Keep Rust CPU behavior and public sklearn API backward-compatible.
- Keep CPU-only builds and imports working on machines without CUDA.
- Add CUDA behind an optional backend path; CPU remains the default.
- Preserve existing dtype, CSR, validation, serialization, and estimator
  contracts unless docs are updated in the same change.

Start with the narrowest useful milestone:
1. Add optional CUDA backend plumbing behind Cargo/Python feature gates.
2. Add backend validation for backend="cuda" with a clear error when CUDA is
   unavailable.
3. Implement FM CSR prediction on CUDA.
4. Add parity tests that skip when CUDA is unavailable.
5. Add a benchmark comparing Rust CPU and CUDA, including transfer-inclusive
   timing.
6. Only after FM prediction is correct, proceed to FFM CSR prediction.

Preferred implementation direction:
- Use Rust + cudarc/dynamic loading if practical.
- Keep CUDA PyO3 functions private and route through python/modern_fm/_backend.py.
- Implement custom kernels before integrating cuSPARSE, unless profiling shows
  cuSPARSE is clearly simpler and faster for FM prediction.
- For v1 training, offload mini-batch gradient accumulation before moving
  optimizer flushes to GPU.

Do not implement multiclass CUDA training, early-stopping optimizer-state
handoff, partial_fit/warm_start optimizer-state persistence, or GPU optimizer
flush until prediction parity and benchmarks are in place.

Acceptance criteria for the first PR:
- CPU-only tests still pass without CUDA.
- CUDA tests are skipped cleanly when CUDA is unavailable.
- FM CSR CUDA prediction matches the Python reference and Rust CPU within a
  documented floating-point tolerance.
- backend="cuda" gives a clear error if CUDA support is not compiled or no CUDA
  device/runtime is available.
- benchmarks/bench_cuda.py reports CPU Rust vs CUDA timings and includes
  transfer-inclusive timing.
```
