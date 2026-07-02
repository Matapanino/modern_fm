# CUDA Validation Runbook

How to validate modern_fm's CUDA kernels on a real NVIDIA machine. **CUDA
kernel PRs merge only after this runbook has been executed green** and the
results are pasted into the PR (docs/gpu_backend_plan.md). The development
machines are CUDA-less (Apple Silicon); CI only compile-checks the
`cuda-backend` feature (`cuda-check` job) — it never executes kernels.

## 0. Automated path: Colab GPU (recommended)

With the Colab CLI installed and authenticated
(`uv tool install google-colab-cli`), the whole runbook is one command from a
committed tree:

```bash
bash scripts/colab_gpu_test.sh              # T4; add --full-bench for the full grid
```

It provisions a T4 VM, uploads `git archive HEAD`, installs rustup + maturin,
builds with `--features cuda-backend`, asserts `has_cuda()`, runs
`tests/test_cuda_parity.py` (pass-not-skip) + the full suite +
`benchmarks/bench_cuda.py`, downloads the report to
`/tmp/<date>-modernfm-cuda-validation.md`, and stops the VM. Paste that report
into the PR. Steps 1–5 below are the manual equivalent for any other GPU box.

## 1. Get a GPU box

Any Linux x86_64 machine with an NVIDIA GPU (compute capability >= 6.0 for
`double` atomics headroom; any T4/L4/A10/RTX class card is fine) and a recent
driver (CUDA 12+). Cheap options: Lambda Cloud, RunPod, Vast.ai, AWS
`g4dn.xlarge`. No CUDA *toolkit* is required — cudarc dynamically loads the
driver and NVRTC ships with it — but installing the toolkit is harmless.

Check the driver:

```bash
nvidia-smi   # should list the GPU and a CUDA version >= 12
```

## 2. Build with the cuda-backend feature

```bash
git clone https://github.com/Matapanino/modern_fm && cd modern_fm
git checkout <the CUDA PR branch>
python3 -m venv .venv
.venv/bin/pip install maturin
# build the extension with the CUDA feature enabled, plus dev deps
.venv/bin/pip install -e ".[dev]" --config-settings=build-args="--features cuda-backend"
```

Confirm the build actually has CUDA:

```bash
.venv/bin/python -c "from modern_fm import _backend; print(_backend.has_cuda())"
# expected: True    (False means the feature flag or the driver is missing)
```

## 3. Correctness: the parity suite must pass, not skip

```bash
.venv/bin/pytest -q tests/test_cuda_parity.py -rs
```

Expected: `N passed` with **zero skipped** (the whole file skips when
`has_cuda()` is false — a skip means step 2 failed). The regular suite must
also stay green: `.venv/bin/pytest -q`.

## 4. Benchmark (transfer-inclusive)

```bash
.venv/bin/python benchmarks/bench_cuda.py
```

Paste the full table plus `nvidia-smi --query-gpu=name,driver_version
--format=csv` output into the PR. Per docs/gpu_backend_plan.md: do **not**
claim a CUDA speedup unless the transfer-inclusive timing beats the Rust CPU
kernel on a realistic large sparse workload (small batches are expected to
lose to the CPU — that is why `rust_cpu` stays the default).

## 5. Report

The PR description gets:

- [ ] `has_cuda() == True` on the box (GPU model + driver version)
- [ ] `tests/test_cuda_parity.py`: all passed, none skipped
- [ ] full `.venv/bin/pytest -q` green
- [ ] `bench_cuda.py` table (transfer-inclusive)

## Notes / troubleshooting

- `RuntimeError: ... requires modern_fm built with the cuda-backend ...` at
  predict time → the wheel in the venv was built without the feature; redo
  step 2 (a plain `pip install -e .` rebuild silently drops the flag).
- NVRTC errors mention the kernel source: all kernels are compiled together
  into one module at the first CUDA call and cached process-wide
  (`rust/src/cuda/mod.rs`), so a driver too old for the generated PTX
  surfaces on that first call.
- The current scope is **every cell**: FM/FFM/FwFM CSR prediction +
  binary/regression/multiclass training (CUDA accumulation, CPU optimizer
  flush — per class via `McState::class_views` for multiclass, with the FwFM
  R group flushed through `GroupStateMut`/`McGroupState`).
  Training parity is tolerance-based on final predictions (atomicAdd makes it
  nondeterministic run-to-run) and needs compute capability >= 6.0.
- `colab exec` runs with `--timeout 7200`: the full FM grid (1M-row cells)
  plus the FFM grid can exceed the old 3600 s ceiling under `--full-bench`.
