# benchmarks

- `bench_synthetic.py` — fit time and predict throughput on synthetic sparse
  data, plus the speedup of the Rust backend over the NumPy reference trainer.
- `bench_vs_baseline.py` — test AUC, fit time, and predict throughput vs a
  scikit-learn `LogisticRegression` baseline on synthetic CTR data with planted
  pairwise interactions, across an `n_jobs` / `batch_size` sweep (auto-includes
  `xlearn` if importable).
- `bench_cuda.py` — FM/FFM prediction, Rust CPU vs CUDA, transfer-inclusive
  (plus a cold-start line for the process-cached context/NVRTC module). Needs
  a `cuda-backend` build + GPU; run via `scripts/colab_gpu_test.sh` per
  `docs/cuda_validation_runbook.md`.

```bash
.venv/bin/python benchmarks/bench_synthetic.py
.venv/bin/python benchmarks/bench_vs_baseline.py
.venv/bin/python benchmarks/bench_cuda.py  # CUDA machine only
```

## CUDA T4 results

_To be filled from the next Colab T4 validation run (the table lands here with
GPU model, driver, and CUDA version — see docs/cuda_validation_runbook.md)._

See `docs/benchmark_plan.md` for goals and rules (fixed seeds, report machine
specs, do not tune the library to a benchmark).
