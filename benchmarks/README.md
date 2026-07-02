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

## CUDA T4 results (2026-07-02)

Colab Tesla T4 (driver 580.82.07, CUDA 13.0), Linux x86_64, Python 3.12,
host CPU 2 vCPU. Transfer-inclusive (every call copies CSR + params to the
device and scores back); `bench_cuda.py --quick`, median of 5 after warmup,
`d=100_000` features. CPU FFM prediction is serial.

```
first CUDA call (context + NVRTC compile, cached afterwards): 315.3 ms
FM prediction (Rust CPU vs CUDA, transfer-inclusive)
      rows  nnz/row    k     cpu ms    cuda ms  speedup
    100000        8    8       46.1       17.4     2.6x
    100000        8   32       86.1       21.0     4.1x
    100000       32    8      114.3       48.6     2.3x
    100000       32   32      235.5       45.7     5.2x

FFM prediction (Rust CPU vs CUDA, transfer-inclusive; CPU FFM is serial)
      rows  nnz/row  fields    k    V MB     cpu ms    cuda ms  speedup
    100000        8       8    8      51      100.6       34.9     2.9x
    100000        8      32    8     205      137.7       67.4     2.0x
    100000       32       8    8      51      895.7       86.9    10.3x
    100000       32      32    8     205     1577.5      130.5    12.1x
```

See `docs/benchmark_plan.md` for goals and rules (fixed seeds, report machine
specs, do not tune the library to a benchmark).
