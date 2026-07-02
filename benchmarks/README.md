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
host CPU 2 vCPU. Transfer-inclusive; full `bench_cuda.py` grid, median of 5
after warmup, `d=100_000` features. CPU FFM prediction is serial; the CPU
training baseline is `n_jobs=1`.

```
first CUDA call (context + NVRTC compile, cached afterwards): 391.3 ms
FM prediction (Rust CPU vs CUDA, transfer-inclusive)
      rows  nnz/row    k     cpu ms    cuda ms  speedup
     10000        8    8        3.8        2.8     1.4x
     10000        8   16        6.0        3.9     1.5x
     10000        8   32       10.9        7.5     1.5x
     10000        8   64       18.6       12.9     1.4x
     10000       32    8       11.7        4.6     2.5x
     10000       32   16       19.7        6.0     3.3x
     10000       32   32       28.0        8.5     3.3x
     10000       32   64       61.3       16.8     3.7x
     10000      128    8       42.1       16.6     2.5x
     10000      128   16       84.1       16.7     5.0x
     10000      128   32      148.0       21.6     6.8x
     10000      128   64      267.2       30.2     8.9x
    100000        8    8       31.1       12.8     2.4x
    100000        8   16       59.5       14.6     4.1x
    100000        8   32       89.6       19.0     4.7x
    100000        8   64      176.4       29.6     6.0x
    100000       32    8      116.7       40.9     2.9x
    100000       32   16      250.2       49.2     5.1x
    100000       32   32      301.6       47.9     6.3x
    100000       32   64      712.2       62.4    11.4x
    100000      128    8      416.4      205.3     2.0x
    100000      128   16      603.5      186.2     3.2x
    100000      128   32     1366.4      192.8     7.1x
    100000      128   64     2587.9      217.2    11.9x
   1000000        8    8      430.9      115.9     3.7x
   1000000        8   16      424.2      125.5     3.4x
   1000000        8   32      830.1      135.1     6.1x
   1000000        8   64     1783.1      157.2    11.3x
   1000000       32    8     1064.5      432.8     2.5x
   1000000       32   16     1625.1      479.9     3.4x
   1000000       32   32     2952.8      492.1     6.0x
   1000000       32   64     6795.9      488.3    13.9x
   1000000      128    8     4015.4     1659.7     2.4x
   1000000      128   16     6519.1     1689.0     3.9x
   1000000      128   32    12355.8     1733.9     7.1x
   1000000      128   64    26895.0     1901.3    14.1x

FFM prediction (Rust CPU vs CUDA, transfer-inclusive; CPU FFM is serial)
      rows  nnz/row  fields    k    V MB     cpu ms    cuda ms  speedup
    100000        8       8    4      26       62.0       19.1     3.2x
    100000        8       8    8      51      191.9       33.3     5.8x
    100000        8       8   16     102      180.5       53.4     3.4x
    100000        8      16    4      51       94.2       28.4     3.3x
    100000        8      16    8     102      140.0       46.5     3.0x
    100000        8      16   16     205      208.3       71.0     2.9x
    100000        8      32    4     102      111.5       37.8     2.9x
    100000        8      32    8     205      146.2       60.5     2.4x
    100000        8      32   16     410      233.6      124.4     1.9x
    100000       16       8    4      26      182.5       40.5     4.5x
    100000       16       8    8      51      341.0       54.6     6.3x
    100000       16       8   16     102      529.9       73.1     7.2x
    100000       16      16    4      51      276.8       47.4     5.8x
    100000       16      16    8     102      582.6       66.2     8.8x
    100000       16      16   16     205      628.7       93.1     6.8x
    100000       16      32    4     102      361.8       58.0     6.2x
    100000       16      32    8     205      544.4       82.7     6.6x
    100000       16      32   16     410      753.9      134.6     5.6x
    100000       32       8    4      26      631.3       75.0     8.4x
    100000       32       8    8      51      874.9       89.5     9.8x
    100000       32       8   16     102     1489.5      109.7    13.6x
    100000       32      16    4      51      827.5       89.0     9.3x
    100000       32      16    8     102     1365.4      103.2    13.2x
    100000       32      16   16     205     1903.2      144.8    13.1x
    100000       32      32    4     102     1122.0       88.5    12.7x
    100000       32      32    8     205     1683.1      116.3    14.5x
    100000       32      32   16     410     2379.1      212.4    11.2x

FM training, 1 epoch (Rust CPU n_jobs=1 vs CUDA accumulation + CPU flush; rows=100000, nnz/row=32, k=8)
     batch     cpu ms    cuda ms  speedup
       256      550.8      768.9     0.7x
      1024      488.1      619.4     0.8x
      8192      334.9      341.6     1.0x
      full      209.3      135.9     1.5x

FFM training, 1 epoch (Rust CPU n_jobs=1 vs CUDA accumulation + CPU flush; rows=100000, nnz/row=32, fields=8, k=4)
     batch     cpu ms    cuda ms  speedup
       256     4281.4     4204.7     1.0x
      1024     4100.6     3669.7     1.1x
      8192     2703.1     2064.9     1.3x
      full     1784.5      500.0     3.6x
```

Training notes (honesty): both training tables are from one run (parameters
device-resident, touched-coordinate transfers). FM training is bounded by
kernel-launch overhead / low occupancy at small batches (an earlier
dense-transfer design was 0.3x at batch 256; the compact path brought CUDA
time from 2256 ms to ~800 ms there). FFM training's higher O(z²k) arithmetic
intensity suits the GPU better — it breaks even at batch 256 and reaches 3.6x
at full batch. Colab T4 *instances vary a lot* (CPU columns moved up to ~2x
between same-day runs), so only same-run columns are comparable.

See `docs/benchmark_plan.md` for goals and rules (fixed seeds, report machine
specs, do not tune the library to a benchmark).
