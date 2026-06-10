# Benchmark Plan

Benchmarks start in Phase 2 (no fast path exists before the Rust backend).

## Goals

- Measure fit speed, predict throughput (rows/sec), peak memory
- Compare dense vs CSR paths
- Compare against the pure-Python reference (sanity floor)
- Optional: compare against libffm / xLearn when installed

## Datasets

1. synthetic dense classification (small/medium)
2. synthetic sparse classification (one-hot, ~1e5–1e6 features)
3. synthetic field-aware sparse classification (FFM)
4. small CTR-like dataset (Criteo-like sample, not full Criteo)
5. Kaggle-style tabular encoded dataset

## Metrics

- wall-clock training time, prediction throughput
- logloss, AUC, balanced accuracy (multiclass)
- peak memory

## Rules

- fixed seeds, report machine specs, run 3x and report median
- do not tune the library to a benchmark (CLAUDE.md rule)

Scripts live in `benchmarks/`: `bench_synthetic.py`, `bench_criteo_like.py`,
`bench_against_libffm.py`.
