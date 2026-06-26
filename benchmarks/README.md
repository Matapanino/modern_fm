# benchmarks

- `bench_synthetic.py` — fit time and predict throughput on synthetic sparse
  data, plus the speedup of the Rust backend over the NumPy reference trainer.
- `bench_vs_baseline.py` — test AUC, fit time, and predict throughput vs a
  scikit-learn `LogisticRegression` baseline on synthetic CTR data with planted
  pairwise interactions, across an `n_jobs` / `batch_size` sweep (auto-includes
  `xlearn` if importable).

```bash
.venv/bin/python benchmarks/bench_synthetic.py
.venv/bin/python benchmarks/bench_vs_baseline.py
```

See `docs/benchmark_plan.md` for goals and rules (fixed seeds, report machine
specs, do not tune the library to a benchmark).
