# benchmarks

- `bench_synthetic.py` — fit time and predict throughput on synthetic sparse
  data, plus the speedup of the Rust backend over the NumPy reference trainer.

```bash
.venv/bin/python benchmarks/bench_synthetic.py
```

See `docs/benchmark_plan.md` for goals and rules (fixed seeds, report machine
specs, do not tune the library to a benchmark).
