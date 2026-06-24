"""Synthetic benchmarks for modern_fm (docs/benchmark_plan.md).

Reports fit time and predict throughput for the Rust backend, and the speedup
over the pure-NumPy reference trainer (the correctness floor). Fixed seeds;
median of a few repeats. Not tuned to any benchmark.

Run from the repo root after `pip install -e .`:

    .venv/bin/python benchmarks/bench_synthetic.py
"""

import time
from statistics import median

import numpy as np
from modern_fm import FMClassifier
from modern_fm._backend import has_rust
from modern_fm._reference_train import fm_train


def make_sparse_classification(n, d, density, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[rng.random(X.shape) > density] = 0.0
    y = (X @ rng.normal(size=d) > 0).astype(int)
    return X, y


def timed(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return median(times)


def bench_fit_vs_reference(n=4000, d=200, density=0.05, epochs=5, k=8):
    X, y = make_sparse_classification(n, d, density)
    rust = timed(
        lambda: FMClassifier(
            n_factors=k, max_iter=epochs, learning_rate=0.1, random_state=0
        ).fit(X, y)
    )
    ref = timed(
        lambda: fm_train(
            X, y.astype(float), epochs=epochs, n_factors=k, learning_rate=0.1, random_state=0
        ),
        repeats=1,
    )
    print(f"FM fit  n={n} d={d} density={density} epochs={epochs} k={k}")
    print(f"  rust backend : {rust * 1e3:8.1f} ms")
    print(f"  numpy reference: {ref * 1e3:8.1f} ms   (speedup x{ref / rust:.1f})")


def bench_predict_throughput(n=200_000, d=500, density=0.02, k=16):
    X, y = make_sparse_classification(n, d, density, seed=1)
    model = FMClassifier(n_factors=k, max_iter=3, learning_rate=0.1, random_state=0).fit(X, y)
    t = timed(lambda: model.decision_function(X))
    print(f"FM predict  n={n} d={d} density={density} k={k}")
    print(f"  {t * 1e3:8.1f} ms   ({n / t / 1e6:.1f}M rows/s)")


if __name__ == "__main__":
    print(f"Rust backend available: {has_rust()}\n")
    bench_fit_vs_reference()
    print()
    bench_predict_throughput()
