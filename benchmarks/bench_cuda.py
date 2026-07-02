"""CUDA vs Rust-CPU FM/FFM prediction + binary/multiclass training benchmark
(docs/gpu_backend_plan.md).

Transfer-INCLUSIVE: every CUDA prediction call copies the CSR arrays +
parameters to the device and the scores back; a CUDA training call uploads the
CSR/targets/row-order/parameters once (parameters stay device-resident), then
per mini-batch moves either compact touched-coordinate gradient buffers +
touched-parameter scatter-backs (small batches) or full dense buffers (large
batches) — the optimizer flush stays on the CPU. That is exactly what
`backend="cuda"` pays today. The CUDA context and NVRTC-compiled
module are process-cached, so only the very first CUDA call pays
initialization — measured separately below. Run on a CUDA machine per
docs/cuda_validation_runbook.md; prints machine info to paste into PRs.

    .venv/bin/python benchmarks/bench_cuda.py
"""

import platform
import sys
import time
from statistics import median

import numpy as np
import scipy.sparse as sp
from modern_fm import _backend


def make_csr(n_rows, n_features, avg_nnz, seed=0):
    rng = np.random.default_rng(seed)
    nnz = n_rows * avg_nnz
    rows = np.repeat(np.arange(n_rows), avg_nnz)
    cols = rng.integers(0, n_features, size=nnz)
    X = sp.csr_matrix((rng.normal(size=nnz), (rows, cols)), shape=(n_rows, n_features))
    X.sum_duplicates()
    return X


def timed(fn, repeats=5):
    fn()  # warmup (first-touch; context/module already cached process-wide)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return median(times)


def cold_start():
    """Time the first CUDA call of the process (context + NVRTC compile of all
    kernels + a tiny FM predict); every later call reuses the cache."""
    X = sp.csr_matrix(np.eye(2))
    w, V = np.zeros(2), np.ones((2, 2))
    t0 = time.perf_counter()
    _backend.fm_predict_fast(X, 0.0, w, V, backend="cuda")
    return time.perf_counter() - t0


def bench_fm(rng, quick):
    rows_grid = (100_000,) if quick else (10_000, 100_000, 1_000_000)
    nnz_grid = (8, 32) if quick else (8, 32, 128)
    k_grid = (8, 32) if quick else (8, 16, 32, 64)
    print("FM prediction (Rust CPU vs CUDA, transfer-inclusive)")
    print(f"{'rows':>10} {'nnz/row':>8} {'k':>4} {'cpu ms':>10} {'cuda ms':>10} {'speedup':>8}")
    for n_rows in rows_grid:
        for avg_nnz in nnz_grid:
            for k in k_grid:
                d = 100_000
                X = make_csr(n_rows, d, avg_nnz)
                w0, w, V = 0.1, rng.normal(size=d), rng.normal(size=(d, k))
                cpu = timed(lambda: _backend.fm_predict_fast(X, w0, w, V))
                cuda = timed(lambda: _backend.fm_predict_fast(X, w0, w, V, backend="cuda"))
                print(
                    f"{n_rows:>10} {avg_nnz:>8} {k:>4} {cpu * 1e3:>10.1f} "
                    f"{cuda * 1e3:>10.1f} {cpu / cuda:>7.1f}x"
                )


def bench_ffm(rng, quick):
    n_rows, d = 100_000, 100_000
    nnz_grid = (8, 32) if quick else (8, 16, 32)
    fields_grid = (8, 32) if quick else (8, 16, 32)
    k_grid = (8,) if quick else (4, 8, 16)
    print()
    print("FFM prediction (Rust CPU vs CUDA, transfer-inclusive; CPU FFM is serial)")
    print(
        f"{'rows':>10} {'nnz/row':>8} {'fields':>7} {'k':>4} "
        f"{'V MB':>7} {'cpu ms':>10} {'cuda ms':>10} {'speedup':>8}"
    )
    for avg_nnz in nnz_grid:
        X = make_csr(n_rows, d, avg_nnz)
        for n_fields in fields_grid:
            field_ids = rng.integers(0, n_fields, size=d)
            for k in k_grid:
                w0, w = 0.1, rng.normal(size=d)
                V = rng.normal(size=(d, n_fields, k))
                cpu = timed(lambda: _backend.ffm_predict(X, field_ids, w0, w, V))
                cuda = timed(
                    lambda: _backend.ffm_predict(X, field_ids, w0, w, V, backend="cuda")
                )
                print(
                    f"{n_rows:>10} {avg_nnz:>8} {n_fields:>7} {k:>4} "
                    f"{V.nbytes / 1e6:>7.0f} {cpu * 1e3:>10.1f} "
                    f"{cuda * 1e3:>10.1f} {cpu / cuda:>7.1f}x"
                )


def bench_fm_train(rng, quick):
    """One binary-logistic epoch at the plan-doc batch sizes. Small batches
    ride the compact touched-coordinate transfer path (parameters
    device-resident, touched-only scatter-back); large batches use full dense
    gradient buffers."""
    n_rows, d, avg_nnz, k = 100_000, 100_000, 32, 8
    X = make_csr(n_rows, d, avg_nnz)
    y = (rng.random(n_rows) > 0.5).astype(np.float64)
    params = (0.0, rng.normal(size=d) * 0.01, rng.normal(size=(d, k)) * 0.01)
    row_orders = rng.permutation(n_rows).astype(np.int64)[None, :]
    kwargs = dict(
        loss="logistic", optimizer="adagrad", learning_rate=0.05,
        l2_linear=1e-6, l2_factors=1e-6, row_orders=row_orders,
    )
    batch_grid = (1024, n_rows) if quick else (256, 1024, 8192, n_rows)
    print()
    print(
        "FM training, 1 epoch (Rust CPU n_jobs=1 vs CUDA accumulation + CPU flush; "
        f"rows={n_rows}, nnz/row={avg_nnz}, k={k})"
    )
    print(f"{'batch':>10} {'cpu ms':>10} {'cuda ms':>10} {'speedup':>8}")
    for bs in batch_grid:
        cpu = timed(lambda: _backend.fm_fit(X, y, params, batch_size=bs, **kwargs), repeats=3)
        cuda = timed(
            lambda: _backend.fm_fit(X, y, params, batch_size=bs, backend="cuda", **kwargs),
            repeats=3,
        )
        label = "full" if bs == n_rows else bs
        print(f"{label:>10} {cpu * 1e3:>10.1f} {cuda * 1e3:>10.1f} {cpu / cuda:>7.1f}x")


def bench_ffm_train(rng, quick):
    """One binary-logistic FFM epoch. The CUDA path keeps V device-resident
    and moves only touched (feature, field) slots per batch; the host pays the
    touched-slot enumeration (the pair loop without the k-dot). CPU baseline
    is n_jobs=1 serial."""
    n_rows, d, avg_nnz, n_fields, k = 100_000, 100_000, 32, 8, 4
    X = make_csr(n_rows, d, avg_nnz)
    y = (rng.random(n_rows) > 0.5).astype(np.float64)
    field_ids = rng.integers(0, n_fields, size=d)
    params = (0.0, rng.normal(size=d) * 0.01, rng.normal(size=(d, n_fields, k)) * 0.01)
    row_orders = rng.permutation(n_rows).astype(np.int64)[None, :]
    kwargs = dict(
        loss="logistic", optimizer="adagrad", learning_rate=0.05,
        l2_linear=1e-6, l2_factors=1e-6, row_orders=row_orders,
    )
    batch_grid = (1024, n_rows) if quick else (256, 1024, 8192, n_rows)
    print()
    print(
        "FFM training, 1 epoch (Rust CPU n_jobs=1 vs CUDA accumulation + CPU flush; "
        f"rows={n_rows}, nnz/row={avg_nnz}, fields={n_fields}, k={k})"
    )
    print(f"{'batch':>10} {'cpu ms':>10} {'cuda ms':>10} {'speedup':>8}")
    for bs in batch_grid:
        cpu = timed(
            lambda: _backend.ffm_fit(X, y, field_ids, params, batch_size=bs, **kwargs),
            repeats=3,
        )
        cuda = timed(
            lambda: _backend.ffm_fit(
                X, y, field_ids, params, batch_size=bs, backend="cuda", **kwargs
            ),
            repeats=3,
        )
        label = "full" if bs == n_rows else bs
        print(f"{label:>10} {cpu * 1e3:>10.1f} {cuda * 1e3:>10.1f} {cpu / cuda:>7.1f}x")


def bench_fm_mc_train(rng, quick):
    """One softmax multiclass FM epoch (gpu_backend_plan milestone 5): all C
    class gradients accumulate in one kernel launch per batch (C-stacked
    compact buffers), the per-class flush stays on the CPU. CPU multiclass is
    serial, so this is the cell where the GPU replaces the most CPU work."""
    n_rows, d, avg_nnz, k = 100_000, 100_000, 32, 8
    X = make_csr(n_rows, d, avg_nnz)
    row_orders = rng.permutation(n_rows).astype(np.int64)[None, :]
    kwargs = dict(
        optimizer="adagrad", learning_rate=0.05,
        l2_linear=1e-6, l2_factors=1e-6, row_orders=row_orders,
    )
    c_grid = (3,) if quick else (3, 10)
    batch_grid = (1024, n_rows) if quick else (1024, 8192, n_rows)
    print()
    print(
        "FM multiclass training, 1 epoch (Rust CPU serial vs CUDA accumulation + "
        f"CPU per-class flush; rows={n_rows}, nnz/row={avg_nnz}, k={k})"
    )
    print(f"{'classes':>8} {'batch':>10} {'cpu ms':>10} {'cuda ms':>10} {'speedup':>8}")
    for n_classes in c_grid:
        y = rng.integers(0, n_classes, size=n_rows)
        params = (
            np.zeros(n_classes),
            rng.normal(size=(n_classes, d)) * 0.01,
            rng.normal(size=(n_classes, d, k)) * 0.01,
        )
        for bs in batch_grid:
            cpu = timed(
                lambda: _backend.fm_fit_multiclass(X, y, params, batch_size=bs, **kwargs),
                repeats=3,
            )
            cuda = timed(
                lambda: _backend.fm_fit_multiclass(
                    X, y, params, batch_size=bs, backend="cuda", **kwargs
                ),
                repeats=3,
            )
            label = "full" if bs == n_rows else bs
            print(
                f"{n_classes:>8} {label:>10} {cpu * 1e3:>10.1f} "
                f"{cuda * 1e3:>10.1f} {cpu / cuda:>7.1f}x"
            )


def bench_ffm_mc_train(rng, quick):
    """One softmax multiclass FFM epoch: the score kernel does all C logits +
    softmax in one launch, then per class a pair-accumulation kernel + gather
    reuse one class-sized dense gv buffer (VRAM stays 1x V per class)."""
    n_rows, d, avg_nnz, n_fields, k = 100_000, 100_000, 32, 8, 4
    X = make_csr(n_rows, d, avg_nnz)
    field_ids = rng.integers(0, n_fields, size=d)
    row_orders = rng.permutation(n_rows).astype(np.int64)[None, :]
    kwargs = dict(
        optimizer="adagrad", learning_rate=0.05,
        l2_linear=1e-6, l2_factors=1e-6, row_orders=row_orders,
    )
    c_grid = (3,) if quick else (3, 10)
    batch_grid = (1024, n_rows) if quick else (1024, 8192, n_rows)
    print()
    print(
        "FFM multiclass training, 1 epoch (Rust CPU serial vs CUDA two-kernel "
        f"accumulation + CPU per-class flush; rows={n_rows}, nnz/row={avg_nnz}, "
        f"fields={n_fields}, k={k})"
    )
    print(f"{'classes':>8} {'batch':>10} {'cpu ms':>10} {'cuda ms':>10} {'speedup':>8}")
    for n_classes in c_grid:
        y = rng.integers(0, n_classes, size=n_rows)
        params = (
            np.zeros(n_classes),
            rng.normal(size=(n_classes, d)) * 0.01,
            rng.normal(size=(n_classes, d, n_fields, k)) * 0.01,
        )
        for bs in batch_grid:
            cpu = timed(
                lambda: _backend.ffm_fit_multiclass(
                    X, y, field_ids, params, batch_size=bs, **kwargs
                ),
                repeats=3,
            )
            cuda = timed(
                lambda: _backend.ffm_fit_multiclass(
                    X, y, field_ids, params, batch_size=bs, backend="cuda", **kwargs
                ),
                repeats=3,
            )
            label = "full" if bs == n_rows else bs
            print(
                f"{n_classes:>8} {label:>10} {cpu * 1e3:>10.1f} "
                f"{cuda * 1e3:>10.1f} {cpu / cuda:>7.1f}x"
            )


def main():
    quick = "--quick" in sys.argv  # trimmed grid for short validation runs
    print(f"machine: {platform.platform()} / python {platform.python_version()}")
    print(f"has_rust={_backend.has_rust()} has_cuda={_backend.has_cuda()}")
    if not _backend.has_cuda():
        print("no CUDA device/build — nothing to benchmark")
        return
    print(
        f"first CUDA call (context + NVRTC compile, cached afterwards): "
        f"{cold_start() * 1e3:.1f} ms"
    )
    rng = np.random.default_rng(0)
    bench_fm(rng, quick)
    bench_ffm(rng, quick)
    bench_fm_train(rng, quick)
    bench_ffm_train(rng, quick)
    bench_fm_mc_train(rng, quick)
    bench_ffm_mc_train(rng, quick)


if __name__ == "__main__":
    main()
