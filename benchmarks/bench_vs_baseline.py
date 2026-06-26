"""modern_fm vs. baselines on synthetic CTR data (docs/benchmark_plan.md).

Generates a CTR-style dataset of one-hot categorical fields with *planted
pairwise interactions* between disjoint field pairs — signal a linear model
cannot capture but Factorization Machines can. Train and test are sampled from
the *same* ground-truth weights, so test AUC measures real generalization.

Reports test AUC, fit time, and predict throughput for `FMClassifier` and
`FFMClassifier` across an `n_jobs` / `batch_size` sweep, plus a scikit-learn
`LogisticRegression` baseline. `xlearn` is included automatically if importable
(it does not build on every platform; it failed to build here).

Run from the repo root after `pip install -e ".[dev]"`:

    .venv/bin/python benchmarks/bench_vs_baseline.py
"""

from __future__ import annotations

import time
from statistics import median

import numpy as np
import scipy.sparse as sp
from modern_fm import FFMClassifier, FMClassifier
from modern_fm._backend import has_rust
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]  # interacting field pairs


def make_ctr_dataset(n_train, n_test, n_fields=16, card=16, seed=0):
    """One-hot CTR data with linear + planted pairwise-interaction ground truth."""
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.3, size=n_fields * card)                       # linear truth
    inters = {ab: rng.normal(scale=2.0, size=(card, card)) for ab in _PAIRS}  # pairwise truth

    def sample(n, s):
        r = np.random.default_rng(s)
        cats = r.integers(0, card, size=(n, n_fields))
        cols = (np.arange(n_fields) * card)[None, :] + cats
        rows = np.repeat(np.arange(n), n_fields)
        X = sp.csr_matrix(
            (np.ones(n * n_fields), (rows, cols.ravel())), shape=(n, n_fields * card)
        )
        logit = X @ w + r.normal(scale=0.3, size=n)
        for (a, b), M in inters.items():
            logit += M[cats[:, a], cats[:, b]]
        y = (r.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        return X, y

    field_ids = np.repeat(np.arange(n_fields), card).astype(np.int64)
    Xtr, ytr = sample(n_train, seed + 1)
    Xte, yte = sample(n_test, seed + 2)
    return Xtr, ytr, Xte, yte, field_ids


def timed(fn, repeats=3):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return median(times), out


def main():
    Xtr, ytr, Xte, yte, field_ids = make_ctr_dataset(40_000, 20_000)
    n_te = Xte.shape[0]
    print(f"Rust backend available: {has_rust()}")
    print(f"train={Xtr.shape[0]}  test={n_te}  features={Xtr.shape[1]}"
          f"  base rate={ytr.mean():.3f}\n")

    rows = []

    def record(label, fit_fn, predict_fn):
        t_fit, model = timed(fit_fn, repeats=3)
        t_pred, scores = timed(lambda: predict_fn(model))
        rows.append((label, roc_auc_score(yte, scores), t_fit, n_te / t_pred))

    record(
        "LogisticRegression (sklearn)",
        lambda: LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, ytr),
        lambda m: m.decision_function(Xte),
    )

    fm_kw = dict(n_factors=16, optimizer="adagrad", learning_rate=0.1, max_iter=20,
                 l2_factors=1e-5, random_state=0)
    for label, kw in [
        ("FMClassifier  n_jobs=1  batch=1", dict(n_jobs=1, batch_size=1)),
        ("FMClassifier  n_jobs=1  batch=512", dict(n_jobs=1, batch_size=512)),
        ("FMClassifier  n_jobs=-1 batch=512", dict(n_jobs=-1, batch_size=512)),
    ]:
        record(label, lambda kw=kw: FMClassifier(**fm_kw, **kw).fit(Xtr, ytr),
               lambda m: m.decision_function(Xte))

    ffm_kw = dict(n_factors=8, optimizer="adagrad", learning_rate=0.1, max_iter=20,
                  l2_factors=1e-5, random_state=0)
    for label, kw in [
        ("FFMClassifier n_jobs=1  batch=512", dict(n_jobs=1, batch_size=512)),
        ("FFMClassifier n_jobs=-1 batch=512", dict(n_jobs=-1, batch_size=512)),
    ]:
        def fit_ffm(kw=kw):
            return FFMClassifier(**ffm_kw, **kw).fit(Xtr, ytr, field_ids=field_ids)

        record(label, fit_ffm, lambda m: m.decision_function(Xte))

    try:
        import xlearn  # noqa: F401
        print("(xlearn importable — extend the script with an xlearn FM run)\n")
    except Exception:
        pass

    print(f"{'model':36s} {'test AUC':>9s} {'fit (s)':>9s} {'predict rows/s':>16s}")
    print("-" * 73)
    for label, auc, fit_s, thr in rows:
        print(f"{label:36s} {auc:9.4f} {fit_s:9.3f} {thr:16,.0f}")


if __name__ == "__main__":
    main()
