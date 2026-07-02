"""Real-data CTR benchmark (docs/roadmap.md v1.0 "Real-data benchmark").

Dataset: the KDD Cup 2012 track-2 click-through sample published on OpenML as
``Click_prediction_small`` (1.5M impressions, 9 integer-categorical columns —
ad/advertiser/keyword/title/description/user ids + depth/position/impression
— binary click label, ~4.5% CTR). It downloads without credentials via
`sklearn.datasets.fetch_openml` and is cached under ``~/scikit_learn_data``.
The original Criteo/Avazu samples are no longer publicly downloadable without
credentials (checked 2026-07: labs.criteo.com redirects to a blog, the S3
mirror 404s, the Hugging Face copy is gated), so this real CTR dataset is the
zero-credential stand-in.

Protocol (benchmark_plan rules: fixed seeds, machine specs, hyperparameters
chosen once and NOT tuned to this benchmark): subsample ``--rows`` rows with
a fixed seed, stratified 80/20 split, one-hot encode every column with
`CategoricalEncoder` (fields = source columns), report held-out AUC, fit
time and predict throughput.

    .venv/bin/python benchmarks/bench_criteo_like.py [--rows 200000] [--quick]

macOS note: if `fetch_openml` fails with CERTIFICATE_VERIFY_FAILED, point
SSL_CERT_FILE at a CA bundle (or run "Install Certificates.command").
"""

import argparse
import platform
import time

import numpy as np
from modern_fm import CategoricalEncoder, FFMClassifier, FMClassifier, FwFMClassifier
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 0


def load_click_data(rows):
    data = fetch_openml("Click_prediction_small", version=1, as_frame=True, parser="auto")
    X = data.data.to_numpy(dtype=np.int64)
    y = data.target.to_numpy().astype(np.int64)
    if rows < len(y):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(y), size=rows, replace=False)
        X, y = X[idx], y[idx]
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def bench_model(name, model, X_tr, y_tr, X_te, y_te):
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_te)[:, 1]
    else:
        scores = model.decision_function(X_te)
    pred_s = time.perf_counter() - t0
    auc = roc_auc_score(y_te, scores)
    krps = X_te.shape[0] / pred_s / 1e3
    print(f"{name:>18} {auc:>8.4f} {fit_s:>9.1f} {krps:>12.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--quick", action="store_true", help="20k rows, 3 epochs")
    args = ap.parse_args()
    rows, epochs = (20_000, 3) if args.quick else (args.rows, 20)

    print(f"machine: {platform.platform()} / python {platform.python_version()}")
    Xc_tr, Xc_te, y_tr, y_te = load_click_data(rows)
    enc = CategoricalEncoder().fit(Xc_tr)
    X_tr, X_te = enc.transform(Xc_tr), enc.transform(Xc_te)
    print(
        f"rows: {rows} (train {X_tr.shape[0]} / test {X_te.shape[0]}), "
        f"one-hot features: {enc.n_features_out_}, fields: {enc.n_fields_}, "
        f"CTR: {y_tr.mean():.4f}"
    )
    # Hyperparameters fixed up front and not tuned to this benchmark:
    # libFM-style L2 (1e-4 — one-hot CTR data is dominated by rare ids, so
    # near-zero regularization degenerates factor models), AdaGrad per-row
    # updates, and the built-in early stopping (train-internal validation
    # split — no test leakage) instead of a hand-picked epoch count. k is
    # halved for FFM's per-field factors.
    fm_kw = dict(
        optimizer="adagrad", learning_rate=0.05, l2_linear=1e-4, l2_factors=1e-4,
        max_iter=epochs, batch_size=1, random_state=SEED,
        early_stopping=True, patience=3,
    )
    print(f"{'model':>18} {'AUC':>8} {'fit s':>9} {'pred krow/s':>12}")
    bench_model(
        "LogisticRegression",
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED),
        X_tr, y_tr, X_te, y_te,
    )
    bench_model("FMClassifier", FMClassifier(n_factors=8, **fm_kw), X_tr, y_tr, X_te, y_te)
    ffm = FFMClassifier(n_factors=4, **fm_kw)
    bench_model(
        "FFMClassifier",
        _WithFields(ffm, enc.field_ids_),
        X_tr, y_tr, X_te, y_te,
    )
    fwfm = FwFMClassifier(n_factors=8, **fm_kw)
    bench_model("FwFMClassifier", _WithFields(fwfm, enc.field_ids_), X_tr, y_tr, X_te, y_te)


class _WithFields:
    """Bind field_ids into fit so bench_model can treat all models alike."""

    def __init__(self, model, field_ids):
        self.model = model
        self.field_ids = field_ids

    def fit(self, X, y):
        self.model.fit(X, y, field_ids=self.field_ids)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


if __name__ == "__main__":
    main()
