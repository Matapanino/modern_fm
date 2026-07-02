"""Calibrated CTR probabilities with CalibratedClassifierCV (docs/roadmap.md).

modern_fm classifiers are scikit-learn compatible, so probability calibration
is the standard sklearn recipe — no library-specific API. This example trains
a deliberately miscalibrated FM (label smoothing compresses probabilities
toward 0.5), wraps it in `CalibratedClassifierCV`, and reports held-out ECE /
Brier / log-loss plus a small reliability table.

    .venv/bin/python examples/calibration.py
"""

import numpy as np
from modern_fm import FMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss


def expected_calibration_error(y, p, bins=10):
    ids = np.clip((p * bins).astype(int), 0, bins - 1)
    return sum(
        abs(p[m].mean() - y[m].mean()) * m.mean()
        for b in range(bins)
        if (m := ids == b).any()
    )


def main():
    rng = np.random.default_rng(0)
    n, d = 6000, 30
    X = rng.normal(size=(n, d))
    X[rng.random(X.shape) > 0.4] = 0.0
    logits = X @ rng.normal(size=d)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    X_tr, y_tr, X_te, y_te = X[: n // 2], y[: n // 2], X[n // 2 :], y[n // 2 :]

    base = FMClassifier(
        n_factors=8, max_iter=30, learning_rate=0.05, label_smoothing=0.3, random_state=0
    )
    p_raw = base.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(X_tr, y_tr)
    p_cal = cal.predict_proba(X_te)[:, 1]

    print(f"{'':>10} {'ECE':>8} {'Brier':>8} {'log-loss':>9}")
    for name, p in (("raw", p_raw), ("calibrated", p_cal)):
        print(
            f"{name:>10} {expected_calibration_error(y_te, p):>8.4f} "
            f"{brier_score_loss(y_te, p):>8.4f} {log_loss(y_te, p):>9.4f}"
        )

    def _bin_cell(p, b):
        m = np.clip((p * 10).astype(int), 0, 9) == b
        if not m.any():
            return f"{'-':>9} {'-':>8}"
        return f"{p[m].mean():>9.3f} {y_te[m].mean():>8.3f}"

    print("\nreliability (10 bins): mean predicted vs observed positive rate")
    print(f"{'bin':>5} {'raw pred':>9} {'raw obs':>8} {'cal pred':>9} {'cal obs':>8}")
    for b in range(10):
        print(f"{b:>5} {_bin_cell(p_raw, b)} {_bin_cell(p_cal, b)}")


if __name__ == "__main__":
    main()
