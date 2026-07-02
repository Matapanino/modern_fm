"""Probability calibration (docs/roadmap.md v1.0 "Probability calibration").

The supported recipe is sklearn's `CalibratedClassifierCV` wrapping a
modern_fm classifier — the estimators are `check_estimator`-clean, so no
library code is involved. These tests pin (1) compatibility for every public
classifier and (2) that calibration actually improves ECE / Brier on
synthetically miscalibrated data (label smoothing compresses predicted
probabilities toward 0.5, which sigmoid recalibration undoes).
"""

import numpy as np
import pytest
from modern_fm import FFMClassifier, FMClassifier, FwFMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def _binary_data(seed, n=2400, d=20, density=0.4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[rng.random(X.shape) > density] = 0.0
    logits = X @ rng.normal(size=d)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    half = n // 2
    return (X[:half], y[:half]), (X[half:], y[half:])


def expected_calibration_error(y, p, bins=10):
    """Standard 10-bin ECE: |mean predicted - observed rate| weighted by bin mass."""
    ids = np.clip((p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            total += abs(p[m].mean() - y[m].mean()) * m.mean()
    return total


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_improves_ece_on_miscalibrated_fm(method):
    """label_smoothing=0.3 compresses probabilities toward 0.5 (systematically
    miscalibrated); CalibratedClassifierCV must improve held-out ECE and Brier."""
    (X_tr, y_tr), (X_te, y_te) = _binary_data(seed=0)
    base = FMClassifier(
        n_factors=4, max_iter=30, learning_rate=0.05, label_smoothing=0.3, random_state=0
    )
    p_raw = base.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    cal = CalibratedClassifierCV(base, method=method, cv=3).fit(X_tr, y_tr)
    p_cal = cal.predict_proba(X_te)[:, 1]
    ece_raw = expected_calibration_error(y_te, p_raw)
    ece_cal = expected_calibration_error(y_te, p_cal)
    assert ece_cal < ece_raw, f"ECE did not improve: {ece_cal:.4f} vs {ece_raw:.4f}"
    assert brier_score_loss(y_te, p_cal) < brier_score_loss(y_te, p_raw)


@pytest.mark.parametrize("cls", [FMClassifier, FFMClassifier, FwFMClassifier])
def test_calibrated_classifier_cv_compatible(cls):
    """Every public classifier works inside CalibratedClassifierCV (clone +
    cv-fold fitting + predict_proba)."""
    (X_tr, y_tr), (X_te, _) = _binary_data(seed=1, n=400)
    base = cls(n_factors=2, max_iter=5, random_state=0)
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=2).fit(X_tr, y_tr)
    proba = cal.predict_proba(X_te)
    assert proba.shape == (X_te.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-9)
    assert set(np.unique(cal.predict(X_te))) <= {0, 1}
