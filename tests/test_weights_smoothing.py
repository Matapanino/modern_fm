"""Estimator-level behavior of sample_weight, class_weight, label_smoothing."""

import numpy as np
import pytest
from modern_fm import FFMClassifier, FFMRegressor, FMClassifier


def _imbalanced(n=200, d=5, pos_frac=0.12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    score = X @ rng.normal(size=d)
    thr = np.quantile(score, 1.0 - pos_frac)  # make positives rare
    return X, (score > thr).astype(int)


def test_sample_weight_shape_validation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = (X[:, 0] > 0).astype(int)
    with pytest.raises(ValueError, match="sample_weight"):
        FMClassifier(random_state=0, max_iter=2).fit(X, y, sample_weight=np.ones(9))


def test_negative_sample_weight_rejected():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = (X[:, 0] > 0).astype(int)
    sw = np.ones(10)
    sw[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        FMClassifier(random_state=0, max_iter=2).fit(X, y, sample_weight=sw)


def test_sample_weight_changes_model():
    # SGD (not AdaGrad, which is ~invariant to a uniform gradient scale)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    sw = rng.uniform(0.1, 3.0, size=60)
    a = FMClassifier(optimizer="sgd", random_state=0, max_iter=20).fit(X, y)
    b = FMClassifier(optimizer="sgd", random_state=0, max_iter=20).fit(X, y, sample_weight=sw)
    assert not np.allclose(a.w_, b.w_)


def test_class_weight_balanced_changes_model():
    X, y = _imbalanced()
    a = FMClassifier(random_state=0, max_iter=20).fit(X, y)
    b = FMClassifier(random_state=0, max_iter=20, class_weight="balanced").fit(X, y)
    assert not np.allclose(a.w_, b.w_)


def test_class_weight_balanced_does_not_reduce_minority_recall():
    X, y = _imbalanced(seed=2)
    pos = y == 1
    base = FMClassifier(random_state=0, max_iter=40).fit(X, y)
    bal = FMClassifier(random_state=0, max_iter=40, class_weight="balanced").fit(X, y)
    assert (bal.predict(X)[pos] == 1).mean() >= (base.predict(X)[pos] == 1).mean()


def test_class_weight_dict_with_string_labels():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    y = np.where(X[:, 0] > 0, "pos", "neg")
    m = FMClassifier(random_state=0, max_iter=20, class_weight={"pos": 2.0, "neg": 1.0})
    m.fit(X, y)
    assert set(np.unique(m.predict(X))).issubset({"pos", "neg"})


def test_label_smoothing_shrinks_confidence():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    sharp = FMClassifier(random_state=0, max_iter=60, learning_rate=0.1).fit(X, y)
    smooth = FMClassifier(
        random_state=0, max_iter=60, learning_rate=0.1, label_smoothing=0.2
    ).fit(X, y)
    assert np.abs(smooth.decision_function(X)).mean() < np.abs(sharp.decision_function(X)).mean()


def test_label_smoothing_out_of_range_rejected():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = (X[:, 0] > 0).astype(int)
    with pytest.raises(ValueError, match="label_smoothing"):
        FMClassifier(random_state=0, max_iter=2, label_smoothing=1.0).fit(X, y)


def test_ffm_sample_weight_and_class_weight_run():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(50, 4))
    y = (X[:, 0] > 0).astype(int)
    fid = np.arange(4) % 2
    sw = rng.uniform(0.5, 2.0, size=50)
    m = FFMClassifier(random_state=0, max_iter=20, class_weight="balanced")
    m.fit(X, y, field_ids=fid, sample_weight=sw)
    proba = m.predict_proba(X)
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_ffm_regressor_sample_weight_runs():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(50, 4))
    y = X @ rng.normal(size=4)
    fid = np.arange(4) % 2
    sw = rng.uniform(0.5, 2.0, size=50)
    m = FFMRegressor(random_state=0, max_iter=20).fit(X, y, field_ids=fid, sample_weight=sw)
    pred = m.predict(X)
    assert pred.shape == (50,)
    assert np.all(np.isfinite(pred))
