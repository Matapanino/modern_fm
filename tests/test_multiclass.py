"""Multiclass (softmax) FMClassifier (docs/api_design.md, docs/math_spec.md).

Training uses the NumPy reference path (v0.1 has no Rust multiclass kernel);
prediction reuses the Rust per-class FM predict kernel. The reference is the
ground truth, so these are behavior tests, not parity tests.
"""

import numpy as np
from modern_fm import FMClassifier
from modern_fm.losses import softmax_loss


def _three_class(n=150, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.argmax(X @ rng.normal(size=(d, 3)), axis=1)
    return X, y


def _clf(**kw):
    params = dict(n_factors=4, random_state=0, max_iter=40, learning_rate=0.1)
    params.update(kw)
    return FMClassifier(**params)


def test_multiclass_shapes():
    X, y = _three_class()
    m = _clf().fit(X, y)
    assert m.V_.shape == (3, X.shape[1], 4)
    assert m.w_.shape == (3, X.shape[1])
    assert np.asarray(m.w0_).shape == (3,)
    np.testing.assert_array_equal(m.classes_, [0, 1, 2])
    assert m.decision_function(X).shape == (len(y), 3)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all((proba >= 0) & (proba <= 1))


def test_multiclass_learns():
    X, y = _three_class()
    m = _clf(max_iter=80).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.7
    assert softmax_loss(y, m.decision_function(X)) < 0.8 * np.log(3)


def test_predict_is_argmax_of_decision_function():
    X, y = _three_class()
    m = _clf().fit(X, y)
    np.testing.assert_array_equal(
        m.predict(X), m.classes_[np.argmax(m.decision_function(X), axis=1)]
    )


def test_multiclass_string_labels():
    X, yi = _three_class(n=90)
    y = np.array(["a", "b", "c"])[yi]
    m = _clf().fit(X, y)
    np.testing.assert_array_equal(m.classes_, ["a", "b", "c"])
    assert set(np.unique(m.predict(X))).issubset({"a", "b", "c"})


def test_multiclass_reproducible():
    X, y = _three_class()
    a = _clf(random_state=1).fit(X, y)
    b = _clf(random_state=1).fit(X, y)
    np.testing.assert_array_equal(a.V_, b.V_)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_multiclass_save_load_roundtrip(tmp_path):
    X, y = _three_class(n=80)
    m = _clf().fit(X, y)
    path = str(tmp_path / "mc.bin")
    m.save_model(path)
    loaded = FMClassifier.load_model(path)
    np.testing.assert_array_equal(loaded.predict_proba(X), m.predict_proba(X))


def test_multiclass_label_smoothing_runs():
    X, y = _three_class(n=90)
    m = _clf(label_smoothing=0.1).fit(X, y)
    np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1.0, atol=1e-12)


def test_two_class_softmax_path():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    m = _clf(loss="softmax").fit(X, y)  # softmax even with 2 classes
    proba = m.predict_proba(X)
    assert proba.shape == (60, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
