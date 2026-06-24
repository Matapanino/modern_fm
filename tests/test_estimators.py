"""End-to-end tests for the public estimator API (fit -> predict).

Complements test_rust_train_parity.py (backend functions) and
test_training_reference.py (reference trainers) by driving the estimator
classes themselves: sklearn-style contracts, output shapes, label round-trip
via classes_, reproducibility, dtype, and that training actually moves
predictions toward the labels. Uses whichever backend is built (Rust or the
NumPy fallback); their parity is covered elsewhere.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from modern_fm import FFMClassifier, FMClassifier, FMRegressor, NotFittedError
from modern_fm.losses import logistic_loss, squared_loss

ALL = [FMClassifier, FMRegressor, FFMClassifier]
CLASSIFIERS = [FMClassifier, FFMClassifier]
LOG2 = np.log(2.0)


def _separable_binary(seed=0, n=80, d=6, labels=(0, 1)):
    """Linearly separable-ish binary data; labels controls the class values."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    pos = X @ rng.normal(size=d) > 0
    return X, np.where(pos, labels[1], labels[0])


def _field_ids(d):
    return np.arange(d) % 3  # a handful of fields


def _make(cls, **kw):
    params = dict(random_state=0, max_iter=40, learning_rate=0.1)
    params.update(kw)
    return cls(**params)


def _fit(model, X, y):
    if isinstance(model, FFMClassifier):
        return model.fit(X, y, field_ids=_field_ids(X.shape[1]))
    return model.fit(X, y)


@pytest.mark.parametrize("cls", ALL)
def test_fit_returns_self(cls):
    X, y = _separable_binary()
    model = _make(cls)
    assert _fit(model, X, y) is model


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_classifier_shapes_and_proba(cls):
    X, y = _separable_binary()
    model = _fit(_make(cls), X, y)
    n = X.shape[0]
    assert model.predict(X).shape == (n,)
    assert model.decision_function(X).shape == (n,)
    proba = model.predict_proba(X)
    assert proba.shape == (n, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_predict_consistent_with_decision_function(cls):
    X, y = _separable_binary()
    model = _fit(_make(cls), X, y)
    expected = model.classes_[(model.decision_function(X) >= 0).astype(int)]
    np.testing.assert_array_equal(model.predict(X), expected)


@pytest.mark.parametrize("cls", CLASSIFIERS)
@pytest.mark.parametrize("labels", [(0, 1), (-1, 1), ("neg", "pos")])
def test_label_roundtrip_via_classes(cls, labels):
    X, y = _separable_binary(labels=labels)
    model = _fit(_make(cls), X, y)
    np.testing.assert_array_equal(model.classes_, np.unique(y))
    assert set(np.unique(model.predict(X))).issubset(set(labels))


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_classifier_learns(cls):
    X, y = _separable_binary(n=120)
    model = _fit(_make(cls), X, y)
    y01 = (y == model.classes_[1]).astype(float)
    assert logistic_loss(y01, model.decision_function(X)) < 0.7 * LOG2
    assert (model.predict(X) == y).mean() > 0.75


def test_regressor_learns_and_predict_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    y = X @ rng.normal(size=5)
    model = FMRegressor(random_state=0, max_iter=60, learning_rate=0.05).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (100,)
    baseline = squared_loss(y, np.full_like(y, y.mean()))
    assert squared_loss(y, pred) < 0.5 * baseline


@pytest.mark.parametrize("cls", ALL)
def test_not_fitted_raises(cls):
    X = np.zeros((3, 4))
    with pytest.raises(NotFittedError):
        cls().predict(X)
    if cls is not FMRegressor:
        with pytest.raises(NotFittedError):
            cls().predict_proba(X)
        with pytest.raises(NotFittedError):
            cls().decision_function(X)


@pytest.mark.parametrize("cls", ALL)
def test_learned_attributes(cls):
    X, y = _separable_binary(d=6)
    model = _fit(_make(cls), X, y)
    assert isinstance(model.w0_, float)
    assert model.w_.shape == (6,)
    assert model.n_features_in_ == 6
    assert model.n_iter_ == 40
    if cls is FFMClassifier:
        assert model.n_fields_ == 3
        assert model.V_.shape == (6, model.n_fields_, model.n_factors)
        np.testing.assert_array_equal(model.field_ids_, _field_ids(6))
    else:
        assert model.V_.shape == (6, model.n_factors)
    if cls is not FMRegressor:
        np.testing.assert_array_equal(model.classes_, np.unique(y))


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("dtype,np_dtype", [("float32", np.float32), ("float64", np.float64)])
def test_dtype_of_learned_arrays(cls, dtype, np_dtype):
    X, y = _separable_binary(d=5)
    model = _fit(_make(cls, dtype=dtype), X, y)
    assert model.w_.dtype == np_dtype
    assert model.V_.dtype == np_dtype


@pytest.mark.parametrize("cls", ALL)
def test_same_seed_identical(cls):
    X, y = _separable_binary()
    m1 = _fit(_make(cls, random_state=42), X, y)
    m2 = _fit(_make(cls, random_state=42), X, y)
    assert m1.w0_ == m2.w0_
    np.testing.assert_array_equal(m1.w_, m2.w_)
    np.testing.assert_array_equal(m1.V_, m2.V_)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


@pytest.mark.parametrize("cls", ALL)
def test_different_seed_differs(cls):
    X, y = _separable_binary()
    m1 = _fit(_make(cls, random_state=0), X, y)
    m2 = _fit(_make(cls, random_state=1), X, y)
    assert not np.array_equal(m1.V_, m2.V_)


@pytest.mark.parametrize("cls", ALL)
def test_predict_feature_mismatch_raises(cls):
    X, y = _separable_binary(d=5)
    model = _fit(_make(cls), X, y)
    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 6)))


@pytest.mark.parametrize("cls", ALL)
def test_dense_and_csr_training_equivalent(cls):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 8))
    X[rng.random(X.shape) > 0.4] = 0.0  # real sparsity so CSR path is exercised
    y = (X @ rng.normal(size=8) > 0).astype(int)
    dense = _fit(_make(cls, dtype="float64"), X, y)
    sparse = _fit(_make(cls, dtype="float64"), sp.csr_matrix(X), y)
    assert dense.w0_ == pytest.approx(sparse.w0_, rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(dense.w_, sparse.w_, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(dense.V_, sparse.V_, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_adam_classifier_learns(cls):
    X, y = _separable_binary(n=120)
    model = _fit(_make(cls, optimizer="adam"), X, y)
    assert (model.predict(X) == y).mean() > 0.75


def test_adam_regressor_learns():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    y = X @ rng.normal(size=5)
    model = FMRegressor(optimizer="adam", random_state=0, max_iter=60, learning_rate=0.05).fit(X, y)
    baseline = squared_loss(y, np.full_like(y, y.mean()))
    assert squared_loss(y, model.predict(X)) < 0.5 * baseline


def test_adam_multiclass_learns():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 6))
    y = (X[:, :3] @ rng.normal(size=(3, 3))).argmax(axis=1)  # 3-class, learnable
    model = FMClassifier(
        optimizer="adam", random_state=0, max_iter=60, learning_rate=0.05
    ).fit(X, y)
    assert model.V_.shape[0] == 3  # one parameter set per class
    np.testing.assert_array_equal(model.classes_, np.array([0, 1, 2]))
    assert (model.predict(X) == y).mean() > 0.6


@pytest.mark.parametrize("cls", ALL)
def test_adam_early_stopping_not_implemented(cls):
    X, y = _separable_binary(n=40)
    with pytest.raises(NotImplementedError):
        _fit(_make(cls, optimizer="adam", early_stopping=True), X, y)
