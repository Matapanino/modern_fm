"""Early stopping and eval_set (docs/requirements.md).

Covers FMClassifier (binary), FMRegressor, and FFMClassifier; multiclass +
early stopping is not supported in v0.1.
"""

import numpy as np
import pytest
from modern_fm import FFMClassifier, FMClassifier, FMRegressor


def _data(n=200, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X @ rng.normal(size=d) > 0).astype(int)
    return X, y


def _holdout(X, y, n_train=150):
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def test_early_stopping_internal_split_runs():
    X, y = _data()
    m = FMClassifier(random_state=0, max_iter=50, early_stopping=True, patience=5).fit(X, y)
    assert 1 <= m.n_iter_ <= 50
    assert m.predict_proba(X).shape == (len(y), 2)


def test_early_stopping_stops_before_max_iter():
    X, y = _data(n=120)
    m = FMClassifier(
        random_state=0, max_iter=200, early_stopping=True, patience=3, learning_rate=0.2
    ).fit(X, y)
    assert m.n_iter_ < 200


def test_large_patience_runs_all_epochs():
    X, y = _data()
    m = FMClassifier(random_state=0, max_iter=20, early_stopping=True, patience=10**6).fit(X, y)
    assert m.n_iter_ == 20


def test_eval_set_tuple_and_list_equivalent():
    X, y = _data()
    Xtr, ytr, Xv, yv = _holdout(X, y)
    m1 = FMClassifier(random_state=0, max_iter=30, patience=5).fit(Xtr, ytr, eval_set=(Xv, yv))
    m2 = FMClassifier(random_state=0, max_iter=30, patience=5).fit(Xtr, ytr, eval_set=[(Xv, yv)])
    assert m1.n_iter_ == m2.n_iter_
    np.testing.assert_array_equal(m1.predict(Xv), m2.predict(Xv))


def test_regressor_early_stopping():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 5))
    y = X @ rng.normal(size=5)
    m = FMRegressor(
        random_state=0, max_iter=60, early_stopping=True, patience=5, learning_rate=0.05
    ).fit(X, y)
    assert 1 <= m.n_iter_ <= 60
    assert m.predict(X).shape == (150,)


def test_ffm_early_stopping_and_eval_set():
    X, y = _data(d=6)
    fid = np.arange(6) % 3
    m = FFMClassifier(random_state=0, max_iter=40, early_stopping=True, patience=5)
    m.fit(X, y, field_ids=fid)
    assert 1 <= m.n_iter_ <= 40
    np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1.0, atol=1e-12)

    Xtr, ytr, Xv, yv = _holdout(X, y)
    m2 = FFMClassifier(random_state=0, max_iter=30, patience=5)
    m2.fit(Xtr, ytr, field_ids=fid, eval_set=(Xv, yv))
    assert m2.n_iter_ <= 30


def test_early_stopping_reproducible():
    X, y = _data()
    a = FMClassifier(random_state=3, max_iter=40, early_stopping=True, patience=4).fit(X, y)
    b = FMClassifier(random_state=3, max_iter=40, early_stopping=True, patience=4).fit(X, y)
    assert a.n_iter_ == b.n_iter_
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_multiclass_early_stopping_works(optimizer):
    # multiclass + early stopping rounds optimizer state via the reference path.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    y = (X[:, :3] @ rng.normal(size=(3, 3))).argmax(axis=1)  # learnable 3-class
    model = FMClassifier(
        optimizer=optimizer, random_state=0, max_iter=50, patience=6,
        learning_rate=0.05, early_stopping=True,
    ).fit(X, y)
    assert 1 <= model.n_iter_ <= 50
    assert model.V_.shape[0] == 3  # one parameter set per class
    assert model.predict(X).shape == (X.shape[0],)


def test_multiclass_early_stopping_reproducible():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    y = (X[:, :3] @ rng.normal(size=(3, 3))).argmax(axis=1)
    a = FMClassifier(random_state=0, max_iter=40, early_stopping=True, patience=5).fit(X, y)
    b = FMClassifier(random_state=0, max_iter=40, early_stopping=True, patience=5).fit(X, y)
    assert a.n_iter_ == b.n_iter_
    np.testing.assert_array_equal(a.V_, b.V_)


def test_ftrl_multiclass_early_stopping_not_implemented():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = rng.integers(0, 3, size=60)
    with pytest.raises(NotImplementedError):
        FMClassifier(optimizer="ftrl", random_state=0, max_iter=10, early_stopping=True).fit(X, y)


def test_invalid_validation_fraction():
    X, y = _data(n=40)
    with pytest.raises(ValueError, match="validation_fraction"):
        FMClassifier(
            random_state=0, max_iter=5, early_stopping=True, validation_fraction=1.5
        ).fit(X, y)


def test_bad_eval_set_raises():
    X, y = _data(n=40)
    with pytest.raises(ValueError, match="eval_set"):
        FMClassifier(random_state=0, max_iter=5).fit(X, y, eval_set=(X,))
