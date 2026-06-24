"""Input validation at fit/predict boundaries (CLAUDE.md coding rules)."""

import numpy as np
import pytest
import scipy.sparse as sp
from modern_fm import FFMClassifier, FMClassifier, FMRegressor


def _binary(n=8, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_fit_rejects_nonfinite_dense_X():
    X, y = _binary()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        FMClassifier(random_state=0, max_iter=2).fit(X, y)


def test_fit_rejects_nonfinite_sparse_X():
    X, y = _binary()
    Xs = sp.csr_matrix(X)
    Xs.data[0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinity"):
        FMClassifier(random_state=0, max_iter=2).fit(Xs, y)


def test_predict_rejects_nonfinite_X():
    X, y = _binary()
    model = FMClassifier(random_state=0, max_iter=2).fit(X, y)
    bad = X.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinity"):
        model.predict(bad)


def test_regressor_rejects_nonfinite_y():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    y = X[:, 0].copy()
    y[0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        FMRegressor(random_state=0, max_iter=2).fit(X, y)


def test_ffm_rejects_nonfinite_X():
    X, y = _binary(d=4)
    X[1, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        FFMClassifier(random_state=0, max_iter=2).fit(X, y, field_ids=np.arange(4) % 2)


def test_ndim_validation():
    with pytest.raises(ValueError, match="2-dimensional"):
        FMClassifier().fit(np.zeros(5), np.zeros(5))
