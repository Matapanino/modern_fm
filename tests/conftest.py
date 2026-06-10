import numpy as np
import pytest


@pytest.fixture(params=[0, 1, 2])
def rng(request):
    return np.random.default_rng(request.param)


def random_fm_params(rng, n_features, n_factors):
    w0 = float(rng.normal())
    w = rng.normal(size=n_features)
    V = rng.normal(size=(n_features, n_factors))
    return w0, w, V


def random_ffm_params(rng, n_features, n_fields, n_factors):
    w0 = float(rng.normal())
    w = rng.normal(size=n_features)
    V = rng.normal(size=(n_features, n_fields, n_factors))
    field_ids = rng.integers(0, n_fields, size=n_features)
    return w0, w, V, field_ids


def random_sparse_dense_X(rng, n_samples, n_features, density=0.3):
    """Dense X with many exact zeros, so CSR and dense paths see the same data."""
    X = rng.normal(size=(n_samples, n_features))
    X[rng.random(size=X.shape) > density] = 0.0
    return X
