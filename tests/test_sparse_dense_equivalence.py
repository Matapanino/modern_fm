import numpy as np
import scipy.sparse as sp
from conftest import random_ffm_params, random_fm_params, random_sparse_dense_X
from modern_fm._reference import ffm_predict, fm_predict_fast


def test_fm_dense_equals_csr(rng):
    X = random_sparse_dense_X(rng, n_samples=40, n_features=20, density=0.2)
    w0, w, V = random_fm_params(rng, n_features=20, n_factors=6)
    np.testing.assert_allclose(
        fm_predict_fast(X, w0, w, V),
        fm_predict_fast(sp.csr_matrix(X), w0, w, V),
        atol=1e-10,
    )


def test_fm_csr_array_input(rng):
    X = random_sparse_dense_X(rng, n_samples=15, n_features=10, density=0.3)
    w0, w, V = random_fm_params(rng, n_features=10, n_factors=4)
    np.testing.assert_allclose(
        fm_predict_fast(X, w0, w, V),
        fm_predict_fast(sp.csr_array(X), w0, w, V),
        atol=1e-10,
    )


def test_ffm_dense_equals_csr(rng):
    X = random_sparse_dense_X(rng, n_samples=30, n_features=16, density=0.25)
    w0, w, V, field_ids = random_ffm_params(rng, n_features=16, n_fields=5, n_factors=3)
    np.testing.assert_allclose(
        ffm_predict(X, field_ids, w0, w, V),
        ffm_predict(sp.csr_matrix(X), field_ids, w0, w, V),
        atol=1e-10,
    )


def test_fm_csr_with_empty_rows(rng):
    X = random_sparse_dense_X(rng, n_samples=10, n_features=8, density=0.3)
    X[3] = 0.0
    X[7] = 0.0
    w0, w, V = random_fm_params(rng, n_features=8, n_factors=3)
    np.testing.assert_allclose(
        fm_predict_fast(X, w0, w, V),
        fm_predict_fast(sp.csr_matrix(X), w0, w, V),
        atol=1e-10,
    )
    assert abs(fm_predict_fast(sp.csr_matrix(X), w0, w, V)[3] - w0) < 1e-10
