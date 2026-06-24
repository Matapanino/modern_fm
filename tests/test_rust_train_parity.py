"""Parity tests: Rust training kernels vs the Python reference trainers.

Both start from identical initial parameters and identical row_orders, so
results must agree to float64 round-off (summation-order differences only).
"""

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import random_sparse_dense_X
from modern_fm import _backend
from modern_fm._reference_train import (
    ffm_fit_reference,
    fm_fit_reference,
    init_ffm_params,
    init_fm_params,
    make_row_orders,
)

pytestmark = pytest.mark.skipif(
    not _backend.has_rust(), reason="modern_fm._rust extension not built"
)

RTOL = 1e-9
ATOL = 1e-12


def _assert_params_close(a, b):
    np.testing.assert_allclose(a[0], b[0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(a[1], b[1], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(a[2], b[2], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("loss", ["logistic", "squared"])
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad"])
def test_fm_training_parity(rng, loss, optimizer):
    n, d, k = 40, 10, 3
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = rng.normal(size=n) if loss == "squared" else (rng.random(n) > 0.4).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    kwargs = dict(
        loss=loss, optimizer=optimizer, learning_rate=0.1,
        l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=3),
    )
    _assert_params_close(
        _backend.fm_fit(X, y, params, **kwargs),
        fm_fit_reference(X, y, params, **kwargs),
    )


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad"])
def test_ffm_training_parity(rng, optimizer):
    n, d, n_fields, k = 30, 8, 3, 2
    X = random_sparse_dense_X(rng, n, d, density=0.5)
    y = (rng.random(n) > 0.5).astype(np.float64)
    field_ids = rng.integers(0, n_fields, size=d)
    params = init_ffm_params(rng, d, n_fields, k, 0.05)
    kwargs = dict(
        optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=2),
    )
    _assert_params_close(
        _backend.ffm_fit(X, y, field_ids, params, **kwargs),
        ffm_fit_reference(X, y, field_ids, params, **kwargs),
    )


def test_fm_training_parity_csr_input(rng):
    n, d, k = 30, 8, 2
    X = random_sparse_dense_X(rng, n, d, density=0.3)
    y = (rng.random(n) > 0.5).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    kwargs = dict(
        loss="logistic", optimizer="adagrad", learning_rate=0.1,
        l2_linear=0.0, l2_factors=0.0, row_orders=make_row_orders(rng, n, epochs=2),
    )
    _assert_params_close(
        _backend.fm_fit(sp.csr_matrix(X), y, params, **kwargs),
        fm_fit_reference(X, y, params, **kwargs),
    )


def test_fit_does_not_mutate_input_params(rng):
    n, d, k = 10, 5, 2
    X = random_sparse_dense_X(rng, n, d)
    y = (rng.random(n) > 0.5).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    w_before, V_before = params[1].copy(), params[2].copy()
    _backend.fm_fit(
        X, y, params, loss="logistic", optimizer="sgd", learning_rate=0.1,
        l2_linear=0.0, l2_factors=0.0, row_orders=make_row_orders(rng, n, epochs=1),
    )
    np.testing.assert_array_equal(params[1], w_before)
    np.testing.assert_array_equal(params[2], V_before)


def test_rust_fit_rejects_bad_row_orders(rng):
    n, d, k = 6, 4, 2
    X = random_sparse_dense_X(rng, n, d)
    y = np.zeros(n)
    params = init_fm_params(rng, d, k, 0.05)
    bad = np.full((1, n), n, dtype=np.int64)  # out of range
    with pytest.raises(ValueError):
        _backend.fm_fit(
            X, y, params, loss="logistic", optimizer="sgd", learning_rate=0.1,
            l2_linear=0.0, l2_factors=0.0, row_orders=bad,
        )
