"""Parity tests: Rust backend vs the pure-NumPy reference implementations.

Skipped entirely when the extension is not built (pure-Python install);
when built, the Rust results must match the reference to float64 precision.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import random_ffm_params, random_fm_params, random_sparse_dense_X
from modern_fm import _backend
from modern_fm._reference import ffm_predict, fm_predict_fast

pytestmark = pytest.mark.skipif(
    not _backend.has_rust(), reason="modern_fm._rust extension not built"
)

ATOL = 1e-12
RTOL = 1e-12


def test_fm_dense_parity(rng):
    X = random_sparse_dense_X(rng, n_samples=50, n_features=24, density=0.4)
    w0, w, V = random_fm_params(rng, n_features=24, n_factors=6)
    np.testing.assert_allclose(
        _backend._rust.fm_predict_fast_dense(X, w0, w, V),
        fm_predict_fast(X, w0, w, V),
        atol=ATOL,
        rtol=RTOL,
    )


def test_fm_csr_parity(rng):
    X = random_sparse_dense_X(rng, n_samples=50, n_features=24, density=0.2)
    w0, w, V = random_fm_params(rng, n_features=24, n_factors=6)
    np.testing.assert_allclose(
        _backend.fm_predict_fast(sp.csr_matrix(X), w0, w, V),
        fm_predict_fast(X, w0, w, V),
        atol=ATOL,
        rtol=RTOL,
    )


def test_ffm_dense_parity(rng):
    X = random_sparse_dense_X(rng, n_samples=40, n_features=18, density=0.3)
    w0, w, V, field_ids = random_ffm_params(rng, n_features=18, n_fields=5, n_factors=4)
    np.testing.assert_allclose(
        _backend.ffm_predict(X, field_ids, w0, w, V),
        ffm_predict(X, field_ids, w0, w, V),
        atol=ATOL,
        rtol=RTOL,
    )


def test_ffm_csr_parity(rng):
    X = random_sparse_dense_X(rng, n_samples=40, n_features=18, density=0.25)
    w0, w, V, field_ids = random_ffm_params(rng, n_features=18, n_fields=5, n_factors=4)
    np.testing.assert_allclose(
        _backend.ffm_predict(sp.csr_matrix(X), field_ids, w0, w, V),
        ffm_predict(X, field_ids, w0, w, V),
        atol=ATOL,
        rtol=RTOL,
    )


def test_fm_zero_rows_and_single_nonzero(rng):
    X = np.zeros((3, 10))
    X[1, 4] = 2.5  # single nonzero -> no pairwise term
    w0, w, V = random_fm_params(rng, n_features=10, n_factors=3)
    out = _backend.fm_predict_fast(X, w0, w, V)
    np.testing.assert_allclose(out[0], w0, atol=ATOL)
    np.testing.assert_allclose(out[1], w0 + w[4] * 2.5, atol=ATOL)
    np.testing.assert_allclose(out, fm_predict_fast(X, w0, w, V), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(
        _backend.fm_predict_fast(sp.csr_matrix(X), w0, w, V), out, atol=ATOL, rtol=RTOL
    )


def test_ffm_zero_rows_and_single_nonzero(rng):
    X = np.zeros((3, 8))
    X[2, 1] = -1.5
    w0, w, V, field_ids = random_ffm_params(rng, n_features=8, n_fields=3, n_factors=2)
    out = _backend.ffm_predict(X, field_ids, w0, w, V)
    np.testing.assert_allclose(out[0], w0, atol=ATOL)
    np.testing.assert_allclose(out[2], w0 + w[1] * -1.5, atol=ATOL)
    np.testing.assert_allclose(
        _backend.ffm_predict(sp.csr_matrix(X), field_ids, w0, w, V), out, atol=ATOL, rtol=RTOL
    )


def test_ffm_tiny_hand_computed_example():
    # Same fixture as test_ffm_correctness.py: expected prediction 9.5.
    V = np.zeros((3, 2, 2))
    V[0, 0] = [1.0, 0.0]
    V[0, 1] = [0.0, 1.0]
    V[1, 0] = [1.0, 1.0]
    V[1, 1] = [2.0, 0.0]
    V[2, 0] = [0.5, 0.5]
    V[2, 1] = [1.0, -1.0]
    field_ids = np.array([0, 0, 1])
    X = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(_backend.ffm_predict(X, field_ids, 0.0, np.zeros(3), V), [9.5])
    np.testing.assert_allclose(
        _backend.ffm_predict(sp.csr_matrix(X), field_ids, 0.0, np.zeros(3), V), [9.5]
    )


def test_fm_hand_computed_example():
    # Same fixture as test_fm_correctness.py: expected prediction 4.25.
    X = np.array([[2.0, 3.0]])
    w0, w = 0.25, np.array([0.5, -1.0])
    V = np.array([[1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(_backend.fm_predict_fast(X, w0, w, V), [4.25])


def test_fm_noncontiguous_and_float32_inputs_handled(rng):
    # The wrapper must coerce dtype/contiguity before calling Rust.
    X64 = random_sparse_dense_X(rng, n_samples=12, n_features=10)
    w0, w, V = random_fm_params(rng, n_features=10, n_factors=3)
    X_f = np.asfortranarray(X64)
    np.testing.assert_allclose(
        _backend.fm_predict_fast(X_f, w0, w, V), fm_predict_fast(X64, w0, w, V), atol=ATOL
    )


def test_rust_rejects_bad_shapes(rng):
    X = np.zeros((2, 5))
    w0, w, V = random_fm_params(rng, n_features=5, n_factors=3)
    with pytest.raises(ValueError):
        _backend.fm_predict_fast(X, w0, w[:4], V)  # wrong w length


def test_rust_rejects_bad_field_ids(rng):
    X = np.zeros((2, 5))
    w0, w, V, field_ids = random_ffm_params(rng, n_features=5, n_fields=2, n_factors=2)
    bad = field_ids.copy()
    bad[0] = 7  # out of range for n_fields=2
    with pytest.raises(ValueError):
        _backend.ffm_predict(X, bad, w0, w, V)
