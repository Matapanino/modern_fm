"""Backend dispatch: Rust extension when built, NumPy reference otherwise.

Private module. The NumPy implementations in `_reference` remain the ground
truth; the Rust extension (`modern_fm._rust`, built via maturin) is an
optimized drop-in whose parity is enforced by tests/test_rust_parity.py.

Estimators do not call this yet (training is Phase 2B); only prediction
plumbing lives here for now.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import _reference, _reference_train

try:
    from . import _rust
except ImportError:  # extension not built — pure-Python install
    _rust = None


def has_rust():
    return _rust is not None


def _prep_dense(X):
    return np.ascontiguousarray(X, dtype=np.float64)


def _prep_vec(a, dtype=np.float64):
    return np.ascontiguousarray(a, dtype=dtype)


def _prep_csr(X):
    X = X.tocsr().astype(np.float64)
    X.sum_duplicates()
    return (
        np.ascontiguousarray(X.indptr, dtype=np.int64),
        np.ascontiguousarray(X.indices, dtype=np.int64),
        np.ascontiguousarray(X.data, dtype=np.float64),
        X.shape[1],
    )


def fm_predict_fast(X, w0, w, V):
    """FM prediction (math_spec.md), Rust-accelerated when available."""
    if _rust is None:
        return _reference.fm_predict_fast(X, w0, w, V)
    w = _prep_vec(w)
    V = _prep_dense(V)
    if sp.issparse(X):
        indptr, indices, data, n_features = _prep_csr(X)
        return _rust.fm_predict_fast_csr(indptr, indices, data, n_features, float(w0), w, V)
    return _rust.fm_predict_fast_dense(_prep_dense(X), float(w0), w, V)


def ffm_predict(X, field_ids, w0, w, V):
    """FFM prediction (math_spec.md), Rust-accelerated when available."""
    if _rust is None:
        return _reference.ffm_predict(X, field_ids, w0, w, V)
    field_ids = _prep_vec(field_ids, dtype=np.int64)
    w = _prep_vec(w)
    V = _prep_dense(V)
    if sp.issparse(X):
        indptr, indices, data, n_features = _prep_csr(X)
        return _rust.ffm_predict_csr(
            indptr, indices, data, n_features, field_ids, float(w0), w, V
        )
    return _rust.ffm_predict_dense(_prep_dense(X), field_ids, float(w0), w, V)


def _prep_fit(X, y, params, row_orders):
    """Common coercion for the Rust fit entry points.

    Dense X is converted to CSR (exact zeros are skipped either way, matching
    the reference). Returns fresh float64 copies of w and V that the Rust
    kernel mutates in place; the caller's `params` are left untouched.
    """
    w0, w, V = params
    w = np.array(w, dtype=np.float64, order="C", copy=True)
    V = np.array(V, dtype=np.float64, order="C", copy=True)
    y = _prep_vec(y)
    row_orders = np.ascontiguousarray(row_orders, dtype=np.int64)
    if row_orders.ndim == 1:
        row_orders = row_orders[None, :]
    Xc = X if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float64))
    return _prep_csr(Xc), y, float(w0), w, V, row_orders


def fm_fit(X, y, params, *, loss, optimizer, learning_rate, l2_linear, l2_factors, row_orders):
    """Train an FM with batch_size=1 (docs/optimization_spec.md).

    `params` = (w0, w, V) initial values (unchanged); returns new float64
    (w0, w, V). Rust-accelerated when available, reference fallback otherwise.
    """
    if _rust is None:
        return _reference_train.fm_fit_reference(
            X, y, params, loss=loss, optimizer=optimizer, learning_rate=learning_rate,
            l2_linear=l2_linear, l2_factors=l2_factors, row_orders=row_orders,
        )
    (indptr, indices, data, n_features), y, w0, w, V, row_orders = _prep_fit(
        X, y, params, row_orders
    )
    w0 = _rust.fm_fit_csr(
        indptr, indices, data, n_features, y, w0, w, V,
        loss, optimizer, learning_rate, l2_linear, l2_factors, row_orders,
    )
    return w0, w, V


def ffm_fit(
    X, y, field_ids, params, *, optimizer, learning_rate, l2_linear, l2_factors, row_orders
):
    """Train an FFM (logistic loss) with batch_size=1; see fm_fit."""
    if _rust is None:
        return _reference_train.ffm_fit_reference(
            X, y, field_ids, params, optimizer=optimizer, learning_rate=learning_rate,
            l2_linear=l2_linear, l2_factors=l2_factors, row_orders=row_orders,
        )
    field_ids = _prep_vec(field_ids, dtype=np.int64)
    (indptr, indices, data, n_features), y, w0, w, V, row_orders = _prep_fit(
        X, y, params, row_orders
    )
    w0 = _rust.ffm_fit_csr(
        indptr, indices, data, n_features, y, field_ids, w0, w, V,
        optimizer, learning_rate, l2_linear, l2_factors, row_orders,
    )
    return w0, w, V
