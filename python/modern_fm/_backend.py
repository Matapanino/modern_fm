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

from . import _reference

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
