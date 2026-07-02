"""Pure-NumPy reference implementations of FM and FFM predictions.

These are the ground truth for every backend (see docs/math_spec.md).
Correctness over speed: the naive variants exist solely so the optimized
variants (and later the Rust backend) can be tested against them.

Shapes:
    X         : (n_samples, n_features), dense ndarray or scipy CSR
    w0        : scalar bias
    w         : (n_features,) linear weights
    V (FM)    : (n_features, n_factors)
    V (FFM)   : (n_features, n_fields, n_factors)
    field_ids : (n_features,) int, field of each feature/column
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _as_dense_rows(X):
    """Yield (nonzero_indices, nonzero_values) per row for dense or CSR input."""
    if sp.issparse(X):
        X = X.tocsr()
        for r in range(X.shape[0]):
            lo, hi = X.indptr[r], X.indptr[r + 1]
            yield X.indices[lo:hi], X.data[lo:hi]
    else:
        X = np.asarray(X)
        for row in X:
            idx = np.nonzero(row)[0]
            yield idx, row[idx]


def fm_predict_naive(X, w0, w, V):
    """FM prediction via the explicit O(n^2 k) pairwise sum.

    y_hat = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j
    """
    w = np.asarray(w, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    out = np.empty(X.shape[0], dtype=np.float64)
    for r, (idx, val) in enumerate(_as_dense_rows(X)):
        s = w0 + w[idx] @ val
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                s += (V[idx[a]] @ V[idx[b]]) * val[a] * val[b]
        out[r] = s
    return out


def fm_predict_fast(X, w0, w, V):
    """FM prediction via Rendle's O(nk) reformulation. Dense or CSR input.

    pairwise = 0.5 * sum_f [(sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2]
    """
    w = np.asarray(w, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if sp.issparse(X):
        X = X.tocsr().astype(np.float64)
        linear = X @ w
        xv = X @ V                       # (n_samples, k)
        x2v2 = (X.multiply(X)) @ (V**2)  # (n_samples, k)
        x2v2 = np.asarray(x2v2)
    else:
        X = np.asarray(X, dtype=np.float64)
        linear = X @ w
        xv = X @ V
        x2v2 = (X**2) @ (V**2)
    pairwise = 0.5 * (xv**2 - x2v2).sum(axis=1)
    return w0 + linear + pairwise


def ffm_predict_naive(X, field_ids, w0, w, V):
    """FFM prediction via the explicit pairwise loop (docs/math_spec.md).

    y_hat = w0 + sum_i w_i x_i + sum_{i<j} <V[i, f_j], V[j, f_i]> x_i x_j
    """
    field_ids = np.asarray(field_ids)
    w = np.asarray(w, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    out = np.empty(X.shape[0], dtype=np.float64)
    for r, (idx, val) in enumerate(_as_dense_rows(X)):
        s = w0 + w[idx] @ val
        for a in range(len(idx)):
            i, xi = idx[a], val[a]
            for b in range(a + 1, len(idx)):
                j, xj = idx[b], val[b]
                s += (V[i, field_ids[j]] @ V[j, field_ids[i]]) * xi * xj
        out[r] = s
    return out


def ffm_predict(X, field_ids, w0, w, V):
    """FFM prediction, vectorized per row. Dense or CSR input.

    For each row with nonzero columns nz, builds G[a, b] =
    <V[nz_a, f_b], V[nz_b, f_a]> x_a x_b and sums the strict upper triangle.
    O(z^2 k) per row in the number of nonzeros z — there is no FM-style
    factorization for FFM.
    """
    field_ids = np.asarray(field_ids)
    w = np.asarray(w, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    out = np.empty(X.shape[0], dtype=np.float64)
    for r, (idx, val) in enumerate(_as_dense_rows(X)):
        s = w0 + w[idx] @ val
        if len(idx) >= 2:
            f = field_ids[idx]
            Vsub = V[np.ix_(idx, f)]                  # Vsub[a, b] = V[nz_a, f_b]
            G = np.einsum("abk,bak->ab", Vsub, Vsub)  # G[a, b] = <V[nz_a,f_b], V[nz_b,f_a]>
            G = G * np.outer(val, val)
            s += np.triu(G, k=1).sum()
        out[r] = s
    return out


def fwfm_predict_naive(X, field_ids, w0, w, V, r):
    """FwFM prediction via the explicit pairwise loop (docs/math_spec_fwfm.md).

    y_hat = w0 + sum_i w_i x_i + sum_{i<j} r_{f_i f_j} <v_i, v_j> x_i x_j

    `V` is FM-shaped (n_features, k); `r` is (n_fields, n_fields) with only the
    upper triangle (incl. diagonal) read via r[min(f_i,f_j), max(f_i,f_j)].
    """
    field_ids = np.asarray(field_ids)
    w = np.asarray(w, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    out = np.empty(X.shape[0], dtype=np.float64)
    for row, (idx, val) in enumerate(_as_dense_rows(X)):
        s = w0 + w[idx] @ val
        for a in range(len(idx)):
            i, xi = idx[a], val[a]
            for b in range(a + 1, len(idx)):
                j, xj = idx[b], val[b]
                fa, fb = field_ids[i], field_ids[j]
                s += r[min(fa, fb), max(fa, fb)] * (V[i] @ V[j]) * xi * xj
        out[row] = s
    return out


def fwfm_predict(X, field_ids, w0, w, V, r):
    """FwFM prediction, vectorized per row. Dense or CSR input.

    For each row with nonzero columns nz, builds
    G[a, b] = r_{f_a f_b} <v_a, v_b> x_a x_b and sums the strict upper
    triangle. O(z^2 k) per row (docs/math_spec_fwfm.md fixes the pair order).
    """
    field_ids = np.asarray(field_ids)
    w = np.asarray(w, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    out = np.empty(X.shape[0], dtype=np.float64)
    for row, (idx, val) in enumerate(_as_dense_rows(X)):
        s = w0 + w[idx] @ val
        if len(idx) >= 2:
            f = field_ids[idx]
            W = r[np.minimum.outer(f, f), np.maximum.outer(f, f)]  # r_{f_a f_b}
            G = (V[idx] @ V[idx].T) * W * np.outer(val, val)
            s += np.triu(G, k=1).sum()
        out[row] = s
    return out


def fm_bi_interaction(X, V):
    """Bi-interaction pooling (He & Chua, SIGIR 2017): the k-dim vector of the
    FM pairwise term before its factor-sum,

        f_BI(x)_f = 0.5 * [(sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2]

    so `fm_predict_fast(X, w0, w, V) == w0 + X @ w + fm_bi_interaction(X, V).sum(axis=1)`
    exactly. Returns (n_samples, k). Dense or CSR input, O(nnz * k), no
    parameters beyond V — used as a feature transform (a linear head over it
    provably collapses to plain FM; see BiInteractionPooling).
    """
    V = np.asarray(V, dtype=np.float64)
    if sp.issparse(X):
        X = X.tocsr().astype(np.float64)
        xv = X @ V
        x2v2 = np.asarray((X.multiply(X)) @ (V**2))
    else:
        X = np.asarray(X, dtype=np.float64)
        xv = X @ V
        x2v2 = (X**2) @ (V**2)
    return 0.5 * (xv**2 - x2v2)
