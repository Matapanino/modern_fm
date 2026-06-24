"""Preprocessing helpers (docs/data_format.md).

``CategoricalEncoder`` one-hot encodes integer-coded categorical columns into a
scipy CSR matrix and records, for every output column, the index of the raw
column it came from. That per-column field mapping is exactly what FFM needs::

    enc = CategoricalEncoder().fit(X_train_int)
    Xtr = enc.transform(X_train_int)
    FFMClassifier(...).fit(Xtr, y, field_ids=enc.field_ids_)

Each raw column becomes one field, so ``n_fields_ == n_features_in_``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._base import ParamsMixin, check_is_fitted


def _check_2d(X):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got ndim={X.ndim}")
    return X


class CategoricalEncoder(ParamsMixin):
    """One-hot encode integer categorical columns to CSR, tracking field ids.

    Parameters
    ----------
    handle_unknown : {"ignore", "error"}, default "ignore"
        At ``transform`` time, categories not seen during ``fit`` either
        contribute no active column for that field ("ignore") or raise
        ("error").

    Learned attributes
    ------------------
    categories_ : list of arrays, the sorted categories of each input column.
    field_ids_ : (n_features_out_,) int64, source column of each output column.
    n_features_in_, n_features_out_, n_fields_ : ints.
    """

    def __init__(self, handle_unknown="ignore"):
        self.handle_unknown = handle_unknown

    def fit(self, X):
        if self.handle_unknown not in ("ignore", "error"):
            raise ValueError(
                f"handle_unknown must be 'ignore' or 'error', got {self.handle_unknown!r}"
            )
        X = _check_2d(X)
        n_cols = X.shape[1]
        self.categories_ = [np.unique(X[:, c]) for c in range(n_cols)]
        sizes = np.array([len(c) for c in self.categories_], dtype=np.int64)
        self.offsets_ = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.n_features_in_ = n_cols
        self.n_features_out_ = int(self.offsets_[-1])
        self.n_fields_ = n_cols
        self.field_ids_ = np.repeat(np.arange(n_cols, dtype=np.int64), sizes)
        return self

    def transform(self, X):
        check_is_fitted(self, "categories_")
        X = _check_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} columns, but the encoder was fitted with "
                f"{self.n_features_in_}"
            )
        n_rows = X.shape[0]
        rows, cols = [], []
        for c in range(self.n_features_in_):
            cats = self.categories_[c]
            vals = X[:, c]
            idx = np.searchsorted(cats, vals)
            in_bounds = idx < len(cats)
            # searchsorted gives an insertion point; the value is a known
            # category only when it lands exactly on an existing entry.
            known = in_bounds.copy()
            known[in_bounds] = cats[idx[in_bounds]] == vals[in_bounds]
            if self.handle_unknown == "error" and not known.all():
                raise ValueError(f"unknown category in column {c}")
            r = np.nonzero(known)[0]
            rows.append(r)
            cols.append(self.offsets_[c] + idx[r])
        rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
        cols = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
        data = np.ones(len(rows), dtype=np.float64)
        return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, self.n_features_out_))

    def fit_transform(self, X):
        return self.fit(X).transform(X)
