"""libffm text-format I/O (docs/data_format.md).

The libffm format has one example per line::

    <label> <field>:<feature>:<value> <field>:<feature>:<value> ...

``field`` and ``feature`` are 0-based integer indices and ``value`` is a float
(``1`` for one-hot). A feature belongs to exactly one field. These helpers map
that format to/from the arrays the estimators use:

    X, y, field_ids = load_libffm("train.ffm")
    FFMClassifier().fit(X, y, field_ids=field_ids)

``load_libffm`` returns a CSR ``X`` (n_samples x n_features), a float64 ``y``,
and an int64 ``field_ids`` (one field per feature/column). ``dump_libffm`` is the
inverse and only writes nonzero entries, so a feature/column that is zero in
every row cannot be represented (libffm infers ``n_features`` from the indices);
round-trips are exact otherwise.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["load_libffm", "dump_libffm"]


def _fmt_number(x):
    """Compact, round-trip-exact string: integers as ``5``, else Python ``repr``."""
    xf = float(x)
    return str(int(xf)) if xf.is_integer() else repr(xf)


def load_libffm(path):
    """Parse a libffm file into ``(X, y, field_ids)``.

    Parameters
    ----------
    path : str or path-like

    Returns
    -------
    X : scipy.sparse.csr_matrix, shape (n_samples, n_features), float64
    y : ndarray, shape (n_samples,), float64
    field_ids : ndarray, shape (n_features,), int64
    """
    labels = []
    rows, cols, vals = [], [], []
    field_of = {}
    n_features = 0
    r = 0
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            labels.append(float(parts[0]))
            for tok in parts[1:]:
                try:
                    f_str, j_str, v_str = tok.split(":")
                    field, feat, val = int(f_str), int(j_str), float(v_str)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{lineno}: bad token {tok!r}; expected field:feature:value"
                    ) from exc
                if feat < 0 or field < 0:
                    raise ValueError(f"{path}:{lineno}: negative field/feature in {tok!r}")
                prev = field_of.setdefault(feat, field)
                if prev != field:
                    raise ValueError(
                        f"{path}:{lineno}: feature {feat} maps to fields {prev} and {field}"
                    )
                rows.append(r)
                cols.append(feat)
                vals.append(val)
                n_features = max(n_features, feat + 1)
            r += 1
    X = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(r, n_features),
    )
    y = np.asarray(labels, dtype=np.float64)
    field_ids = np.zeros(n_features, dtype=np.int64)
    for feat, field in field_of.items():
        field_ids[feat] = field
    return X, y, field_ids


def dump_libffm(path, X, y, field_ids):
    """Write ``(X, y, field_ids)`` to ``path`` in libffm text format.

    Only nonzero entries of ``X`` (dense or sparse) are emitted. ``field_ids``
    must have one entry per column of ``X``.
    """
    X = sp.csr_matrix(X)
    field_ids = np.asarray(field_ids, dtype=np.int64)
    y = np.asarray(y)
    if field_ids.shape != (X.shape[1],):
        raise ValueError(f"field_ids has shape {field_ids.shape}, expected ({X.shape[1]},)")
    if y.shape != (X.shape[0],):
        raise ValueError(f"y has shape {y.shape}, expected ({X.shape[0]},)")
    indptr, indices, data = X.indptr, X.indices, X.data
    with open(path, "w") as fh:
        for i in range(X.shape[0]):
            toks = [_fmt_number(y[i])]
            for k in range(indptr[i], indptr[i + 1]):
                j = indices[k]
                toks.append(f"{field_ids[j]}:{j}:{_fmt_number(data[k])}")
            fh.write(" ".join(toks))
            fh.write("\n")
