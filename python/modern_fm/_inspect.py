"""Model inspection: strongest learned pairwise interactions
(docs/roadmap.md v1.0 "Model inspection (top interactions)").

The interaction strength of a feature pair (i, j) is the magnitude of its
learned pairwise coefficient — the model-side quantity multiplied by
`x_i x_j` in the score (docs/math_spec.md):

- FM:   ``|<V_i, V_j>|``
- FwFM: ``|r[min(f_i, f_j), max(f_i, f_j)] * <V_i, V_j>|``
- FFM:  ``|<V[i, f_j], V[j, f_i]>|``

All functions scan the full upper triangle blockwise with BLAS matmuls and a
running top-n merge — exact, no sampling — so the cost is O(d^2 k) in
n_features d (fine up to tens of thousands of features; document before
pointing it at millions).
"""

import numpy as np

_BLOCK = 1024


def _merge_topn(best, cand, n_top):
    """Merge candidate (strength, i, j) arrays into the running top-n."""
    s = np.concatenate([best[0], cand[0]])
    i = np.concatenate([best[1], cand[1]])
    j = np.concatenate([best[2], cand[2]])
    if s.size > n_top:
        keep = np.argpartition(s, s.size - n_top)[-n_top:]
        s, i, j = s[keep], i[keep], j[keep]
    return s, i, j


def _empty_best():
    return (
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )


def _block_candidates(S, row_offset, n_top):
    """Top candidates from a strict-upper-triangle block of strengths.

    `S` is (b, d) strengths for global rows [row_offset, row_offset + b);
    entries with j <= i are masked out.
    """
    b, d = S.shape
    rows = np.arange(row_offset, row_offset + b)
    mask = np.arange(d)[None, :] <= rows[:, None]
    S = np.where(mask, -np.inf, S)
    flat = S.ravel()
    n_cand = min(n_top, flat.size)
    top = np.argpartition(flat, flat.size - n_cand)[-n_cand:]
    s = flat[top]
    finite = np.isfinite(s)
    top, s = top[finite], s[finite]
    return s, rows[top // d].astype(np.int64), (top % d).astype(np.int64)


def fm_top_interactions(V, n_top, r=None, field_ids=None):
    """Top-n |<V_i, V_j>| pairs of an FM factor matrix (d, k); with `r` and
    `field_ids` given, FwFM's field-pair weight scales each entry."""
    V = np.asarray(V, dtype=np.float64)
    d = V.shape[0]
    if r is not None:
        r = np.asarray(r, dtype=np.float64)
        r_full = np.triu(r) + np.triu(r, 1).T  # r[min, max] read for any order
        f = np.asarray(field_ids)
    best = _empty_best()
    for start in range(0, d, _BLOCK):
        stop = min(start + _BLOCK, d)
        S = np.abs(V[start:stop] @ V.T)
        if r is not None:
            S = S * np.abs(r_full[f[start:stop, None], f[None, :]])
        best = _merge_topn(best, _block_candidates(S, start, n_top), n_top)
    return _sorted_pairs(best)


def ffm_top_interactions(V, field_ids, n_top):
    """Top-n |<V[i, f_j], V[j, f_i]>| pairs of an FFM factor tensor
    (d, n_fields, k), grouped by field pair so each group is one matmul."""
    V = np.asarray(V, dtype=np.float64)
    field_ids = np.asarray(field_ids)
    n_fields = V.shape[1]
    idx_by_field = [np.where(field_ids == f)[0] for f in range(n_fields)]
    best = _empty_best()
    for f in range(n_fields):
        idx_f = idx_by_field[f]
        if idx_f.size == 0:
            continue
        for g in range(f, n_fields):
            idx_g = idx_by_field[g]
            if idx_g.size == 0:
                continue
            for start in range(0, idx_f.size, _BLOCK):
                rows = idx_f[start : start + _BLOCK]
                # i in field f interacts with j in field g through slots
                # V[i, g] and V[j, f].
                S = np.abs(V[rows][:, g, :] @ V[idx_g][:, f, :].T)
                if f == g:
                    # same-field blocks contain both orders and the diagonal;
                    # keep the strict upper triangle on global feature ids.
                    # Distinct field pairs list each feature pair exactly once
                    # (a feature's field decides its side), so no mask there.
                    S = np.where(idx_g[None, :] <= rows[:, None], -np.inf, S)
                flat = S.ravel()
                n_cand = min(n_top, flat.size)
                top = np.argpartition(flat, flat.size - n_cand)[-n_cand:]
                s = flat[top]
                finite = np.isfinite(s)
                top, s = top[finite], s[finite]
                cand_i = rows[top // idx_g.size].astype(np.int64)
                cand_j = idx_g[top % idx_g.size].astype(np.int64)
                # report pairs as (min, max) feature id
                lo = np.minimum(cand_i, cand_j)
                hi = np.maximum(cand_i, cand_j)
                best = _merge_topn(best, (s, lo, hi), n_top)
    return _sorted_pairs(best)


def _sorted_pairs(best):
    s, i, j = best
    order = np.argsort(s)[::-1]
    return [(int(i[o]), int(j[o]), float(s[o])) for o in order]
