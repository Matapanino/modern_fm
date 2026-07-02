"""top_interactions (docs/roadmap.md v1.0 "Model inspection").

Strength definitions (magnitude of the learned pairwise coefficient of
x_i x_j, docs/math_spec.md / math_spec_fwfm.md):
FM |<V_i, V_j>|; FwFM |r[min,max] * <V_i, V_j>|; FFM |<V[i,f_j], V[j,f_i]>|.
Blockwise-exactness is pinned against a naive O(d^2) reference.
"""

import numpy as np
import pytest
from modern_fm import (
    FFMClassifier,
    FFMRegressor,
    FMClassifier,
    FMRegressor,
    FwFMClassifier,
    _inspect,
)
from sklearn.exceptions import NotFittedError


def _fit_tiny(cls, rng, n=40, d=8, multiclass=False, **kw):
    X = rng.normal(size=(n, d))
    if cls in (FMRegressor, FFMRegressor):
        y = X @ rng.normal(size=d)
    elif multiclass:
        y = np.arange(n) % 3
    else:
        y = (X[:, 0] > 0).astype(int)
    return cls(n_factors=3, max_iter=2, random_state=0, **kw).fit(X, y)


def _naive_fm(V, r=None, field_ids=None):
    d = V.shape[0]
    out = []
    for i in range(d):
        for j in range(i + 1, d):
            s = abs(float(V[i] @ V[j]))
            if r is not None:
                fi, fj = field_ids[i], field_ids[j]
                s *= abs(float(r[min(fi, fj), max(fi, fj)]))
            out.append((i, j, s))
    return sorted(out, key=lambda t: -t[2])


def _naive_ffm(V, field_ids):
    d = V.shape[0]
    out = []
    for i in range(d):
        for j in range(i + 1, d):
            s = abs(float(V[i, field_ids[j]] @ V[j, field_ids[i]]))
            out.append((i, j, s))
    return sorted(out, key=lambda t: -t[2])


def _assert_matches(got, want):
    assert len(got) == len(want)
    for (gi, gj, gs), (wi, wj, ws) in zip(got, want):
        assert (gi, gj) == (wi, wj)
        np.testing.assert_allclose(gs, ws, rtol=1e-12)


def test_fm_planted_dominant_pair(rng):
    m = _fit_tiny(FMClassifier, rng)
    V = np.zeros((8, 3))
    V[2] = [10.0, 0, 0]
    V[5] = [10.0, 0, 0]  # <V_2, V_5> = 100, everything else 0
    V[1] = [0, 0.1, 0]
    V[6] = [0, 0.1, 0]  # runner-up 0.01
    m.V_ = V
    top = m.top_interactions(2)
    assert top[0][:2] == (2, 5)
    np.testing.assert_allclose(top[0][2], 100.0)
    assert top[1][:2] == (1, 6)


def test_ffm_planted_dominant_pair(rng):
    m = _fit_tiny(FFMClassifier, rng)
    d, F, k = 8, m.n_fields_, 3
    V = np.zeros((d, F, k))
    fi, fj = m.field_ids_[3], m.field_ids_[6]
    V[3, fj] = [7.0, 0, 0]
    V[6, fi] = [7.0, 0, 0]  # <V[3,f_6], V[6,f_3]> = 49
    m.V_ = V
    top = m.top_interactions(1)
    assert top[0][:2] == (3, 6)
    np.testing.assert_allclose(top[0][2], 49.0)


def test_fwfm_r_weights_the_pair(rng):
    m = _fit_tiny(FwFMClassifier, rng)
    d = 8
    V = np.zeros((d, 3))
    V[0] = [2.0, 0, 0]
    V[1] = [2.0, 0, 0]  # dot 4
    V[2] = [1.0, 0, 0]
    V[3] = [1.0, 0, 0]  # dot 1
    m.V_ = V
    r = np.zeros((m.n_fields_, m.n_fields_))
    f = m.field_ids_
    r[min(f[0], f[1]), max(f[0], f[1])] = 0.1  # 4 * 0.1 = 0.4
    r[min(f[2], f[3]), max(f[2], f[3])] = 1.0  # 1 * 1.0 = 1.0 -> wins
    m.r_ = r
    top = m.top_interactions(2)
    assert top[0][:2] == (2, 3)
    np.testing.assert_allclose(top[0][2], 1.0)


@pytest.mark.parametrize("d", [5, 60])
def test_fm_blockwise_matches_naive(rng, d):
    V = rng.normal(size=(d, 4))
    _assert_matches(_inspect.fm_top_interactions(V, 15), _naive_fm(V)[:15])


def test_fwfm_blockwise_matches_naive(rng):
    d, F = 40, 5
    V = rng.normal(size=(d, 4))
    field_ids = rng.integers(0, F, size=d)
    r = np.triu(rng.normal(size=(F, F)))
    got = _inspect.fm_top_interactions(V, 10, r=r, field_ids=field_ids)
    _assert_matches(got, _naive_fm(V, r=r, field_ids=field_ids)[:10])


def test_ffm_blockwise_matches_naive(rng):
    d, F = 40, 5
    V = rng.normal(size=(d, F, 4))
    field_ids = rng.integers(0, F, size=d)
    got = _inspect.ffm_top_interactions(V, field_ids, 10)
    _assert_matches(got, _naive_ffm(V, field_ids)[:10])


def test_n_top_larger_than_pairs(rng):
    V = rng.normal(size=(4, 2))
    got = _inspect.fm_top_interactions(V, 100)
    assert len(got) == 6  # all C(4,2) pairs, no padding


@pytest.mark.parametrize("cls", [FMClassifier, FMRegressor, FFMClassifier, FFMRegressor])
def test_estimator_end_to_end(rng, cls):
    m = _fit_tiny(cls, rng)
    top = m.top_interactions(3)
    assert len(top) == 3
    for i, j, s in top:
        assert 0 <= i < j < 8 and s >= 0.0


def test_multiclass_requires_class_idx(rng):
    m = _fit_tiny(FMClassifier, rng, multiclass=True)
    with pytest.raises(ValueError, match="class_idx"):
        m.top_interactions(3)
    top = m.top_interactions(3, class_idx=1)
    assert len(top) == 3


def test_binary_rejects_class_idx(rng):
    m = _fit_tiny(FMClassifier, rng)
    with pytest.raises(ValueError, match="only valid for a multiclass"):
        m.top_interactions(3, class_idx=0)


def test_validation_errors(rng):
    with pytest.raises(NotFittedError):
        FMClassifier().top_interactions()
    m = _fit_tiny(FMClassifier, rng)
    with pytest.raises(ValueError, match="n_top"):
        m.top_interactions(0)
