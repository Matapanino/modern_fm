"""partial_fit / warm_start: incremental & streaming training (docs/roadmap.md v0.4).

Core contract: N sequential ``partial_fit`` calls over consecutive chunks equal one
``partial_fit`` over the concatenation, bit-for-bit, via the persisted
optimizer-state round-trip (``dtype="float64"``, ``n_jobs=1``, ``batch_size``
dividing the chunk lengths). Also covers the sklearn first-call ``classes``
convention, field-id plumbing, and ``warm_start`` resume.
"""

import numpy as np
import pytest
from modern_fm import FFMClassifier, FFMRegressor, FMClassifier, FMRegressor
from modern_fm._reference_train import fm_fit_reference, init_fm_params, make_row_orders

FIELD_IDS = np.arange(6) % 3


def _data(seed=0, n=90, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    yb = (X[:, 0] + X[:, 1] > 0).astype(int)
    ymc = (X[:, :3] @ rng.normal(size=(3, 3))).argmax(axis=1)
    yr = X[:, 0] * 2.0 + rng.normal(size=n) * 0.1
    return X, yb, ymc, yr


def _make(cls, optimizer, **kw):
    return cls(optimizer=optimizer, n_factors=4, learning_rate=0.1, random_state=0,
               n_jobs=1, dtype="float64", **kw)


# (label, class, y-key, classes-or-None, uses_field_ids)
CASES = [
    ("fm-binary", FMClassifier, "yb", np.array([0, 1]), False),
    ("fm-multiclass", FMClassifier, "ymc", np.array([0, 1, 2]), False),
    ("fm-reg", FMRegressor, "yr", None, False),
    ("ffm-binary", FFMClassifier, "yb", np.array([0, 1]), True),
    ("ffm-multiclass", FFMClassifier, "ymc", np.array([0, 1, 2]), True),
    ("ffm-reg", FFMRegressor, "yr", None, True),
]


def _pfit(model, X, y, classes, uses_field_ids):
    kw = {}
    if classes is not None:
        kw["classes"] = classes
    if uses_field_ids:
        kw["field_ids"] = FIELD_IDS
    return model.partial_fit(X, y, **kw)


def _params(m):
    return np.asarray(m.w0_, dtype=np.float64), np.asarray(m.w_), np.asarray(m.V_)


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
@pytest.mark.parametrize("label,cls,ykey,classes,fids", CASES, ids=[c[0] for c in CASES])
def test_chunks_equal_single_pass(optimizer, label, cls, ykey, classes, fids):
    """N partial_fit calls over consecutive chunks == one partial_fit over all rows,
    bit-for-bit (the persisted optimizer-state round-trip; float64, n_jobs=1)."""
    X, yb, ymc, yr = _data()
    y = {"yb": yb, "ymc": ymc, "yr": yr}[ykey]
    one = _make(cls, optimizer)
    _pfit(one, X, y, classes, fids)
    chunks = _make(cls, optimizer)
    for s in range(0, len(X), 30):
        _pfit(chunks, X[s:s + 30], y[s:s + 30], classes, fids)
    for a, b in zip(_params(one), _params(chunks)):
        np.testing.assert_array_equal(a, b)


def test_ground_truth_matches_reference():
    """Chunked partial_fit equals a direct reference-trainer single natural-order
    epoch — anchors the numerics, not just self-consistency."""
    X, yb, _, _ = _data()
    classes = np.array([0, 1])
    chunks = _make(FMClassifier, "adagrad")
    for s in range(0, len(X), 30):
        chunks.partial_fit(X[s:s + 30], yb[s:s + 30], classes=classes)
    init = init_fm_params(np.random.default_rng(0), X.shape[1], 4, 0.01)
    ro = make_row_orders(np.random.default_rng(0), len(X), 1, shuffle=False)
    w0, w, V = fm_fit_reference(
        X, yb.astype(np.float64), init, optimizer="adagrad", learning_rate=0.1,
        l2_linear=1e-5, l2_factors=1e-5, row_orders=ro,
    )
    assert np.isclose(chunks.w0_, w0)
    np.testing.assert_allclose(chunks.w_, w)
    np.testing.assert_allclose(chunks.V_, V)


def test_batch_size_alignment_exact():
    """Exactness holds for batch_size>1 when chunk lengths are multiples of it."""
    X, yb, *_ = _data(n=80)
    classes = np.array([0, 1])
    one = _make(FMClassifier, "adagrad", batch_size=4)
    one.partial_fit(X, yb, classes=classes)
    chunks = _make(FMClassifier, "adagrad", batch_size=4)
    for s in range(0, 80, 20):  # 20 is a multiple of batch_size=4
        chunks.partial_fit(X[s:s + 20], yb[s:s + 20], classes=classes)
    np.testing.assert_array_equal(one.w_, chunks.w_)
    np.testing.assert_array_equal(one.V_, chunks.V_)


# --- sklearn semantics ---

def test_first_call_requires_classes():
    X, yb, *_ = _data()
    with pytest.raises(ValueError):
        FMClassifier().partial_fit(X, yb)


def test_later_call_class_mismatch_raises():
    X, yb, *_ = _data()
    m = FMClassifier(random_state=0)
    m.partial_fit(X[:30], yb[:30], classes=np.array([0, 1]))
    with pytest.raises(ValueError):
        m.partial_fit(X[30:60], yb[30:60], classes=np.array([0, 2]))


def test_unseen_label_raises():
    X, _, ymc, _ = _data()
    with pytest.raises(ValueError):
        FMClassifier(random_state=0).partial_fit(X, ymc, classes=np.array([0, 1]))


def test_feature_count_mismatch_raises():
    X, yb, *_ = _data()
    m = FMClassifier(random_state=0)
    m.partial_fit(X[:30], yb[:30], classes=np.array([0, 1]))
    with pytest.raises(ValueError):
        m.partial_fit(X[30:60, :5], yb[30:60], classes=np.array([0, 1]))


def test_field_ids_mismatch_raises():
    X, yb, *_ = _data()
    m = FFMClassifier(random_state=0)
    m.partial_fit(X[:30], yb[:30], classes=np.array([0, 1]), field_ids=FIELD_IDS)
    with pytest.raises(ValueError, match="field_ids"):
        m.partial_fit(X[30:60], yb[30:60], classes=np.array([0, 1]), field_ids=np.arange(6))


def test_class_weight_balanced_rejected():
    X, yb, *_ = _data()
    with pytest.raises(ValueError, match="balanced"):
        FMClassifier(class_weight="balanced").partial_fit(X, yb, classes=np.array([0, 1]))


def test_attributes_and_n_iter_accumulate():
    X, yb, *_ = _data()
    m = FFMClassifier(random_state=0)
    m.partial_fit(X[:30], yb[:30], classes=np.array([0, 1]), field_ids=FIELD_IDS)
    m.partial_fit(X[30:60], yb[30:60])  # field_ids reused from the first call
    m.partial_fit(X[60:], yb[60:])
    assert m.n_iter_ == 3
    assert m.n_features_in_ == 6
    np.testing.assert_array_equal(m.classes_, [0, 1])
    np.testing.assert_array_equal(m.field_ids_, FIELD_IDS)
    assert m.predict(X).shape == (len(X),)


def test_single_class_first_chunk_trains():
    # a chunk may hold a single class as long as `classes` lists them all
    X, yb, *_ = _data()
    zeros = X[yb == 0][:20]
    m = FMClassifier(random_state=0)
    m.partial_fit(zeros, np.zeros(len(zeros), dtype=int), classes=np.array([0, 1]))
    assert m.predict_proba(X).shape == (len(X), 2)


# --- warm_start ---

def test_warm_start_false_is_a_fresh_fit():
    X, yb, *_ = _data()
    a = FMClassifier(random_state=0, max_iter=15).fit(X, yb)
    b = FMClassifier(random_state=0, max_iter=15, warm_start=False)
    b.fit(X, yb)
    b.fit(X, yb)  # a fresh fit must discard the previous one (no state leakage)
    np.testing.assert_array_equal(a.w_, b.w_)
    np.testing.assert_array_equal(a.V_, b.V_)


def test_warm_start_resumes_and_changes_params():
    X, yb, *_ = _data()
    m = FMClassifier(random_state=0, max_iter=10, warm_start=True, dtype="float64")
    m.fit(X, yb)
    before = m.w_.copy()
    m.fit(X, yb)  # continue from the previous solution
    assert not np.allclose(before, m.w_)


@pytest.mark.parametrize("cls", [FMClassifier, FMRegressor, FFMClassifier, FFMRegressor])
def test_warm_start_early_stopping_reproducible(cls):
    X, yb, _, yr = _data(n=150)
    y = yb if cls in (FMClassifier, FFMClassifier) else yr
    kw = dict(random_state=0, max_iter=15, warm_start=True, early_stopping=True, patience=4)
    fkw = dict(field_ids=FIELD_IDS) if cls in (FFMClassifier, FFMRegressor) else {}
    a = cls(**kw)
    a.fit(X, y, **fkw)
    a.fit(X, y, **fkw)
    b = cls(**kw)
    b.fit(X, y, **fkw)
    b.fit(X, y, **fkw)
    np.testing.assert_array_equal(np.asarray(a.w_), np.asarray(b.w_))
    assert 1 <= a.n_iter_ <= 15
