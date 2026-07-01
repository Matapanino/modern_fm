"""FwFM tests (docs/math_spec_fwfm.md): reference correctness, Rust parity at
every layer, the collapse-to-FM property, and FwFMClassifier behavior."""

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import random_sparse_dense_X
from modern_fm import FMClassifier, FwFMClassifier, _backend
from modern_fm._partial import make_opt_state_fwfm
from modern_fm._reference import fm_predict_naive, fwfm_predict, fwfm_predict_naive
from modern_fm._reference_train import (
    fwfm_fit_multiclass_reference,
    fwfm_fit_reference,
    init_fwfm_multiclass_params,
    init_fwfm_params,
    make_row_orders,
)

RTOL = 1e-9
ATOL = 1e-12


def _fwfm_setup(rng, n=40, d=10, n_fields=4, k=3, density=0.4):
    X = random_sparse_dense_X(rng, n, d, density=density)
    field_ids = rng.integers(0, n_fields, size=d)
    return X, field_ids


# --- reference correctness ---------------------------------------------------


def test_naive_matches_vectorized(rng):
    for seed in range(3):
        r2 = np.random.default_rng(seed)
        X, field_ids = _fwfm_setup(r2)
        w0, w, V, R = 0.3, r2.normal(size=10), r2.normal(size=(10, 3)), r2.normal(size=(4, 4))
        np.testing.assert_allclose(
            fwfm_predict_naive(X, field_ids, w0, w, V, R),
            fwfm_predict(X, field_ids, w0, w, V, R),
            rtol=1e-12, atol=1e-12,
        )


def test_hand_computed_example():
    # x = (2, 3), fields (0, 1), v0 = (1, 0), v1 = (1, 1), w = (0.5, -1),
    # w0 = 0.25, r_{01} = 2 -> 0.25 + (1 - 3) + 2 * <v0, v1> * 6 = 10.25
    X = np.array([[2.0, 3.0]])
    R = np.ones((2, 2))
    R[0, 1] = 2.0
    out = fwfm_predict(X, [0, 1], 0.25, [0.5, -1.0], [[1.0, 0.0], [1.0, 1.0]], R)
    assert abs(out[0] - 10.25) < 1e-12


def test_collapse_to_fm_with_ones_r(rng):
    """R = ones makes FwFM exactly a plain FM (docs/math_spec_fwfm.md)."""
    X, field_ids = _fwfm_setup(rng)
    w0, w, V = 0.1, rng.normal(size=10), rng.normal(size=(10, 3))
    np.testing.assert_allclose(
        fwfm_predict(X, field_ids, w0, w, V, np.ones((4, 4))),
        fm_predict_naive(X, w0, w, V),
        rtol=1e-12, atol=1e-12,
    )


def test_zero_row_is_bias_and_single_nonzero_has_no_pairwise(rng):
    X = np.zeros((2, 6))
    X[1, 3] = -1.5
    field_ids = np.arange(6) % 3
    w = rng.normal(size=6)
    out = fwfm_predict(X, field_ids, 0.5, w, rng.normal(size=(6, 2)), rng.normal(size=(3, 3)))
    assert out[0] == 0.5
    assert abs(out[1] - (0.5 + w[3] * -1.5)) < 1e-12


# --- Rust parity --------------------------------------------------------------

needs_rust = pytest.mark.skipif(
    not _backend.has_rust(), reason="modern_fm._rust extension not built"
)


@needs_rust
def test_predict_parity_dense_and_csr(rng):
    X, field_ids = _fwfm_setup(rng)
    w0, w, V, R = 0.3, rng.normal(size=10), rng.normal(size=(10, 3)), rng.normal(size=(4, 4))
    want = fwfm_predict(X, field_ids, w0, w, V, R)
    np.testing.assert_allclose(
        _backend.fwfm_predict(X, field_ids, w0, w, V, R), want, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _backend.fwfm_predict(sp.csr_matrix(X), field_ids, w0, w, V, R),
        want, rtol=1e-12, atol=1e-12,
    )


def _assert_fwfm_params_close(a, b):
    for got, want in zip(a, b):
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


@needs_rust
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("loss", ["logistic", "squared"])
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
def test_training_parity(rng, optimizer, loss, batch_size):
    X, field_ids = _fwfm_setup(rng)
    n = X.shape[0]
    y = rng.normal(size=n) if loss == "squared" else (rng.random(n) > 0.4).astype(np.float64)
    params = init_fwfm_params(rng, 10, 4, 3, 0.05)
    kwargs = dict(
        loss=loss, optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        l1_linear=1e-4 if optimizer == "ftrl" else 0.0,
        l1_factors=1e-4 if optimizer == "ftrl" else 0.0,
        row_orders=make_row_orders(rng, n, epochs=3), batch_size=batch_size,
    )
    _assert_fwfm_params_close(
        _backend.fwfm_fit(X, y, field_ids, params, **kwargs),
        fwfm_fit_reference(X, y, field_ids, params, **kwargs),
    )


@needs_rust
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
def test_multiclass_training_parity(rng, optimizer):
    X, field_ids = _fwfm_setup(rng)
    n = X.shape[0]
    y = rng.integers(0, 3, size=n)
    params = init_fwfm_multiclass_params(rng, 3, 10, 4, 3, 0.05)
    kwargs = dict(
        optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        label_smoothing=0.05, row_orders=make_row_orders(rng, n, epochs=2),
    )
    _assert_fwfm_params_close(
        _backend.fwfm_fit_multiclass(X, y, field_ids, params, **kwargs),
        fwfm_fit_multiclass_reference(X, y, field_ids, params, **kwargs),
    )


@needs_rust
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
def test_es_epoch_handoff_bit_exact(rng, optimizer):
    """Epoch-driven training with the four-group state hand-off equals one
    multi-epoch Rust call bit-for-bit (as for FM/FFM in v0.5)."""
    X, field_ids = _fwfm_setup(rng)
    n = X.shape[0]
    y = (rng.random(n) > 0.4).astype(np.float64)
    params = init_fwfm_params(rng, 10, 4, 3, 0.05)
    ro = make_row_orders(rng, n, epochs=4)
    kwargs = dict(
        loss="logistic", optimizer=optimizer, learning_rate=0.1,
        l2_linear=1e-3, l2_factors=1e-3,
    )
    full = _backend.fwfm_fit(X, y, field_ids, params, row_orders=ro, **kwargs)
    opt_state = make_opt_state_fwfm(optimizer, *params)
    work = params
    for e in range(len(ro)):
        work = _backend.fwfm_fit(
            X, y, field_ids, work, row_orders=ro[e : e + 1], **kwargs, **opt_state
        )
    for got, want in zip(work, full):
        np.testing.assert_array_equal(got, want)
    # the same loop on the reference matches within parity tolerance
    ref_state = make_opt_state_fwfm(optimizer, *params)
    ref = params
    for e in range(len(ro)):
        ref = fwfm_fit_reference(
            X, y, field_ids, ref, row_orders=ro[e : e + 1], **kwargs, **ref_state
        )
    _assert_fwfm_params_close(work, ref)


@needs_rust
@pytest.mark.parametrize("optimizer", ["adagrad", "adam", "ftrl"])
def test_multiclass_es_epoch_handoff_bit_exact(rng, optimizer):
    X, field_ids = _fwfm_setup(rng)
    n = X.shape[0]
    y = rng.integers(0, 3, size=n)
    params = init_fwfm_multiclass_params(rng, 3, 10, 4, 3, 0.05)
    ro = make_row_orders(rng, n, epochs=3)
    kwargs = dict(
        optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        label_smoothing=0.05,
    )
    full = _backend.fwfm_fit_multiclass(X, y, field_ids, params, row_orders=ro, **kwargs)
    opt_state = make_opt_state_fwfm(optimizer, *params)
    work = params
    for e in range(len(ro)):
        work = _backend.fwfm_fit_multiclass(
            X, y, field_ids, work, row_orders=ro[e : e + 1], **kwargs, **opt_state
        )
    for got, want in zip(work, full):
        np.testing.assert_array_equal(got, want)


@needs_rust
def test_state_kwargs_validated(rng):
    X, field_ids = _fwfm_setup(rng, n=10)
    y = (rng.random(10) > 0.5).astype(np.float64)
    params = init_fwfm_params(rng, 10, 4, 3, 0.05)
    ro = make_row_orders(rng, 10, epochs=1)
    adam_state = make_opt_state_fwfm("adam", *params)["adam_state"]
    with pytest.raises(ValueError, match="adam_state"):
        _backend.fwfm_fit(
            X, y, field_ids, params, loss="logistic", optimizer="sgd", learning_rate=0.1,
            l2_linear=0.0, l2_factors=0.0, row_orders=ro, adam_state=adam_state,
        )


# --- estimator ----------------------------------------------------------------


def _class_data(rng, n=200, d=12, n_fields=4, n_classes=2):
    X = rng.normal(size=(n, d))
    X[rng.random(X.shape) > 0.4] = 0.0
    field_ids = np.arange(d) % n_fields
    if n_classes == 2:
        y = (X @ rng.normal(size=d) > 0).astype(int)
    else:
        y = np.digitize(X @ rng.normal(size=d), [-0.5, 0.5])
    return X, y, field_ids


def test_estimator_learns_binary(rng):
    X, y, field_ids = _class_data(rng)
    m = FwFMClassifier(n_factors=4, max_iter=20, random_state=0).fit(X, y, field_ids=field_ids)
    assert (m.predict(X) == y).mean() > 0.85
    assert m.r_.shape == (4, 4)
    p = m.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_estimator_learns_multiclass(rng):
    X, y, field_ids = _class_data(rng, n_classes=3)
    m = FwFMClassifier(n_factors=4, max_iter=20, random_state=0).fit(X, y, field_ids=field_ids)
    assert m.V_.ndim == 3 and m.r_.shape == (3, 4, 4)
    assert (m.predict(X) == y).mean() > 0.7
    np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1.0)


def test_estimator_collapse_property(rng):
    """A fitted FM's parameters loaded into an FwFM with r_ = ones predict
    identically to the FM (the estimator-level collapse check)."""
    X, y, _ = _class_data(rng)
    fm = FMClassifier(n_factors=4, max_iter=10, random_state=0, dtype="float64").fit(X, y)
    fw = FwFMClassifier(n_factors=4, dtype="float64")
    fw.classes_ = fm.classes_
    fw.w0_, fw.w_, fw.V_ = fm.w0_, fm.w_, fm.V_
    fw.field_ids_ = np.arange(X.shape[1], dtype=np.int64)
    fw.n_fields_ = X.shape[1]
    fw.n_features_in_ = X.shape[1]
    fw.n_iter_ = fm.n_iter_
    fw.r_ = np.ones((fw.n_fields_, fw.n_fields_))
    np.testing.assert_allclose(
        fw.decision_function(X), fm.decision_function(X), rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("optimizer", ["adagrad", "adam", "ftrl"])
def test_estimator_early_stopping(rng, optimizer):
    X, y, field_ids = _class_data(rng)
    m = FwFMClassifier(
        n_factors=4, max_iter=40, early_stopping=True, patience=5, optimizer=optimizer,
        random_state=0,
    ).fit(X, y, field_ids=field_ids)
    assert 1 <= m.n_iter_ <= 40
    np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1.0)


def test_estimator_eval_set(rng):
    X, y, field_ids = _class_data(rng)
    m = FwFMClassifier(n_factors=4, max_iter=30, patience=3, random_state=0).fit(
        X[:150], y[:150], field_ids=field_ids, eval_set=(X[150:], y[150:])
    )
    assert m.n_iter_ <= 30


def test_estimator_multiclass_early_stopping(rng):
    X, y, field_ids = _class_data(rng, n_classes=3)
    m = FwFMClassifier(
        n_factors=4, max_iter=25, early_stopping=True, patience=4, random_state=0
    ).fit(X, y, field_ids=field_ids)
    assert 1 <= m.n_iter_ <= 25
    assert m.r_.shape == (3, 4, 4)


def test_partial_fit_matches_fit_binary(rng):
    """Two chunked partial_fit calls equal one partial_fit over the
    concatenated data (both run natural order with an exact optimizer-state
    round-trip) — the FM/FFM partial_fit contract."""
    X, y, field_ids = _class_data(rng)
    kw = dict(n_factors=4, random_state=0, dtype="float64")
    a = FwFMClassifier(**kw)
    a.partial_fit(X[:100], y[:100], classes=[0, 1], field_ids=field_ids)
    a.partial_fit(X[100:], y[100:])
    b = FwFMClassifier(**kw)
    b.partial_fit(X, y, classes=[0, 1], field_ids=field_ids)
    np.testing.assert_allclose(a.w_, b.w_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.V_, b.V_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.r_, b.r_, rtol=1e-12, atol=1e-12)


def test_warm_start_continues(rng):
    X, y, field_ids = _class_data(rng)
    m = FwFMClassifier(n_factors=4, max_iter=5, warm_start=True, random_state=0)
    m.fit(X, y, field_ids=field_ids)
    w_after_5 = m.w_.copy()
    m.fit(X, y, field_ids=field_ids)  # resumes, does not reinit
    assert not np.allclose(m.w_, w_after_5)


def test_default_field_ids_per_column(rng):
    X, y, _ = _class_data(rng)
    m = FwFMClassifier(n_factors=4, max_iter=5, random_state=0).fit(X, y)
    assert m.n_fields_ == X.shape[1]
    np.testing.assert_array_equal(m.field_ids_, np.arange(X.shape[1]))
