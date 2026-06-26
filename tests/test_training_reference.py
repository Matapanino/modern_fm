"""Tests for the pure-Python reference trainers (ground truth for Rust)."""

import numpy as np
import pytest
from modern_fm._reference import ffm_predict, fm_predict_fast
from modern_fm._reference_train import (
    ffm_train,
    fm_fit_multiclass_reference,
    fm_fit_reference,
    fm_train,
    init_fm_multiclass_params,
    init_fm_params,
    make_row_orders,
    new_adam_state,
)
from modern_fm.losses import logistic_loss, squared_loss

LOG2 = np.log(2.0)  # initial logistic loss with near-zero parameters


def _binary_data(seed=0, n=80, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = rng.normal(size=d)
    y = (X @ beta > 0).astype(np.float64)
    return X, y


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
def test_fm_logistic_loss_decreases(optimizer):
    X, y = _binary_data()
    w0, w, V = fm_train(
        X, y, optimizer=optimizer, learning_rate=0.1, epochs=15, n_factors=3, random_state=0
    )
    assert logistic_loss(y, fm_predict_fast(X, w0, w, V)) < 0.7 * LOG2


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
def test_fm_squared_loss_decreases(optimizer):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 5))
    y = X @ rng.normal(size=5)
    w0, w, V = fm_train(
        X, y, loss="squared", optimizer=optimizer, learning_rate=0.05, epochs=25,
        n_factors=3, random_state=0,
    )
    baseline = squared_loss(y, np.full_like(y, y.mean()))
    assert squared_loss(y, fm_predict_fast(X, w0, w, V)) < 0.3 * baseline


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
def test_ffm_logistic_loss_decreases(optimizer):
    X, y = _binary_data(n=60, d=6)
    field_ids = np.array([0, 0, 1, 1, 2, 2])
    w0, w, V = ffm_train(
        X, y, field_ids, optimizer=optimizer, learning_rate=0.1, epochs=10,
        n_factors=2, random_state=0,
    )
    assert logistic_loss(y, ffm_predict(X, field_ids, w0, w, V)) < 0.7 * LOG2


def test_fm_same_seed_identical_params():
    X, y = _binary_data()
    r1 = fm_train(X, y, epochs=5, random_state=42)
    r2 = fm_train(X, y, epochs=5, random_state=42)
    assert r1[0] == r2[0]
    np.testing.assert_array_equal(r1[1], r2[1])
    np.testing.assert_array_equal(r1[2], r2[2])


def test_fm_different_seed_differs():
    X, y = _binary_data()
    r1 = fm_train(X, y, epochs=5, random_state=0)
    r2 = fm_train(X, y, epochs=5, random_state=1)
    assert not np.array_equal(r1[2], r2[2])


def test_ffm_same_seed_identical_params():
    X, y = _binary_data(n=40, d=4)
    field_ids = np.array([0, 0, 1, 1])
    r1 = ffm_train(X, y, field_ids, epochs=3, random_state=7)
    r2 = ffm_train(X, y, field_ids, epochs=3, random_state=7)
    assert r1[0] == r2[0]
    np.testing.assert_array_equal(r1[1], r2[1])
    np.testing.assert_array_equal(r1[2], r2[2])


@pytest.mark.parametrize("batch_size", [1, 8, 80])
def test_fm_minibatch_loss_decreases(batch_size):
    # 80 rows: batch_size 80 is a single full-batch step per epoch.
    X, y = _binary_data()
    rng = np.random.default_rng(0)
    params = init_fm_params(rng, X.shape[1], 3, 0.02)
    init_loss = logistic_loss(y, fm_predict_fast(X, *params))
    ro = make_row_orders(rng, X.shape[0], epochs=40)
    w0, w, V = fm_fit_reference(
        X, y, params, loss="logistic", optimizer="adagrad", learning_rate=0.1,
        row_orders=ro, batch_size=batch_size,
    )
    assert logistic_loss(y, fm_predict_fast(X, w0, w, V)) < init_loss


def test_fm_full_batch_is_one_averaged_gradient_step():
    """One full-batch SGD epoch equals one manual averaged-gradient step from the
    init parameters (docs/optimization_spec.md, "Mini-batch")."""
    rng = np.random.default_rng(0)
    n, d, lr = 50, 6, 0.05
    X = rng.normal(size=(n, d))  # dense: every feature touched by every row
    y = (rng.normal(size=n) > 0).astype(np.float64)
    params = init_fm_params(rng, d, 3, 0.02)
    ro = make_row_orders(rng, n, epochs=1, shuffle=False)
    w0, w, _ = fm_fit_reference(
        X, y, params, loss="logistic", optimizer="sgd", learning_rate=lr,
        row_orders=ro, batch_size=n,
    )
    w0_0, w_0, V_0 = params
    g = 1.0 / (1.0 + np.exp(-fm_predict_fast(X, w0_0, w_0, V_0))) - y  # dL/ds at init
    assert np.isclose(w0, w0_0 - lr * g.mean())
    np.testing.assert_allclose(w, w_0 - lr * (g[:, None] * X).mean(axis=0))


def test_adam_state_roundtrip_matches_single_call():
    """Per-epoch Adam training with `adam_state` round-trip equals one multi-epoch
    call — the early-stopping moment hand-off preserves (m, v, t) exactly."""
    rng = np.random.default_rng(0)
    n, d, k, epochs = 30, 8, 3, 6
    X = rng.normal(size=(n, d))
    y = (X @ rng.normal(size=d) > 0).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    ro = make_row_orders(rng, n, epochs=epochs)
    one = fm_fit_reference(
        X, y, params, optimizer="adam", learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        row_orders=ro,
    )
    w0, w, V = params
    adam_state = new_adam_state(params[0], params[1], params[2])
    for e in range(epochs):
        w0, w, V = fm_fit_reference(
            X, y, (w0, w, V), optimizer="adam", learning_rate=0.1, l2_linear=1e-3,
            l2_factors=1e-3, row_orders=ro[e : e + 1], adam_state=adam_state,
        )
    assert np.isclose(one[0], w0)
    np.testing.assert_allclose(one[1], w)
    np.testing.assert_allclose(one[2], V)


@pytest.mark.parametrize("optimizer", ["adagrad", "adam"])
def test_multiclass_state_roundtrip_matches_single_call(optimizer):
    """Per-epoch multiclass training with state/adam_state round-trip equals one
    multi-epoch call — the early-stopping hand-off preserves per-class state."""
    rng = np.random.default_rng(0)
    n, d, k, n_classes, epochs = 30, 8, 3, 4, 5
    X = rng.normal(size=(n, d))
    y = rng.integers(0, n_classes, size=n)
    params = init_fm_multiclass_params(rng, n_classes, d, k, 0.05)
    ro = make_row_orders(rng, n, epochs=epochs)
    kw = dict(optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
              label_smoothing=0.1)
    one = fm_fit_multiclass_reference(X, y, params, row_orders=ro, **kw)
    if optimizer == "adam":
        carry = dict(adam_state=new_adam_state(params[0], params[1], params[2]))
    else:
        carry = dict(state=[np.zeros_like(p) for p in params])
    w0, w, V = params
    for e in range(epochs):
        w0, w, V = fm_fit_multiclass_reference(
            X, y, (w0, w, V), row_orders=ro[e : e + 1], **kw, **carry
        )
    np.testing.assert_allclose(one[0], w0)
    np.testing.assert_allclose(one[1], w)
    np.testing.assert_allclose(one[2], V)


def test_unknown_loss_and_optimizer_raise():
    X, y = _binary_data(n=10, d=3)
    with pytest.raises(ValueError, match="loss"):
        fm_train(X, y, loss="hinge")
    with pytest.raises(ValueError, match="optimizer"):
        fm_train(X, y, optimizer="rmsprop")
