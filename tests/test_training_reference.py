"""Tests for the pure-Python reference trainers (ground truth for Rust)."""

import numpy as np
import pytest
from modern_fm._reference import ffm_predict, fm_predict_fast
from modern_fm._reference_train import ffm_train, fm_train
from modern_fm.losses import logistic_loss, squared_loss

LOG2 = np.log(2.0)  # initial logistic loss with near-zero parameters


def _binary_data(seed=0, n=80, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = rng.normal(size=d)
    y = (X @ beta > 0).astype(np.float64)
    return X, y


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_fm_logistic_loss_decreases(optimizer):
    X, y = _binary_data()
    w0, w, V = fm_train(
        X, y, optimizer=optimizer, learning_rate=0.1, epochs=15, n_factors=3, random_state=0
    )
    assert logistic_loss(y, fm_predict_fast(X, w0, w, V)) < 0.7 * LOG2


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
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


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
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


def test_unknown_loss_and_optimizer_raise():
    X, y = _binary_data(n=10, d=3)
    with pytest.raises(ValueError, match="loss"):
        fm_train(X, y, loss="hinge")
    with pytest.raises(ValueError, match="optimizer"):
        fm_train(X, y, optimizer="rmsprop")
