"""Parity tests: Rust training kernels vs the Python reference trainers.

Both start from identical initial parameters and identical row_orders, so
results must agree to float64 round-off (summation-order differences only).
"""

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import random_sparse_dense_X
from modern_fm import _backend
from modern_fm._reference_train import (
    ffm_fit_multiclass_reference,
    ffm_fit_reference,
    fm_fit_multiclass_reference,
    fm_fit_reference,
    init_ffm_multiclass_params,
    init_ffm_params,
    init_fm_multiclass_params,
    init_fm_params,
    make_row_orders,
)

pytestmark = pytest.mark.skipif(
    not _backend.has_rust(), reason="modern_fm._rust extension not built"
)

RTOL = 1e-9
ATOL = 1e-12


def _assert_params_close(a, b):
    np.testing.assert_allclose(a[0], b[0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(a[1], b[1], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(a[2], b[2], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("loss", ["logistic", "squared"])
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_fm_training_parity(rng, loss, optimizer):
    n, d, k = 40, 10, 3
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = rng.normal(size=n) if loss == "squared" else (rng.random(n) > 0.4).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    kwargs = dict(
        loss=loss, optimizer=optimizer, learning_rate=0.1,
        l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=3),
    )
    _assert_params_close(
        _backend.fm_fit(X, y, params, **kwargs),
        fm_fit_reference(X, y, params, **kwargs),
    )


@pytest.mark.parametrize("loss", ["logistic", "squared"])
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_ffm_training_parity(rng, loss, optimizer):
    n, d, n_fields, k = 30, 8, 3, 2
    X = random_sparse_dense_X(rng, n, d, density=0.5)
    y = rng.normal(size=n) if loss == "squared" else (rng.random(n) > 0.5).astype(np.float64)
    field_ids = rng.integers(0, n_fields, size=d)
    params = init_ffm_params(rng, d, n_fields, k, 0.05)
    kwargs = dict(
        loss=loss, optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=2),
    )
    _assert_params_close(
        _backend.ffm_fit(X, y, field_ids, params, **kwargs),
        ffm_fit_reference(X, y, field_ids, params, **kwargs),
    )


def test_fm_training_parity_csr_input(rng):
    n, d, k = 30, 8, 2
    X = random_sparse_dense_X(rng, n, d, density=0.3)
    y = (rng.random(n) > 0.5).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    kwargs = dict(
        loss="logistic", optimizer="adagrad", learning_rate=0.1,
        l2_linear=0.0, l2_factors=0.0, row_orders=make_row_orders(rng, n, epochs=2),
    )
    _assert_params_close(
        _backend.fm_fit(sp.csr_matrix(X), y, params, **kwargs),
        fm_fit_reference(X, y, params, **kwargs),
    )


def test_fit_does_not_mutate_input_params(rng):
    n, d, k = 10, 5, 2
    X = random_sparse_dense_X(rng, n, d)
    y = (rng.random(n) > 0.5).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    w_before, V_before = params[1].copy(), params[2].copy()
    _backend.fm_fit(
        X, y, params, loss="logistic", optimizer="sgd", learning_rate=0.1,
        l2_linear=0.0, l2_factors=0.0, row_orders=make_row_orders(rng, n, epochs=1),
    )
    np.testing.assert_array_equal(params[1], w_before)
    np.testing.assert_array_equal(params[2], V_before)


def test_rust_fit_rejects_bad_row_orders(rng):
    n, d, k = 6, 4, 2
    X = random_sparse_dense_X(rng, n, d)
    y = np.zeros(n)
    params = init_fm_params(rng, d, k, 0.05)
    bad = np.full((1, n), n, dtype=np.int64)  # out of range
    with pytest.raises(ValueError):
        _backend.fm_fit(
            X, y, params, loss="logistic", optimizer="sgd", learning_rate=0.1,
            l2_linear=0.0, l2_factors=0.0, row_orders=bad,
        )


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_fm_training_parity_sample_weight(rng, optimizer):
    n, d, k = 35, 9, 3
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = (rng.random(n) > 0.5).astype(np.float64)
    sw = rng.uniform(0.1, 3.0, size=n)
    params = init_fm_params(rng, d, k, 0.05)
    kwargs = dict(
        loss="logistic", optimizer=optimizer, learning_rate=0.1,
        l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=3), sample_weight=sw,
    )
    _assert_params_close(
        _backend.fm_fit(X, y, params, **kwargs),
        fm_fit_reference(X, y, params, **kwargs),
    )


@pytest.mark.parametrize("loss", ["logistic", "squared"])
def test_ffm_training_parity_sample_weight(rng, loss):
    n, d, n_fields, k = 25, 7, 3, 2
    X = random_sparse_dense_X(rng, n, d, density=0.5)
    y = rng.normal(size=n) if loss == "squared" else (rng.random(n) > 0.5).astype(np.float64)
    field_ids = rng.integers(0, n_fields, size=d)
    sw = rng.uniform(0.1, 3.0, size=n)
    params = init_ffm_params(rng, d, n_fields, k, 0.05)
    kwargs = dict(
        loss=loss, optimizer="adagrad", learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=2), sample_weight=sw,
    )
    _assert_params_close(
        _backend.ffm_fit(X, y, field_ids, params, **kwargs),
        ffm_fit_reference(X, y, field_ids, params, **kwargs),
    )


def test_rust_fit_rejects_bad_sample_weight_length(rng):
    n, d, k = 6, 4, 2
    X = random_sparse_dense_X(rng, n, d)
    y = np.zeros(n)
    params = init_fm_params(rng, d, k, 0.05)
    with pytest.raises(ValueError):
        _backend.fm_fit(
            X, y, params, loss="logistic", optimizer="sgd", learning_rate=0.1,
            l2_linear=0.0, l2_factors=0.0, row_orders=make_row_orders(rng, n, epochs=1),
            sample_weight=np.ones(n + 1),
        )


@pytest.mark.parametrize("n_classes", [3, 4])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_fm_multiclass_training_parity(rng, optimizer, label_smoothing, n_classes):
    n, d, k = 40, 10, 3
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = rng.integers(0, n_classes, size=n)
    params = init_fm_multiclass_params(rng, n_classes, d, k, 0.05)
    kwargs = dict(
        optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=3), label_smoothing=label_smoothing,
    )
    _assert_params_close(
        _backend.fm_fit_multiclass(X, y, params, **kwargs),
        fm_fit_multiclass_reference(X, y, params, **kwargs),
    )


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_fm_multiclass_training_parity_sample_weight(rng, optimizer):
    n, d, k, n_classes = 35, 9, 3, 4
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = rng.integers(0, n_classes, size=n)
    sw = rng.uniform(0.1, 3.0, size=n)
    params = init_fm_multiclass_params(rng, n_classes, d, k, 0.05)
    kwargs = dict(
        optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=3), sample_weight=sw,
    )
    _assert_params_close(
        _backend.fm_fit_multiclass(X, y, params, **kwargs),
        fm_fit_multiclass_reference(X, y, params, **kwargs),
    )


def test_fm_multiclass_training_parity_csr_input(rng):
    n, d, k, n_classes = 30, 8, 2, 3
    X = random_sparse_dense_X(rng, n, d, density=0.3)
    y = rng.integers(0, n_classes, size=n)
    params = init_fm_multiclass_params(rng, n_classes, d, k, 0.05)
    kwargs = dict(
        optimizer="adagrad", learning_rate=0.1, l2_linear=0.0, l2_factors=0.0,
        row_orders=make_row_orders(rng, n, epochs=2),
    )
    _assert_params_close(
        _backend.fm_fit_multiclass(sp.csr_matrix(X), y, params, **kwargs),
        fm_fit_multiclass_reference(X, y, params, **kwargs),
    )


def test_fm_multiclass_fit_does_not_mutate_input_params(rng):
    n, d, k, n_classes = 10, 5, 2, 3
    X = random_sparse_dense_X(rng, n, d)
    y = rng.integers(0, n_classes, size=n)
    params = init_fm_multiclass_params(rng, n_classes, d, k, 0.05)
    before = [p.copy() for p in params]
    _backend.fm_fit_multiclass(
        X, y, params, optimizer="adagrad", learning_rate=0.1, l2_linear=0.0,
        l2_factors=0.0, row_orders=make_row_orders(rng, n, epochs=1),
    )
    for p, p0 in zip(params, before):
        np.testing.assert_array_equal(p, p0)


def test_rust_multiclass_fit_rejects_bad_row_orders(rng):
    n, d, k, n_classes = 6, 4, 2, 3
    X = random_sparse_dense_X(rng, n, d)
    y = rng.integers(0, n_classes, size=n)
    params = init_fm_multiclass_params(rng, n_classes, d, k, 0.05)
    bad = np.full((1, n), n, dtype=np.int64)  # out of range
    with pytest.raises(ValueError):
        _backend.fm_fit_multiclass(
            X, y, params, optimizer="sgd", learning_rate=0.1, l2_linear=0.0,
            l2_factors=0.0, row_orders=bad,
        )


def test_rust_multiclass_fit_rejects_bad_class_index(rng):
    n, d, k, n_classes = 6, 4, 2, 3
    X = random_sparse_dense_X(rng, n, d)
    y = rng.integers(0, n_classes, size=n)
    y[0] = n_classes  # out of range [0, n_classes)
    params = init_fm_multiclass_params(rng, n_classes, d, k, 0.05)
    with pytest.raises(ValueError):
        _backend.fm_fit_multiclass(
            X, y, params, optimizer="sgd", learning_rate=0.1, l2_linear=0.0,
            l2_factors=0.0, row_orders=make_row_orders(rng, n, epochs=1),
        )


def test_adam_nondefault_hyperparams_parity(rng):
    """Custom beta_1/beta_2/epsilon must reach both backends (FM, FFM, multiclass),
    not be hard-coded — parity would break if either path ignored them. A
    divergence check confirms the custom values actually change the result, so a
    'both silently default' false pass cannot hide a plumbing bug."""
    betas = dict(beta_1=0.8, beta_2=0.9, epsilon=1e-6)
    n, d, k = 40, 10, 3
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    ro = make_row_orders(rng, n, epochs=3)
    y = (rng.random(n) > 0.5).astype(np.float64)

    # FM binary: parity at custom betas, and custom betas differ from defaults.
    params = init_fm_params(rng, d, k, 0.05)
    base = dict(loss="logistic", optimizer="adam", learning_rate=0.1,
                l2_linear=1e-3, l2_factors=1e-3, row_orders=ro)
    rust_custom = _backend.fm_fit(X, y, params, **base, **betas)
    _assert_params_close(rust_custom, fm_fit_reference(X, y, params, **base, **betas))
    ref_default = fm_fit_reference(X, y, params, **base)
    assert not np.allclose(rust_custom[2], ref_default[2]), "betas had no effect"

    # FFM
    n_fields = 3
    field_ids = rng.integers(0, n_fields, size=d)
    pf = init_ffm_params(rng, d, n_fields, k, 0.05)
    kwf = dict(loss="logistic", optimizer="adam", learning_rate=0.1, l2_linear=1e-3,
               l2_factors=1e-3, row_orders=ro, **betas)
    _assert_params_close(
        _backend.ffm_fit(X, y, field_ids, pf, **kwf),
        ffm_fit_reference(X, y, field_ids, pf, **kwf),
    )

    # FM multiclass (with label smoothing)
    n_classes = 4
    ym = rng.integers(0, n_classes, size=n)
    pm = init_fm_multiclass_params(rng, n_classes, d, k, 0.05)
    kwm = dict(optimizer="adam", learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
               row_orders=ro, label_smoothing=0.1, **betas)
    _assert_params_close(
        _backend.fm_fit_multiclass(X, ym, pm, **kwm),
        fm_fit_multiclass_reference(X, ym, pm, **kwm),
    )


# --- mini-batch (docs/optimization_spec.md, "Mini-batch") ------------------
# batch_size 1 (per-row), 4 (several batches, partial last batch when n % 4),
# and >= n (one full-batch averaged step per epoch).
BATCH_SIZES = [1, 4, 1000]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
@pytest.mark.parametrize("loss", ["logistic", "squared"])
def test_fm_minibatch_parity(rng, loss, optimizer, batch_size):
    n, d, k = 38, 10, 3  # 38 % 4 != 0 -> exercises a short final batch
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = rng.normal(size=n) if loss == "squared" else (rng.random(n) > 0.4).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    kwargs = dict(
        loss=loss, optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=3), batch_size=batch_size,
    )
    _assert_params_close(
        _backend.fm_fit(X, y, params, **kwargs),
        fm_fit_reference(X, y, params, **kwargs),
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
@pytest.mark.parametrize("loss", ["logistic", "squared"])
def test_ffm_minibatch_parity(rng, loss, optimizer, batch_size):
    n, d, n_fields, k = 30, 8, 3, 2
    X = random_sparse_dense_X(rng, n, d, density=0.5)
    y = rng.normal(size=n) if loss == "squared" else (rng.random(n) > 0.5).astype(np.float64)
    field_ids = rng.integers(0, n_fields, size=d)
    params = init_ffm_params(rng, d, n_fields, k, 0.05)
    kwargs = dict(
        loss=loss, optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=2), batch_size=batch_size,
    )
    _assert_params_close(
        _backend.ffm_fit(X, y, field_ids, params, **kwargs),
        ffm_fit_reference(X, y, field_ids, params, **kwargs),
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_fm_multiclass_minibatch_parity(rng, optimizer, batch_size):
    n, d, k, n_classes = 38, 10, 3, 4
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = rng.integers(0, n_classes, size=n)
    params = init_fm_multiclass_params(rng, n_classes, d, k, 0.05)
    kwargs = dict(
        optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=3), label_smoothing=0.1,
        batch_size=batch_size,
    )
    _assert_params_close(
        _backend.fm_fit_multiclass(X, y, params, **kwargs),
        fm_fit_multiclass_reference(X, y, params, **kwargs),
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
def test_ffm_multiclass_parity(rng, optimizer, batch_size):
    n, d, n_fields, k, n_classes = 38, 9, 3, 2, 4
    X = random_sparse_dense_X(rng, n, d, density=0.5)
    y = rng.integers(0, n_classes, size=n)
    field_ids = rng.integers(0, n_fields, size=d)
    params = init_ffm_multiclass_params(rng, n_classes, d, n_fields, k, 0.05)
    l1 = 0.01 if optimizer == "ftrl" else 0.0
    kwargs = dict(
        optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3, l1_linear=l1,
        l1_factors=l1, ftrl_beta=1.0, row_orders=make_row_orders(rng, n, epochs=3),
        label_smoothing=0.1, batch_size=batch_size,
    )
    _assert_params_close(
        _backend.ffm_fit_multiclass(X, y, field_ids, params, **kwargs),
        ffm_fit_multiclass_reference(X, y, field_ids, params, **kwargs),
    )


def test_minibatch_actually_changes_result(rng):
    """batch_size > 1 must change the result vs batch_size=1, so a silently
    ignored batch_size can't make the parity tests above pass vacuously."""
    n, d, k = 30, 8, 3
    X = random_sparse_dense_X(rng, n, d, density=0.5)
    y = (rng.random(n) > 0.5).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    base = dict(loss="logistic", optimizer="adagrad", learning_rate=0.1,
                l2_linear=1e-3, l2_factors=1e-3, row_orders=make_row_orders(rng, n, epochs=3))
    bs1 = _backend.fm_fit(X, y, params, **base, batch_size=1)
    bsN = _backend.fm_fit(X, y, params, **base, batch_size=n)
    assert not np.allclose(bs1[2], bsN[2]), "batch_size had no effect on training"


# --- rayon parallelism (docs/optimization_spec.md, "Parallelism") -----------
# n_jobs>1 splits each batch across threads; it differs from the serial path
# only in float summation order (loose tolerance), and is bit-reproducible for a
# fixed n_jobs. n_jobs only acts when batch_size > 1 (a batch_size=1 batch is one
# chunk), so these use batch_size > 1.
PAR_TOL = dict(rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n_jobs", [2, 4])
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_fm_parallel_matches_serial(rng, optimizer, n_jobs):
    n, d, k = 64, 12, 3
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = (rng.random(n) > 0.5).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    kw = dict(
        loss="logistic", optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3,
        l2_factors=1e-3, row_orders=make_row_orders(rng, n, epochs=3), batch_size=8,
    )
    serial = _backend.fm_fit(X, y, params, **kw, n_jobs=1)
    par = _backend.fm_fit(X, y, params, **kw, n_jobs=n_jobs)
    for a, b in zip(serial, par):
        np.testing.assert_allclose(a, b, **PAR_TOL)


@pytest.mark.parametrize("n_jobs", [2, 4])
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
@pytest.mark.parametrize("loss", ["logistic", "squared"])
def test_ffm_parallel_matches_serial(rng, loss, optimizer, n_jobs):
    n, d, n_fields, k = 50, 10, 3, 2
    X = random_sparse_dense_X(rng, n, d, density=0.6)
    y = rng.normal(size=n) if loss == "squared" else (rng.random(n) > 0.5).astype(np.float64)
    field_ids = rng.integers(0, n_fields, size=d)
    params = init_ffm_params(rng, d, n_fields, k, 0.05)
    kw = dict(
        loss=loss, optimizer=optimizer, learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        row_orders=make_row_orders(rng, n, epochs=3), batch_size=8,
    )
    serial = _backend.ffm_fit(X, y, field_ids, params, **kw, n_jobs=1)
    par = _backend.ffm_fit(X, y, field_ids, params, **kw, n_jobs=n_jobs)
    for a, b in zip(serial, par):
        np.testing.assert_allclose(a, b, **PAR_TOL)


# --- FTRL-Proximal (docs/optimization_spec.md, "Optimizers") ----------------
@pytest.mark.parametrize("l1", [0.0, 0.02])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("loss", ["logistic", "squared"])
def test_fm_ftrl_parity(rng, loss, batch_size, l1):
    n, d, k = 40, 10, 3
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = rng.normal(size=n) if loss == "squared" else (rng.random(n) > 0.4).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    kw = dict(
        loss=loss, optimizer="ftrl", learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        l1_linear=l1, l1_factors=l1, ftrl_beta=1.0, row_orders=make_row_orders(rng, n, epochs=3),
        batch_size=batch_size,
    )
    _assert_params_close(
        _backend.fm_fit(X, y, params, **kw),
        fm_fit_reference(X, y, params, **kw),
    )


@pytest.mark.parametrize("l1", [0.0, 0.02])
@pytest.mark.parametrize("loss", ["logistic", "squared"])
def test_ffm_ftrl_parity(rng, loss, l1):
    n, d, n_fields, k = 30, 8, 3, 2
    X = random_sparse_dense_X(rng, n, d, density=0.5)
    y = rng.normal(size=n) if loss == "squared" else (rng.random(n) > 0.5).astype(np.float64)
    field_ids = rng.integers(0, n_fields, size=d)
    params = init_ffm_params(rng, d, n_fields, k, 0.05)
    kw = dict(
        loss=loss, optimizer="ftrl", learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        l1_linear=l1, l1_factors=l1, ftrl_beta=1.0,
        row_orders=make_row_orders(rng, n, epochs=2), batch_size=4,
    )
    _assert_params_close(
        _backend.ffm_fit(X, y, field_ids, params, **kw),
        ffm_fit_reference(X, y, field_ids, params, **kw),
    )


@pytest.mark.parametrize("l1", [0.0, 0.02])
def test_fm_multiclass_ftrl_parity(rng, l1):
    n, d, k, n_classes = 40, 10, 3, 4
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = rng.integers(0, n_classes, size=n)
    params = init_fm_multiclass_params(rng, n_classes, d, k, 0.05)
    kw = dict(
        optimizer="ftrl", learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3, l1_linear=l1,
        l1_factors=l1, ftrl_beta=1.0, row_orders=make_row_orders(rng, n, epochs=3),
        label_smoothing=0.1, batch_size=5,
    )
    _assert_params_close(
        _backend.fm_fit_multiclass(X, y, params, **kw),
        fm_fit_multiclass_reference(X, y, params, **kw),
    )


def test_fm_ftrl_parallel_matches_serial(rng):
    n, d, k = 64, 12, 3
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = (rng.random(n) > 0.5).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    kw = dict(
        loss="logistic", optimizer="ftrl", learning_rate=0.1, l2_linear=1e-3, l2_factors=1e-3,
        l1_linear=0.02, l1_factors=0.02, ftrl_beta=1.0,
        row_orders=make_row_orders(rng, n, epochs=3), batch_size=8,
    )
    serial = _backend.fm_fit(X, y, params, **kw, n_jobs=1)
    par = _backend.fm_fit(X, y, params, **kw, n_jobs=4)
    for a, b in zip(serial, par):
        np.testing.assert_allclose(a, b, **PAR_TOL)


def test_ftrl_l1_sparsifies_linear_weights(rng):
    """FTRL with L1 drives irrelevant linear weights to exact zero; AdaGrad never
    produces exact zeros. The label depends only on the first few features."""
    n, d, k = 300, 25, 3
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    true_w = np.zeros(d)
    true_w[:3] = rng.normal(size=3) * 3.0
    y = (X @ true_w + rng.normal(scale=0.1, size=n) > 0).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    ro = make_row_orders(rng, n, epochs=2)
    _, w_ftrl, _ = _backend.fm_fit(
        X, y, params, loss="logistic", optimizer="ftrl", learning_rate=0.2,
        l2_linear=0.0, l2_factors=0.0, l1_linear=5.0, l1_factors=0.0, ftrl_beta=1.0, row_orders=ro,
    )
    _, w_ada, _ = _backend.fm_fit(
        X, y, params, loss="logistic", optimizer="adagrad", learning_rate=0.2,
        l2_linear=0.0, l2_factors=0.0, row_orders=ro,
    )
    assert (w_ftrl == 0.0).sum() > 0  # L1 zeroed several weights
    assert (w_ada == 0.0).sum() == 0  # AdaGrad never produces exact zeros


def test_parallel_is_reproducible_for_fixed_n_jobs(rng):
    """Same n_jobs across two runs is bit-identical: contiguous chunking + a
    fixed-order reduction make the float operations deterministic."""
    n, d, k = 64, 12, 3
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = (rng.random(n) > 0.5).astype(np.float64)
    params = init_fm_params(rng, d, k, 0.05)
    kw = dict(
        loss="logistic", optimizer="adam", learning_rate=0.1, l2_linear=1e-3,
        l2_factors=1e-3, row_orders=make_row_orders(rng, n, epochs=3), batch_size=8, n_jobs=4,
    )
    a = _backend.fm_fit(X, y, params, **kw)
    b = _backend.fm_fit(X, y, params, **kw)
    np.testing.assert_array_equal(a[2], b[2])  # exact, not just close
    np.testing.assert_array_equal(a[1], b[1])
