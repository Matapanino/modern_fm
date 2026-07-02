"""CUDA FM/FFM/FwFM prediction + binary and multiclass training parity —
every backend cell (docs/gpu_backend_plan.md milestones 1-6).

Skipped entirely without a CUDA build + device — run on a real GPU per
docs/cuda_validation_runbook.md before merging CUDA kernel changes.
Prediction is tolerance-based at rtol/atol 1e-10; training compares final
predictions at rtol 1e-7 / atol 1e-8 (plan-doc tolerances): CUDA atomics make
the gradient summation order nondeterministic run-to-run, so training parity
can never be bit-exact.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import random_sparse_dense_X
from modern_fm import (
    FFMClassifier,
    FFMRegressor,
    FMClassifier,
    FMRegressor,
    FwFMClassifier,
    _backend,
)
from modern_fm._reference import ffm_predict, fm_predict_fast, fwfm_predict

pytestmark = pytest.mark.skipif(
    not _backend.has_cuda(), reason="requires a cuda-backend build and a CUDA device"
)

RTOL = 1e-10
ATOL = 1e-10


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("k", [1, 4, 16, 64])
def test_fm_prediction_parity(seed, k):
    rng = np.random.default_rng(seed)
    n, d = 200, 50
    X = random_sparse_dense_X(rng, n, d, density=0.2)
    w0, w, V = rng.normal(), rng.normal(size=d), rng.normal(size=(d, k))
    want = fm_predict_fast(X, w0, w, V)
    got = _backend.fm_predict_fast(X, w0, w, V, backend="cuda")
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
    got_csr = _backend.fm_predict_fast(sp.csr_matrix(X), w0, w, V, backend="cuda")
    np.testing.assert_allclose(got_csr, want, rtol=RTOL, atol=ATOL)
    # and vs the Rust CPU kernel
    cpu = _backend.fm_predict_fast(X, w0, w, V)
    np.testing.assert_allclose(got, cpu, rtol=RTOL, atol=ATOL)


def test_empty_rows_return_bias():
    X = sp.csr_matrix((4, 6))
    out = _backend.fm_predict_fast(X, 1.5, np.zeros(6), np.ones((6, 3)), backend="cuda")
    np.testing.assert_allclose(out, np.full(4, 1.5), rtol=RTOL, atol=ATOL)


def test_single_nonzero_has_no_pairwise():
    rng = np.random.default_rng(0)
    X = np.zeros((3, 5))
    X[1, 2] = -2.0
    w = rng.normal(size=5)
    out = _backend.fm_predict_fast(X, 0.5, w, rng.normal(size=(5, 4)), backend="cuda")
    np.testing.assert_allclose(out[0], 0.5, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(out[1], 0.5 + w[2] * -2.0, rtol=RTOL, atol=ATOL)


def test_estimator_inference_flow():
    """fit on rust_cpu, set_params(backend='cuda') for inference — the
    documented usage until CUDA training lands."""
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 300, 20, density=0.3)
    y = (X @ rng.normal(size=20) > 0).astype(int)
    m = FMClassifier(n_factors=8, max_iter=10, random_state=0).fit(X, y)
    cpu_scores = m.decision_function(X)
    m.set_params(backend="cuda")
    cuda_scores = m.decision_function(X)
    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=1e-6, atol=1e-6)
    assert m.predict_proba(X).shape == (300, 2)


def _random_ffm(rng, d, n_fields, k):
    w0 = float(rng.normal())
    w = rng.normal(size=d)
    V = rng.normal(size=(d, n_fields, k))
    field_ids = rng.integers(0, n_fields, size=d)
    return w0, w, V, field_ids


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("k", [1, 4, 16])
@pytest.mark.parametrize("n_fields", [1, 4, 8])
def test_ffm_prediction_parity(seed, k, n_fields):
    rng = np.random.default_rng(seed)
    n, d = 200, 50
    X = random_sparse_dense_X(rng, n, d, density=0.2)
    w0, w, V, field_ids = _random_ffm(rng, d, n_fields, k)
    want = ffm_predict(X, field_ids, w0, w, V)
    got = _backend.ffm_predict(X, field_ids, w0, w, V, backend="cuda")
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
    got_csr = _backend.ffm_predict(sp.csr_matrix(X), field_ids, w0, w, V, backend="cuda")
    np.testing.assert_allclose(got_csr, want, rtol=RTOL, atol=ATOL)
    # and vs the Rust CPU kernel
    cpu = _backend.ffm_predict(X, field_ids, w0, w, V)
    np.testing.assert_allclose(got, cpu, rtol=RTOL, atol=ATOL)


def test_ffm_empty_rows_return_bias():
    X = sp.csr_matrix((4, 6))
    field_ids = np.zeros(6, dtype=np.int64)
    out = _backend.ffm_predict(X, field_ids, 1.5, np.zeros(6), np.ones((6, 2, 3)), backend="cuda")
    np.testing.assert_allclose(out, np.full(4, 1.5), rtol=RTOL, atol=ATOL)


def test_ffm_single_nonzero_has_no_pairwise():
    rng = np.random.default_rng(0)
    X = np.zeros((3, 5))
    X[1, 2] = -2.0
    w = rng.normal(size=5)
    V = rng.normal(size=(5, 3, 4))
    field_ids = rng.integers(0, 3, size=5)
    out = _backend.ffm_predict(X, field_ids, 0.5, w, V, backend="cuda")
    np.testing.assert_allclose(out[0], 0.5, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(out[1], 0.5 + w[2] * -2.0, rtol=RTOL, atol=ATOL)


def test_ffm_large_row_grid_stride():
    """A single row with z=2048 nonzeros: pairs >> blockDim exercises the
    b-strided pair loop. Compared vs Rust CPU (itself reference-parity-tested);
    the NumPy reference materializes z*z*k per row and would blow memory here."""
    rng = np.random.default_rng(0)
    d = 2048
    X = sp.csr_matrix(rng.normal(size=(1, d)))
    w0, w, V, field_ids = _random_ffm(rng, d, 4, 4)
    cpu = _backend.ffm_predict(X, field_ids, w0, w, V)
    got = _backend.ffm_predict(X, field_ids, w0, w, V, backend="cuda")
    np.testing.assert_allclose(got, cpu, rtol=RTOL, atol=ATOL)


def test_ffm_estimator_inference_flow():
    """FFMClassifier (binary + multiclass) and FFMRegressor: fit on rust_cpu,
    set_params(backend='cuda') for inference. The multiclass per-class loop
    makes several consecutive CUDA calls — cache-reuse coverage for free."""
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 300, 20, density=0.3)
    margin = X @ rng.normal(size=20)

    y_bin = (margin > 0).astype(int)
    m = FFMClassifier(n_factors=4, max_iter=10, random_state=0).fit(X, y_bin)
    cpu_scores = m.decision_function(X)
    m.set_params(backend="cuda")
    np.testing.assert_allclose(m.decision_function(X), cpu_scores, rtol=1e-6, atol=1e-6)
    assert m.predict_proba(X).shape == (300, 2)

    y_multi = np.digitize(margin, np.quantile(margin, [1 / 3, 2 / 3]))
    m = FFMClassifier(n_factors=4, max_iter=10, random_state=0).fit(X, y_multi)
    cpu_scores = m.decision_function(X)
    m.set_params(backend="cuda")
    np.testing.assert_allclose(m.decision_function(X), cpu_scores, rtol=1e-6, atol=1e-6)
    assert m.predict_proba(X).shape == (300, 3)

    r = FFMRegressor(n_factors=4, max_iter=10, random_state=0).fit(X, margin)
    cpu_pred = r.predict(X)
    r.set_params(backend="cuda")
    np.testing.assert_allclose(r.predict(X), cpu_pred, rtol=1e-6, atol=1e-6)


TRAIN_RTOL = 1e-7
TRAIN_ATOL = 1e-8


def _train_setup(seed=0, n=120, d=25, k=4, epochs=3):
    rng = np.random.default_rng(seed)
    X = random_sparse_dense_X(rng, n, d, density=0.3)
    margin = X @ rng.normal(size=d)
    params = (0.0, rng.normal(size=d) * 0.01, rng.normal(size=(d, k)) * 0.01)
    row_orders = np.vstack([rng.permutation(n) for _ in range(epochs)]).astype(np.int64)
    return X, margin, params, row_orders


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
@pytest.mark.parametrize("loss", ["logistic", "squared"])
@pytest.mark.parametrize("batch_size", [1, 7, 120])
def test_fm_train_accumulation_parity(optimizer, loss, batch_size):
    """CUDA batch-gradient accumulation + CPU optimizer flush vs the all-CPU
    kernel: same init and row_orders, compare final predictions."""
    X, margin, params, row_orders = _train_setup()
    y = (margin > 0).astype(np.float64) if loss == "logistic" else margin
    kwargs = dict(
        loss=loss, optimizer=optimizer, learning_rate=0.1,
        l2_linear=1e-4, l2_factors=1e-4, row_orders=row_orders, batch_size=batch_size,
    )
    if optimizer == "ftrl":
        kwargs.update(l1_linear=0.01, l1_factors=0.001)
    w0_c, w_c, V_c = _backend.fm_fit(X, y, params, **kwargs)
    w0_g, w_g, V_g = _backend.fm_fit(X, y, params, backend="cuda", **kwargs)
    pred_c = _backend.fm_predict_fast(X, w0_c, w_c, V_c)
    pred_g = _backend.fm_predict_fast(X, w0_g, w_g, V_g)
    np.testing.assert_allclose(pred_g, pred_c, rtol=TRAIN_RTOL, atol=TRAIN_ATOL)


@pytest.mark.parametrize("optimizer", ["adagrad", "adam"])
def test_fm_train_compact_multirow_parity(optimizer):
    """Shapes where a multi-row batch touches far fewer features than d force
    the compact (sparse touched-coordinate) transfer path: atomics across rows
    plus slot dedup, vs the all-CPU kernel."""
    rng = np.random.default_rng(3)
    n, d, k = 60, 400, 4
    X = random_sparse_dense_X(rng, n, d, density=0.05)
    y = (X @ rng.normal(size=d) > 0).astype(np.float64)
    params = (0.0, rng.normal(size=d) * 0.01, rng.normal(size=(d, k)) * 0.01)
    row_orders = np.vstack([rng.permutation(n) for _ in range(3)]).astype(np.int64)
    kwargs = dict(
        loss="logistic", optimizer=optimizer, learning_rate=0.1,
        l2_linear=1e-4, l2_factors=1e-4, row_orders=row_orders, batch_size=4,
    )
    w0_c, w_c, V_c = _backend.fm_fit(X, y, params, **kwargs)
    w0_g, w_g, V_g = _backend.fm_fit(X, y, params, backend="cuda", **kwargs)
    np.testing.assert_allclose(
        _backend.fm_predict_fast(X, w0_g, w_g, V_g),
        _backend.fm_predict_fast(X, w0_c, w_c, V_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fm_train_mixed_dense_compact_parity():
    """A shape where full batches take the dense-gradient path while the small
    trailing partial batch takes the compact path — both modes must keep the
    device-resident parameters in sync within one fit."""
    rng = np.random.default_rng(4)
    n, d, k = 60, 280, 4
    X = random_sparse_dense_X(rng, n, d, density=10 / 280)
    y = (X @ rng.normal(size=d) > 0).astype(np.float64)
    params = (0.0, rng.normal(size=d) * 0.01, rng.normal(size=(d, k)) * 0.01)
    row_orders = np.vstack([rng.permutation(n) for _ in range(3)]).astype(np.int64)
    kwargs = dict(
        loss="logistic", optimizer="adagrad", learning_rate=0.1,
        l2_linear=1e-4, l2_factors=1e-4, row_orders=row_orders, batch_size=16,
    )
    w0_c, w_c, V_c = _backend.fm_fit(X, y, params, **kwargs)
    w0_g, w_g, V_g = _backend.fm_fit(X, y, params, backend="cuda", **kwargs)
    np.testing.assert_allclose(
        _backend.fm_predict_fast(X, w0_g, w_g, V_g),
        _backend.fm_predict_fast(X, w0_c, w_c, V_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fm_train_sample_weight_parity():
    X, margin, params, row_orders = _train_setup(seed=1)
    y = (margin > 0).astype(np.float64)
    sw = np.random.default_rng(1).uniform(0.5, 2.0, size=len(y))
    kwargs = dict(
        loss="logistic", optimizer="adagrad", learning_rate=0.1,
        l2_linear=1e-4, l2_factors=1e-4, row_orders=row_orders, batch_size=8,
        sample_weight=sw,
    )
    w0_c, w_c, V_c = _backend.fm_fit(X, y, params, **kwargs)
    w0_g, w_g, V_g = _backend.fm_fit(X, y, params, backend="cuda", **kwargs)
    np.testing.assert_allclose(
        _backend.fm_predict_fast(X, w0_g, w_g, V_g),
        _backend.fm_predict_fast(X, w0_c, w_c, V_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fm_ftrl_l1_yields_exact_zeros_on_cuda():
    """The FTRL flush stays on the CPU, so L1's exact zeros survive the CUDA
    accumulation path."""
    X, margin, params, row_orders = _train_setup(seed=2, epochs=5)
    y = (margin > 0).astype(np.float64)
    _w0, w, _V = _backend.fm_fit(
        X, y, params, loss="logistic", optimizer="ftrl", learning_rate=0.5,
        l2_linear=0.0, l2_factors=0.0, l1_linear=0.5, l1_factors=0.1,
        row_orders=row_orders, batch_size=4, backend="cuda",
    )
    assert np.sum(w == 0.0) > 0


@pytest.mark.parametrize("cls", [FMClassifier, FMRegressor])
def test_fm_estimator_fit_cuda_end_to_end(cls):
    """FMClassifier/FMRegressor fit entirely with backend='cuda' vs an
    identically-seeded CPU twin (float64 attrs to avoid casting noise)."""
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 200, 20, density=0.3)
    margin = X @ rng.normal(size=20)
    y = (margin > 0).astype(int) if cls is FMClassifier else margin
    common = dict(n_factors=4, max_iter=5, random_state=0, dtype="float64", batch_size=16)
    cpu = cls(**common).fit(X, y)
    gpu = cls(backend="cuda", **common).fit(X, y)
    score = "decision_function" if cls is FMClassifier else "predict"
    np.testing.assert_allclose(
        getattr(gpu, score)(X), getattr(cpu, score)(X), rtol=1e-6, atol=1e-6
    )


def test_fm_early_stopping_fit_cuda():
    """ES drives the CUDA kernel epoch-by-epoch through the same CPU-side
    optimizer-state hand-off; with patience >= max_iter both backends run all
    epochs, so the trajectories match up to atomic-order noise."""
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 300, 20, density=0.3)
    y = (X @ rng.normal(size=20) > 0).astype(int)
    common = dict(
        n_factors=4, max_iter=6, random_state=0, dtype="float64",
        early_stopping=True, patience=6, batch_size=32,
    )
    cpu = FMClassifier(**common).fit(X, y)
    gpu = FMClassifier(backend="cuda", **common).fit(X, y)
    assert gpu.n_iter_ == cpu.n_iter_
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )


def test_fm_partial_fit_cuda():
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 200, 20, density=0.3)
    y = (X @ rng.normal(size=20) > 0).astype(int)
    common = dict(n_factors=4, max_iter=2, random_state=0, dtype="float64")
    cpu = FMClassifier(**common)
    gpu = FMClassifier(backend="cuda", **common)
    for m in (cpu, gpu):
        m.partial_fit(X[:100], y[:100], classes=[0, 1])
        m.partial_fit(X[100:], y[100:])
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )


def _ffm_train_setup(seed=0, n=100, d=25, n_fields=4, k=3, epochs=3):
    rng = np.random.default_rng(seed)
    X = random_sparse_dense_X(rng, n, d, density=0.3)
    margin = X @ rng.normal(size=d)
    field_ids = rng.integers(0, n_fields, size=d)
    params = (0.0, rng.normal(size=d) * 0.01, rng.normal(size=(d, n_fields, k)) * 0.01)
    row_orders = np.vstack([rng.permutation(n) for _ in range(epochs)]).astype(np.int64)
    return X, margin, field_ids, params, row_orders


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
@pytest.mark.parametrize("loss", ["logistic", "squared"])
@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_ffm_train_accumulation_parity(optimizer, loss, batch_size):
    """CUDA FFM batch-gradient accumulation + CPU optimizer flush vs the
    all-CPU kernel: same init and row_orders, compare final predictions."""
    X, margin, field_ids, params, row_orders = _ffm_train_setup()
    y = (margin > 0).astype(np.float64) if loss == "logistic" else margin
    kwargs = dict(
        loss=loss, optimizer=optimizer, learning_rate=0.1,
        l2_linear=1e-4, l2_factors=1e-4, row_orders=row_orders, batch_size=batch_size,
    )
    if optimizer == "ftrl":
        kwargs.update(l1_linear=0.01, l1_factors=0.001)
    w0_c, w_c, V_c = _backend.ffm_fit(X, y, field_ids, params, **kwargs)
    w0_g, w_g, V_g = _backend.ffm_fit(X, y, field_ids, params, backend="cuda", **kwargs)
    pred_c = _backend.ffm_predict(X, field_ids, w0_c, w_c, V_c)
    pred_g = _backend.ffm_predict(X, field_ids, w0_g, w_g, V_g)
    np.testing.assert_allclose(pred_g, pred_c, rtol=TRAIN_RTOL, atol=TRAIN_ATOL)


def test_ffm_train_sample_weight_parity():
    X, margin, field_ids, params, row_orders = _ffm_train_setup(seed=1)
    y = (margin > 0).astype(np.float64)
    sw = np.random.default_rng(1).uniform(0.5, 2.0, size=len(y))
    kwargs = dict(
        loss="logistic", optimizer="adagrad", learning_rate=0.1,
        l2_linear=1e-4, l2_factors=1e-4, row_orders=row_orders, batch_size=8,
        sample_weight=sw,
    )
    w0_c, w_c, V_c = _backend.ffm_fit(X, y, field_ids, params, **kwargs)
    w0_g, w_g, V_g = _backend.ffm_fit(X, y, field_ids, params, backend="cuda", **kwargs)
    np.testing.assert_allclose(
        _backend.ffm_predict(X, field_ids, w0_g, w_g, V_g),
        _backend.ffm_predict(X, field_ids, w0_c, w_c, V_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_ffm_train_ftrl_l1_yields_exact_zeros_on_cuda():
    X, margin, field_ids, params, row_orders = _ffm_train_setup(seed=2, epochs=5)
    y = (margin > 0).astype(np.float64)
    _w0, w, _V = _backend.ffm_fit(
        X, y, field_ids, params, loss="logistic", optimizer="ftrl", learning_rate=0.5,
        l2_linear=0.0, l2_factors=0.0, l1_linear=0.5, l1_factors=0.1,
        row_orders=row_orders, batch_size=4, backend="cuda",
    )
    assert np.sum(w == 0.0) > 0


@pytest.mark.parametrize("cls", [FFMClassifier, FFMRegressor])
def test_ffm_estimator_fit_cuda_end_to_end(cls):
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 150, 20, density=0.3)
    margin = X @ rng.normal(size=20)
    y = (margin > 0).astype(int) if cls is FFMClassifier else margin
    common = dict(n_factors=3, max_iter=5, random_state=0, dtype="float64", batch_size=16)
    cpu = cls(**common).fit(X, y)
    gpu = cls(backend="cuda", **common).fit(X, y)
    score = "decision_function" if cls is FFMClassifier else "predict"
    np.testing.assert_allclose(
        getattr(gpu, score)(X), getattr(cpu, score)(X), rtol=1e-6, atol=1e-6
    )


def test_ffm_early_stopping_fit_cuda():
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 200, 20, density=0.3)
    y = (X @ rng.normal(size=20) > 0).astype(int)
    common = dict(
        n_factors=3, max_iter=5, random_state=0, dtype="float64",
        early_stopping=True, patience=5, batch_size=32,
    )
    cpu = FFMClassifier(**common).fit(X, y)
    gpu = FFMClassifier(backend="cuda", **common).fit(X, y)
    assert gpu.n_iter_ == cpu.n_iter_
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )


def test_ffm_partial_fit_cuda():
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 160, 20, density=0.3)
    y = (X @ rng.normal(size=20) > 0).astype(int)
    common = dict(n_factors=3, max_iter=2, random_state=0, dtype="float64")
    cpu = FFMClassifier(**common)
    gpu = FFMClassifier(backend="cuda", **common)
    for m in (cpu, gpu):
        m.partial_fit(X[:80], y[:80], classes=[0, 1])
        m.partial_fit(X[80:], y[80:])
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )


def _mc_train_setup(seed=0, n=120, d=25, k=4, n_classes=3, epochs=3, density=0.3):
    rng = np.random.default_rng(seed)
    X = random_sparse_dense_X(rng, n, d, density=density)
    margin = X @ rng.normal(size=d)
    y = np.digitize(margin, np.quantile(margin, np.arange(1, n_classes) / n_classes))
    params = (
        np.zeros(n_classes),
        rng.normal(size=(n_classes, d)) * 0.01,
        rng.normal(size=(n_classes, d, k)) * 0.01,
    )
    row_orders = np.vstack([rng.permutation(n) for _ in range(epochs)]).astype(np.int64)
    return X, y, params, row_orders


def _fm_mc_logits(X, w0, w, V):
    return np.column_stack(
        [_backend.fm_predict_fast(X, float(w0[c]), w[c], V[c]) for c in range(len(w0))]
    )


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
@pytest.mark.parametrize("batch_size", [1, 7, 120])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_fm_multiclass_train_accumulation_parity(optimizer, batch_size, label_smoothing):
    """CUDA per-class batch-gradient accumulation (softmax coupling on the
    GPU) + CPU per-class optimizer flush vs the all-CPU multiclass kernel:
    same init and row_orders, compare final per-class logits."""
    X, y, params, row_orders = _mc_train_setup()
    kwargs = dict(
        optimizer=optimizer, learning_rate=0.1, l2_linear=1e-4, l2_factors=1e-4,
        row_orders=row_orders, batch_size=batch_size, label_smoothing=label_smoothing,
    )
    if optimizer == "ftrl":
        kwargs.update(l1_linear=0.01, l1_factors=0.001)
    w0_c, w_c, V_c = _backend.fm_fit_multiclass(X, y, params, **kwargs)
    w0_g, w_g, V_g = _backend.fm_fit_multiclass(X, y, params, backend="cuda", **kwargs)
    np.testing.assert_allclose(
        _fm_mc_logits(X, w0_g, w_g, V_g), _fm_mc_logits(X, w0_c, w_c, V_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fm_multiclass_train_compact_multirow_parity():
    """Multi-row batches touching far fewer features than d force the compact
    C-stacked transfer path (shared slot map across classes)."""
    X, y, params, row_orders = _mc_train_setup(seed=3, n=60, d=400, density=0.05)
    kwargs = dict(
        optimizer="adagrad", learning_rate=0.1, l2_linear=1e-4, l2_factors=1e-4,
        row_orders=row_orders, batch_size=4,
    )
    w0_c, w_c, V_c = _backend.fm_fit_multiclass(X, y, params, **kwargs)
    w0_g, w_g, V_g = _backend.fm_fit_multiclass(X, y, params, backend="cuda", **kwargs)
    np.testing.assert_allclose(
        _fm_mc_logits(X, w0_g, w_g, V_g), _fm_mc_logits(X, w0_c, w_c, V_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fm_multiclass_train_mixed_dense_compact_parity():
    """Full batches take the dense path, the trailing partial batch the
    compact path — both must keep the C-stacked device parameters in sync."""
    X, y, params, row_orders = _mc_train_setup(seed=4, n=60, d=280, density=10 / 280)
    kwargs = dict(
        optimizer="adagrad", learning_rate=0.1, l2_linear=1e-4, l2_factors=1e-4,
        row_orders=row_orders, batch_size=16,
    )
    w0_c, w_c, V_c = _backend.fm_fit_multiclass(X, y, params, **kwargs)
    w0_g, w_g, V_g = _backend.fm_fit_multiclass(X, y, params, backend="cuda", **kwargs)
    np.testing.assert_allclose(
        _fm_mc_logits(X, w0_g, w_g, V_g), _fm_mc_logits(X, w0_c, w_c, V_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fm_multiclass_train_sample_weight_parity():
    X, y, params, row_orders = _mc_train_setup(seed=1)
    sw = np.random.default_rng(1).uniform(0.5, 2.0, size=len(y))
    kwargs = dict(
        optimizer="adagrad", learning_rate=0.1, l2_linear=1e-4, l2_factors=1e-4,
        row_orders=row_orders, batch_size=8, sample_weight=sw,
    )
    w0_c, w_c, V_c = _backend.fm_fit_multiclass(X, y, params, **kwargs)
    w0_g, w_g, V_g = _backend.fm_fit_multiclass(X, y, params, backend="cuda", **kwargs)
    np.testing.assert_allclose(
        _fm_mc_logits(X, w0_g, w_g, V_g), _fm_mc_logits(X, w0_c, w_c, V_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fm_multiclass_ftrl_l1_yields_exact_zeros_on_cuda():
    """The per-class FTRL flush stays on the CPU, so L1's exact zeros survive
    the CUDA accumulation path in every class."""
    X, y, params, row_orders = _mc_train_setup(seed=2, epochs=5)
    _w0, w, _V = _backend.fm_fit_multiclass(
        X, y, params, optimizer="ftrl", learning_rate=0.5,
        l2_linear=0.0, l2_factors=0.0, l1_linear=0.5, l1_factors=0.1,
        row_orders=row_orders, batch_size=4, backend="cuda",
    )
    assert np.sum(w == 0.0) > 0


def test_fm_multiclass_estimator_fit_cuda_end_to_end():
    """FMClassifier(loss='softmax') fit entirely with backend='cuda' vs an
    identically-seeded CPU twin."""
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 200, 20, density=0.3)
    margin = X @ rng.normal(size=20)
    y = np.digitize(margin, np.quantile(margin, [1 / 3, 2 / 3]))
    common = dict(n_factors=4, max_iter=5, random_state=0, dtype="float64", batch_size=16)
    cpu = FMClassifier(**common).fit(X, y)
    gpu = FMClassifier(backend="cuda", **common).fit(X, y)
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )
    assert gpu.predict_proba(X).shape == (200, 3)


def test_fm_multiclass_early_stopping_fit_cuda():
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 300, 20, density=0.3)
    margin = X @ rng.normal(size=20)
    y = np.digitize(margin, np.quantile(margin, [1 / 3, 2 / 3]))
    common = dict(
        n_factors=4, max_iter=6, random_state=0, dtype="float64",
        early_stopping=True, patience=6, batch_size=32,
    )
    cpu = FMClassifier(**common).fit(X, y)
    gpu = FMClassifier(backend="cuda", **common).fit(X, y)
    assert gpu.n_iter_ == cpu.n_iter_
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )


def test_fm_multiclass_partial_fit_cuda():
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 200, 20, density=0.3)
    margin = X @ rng.normal(size=20)
    y = np.digitize(margin, np.quantile(margin, [1 / 3, 2 / 3]))
    common = dict(n_factors=4, max_iter=2, random_state=0, dtype="float64")
    cpu = FMClassifier(**common)
    gpu = FMClassifier(backend="cuda", **common)
    for m in (cpu, gpu):
        m.partial_fit(X[:100], y[:100], classes=[0, 1, 2])
        m.partial_fit(X[100:], y[100:])
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )


def _ffm_mc_train_setup(seed=0, n=100, d=25, n_fields=4, k=3, n_classes=3, epochs=3):
    rng = np.random.default_rng(seed)
    X = random_sparse_dense_X(rng, n, d, density=0.3)
    margin = X @ rng.normal(size=d)
    y = np.digitize(margin, np.quantile(margin, np.arange(1, n_classes) / n_classes))
    field_ids = rng.integers(0, n_fields, size=d)
    params = (
        np.zeros(n_classes),
        rng.normal(size=(n_classes, d)) * 0.01,
        rng.normal(size=(n_classes, d, n_fields, k)) * 0.01,
    )
    row_orders = np.vstack([rng.permutation(n) for _ in range(epochs)]).astype(np.int64)
    return X, y, field_ids, params, row_orders


def _ffm_mc_logits(X, field_ids, w0, w, V):
    return np.column_stack(
        [
            _backend.ffm_predict(X, field_ids, float(w0[c]), w[c], V[c])
            for c in range(len(w0))
        ]
    )


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_ffm_multiclass_train_accumulation_parity(optimizer, batch_size):
    """CUDA FFM multiclass (two-kernel: score/softmax + per-class pair
    accumulation into one shared class-local gv) vs the all-CPU kernel."""
    X, y, field_ids, params, row_orders = _ffm_mc_train_setup()
    kwargs = dict(
        optimizer=optimizer, learning_rate=0.1, l2_linear=1e-4, l2_factors=1e-4,
        row_orders=row_orders, batch_size=batch_size,
    )
    if optimizer == "ftrl":
        kwargs.update(l1_linear=0.01, l1_factors=0.001)
    w0_c, w_c, V_c = _backend.ffm_fit_multiclass(X, y, field_ids, params, **kwargs)
    w0_g, w_g, V_g = _backend.ffm_fit_multiclass(
        X, y, field_ids, params, backend="cuda", **kwargs
    )
    np.testing.assert_allclose(
        _ffm_mc_logits(X, field_ids, w0_g, w_g, V_g),
        _ffm_mc_logits(X, field_ids, w0_c, w_c, V_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_ffm_multiclass_train_label_smoothing_parity():
    X, y, field_ids, params, row_orders = _ffm_mc_train_setup(seed=1)
    kwargs = dict(
        optimizer="adagrad", learning_rate=0.1, l2_linear=1e-4, l2_factors=1e-4,
        row_orders=row_orders, batch_size=8, label_smoothing=0.1,
    )
    w0_c, w_c, V_c = _backend.ffm_fit_multiclass(X, y, field_ids, params, **kwargs)
    w0_g, w_g, V_g = _backend.ffm_fit_multiclass(
        X, y, field_ids, params, backend="cuda", **kwargs
    )
    np.testing.assert_allclose(
        _ffm_mc_logits(X, field_ids, w0_g, w_g, V_g),
        _ffm_mc_logits(X, field_ids, w0_c, w_c, V_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_ffm_multiclass_train_sample_weight_parity():
    X, y, field_ids, params, row_orders = _ffm_mc_train_setup(seed=1)
    sw = np.random.default_rng(1).uniform(0.5, 2.0, size=len(y))
    kwargs = dict(
        optimizer="adagrad", learning_rate=0.1, l2_linear=1e-4, l2_factors=1e-4,
        row_orders=row_orders, batch_size=8, sample_weight=sw,
    )
    w0_c, w_c, V_c = _backend.ffm_fit_multiclass(X, y, field_ids, params, **kwargs)
    w0_g, w_g, V_g = _backend.ffm_fit_multiclass(
        X, y, field_ids, params, backend="cuda", **kwargs
    )
    np.testing.assert_allclose(
        _ffm_mc_logits(X, field_ids, w0_g, w_g, V_g),
        _ffm_mc_logits(X, field_ids, w0_c, w_c, V_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_ffm_multiclass_ftrl_l1_yields_exact_zeros_on_cuda():
    X, y, field_ids, params, row_orders = _ffm_mc_train_setup(seed=2, epochs=5)
    _w0, w, _V = _backend.ffm_fit_multiclass(
        X, y, field_ids, params, optimizer="ftrl", learning_rate=0.5,
        l2_linear=0.0, l2_factors=0.0, l1_linear=0.5, l1_factors=0.1,
        row_orders=row_orders, batch_size=4, backend="cuda",
    )
    assert np.sum(w == 0.0) > 0


def test_ffm_multiclass_estimator_fit_cuda_end_to_end():
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 150, 20, density=0.3)
    margin = X @ rng.normal(size=20)
    y = np.digitize(margin, np.quantile(margin, [1 / 3, 2 / 3]))
    common = dict(n_factors=3, max_iter=5, random_state=0, dtype="float64", batch_size=16)
    cpu = FFMClassifier(**common).fit(X, y)
    gpu = FFMClassifier(backend="cuda", **common).fit(X, y)
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )
    assert gpu.predict_proba(X).shape == (150, 3)


def test_ffm_multiclass_early_stopping_fit_cuda():
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 200, 20, density=0.3)
    margin = X @ rng.normal(size=20)
    y = np.digitize(margin, np.quantile(margin, [1 / 3, 2 / 3]))
    common = dict(
        n_factors=3, max_iter=5, random_state=0, dtype="float64",
        early_stopping=True, patience=5, batch_size=32,
    )
    cpu = FFMClassifier(**common).fit(X, y)
    gpu = FFMClassifier(backend="cuda", **common).fit(X, y)
    assert gpu.n_iter_ == cpu.n_iter_
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )


def test_ffm_multiclass_partial_fit_cuda():
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 160, 20, density=0.3)
    margin = X @ rng.normal(size=20)
    y = np.digitize(margin, np.quantile(margin, [1 / 3, 2 / 3]))
    common = dict(n_factors=3, max_iter=2, random_state=0, dtype="float64")
    cpu = FFMClassifier(**common)
    gpu = FFMClassifier(backend="cuda", **common)
    for m in (cpu, gpu):
        m.partial_fit(X[:80], y[:80], classes=[0, 1, 2])
        m.partial_fit(X[80:], y[80:])
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )


def _random_fwfm(rng, d, n_fields, k):
    w0 = float(rng.normal())
    w = rng.normal(size=d)
    V = rng.normal(size=(d, k))
    R = rng.normal(size=(n_fields, n_fields))
    field_ids = rng.integers(0, n_fields, size=d)
    return w0, w, V, R, field_ids


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("k", [1, 4, 16])
@pytest.mark.parametrize("n_fields", [1, 4, 8])
def test_fwfm_prediction_parity(seed, k, n_fields):
    rng = np.random.default_rng(seed)
    n, d = 200, 50
    X = random_sparse_dense_X(rng, n, d, density=0.2)
    w0, w, V, R, field_ids = _random_fwfm(rng, d, n_fields, k)
    want = fwfm_predict(X, field_ids, w0, w, V, R)
    got = _backend.fwfm_predict(X, field_ids, w0, w, V, R, backend="cuda")
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
    got_csr = _backend.fwfm_predict(sp.csr_matrix(X), field_ids, w0, w, V, R, backend="cuda")
    np.testing.assert_allclose(got_csr, want, rtol=RTOL, atol=ATOL)
    # and vs the Rust CPU kernel
    cpu = _backend.fwfm_predict(X, field_ids, w0, w, V, R)
    np.testing.assert_allclose(got, cpu, rtol=RTOL, atol=ATOL)


def test_fwfm_empty_rows_return_bias():
    X = sp.csr_matrix((4, 6))
    field_ids = np.zeros(6, dtype=np.int64)
    out = _backend.fwfm_predict(
        X, field_ids, 1.5, np.zeros(6), np.ones((6, 3)), np.ones((1, 1)), backend="cuda"
    )
    np.testing.assert_allclose(out, np.full(4, 1.5), rtol=RTOL, atol=ATOL)


def test_fwfm_single_nonzero_has_no_pairwise():
    rng = np.random.default_rng(0)
    X = np.zeros((3, 5))
    X[1, 2] = -2.0
    w = rng.normal(size=5)
    V = rng.normal(size=(5, 4))
    R = rng.normal(size=(3, 3))
    field_ids = rng.integers(0, 3, size=5)
    out = _backend.fwfm_predict(X, field_ids, 0.5, w, V, R, backend="cuda")
    np.testing.assert_allclose(out[0], 0.5, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(out[1], 0.5 + w[2] * -2.0, rtol=RTOL, atol=ATOL)


def test_fwfm_large_row_grid_stride():
    """A single row with z=2048 nonzeros exercises the b-strided pair loop;
    compared vs Rust CPU (itself reference-parity-tested)."""
    rng = np.random.default_rng(0)
    d = 2048
    X = sp.csr_matrix(rng.normal(size=(1, d)))
    w0, w, V, R, field_ids = _random_fwfm(rng, d, 4, 4)
    cpu = _backend.fwfm_predict(X, field_ids, w0, w, V, R)
    got = _backend.fwfm_predict(X, field_ids, w0, w, V, R, backend="cuda")
    np.testing.assert_allclose(got, cpu, rtol=RTOL, atol=ATOL)


def _fwfm_train_setup(seed=0, n=100, d=25, n_fields=4, k=3, epochs=3, density=0.3):
    rng = np.random.default_rng(seed)
    X = random_sparse_dense_X(rng, n, d, density=density)
    margin = X @ rng.normal(size=d)
    field_ids = rng.integers(0, n_fields, size=d)
    params = (
        0.0,
        rng.normal(size=d) * 0.01,
        rng.normal(size=(d, k)) * 0.01,
        np.ones((n_fields, n_fields)),
    )
    row_orders = np.vstack([rng.permutation(n) for _ in range(epochs)]).astype(np.int64)
    return X, margin, field_ids, params, row_orders


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
@pytest.mark.parametrize("loss", ["logistic", "squared"])
@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_fwfm_train_accumulation_parity(optimizer, loss, batch_size):
    """CUDA FwFM batch-gradient accumulation (incl. the R group) + CPU
    optimizer flush vs the all-CPU kernel: same init and row_orders, compare
    final predictions."""
    X, margin, field_ids, params, row_orders = _fwfm_train_setup()
    y = (margin > 0).astype(np.float64) if loss == "logistic" else margin
    kwargs = dict(
        loss=loss, optimizer=optimizer, learning_rate=0.1,
        l2_linear=1e-4, l2_factors=1e-4, row_orders=row_orders, batch_size=batch_size,
    )
    if optimizer == "ftrl":
        kwargs.update(l1_linear=0.01, l1_factors=0.001)
    w0_c, w_c, V_c, R_c = _backend.fwfm_fit(X, y, field_ids, params, **kwargs)
    w0_g, w_g, V_g, R_g = _backend.fwfm_fit(X, y, field_ids, params, backend="cuda", **kwargs)
    pred_c = _backend.fwfm_predict(X, field_ids, w0_c, w_c, V_c, R_c)
    pred_g = _backend.fwfm_predict(X, field_ids, w0_g, w_g, V_g, R_g)
    np.testing.assert_allclose(pred_g, pred_c, rtol=TRAIN_RTOL, atol=TRAIN_ATOL)


def test_fwfm_train_compact_multirow_parity():
    """Multi-row batches touching far fewer features than d force the compact
    transfer path (per-nonzero slots cover both pair endpoints)."""
    X, margin, field_ids, params, row_orders = _fwfm_train_setup(
        seed=3, n=60, d=400, density=0.05
    )
    y = (margin > 0).astype(np.float64)
    kwargs = dict(
        loss="logistic", optimizer="adagrad", learning_rate=0.1,
        l2_linear=1e-4, l2_factors=1e-4, row_orders=row_orders, batch_size=4,
    )
    w0_c, w_c, V_c, R_c = _backend.fwfm_fit(X, y, field_ids, params, **kwargs)
    w0_g, w_g, V_g, R_g = _backend.fwfm_fit(X, y, field_ids, params, backend="cuda", **kwargs)
    np.testing.assert_allclose(
        _backend.fwfm_predict(X, field_ids, w0_g, w_g, V_g, R_g),
        _backend.fwfm_predict(X, field_ids, w0_c, w_c, V_c, R_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fwfm_train_mixed_dense_compact_parity():
    X, margin, field_ids, params, row_orders = _fwfm_train_setup(
        seed=4, n=60, d=280, density=10 / 280
    )
    y = (margin > 0).astype(np.float64)
    kwargs = dict(
        loss="logistic", optimizer="adagrad", learning_rate=0.1,
        l2_linear=1e-4, l2_factors=1e-4, row_orders=row_orders, batch_size=16,
    )
    w0_c, w_c, V_c, R_c = _backend.fwfm_fit(X, y, field_ids, params, **kwargs)
    w0_g, w_g, V_g, R_g = _backend.fwfm_fit(X, y, field_ids, params, backend="cuda", **kwargs)
    np.testing.assert_allclose(
        _backend.fwfm_predict(X, field_ids, w0_g, w_g, V_g, R_g),
        _backend.fwfm_predict(X, field_ids, w0_c, w_c, V_c, R_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fwfm_train_sample_weight_parity():
    X, margin, field_ids, params, row_orders = _fwfm_train_setup(seed=1)
    y = (margin > 0).astype(np.float64)
    sw = np.random.default_rng(1).uniform(0.5, 2.0, size=len(y))
    kwargs = dict(
        loss="logistic", optimizer="adagrad", learning_rate=0.1,
        l2_linear=1e-4, l2_factors=1e-4, row_orders=row_orders, batch_size=8,
        sample_weight=sw,
    )
    w0_c, w_c, V_c, R_c = _backend.fwfm_fit(X, y, field_ids, params, **kwargs)
    w0_g, w_g, V_g, R_g = _backend.fwfm_fit(X, y, field_ids, params, backend="cuda", **kwargs)
    np.testing.assert_allclose(
        _backend.fwfm_predict(X, field_ids, w0_g, w_g, V_g, R_g),
        _backend.fwfm_predict(X, field_ids, w0_c, w_c, V_c, R_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fwfm_ftrl_l1_yields_exact_zeros_on_cuda():
    """The FTRL flush stays on the CPU, so L1's exact zeros survive — for w
    AND for the R group (regularized with l1_factors)."""
    X, margin, field_ids, params, row_orders = _fwfm_train_setup(seed=2, epochs=5)
    y = (margin > 0).astype(np.float64)
    _w0, w, _V, R = _backend.fwfm_fit(
        X, y, field_ids, params, loss="logistic", optimizer="ftrl", learning_rate=0.5,
        l2_linear=0.0, l2_factors=0.0, l1_linear=0.5, l1_factors=0.5,
        row_orders=row_orders, batch_size=4, backend="cuda",
    )
    assert np.sum(w == 0.0) > 0
    assert np.sum(R == 0.0) > 0


def test_fwfm_estimator_fit_cuda_end_to_end():
    """FwFMClassifier fit entirely with backend='cuda' vs an
    identically-seeded CPU twin — training AND prediction on the GPU."""
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 150, 20, density=0.3)
    y = (X @ rng.normal(size=20) > 0).astype(int)
    common = dict(n_factors=3, max_iter=5, random_state=0, dtype="float64", batch_size=16)
    cpu = FwFMClassifier(**common).fit(X, y)
    gpu = FwFMClassifier(backend="cuda", **common).fit(X, y)
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )
    assert gpu.predict_proba(X).shape == (150, 2)


def test_fwfm_early_stopping_fit_cuda():
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 200, 20, density=0.3)
    y = (X @ rng.normal(size=20) > 0).astype(int)
    common = dict(
        n_factors=3, max_iter=5, random_state=0, dtype="float64",
        early_stopping=True, patience=5, batch_size=32,
    )
    cpu = FwFMClassifier(**common).fit(X, y)
    gpu = FwFMClassifier(backend="cuda", **common).fit(X, y)
    assert gpu.n_iter_ == cpu.n_iter_
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )


def test_fwfm_partial_fit_cuda():
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 160, 20, density=0.3)
    y = (X @ rng.normal(size=20) > 0).astype(int)
    common = dict(n_factors=3, max_iter=2, random_state=0, dtype="float64")
    cpu = FwFMClassifier(**common)
    gpu = FwFMClassifier(backend="cuda", **common)
    for m in (cpu, gpu):
        m.partial_fit(X[:80], y[:80], classes=[0, 1])
        m.partial_fit(X[80:], y[80:])
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )


def _fwfm_mc_train_setup(seed=0, n=100, d=25, n_fields=4, k=3, n_classes=3, epochs=3):
    rng = np.random.default_rng(seed)
    X = random_sparse_dense_X(rng, n, d, density=0.3)
    margin = X @ rng.normal(size=d)
    y = np.digitize(margin, np.quantile(margin, np.arange(1, n_classes) / n_classes))
    field_ids = rng.integers(0, n_fields, size=d)
    params = (
        np.zeros(n_classes),
        rng.normal(size=(n_classes, d)) * 0.01,
        rng.normal(size=(n_classes, d, k)) * 0.01,
        np.ones((n_classes, n_fields, n_fields)),
    )
    row_orders = np.vstack([rng.permutation(n) for _ in range(epochs)]).astype(np.int64)
    return X, y, field_ids, params, row_orders


def _fwfm_mc_logits(X, field_ids, w0, w, V, R):
    return np.column_stack(
        [
            _backend.fwfm_predict(X, field_ids, float(w0[c]), w[c], V[c], R[c])
            for c in range(len(w0))
        ]
    )


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam", "ftrl"])
@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_fwfm_multiclass_train_accumulation_parity(optimizer, batch_size):
    X, y, field_ids, params, row_orders = _fwfm_mc_train_setup()
    kwargs = dict(
        optimizer=optimizer, learning_rate=0.1, l2_linear=1e-4, l2_factors=1e-4,
        row_orders=row_orders, batch_size=batch_size,
    )
    if optimizer == "ftrl":
        kwargs.update(l1_linear=0.01, l1_factors=0.001)
    w0_c, w_c, V_c, R_c = _backend.fwfm_fit_multiclass(X, y, field_ids, params, **kwargs)
    w0_g, w_g, V_g, R_g = _backend.fwfm_fit_multiclass(
        X, y, field_ids, params, backend="cuda", **kwargs
    )
    np.testing.assert_allclose(
        _fwfm_mc_logits(X, field_ids, w0_g, w_g, V_g, R_g),
        _fwfm_mc_logits(X, field_ids, w0_c, w_c, V_c, R_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fwfm_multiclass_train_label_smoothing_parity():
    X, y, field_ids, params, row_orders = _fwfm_mc_train_setup(seed=1)
    kwargs = dict(
        optimizer="adagrad", learning_rate=0.1, l2_linear=1e-4, l2_factors=1e-4,
        row_orders=row_orders, batch_size=8, label_smoothing=0.1,
    )
    w0_c, w_c, V_c, R_c = _backend.fwfm_fit_multiclass(X, y, field_ids, params, **kwargs)
    w0_g, w_g, V_g, R_g = _backend.fwfm_fit_multiclass(
        X, y, field_ids, params, backend="cuda", **kwargs
    )
    np.testing.assert_allclose(
        _fwfm_mc_logits(X, field_ids, w0_g, w_g, V_g, R_g),
        _fwfm_mc_logits(X, field_ids, w0_c, w_c, V_c, R_c),
        rtol=TRAIN_RTOL, atol=TRAIN_ATOL,
    )


def test_fwfm_multiclass_estimator_fit_cuda_end_to_end():
    rng = np.random.default_rng(0)
    X = random_sparse_dense_X(rng, 150, 20, density=0.3)
    margin = X @ rng.normal(size=20)
    y = np.digitize(margin, np.quantile(margin, [1 / 3, 2 / 3]))
    common = dict(n_factors=3, max_iter=5, random_state=0, dtype="float64", batch_size=16)
    cpu = FwFMClassifier(**common).fit(X, y)
    gpu = FwFMClassifier(backend="cuda", **common).fit(X, y)
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-6, atol=1e-6
    )
    assert gpu.predict_proba(X).shape == (150, 3)


def test_repeated_calls_reuse_cached_context():
    """Alternating FM/FFM/FwFM CUDA predicts: a functional regression test for
    the process-wide context/module cache (lifetime/refcount bugs would
    surface as errors or wrong scores on later iterations)."""
    rng = np.random.default_rng(0)
    n, d, k, n_fields = 50, 20, 4, 3
    X = random_sparse_dense_X(rng, n, d, density=0.3)
    fm_w0, fm_w, fm_V = rng.normal(), rng.normal(size=d), rng.normal(size=(d, k))
    w0, w, V, field_ids = _random_ffm(rng, d, n_fields, k)
    fw_w0, fw_w, fw_V, fw_R, fw_fields = _random_fwfm(rng, d, n_fields, k)
    fm_cpu = _backend.fm_predict_fast(X, fm_w0, fm_w, fm_V)
    ffm_cpu = _backend.ffm_predict(X, field_ids, w0, w, V)
    fwfm_cpu = _backend.fwfm_predict(X, fw_fields, fw_w0, fw_w, fw_V, fw_R)
    for _ in range(10):
        fm_cuda = _backend.fm_predict_fast(X, fm_w0, fm_w, fm_V, backend="cuda")
        np.testing.assert_allclose(fm_cuda, fm_cpu, rtol=RTOL, atol=ATOL)
        ffm_cuda = _backend.ffm_predict(X, field_ids, w0, w, V, backend="cuda")
        np.testing.assert_allclose(ffm_cuda, ffm_cpu, rtol=RTOL, atol=ATOL)
        fwfm_cuda = _backend.fwfm_predict(X, fw_fields, fw_w0, fw_w, fw_V, fw_R, backend="cuda")
        np.testing.assert_allclose(fwfm_cuda, fwfm_cpu, rtol=RTOL, atol=ATOL)
