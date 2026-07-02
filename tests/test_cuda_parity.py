"""CUDA FM/FFM prediction + FM training parity (docs/gpu_backend_plan.md
milestones 1-3).

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
from modern_fm import FFMClassifier, FFMRegressor, FMClassifier, FMRegressor, _backend
from modern_fm._reference import ffm_predict, fm_predict_fast

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


def test_repeated_calls_reuse_cached_context():
    """Alternating FM/FFM CUDA predicts: a functional regression test for the
    process-wide context/module cache (lifetime/refcount bugs would surface as
    errors or wrong scores on later iterations)."""
    rng = np.random.default_rng(0)
    n, d, k, n_fields = 50, 20, 4, 3
    X = random_sparse_dense_X(rng, n, d, density=0.3)
    fm_w0, fm_w, fm_V = rng.normal(), rng.normal(size=d), rng.normal(size=(d, k))
    w0, w, V, field_ids = _random_ffm(rng, d, n_fields, k)
    fm_cpu = _backend.fm_predict_fast(X, fm_w0, fm_w, fm_V)
    ffm_cpu = _backend.ffm_predict(X, field_ids, w0, w, V)
    for _ in range(10):
        fm_cuda = _backend.fm_predict_fast(X, fm_w0, fm_w, fm_V, backend="cuda")
        np.testing.assert_allclose(fm_cuda, fm_cpu, rtol=RTOL, atol=ATOL)
        ffm_cuda = _backend.ffm_predict(X, field_ids, w0, w, V, backend="cuda")
        np.testing.assert_allclose(ffm_cuda, ffm_cpu, rtol=RTOL, atol=ATOL)
