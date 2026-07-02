"""CUDA FM + FFM prediction parity (docs/gpu_backend_plan.md milestones 1-2).

Skipped entirely without a CUDA build + device — run on a real GPU per
docs/cuda_validation_runbook.md before merging CUDA kernel changes.
Tolerance-based (rtol/atol 1e-10): CUDA reduction order is not bit-identical
to the CPU paths (floating-point addition is not associative).
"""

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import random_sparse_dense_X
from modern_fm import FFMClassifier, FFMRegressor, FMClassifier, _backend
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
