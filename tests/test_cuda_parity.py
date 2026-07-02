"""CUDA FM prediction parity (docs/gpu_backend_plan.md milestone 1).

Skipped entirely without a CUDA build + device — run on a real GPU per
docs/cuda_validation_runbook.md before merging CUDA kernel changes.
Tolerance-based (rtol/atol 1e-10): CUDA reduction order is not bit-identical
to the CPU paths (floating-point addition is not associative).
"""

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import random_sparse_dense_X
from modern_fm import FMClassifier, _backend
from modern_fm._reference import fm_predict_fast

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
