"""CUDA backend plumbing (docs/gpu_backend_plan.md): `backend="cuda"` is
accepted with clear errors, never a silent CPU fallback. Kernel parity tests
live separately and skip without a GPU; these run everywhere."""

import numpy as np
import pytest
from modern_fm import (
    FFMClassifier,
    FFMRegressor,
    FMClassifier,
    FMRegressor,
    FwFMClassifier,
    _backend,
)


def _xy(rng, n=20, d=5):
    X = rng.normal(size=(n, d))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_has_cuda_returns_bool():
    assert isinstance(_backend.has_cuda(), bool)


def test_import_never_requires_cuda():
    # importing modern_fm must work on CUDA-less machines (this test running
    # at all is the check); has_cuda() must not raise either
    _backend.has_cuda()


@pytest.mark.parametrize("cls", [FMClassifier, FMRegressor])
def test_fm_fit_cuda_supported(rng, cls):
    """FM binary/regression training accumulates on CUDA (gpu_backend_plan
    milestone 3); without CUDA the error stays clear, never a CPU fallback."""
    X, y = _xy(rng)
    model = cls(backend="cuda", max_iter=2)
    if _backend.has_cuda():
        model.fit(X, y)
        assert model.predict(X).shape == (X.shape[0],)
    else:
        with pytest.raises(RuntimeError, match="cuda-backend"):
            model.fit(X, y)


@pytest.mark.parametrize("cls", [FFMClassifier, FFMRegressor, FwFMClassifier])
def test_fit_cuda_unsupported_models_error_clearly(rng, cls):
    X, y = _xy(rng)
    model = cls(backend="cuda", max_iter=2)
    if _backend.has_cuda():
        # CUDA present: fitting names the exact unsupported surface
        with pytest.raises(NotImplementedError, match="FM binary/regression training"):
            model.fit(X, y)
    else:
        with pytest.raises(RuntimeError, match="cuda-backend"):
            model.fit(X, y)


def test_fm_multiclass_fit_cuda_not_implemented(rng):
    X, _ = _xy(rng, n=30)
    y3 = np.arange(30) % 3
    model = FMClassifier(backend="cuda", max_iter=2)
    if _backend.has_cuda():
        with pytest.raises(NotImplementedError, match="multiclass FM training"):
            model.fit(X, y3)
    else:
        with pytest.raises(RuntimeError, match="cuda-backend"):
            model.fit(X, y3)


def test_fm_predict_cuda_requires_cuda(rng):
    """The predict-time CUDA entry never falls back to CPU silently."""
    X, y = _xy(rng)
    m = FMClassifier(max_iter=3, random_state=0).fit(X, y)
    m.set_params(backend="cuda")
    if _backend.has_cuda():
        p = m.decision_function(X)
        assert p.shape == (X.shape[0],)
    else:
        with pytest.raises(RuntimeError, match="cuda-backend"):
            m.decision_function(X)


@pytest.mark.parametrize("cls", [FFMClassifier, FFMRegressor])
def test_ffm_predict_cuda_requires_cuda(rng, cls):
    """FFM prediction has a CUDA kernel (gpu_backend_plan milestone 2); like FM,
    it never falls back to CPU silently."""
    X, y = _xy(rng)
    m = cls(max_iter=3, random_state=0).fit(X, y)
    m.set_params(backend="cuda")
    score = m.decision_function if cls is FFMClassifier else m.predict
    if _backend.has_cuda():
        p = score(X)
        assert p.shape == (X.shape[0],)
    else:
        with pytest.raises(RuntimeError, match="cuda-backend"):
            score(X)


def test_fwfm_predict_cuda_not_implemented(rng):
    X, y = _xy(rng)
    m = FwFMClassifier(max_iter=3, random_state=0).fit(X, y)
    m.set_params(backend="cuda")
    with pytest.raises(NotImplementedError, match="FwFM prediction"):
        m.decision_function(X)


def test_backend_bogus_still_valueerror(rng):
    X, y = _xy(rng)
    with pytest.raises(ValueError, match="unknown backend"):
        FMClassifier(backend="bogus").fit(X, y)


def test_backend_rust_cpu_unaffected(rng):
    X, y = _xy(rng)
    m = FMClassifier(backend="rust_cpu", max_iter=5, random_state=0).fit(X, y)
    assert m.predict(X).shape == (X.shape[0],)


def test_error_at_fit_not_init():
    # sklearn convention: __init__ stores only; the backend error surfaces at fit
    FMClassifier(backend="cuda")
    FMClassifier(backend="bogus")


def test_partial_fit_backend_cuda(rng):
    """Binary FM partial_fit rides the same CUDA kernel as fit (state stays on
    the CPU); without CUDA it errors clearly."""
    X, y = _xy(rng)
    m = FMClassifier(backend="cuda", max_iter=2)
    if _backend.has_cuda():
        m.partial_fit(X, y, classes=[0, 1])
        assert m.predict(X).shape == (X.shape[0],)
    else:
        with pytest.raises(RuntimeError, match="cuda-backend"):
            m.partial_fit(X, y, classes=[0, 1])


def test_partial_fit_cuda_unsupported_cells(rng):
    X, y = _xy(rng, n=30)
    ffm = FFMClassifier(backend="cuda", max_iter=2)
    fm3 = FMClassifier(backend="cuda", max_iter=2)
    if _backend.has_cuda():
        with pytest.raises(NotImplementedError, match="FM binary/regression training"):
            ffm.partial_fit(X, y, classes=[0, 1])
        with pytest.raises(NotImplementedError, match="multiclass FM training"):
            fm3.partial_fit(X, np.arange(30) % 3, classes=[0, 1, 2])
    else:
        with pytest.raises(RuntimeError, match="cuda-backend"):
            ffm.partial_fit(X, y, classes=[0, 1])
