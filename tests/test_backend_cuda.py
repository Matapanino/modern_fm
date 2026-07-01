"""CUDA backend plumbing (docs/gpu_backend_plan.md): `backend="cuda"` is
accepted with clear errors, never a silent CPU fallback. Kernel parity tests
live separately and skip without a GPU; these run everywhere."""

import pytest
from modern_fm import FFMClassifier, FMClassifier, FMRegressor, FwFMClassifier, _backend


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


@pytest.mark.parametrize("cls", [FMClassifier, FMRegressor, FFMClassifier, FwFMClassifier])
def test_backend_cuda_errors_clearly(rng, cls):
    X, y = _xy(rng)
    model = cls(backend="cuda", max_iter=2)
    if _backend.has_cuda():
        # CUDA present but no kernels yet: the exact unsupported surface
        with pytest.raises(NotImplementedError, match="no kernels yet"):
            model.fit(X, y)
    else:
        with pytest.raises(RuntimeError, match="cuda-backend"):
            model.fit(X, y)


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


def test_partial_fit_validates_backend_too(rng):
    X, y = _xy(rng)
    with pytest.raises((RuntimeError, NotImplementedError)):
        FMClassifier(backend="cuda").partial_fit(X, y, classes=[0, 1])
