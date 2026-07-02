"""save_model / load_model and pickle round-trips (docs/test_plan.md)."""

import pickle

import numpy as np
import pytest
from modern_fm import (
    FFMClassifier,
    FFMRegressor,
    FMClassifier,
    FMRegressor,
    FwFMClassifier,
    NotFittedError,
)

ESTIMATORS = [FMClassifier, FMRegressor, FFMClassifier, FFMRegressor, FwFMClassifier]


def _fit(cls, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 5))
    y = (X[:, 0] > 0).astype(int)
    model = cls(random_state=0, max_iter=20)
    if cls in (FFMClassifier, FwFMClassifier):
        model.fit(X, y, field_ids=np.arange(5) % 2)
    elif cls is FFMRegressor:
        model.fit(X, X[:, 0], field_ids=np.arange(5) % 2)
    elif cls is FMRegressor:
        model.fit(X, X[:, 0])
    else:
        model.fit(X, y)
    return model, X


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_save_load_preserves_predictions(cls, tmp_path):
    model, X = _fit(cls)
    path = str(tmp_path / "model.bin")
    model.save_model(path)
    loaded = cls.load_model(path)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
    assert loaded.get_params() == model.get_params()
    if cls not in (FMRegressor, FFMRegressor):
        np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
        np.testing.assert_array_equal(loaded.classes_, model.classes_)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_pickle_roundtrip_preserves_predictions(cls):
    model, X = _fit(cls)
    loaded = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_load_model_wrong_class_raises(tmp_path):
    model, _ = _fit(FMClassifier)
    path = str(tmp_path / "m.bin")
    model.save_model(path)
    with pytest.raises(ValueError, match="not a"):
        FMRegressor.load_model(path)


def test_save_model_requires_fitted(tmp_path):
    with pytest.raises(NotFittedError):
        FMClassifier().save_model(str(tmp_path / "m.bin"))


def test_load_model_rejects_newer_format(tmp_path):
    """docs/compat_policy.md: files from a newer modern_fm fail with a clear
    upgrade error instead of loading garbage."""
    model, _ = _fit(FMClassifier)
    path = str(tmp_path / "m.bin")
    model.save_model(path)
    with open(path, "rb") as f:
        state = pickle.load(f)
    state["format_version"] = 999
    with open(path, "wb") as f:
        pickle.dump(state, f)
    with pytest.raises(ValueError, match="newer modern_fm"):
        FMClassifier.load_model(path)


def test_save_model_carries_format_version(tmp_path):
    model, _ = _fit(FMClassifier)
    path = str(tmp_path / "m.bin")
    model.save_model(path)
    with open(path, "rb") as f:
        state = pickle.load(f)
    assert state["format_version"] == 1
