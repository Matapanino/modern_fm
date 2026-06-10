import numpy as np
import pytest
from modern_fm import FFMClassifier, FMClassifier, FMRegressor

ESTIMATORS = [FMClassifier, FMRegressor, FFMClassifier]


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_init_stores_params_only(cls):
    model = cls()
    # no learned attributes (trailing underscore) and no extra state after init
    assert all(not k.endswith("_") for k in vars(model))
    assert set(vars(model)) == set(model._param_names())


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_get_set_params_roundtrip(cls):
    model = cls(n_factors=7, learning_rate=0.123, random_state=99)
    params = model.get_params()
    assert params["n_factors"] == 7
    assert params["learning_rate"] == 0.123
    assert params["random_state"] == 99

    clone = cls().set_params(**params)
    assert clone.get_params() == params


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_set_params_rejects_unknown(cls):
    with pytest.raises(ValueError, match="Invalid parameter"):
        cls().set_params(definitely_not_a_param=1)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_set_params_returns_self(cls):
    model = cls()
    assert model.set_params(n_factors=3) is model
    assert model.n_factors == 3


def test_ffm_fit_requires_field_ids():
    X = np.zeros((4, 3))
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="field_ids"):
        FFMClassifier().fit(X, y)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_fit_not_implemented_yet(cls):
    X = np.zeros((4, 3))
    y = np.array([0, 1, 0, 1])
    kwargs = {"field_ids": np.zeros(3, dtype=int)} if cls is FFMClassifier else {}
    with pytest.raises(NotImplementedError):
        cls().fit(X, y, **kwargs)
