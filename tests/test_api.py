import numpy as np
import pytest
from modern_fm import FFMClassifier, FMClassifier, FMRegressor

ESTIMATORS = [FMClassifier, FMRegressor, FFMClassifier]


def _tiny_binary(n=6, d=3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    y = np.array([0, 1] * (n // 2))
    return X, y


def _ffm_kwargs(cls, n_features):
    """field_ids kwarg required by FFMClassifier.fit, empty for FM estimators."""
    return {"field_ids": np.zeros(n_features, dtype=int)} if cls is FFMClassifier else {}


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
def test_adam_params_roundtrip(cls):
    model = cls(optimizer="adam", beta_1=0.85, beta_2=0.99, epsilon=1e-7)
    params = model.get_params()
    assert (params["beta_1"], params["beta_2"], params["epsilon"]) == (0.85, 0.99, 1e-7)
    assert cls().set_params(**params).get_params() == params


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


# fit is implemented for FM binary/regression and FFM binary (Phase 2B). The
# guards below cover the v0.1 features that are not wired up yet; each lands in
# a later phase (docs/roadmap.md) and its guard test moves from "raises" to
# "works" then.


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_batch_size_not_implemented(cls):
    X, y = _tiny_binary()
    with pytest.raises(NotImplementedError):
        cls(batch_size=2).fit(X, y, **_ffm_kwargs(cls, X.shape[1]))


def test_ffm_softmax_not_implemented():
    # FMClassifier supports softmax/multiclass; FFM stays binary in v0.1.
    X, y = _tiny_binary()
    with pytest.raises(NotImplementedError):
        FFMClassifier(loss="softmax").fit(X, y, field_ids=np.zeros(X.shape[1], dtype=int))


def test_ffm_multiclass_not_implemented():
    X, _ = _tiny_binary()
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(NotImplementedError):
        FFMClassifier().fit(X, y, field_ids=np.zeros(X.shape[1], dtype=int))
