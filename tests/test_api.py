import numpy as np
import pytest
from modern_fm import FFMClassifier, FMClassifier, FMRegressor

ESTIMATORS = [FMClassifier, FMRegressor, FFMClassifier]
CLASSIFIERS = [FMClassifier, FFMClassifier]


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
@pytest.mark.parametrize("ctor", [{"batch_size": 2}, {"early_stopping": True}])
def test_unimplemented_ctor_options_raise(cls, ctor):
    X, y = _tiny_binary()
    with pytest.raises(NotImplementedError):
        cls(**ctor).fit(X, y, **_ffm_kwargs(cls, X.shape[1]))


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_eval_set_not_implemented(cls):
    X, y = _tiny_binary()
    with pytest.raises(NotImplementedError):
        cls().fit(X, y, **_ffm_kwargs(cls, X.shape[1]), eval_set=())


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_softmax_loss_not_implemented(cls):
    X, y = _tiny_binary()
    with pytest.raises(NotImplementedError):
        cls(loss="softmax").fit(X, y, **_ffm_kwargs(cls, X.shape[1]))


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_multiclass_not_implemented(cls):
    X, _ = _tiny_binary()
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(NotImplementedError):
        cls().fit(X, y, **_ffm_kwargs(cls, X.shape[1]))


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_save_load_not_implemented(cls, tmp_path):
    path = str(tmp_path / "model.bin")
    with pytest.raises(NotImplementedError):
        cls().save_model(path)
    with pytest.raises(NotImplementedError):
        cls.load_model(path)
