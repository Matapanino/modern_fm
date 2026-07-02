import numpy as np
import pytest
from modern_fm import FFMClassifier, FFMRegressor, FMClassifier, FMRegressor

ESTIMATORS = [FMClassifier, FMRegressor, FFMClassifier, FFMRegressor]


def _tiny_binary(n=6, d=3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    y = np.array([0, 1] * (n // 2))
    return X, y


def _ffm_kwargs(cls, n_features):
    """field_ids kwarg accepted by FFM estimators' fit, empty for FM estimators."""
    return (
        {"field_ids": np.zeros(n_features, dtype=int)}
        if cls in (FFMClassifier, FFMRegressor)
        else {}
    )


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_init_stores_params_only(cls):
    model = cls()
    # no learned attributes (trailing underscore) and no extra state after init
    assert all(not k.endswith("_") for k in vars(model))
    assert set(vars(model)) == set(model.get_params())


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


def test_ffm_fit_defaults_field_ids_to_per_column():
    # Without field_ids, each column becomes its own field so fit(X, y) works
    # under the sklearn API; n_fields_ == n_features.
    X = np.zeros((4, 3))
    y = np.array([0, 1, 0, 1])
    model = FFMClassifier(max_iter=2, random_state=0).fit(X, y)
    assert model.n_fields_ == 3
    assert np.array_equal(model.field_ids_, np.arange(3))


# fit is implemented for FM binary/regression and FFM binary (Phase 2B). The
# guards below cover the v0.1 features that are not wired up yet; each lands in
# a later phase (docs/roadmap.md) and its guard test moves from "raises" to
# "works" then.


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_batch_size_gt_one_fits(cls):
    # mini-batch (batch_size > 1) is supported (docs/optimization_spec.md).
    X, y = _tiny_binary()
    model = cls(batch_size=4).fit(X, y, **_ffm_kwargs(cls, X.shape[1]))
    assert model is model
    assert model.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_batch_size_must_be_positive_int(cls, bad):
    X, y = _tiny_binary()
    with pytest.raises(ValueError, match="batch_size"):
        cls(batch_size=bad).fit(X, y, **_ffm_kwargs(cls, X.shape[1]))


def test_ffm_softmax_loss_fits_multiclass():
    # loss="softmax" routes through the multiclass (per-class) FFM path.
    X, y = _tiny_binary()
    model = FFMClassifier(loss="softmax", max_iter=5).fit(
        X, y, field_ids=np.zeros(X.shape[1], dtype=int)
    )
    assert model.V_.ndim == 4  # (n_classes, n_features, n_fields, k)
    assert model.predict_proba(X).shape == (X.shape[0], 2)


def test_ffm_multiclass_fits():
    X, _ = _tiny_binary()
    y = np.array([0, 1, 2, 0, 1, 2])
    model = FFMClassifier(max_iter=5).fit(X, y, field_ids=np.zeros(X.shape[1], dtype=int))
    assert model.V_.shape[0] == 3  # one FFM per class
    np.testing.assert_array_equal(model.classes_, np.array([0, 1, 2]))


def test_version_matches_installed_metadata():
    """__version__ (frozen public API) must track the distribution version —
    v1.1.0 shipped self-reporting "1.0.0" because the hardcoded string was
    missed in the release bump; this pins the two together."""
    import importlib.metadata

    import modern_fm

    assert modern_fm.__version__ == importlib.metadata.version("modern-fm")
