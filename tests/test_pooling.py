"""BiInteractionPooling tests: the collapse-to-FM identity, dense/CSR
equivalence, Pipeline integration, sklearn compliance, and round-trips."""

import pickle

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import random_sparse_dense_X
from modern_fm import BiInteractionPooling, FMClassifier, FMRegressor
from modern_fm._reference import fm_bi_interaction, fm_predict_fast
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import check_estimator


def _data(rng, n=120, d=10):
    X = random_sparse_dense_X(rng, n, d, density=0.4)
    y = (X @ rng.normal(size=d) > 0).astype(int)
    return X, y


def test_collapse_identity(rng):
    """fm_predict_fast == w0 + X @ w + bi_interaction.sum(axis=1) exactly —
    the reason pooling ships as a transform: a linear head over it can only
    re-learn FM."""
    for seed in range(3):
        r2 = np.random.default_rng(seed)
        X = random_sparse_dense_X(r2, 50, 8, density=0.5)
        w0, w, V = 0.3, r2.normal(size=8), r2.normal(size=(8, 4))
        for Xi in (X, sp.csr_matrix(X)):
            pooled = fm_bi_interaction(Xi, V)
            assert pooled.shape == (50, 4)
            np.testing.assert_allclose(
                w0 + Xi @ w + pooled.sum(axis=1),
                fm_predict_fast(Xi, w0, w, V),
                rtol=1e-12, atol=1e-12,
            )


def test_dense_csr_equivalence(rng):
    X, y = _data(rng)
    t = BiInteractionPooling(FMRegressor(n_factors=4, max_iter=5, random_state=0))
    t.fit(X, y)
    np.testing.assert_allclose(
        t.transform(X), t.transform(sp.csr_matrix(X)), rtol=1e-12, atol=1e-12
    )


def test_transform_shape_and_feature_names(rng):
    X, y = _data(rng)
    t = BiInteractionPooling(FMRegressor(n_factors=6, max_iter=5, random_state=0)).fit(X, y)
    out = t.transform(X)
    assert out.shape == (X.shape[0], 6)
    names = t.get_feature_names_out()
    assert len(names) == 6


def test_multiclass_concatenates_per_class(rng):
    X, _ = _data(rng)
    y3 = np.digitize(X @ np.random.default_rng(1).normal(size=X.shape[1]), [-0.5, 0.5])
    t = BiInteractionPooling(FMClassifier(n_factors=3, max_iter=5, random_state=0)).fit(X, y3)
    out = t.transform(X)
    assert out.shape == (X.shape[0], 3 * 3)  # n_classes * n_factors
    assert len(t.get_feature_names_out()) == 9


def test_matches_estimator_bi_interaction_method(rng):
    X, y = _data(rng)
    fm = FMClassifier(n_factors=4, max_iter=5, random_state=0).fit(X, y)
    t = BiInteractionPooling(FMClassifier(n_factors=4, max_iter=5, random_state=0)).fit(X, y)
    np.testing.assert_allclose(fm.bi_interaction(X), t.transform(X), rtol=1e-12, atol=1e-12)


def test_pipeline_integration(rng):
    X, y = _data(rng)
    pipe = make_pipeline(
        BiInteractionPooling(FMRegressor(n_factors=4, max_iter=8, random_state=0)),
        LogisticRegression(max_iter=200),
    ).fit(X, y)
    assert pipe.predict(X).shape == (X.shape[0],)
    assert pipe.score(X, y) > 0.5


def test_default_estimator_is_fm_regressor(rng):
    X, y = _data(rng)
    t = BiInteractionPooling().fit(X, y)
    assert isinstance(t.estimator_, FMRegressor)
    assert t.transform(X).shape == (X.shape[0], 8)  # default n_factors=8


def test_fit_does_not_mutate_given_estimator(rng):
    X, y = _data(rng)
    est = FMRegressor(n_factors=4, max_iter=3, random_state=0)
    BiInteractionPooling(est).fit(X, y)
    assert not hasattr(est, "V_")  # cloned, not fitted in place


def test_pickle_roundtrip(rng):
    X, y = _data(rng)
    t = BiInteractionPooling(FMRegressor(n_factors=4, max_iter=5, random_state=0)).fit(X, y)
    loaded = pickle.loads(pickle.dumps(t))
    np.testing.assert_array_equal(loaded.transform(X), t.transform(X))


def test_transform_before_fit_raises():
    from modern_fm import NotFittedError

    with pytest.raises(NotFittedError):
        BiInteractionPooling().transform(np.zeros((2, 3)))


def test_wrong_feature_count_rejected(rng):
    X, y = _data(rng)
    t = BiInteractionPooling(FMRegressor(n_factors=4, max_iter=3, random_state=0)).fit(X, y)
    with pytest.raises(ValueError):
        t.transform(X[:, :-1])


def test_sklearn_check_estimator():
    check_estimator(
        BiInteractionPooling(FMRegressor(n_factors=2, max_iter=5, n_jobs=1, random_state=0))
    )
