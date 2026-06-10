import numpy as np
from conftest import random_fm_params, random_sparse_dense_X
from modern_fm._reference import fm_predict_fast, fm_predict_naive


def test_naive_equals_fast_dense(rng):
    X = rng.normal(size=(20, 12))
    w0, w, V = random_fm_params(rng, n_features=12, n_factors=5)
    np.testing.assert_allclose(
        fm_predict_naive(X, w0, w, V), fm_predict_fast(X, w0, w, V), atol=1e-10
    )


def test_naive_equals_fast_with_zeros(rng):
    X = random_sparse_dense_X(rng, n_samples=30, n_features=15)
    w0, w, V = random_fm_params(rng, n_features=15, n_factors=4)
    np.testing.assert_allclose(
        fm_predict_naive(X, w0, w, V), fm_predict_fast(X, w0, w, V), atol=1e-10
    )


def test_all_zero_row_predicts_bias(rng):
    X = np.zeros((1, 8))
    w0, w, V = random_fm_params(rng, n_features=8, n_factors=3)
    np.testing.assert_allclose(fm_predict_fast(X, w0, w, V), [w0], atol=1e-12)
    np.testing.assert_allclose(fm_predict_naive(X, w0, w, V), [w0], atol=1e-12)


def test_single_nonzero_feature_has_no_pairwise_term(rng):
    X = np.zeros((1, 8))
    X[0, 3] = 2.5
    w0, w, V = random_fm_params(rng, n_features=8, n_factors=3)
    expected = w0 + w[3] * 2.5
    np.testing.assert_allclose(fm_predict_fast(X, w0, w, V), [expected], atol=1e-10)
    np.testing.assert_allclose(fm_predict_naive(X, w0, w, V), [expected], atol=1e-10)


def test_two_features_hand_computed():
    # x = (2, 3), v_0=(1, 0), v_1=(1, 1), w=(0.5, -1), w0=0.25
    # y = 0.25 + (0.5*2 - 1*3) + <v_0, v_1> * 2 * 3 = 0.25 - 2 + 1*6 = 4.25
    X = np.array([[2.0, 3.0]])
    w0, w = 0.25, np.array([0.5, -1.0])
    V = np.array([[1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(fm_predict_naive(X, w0, w, V), [4.25])
    np.testing.assert_allclose(fm_predict_fast(X, w0, w, V), [4.25])


def test_k1_factor(rng):
    X = rng.normal(size=(10, 6))
    w0, w, V = random_fm_params(rng, n_features=6, n_factors=1)
    np.testing.assert_allclose(
        fm_predict_naive(X, w0, w, V), fm_predict_fast(X, w0, w, V), atol=1e-10
    )
