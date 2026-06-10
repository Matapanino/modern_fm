import numpy as np
from conftest import random_ffm_params, random_sparse_dense_X
from modern_fm._reference import ffm_predict, ffm_predict_naive


def test_tiny_hand_computed_example():
    # 3 features, 2 fields: f = (0, 0, 1). x = (1, 2, 3). w0 = 0, w = 0.
    # V[i, g] is the vector feature i uses against field g (k = 2).
    V = np.zeros((3, 2, 2))
    V[0, 0] = [1.0, 0.0]
    V[0, 1] = [0.0, 1.0]
    V[1, 0] = [1.0, 1.0]
    V[1, 1] = [2.0, 0.0]
    V[2, 0] = [0.5, 0.5]
    V[2, 1] = [1.0, -1.0]
    field_ids = np.array([0, 0, 1])
    X = np.array([[1.0, 2.0, 3.0]])
    # pair (0,1): fields (0,0) -> <V[0,0], V[1,0]> * 1*2 = <(1,0),(1,1)> * 2 = 2
    # pair (0,2): fields (0,1) -> <V[0,1], V[2,0]> * 1*3 = <(0,1),(0.5,0.5)> * 3 = 1.5
    # pair (1,2): fields (0,1) -> <V[1,1], V[2,0]> * 2*3 = <(2,0),(0.5,0.5)> * 6 = 6
    expected = 2.0 + 1.5 + 6.0
    np.testing.assert_allclose(ffm_predict_naive(X, field_ids, 0.0, np.zeros(3), V), [expected])
    np.testing.assert_allclose(ffm_predict(X, field_ids, 0.0, np.zeros(3), V), [expected])


def test_naive_equals_vectorized(rng):
    X = random_sparse_dense_X(rng, n_samples=25, n_features=14)
    w0, w, V, field_ids = random_ffm_params(rng, n_features=14, n_fields=4, n_factors=3)
    np.testing.assert_allclose(
        ffm_predict_naive(X, field_ids, w0, w, V),
        ffm_predict(X, field_ids, w0, w, V),
        atol=1e-10,
    )


def test_single_field_reduces_to_fm_form(rng):
    # With one field, V[i, 0] plays the role of FM's v_i.
    from modern_fm._reference import fm_predict_naive

    X = rng.normal(size=(10, 7))
    w0, w, V, _ = random_ffm_params(rng, n_features=7, n_fields=1, n_factors=4)
    field_ids = np.zeros(7, dtype=int)
    np.testing.assert_allclose(
        ffm_predict(X, field_ids, w0, w, V),
        fm_predict_naive(X, w0, w, V[:, 0, :]),
        atol=1e-10,
    )


def test_all_zero_row_predicts_bias(rng):
    X = np.zeros((1, 6))
    w0, w, V, field_ids = random_ffm_params(rng, n_features=6, n_fields=3, n_factors=2)
    np.testing.assert_allclose(ffm_predict(X, field_ids, w0, w, V), [w0], atol=1e-12)


def test_single_nonzero_feature_has_no_pairwise_term(rng):
    X = np.zeros((1, 6))
    X[0, 2] = -1.5
    w0, w, V, field_ids = random_ffm_params(rng, n_features=6, n_fields=3, n_factors=2)
    expected = w0 + w[2] * (-1.5)
    np.testing.assert_allclose(ffm_predict(X, field_ids, w0, w, V), [expected], atol=1e-10)
