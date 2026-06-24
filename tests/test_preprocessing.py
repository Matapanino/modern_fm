"""Tests for CategoricalEncoder (docs/data_format.md)."""

import numpy as np
import pytest
import scipy.sparse as sp
from modern_fm import CategoricalEncoder, FFMClassifier, NotFittedError


def test_basic_one_hot_and_field_ids():
    X = np.array([[0, 10], [1, 10], [0, 20]])
    enc = CategoricalEncoder()
    Xt = enc.fit_transform(X)
    assert sp.issparse(Xt)
    # column 0 categories {0,1} -> 2 cols; column 1 categories {10,20} -> 2 cols
    assert enc.n_features_out_ == 4
    assert enc.n_fields_ == 2
    np.testing.assert_array_equal(enc.field_ids_, [0, 0, 1, 1])
    expected = np.array(
        [
            [1, 0, 1, 0],  # cat0=0 -> col0 ; cat1=10 -> col2
            [0, 1, 1, 0],  # cat0=1 -> col1 ; cat1=10 -> col2
            [1, 0, 0, 1],  # cat0=0 -> col0 ; cat1=20 -> col3
        ]
    )
    np.testing.assert_array_equal(Xt.toarray(), expected)
    # exactly one active column per original column, per row
    np.testing.assert_array_equal(np.asarray(Xt.sum(axis=1)).ravel(), [2, 2, 2])


def test_non_contiguous_integer_categories():
    # categories need not be 0..K-1; encoder maps via sorted unique
    enc = CategoricalEncoder().fit(np.array([[5], [9], [5]]))
    np.testing.assert_array_equal(enc.categories_[0], [5, 9])
    np.testing.assert_array_equal(enc.transform(np.array([[9], [5]])).toarray(), [[0, 1], [1, 0]])


def test_unknown_category_ignored():
    enc = CategoricalEncoder().fit(np.array([[0], [1]]))
    Xt = enc.transform(np.array([[0], [5]]))  # 5 is unseen -> no active column
    np.testing.assert_array_equal(Xt.toarray(), [[1, 0], [0, 0]])


def test_unknown_category_error():
    enc = CategoricalEncoder(handle_unknown="error").fit(np.array([[0], [1]]))
    with pytest.raises(ValueError, match="unknown category"):
        enc.transform(np.array([[2]]))


def test_invalid_handle_unknown_raises_at_fit():
    with pytest.raises(ValueError, match="handle_unknown"):
        CategoricalEncoder(handle_unknown="boom").fit(np.array([[0]]))


def test_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        CategoricalEncoder().transform(np.array([[0]]))


def test_feature_count_mismatch_raises():
    enc = CategoricalEncoder().fit(np.array([[0, 1], [1, 0]]))
    with pytest.raises(ValueError, match="columns"):
        enc.transform(np.array([[0]]))


def test_params_roundtrip():
    enc = CategoricalEncoder(handle_unknown="error")
    assert enc.get_params() == {"handle_unknown": "error"}
    clone = CategoricalEncoder().set_params(handle_unknown="error")
    assert clone.handle_unknown == "error"


def test_output_feeds_ffm_classifier():
    rng = np.random.default_rng(0)
    Xint = rng.integers(0, 4, size=(60, 3))
    y = (Xint[:, 0] + Xint[:, 1] >= 4).astype(int)
    enc = CategoricalEncoder()
    Xt = enc.fit_transform(Xint)
    model = FFMClassifier(n_factors=4, max_iter=30, random_state=0)
    model.fit(Xt, y, field_ids=enc.field_ids_)
    proba = model.predict_proba(Xt)
    assert proba.shape == (60, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
