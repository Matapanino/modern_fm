"""libffm text-format loader/exporter round-trips."""

import numpy as np
import pytest
import scipy.sparse as sp
from modern_fm import FFMClassifier, dump_libffm, load_libffm


def test_load_parses_known_text(tmp_path):
    p = tmp_path / "a.ffm"
    p.write_text("1 0:0:1 1:2:0.5\n0 0:1:1 1:2:2\n")
    X, y, fields = load_libffm(p)
    assert X.shape == (2, 3)
    np.testing.assert_array_equal(y, [1.0, 0.0])
    np.testing.assert_array_equal(fields, [0, 0, 1])  # feat0,feat1->field0; feat2->field1
    np.testing.assert_allclose(X.toarray(), [[1.0, 0.0, 0.5], [0.0, 1.0, 2.0]])


def test_round_trip_exact(tmp_path):
    rng = np.random.default_rng(0)
    M = rng.integers(0, 3, size=(20, 5)).astype(np.float64)
    M[0, :] = 1.0  # every column appears (libffm infers n_features from indices)
    X = sp.csr_matrix(M)
    y = rng.integers(0, 2, size=20).astype(np.float64)
    fields = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    p = tmp_path / "rt.ffm"
    dump_libffm(p, X, y, fields)
    X2, y2, f2 = load_libffm(p)
    np.testing.assert_array_equal(y, y2)
    np.testing.assert_array_equal(fields, f2)
    np.testing.assert_allclose(X.toarray(), X2.toarray())


def test_round_trip_float_values(tmp_path):
    X = sp.csr_matrix(np.array([[0.25, 0.0, 1.0], [0.0, 1.5, 1.0]]))
    p = tmp_path / "f.ffm"
    dump_libffm(p, X, np.array([1.0, 0.0]), np.array([0, 1, 1], dtype=np.int64))
    X2, _, f2 = load_libffm(p)
    np.testing.assert_allclose(X.toarray(), X2.toarray())  # repr() is round-trip exact
    np.testing.assert_array_equal(f2, [0, 1, 1])


def test_loaded_data_trains_ffm(tmp_path):
    rng = np.random.default_rng(1)
    M = (rng.random((60, 6)) < 0.5).astype(np.float64)
    M[0, :] = 1.0
    X = sp.csr_matrix(M)
    y = (M[:, 1] + M[:, 2] > 0).astype(np.float64)
    fields = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    p = tmp_path / "train.ffm"
    dump_libffm(p, X, y, fields)
    Xl, yl, fl = load_libffm(p)
    model = FFMClassifier(max_iter=20, n_jobs=1, random_state=0).fit(Xl, yl, field_ids=fl)
    assert model.predict(Xl).shape == (60,)


def test_load_rejects_bad_token(tmp_path):
    p = tmp_path / "bad.ffm"
    p.write_text("1 0:0:1 not_a_triple\n")
    with pytest.raises(ValueError, match="field:feature:value"):
        load_libffm(p)


def test_dump_validates_shapes(tmp_path):
    X = sp.csr_matrix(np.ones((3, 2)))
    with pytest.raises(ValueError, match="field_ids"):
        dump_libffm(tmp_path / "x.ffm", X, np.zeros(3), np.array([0]))
    with pytest.raises(ValueError, match="y has shape"):
        dump_libffm(tmp_path / "x.ffm", X, np.zeros(5), np.array([0, 1]))
