"""pandas / polars DataFrame input.

DataFrame support comes for free from sklearn's ``validate_data`` (used in
fit/predict): columns are taken in order, ``feature_names_in_`` is recorded, and
a column reorder at predict time is rejected. These tests pin that behavior and
prove parity with the equivalent ndarray fit.
"""

import numpy as np
import pytest
from modern_fm import FFMClassifier, FMClassifier, FMRegressor


def _data(n=40, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y_bin = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_reg = X[:, 0] - 0.5 * X[:, 2]
    cols = [f"f{i}" for i in range(d)]
    return X, y_bin, y_reg, cols


def test_pandas_classifier_parity_and_feature_names():
    pd = pytest.importorskip("pandas")
    X, y_bin, _, cols = _data()
    df = pd.DataFrame(X, columns=cols)
    m_df = FMClassifier(max_iter=30, n_jobs=1, random_state=0).fit(df, pd.Series(y_bin))
    m_np = FMClassifier(max_iter=30, n_jobs=1, random_state=0).fit(X, y_bin)
    assert list(m_df.feature_names_in_) == cols
    assert m_df.n_features_in_ == len(cols)
    np.testing.assert_allclose(m_df.decision_function(df), m_np.decision_function(X), atol=1e-5)
    np.testing.assert_array_equal(m_df.predict(df), m_np.predict(X))


def test_pandas_column_reorder_rejected_at_predict():
    pd = pytest.importorskip("pandas")
    X, y_bin, _, cols = _data()
    df = pd.DataFrame(X, columns=cols)
    model = FMClassifier(max_iter=10, n_jobs=1, random_state=0).fit(df, y_bin)
    with pytest.raises(ValueError, match="feature names"):
        model.predict(df[cols[::-1]])


def test_pandas_regressor_and_ffm():
    pd = pytest.importorskip("pandas")
    X, y_bin, y_reg, cols = _data()
    df = pd.DataFrame(X, columns=cols)
    r_df = FMRegressor(max_iter=30, n_jobs=1, random_state=0).fit(df, y_reg)
    r_np = FMRegressor(max_iter=30, n_jobs=1, random_state=0).fit(X, y_reg)
    np.testing.assert_allclose(r_df.predict(df), r_np.predict(X), atol=1e-5)
    f_df = FFMClassifier(max_iter=20, n_jobs=1, random_state=0).fit(df, y_bin)
    assert list(f_df.feature_names_in_) == cols
    assert f_df.predict(df).shape == (X.shape[0],)


def test_polars_input_parity():
    pl = pytest.importorskip("polars")
    X, y_bin, _, cols = _data()
    df = pl.DataFrame({c: X[:, i] for i, c in enumerate(cols)})
    m_pl = FMClassifier(max_iter=30, n_jobs=1, random_state=0).fit(df, y_bin)
    m_np = FMClassifier(max_iter=30, n_jobs=1, random_state=0).fit(X, y_bin)
    assert list(m_pl.feature_names_in_) == cols
    np.testing.assert_allclose(m_pl.decision_function(df), m_np.decision_function(X), atol=1e-5)
    np.testing.assert_array_equal(m_pl.predict(df), m_np.predict(X))
