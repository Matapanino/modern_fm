"""modern_fm basic usage: FM/FFM classification, regression, and the encoder.

Run from the repo root after `pip install -e .`:

    .venv/bin/python examples/basic_usage.py
"""

import numpy as np
from modern_fm import CategoricalEncoder, FFMClassifier, FMClassifier, FMRegressor


def fm_binary_classification():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 10))
    y = (X @ rng.normal(size=10) > 0).astype(int)
    clf = FMClassifier(n_factors=8, max_iter=50, learning_rate=0.1, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    acc = (clf.predict(X) == y).mean()
    print(f"FMClassifier  train acc={acc:.3f}  proba row sum={proba.sum(axis=1)[0]:.3f}")


def fm_multiclass():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 10))
    y = np.argmax(X @ rng.normal(size=(10, 3)), axis=1)  # 3 classes
    clf = FMClassifier(n_factors=6, max_iter=60, learning_rate=0.1, random_state=0)
    clf.fit(X, y)  # softmax path is selected automatically for >2 classes
    print(f"FMClassifier (multiclass)  train acc={(clf.predict(X) == y).mean():.3f}")


def fm_regression_with_early_stopping():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 8))
    y = X @ rng.normal(size=8) + 0.1 * rng.normal(size=500)
    reg = FMRegressor(
        n_factors=8, max_iter=200, learning_rate=0.05, random_state=0,
        early_stopping=True, validation_fraction=0.2, patience=10,
    )
    reg.fit(X, y)
    rmse = np.sqrt(np.mean((reg.predict(X) - y) ** 2))
    print(f"FMRegressor   stopped at n_iter={reg.n_iter_}  train rmse={rmse:.3f}")


def ffm_with_categorical_encoder():
    rng = np.random.default_rng(3)
    # three integer categorical columns -> one-hot CSR + field ids
    X_cat = rng.integers(0, 5, size=(600, 3))
    y = ((X_cat[:, 0] + X_cat[:, 1]) >= 5).astype(int)
    enc = CategoricalEncoder()
    X_csr = enc.fit_transform(X_cat)
    ffm = FFMClassifier(n_factors=4, max_iter=40, learning_rate=0.1, random_state=0)
    ffm.fit(X_csr, y, field_ids=enc.field_ids_)
    print(
        f"FFMClassifier n_fields={ffm.n_fields_} n_features={ffm.n_features_in_} "
        f"train acc={(ffm.predict(X_csr) == y).mean():.3f}"
    )


def save_and_load(tmp="/tmp/modern_fm_demo.bin"):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 5))
    y = (X[:, 0] > 0).astype(int)
    clf = FMClassifier(max_iter=20, random_state=0).fit(X, y)
    clf.save_model(tmp)
    reloaded = FMClassifier.load_model(tmp)
    same = np.array_equal(reloaded.predict(X), clf.predict(X))
    print(f"save_model/load_model round-trip predictions match: {same}")


if __name__ == "__main__":
    fm_binary_classification()
    fm_multiclass()
    fm_regression_with_early_stopping()
    ffm_with_categorical_encoder()
    save_and_load()
