"""FM estimators (docs/api_design.md).

Binary classification (logistic loss) and regression (squared loss), trained
through `_backend` (Rust when built, NumPy reference otherwise) with
batch_size=1, single-threaded. Training always runs in float64; learned
attributes are stored in the requested `dtype`.

Not implemented yet (raise NotImplementedError at fit time): multiclass,
mini-batches (batch_size != 1), early stopping, class_weight, sample_weight,
label_smoothing, save/load. See docs/roadmap.md.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import _backend
from ._base import ParamsMixin, check_is_fitted
from ._reference_train import OPTIMIZERS, init_fm_params, make_row_orders
from .losses import sigmoid

_PHASE4 = "lands in a later phase (see docs/roadmap.md)"


def _check_X(X, n_features=None):
    if not (sp.issparse(X) or isinstance(X, np.ndarray)):
        X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got ndim={X.ndim}")
    data = X.data if sp.issparse(X) else X
    if not np.all(np.isfinite(data)):
        raise ValueError("X contains NaN or infinity, which modern_fm does not accept")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but this estimator was fitted with {n_features}"
        )
    return X


def _check_binary_classes(y):
    """Return (classes_, y01 float64) for a binary target."""
    classes = np.unique(y)
    if classes.shape[0] < 2:
        raise ValueError("y must contain at least 2 classes")
    if classes.shape[0] > 2:
        raise NotImplementedError(
            f"multiclass classification (got {classes.shape[0]} classes) {_PHASE4}"
        )
    return classes, (y == classes[1]).astype(np.float64)


class _FMBase(ParamsMixin):
    def _validate_common(self):
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optimizer!r}; expected one of {OPTIMIZERS}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"unknown dtype {self.dtype!r}; expected 'float32' or 'float64'")
        if self.backend != "rust_cpu":
            raise ValueError(f"unknown backend {self.backend!r}; only 'rust_cpu' exists in v0.1")
        if self.batch_size != 1:
            raise NotImplementedError(f"mini-batch training (batch_size != 1) {_PHASE4}")
        if self.early_stopping:
            raise NotImplementedError(f"early_stopping {_PHASE4}")

    def _validate_fit_extras(self, sample_weight, eval_set):
        if sample_weight is not None:
            raise NotImplementedError(f"sample_weight {_PHASE4}")
        if eval_set is not None:
            raise NotImplementedError(f"eval_set {_PHASE4}")

    def _fit_core(self, X, y, loss):
        n_rows, n_features = X.shape
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (n_rows,):
            raise ValueError(f"y has shape {y.shape}, expected ({n_rows},)")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinity")
        rng = np.random.default_rng(self.random_state)
        params = init_fm_params(rng, n_features, self.n_factors, self.init_scale)
        row_orders = make_row_orders(rng, n_rows, self.max_iter)
        w0, w, V = _backend.fm_fit(
            X, y, params,
            loss=loss,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            l2_linear=self.l2_linear,
            l2_factors=self.l2_factors,
            row_orders=row_orders,
        )
        out_dtype = np.float32 if self.dtype == "float32" else np.float64
        self.w0_ = float(w0)
        self.w_ = w.astype(out_dtype)
        self.V_ = V.astype(out_dtype)
        self.n_features_in_ = n_features
        self.n_iter_ = self.max_iter
        return self

    def _raw_scores(self, X):
        check_is_fitted(self)
        X = _check_X(X, self.n_features_in_)
        return _backend.fm_predict_fast(X, self.w0_, self.w_, self.V_)

    def save_model(self, path):
        raise NotImplementedError(f"save_model {_PHASE4}")

    @classmethod
    def load_model(cls, path):
        raise NotImplementedError(f"load_model {_PHASE4}")


class FMClassifier(_FMBase):
    """Factorization Machine binary classifier (logistic loss).

    See docs/api_design.md and docs/math_spec.md. Multiclass (softmax) is a
    later phase."""

    def __init__(
        self,
        n_factors=16,
        loss="logistic",
        optimizer="adagrad",
        learning_rate=0.05,
        max_iter=100,
        batch_size=1,
        l2_linear=1e-5,
        l2_factors=1e-5,
        init_scale=0.01,
        label_smoothing=0.0,
        class_weight=None,
        early_stopping=False,
        validation_fraction=0.1,
        patience=10,
        min_delta=0.0,
        dtype="float32",
        backend="rust_cpu",
        random_state=None,
        n_jobs=-1,
        verbose=0,
    ):
        self.n_factors = n_factors
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.l2_linear = l2_linear
        self.l2_factors = l2_factors
        self.init_scale = init_scale
        self.label_smoothing = label_smoothing
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.dtype = dtype
        self.backend = backend
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, eval_set=None):
        self._validate_common()
        self._validate_fit_extras(sample_weight, eval_set)
        if self.class_weight is not None:
            raise NotImplementedError(f"class_weight {_PHASE4}")
        if self.label_smoothing != 0.0:
            raise NotImplementedError(f"label_smoothing {_PHASE4}")
        if self.loss == "softmax":
            raise NotImplementedError(f"multiclass (softmax) {_PHASE4}")
        if self.loss != "logistic":
            raise ValueError(f"unknown loss {self.loss!r} for FMClassifier")
        X = _check_X(X)
        self.classes_, y01 = _check_binary_classes(np.asarray(y))
        return self._fit_core(X, y01, "logistic")

    def decision_function(self, X):
        return self._raw_scores(X)

    def predict_proba(self, X):
        p = sigmoid(self._raw_scores(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        # _raw_scores first so check_is_fitted runs before classes_ is touched
        # (predict before fit must raise NotFittedError, not AttributeError).
        scores = self._raw_scores(X)
        return self.classes_[(scores >= 0.0).astype(int)]


class FMRegressor(_FMBase):
    """Factorization Machine regressor (squared loss)."""

    def __init__(
        self,
        n_factors=16,
        optimizer="adagrad",
        learning_rate=0.05,
        max_iter=100,
        batch_size=1,
        l2_linear=1e-5,
        l2_factors=1e-5,
        init_scale=0.01,
        early_stopping=False,
        validation_fraction=0.1,
        patience=10,
        min_delta=0.0,
        dtype="float32",
        backend="rust_cpu",
        random_state=None,
        n_jobs=-1,
        verbose=0,
    ):
        self.n_factors = n_factors
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.l2_linear = l2_linear
        self.l2_factors = l2_factors
        self.init_scale = init_scale
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.dtype = dtype
        self.backend = backend
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, eval_set=None):
        self._validate_common()
        self._validate_fit_extras(sample_weight, eval_set)
        X = _check_X(X)
        return self._fit_core(X, y, "squared")

    def predict(self, X):
        return self._raw_scores(X)
