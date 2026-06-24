"""FFM estimator (docs/api_design.md).

Binary classification with logistic loss, trained through `_backend`
(batch_size=1, single-threaded). `field_ids` is explicit and required at fit
time — no automatic field inference in v0.1; the mapping is stored on the
model (`field_ids_`, `n_fields_`) so predict-time calls do not take it.

Training runs in float64; learned attributes are stored in the requested
`dtype`. Multiclass, mini-batches, early stopping, class_weight,
sample_weight, label_smoothing and save/load land in later phases.
"""

from __future__ import annotations

import numpy as np

from . import _backend
from ._base import ParamsMixin, check_is_fitted
from ._reference_train import OPTIMIZERS, init_ffm_params, make_row_orders
from .fm import _check_binary_classes, _check_X, _combine_weights, _smooth
from .losses import sigmoid

_PHASE4 = "lands in a later phase (see docs/roadmap.md)"


class FFMClassifier(ParamsMixin):
    """Field-aware Factorization Machine binary classifier.
    See docs/api_design.md and docs/math_spec.md."""

    def __init__(
        self,
        n_factors=8,
        loss="logistic",
        optimizer="adagrad",
        learning_rate=0.05,
        max_iter=50,
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

    def fit(self, X, y, field_ids=None, sample_weight=None, eval_set=None):
        if field_ids is None:
            raise ValueError(
                "FFMClassifier.fit requires field_ids (shape (n_features,)); "
                "automatic field inference is not supported in v0.1."
            )
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
        if eval_set is not None:
            raise NotImplementedError(f"eval_set {_PHASE4}")
        if self.loss == "softmax":
            raise NotImplementedError(f"multiclass (softmax) {_PHASE4}")
        if self.loss != "logistic":
            raise ValueError(f"unknown loss {self.loss!r} for FFMClassifier")

        X = _check_X(X)
        n_rows, n_features = X.shape
        field_ids = np.asarray(field_ids, dtype=np.int64)
        if field_ids.shape != (n_features,):
            raise ValueError(
                f"field_ids has shape {field_ids.shape}, expected ({n_features},) "
                "(one field id per feature/column)"
            )
        if field_ids.min() < 0:
            raise ValueError("field_ids must be non-negative integers")
        self.classes_, y01 = _check_binary_classes(np.asarray(y))
        self.field_ids_ = field_ids
        self.n_fields_ = int(field_ids.max()) + 1
        sw = _combine_weights(y01, self.classes_, sample_weight, self.class_weight, n_rows)

        rng = np.random.default_rng(self.random_state)
        params = init_ffm_params(
            rng, n_features, self.n_fields_, self.n_factors, self.init_scale
        )
        row_orders = make_row_orders(rng, n_rows, self.max_iter)
        w0, w, V = _backend.ffm_fit(
            X, _smooth(y01, self.label_smoothing), field_ids, params,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            l2_linear=self.l2_linear,
            l2_factors=self.l2_factors,
            row_orders=row_orders,
            sample_weight=sw,
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
        return _backend.ffm_predict(X, self.field_ids_, self.w0_, self.w_, self.V_)

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

    def save_model(self, path):
        raise NotImplementedError(f"save_model {_PHASE4}")

    @classmethod
    def load_model(cls, path):
        raise NotImplementedError(f"load_model {_PHASE4}")
