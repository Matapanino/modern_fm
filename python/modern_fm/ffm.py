"""FFM estimator (docs/api_design.md).

Binary (logistic) and multiclass (softmax, one FFM per class) classification,
trained through `_backend` with configurable `batch_size` (mini-batch gradient
averaging) and `n_jobs` rayon row-parallelism (binary; multiclass is serial).
`field_ids` may be passed at fit time (one field id per feature/column); when
omitted, each column becomes its own field (so `fit(X, y)` works under the
sklearn API). The mapping is stored on the model (`field_ids_`, `n_fields_`) so
predict-time calls do not take it.

Training runs in float64; learned attributes are stored in the requested
`dtype`. Supports class_weight, sample_weight, label_smoothing, early
stopping/eval_set, save/load, and the SGD/AdaGrad/Adam/FTRL optimizers (FTRL
adds l1_linear/l1_factors/ftrl_beta). Binary Adam + early stopping rounds the
moments through the NumPy reference path. Early stopping with the FTRL optimizer
or with multiclass raises NotImplementedError (see docs/roadmap.md).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_consistent_length, column_or_1d

from . import _backend
from ._base import ModelIOMixin, check_is_fitted
from ._early_stop import normalize_eval_set, run_epochs, split_indices
from ._reference_train import (
    OPTIMIZERS,
    init_ffm_multiclass_params,
    init_ffm_params,
    make_row_orders,
    new_adam_state,
)
from .fm import (
    _check_X,
    _combine_weights,
    _no_ftrl_early_stopping,
    _resolve_n_jobs,
    _smooth,
    _validate_X,
)
from .losses import logistic_loss, sigmoid, softmax

_PHASE4 = "lands in a later phase (see docs/roadmap.md)"


class FFMClassifier(ClassifierMixin, BaseEstimator, ModelIOMixin):
    """Field-aware Factorization Machine binary classifier.
    See docs/api_design.md and docs/math_spec.md."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def __init__(
        self,
        n_factors=8,
        loss="logistic",
        optimizer="adagrad",
        learning_rate=0.05,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
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
        l1_linear=0.0,
        l1_factors=0.0,
        ftrl_beta=1.0,
    ):
        self.n_factors = n_factors
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
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
        self.l1_linear = l1_linear
        self.l1_factors = l1_factors
        self.ftrl_beta = ftrl_beta

    def fit(self, X, y, field_ids=None, sample_weight=None, eval_set=None):
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optimizer!r}; expected one of {OPTIMIZERS}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"unknown dtype {self.dtype!r}; expected 'float32' or 'float64'")
        if self.backend != "rust_cpu":
            raise ValueError(f"unknown backend {self.backend!r}; only 'rust_cpu' exists in v0.1")
        if not (isinstance(self.batch_size, (int, np.integer)) and self.batch_size >= 1):
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size!r}")
        _resolve_n_jobs(self.n_jobs)  # validate (raises on a bad n_jobs)
        if (self.l1_linear or self.l1_factors) and self.optimizer != "ftrl":
            raise ValueError("l1_linear/l1_factors are only used by optimizer='ftrl'")
        if self.loss not in ("logistic", "softmax"):
            raise ValueError(f"unknown loss {self.loss!r} for FFMClassifier")
        _no_ftrl_early_stopping(self.optimizer, self.early_stopping, eval_set)

        X = _validate_X(self, X, reset=True)
        n_rows, n_features = X.shape
        if field_ids is None:
            field_ids = np.arange(n_features, dtype=np.int64)  # default: each column its own field
        field_ids = np.asarray(field_ids, dtype=np.int64)
        if field_ids.shape != (n_features,):
            raise ValueError(
                f"field_ids has shape {field_ids.shape}, expected ({n_features},) "
                "(one field id per feature/column)"
            )
        if field_ids.min() < 0:
            raise ValueError("field_ids must be non-negative integers")
        self.field_ids_ = field_ids
        self.n_fields_ = int(field_ids.max()) + 1
        y = column_or_1d(y, warn=True)
        check_consistent_length(X, y)
        check_classification_targets(y)
        self.classes_ = unique_labels(y)
        if self.classes_.shape[0] < 2:
            raise ValueError(
                "Classifier can't train when only one class is present in y; "
                "need at least 2 classes."
            )
        if self.classes_.shape[0] > 2 or self.loss == "softmax":
            if self.early_stopping or eval_set is not None:
                raise NotImplementedError(f"early stopping with multiclass FFM {_PHASE4}")
            return self._fit_multiclass(X, y, field_ids, sample_weight)
        y01 = (np.asarray(y) == self.classes_[1]).astype(np.float64)
        sw = _combine_weights(y01, self.classes_, sample_weight, self.class_weight, n_rows)
        y_train = _smooth(y01, self.label_smoothing)

        if self.early_stopping or eval_set is not None:
            eval_val = None
            if eval_set is not None:
                Xv, yv = normalize_eval_set(eval_set)
                eval_val = (_check_X(Xv, n_features), (yv == self.classes_[1]).astype(np.float64))
            return self._fit_es(X, y_train, y01, sw, eval_val)

        rng = np.random.default_rng(self.random_state)
        params = init_ffm_params(
            rng, n_features, self.n_fields_, self.n_factors, self.init_scale
        )
        row_orders = make_row_orders(rng, n_rows, self.max_iter)
        w0, w, V = _backend.ffm_fit(
            X, y_train, field_ids, params,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            l2_linear=self.l2_linear,
            l2_factors=self.l2_factors,
            l1_linear=self.l1_linear,
            l1_factors=self.l1_factors,
            row_orders=row_orders,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            ftrl_beta=self.ftrl_beta,
            batch_size=self.batch_size,
            n_jobs=_resolve_n_jobs(self.n_jobs),
            sample_weight=sw,
        )
        out_dtype = np.float32 if self.dtype == "float32" else np.float64
        self.w0_ = float(w0)
        self.w_ = w.astype(out_dtype)
        self.V_ = V.astype(out_dtype)
        self.n_features_in_ = n_features
        self.n_iter_ = self.max_iter
        return self

    def _fit_es(self, X, y_train, y_eval, sample_weight, eval_val):
        """Epoch-by-epoch FFM fit with early stopping (logistic logloss metric)."""
        n_rows, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        if eval_val is None:
            tr_idx, val_idx = split_indices(n_rows, self.validation_fraction, rng)
            X_tr, y_tr = X[tr_idx], y_train[tr_idx]
            sw_tr = None if sample_weight is None else np.asarray(sample_weight)[tr_idx]
            X_val, y_val = X[val_idx], y_eval[val_idx]
        else:
            X_tr, y_tr, sw_tr = X, y_train, sample_weight
            X_val, y_val = eval_val
        n_tr = X_tr.shape[0]
        init = init_ffm_params(rng, n_features, self.n_fields_, self.n_factors, self.init_scale)
        row_orders = make_row_orders(rng, n_tr, self.max_iter)
        # Adam round-trips its moments via `adam_state` (NumPy reference path);
        # SGD/AdaGrad round-trip accumulators via `state`.
        is_adam = self.optimizer == "adam"
        state = (
            None if is_adam
            else [0.0, np.zeros(n_features), np.zeros((n_features, self.n_fields_, self.n_factors))]
        )
        adam_state = new_adam_state(init[0], init[1], init[2]) if is_adam else None
        work = [init[0], init[1], init[2]]

        def train_epoch(e):
            work[0], work[1], work[2] = _backend.ffm_fit(
                X_tr, y_tr, self.field_ids_, (work[0], work[1], work[2]),
                optimizer=self.optimizer, learning_rate=self.learning_rate,
                l2_linear=self.l2_linear, l2_factors=self.l2_factors,
                l1_linear=self.l1_linear, l1_factors=self.l1_factors,
                row_orders=row_orders[e : e + 1], ftrl_beta=self.ftrl_beta,
                batch_size=self.batch_size, n_jobs=_resolve_n_jobs(self.n_jobs),
                sample_weight=sw_tr, state=state, adam_state=adam_state,
            )

        def evaluate():
            scores = _backend.ffm_predict(X_val, self.field_ids_, work[0], work[1], work[2])
            return logistic_loss(y_val, scores)

        def snapshot():
            return (work[0], work[1].copy(), work[2].copy())

        best, n_iter = run_epochs(
            self.max_iter, self.patience, self.min_delta, train_epoch, evaluate, snapshot
        )
        w0, w, V = best
        out_dtype = np.float32 if self.dtype == "float32" else np.float64
        self.w0_ = float(w0)
        self.w_ = w.astype(out_dtype)
        self.V_ = V.astype(out_dtype)
        self.n_features_in_ = n_features
        self.n_iter_ = n_iter
        return self

    def _fit_multiclass(self, X, y, field_ids, sample_weight):
        n_rows, n_features = X.shape
        n_classes = self.classes_.shape[0]
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        y_idx = np.searchsorted(self.classes_, y)  # classes_ is sorted (np.unique)
        sw = _combine_weights(y_idx, self.classes_, sample_weight, self.class_weight, n_rows)
        rng = np.random.default_rng(self.random_state)
        params = init_ffm_multiclass_params(
            rng, n_classes, n_features, self.n_fields_, self.n_factors, self.init_scale
        )
        row_orders = make_row_orders(rng, n_rows, self.max_iter)
        w0, w, V = _backend.ffm_fit_multiclass(
            X, y_idx, field_ids, params,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            l2_linear=self.l2_linear,
            l2_factors=self.l2_factors,
            l1_linear=self.l1_linear,
            l1_factors=self.l1_factors,
            row_orders=row_orders,
            label_smoothing=self.label_smoothing,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            ftrl_beta=self.ftrl_beta,
            batch_size=self.batch_size,
            sample_weight=sw,
        )
        out_dtype = np.float32 if self.dtype == "float32" else np.float64
        self.w0_ = w0.astype(out_dtype)
        self.w_ = w.astype(out_dtype)
        self.V_ = V.astype(out_dtype)
        self.n_features_in_ = n_features
        self.n_iter_ = self.max_iter
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = _validate_X(self, X, reset=False)
        if self.V_.ndim == 4:  # multiclass: per-class FFM logits -> (n, n_classes)
            return np.column_stack(
                [
                    _backend.ffm_predict(
                        X, self.field_ids_, float(self.w0_[c]), self.w_[c], self.V_[c]
                    )
                    for c in range(self.V_.shape[0])
                ]
            )
        return _backend.ffm_predict(X, self.field_ids_, self.w0_, self.w_, self.V_)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1:  # binary logistic
            p = sigmoid(scores)
            return np.column_stack([1.0 - p, p])
        return softmax(scores)  # multiclass softmax, rows sum to 1

    def predict(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1:
            return self.classes_[(scores >= 0.0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]
