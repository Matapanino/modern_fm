"""FFM estimator (docs/api_design.md).

Binary classification with logistic loss, trained through `_backend`
(batch_size=1, single-threaded). `field_ids` is explicit and required at fit
time — no automatic field inference in v0.1; the mapping is stored on the
model (`field_ids_`, `n_fields_`) so predict-time calls do not take it.

Training runs in float64; learned attributes are stored in the requested
`dtype`. Supports class_weight, sample_weight, label_smoothing, early
stopping/eval_set, save/load, and the SGD/AdaGrad/Adam optimizers. Multiclass
(softmax), mini-batches (batch_size != 1), and early stopping combined with
the Adam optimizer raise NotImplementedError (see docs/roadmap.md).
"""

from __future__ import annotations

import numpy as np

from . import _backend
from ._base import ModelIOMixin, ParamsMixin, check_is_fitted
from ._early_stop import normalize_eval_set, run_epochs, split_indices
from ._reference_train import OPTIMIZERS, init_ffm_params, make_row_orders
from .fm import (
    _check_binary_classes,
    _check_X,
    _combine_weights,
    _no_adam_early_stopping,
    _smooth,
)
from .losses import logistic_loss, sigmoid

_PHASE4 = "lands in a later phase (see docs/roadmap.md)"


class FFMClassifier(ModelIOMixin, ParamsMixin):
    """Field-aware Factorization Machine binary classifier.
    See docs/api_design.md and docs/math_spec.md."""

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
        if self.loss == "softmax":
            raise NotImplementedError(f"multiclass (softmax) {_PHASE4}")
        if self.loss != "logistic":
            raise ValueError(f"unknown loss {self.loss!r} for FFMClassifier")
        _no_adam_early_stopping(self.optimizer, self.early_stopping, eval_set)

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
            row_orders=row_orders,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
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
        state = [0.0, np.zeros(n_features), np.zeros((n_features, self.n_fields_, self.n_factors))]
        work = [init[0], init[1], init[2]]

        def train_epoch(e):
            work[0], work[1], work[2] = _backend.ffm_fit(
                X_tr, y_tr, self.field_ids_, (work[0], work[1], work[2]),
                optimizer=self.optimizer, learning_rate=self.learning_rate,
                l2_linear=self.l2_linear, l2_factors=self.l2_factors,
                row_orders=row_orders[e : e + 1], sample_weight=sw_tr, state=state,
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
