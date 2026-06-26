"""FM estimators (docs/api_design.md).

Binary classification (logistic loss), multiclass classification (softmax loss),
and regression (squared loss), trained through `_backend` (Rust when built, NumPy
reference otherwise). Configurable `batch_size` (mini-batch gradient averaging;
batch_size=1 is per-row SGD) and `n_jobs` rayon row-parallelism (binary/regression;
multiclass is serial in v0.2). Supports class_weight, sample_weight,
label_smoothing, early stopping/eval_set, and save/load. Training always runs in
float64; learned attributes are stored in the requested `dtype`.

Optimizers: SGD, AdaGrad, Adam ("adam", with beta_1/beta_2/epsilon), and
FTRL-Proximal ("ftrl", with l1_linear/l1_factors/ftrl_beta; L1 yields exact zeros).

Early stopping supports every optimizer except FTRL and works with multiclass;
those state hand-offs round-trip through the NumPy reference path. Not yet
supported (raises NotImplementedError at fit time): early stopping with the FTRL
optimizer. See docs/roadmap.md.
"""

from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

from . import _backend
from ._base import ModelIOMixin, ParamsMixin, check_is_fitted
from ._early_stop import normalize_eval_set, run_epochs, split_indices
from ._reference_train import (
    OPTIMIZERS,
    init_fm_multiclass_params,
    init_fm_params,
    make_row_orders,
    new_adam_state,
)
from .losses import logistic_loss, sigmoid, softmax, softmax_loss, squared_loss

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


def _check_sample_weight(sample_weight, n_rows):
    """Validate sample_weight, or return None when not given."""
    if sample_weight is None:
        return None
    sw = np.asarray(sample_weight, dtype=np.float64)
    if sw.shape != (n_rows,):
        raise ValueError(f"sample_weight has shape {sw.shape}, expected ({n_rows},)")
    if not np.all(np.isfinite(sw)):
        raise ValueError("sample_weight contains NaN or infinity")
    if np.any(sw < 0.0):
        raise ValueError("sample_weight must be non-negative")
    return sw


def _cw_lookup(class_weight, label):
    if label in class_weight:
        return float(class_weight[label])
    key = label.item() if hasattr(label, "item") else label
    return float(class_weight.get(key, 1.0))


def _class_weight_vector(y01, classes, class_weight):
    """Per-row weight from class_weight ('balanced' or {label: weight} dict)."""
    yi = y01.astype(int)
    if isinstance(class_weight, str) and class_weight == "balanced":
        counts = np.bincount(yi, minlength=len(classes)).astype(np.float64)
        per_class = len(yi) / (len(classes) * np.maximum(counts, 1.0))
    elif isinstance(class_weight, dict):
        per_class = np.array([_cw_lookup(class_weight, c) for c in classes], dtype=np.float64)
    else:
        raise ValueError(
            f"class_weight must be None, 'balanced', or a dict; got {class_weight!r}"
        )
    return per_class[yi]


def _combine_weights(y01, classes, sample_weight, class_weight, n_rows):
    """Fold class_weight into sample_weight; returns per-row weights or None."""
    sw = _check_sample_weight(sample_weight, n_rows)
    if class_weight is None:
        return sw
    cw = _class_weight_vector(y01, classes, class_weight)
    return cw if sw is None else sw * cw


def _smooth(y01, label_smoothing):
    """y_smooth = y*(1-eps) + 0.5*eps (docs/math_spec.md); identity when eps=0."""
    if not label_smoothing:
        return y01
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
    return y01 * (1.0 - label_smoothing) + 0.5 * label_smoothing


def _resolve_n_jobs(n_jobs):
    """Rust thread count: -1 -> all cores, otherwise a positive integer.

    n_jobs=1 is the serial, bit-reproducible path; n_jobs>1 splits each batch
    across threads (reproducible for a fixed thread count, see optimization_spec.md).
    """
    if n_jobs == -1:
        return os.cpu_count() or 1
    if not (isinstance(n_jobs, (int, np.integer)) and n_jobs >= 1):
        raise ValueError(f"n_jobs must be -1 or a positive integer, got {n_jobs!r}")
    return int(n_jobs)


def _no_ftrl_early_stopping(optimizer, early_stopping, eval_set):
    """FTRL keeps per-coordinate (z, n) state that is not round-tripped across
    epochs, so it is incompatible with early stopping in v0.2. (Adam IS supported
    via the reference-path moment round-trip in `_fit_es`; see docs/roadmap.md.)"""
    if optimizer == "ftrl" and (early_stopping or eval_set is not None):
        raise NotImplementedError(f"early stopping with the ftrl optimizer {_PHASE4}")


class _FMBase(ModelIOMixin, ParamsMixin):
    def _validate_common(self):
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

    def _fit_core(self, X, y, loss, sample_weight=None):
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
            l1_linear=self.l1_linear,
            l1_factors=self.l1_factors,
            row_orders=row_orders,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            ftrl_beta=self.ftrl_beta,
            batch_size=self.batch_size,
            n_jobs=_resolve_n_jobs(self.n_jobs),
            sample_weight=sample_weight,
        )
        out_dtype = np.float32 if self.dtype == "float32" else np.float64
        self.w0_ = float(w0)
        self.w_ = w.astype(out_dtype)
        self.V_ = V.astype(out_dtype)
        self.n_features_in_ = n_features
        self.n_iter_ = self.max_iter
        return self

    def _fit_es(self, X, y_train, y_eval, loss, sample_weight, eval_val):
        """Epoch-by-epoch fit with early stopping (binary FM / regressor).

        `y_train` are the training targets (smoothed for logistic); `y_eval`
        the true targets used for the validation metric. `eval_val` is a
        prepared (X_val, y_val) pair, or None to split off validation_fraction.
        """
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
        init = init_fm_params(rng, n_features, self.n_factors, self.init_scale)
        row_orders = make_row_orders(rng, n_tr, self.max_iter)
        # AdaGrad/SGD round-trip accumulators via `state`; Adam round-trips its
        # moments via `adam_state` (which routes through the NumPy reference).
        is_adam = self.optimizer == "adam"
        state = (
            None if is_adam
            else [0.0, np.zeros(n_features), np.zeros((n_features, self.n_factors))]
        )
        adam_state = new_adam_state(init[0], init[1], init[2]) if is_adam else None
        work = [init[0], init[1], init[2]]
        metric = logistic_loss if loss == "logistic" else squared_loss

        def train_epoch(e):
            work[0], work[1], work[2] = _backend.fm_fit(
                X_tr, y_tr, (work[0], work[1], work[2]), loss=loss, optimizer=self.optimizer,
                learning_rate=self.learning_rate, l2_linear=self.l2_linear,
                l2_factors=self.l2_factors, l1_linear=self.l1_linear, l1_factors=self.l1_factors,
                row_orders=row_orders[e : e + 1], ftrl_beta=self.ftrl_beta,
                batch_size=self.batch_size, n_jobs=_resolve_n_jobs(self.n_jobs),
                sample_weight=sw_tr, state=state, adam_state=adam_state,
            )

        def evaluate():
            return metric(y_val, _backend.fm_predict_fast(X_val, work[0], work[1], work[2]))

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
        return _backend.fm_predict_fast(X, self.w0_, self.w_, self.V_)


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
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
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

    def fit(self, X, y, sample_weight=None, eval_set=None):
        self._validate_common()
        _no_ftrl_early_stopping(self.optimizer, self.early_stopping, eval_set)
        if self.loss not in ("logistic", "softmax"):
            raise ValueError(f"unknown loss {self.loss!r} for FMClassifier")
        X = _check_X(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] < 2:
            raise ValueError("y must contain at least 2 classes")
        if self.classes_.shape[0] > 2 or self.loss == "softmax":
            if self.early_stopping or eval_set is not None:
                eval_val = None
                if eval_set is not None:
                    Xv, yv = normalize_eval_set(eval_set)
                    eval_val = (
                        _check_X(Xv, X.shape[1]),
                        np.searchsorted(self.classes_, np.asarray(yv)),
                    )
                return self._fit_multiclass_es(X, y, sample_weight, eval_val)
            return self._fit_multiclass(X, y, sample_weight)
        y01 = (y == self.classes_[1]).astype(np.float64)
        sw = _combine_weights(y01, self.classes_, sample_weight, self.class_weight, X.shape[0])
        y_train = _smooth(y01, self.label_smoothing)
        if self.early_stopping or eval_set is not None:
            eval_val = None
            if eval_set is not None:
                Xv, yv = normalize_eval_set(eval_set)
                eval_val = (_check_X(Xv, X.shape[1]), (yv == self.classes_[1]).astype(np.float64))
            return self._fit_es(X, y_train, y01, "logistic", sw, eval_val)
        return self._fit_core(X, y_train, "logistic", sample_weight=sw)

    def _fit_multiclass(self, X, y, sample_weight):
        n_rows, n_features = X.shape
        n_classes = self.classes_.shape[0]
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        y_idx = np.searchsorted(self.classes_, y)  # classes_ is sorted (np.unique)
        sw = _combine_weights(y_idx, self.classes_, sample_weight, self.class_weight, n_rows)
        rng = np.random.default_rng(self.random_state)
        params = init_fm_multiclass_params(
            rng, n_classes, n_features, self.n_factors, self.init_scale
        )
        row_orders = make_row_orders(rng, n_rows, self.max_iter)
        w0, w, V = _backend.fm_fit_multiclass(
            X, y_idx, params,
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

    def _fit_multiclass_es(self, X, y, sample_weight, eval_val):
        """Epoch-by-epoch multiclass fit with early stopping (softmax cross-entropy
        metric). The Rust multiclass kernel keeps its state internal, so the
        per-epoch optimizer-state hand-off runs on the NumPy reference path."""
        n_rows, n_features = X.shape
        n_classes = self.classes_.shape[0]
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        y_idx = np.searchsorted(self.classes_, y)
        sw = _combine_weights(y_idx, self.classes_, sample_weight, self.class_weight, n_rows)
        rng = np.random.default_rng(self.random_state)
        if eval_val is None:
            tr_idx, val_idx = split_indices(n_rows, self.validation_fraction, rng)
            X_tr, y_tr = X[tr_idx], y_idx[tr_idx]
            sw_tr = None if sw is None else np.asarray(sw)[tr_idx]
            X_val, y_val = X[val_idx], y_idx[val_idx]
        else:
            X_tr, y_tr, sw_tr = X, y_idx, sw
            X_val, y_val = eval_val
        n_tr = X_tr.shape[0]
        init = init_fm_multiclass_params(
            rng, n_classes, n_features, self.n_factors, self.init_scale
        )
        row_orders = make_row_orders(rng, n_tr, self.max_iter)
        is_adam = self.optimizer == "adam"
        state = None if is_adam else [
            np.zeros(n_classes), np.zeros((n_classes, n_features)),
            np.zeros((n_classes, n_features, self.n_factors)),
        ]
        adam_state = new_adam_state(init[0], init[1], init[2]) if is_adam else None
        work = [init[0], init[1], init[2]]

        def train_epoch(e):
            work[0], work[1], work[2] = _backend.fm_fit_multiclass(
                X_tr, y_tr, (work[0], work[1], work[2]), optimizer=self.optimizer,
                learning_rate=self.learning_rate, l2_linear=self.l2_linear,
                l2_factors=self.l2_factors, l1_linear=self.l1_linear, l1_factors=self.l1_factors,
                row_orders=row_orders[e : e + 1], label_smoothing=self.label_smoothing,
                beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon,
                ftrl_beta=self.ftrl_beta, batch_size=self.batch_size, sample_weight=sw_tr,
                state=state, adam_state=adam_state,
            )

        def evaluate():
            logits = np.column_stack(
                [
                    _backend.fm_predict_fast(X_val, float(work[0][c]), work[1][c], work[2][c])
                    for c in range(n_classes)
                ]
            )
            return softmax_loss(y_val, logits)  # true-label cross-entropy

        def snapshot():
            return (work[0].copy(), work[1].copy(), work[2].copy())

        best, n_iter = run_epochs(
            self.max_iter, self.patience, self.min_delta, train_epoch, evaluate, snapshot
        )
        w0, w, V = best
        out_dtype = np.float32 if self.dtype == "float32" else np.float64
        self.w0_ = w0.astype(out_dtype)
        self.w_ = w.astype(out_dtype)
        self.V_ = V.astype(out_dtype)
        self.n_features_in_ = n_features
        self.n_iter_ = n_iter
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = _check_X(X, self.n_features_in_)
        if self.V_.ndim == 3:  # multiclass: per-class FM logits -> (n, n_classes)
            return np.column_stack(
                [
                    _backend.fm_predict_fast(X, float(self.w0_[c]), self.w_[c], self.V_[c])
                    for c in range(self.V_.shape[0])
                ]
            )
        return _backend.fm_predict_fast(X, self.w0_, self.w_, self.V_)

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


class FMRegressor(_FMBase):
    """Factorization Machine regressor (squared loss)."""

    def __init__(
        self,
        n_factors=16,
        optimizer="adagrad",
        learning_rate=0.05,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
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
        l1_linear=0.0,
        l1_factors=0.0,
        ftrl_beta=1.0,
    ):
        self.n_factors = n_factors
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

    def fit(self, X, y, sample_weight=None, eval_set=None):
        self._validate_common()
        _no_ftrl_early_stopping(self.optimizer, self.early_stopping, eval_set)
        X = _check_X(X)
        sw = _check_sample_weight(sample_weight, X.shape[0])
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (X.shape[0],):
            raise ValueError(f"y has shape {y.shape}, expected ({X.shape[0]},)")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinity")
        if self.early_stopping or eval_set is not None:
            eval_val = None
            if eval_set is not None:
                Xv, yv = normalize_eval_set(eval_set)
                eval_val = (_check_X(Xv, X.shape[1]), np.asarray(yv, dtype=np.float64))
            return self._fit_es(X, y, y, "squared", sw, eval_val)
        return self._fit_core(X, y, "squared", sample_weight=sw)

    def predict(self, X):
        return self._raw_scores(X)
