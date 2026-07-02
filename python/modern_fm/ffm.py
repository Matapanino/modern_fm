"""FFM estimators (docs/api_design.md).

Binary (logistic) and multiclass (softmax, one FFM per class) classification and
regression (squared loss), trained through `_backend` with configurable
`batch_size` (mini-batch gradient averaging) and `n_jobs` rayon row-parallelism
(binary/regression; multiclass is serial). `field_ids` may be passed at fit time
(one field id per feature/column); when omitted, each column becomes its own
field (so `fit(X, y)` works under the sklearn API). The mapping is stored on the
model (`field_ids_`, `n_fields_`) so predict-time calls do not take it.

Training runs in float64; learned attributes are stored in the requested
`dtype`. Supports class_weight, sample_weight, label_smoothing (classifier),
early stopping/eval_set, save/load, and the SGD/AdaGrad/Adam/FTRL optimizers
(FTRL adds l1_linear/l1_factors/ftrl_beta). Early stopping supports every
optimizer and both binary and multiclass; every per-epoch optimizer-state
hand-off round-trips through the Rust kernel (see docs/roadmap.md).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_consistent_length, column_or_1d

from . import _backend
from ._base import ModelIOMixin, check_is_fitted
from ._early_stop import normalize_eval_set, run_epochs, split_indices
from ._partial import make_opt_state, partial_fit_classes, warm_resume
from ._reference_train import (
    OPTIMIZERS,
    init_ffm_multiclass_params,
    init_ffm_params,
    make_row_orders,
    new_adam_state,
    new_ftrl_state,
)
from .fm import (
    _check_sample_weight,
    _check_X,
    _combine_weights,
    _resolve_n_jobs,
    _smooth,
    _validate_backend,
    _validate_X,
)
from .losses import logistic_loss, sigmoid, softmax, softmax_loss, squared_loss


class _FFMBase(BaseEstimator, ModelIOMixin):
    """Shared FFM machinery for FFMClassifier and FFMRegressor.

    Holds parameter validation, the `field_ids` -> (`field_ids_`, `n_fields_`)
    plumbing, the loss-parameterized core / early-stopping training loops, and
    raw FFM scoring. Mirrors `_FMBase`; subclasses add the task-specific fit
    target and prediction head.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def _validate_common(self):
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optimizer!r}; expected one of {OPTIMIZERS}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"unknown dtype {self.dtype!r}; expected 'float32' or 'float64'")
        _validate_backend(self.backend)
        if not (isinstance(self.batch_size, (int, np.integer)) and self.batch_size >= 1):
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size!r}")
        _resolve_n_jobs(self.n_jobs)  # validate (raises on a bad n_jobs)
        if (self.l1_linear or self.l1_factors) and self.optimizer != "ftrl":
            raise ValueError("l1_linear/l1_factors are only used by optimizer='ftrl'")

    def _set_field_ids(self, field_ids, n_features):
        """Validate `field_ids` (default: each column its own field) and store
        `field_ids_` / `n_fields_`; returns the int64 field id array."""
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
        return field_ids

    def _fit_core(self, X, y, loss, sample_weight=None):
        """Single all-epochs FFM fit; stores w0_/w_/V_/n_features_in_/n_iter_."""
        n_rows, n_features = X.shape
        resumed = warm_resume(self)
        rng = np.random.default_rng(self.random_state)
        if resumed is None:
            params = init_ffm_params(
                rng, n_features, self.n_fields_, self.n_factors, self.init_scale
            )
            opt = {}
        else:
            params, opt = resumed
        row_orders = make_row_orders(rng, n_rows, self.max_iter)
        w0, w, V = _backend.ffm_fit(
            X, y, self.field_ids_, params,
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
            **opt,
        )
        out_dtype = np.float32 if self.dtype == "float32" else np.float64
        self.w0_ = float(w0)
        self.w_ = w.astype(out_dtype)
        self.V_ = V.astype(out_dtype)
        self.n_features_in_ = n_features
        self.n_iter_ = self.max_iter
        return self

    def _fit_es(self, X, y_train, y_eval, loss, sample_weight, eval_val):
        """Epoch-by-epoch FFM fit with early stopping.

        Metric is logistic logloss for `loss="logistic"`, squared error for
        `loss="squared"`. `y_train` are the training targets (smoothed for
        logistic), `y_eval` the targets for the validation metric; `eval_val`
        is a prepared (X_val, y_val) pair, or None to split off
        validation_fraction.
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
        resumed = warm_resume(self)
        if resumed is None:
            init = init_ffm_params(rng, n_features, self.n_fields_, self.n_factors, self.init_scale)
            opt = None
        else:
            init, opt = resumed
        row_orders = make_row_orders(rng, n_tr, self.max_iter)
        # SGD/AdaGrad round-trip accumulators via `state`; Adam and FTRL round-trip
        # their state (moments / (z, n)) via `adam_state` / `ftrl_state` (NumPy path).
        # warm_start resumes the persisted state.
        is_adam = self.optimizer == "adam"
        is_ftrl = self.optimizer == "ftrl"
        if opt is not None:
            state, adam_state, ftrl_state = (
                opt.get("state"), opt.get("adam_state"), opt.get("ftrl_state")
            )
        else:
            state = (
                None if (is_adam or is_ftrl)
                else [0.0, np.zeros(n_features),
                      np.zeros((n_features, self.n_fields_, self.n_factors))]
            )
            adam_state = new_adam_state(init[0], init[1], init[2]) if is_adam else None
            ftrl_state = new_ftrl_state(init[0], init[1], init[2]) if is_ftrl else None
        work = [init[0], init[1], init[2]]
        metric = logistic_loss if loss == "logistic" else squared_loss

        def train_epoch(e):
            work[0], work[1], work[2] = _backend.ffm_fit(
                X_tr, y_tr, self.field_ids_, (work[0], work[1], work[2]),
                loss=loss, optimizer=self.optimizer, learning_rate=self.learning_rate,
                l2_linear=self.l2_linear, l2_factors=self.l2_factors,
                l1_linear=self.l1_linear, l1_factors=self.l1_factors,
                row_orders=row_orders[e : e + 1], ftrl_beta=self.ftrl_beta,
                batch_size=self.batch_size, n_jobs=_resolve_n_jobs(self.n_jobs),
                sample_weight=sw_tr, state=state, adam_state=adam_state, ftrl_state=ftrl_state,
            )

        def evaluate():
            scores = _backend.ffm_predict(X_val, self.field_ids_, work[0], work[1], work[2])
            return metric(y_val, scores)

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

    def _ffm_raw_scores(self, X):
        """Validated raw FFM score (single parameter set, V_.ndim == 3)."""
        check_is_fitted(self)
        X = _validate_X(self, X, reset=False)
        return _backend.ffm_predict(
            X, self.field_ids_, self.w0_, self.w_, self.V_, backend=self.backend
        )

    def _partial_field_ids(self, field_ids, n_features, first_call):
        """Set field_ids_/n_fields_ on the first partial_fit; on later calls reuse
        them and reject a re-passed field_ids that disagrees."""
        if first_call:
            return self._set_field_ids(field_ids, n_features)
        if field_ids is not None:
            fid = np.asarray(field_ids, dtype=np.int64)
            if not np.array_equal(fid, self.field_ids_):
                raise ValueError("field_ids passed to partial_fit differ from the first call")
        return self.field_ids_

    def _advance_one_epoch(self, X, y, loss, sample_weight, *, multiclass, first_call, n_classes):
        """One natural-order pass continuing ``self._opt_state`` (FFM partial_fit
        primitive); mirrors ``_FMBase._advance_one_epoch`` with field plumbing."""
        n_rows, n_features = X.shape
        out_dtype = np.float32 if self.dtype == "float32" else np.float64
        if first_call:
            rng = np.random.default_rng(self.random_state)
            if multiclass:
                w0, w, V = init_ffm_multiclass_params(
                    rng, n_classes, n_features, self.n_fields_, self.n_factors, self.init_scale
                )
            else:
                w0, w, V = init_ffm_params(
                    rng, n_features, self.n_fields_, self.n_factors, self.init_scale
                )
            self._opt_state = make_opt_state(self.optimizer, w0, w, V)
            self.n_iter_ = 0
        else:
            w0 = self.w0_.astype(np.float64) if multiclass else float(self.w0_)
            w = self.w_.astype(np.float64)
            V = self.V_.astype(np.float64)
            if getattr(self, "_opt_state", None) is None:
                self._opt_state = make_opt_state(self.optimizer, w0, w, V)
        row_orders = np.arange(n_rows, dtype=np.int64)[None, :]  # one pass, natural order
        common = dict(
            optimizer=self.optimizer, learning_rate=self.learning_rate,
            l2_linear=self.l2_linear, l2_factors=self.l2_factors,
            l1_linear=self.l1_linear, l1_factors=self.l1_factors, row_orders=row_orders,
            beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon,
            ftrl_beta=self.ftrl_beta, batch_size=self.batch_size, sample_weight=sample_weight,
        )
        if multiclass:
            w0, w, V = _backend.ffm_fit_multiclass(
                X, y, self.field_ids_, (w0, w, V), label_smoothing=self.label_smoothing,
                **common, **self._opt_state,
            )
            self.w0_ = w0.astype(out_dtype)
        else:
            w0, w, V = _backend.ffm_fit(
                X, y, self.field_ids_, (w0, w, V), loss=loss,
                n_jobs=_resolve_n_jobs(self.n_jobs), **common, **self._opt_state,
            )
            self.w0_ = float(w0)
        self.w_ = w.astype(out_dtype)
        self.V_ = V.astype(out_dtype)
        self.n_iter_ += 1
        return self


class FFMClassifier(ClassifierMixin, _FFMBase):
    """Field-aware Factorization Machine classifier (logistic / softmax).
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
        warm_start=False,
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
        self.warm_start = warm_start
        self.dtype = dtype
        self.backend = backend
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.l1_linear = l1_linear
        self.l1_factors = l1_factors
        self.ftrl_beta = ftrl_beta

    def fit(self, X, y, field_ids=None, sample_weight=None, eval_set=None):
        self._validate_common()
        if self.loss not in ("logistic", "softmax"):
            raise ValueError(f"unknown loss {self.loss!r} for FFMClassifier")

        X = _validate_X(self, X, reset=True)
        n_rows, n_features = X.shape
        field_ids = self._set_field_ids(field_ids, n_features)
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
                eval_val = None
                if eval_set is not None:
                    Xv, yv = normalize_eval_set(eval_set)
                    eval_val = (
                        _check_X(Xv, n_features),
                        np.searchsorted(self.classes_, np.asarray(yv)),
                    )
                return self._fit_multiclass_es(X, y, field_ids, sample_weight, eval_val)
            return self._fit_multiclass(X, y, field_ids, sample_weight)
        y01 = (np.asarray(y) == self.classes_[1]).astype(np.float64)
        sw = _combine_weights(y01, self.classes_, sample_weight, self.class_weight, n_rows)
        y_train = _smooth(y01, self.label_smoothing)

        if self.early_stopping or eval_set is not None:
            eval_val = None
            if eval_set is not None:
                Xv, yv = normalize_eval_set(eval_set)
                eval_val = (_check_X(Xv, n_features), (yv == self.classes_[1]).astype(np.float64))
            return self._fit_es(X, y_train, y01, "logistic", sw, eval_val)
        return self._fit_core(X, y_train, "logistic", sample_weight=sw)

    def partial_fit(self, X, y, classes=None, field_ids=None, sample_weight=None):
        """Incremental fit on a chunk: one pass in natural row order, continuing the
        optimizer state. Pass ``classes`` (all labels) on the first call; ``field_ids``
        (the field map) is set on the first call and validated thereafter. See
        docs/api_design.md."""
        self._validate_common()
        if self.loss not in ("logistic", "softmax"):
            raise ValueError(f"unknown loss {self.loss!r} for FFMClassifier")
        if isinstance(self.class_weight, str) and self.class_weight == "balanced":
            raise ValueError("class_weight='balanced' is not supported by partial_fit")
        first_call = not hasattr(self, "n_features_in_")
        X = _validate_X(self, X, reset=first_call)
        self._partial_field_ids(field_ids, X.shape[1], first_call)
        _, y = partial_fit_classes(self, y, classes)
        check_consistent_length(X, y)
        n_classes = self.classes_.shape[0]
        multiclass = n_classes > 2 or self.loss == "softmax"
        if multiclass:
            if not 0.0 <= self.label_smoothing < 1.0:
                raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
            y_target = np.searchsorted(self.classes_, y)
            sw = _combine_weights(
                y_target, self.classes_, sample_weight, self.class_weight, X.shape[0]
            )
            return self._advance_one_epoch(
                X, y_target, "softmax", sw, multiclass=True, first_call=first_call,
                n_classes=n_classes,
            )
        y01 = (y == self.classes_[1]).astype(np.float64)
        sw = _combine_weights(y01, self.classes_, sample_weight, self.class_weight, X.shape[0])
        y_train = _smooth(y01, self.label_smoothing)
        return self._advance_one_epoch(
            X, y_train, "logistic", sw, multiclass=False, first_call=first_call, n_classes=n_classes
        )

    def _fit_multiclass(self, X, y, field_ids, sample_weight):
        n_rows, n_features = X.shape
        n_classes = self.classes_.shape[0]
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        y_idx = np.searchsorted(self.classes_, y)  # classes_ is sorted (np.unique)
        sw = _combine_weights(y_idx, self.classes_, sample_weight, self.class_weight, n_rows)
        resumed = warm_resume(self)
        rng = np.random.default_rng(self.random_state)
        if resumed is None:
            params = init_ffm_multiclass_params(
                rng, n_classes, n_features, self.n_fields_, self.n_factors, self.init_scale
            )
            opt = {}
        else:
            params, opt = resumed
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
            **opt,
        )
        out_dtype = np.float32 if self.dtype == "float32" else np.float64
        self.w0_ = w0.astype(out_dtype)
        self.w_ = w.astype(out_dtype)
        self.V_ = V.astype(out_dtype)
        self.n_features_in_ = n_features
        self.n_iter_ = self.max_iter
        return self

    def _fit_multiclass_es(self, X, y, field_ids, sample_weight, eval_val):
        """Epoch-by-epoch multiclass FFM fit with early stopping (softmax
        cross-entropy metric); the per-class optimizer state round-trips through
        the Rust kernel (mirrors FMClassifier._fit_multiclass_es)."""
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
        resumed = warm_resume(self)
        if resumed is None:
            init = init_ffm_multiclass_params(
                rng, n_classes, n_features, self.n_fields_, self.n_factors, self.init_scale
            )
            opt = None
        else:
            init, opt = resumed
        row_orders = make_row_orders(rng, n_tr, self.max_iter)
        is_adam = self.optimizer == "adam"
        is_ftrl = self.optimizer == "ftrl"
        if opt is not None:
            state, adam_state, ftrl_state = (
                opt.get("state"), opt.get("adam_state"), opt.get("ftrl_state")
            )
        else:
            state = None if (is_adam or is_ftrl) else [
                np.zeros(n_classes), np.zeros((n_classes, n_features)),
                np.zeros((n_classes, n_features, self.n_fields_, self.n_factors)),
            ]
            adam_state = new_adam_state(init[0], init[1], init[2]) if is_adam else None
            ftrl_state = new_ftrl_state(init[0], init[1], init[2]) if is_ftrl else None
        work = [init[0], init[1], init[2]]

        def train_epoch(e):
            work[0], work[1], work[2] = _backend.ffm_fit_multiclass(
                X_tr, y_tr, field_ids, (work[0], work[1], work[2]), optimizer=self.optimizer,
                learning_rate=self.learning_rate, l2_linear=self.l2_linear,
                l2_factors=self.l2_factors, l1_linear=self.l1_linear, l1_factors=self.l1_factors,
                row_orders=row_orders[e : e + 1], label_smoothing=self.label_smoothing,
                beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon,
                ftrl_beta=self.ftrl_beta, batch_size=self.batch_size, sample_weight=sw_tr,
                state=state, adam_state=adam_state, ftrl_state=ftrl_state,
            )

        def evaluate():
            logits = np.column_stack(
                [
                    _backend.ffm_predict(
                        X_val, self.field_ids_, float(work[0][c]), work[1][c], work[2][c]
                    )
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
        X = _validate_X(self, X, reset=False)
        if self.V_.ndim == 4:  # multiclass: per-class FFM logits -> (n, n_classes)
            return np.column_stack(
                [
                    _backend.ffm_predict(
                        X,
                        self.field_ids_,
                        float(self.w0_[c]),
                        self.w_[c],
                        self.V_[c],
                        backend=self.backend,
                    )
                    for c in range(self.V_.shape[0])
                ]
            )
        return _backend.ffm_predict(
            X, self.field_ids_, self.w0_, self.w_, self.V_, backend=self.backend
        )

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


class FFMRegressor(RegressorMixin, _FFMBase):
    """Field-aware Factorization Machine regressor (squared loss).

    The regression counterpart to `FMRegressor`; see docs/api_design.md and
    docs/math_spec.md. `field_ids` is passed at fit time (one field id per
    feature/column; defaults to each column its own field) and stored on the
    model. predict returns the raw FFM score (no link function)."""

    def __init__(
        self,
        n_factors=8,
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
        early_stopping=False,
        validation_fraction=0.1,
        patience=10,
        min_delta=0.0,
        warm_start=False,
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
        self.warm_start = warm_start
        self.dtype = dtype
        self.backend = backend
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.l1_linear = l1_linear
        self.l1_factors = l1_factors
        self.ftrl_beta = ftrl_beta

    def fit(self, X, y, field_ids=None, sample_weight=None, eval_set=None):
        self._validate_common()
        X = _validate_X(self, X, reset=True)
        n_rows, n_features = X.shape
        field_ids = self._set_field_ids(field_ids, n_features)
        sw = _check_sample_weight(sample_weight, n_rows)
        y = column_or_1d(y, warn=True).astype(np.float64)
        check_consistent_length(X, y)
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinity")
        if self.early_stopping or eval_set is not None:
            eval_val = None
            if eval_set is not None:
                Xv, yv = normalize_eval_set(eval_set)
                eval_val = (_check_X(Xv, n_features), np.asarray(yv, dtype=np.float64))
            return self._fit_es(X, y, y, "squared", sw, eval_val)
        return self._fit_core(X, y, "squared", sample_weight=sw)

    def partial_fit(self, X, y, field_ids=None, sample_weight=None):
        """Incremental fit on a chunk: one pass in natural row order, continuing the
        optimizer state. ``field_ids`` is set on the first call and validated
        thereafter. See docs/api_design.md."""
        self._validate_common()
        first_call = not hasattr(self, "n_features_in_")
        X = _validate_X(self, X, reset=first_call)
        self._partial_field_ids(field_ids, X.shape[1], first_call)
        sw = _check_sample_weight(sample_weight, X.shape[0])
        y = column_or_1d(y, warn=True).astype(np.float64)
        check_consistent_length(X, y)
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinity")
        return self._advance_one_epoch(
            X, y, "squared", sw, multiclass=False, first_call=first_call, n_classes=None
        )

    def predict(self, X):
        return self._ffm_raw_scores(X)
