"""FwFM estimator (docs/math_spec_fwfm.md, docs/api_design.md).

Field-weighted Factorization Machine (Pan et al., WWW 2018): FM-shaped latent
factors `V (n, k)` plus one learned scalar weight per field pair,
`R (F, F)` (upper triangle used), scaling each pairwise interaction:

    y_hat = w0 + sum_i w_i x_i + sum_{i<j} r_{f_i f_j} <v_i, v_j> x_i x_j

`R` initializes to ones, so a fresh FwFM is exactly a plain FM. Binary
(logistic) and multiclass (softmax, one FwFM per class) classification through
`_backend` with mini-batch `batch_size`; training is serial in v0.5 (`n_jobs`
is accepted for API symmetry but does not parallelize FwFM). Early stopping,
`partial_fit` / `warm_start`, save/load, and all four optimizers work exactly
as for `FFMClassifier`; the optimizer-state hand-off (four parameter groups)
round-trips through the Rust kernel.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_consistent_length, column_or_1d

from . import _backend, _inspect
from ._base import check_is_fitted
from ._early_stop import normalize_eval_set, run_epochs, split_indices
from ._partial import make_opt_state_fwfm, partial_fit_classes, warm_resume_fwfm
from ._reference_train import (
    init_fwfm_multiclass_params,
    init_fwfm_params,
    make_row_orders,
    new_adam_state_fwfm,
    new_ftrl_state_fwfm,
)
from .ffm import _FFMBase
from .fm import (
    _check_n_top,
    _check_X,
    _combine_weights,
    _fit_backend_guard,
    _predict_backend_guard,
    _select_class_slice,
    _smooth,
    _validate_X,
)
from .losses import logistic_loss, sigmoid, softmax, softmax_loss


class _FwFMBase(_FFMBase):
    """Shared FwFM machinery: overrides `_FFMBase`'s parameter-shape-specific
    training/scoring for the four-group (w0, w, V, R) parameterization while
    reusing its validation and `field_ids` plumbing."""

    def _validate_common(self):
        super()._validate_common()
        _fit_backend_guard(self.backend, "FwFM training")

    def top_interactions(self, n_top=10, class_idx=None):
        """The `n_top` strongest learned pairwise interactions.

        Returns a list of ``(i, j, strength)`` tuples (feature indices,
        ``i < j``) sorted by descending
        ``strength = |r[min(f_i, f_j), max(f_i, f_j)] * <V_i, V_j>|`` — the
        magnitude of the learned pairwise coefficient of ``x_i x_j``
        (docs/math_spec_fwfm.md). Multiclass models require ``class_idx``.
        """
        check_is_fitted(self)
        _check_n_top(n_top)
        V, r = _select_class_slice(
            (self.V_, self.r_), self.V_.ndim == 3, class_idx, "FwFM"
        )
        return _inspect.fm_top_interactions(V, n_top, r=r, field_ids=self.field_ids_)

    def _store_fitted(self, w0, w, V, R, n_features, n_iter, multiclass):
        out_dtype = np.float32 if self.dtype == "float32" else np.float64
        self.w0_ = w0.astype(out_dtype) if multiclass else float(w0)
        self.w_ = w.astype(out_dtype)
        self.V_ = V.astype(out_dtype)
        self.r_ = R.astype(out_dtype)
        self.n_features_in_ = n_features
        self.n_iter_ = n_iter
        return self

    def _backend_common(self, sample_weight, row_orders):
        return dict(
            optimizer=self.optimizer, learning_rate=self.learning_rate,
            l2_linear=self.l2_linear, l2_factors=self.l2_factors,
            l1_linear=self.l1_linear, l1_factors=self.l1_factors,
            row_orders=row_orders, beta_1=self.beta_1, beta_2=self.beta_2,
            epsilon=self.epsilon, ftrl_beta=self.ftrl_beta,
            batch_size=self.batch_size, sample_weight=sample_weight,
        )

    def _fit_core(self, X, y, loss, sample_weight=None):
        """Single all-epochs FwFM fit; stores w0_/w_/V_/r_/n_features_in_/n_iter_."""
        n_rows, n_features = X.shape
        resumed = warm_resume_fwfm(self)
        rng = np.random.default_rng(self.random_state)
        if resumed is None:
            params = init_fwfm_params(
                rng, n_features, self.n_fields_, self.n_factors, self.init_scale
            )
            opt = {}
        else:
            params, opt = resumed
        row_orders = make_row_orders(rng, n_rows, self.max_iter)
        w0, w, V, R = _backend.fwfm_fit(
            X, y, self.field_ids_, params, loss=loss,
            **self._backend_common(sample_weight, row_orders), **opt,
        )
        return self._store_fitted(w0, w, V, R, n_features, self.max_iter, multiclass=False)

    def _fit_es(self, X, y_train, y_eval, loss, sample_weight, eval_val):
        """Epoch-by-epoch FwFM fit with early stopping (mirrors `_FFMBase._fit_es`;
        the four-group optimizer state round-trips through the Rust kernel)."""
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
        resumed = warm_resume_fwfm(self)
        if resumed is None:
            init = init_fwfm_params(
                rng, n_features, self.n_fields_, self.n_factors, self.init_scale
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
            state = (
                None if (is_adam or is_ftrl)
                else [0.0, np.zeros(n_features),
                      np.zeros((n_features, self.n_factors)),
                      np.zeros((self.n_fields_, self.n_fields_))]
            )
            adam_state = new_adam_state_fwfm(*init) if is_adam else None
            ftrl_state = new_ftrl_state_fwfm(*init) if is_ftrl else None
        work = list(init)
        metric = logistic_loss  # FwFMClassifier is logistic/softmax only

        def train_epoch(e):
            work[0], work[1], work[2], work[3] = _backend.fwfm_fit(
                X_tr, y_tr, self.field_ids_, tuple(work), loss=loss,
                **self._backend_common(sw_tr, row_orders[e : e + 1]),
                state=state, adam_state=adam_state, ftrl_state=ftrl_state,
            )

        def evaluate():
            scores = _backend.fwfm_predict(
                X_val, self.field_ids_, work[0], work[1], work[2], work[3]
            )
            return metric(y_val, scores)

        def snapshot():
            return (work[0], work[1].copy(), work[2].copy(), work[3].copy())

        best, n_iter = run_epochs(
            self.max_iter, self.patience, self.min_delta, train_epoch, evaluate, snapshot
        )
        w0, w, V, R = best
        return self._store_fitted(w0, w, V, R, n_features, n_iter, multiclass=False)

    def _advance_one_epoch(self, X, y, loss, sample_weight, *, multiclass, first_call, n_classes):
        """One natural-order pass continuing ``self._opt_state`` (FwFM partial_fit
        primitive); mirrors ``_FFMBase._advance_one_epoch`` with the R group."""
        n_rows, n_features = X.shape
        if first_call:
            rng = np.random.default_rng(self.random_state)
            if multiclass:
                w0, w, V, R = init_fwfm_multiclass_params(
                    rng, n_classes, n_features, self.n_fields_, self.n_factors, self.init_scale
                )
            else:
                w0, w, V, R = init_fwfm_params(
                    rng, n_features, self.n_fields_, self.n_factors, self.init_scale
                )
            self._opt_state = make_opt_state_fwfm(self.optimizer, w0, w, V, R)
            self.n_iter_ = 0
        else:
            w0 = self.w0_.astype(np.float64) if multiclass else float(self.w0_)
            w = self.w_.astype(np.float64)
            V = self.V_.astype(np.float64)
            R = self.r_.astype(np.float64)
            if getattr(self, "_opt_state", None) is None:
                self._opt_state = make_opt_state_fwfm(self.optimizer, w0, w, V, R)
        row_orders = np.arange(n_rows, dtype=np.int64)[None, :]  # one pass, natural order
        common = self._backend_common(sample_weight, row_orders)
        if multiclass:
            w0, w, V, R = _backend.fwfm_fit_multiclass(
                X, y, self.field_ids_, (w0, w, V, R), label_smoothing=self.label_smoothing,
                **common, **self._opt_state,
            )
        else:
            w0, w, V, R = _backend.fwfm_fit(
                X, y, self.field_ids_, (w0, w, V, R), loss=loss, **common, **self._opt_state,
            )
        n_iter = self.n_iter_ + 1
        return self._store_fitted(w0, w, V, R, n_features, n_iter, multiclass=multiclass)


class FwFMClassifier(ClassifierMixin, _FwFMBase):
    """Field-weighted Factorization Machine classifier (logistic / softmax).
    See docs/math_spec_fwfm.md and docs/api_design.md."""

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
            raise ValueError(f"unknown loss {self.loss!r} for FwFMClassifier")

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
        optimizer state (see FFMClassifier.partial_fit and docs/api_design.md)."""
        self._validate_common()
        if self.loss not in ("logistic", "softmax"):
            raise ValueError(f"unknown loss {self.loss!r} for FwFMClassifier")
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
            X, y_train, "logistic", sw, multiclass=False, first_call=first_call,
            n_classes=n_classes,
        )

    def _fit_multiclass(self, X, y, field_ids, sample_weight):
        n_rows, n_features = X.shape
        n_classes = self.classes_.shape[0]
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        y_idx = np.searchsorted(self.classes_, y)  # classes_ is sorted (np.unique)
        sw = _combine_weights(y_idx, self.classes_, sample_weight, self.class_weight, n_rows)
        resumed = warm_resume_fwfm(self)
        rng = np.random.default_rng(self.random_state)
        if resumed is None:
            params = init_fwfm_multiclass_params(
                rng, n_classes, n_features, self.n_fields_, self.n_factors, self.init_scale
            )
            opt = {}
        else:
            params, opt = resumed
        row_orders = make_row_orders(rng, n_rows, self.max_iter)
        w0, w, V, R = _backend.fwfm_fit_multiclass(
            X, y_idx, field_ids, params, label_smoothing=self.label_smoothing,
            **self._backend_common(sw, row_orders), **opt,
        )
        return self._store_fitted(w0, w, V, R, n_features, self.max_iter, multiclass=True)

    def _fit_multiclass_es(self, X, y, field_ids, sample_weight, eval_val):
        """Epoch-by-epoch multiclass FwFM fit with early stopping (softmax
        cross-entropy metric); mirrors FFMClassifier._fit_multiclass_es."""
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
        resumed = warm_resume_fwfm(self)
        if resumed is None:
            init = init_fwfm_multiclass_params(
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
                np.zeros((n_classes, n_features, self.n_factors)),
                np.zeros((n_classes, self.n_fields_, self.n_fields_)),
            ]
            adam_state = new_adam_state_fwfm(*init) if is_adam else None
            ftrl_state = new_ftrl_state_fwfm(*init) if is_ftrl else None
        work = list(init)

        def train_epoch(e):
            work[0], work[1], work[2], work[3] = _backend.fwfm_fit_multiclass(
                X_tr, y_tr, field_ids, tuple(work), label_smoothing=self.label_smoothing,
                **self._backend_common(sw_tr, row_orders[e : e + 1]),
                state=state, adam_state=adam_state, ftrl_state=ftrl_state,
            )

        def evaluate():
            logits = np.column_stack(
                [
                    _backend.fwfm_predict(
                        X_val, self.field_ids_, float(work[0][c]), work[1][c], work[2][c],
                        work[3][c],
                    )
                    for c in range(n_classes)
                ]
            )
            return softmax_loss(y_val, logits)  # true-label cross-entropy

        def snapshot():
            return (work[0].copy(), work[1].copy(), work[2].copy(), work[3].copy())

        best, n_iter = run_epochs(
            self.max_iter, self.patience, self.min_delta, train_epoch, evaluate, snapshot
        )
        w0, w, V, R = best
        return self._store_fitted(w0, w, V, R, n_features, n_iter, multiclass=True)

    def decision_function(self, X):
        _predict_backend_guard(self.backend, "FwFM")
        check_is_fitted(self)
        X = _validate_X(self, X, reset=False)
        if self.V_.ndim == 3:  # multiclass: per-class FwFM logits -> (n, n_classes)
            return np.column_stack(
                [
                    _backend.fwfm_predict(
                        X, self.field_ids_, float(self.w0_[c]), self.w_[c], self.V_[c],
                        self.r_[c],
                    )
                    for c in range(self.V_.shape[0])
                ]
            )
        return _backend.fwfm_predict(X, self.field_ids_, self.w0_, self.w_, self.V_, self.r_)

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
