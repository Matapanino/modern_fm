"""FFM estimator — public API skeleton (docs/api_design.md).

field_ids is explicit and required at fit time; no automatic field inference
in v0.1. Training arrives with the Rust backend in Phase 2.
"""

from __future__ import annotations

from ._base import ParamsMixin

_FIT_MSG = "Training lands with the Rust backend in v0.1 Phase 2 (see docs/roadmap.md)."


class FFMClassifier(ParamsMixin):
    """Field-aware Factorization Machine classifier.
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
        raise NotImplementedError(_FIT_MSG)

    def predict(self, X):
        raise NotImplementedError(_FIT_MSG)

    def predict_proba(self, X):
        raise NotImplementedError(_FIT_MSG)

    def decision_function(self, X):
        raise NotImplementedError(_FIT_MSG)

    def save_model(self, path):
        raise NotImplementedError(_FIT_MSG)

    @classmethod
    def load_model(cls, path):
        raise NotImplementedError(_FIT_MSG)
