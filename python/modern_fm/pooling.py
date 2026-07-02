"""Bi-interaction pooling transformer — the honest "NFM pooling".

NFM (He & Chua, SIGIR 2017) is bi-interaction pooling followed by an MLP; the
MLP is what makes it a deep model (out of scope, docs/requirements.md). Without
it, a linear head over the pooled vector provably collapses to plain FM: for
weights h, `h^T f_BI(x) = sum_{i<j} (sum_f h_f v_{if} v_{jf}) x_i x_j`, which is
FM with `sqrt(h_f)` folded into `V` (the NFM paper itself notes FM is the
no-hidden-layer special case). So modern_fm ships bi-interaction pooling as a
**feature transform**, not a predictor: `BiInteractionPooling` fits an FM and
`transform` emits the k-dim second-order interaction vector

    f_BI(x)_f = 0.5 * [(sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2]

for downstream models (GBDT, linear models, ...). O(nnz * k) per call with no
parameters beyond the fitted `V_`. The collapse identity
`fm_predict_fast(X, w0, w, V) == w0 + X @ w + fm_bi_interaction(X, V).sum(1)`
is pinned by tests. The FM estimators also expose the same features directly
via `bi_interaction(X)` (deliberately not named `transform`, so plain FMs keep
plain-estimator semantics in sklearn tooling).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin, clone

from . import _backend
from ._base import check_is_fitted
from .fm import FMRegressor


class BiInteractionPooling(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Fit an FM and transform rows into bi-interaction pooled features.

    Parameters
    ----------
    estimator : a modern_fm FM estimator instance or None
        The FM whose factors supply the pooling (cloned at fit time). None
        defaults to ``FMRegressor(n_factors=8)`` — pass an estimator with
        ``random_state`` set for deterministic transforms. Multiclass
        classifiers pool per class, concatenated to
        ``(n_samples, n_classes * n_factors)``.

    Attributes
    ----------
    estimator_ : the fitted FM
    n_features_in_ : int
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.target_tags.required = True  # the inner FM is supervised
        return tags

    def fit(self, X, y, **fit_params):
        """Fit the inner FM on (X, y); `fit_params` (e.g. sample_weight) pass
        through to its `fit`."""
        est = FMRegressor(n_factors=8) if self.estimator is None else self.estimator
        self.estimator_ = clone(est).fit(X, y, **fit_params)
        self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        return self

    def transform(self, X):
        """Bi-interaction pooled features: (n_samples, n_factors), or
        (n_samples, n_classes * n_factors) for a multiclass inner FM."""
        check_is_fitted(self, "estimator_")
        # reuse the inner estimator's validation (feature count/names, dtype,
        # CSR conversion, finiteness) without re-recording fit-time state
        from .fm import _validate_X

        X = _validate_X(self.estimator_, X, reset=False)
        V = self.estimator_.V_
        if V.ndim == 3:  # multiclass: per-class pooling, concatenated
            return np.hstack([_backend.fm_bi_interaction(X, V[c]) for c in range(V.shape[0])])
        return _backend.fm_bi_interaction(X, V)

    @property
    def _n_features_out(self):
        V = self.estimator_.V_
        return V.shape[0] * V.shape[2] if V.ndim == 3 else V.shape[1]

    def _more_tags(self):  # pragma: no cover - sklearn <1.6 compatibility only
        return {"requires_y": True}
