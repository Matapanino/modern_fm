"""Shared estimator plumbing.

`NotFittedError` and `check_is_fitted` are re-exported from scikit-learn so the
estimators raise the exact exception sklearn tooling expects. `ModelIOMixin`
adds save_model / load_model; the estimators (and `CategoricalEncoder`) inherit
sklearn's `BaseEstimator`, which supplies get_params / set_params / tags.
"""

from __future__ import annotations

import pickle

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

__all__ = ["NotFittedError", "check_is_fitted", "ModelIOMixin"]


class ModelIOMixin:
    """save_model / load_model for fitted estimators.

    Stores `{format_version, class, params, attrs}` via pickle, where `attrs`
    are the learned trailing-underscore attributes; this is generic over the
    estimator (binary, regression, multiclass) and round-trips constructor
    params too. The estimators also pickle natively (plain attributes), so
    `pickle.dumps(model)` works as an alternative.
    """

    _IO_VERSION = 1

    def save_model(self, path):
        check_is_fitted(self)
        state = {
            "format_version": self._IO_VERSION,
            "class": type(self).__name__,
            "params": self.get_params(),
            "attrs": {k: getattr(self, k) for k in vars(self) if k.endswith("_")},
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_model(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state.get("class") != cls.__name__:
            raise ValueError(f"{path!r} holds a {state.get('class')!r}, not a {cls.__name__!r}")
        model = cls(**state["params"])
        for key, value in state["attrs"].items():
            setattr(model, key, value)
        return model
