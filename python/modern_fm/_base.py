"""Minimal sklearn-compatible parameter handling.

Avoids a hard scikit-learn dependency in v0.1 while keeping the contract:
__init__ stores constructor arguments verbatim; get_params/set_params
round-trip them. Phase 3 may swap this for sklearn's BaseEstimator.
"""

from __future__ import annotations

import inspect


class NotFittedError(ValueError, AttributeError):
    """Raised when a predict-like method is called before fit.

    Inherits ValueError and AttributeError to match sklearn's exception of
    the same name, so generic sklearn-style error handling keeps working.
    """


def check_is_fitted(estimator, attribute="w0_"):
    if not hasattr(estimator, attribute):
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet; "
            "call 'fit' before using this method."
        )


class ParamsMixin:
    @classmethod
    def _param_names(cls):
        sig = inspect.signature(cls.__init__)
        return [name for name in sig.parameters if name != "self"]

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params):
        valid = set(self._param_names())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {type(self).__name__}. "
                    f"Valid parameters are: {sorted(valid)}."
                )
            setattr(self, key, value)
        return self

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"
