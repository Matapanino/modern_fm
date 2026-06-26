"""modern_fm: fast, sklearn-compatible FM / FFM for Python."""

from ._base import NotFittedError
from .ffm import FFMClassifier
from .fm import FMClassifier, FMRegressor
from .preprocessing import CategoricalEncoder

__version__ = "0.2.0"

__all__ = [
    "FMClassifier",
    "FMRegressor",
    "FFMClassifier",
    "CategoricalEncoder",
    "NotFittedError",
    "__version__",
]
