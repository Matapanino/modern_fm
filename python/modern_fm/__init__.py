"""modern_fm: fast, sklearn-compatible FM / FFM for Python."""

from ._base import NotFittedError
from .ffm import FFMClassifier
from .fm import FMClassifier, FMRegressor
from .libffm import dump_libffm, load_libffm
from .preprocessing import CategoricalEncoder

__version__ = "0.2.1"

__all__ = [
    "FMClassifier",
    "FMRegressor",
    "FFMClassifier",
    "CategoricalEncoder",
    "NotFittedError",
    "load_libffm",
    "dump_libffm",
    "__version__",
]
