"""modern_fm: fast, sklearn-compatible FM / FFM for Python."""

from ._base import NotFittedError
from .ffm import FFMClassifier, FFMRegressor
from .fm import FMClassifier, FMRegressor
from .fwfm import FwFMClassifier
from .libffm import dump_libffm, load_libffm
from .pooling import BiInteractionPooling
from .preprocessing import CategoricalEncoder

__version__ = "0.5.0"

__all__ = [
    "FMClassifier",
    "FMRegressor",
    "FFMClassifier",
    "FFMRegressor",
    "FwFMClassifier",
    "BiInteractionPooling",
    "CategoricalEncoder",
    "NotFittedError",
    "load_libffm",
    "dump_libffm",
    "__version__",
]
