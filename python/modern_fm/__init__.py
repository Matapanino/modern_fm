"""modern_fm: fast, sklearn-compatible FM / FFM for Python."""

from .ffm import FFMClassifier
from .fm import FMClassifier, FMRegressor

__version__ = "0.1.0.dev0"

__all__ = ["FMClassifier", "FMRegressor", "FFMClassifier", "__version__"]
