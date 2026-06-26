"""Helpers for incremental / streaming training (``partial_fit`` & ``warm_start``).

One ``partial_fit`` call runs a single pass over its chunk in natural row order,
continuing the persistent optimizer state stored on the estimator as
``_opt_state`` (a leading-underscore, non-trailing attribute so it is excluded
from ``check_is_fitted`` and ``save_model`` yet still pickles via ``__dict__``).
N sequential ``partial_fit`` calls therefore equal one ``fit`` over the
concatenated data under a matched schedule — exact for ``batch_size=1`` (or chunk
lengths that are multiples of ``batch_size``) with ``n_jobs=1``. See
docs/api_design.md and docs/roadmap.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import column_or_1d

from ._reference_train import new_adam_state, new_ftrl_state


def make_opt_state(optimizer, w0, w, V):
    """Fresh persistent optimizer state for incremental training.

    Returns a dict with exactly one of ``state`` (sgd/adagrad AdaGrad
    accumulators), ``adam_state`` (Adam moments, see ``new_adam_state``), or
    ``ftrl_state`` (FTRL ``(z, n)``, see ``new_ftrl_state``), keyed to pass
    straight through to the backend as ``**opt_state``. Shapes follow
    ``(w0, w, V)``, so the same helper serves binary/multiclass FM and FFM.
    """
    if optimizer == "adam":
        return {"adam_state": new_adam_state(w0, w, V)}
    if optimizer == "ftrl":
        return {"ftrl_state": new_ftrl_state(w0, w, V)}
    z = np.zeros_like
    return {"state": [z(w0), z(w), z(V)]}  # sgd ignores these (harmless no-op round-trip)


def partial_fit_classes(estimator, y, classes):
    """scikit-learn's first-call ``classes`` convention for incremental classifiers.

    Returns ``(first_call, y_1d)``. On the first call ``classes`` is required and
    sets ``estimator.classes_``; later calls validate a re-passed ``classes``
    against the stored set. Labels in ``y`` outside ``classes_`` are rejected.
    """
    first_call = _check_partial_fit_first_call(estimator, classes)
    y = column_or_1d(y, warn=True)
    if not np.isin(np.asarray(y), estimator.classes_).all():
        raise ValueError(
            "partial_fit got y with labels not present in `classes` "
            f"(classes_={estimator.classes_.tolist()})"
        )
    return first_call, y


def warm_resume(estimator):
    """``warm_start`` continuation for ``fit``: returns ``(params, opt_state)`` of
    float64 copies to resume from when ``warm_start`` is set and the estimator is
    already fitted (persisting/creating ``_opt_state``), else ``None`` after
    clearing any prior ``_opt_state`` (a fresh ``fit`` discards streamed state)."""
    if estimator.warm_start and hasattr(estimator, "w0_"):
        multiclass = np.ndim(estimator.w0_) > 0
        w0 = estimator.w0_.astype(np.float64) if multiclass else float(estimator.w0_)
        w = estimator.w_.astype(np.float64)
        V = estimator.V_.astype(np.float64)
        opt = getattr(estimator, "_opt_state", None)
        if opt is None:
            opt = make_opt_state(estimator.optimizer, w0, w, V)
        estimator._opt_state = opt
        return (w0, w, V), opt
    estimator._opt_state = None
    return None
