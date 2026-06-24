"""Shared early-stopping epoch loop (docs/requirements.md).

The estimators drive training one epoch at a time through the backend (which
carries AdaGrad accumulator state across calls), evaluating a validation metric
after each epoch and stopping after `patience` epochs without `min_delta`
improvement. Best weights are always restored.
"""

from __future__ import annotations

import numpy as np


def split_indices(n_rows, validation_fraction, rng):
    """Random (train_idx, val_idx) for an internal holdout split."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    n_val = max(1, int(round(validation_fraction * n_rows)))
    if n_val >= n_rows:
        raise ValueError("validation_fraction leaves no training rows")
    perm = rng.permutation(n_rows)
    return perm[n_val:], perm[:n_val]


def normalize_eval_set(eval_set):
    """Accept (X_val, y_val) or [(X_val, y_val), ...]; return (X_val, y_val)."""
    item = eval_set[0] if isinstance(eval_set, list) else eval_set
    try:
        X_val, y_val = item
    except (TypeError, ValueError) as exc:
        raise ValueError("eval_set must be (X_val, y_val) or a list of such tuples") from exc
    return X_val, np.asarray(y_val)


def run_epochs(max_iter, patience, min_delta, train_epoch, evaluate, snapshot):
    """Train epoch-by-epoch with patience-based early stopping; restore best.

    - train_epoch(e): advance one epoch in place
    - evaluate(): validation metric, lower is better
    - snapshot(): restorable copy of the current parameters

    Returns (best_snapshot, n_iter_run). The first epoch always improves over
    the initial +inf, so best_snapshot is set whenever max_iter >= 1.
    """
    best_metric = np.inf
    best_snapshot = None
    since_improve = 0
    n_iter = 0
    for e in range(max_iter):
        train_epoch(e)
        n_iter = e + 1
        metric = evaluate()
        if metric < best_metric - min_delta:
            best_metric = metric
            best_snapshot = snapshot()
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break
    return best_snapshot, n_iter
