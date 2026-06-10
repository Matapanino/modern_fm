"""Loss functions (docs/math_spec.md). Numerically stable, pure NumPy."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def sigmoid(s):
    s = np.asarray(s, dtype=np.float64)
    out = np.empty_like(s)
    pos = s >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-s[pos]))
    es = np.exp(s[~pos])
    out[~pos] = es / (1.0 + es)
    return out


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))


def _weighted_mean(values, sample_weight):
    if sample_weight is None:
        return values.mean()
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    return (values * sample_weight).sum() / sample_weight.sum()


def logistic_loss(y, raw, label_smoothing=0.0, sample_weight=None):
    """Mean binary cross-entropy from raw scores.

    Stable form: loss = max(s, 0) - s * y + log1p(exp(-|s|)).
    With label smoothing eps: y <- y * (1 - eps) + 0.5 * eps.
    """
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(raw, dtype=np.float64)
    if label_smoothing:
        y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
    per_sample = np.maximum(s, 0.0) - s * y + np.log1p(np.exp(-np.abs(s)))
    return _weighted_mean(per_sample, sample_weight)


def softmax_loss(y, logits, label_smoothing=0.0, sample_weight=None):
    """Mean multiclass cross-entropy from logits.

    y: int class indices in [0, n_classes). With label smoothing eps the
    target is 1 - eps for the true class and eps / (n_classes - 1) elsewhere.
    """
    y = np.asarray(y)
    logits = np.asarray(logits, dtype=np.float64)
    n_classes = logits.shape[1]
    log_p = logits - logsumexp(logits, axis=1, keepdims=True)
    if label_smoothing:
        targets = np.full_like(log_p, label_smoothing / (n_classes - 1))
        targets[np.arange(len(y)), y] = 1.0 - label_smoothing
        per_sample = -(targets * log_p).sum(axis=1)
    else:
        per_sample = -log_p[np.arange(len(y)), y]
    return _weighted_mean(per_sample, sample_weight)


def squared_loss(y, raw, sample_weight=None):
    """Mean squared-error loss: 0.5 * (y_hat - y)^2."""
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(raw, dtype=np.float64)
    return _weighted_mean(0.5 * (s - y) ** 2, sample_weight)
