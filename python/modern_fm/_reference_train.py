"""Pure-Python reference trainers for FM and FFM (docs/optimization_spec.md).

Ground truth for the Rust training kernels. batch_size=1: one update per row,
per-row gradients computed from pre-update parameters and applied once.
Deliberately unoptimized.

Conventions shared bit-for-bit with the Rust backend (rust/src/{fm,ffm}.rs):
- rows are visited in the order given by `row_orders` of shape
  (epochs, n_rows); callers generate permutations (see `make_row_orders`),
  so no RNG lives inside the training loop
- exact-zero entries contribute nothing (dense input behaves like CSR)
- L2 is lazy: only parameters touched by a row are regularized
- AdaGrad: G += g^2; theta -= lr * g / sqrt(G + 1e-10)
- update order within a row: w0, then w, then V
- FFM V gradients are accumulated per (feature, field) slot over all pairs
  (a ascending, then b ascending), then applied once per touched slot
"""

from __future__ import annotations

import math

import numpy as np

from ._reference import _as_dense_rows

ADAGRAD_EPS = 1e-10
LOSSES = ("logistic", "squared")
OPTIMIZERS = ("sgd", "adagrad")


def _sigmoid(s):
    # Stable scalar sigmoid; the Rust optimizer::sigmoid must match exactly.
    if s >= 0.0:
        return 1.0 / (1.0 + math.exp(-s))
    e = math.exp(s)
    return e / (1.0 + e)


def _check_loss_optimizer(loss, optimizer):
    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r}; expected one of {LOSSES}")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {OPTIMIZERS}")


def init_fm_params(rng, n_features, n_factors, init_scale):
    """w0 = 0, w = 0, V ~ Normal(0, init_scale)."""
    w0 = 0.0
    w = np.zeros(n_features)
    V = rng.normal(0.0, init_scale, size=(n_features, n_factors))
    return w0, w, V


def init_fm_multiclass_params(rng, n_classes, n_features, n_factors, init_scale):
    """Per-class FM params: w0 (C,), w (C, n), V (C, n, k) ~ Normal(0, init_scale)."""
    w0 = np.zeros(n_classes)
    w = np.zeros((n_classes, n_features))
    V = rng.normal(0.0, init_scale, size=(n_classes, n_features, n_factors))
    return w0, w, V


def init_ffm_params(rng, n_features, n_fields, n_factors, init_scale):
    """w0 = 0, w = 0, V ~ Uniform(0, init_scale / sqrt(k)) (libffm-style)."""
    w0 = 0.0
    w = np.zeros(n_features)
    V = rng.uniform(0.0, init_scale / math.sqrt(n_factors), size=(n_features, n_fields, n_factors))
    return w0, w, V


def make_row_orders(rng, n_rows, epochs, shuffle=True):
    """(epochs, n_rows) row visit order; one fresh permutation per epoch."""
    if shuffle:
        return np.stack([rng.permutation(n_rows) for _ in range(epochs)])
    return np.tile(np.arange(n_rows), (epochs, 1))


def fm_fit_reference(
    X,
    y,
    params,
    *,
    loss="logistic",
    optimizer="adagrad",
    learning_rate=0.05,
    l2_linear=0.0,
    l2_factors=0.0,
    row_orders=None,
    sample_weight=None,
    state=None,
):
    """Train an FM from `params` = (w0, w, V); returns new (w0, w, V) copies.

    For logistic loss, y must be 0/1 (label smoothing produces a soft target in
    [0, 1], which is also fine). `sample_weight` scales each row's gradient
    (batch_size=1); L2 is unscaled. CSR input must be canonical (no duplicate
    column indices within a row).
    """
    _check_loss_optimizer(loss, optimizer)
    w0, w, V = params
    w0 = float(w0)
    w = np.array(w, dtype=np.float64, copy=True)
    V = np.array(V, dtype=np.float64, copy=True)
    y = np.asarray(y, dtype=np.float64)
    rows = list(_as_dense_rows(X))
    sw = (
        np.ones(len(rows))
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if row_orders is None:
        row_orders = np.arange(len(rows))[None, :]
    logistic = loss == "logistic"
    adagrad = optimizer == "adagrad"
    lr = learning_rate
    if state is None:
        a_w0, a_w, a_V = 0.0, np.zeros_like(w), np.zeros_like(V)
    else:
        a_w0, a_w, a_V = state  # accumulators persist across epoch-driven calls
    for order in np.asarray(row_orders):
        for r in order:
            idx, val = rows[r]
            Vi = V[idx]  # (z, k) copy: pre-update factors for this row
            cache = Vi.T @ val  # (k,) = sum_i v_{i,f} x_i
            s = w0 + w[idx] @ val + 0.5 * (cache @ cache - ((Vi * val[:, None]) ** 2).sum())
            g = (_sigmoid(s) - y[r]) if logistic else (s - y[r])
            g *= sw[r]
            g_w = g * val + l2_linear * w[idx]
            g_V = g * (val[:, None] * cache[None, :] - Vi * (val**2)[:, None]) + l2_factors * Vi
            if adagrad:
                a_w0 += g * g
                w0 -= lr * g / math.sqrt(a_w0 + ADAGRAD_EPS)
                a_w[idx] += g_w**2
                w[idx] -= lr * g_w / np.sqrt(a_w[idx] + ADAGRAD_EPS)
                a_V[idx] += g_V**2
                V[idx] = Vi - lr * g_V / np.sqrt(a_V[idx] + ADAGRAD_EPS)
            else:
                w0 -= lr * g
                w[idx] -= lr * g_w
                V[idx] = Vi - lr * g_V
    if state is not None:
        state[0], state[1], state[2] = a_w0, a_w, a_V
    return w0, w, V


def fm_fit_multiclass_reference(
    X,
    y,
    params,
    *,
    optimizer="adagrad",
    learning_rate=0.05,
    l2_linear=0.0,
    l2_factors=0.0,
    row_orders=None,
    label_smoothing=0.0,
    sample_weight=None,
):
    """Train a multiclass (softmax) FM: one FM per class, coupled by softmax.

    `params` = (w0 (C,), w (C, n), V (C, n, k)); `y` holds integer class indices
    in [0, C). The gradient on class-c's logit is
    sample_weight * (p_c - target_c), where target uses label smoothing
    (target_c = 1-eps if c == y else eps/(C-1)). Per-class FM updates reuse the
    binary FM gradient; classes share no parameters. NumPy ground truth for the
    Rust `fm_fit_multiclass_csr` kernel (parity in tests/test_rust_train_parity.py).
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {OPTIMIZERS}")
    w0, w, V = params
    w0 = np.array(w0, dtype=np.float64, copy=True)
    w = np.array(w, dtype=np.float64, copy=True)
    V = np.array(V, dtype=np.float64, copy=True)
    y = np.asarray(y)
    n_classes, _, k = V.shape
    rows = list(_as_dense_rows(X))
    sw = (
        np.ones(len(rows))
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if row_orders is None:
        row_orders = np.arange(len(rows))[None, :]
    adagrad = optimizer == "adagrad"
    lr = learning_rate
    eps = label_smoothing
    off = eps / (n_classes - 1) if n_classes > 1 else 0.0
    a_w0 = np.zeros_like(w0)
    a_w = np.zeros_like(w)
    a_V = np.zeros_like(V)
    for order in np.asarray(row_orders):
        for r in order:
            idx, val = rows[r]
            yc = int(y[r])
            # class logits and the per-class factor cache (pre-update params)
            logits = np.empty(n_classes)
            caches = np.empty((n_classes, k))
            for c in range(n_classes):
                Vi = V[c][idx]
                cache = Vi.T @ val
                caches[c] = cache
                logits[c] = (
                    w0[c]
                    + w[c][idx] @ val
                    + 0.5 * (cache @ cache - ((Vi * val[:, None]) ** 2).sum())
                )
            ex = np.exp(logits - logits.max())  # stable softmax
            p = ex / ex.sum()
            for c in range(n_classes):
                target = (1.0 - eps) if c == yc else off
                g = sw[r] * (p[c] - target)
                Vi = V[c][idx]
                cache = caches[c]
                g_w = g * val + l2_linear * w[c][idx]
                g_V = g * (val[:, None] * cache[None, :] - Vi * (val**2)[:, None]) + l2_factors * Vi
                if adagrad:
                    a_w0[c] += g * g
                    w0[c] -= lr * g / math.sqrt(a_w0[c] + ADAGRAD_EPS)
                    a_w[c][idx] += g_w**2
                    w[c][idx] -= lr * g_w / np.sqrt(a_w[c][idx] + ADAGRAD_EPS)
                    a_V[c][idx] += g_V**2
                    V[c][idx] = Vi - lr * g_V / np.sqrt(a_V[c][idx] + ADAGRAD_EPS)
                else:
                    w0[c] -= lr * g
                    w[c][idx] -= lr * g_w
                    V[c][idx] = Vi - lr * g_V
    return w0, w, V


def ffm_fit_reference(
    X,
    y,
    field_ids,
    params,
    *,
    optimizer="adagrad",
    learning_rate=0.05,
    l2_linear=0.0,
    l2_factors=0.0,
    row_orders=None,
    sample_weight=None,
    state=None,
):
    """Train an FFM (logistic loss) from `params` = (w0, w, V); returns copies.

    V has shape (n_features, n_fields, k). y must be 0/1 (or a soft target in
    [0, 1]). `sample_weight` scales each row's gradient (batch_size=1).
    """
    _check_loss_optimizer("logistic", optimizer)
    field_ids = np.asarray(field_ids, dtype=np.int64)
    w0, w, V = params
    w0 = float(w0)
    w = np.array(w, dtype=np.float64, copy=True)
    V = np.array(V, dtype=np.float64, copy=True)
    y = np.asarray(y, dtype=np.float64)
    n_fields, k = V.shape[1], V.shape[2]
    rows = list(_as_dense_rows(X))
    sw = (
        np.ones(len(rows))
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if row_orders is None:
        row_orders = np.arange(len(rows))[None, :]
    adagrad = optimizer == "adagrad"
    lr = learning_rate
    if state is None:
        a_w0, a_w, a_V = 0.0, np.zeros_like(w), np.zeros_like(V)
    else:
        a_w0, a_w, a_V = state  # accumulators persist across epoch-driven calls
    for order in np.asarray(row_orders):
        for r in order:
            idx, val = rows[r]
            z = len(idx)
            f = field_ids[idx]
            # pass 1: score from pre-update parameters
            s = w0 + w[idx] @ val
            for a in range(z):
                for b in range(a + 1, z):
                    s += (V[idx[a], f[b]] @ V[idx[b], f[a]]) * val[a] * val[b]
            g = (_sigmoid(s) - y[r]) * sw[r]
            g_w = g * val + l2_linear * w[idx]
            # pass 2: accumulate V gradients per touched (feature, field) slot
            gV = np.zeros((z, n_fields, k))
            touched = np.zeros((z, n_fields), dtype=bool)
            for a in range(z):
                for b in range(a + 1, z):
                    coef = g * val[a] * val[b]
                    gV[a, f[b]] += coef * V[idx[b], f[a]]
                    gV[b, f[a]] += coef * V[idx[a], f[b]]
                    touched[a, f[b]] = True
                    touched[b, f[a]] = True
            # updates: w0, w, then touched V slots (a ascending, field ascending)
            if adagrad:
                a_w0 += g * g
                w0 -= lr * g / math.sqrt(a_w0 + ADAGRAD_EPS)
                a_w[idx] += g_w**2
                w[idx] -= lr * g_w / np.sqrt(a_w[idx] + ADAGRAD_EPS)
                for a in range(z):
                    i = idx[a]
                    for fld in range(n_fields):
                        if touched[a, fld]:
                            grad = gV[a, fld] + l2_factors * V[i, fld]
                            a_V[i, fld] += grad**2
                            V[i, fld] -= lr * grad / np.sqrt(a_V[i, fld] + ADAGRAD_EPS)
            else:
                w0 -= lr * g
                w[idx] -= lr * g_w
                for a in range(z):
                    i = idx[a]
                    for fld in range(n_fields):
                        if touched[a, fld]:
                            grad = gV[a, fld] + l2_factors * V[i, fld]
                            V[i, fld] -= lr * grad
    if state is not None:
        state[0], state[1], state[2] = a_w0, a_w, a_V
    return w0, w, V


def fm_train(
    X,
    y,
    *,
    loss="logistic",
    optimizer="adagrad",
    learning_rate=0.05,
    epochs=10,
    n_factors=4,
    l2_linear=0.0,
    l2_factors=0.0,
    init_scale=0.01,
    random_state=None,
    shuffle=True,
):
    """Seeded end-to-end FM training: init + per-epoch shuffling + fit."""
    rng = np.random.default_rng(random_state)
    params = init_fm_params(rng, X.shape[1], n_factors, init_scale)
    row_orders = make_row_orders(rng, X.shape[0], epochs, shuffle)
    return fm_fit_reference(
        X,
        y,
        params,
        loss=loss,
        optimizer=optimizer,
        learning_rate=learning_rate,
        l2_linear=l2_linear,
        l2_factors=l2_factors,
        row_orders=row_orders,
    )


def ffm_train(
    X,
    y,
    field_ids,
    *,
    optimizer="adagrad",
    learning_rate=0.05,
    epochs=10,
    n_factors=4,
    l2_linear=0.0,
    l2_factors=0.0,
    init_scale=0.01,
    random_state=None,
    shuffle=True,
):
    """Seeded end-to-end FFM training (logistic loss)."""
    rng = np.random.default_rng(random_state)
    field_ids = np.asarray(field_ids, dtype=np.int64)
    n_fields = int(field_ids.max()) + 1
    params = init_ffm_params(rng, X.shape[1], n_fields, n_factors, init_scale)
    row_orders = make_row_orders(rng, X.shape[0], epochs, shuffle)
    return ffm_fit_reference(
        X,
        y,
        field_ids,
        params,
        optimizer=optimizer,
        learning_rate=learning_rate,
        l2_linear=l2_linear,
        l2_factors=l2_factors,
        row_orders=row_orders,
    )
