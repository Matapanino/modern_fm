"""Pure-Python reference trainers for FM and FFM (docs/optimization_spec.md).

Ground truth for the Rust training kernels. Each epoch's rows are consumed in
contiguous `batch_size` chunks; per-row gradients are computed from the frozen
batch-start parameters, averaged over the batch, and applied once per touched
coordinate (batch_size=1 is the per-row path). Deliberately unoptimized.

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
OPTIMIZERS = ("sgd", "adagrad", "adam", "ftrl")
ADAM_DEFAULTS = dict(beta_1=0.9, beta_2=0.999, epsilon=1e-8)


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


def _adam_scalar(theta, g, m, v, t, lr, beta_1, beta_2, epsilon):
    """One lazy-Adam step for a scalar parameter (docs/optimization_spec.md).

    Returns the updated (theta, m, v, t). `t` is this parameter's update count;
    `beta ** t` matches the Rust kernel's `beta.powf(t)`.
    """
    t += 1.0
    m = beta_1 * m + (1.0 - beta_1) * g
    v = beta_2 * v + (1.0 - beta_2) * g * g
    m_hat = m / (1.0 - beta_1**t)
    v_hat = v / (1.0 - beta_2**t)
    theta -= lr * m_hat / (math.sqrt(v_hat) + epsilon)
    return theta, m, v, t


def _adam_array(theta, g, m, v, t, idx, lr, beta_1, beta_2, epsilon):
    """One lazy-Adam step, in place, for the coordinates theta[idx].

    `m`, `v`, `t` are per-coordinate accumulators shaped like `theta`; `g` is the
    gradient at those coordinates (already including lazy L2). idx is duplicate-free
    (canonical CSR), so the fancy-indexed update is well defined.
    """
    t[idx] += 1.0
    m[idx] = beta_1 * m[idx] + (1.0 - beta_1) * g
    v[idx] = beta_2 * v[idx] + (1.0 - beta_2) * g * g
    m_hat = m[idx] / (1.0 - beta_1 ** t[idx])
    v_hat = v[idx] / (1.0 - beta_2 ** t[idx])
    theta[idx] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)


def _ftrl_scalar(theta, g, z, n, alpha, beta, l1, l2):
    """One FTRL-Proximal step for a scalar parameter (docs/optimization_spec.md).

    `g` is the data gradient (L1/L2 are folded into the update, not the gradient).
    Returns the updated (theta, z, n); the Rust `ftrl_step` must match this exactly.
    """
    n_new = n + g * g
    sigma = (math.sqrt(n_new) - math.sqrt(n)) / alpha
    z += g - sigma * theta
    n = n_new
    if abs(z) <= l1:
        theta = 0.0
    else:
        sign = -1.0 if z < 0.0 else 1.0
        theta = -(z - sign * l1) / ((beta + math.sqrt(n_new)) / alpha + l2)
    return theta, z, n


def _ftrl_array(theta, g, z, n, idx, alpha, beta, l1, l2):
    """One FTRL-Proximal step, in place, for the coordinates theta[idx].

    `z`, `n` are per-coordinate state shaped like `theta`; `g` is the data
    gradient at those coordinates. idx is duplicate-free (canonical CSR).
    """
    n_old = n[idx]
    n_new = n_old + g * g
    sigma = (np.sqrt(n_new) - np.sqrt(n_old)) / alpha
    z[idx] += g - sigma * theta[idx]
    n[idx] = n_new
    zi = z[idx]
    sign = np.where(zi < 0.0, -1.0, 1.0)
    reconstructed = -(zi - sign * l1) / ((beta + np.sqrt(n_new)) / alpha + l2)
    theta[idx] = np.where(np.abs(zi) <= l1, 0.0, reconstructed)


def new_adam_state(w0, w, V):
    """Fresh per-coordinate Adam moment state [m_w0, v_w0, t_w0, m_w, v_w, t_w,
    m_V, v_V, t_V] for round-tripping across epoch-by-epoch calls (early stopping).
    Shapes follow (w0, w, V), so this serves both binary FM (scalar w0) and
    multiclass FM (per-class w0). Arrays are mutated in place; the scalar binary
    w0 moments are written back by the trainer. See the `adam_state` argument."""
    z = np.zeros_like
    return [z(w0), z(w0), z(w0), z(w), z(w), z(w), z(V), z(V), z(V)]


def new_ftrl_state(w0, w, V):
    """Fresh per-coordinate FTRL state [z_w0, n_w0, z_w, n_w, z_V, n_V] for
    round-tripping the (z, n) state across epoch-by-epoch calls (early stopping).
    Shapes follow (w0, w, V), so this serves both binary FM (scalar w0) and
    multiclass FM (per-class w0); the scalar binary z_w0/n_w0 are written back by
    the trainer (the array slots mutate in place). The FTRL counterpart of
    `new_adam_state`. See the `ftrl_state` argument."""
    z = np.zeros_like
    return [z(w0), z(w0), z(w), z(w), z(V), z(V)]


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


def init_ffm_multiclass_params(rng, n_classes, n_features, n_fields, n_factors, init_scale):
    """Per-class FFM params: w0 (C,), w (C, n), V (C, n, n_fields, k) (libffm-style)."""
    w0 = np.zeros(n_classes)
    w = np.zeros((n_classes, n_features))
    V = rng.uniform(
        0.0, init_scale / math.sqrt(n_factors), size=(n_classes, n_features, n_fields, n_factors)
    )
    return w0, w, V


def make_row_orders(rng, n_rows, epochs, shuffle=True):
    """(epochs, n_rows) row visit order; one fresh permutation per epoch."""
    if shuffle:
        return np.stack([rng.permutation(n_rows) for _ in range(epochs)])
    return np.tile(np.arange(n_rows), (epochs, 1))


def _iter_batches(order, batch_size):
    """Yield contiguous row-index chunks of length <= batch_size, in order.

    batch_size=1 yields one row per chunk, so the per-batch update below reduces
    exactly to the per-row update (docs/optimization_spec.md, "Mini-batch").
    """
    order = np.asarray(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def _touched(batch, rows):
    """Sorted unique feature indices touched by any row in the batch."""
    parts = [rows[r][0] for r in batch]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts))


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
    l1_linear=0.0,
    l1_factors=0.0,
    row_orders=None,
    sample_weight=None,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    ftrl_beta=1.0,
    batch_size=1,
    state=None,
    adam_state=None,
    ftrl_state=None,
):
    """Train an FM from `params` = (w0, w, V); returns new (w0, w, V) copies.

    For logistic loss, y must be 0/1 (label smoothing produces a soft target in
    [0, 1], which is also fine). `sample_weight` scales each row's gradient; L2
    is unscaled. `batch_size` averages each batch's gradient and applies one
    update per touched coordinate (docs/optimization_spec.md, "Mini-batch");
    batch_size=1 is the per-row path. CSR input must be canonical (no duplicate
    column indices within a row).

    `state` round-trips AdaGrad accumulators across epoch-by-epoch calls (early
    stopping). `adam_state` (see `new_adam_state`) does the same for Adam moments;
    it is the Adam counterpart of `state` and the two are mutually exclusive.
    `ftrl_state` (see `new_ftrl_state`) does the same for FTRL's per-coordinate
    (z, n) state.
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
    adam = optimizer == "adam"
    ftrl = optimizer == "ftrl"
    if adam_state is not None and not adam:
        raise ValueError("adam_state is only valid for optimizer='adam'")
    if ftrl_state is not None and not ftrl:
        raise ValueError("ftrl_state is only valid for optimizer='ftrl'")
    lr = learning_rate
    if state is None:
        a_w0, a_w, a_V = 0.0, np.zeros_like(w), np.zeros_like(V)
    else:
        a_w0, a_w, a_V = state  # accumulators persist across epoch-driven calls
    if adam and adam_state is None:
        m_w0, v_w0, t_w0 = 0.0, 0.0, 0.0
        m_w, v_w, t_w = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
        m_V, v_V, t_V = np.zeros_like(V), np.zeros_like(V), np.zeros_like(V)
    elif adam:  # round-trip moments across epochs (arrays mutated in place)
        m_w0, v_w0, t_w0 = adam_state[0], adam_state[1], adam_state[2]
        m_w, v_w, t_w = adam_state[3], adam_state[4], adam_state[5]
        m_V, v_V, t_V = adam_state[6], adam_state[7], adam_state[8]
    if ftrl and ftrl_state is None:
        z_w0, n_w0 = 0.0, 0.0
        z_w, n_w = np.zeros_like(w), np.zeros_like(w)
        z_V, n_V = np.zeros_like(V), np.zeros_like(V)
    elif ftrl:  # round-trip (z, n) across epochs (arrays mutated in place)
        z_w0, n_w0 = ftrl_state[0], ftrl_state[1]
        z_w, n_w = ftrl_state[2], ftrl_state[3]
        z_V, n_V = ftrl_state[4], ftrl_state[5]
    for order in np.asarray(row_orders):
        for batch in _iter_batches(order, batch_size):
            bsz = len(batch)
            # pass 1: per-row data-gradients from the frozen (batch-start) params
            g_w0 = 0.0
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)
            for r in batch:
                idx, val = rows[r]
                Vi = V[idx]  # (z, k): pre-update factors, frozen for the batch
                cache = Vi.T @ val  # (k,) = sum_i v_{i,f} x_i
                s = w0 + w[idx] @ val + 0.5 * (cache @ cache - ((Vi * val[:, None]) ** 2).sum())
                g = (_sigmoid(s) - y[r]) if logistic else (s - y[r])
                g *= sw[r]
                g_w0 += g
                gw[idx] += g * val
                gV[idx] += g * (val[:, None] * cache[None, :] - Vi * (val**2)[:, None])
            # pass 2: one update per touched coordinate (batch-mean grad + lazy L2)
            tch = _touched(batch, rows)
            g0 = g_w0 / bsz
            if ftrl:  # FTRL folds L1/L2 into its own update; pass the data gradient
                w0, z_w0, n_w0 = _ftrl_scalar(w0, g0, z_w0, n_w0, lr, ftrl_beta, 0.0, 0.0)
                _ftrl_array(w, gw[tch] / bsz, z_w, n_w, tch, lr, ftrl_beta, l1_linear, l2_linear)
                _ftrl_array(V, gV[tch] / bsz, z_V, n_V, tch, lr, ftrl_beta, l1_factors, l2_factors)
                continue
            g_w = gw[tch] / bsz + l2_linear * w[tch]
            g_V = gV[tch] / bsz + l2_factors * V[tch]
            Vt = V[tch]  # frozen factors for the SGD-form "Vt - lr * g_V"
            if adagrad:
                a_w0 += g0 * g0
                w0 -= lr * g0 / math.sqrt(a_w0 + ADAGRAD_EPS)
                a_w[tch] += g_w**2
                w[tch] -= lr * g_w / np.sqrt(a_w[tch] + ADAGRAD_EPS)
                a_V[tch] += g_V**2
                V[tch] = Vt - lr * g_V / np.sqrt(a_V[tch] + ADAGRAD_EPS)
            elif adam:
                w0, m_w0, v_w0, t_w0 = _adam_scalar(
                    w0, g0, m_w0, v_w0, t_w0, lr, beta_1, beta_2, epsilon
                )
                _adam_array(w, g_w, m_w, v_w, t_w, tch, lr, beta_1, beta_2, epsilon)
                _adam_array(V, g_V, m_V, v_V, t_V, tch, lr, beta_1, beta_2, epsilon)
            else:
                w0 -= lr * g0
                w[tch] -= lr * g_w
                V[tch] = Vt - lr * g_V
    if state is not None:
        state[0], state[1], state[2] = a_w0, a_w, a_V
    if adam_state is not None:  # moment arrays mutate in place; write the scalars
        adam_state[0], adam_state[1], adam_state[2] = m_w0, v_w0, t_w0
    if ftrl_state is not None:  # (z, n) arrays mutate in place; write the scalars
        ftrl_state[0], ftrl_state[1] = z_w0, n_w0
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
    l1_linear=0.0,
    l1_factors=0.0,
    row_orders=None,
    label_smoothing=0.0,
    sample_weight=None,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    ftrl_beta=1.0,
    batch_size=1,
    state=None,
    adam_state=None,
    ftrl_state=None,
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
    adam = optimizer == "adam"
    ftrl = optimizer == "ftrl"
    if adam_state is not None and not adam:
        raise ValueError("adam_state is only valid for optimizer='adam'")
    if ftrl_state is not None and not ftrl:
        raise ValueError("ftrl_state is only valid for optimizer='ftrl'")
    lr = learning_rate
    eps = label_smoothing
    off = eps / (n_classes - 1) if n_classes > 1 else 0.0
    if state is None:
        a_w0, a_w, a_V = np.zeros_like(w0), np.zeros_like(w), np.zeros_like(V)
    else:
        a_w0, a_w, a_V = state  # AdaGrad accumulators persist across epoch-driven calls
    if adam and adam_state is None:
        m_w0, v_w0, t_w0 = np.zeros_like(w0), np.zeros_like(w0), np.zeros_like(w0)
        m_w, v_w, t_w = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
        m_V, v_V, t_V = np.zeros_like(V), np.zeros_like(V), np.zeros_like(V)
    elif adam:  # round-trip moments across epochs (all arrays mutated in place)
        m_w0, v_w0, t_w0 = adam_state[0], adam_state[1], adam_state[2]
        m_w, v_w, t_w = adam_state[3], adam_state[4], adam_state[5]
        m_V, v_V, t_V = adam_state[6], adam_state[7], adam_state[8]
    if ftrl and ftrl_state is None:
        z_w0, n_w0 = np.zeros_like(w0), np.zeros_like(w0)
        z_w, n_w = np.zeros_like(w), np.zeros_like(w)
        z_V, n_V = np.zeros_like(V), np.zeros_like(V)
    elif ftrl:  # round-trip (z, n) across epochs (arrays mutated in place)
        z_w0, n_w0 = ftrl_state[0], ftrl_state[1]
        z_w, n_w = ftrl_state[2], ftrl_state[3]
        z_V, n_V = ftrl_state[4], ftrl_state[5]
    for order in np.asarray(row_orders):
        for batch in _iter_batches(order, batch_size):
            bsz = len(batch)
            # pass 1: per-row softmax gradients from the frozen (batch-start) params
            g_w0 = np.zeros(n_classes)
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)
            for r in batch:
                idx, val = rows[r]
                yc = int(y[r])
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
                    g_w0[c] += g
                    gw[c][idx] += g * val
                    gV[c][idx] += g * (val[:, None] * cache[None, :] - Vi * (val**2)[:, None])
            # pass 2: one update per (class, touched coordinate)
            tch = _touched(batch, rows)
            for c in range(n_classes):
                g0 = g_w0[c] / bsz
                if ftrl:
                    w0[c], z_w0[c], n_w0[c] = _ftrl_scalar(
                        w0[c], g0, z_w0[c], n_w0[c], lr, ftrl_beta, 0.0, 0.0
                    )
                    _ftrl_array(w[c], gw[c][tch] / bsz, z_w[c], n_w[c], tch, lr, ftrl_beta,
                                l1_linear, l2_linear)
                    _ftrl_array(V[c], gV[c][tch] / bsz, z_V[c], n_V[c], tch, lr, ftrl_beta,
                                l1_factors, l2_factors)
                    continue
                g_w = gw[c][tch] / bsz + l2_linear * w[c][tch]
                g_V = gV[c][tch] / bsz + l2_factors * V[c][tch]
                Vt = V[c][tch]
                if adagrad:
                    a_w0[c] += g0 * g0
                    w0[c] -= lr * g0 / math.sqrt(a_w0[c] + ADAGRAD_EPS)
                    a_w[c][tch] += g_w**2
                    w[c][tch] -= lr * g_w / np.sqrt(a_w[c][tch] + ADAGRAD_EPS)
                    a_V[c][tch] += g_V**2
                    V[c][tch] = Vt - lr * g_V / np.sqrt(a_V[c][tch] + ADAGRAD_EPS)
                elif adam:
                    w0[c], m_w0[c], v_w0[c], t_w0[c] = _adam_scalar(
                        w0[c], g0, m_w0[c], v_w0[c], t_w0[c], lr, beta_1, beta_2, epsilon
                    )
                    _adam_array(w[c], g_w, m_w[c], v_w[c], t_w[c], tch, lr, beta_1, beta_2, epsilon)
                    _adam_array(V[c], g_V, m_V[c], v_V[c], t_V[c], tch, lr, beta_1, beta_2, epsilon)
                else:
                    w0[c] -= lr * g0
                    w[c][tch] -= lr * g_w
                    V[c][tch] = Vt - lr * g_V
    # multiclass acc/moments are per-class arrays mutated in place; the writes
    # keep parity with the binary trainer's hand-off (no-ops here).
    if state is not None:
        state[0], state[1], state[2] = a_w0, a_w, a_V
    if adam_state is not None:
        adam_state[0], adam_state[1], adam_state[2] = m_w0, v_w0, t_w0
    if ftrl_state is not None:
        ftrl_state[0], ftrl_state[1] = z_w0, n_w0
    return w0, w, V


def ffm_fit_reference(
    X,
    y,
    field_ids,
    params,
    *,
    loss="logistic",
    optimizer="adagrad",
    learning_rate=0.05,
    l2_linear=0.0,
    l2_factors=0.0,
    l1_linear=0.0,
    l1_factors=0.0,
    row_orders=None,
    sample_weight=None,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    ftrl_beta=1.0,
    batch_size=1,
    state=None,
    adam_state=None,
    ftrl_state=None,
):
    """Train an FFM from `params` = (w0, w, V); returns copies.

    `loss` is "logistic" (y in {0, 1}, or a soft target in [0, 1]) or "squared"
    (regression, real-valued y). V has shape (n_features, n_fields, k).
    `sample_weight` scales each row's gradient. `batch_size` averages each
    batch's gradient and applies one update per touched (feature, field) slot
    (docs/optimization_spec.md); batch_size=1 is the per-row path.
    """
    _check_loss_optimizer(loss, optimizer)
    field_ids = np.asarray(field_ids, dtype=np.int64)
    w0, w, V = params
    w0 = float(w0)
    w = np.array(w, dtype=np.float64, copy=True)
    V = np.array(V, dtype=np.float64, copy=True)
    y = np.asarray(y, dtype=np.float64)
    n_fields = V.shape[1]
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
    adam = optimizer == "adam"
    ftrl = optimizer == "ftrl"
    if adam_state is not None and not adam:
        raise ValueError("adam_state is only valid for optimizer='adam'")
    if ftrl_state is not None and not ftrl:
        raise ValueError("ftrl_state is only valid for optimizer='ftrl'")
    lr = learning_rate
    if state is None:
        a_w0, a_w, a_V = 0.0, np.zeros_like(w), np.zeros_like(V)
    else:
        a_w0, a_w, a_V = state  # accumulators persist across epoch-driven calls
    if adam and adam_state is None:
        m_w0, v_w0, t_w0 = 0.0, 0.0, 0.0
        m_w, v_w, t_w = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
        m_V, v_V, t_V = np.zeros_like(V), np.zeros_like(V), np.zeros_like(V)
    elif adam:  # round-trip moments across epochs (arrays mutated in place)
        m_w0, v_w0, t_w0 = adam_state[0], adam_state[1], adam_state[2]
        m_w, v_w, t_w = adam_state[3], adam_state[4], adam_state[5]
        m_V, v_V, t_V = adam_state[6], adam_state[7], adam_state[8]
    if ftrl and ftrl_state is None:
        z_w0, n_w0 = 0.0, 0.0
        z_w, n_w = np.zeros_like(w), np.zeros_like(w)
        z_V, n_V = np.zeros_like(V), np.zeros_like(V)
    elif ftrl:  # round-trip (z, n) across epochs (arrays mutated in place)
        z_w0, n_w0 = ftrl_state[0], ftrl_state[1]
        z_w, n_w = ftrl_state[2], ftrl_state[3]
        z_V, n_V = ftrl_state[4], ftrl_state[5]
    for order in np.asarray(row_orders):
        for batch in _iter_batches(order, batch_size):
            bsz = len(batch)
            # pass 1: per-row data-gradients from the frozen (batch-start) params,
            # accumulated globally per feature and per (feature, field) slot
            g_w0 = 0.0
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)  # (n_features, n_fields, k)
            tslot = np.zeros((len(w), n_fields), dtype=bool)  # touched (feature, field)
            for r in batch:
                idx, val = rows[r]
                z = len(idx)
                f = field_ids[idx]
                s = w0 + w[idx] @ val
                for a in range(z):
                    for b in range(a + 1, z):
                        s += (V[idx[a], f[b]] @ V[idx[b], f[a]]) * val[a] * val[b]
                g = ((_sigmoid(s) - y[r]) if logistic else (s - y[r])) * sw[r]
                g_w0 += g
                gw[idx] += g * val
                for a in range(z):
                    for b in range(a + 1, z):
                        coef = g * val[a] * val[b]
                        gV[idx[a], f[b]] += coef * V[idx[b], f[a]]
                        gV[idx[b], f[a]] += coef * V[idx[a], f[b]]
                        tslot[idx[a], f[b]] = True
                        tslot[idx[b], f[a]] = True
            # pass 2: w0, w, then touched V slots (feature ascending, field ascending)
            tch = _touched(batch, rows)
            g0 = g_w0 / bsz
            if ftrl:
                w0, z_w0, n_w0 = _ftrl_scalar(w0, g0, z_w0, n_w0, lr, ftrl_beta, 0.0, 0.0)
                _ftrl_array(w, gw[tch] / bsz, z_w, n_w, tch, lr, ftrl_beta, l1_linear, l2_linear)
                for i in tch:
                    for fld in range(n_fields):
                        if tslot[i, fld]:
                            _ftrl_array(V, gV[i, fld] / bsz, z_V, n_V, (i, fld), lr, ftrl_beta,
                                        l1_factors, l2_factors)
                continue
            g_w = gw[tch] / bsz + l2_linear * w[tch]
            if adagrad:
                a_w0 += g0 * g0
                w0 -= lr * g0 / math.sqrt(a_w0 + ADAGRAD_EPS)
                a_w[tch] += g_w**2
                w[tch] -= lr * g_w / np.sqrt(a_w[tch] + ADAGRAD_EPS)
                for i in tch:
                    for fld in range(n_fields):
                        if tslot[i, fld]:
                            grad = gV[i, fld] / bsz + l2_factors * V[i, fld]
                            a_V[i, fld] += grad**2
                            V[i, fld] -= lr * grad / np.sqrt(a_V[i, fld] + ADAGRAD_EPS)
            elif adam:
                w0, m_w0, v_w0, t_w0 = _adam_scalar(
                    w0, g0, m_w0, v_w0, t_w0, lr, beta_1, beta_2, epsilon
                )
                _adam_array(w, g_w, m_w, v_w, t_w, tch, lr, beta_1, beta_2, epsilon)
                for i in tch:
                    for fld in range(n_fields):
                        if tslot[i, fld]:
                            grad = gV[i, fld] / bsz + l2_factors * V[i, fld]
                            _adam_array(
                                V, grad, m_V, v_V, t_V, (i, fld), lr, beta_1, beta_2, epsilon
                            )
            else:
                w0 -= lr * g0
                w[tch] -= lr * g_w
                for i in tch:
                    for fld in range(n_fields):
                        if tslot[i, fld]:
                            grad = gV[i, fld] / bsz + l2_factors * V[i, fld]
                            V[i, fld] -= lr * grad
    if state is not None:
        state[0], state[1], state[2] = a_w0, a_w, a_V
    if adam_state is not None:  # moment arrays mutate in place; write the scalars
        adam_state[0], adam_state[1], adam_state[2] = m_w0, v_w0, t_w0
    if ftrl_state is not None:  # (z, n) arrays mutate in place; write the scalars
        ftrl_state[0], ftrl_state[1] = z_w0, n_w0
    return w0, w, V


def ffm_fit_multiclass_reference(
    X,
    y,
    field_ids,
    params,
    *,
    optimizer="adagrad",
    learning_rate=0.05,
    l2_linear=0.0,
    l2_factors=0.0,
    l1_linear=0.0,
    l1_factors=0.0,
    row_orders=None,
    label_smoothing=0.0,
    sample_weight=None,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    ftrl_beta=1.0,
    batch_size=1,
    state=None,
    adam_state=None,
    ftrl_state=None,
):
    """Train a multiclass (softmax) FFM: one FFM per class, coupled by softmax.

    `params` = (w0 (C,), w (C, n), V (C, n, n_fields, k)); `y` holds class indices
    in [0, C). The gradient on class-c's logit is sample_weight * (p_c - target_c)
    (target uses label smoothing); each class's FFM is then updated with the binary
    FFM gradient (classes share no parameters). NumPy ground truth for the Rust
    `ffm_fit_multiclass_csr` kernel.
    """
    _check_loss_optimizer("logistic", optimizer)
    field_ids = np.asarray(field_ids, dtype=np.int64)
    w0, w, V = params
    w0 = np.array(w0, dtype=np.float64, copy=True)  # (C,)
    w = np.array(w, dtype=np.float64, copy=True)  # (C, n)
    V = np.array(V, dtype=np.float64, copy=True)  # (C, n, n_fields, k)
    y = np.asarray(y)
    n_classes, n_features, n_fields = V.shape[0], V.shape[1], V.shape[2]
    rows = list(_as_dense_rows(X))
    sw = (
        np.ones(len(rows))
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if row_orders is None:
        row_orders = np.arange(len(rows))[None, :]
    adagrad = optimizer == "adagrad"
    adam = optimizer == "adam"
    ftrl = optimizer == "ftrl"
    if adam_state is not None and not adam:
        raise ValueError("adam_state is only valid for optimizer='adam'")
    if ftrl_state is not None and not ftrl:
        raise ValueError("ftrl_state is only valid for optimizer='ftrl'")
    lr = learning_rate
    eps = label_smoothing
    off = eps / (n_classes - 1) if n_classes > 1 else 0.0
    if state is None:
        a_w0, a_w, a_V = np.zeros_like(w0), np.zeros_like(w), np.zeros_like(V)
    else:
        a_w0, a_w, a_V = state  # AdaGrad accumulators persist across epoch-driven calls
    if adam and adam_state is None:
        m_w0, v_w0, t_w0 = np.zeros_like(w0), np.zeros_like(w0), np.zeros_like(w0)
        m_w, v_w, t_w = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
        m_V, v_V, t_V = np.zeros_like(V), np.zeros_like(V), np.zeros_like(V)
    elif adam:  # round-trip moments across epochs (all arrays mutated in place)
        m_w0, v_w0, t_w0 = adam_state[0], adam_state[1], adam_state[2]
        m_w, v_w, t_w = adam_state[3], adam_state[4], adam_state[5]
        m_V, v_V, t_V = adam_state[6], adam_state[7], adam_state[8]
    if ftrl and ftrl_state is None:
        z_w0, n_w0 = np.zeros_like(w0), np.zeros_like(w0)
        z_w, n_w = np.zeros_like(w), np.zeros_like(w)
        z_V, n_V = np.zeros_like(V), np.zeros_like(V)
    elif ftrl:  # round-trip (z, n) across epochs (arrays mutated in place)
        z_w0, n_w0 = ftrl_state[0], ftrl_state[1]
        z_w, n_w = ftrl_state[2], ftrl_state[3]
        z_V, n_V = ftrl_state[4], ftrl_state[5]
    for order in np.asarray(row_orders):
        for batch in _iter_batches(order, batch_size):
            bsz = len(batch)
            g_w0 = np.zeros(n_classes)
            gw = np.zeros_like(w)  # (C, n)
            gV = np.zeros_like(V)  # (C, n, n_fields, k)
            tslot = np.zeros((n_features, n_fields), dtype=bool)  # shared across classes
            for r in batch:
                idx, val = rows[r]
                z = len(idx)
                f = field_ids[idx]
                yc = int(y[r])
                # pass 1: per-class FFM logit from the frozen parameters
                logits = np.empty(n_classes)
                for c in range(n_classes):
                    s = w0[c] + w[c][idx] @ val
                    for a in range(z):
                        for b in range(a + 1, z):
                            s += (V[c][idx[a], f[b]] @ V[c][idx[b], f[a]]) * val[a] * val[b]
                    logits[c] = s
                ex = np.exp(logits - logits.max())  # stable softmax
                p = ex / ex.sum()
                # pass 2: accumulate each class's FFM gradient
                for c in range(n_classes):
                    target = (1.0 - eps) if c == yc else off
                    g = sw[r] * (p[c] - target)
                    g_w0[c] += g
                    gw[c][idx] += g * val
                    for a in range(z):
                        for b in range(a + 1, z):
                            coef = g * val[a] * val[b]
                            gV[c][idx[a], f[b]] += coef * V[c][idx[b], f[a]]
                            gV[c][idx[b], f[a]] += coef * V[c][idx[a], f[b]]
                            tslot[idx[a], f[b]] = True
                            tslot[idx[b], f[a]] = True
            # flush each class over the shared touched feature/slot sets
            tch = _touched(batch, rows)
            for c in range(n_classes):
                g0 = g_w0[c] / bsz
                if ftrl:
                    w0[c], z_w0[c], n_w0[c] = _ftrl_scalar(
                        w0[c], g0, z_w0[c], n_w0[c], lr, ftrl_beta, 0.0, 0.0
                    )
                    _ftrl_array(w[c], gw[c][tch] / bsz, z_w[c], n_w[c], tch, lr, ftrl_beta,
                                l1_linear, l2_linear)
                    for i in tch:
                        for fld in range(n_fields):
                            if tslot[i, fld]:
                                _ftrl_array(V[c], gV[c][i, fld] / bsz, z_V[c], n_V[c], (i, fld),
                                            lr, ftrl_beta, l1_factors, l2_factors)
                    continue
                g_w = gw[c][tch] / bsz + l2_linear * w[c][tch]
                if adagrad:
                    a_w0[c] += g0 * g0
                    w0[c] -= lr * g0 / math.sqrt(a_w0[c] + ADAGRAD_EPS)
                    a_w[c][tch] += g_w**2
                    w[c][tch] -= lr * g_w / np.sqrt(a_w[c][tch] + ADAGRAD_EPS)
                    for i in tch:
                        for fld in range(n_fields):
                            if tslot[i, fld]:
                                grad = gV[c][i, fld] / bsz + l2_factors * V[c][i, fld]
                                a_V[c][i, fld] += grad**2
                                V[c][i, fld] -= lr * grad / np.sqrt(a_V[c][i, fld] + ADAGRAD_EPS)
                elif adam:
                    w0[c], m_w0[c], v_w0[c], t_w0[c] = _adam_scalar(
                        w0[c], g0, m_w0[c], v_w0[c], t_w0[c], lr, beta_1, beta_2, epsilon
                    )
                    _adam_array(w[c], g_w, m_w[c], v_w[c], t_w[c], tch, lr, beta_1, beta_2, epsilon)
                    for i in tch:
                        for fld in range(n_fields):
                            if tslot[i, fld]:
                                grad = gV[c][i, fld] / bsz + l2_factors * V[c][i, fld]
                                _adam_array(V[c], grad, m_V[c], v_V[c], t_V[c], (i, fld), lr,
                                            beta_1, beta_2, epsilon)
                else:  # sgd
                    w0[c] -= lr * g0
                    w[c][tch] -= lr * g_w
                    for i in tch:
                        for fld in range(n_fields):
                            if tslot[i, fld]:
                                grad = gV[c][i, fld] / bsz + l2_factors * V[c][i, fld]
                                V[c][i, fld] -= lr * grad
    if state is not None:
        state[0], state[1], state[2] = a_w0, a_w, a_V
    if adam_state is not None:
        adam_state[0], adam_state[1], adam_state[2] = m_w0, v_w0, t_w0
    if ftrl_state is not None:
        ftrl_state[0], ftrl_state[1] = z_w0, n_w0
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
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
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
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )


def ffm_train(
    X,
    y,
    field_ids,
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
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
):
    """Seeded end-to-end FFM training (logistic or squared loss)."""
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
        loss=loss,
        optimizer=optimizer,
        learning_rate=learning_rate,
        l2_linear=l2_linear,
        l2_factors=l2_factors,
        row_orders=row_orders,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )
