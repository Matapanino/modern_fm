# Optimization Specification (v0.1)

Defines the training algorithms the Rust backend implements in Phase 2.
The gradients follow directly from `math_spec.md`.

## Gradients

Let `s = y_hat(x)` and `dL/ds` the loss derivative
(logistic: `sigmoid(s) - y_smooth`; squared: `s - y`;
softmax: `p_c - target_c` per class logit).

FM parameter gradients for one row `x`:

```
ds/dw0        = 1
ds/dw_i       = x_i
ds/dv_{i,f}   = x_i * (sum_j v_{j,f} x_j) - v_{i,f} * x_i^2
```

(The term `sum_j v_{j,f} x_j` is shared across i — compute once per row/factor.)

FFM gradients for nonzero pair (i, j):

```
ds/dV[i, f_j] += x_i x_j * V[j, f_i]
ds/dV[j, f_i] += x_i x_j * V[i, f_j]
```

L2 adds `l2_linear * w_i` and `l2_factors * v` to the respective gradients
(only for parameters touched by the row — lazy/sparse regularization).

## SGD

```
theta -= learning_rate * grad
```

## AdaGrad (v0.1 default)

Per-parameter accumulator `G` (init 0), epsilon `1e-10`:

```
G     += grad^2
theta -= learning_rate * grad / sqrt(G + epsilon)
```

This is what libffm uses for FFM and is robust on sparse data.

## Adam (v0.2)

Per-parameter **lazy** Adam. Each coordinate keeps its own moments `(m, v)` and an
update count `t`; `t` increments only when the coordinate is touched by a row
(consistent with lazy L2 above), so bias correction adapts per coordinate and the
update is exactly reproducible. Hyperparameters `beta_1` (default `0.9`), `beta_2`
(default `0.999`), `epsilon` (default `1e-8`); `learning_rate` is the step size α.

```
t      += 1
m       = beta_1 * m + (1 - beta_1) * grad
v       = beta_2 * v + (1 - beta_2) * grad^2
m_hat   = m / (1 - beta_1^t)
v_hat   = v / (1 - beta_2^t)
theta  -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

`epsilon` is added **outside** the square root. `grad` already includes lazy L2
(`+ l2_linear * w_i` / `+ l2_factors * v`), exactly as for SGD/AdaGrad. The Rust
kernel uses `beta.powf(t)` to match Python's `beta ** t`.

In the Rust kernel the `(m, v, t)` accumulators are internal to a single `fit`
call. Adam **+ early stopping** is supported by round-tripping the moments across
epochs through the NumPy reference trainer (`fm_fit_reference`'s `adam_state`
argument; the per-epoch hand-off equals one multi-epoch call exactly). Because
that path is the reference, not the Rust kernel, Adam + early stopping is suited
to the moderate data early stopping is typically used on; for large data use Adam
without early stopping (Rust) or AdaGrad with early stopping (Rust round-trip).

## FTRL-Proximal (v0.2)

Per-coordinate FTRL-Proximal (McMahan et al. 2013), the standard CTR optimizer.
Unlike SGD/AdaGrad/Adam, **L1/L2 are folded into the update, not the gradient** —
so the kernels pass FTRL the *pure data gradient* `grad` (no `+ l2 * theta`). Each
coordinate keeps state `(z, n)` (init 0); the weight is reconstructed from `(z, n)`
each step, so `theta` always holds the current FTRL weight. Hyperparameters:
`learning_rate` is α (step size); `ftrl_beta` is β (default `1.0`); `l1_linear`/
`l1_factors` are the L1 strengths (default `0.0`); `l2_linear`/`l2_factors` reused
as the L2 strengths. The bias `w0` uses `l1 = l2 = 0`.

```
sigma  = (sqrt(n + grad^2) - sqrt(n)) / alpha
z      += grad - sigma * theta
n      += grad^2
theta   = 0                                          if |z| <= l1
        = -(z - sign(z)*l1) / ((beta + sqrt(n))/alpha + l2)   otherwise
```

The `(z, n)` state is internal to one `fit` call; L1 produces **exact zeros**
(`|z| <= l1`), which AdaGrad/Adam never do. With mini-batch, FTRL steps once per
batch on the batch-mean data gradient; with `n_jobs>1` the parallel accumulator
sums only data gradients, so FTRL composes with both unchanged. FTRL + early
stopping is **deferred in v0.2** (its `(z, n)` state is not round-tripped), like
Adam. `l1_* > 0` with a non-FTRL optimizer raises `ValueError`.

## Training loop

- shuffle row order each epoch with the seeded RNG
- mini-batch size `batch_size` (gradient averaged over batch); FFM commonly
  uses batch_size=1 (pure SGD) — both must work
- initialization: `w0 = 0`, `w = 0`,
  `V ~ Normal(0, init_scale)` (FM) / `Uniform(0, init_scale/sqrt(k))` (FFM,
  following libffm) from the seeded RNG
- early stopping: evaluate `eval_metric` on eval_set each epoch; stop after
  `patience` epochs without `min_delta` improvement; optionally restore best

## Mini-batch (v0.2)

Each epoch's shuffled row order is consumed in contiguous chunks of `batch_size`
(the last chunk may be shorter). Within one batch:

1. **Parameters are frozen at batch start.** Every row's score `s` and per-row
   data-gradient `dL/ds * ds/dtheta` are computed against the batch-start
   parameters (no row in a batch sees another row's update).
2. **Accumulate the data-gradient** per touched coordinate, summed over the rows
   in the batch (FM linear `g x_i`; FM factor
   `g (x_i * cache_f - v_{i,f} x_i^2)`; FFM per `(feature, field)` slot — exactly
   the batch_size=1 terms, just summed).
3. **One update per touched coordinate**, applied once at batch end:
   `g_theta = (1/B) * sum_rows(data-grad) + l2 * theta`, where `B` is the number
   of rows in the batch and `theta` is the batch-start value. Lazy L2 is added
   once here (not per row). The optimizer step (SGD / AdaGrad / Adam) then runs
   once: AdaGrad accumulators `G += g_theta^2` and Adam's per-coordinate `t`
   advance **once per batch**, not once per row.

Coordinates untouched by the batch are neither regularized nor stepped (lazy).
`w0` is touched by every row, so it is updated every batch. The batch-mean
divides by the actual batch length, so a full final partial batch is handled
correctly.

**`batch_size=1` reduces exactly** to the per-row update above (one row per
batch ⇒ mean over one row, L2 on the frozen value = current value), so it is
bit-for-bit identical to the v0.1 path. Rows within a batch are accumulated in
`row_orders` order so the floating-point sum matches between the NumPy reference
and the Rust kernel.

## Parallelism (Rust, v0.2)

Deterministic **parallel-accumulate, serial-apply** with `rayon` (`n_jobs`
threads), not lock-free Hogwild — chosen so results stay reproducible and
parity-testable. Within each batch:

- the batch's rows are split into `n_jobs` contiguous chunks; each chunk's
  gradients are accumulated into its own partial accumulator on a separate
  thread (the frozen batch-start parameters are read-only, so no races);
- the partial accumulators are reduced in chunk order into one, then a single
  serial flush applies the updates (and steps the AdaGrad/Adam state once).

Parallelism therefore only acts when `batch_size > 1` (a `batch_size=1` batch is
a single row/chunk). The contract:

- **`n_jobs=1` is the canonical serial path** and is bit-identical to the NumPy
  reference;
- **`n_jobs>1` is reproducible for a fixed `n_jobs`** (contiguous chunking +
  fixed-order reduction), differing from `n_jobs=1` only in floating-point
  summation order (parity-tested to a loose tolerance);
- different `n_jobs` values may differ at that tolerance.

FM binary and FFM parallelize; multiclass-softmax FM trains serially in v0.2
(`n_jobs` is accepted but ignored for it).
