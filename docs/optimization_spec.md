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

The `(m, v, t)` accumulators are internal to a single `fit` call. Adam combined
with early stopping (which drives epochs one at a time and would need the moments
handed back and forth) is **deferred in v0.2** — the estimators raise
`NotImplementedError`, mirroring the multiclass + early-stopping deferral.

## Training loop

- shuffle row order each epoch with the seeded RNG
- mini-batch size `batch_size` (gradient averaged over batch); FFM commonly
  uses batch_size=1 (pure SGD) — both must work
- initialization: `w0 = 0`, `w = 0`,
  `V ~ Normal(0, init_scale)` (FM) / `Uniform(0, init_scale/sqrt(k))` (FFM,
  following libffm) from the seeded RNG
- early stopping: evaluate `eval_metric` on eval_set each epoch; stop after
  `patience` epochs without `min_delta` improvement; optionally restore best

## Parallelism (Rust, Phase 2)

- row-parallel Hogwild-style updates with `rayon` (n_jobs threads);
  acceptable for sparse data, documented as slightly non-deterministic when
  n_jobs > 1 — `n_jobs=1` is the reproducible path
