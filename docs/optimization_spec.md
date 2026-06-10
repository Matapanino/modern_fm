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
