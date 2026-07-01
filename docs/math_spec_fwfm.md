# Mathematical Specification — FwFM (Field-weighted Factorization Machine)

This file fixes the exact FwFM formulas (Pan et al., "Field-weighted
Factorization Machines for Click-Through Rate Prediction in Display
Advertising", WWW 2018). Implementations (NumPy reference, Rust kernel, future
GPU backends) must match these definitions. Do not substitute other variants
(FFM, FmFM, FEFM, AFM); `docs/math_spec.md` still governs FM/FFM.

Notation: `n` features, `k` latent factors, `F` fields, input row `x ∈ R^n`,
feature `i` belongs to field `f_i ∈ {0, ..., F-1}`, `<a, b>` is the dot
product.

## Model

Parameters:

- bias `w0 ∈ R`
- linear weights `w ∈ R^n`
- latent factors `V ∈ R^{n×k}` (row `v_i` — FM-shaped, NOT per-field like FFM)
- field-pair weights `R ∈ R^{F×F}`, of which only the upper triangle including
  the diagonal is used: the pair `(a, b)` reads `r_{ab} = R[min(a,b), max(a,b)]`.
  The diagonal `R[a, a]` is required because two co-occurring features may
  share a field. Entries below the diagonal are never read or updated.

Prediction:

```
y_hat(x) = w0 + sum_i w_i x_i + sum_{i<j} r_{f_i f_j} <v_i, v_j> x_i x_j
```

With `R = ones` this is exactly the plain FM of `docs/math_spec.md` — the
implementations initialize `R` to ones, so a freshly initialized FwFM predicts
identically to a freshly initialized FM with the same `V` (the "collapse"
property, pinned by tests).

## Computation order (shared bit-for-bit by reference and Rust)

The pairwise term is computed as an explicit O(z²k) double loop over the row's
nonzeros in CSR column order (`a` ascending, then `b > a` ascending), exactly
like the FFM kernel — no FM-style O(zk) factorization is used, so that the
floating-point operation order is identical between the NumPy reference and
the Rust kernel. (A per-field cache factorization exists for FwFM but is a
post-1.0 optimization; introducing it requires changing this spec first.)

## Gradients

For a row with score gradient `g = dL/ds` (see `docs/math_spec.md` for the
losses; FwFM uses the same logistic / squared / softmax losses):

```
ds/dw0        = 1
ds/dw_i       = x_i
ds/dv_i       = x_i * sum_{j != i} r_{f_i f_j} v_j x_j        (k-vector)
ds/dr_{ab}    = sum over row pairs (i<j) with {f_i, f_j} = {a, b} of
                <v_i, v_j> x_i x_j
```

Gradient accumulation over a mini-batch follows the FM/FFM contract
(docs/optimization_spec.md): per-row data-gradients are computed from the
frozen batch-start parameters via the same `a` ascending / `b` ascending pair
loop (each pair `(i, j)` adds `g * r_{f_i f_j} * x_i x_j * v_j` to `gV[i]` and
symmetrically to `gV[j]`, and `g * <v_i, v_j> x_i x_j` to `gR[min,max]`),
averaged over the batch, and applied once per touched coordinate.

## Updates

Update order within a batch flush: `w0`, then `w` over touched features
(ascending), then `V` over touched features (ascending, factor index inner),
then `R` over touched field pairs in ascending `(a, b)` order (a touched pair
is any `(min(f_i,f_j), max(f_i,f_j))` hit by a pair of the batch's rows).

Regularization (lazy, touched coordinates only, matching FM/FFM):

- `w`: `l2_linear` (and `l1_linear` under FTRL)
- `V` **and `R`**: `l2_factors` (and `l1_factors` under FTRL) — `R` is treated
  as part of the interaction parameterization; no separate hyperparameter in
  v0.5
- `w0`: never regularized

All four optimizers (SGD / AdaGrad / Adam / FTRL) apply per coordinate exactly
as specified in docs/optimization_spec.md; `R` coordinates carry their own
optimizer state (AdaGrad accumulator, Adam `(m, v, t)`, FTRL `(z, n)`) shaped
like `R`.

## Multiclass

One FwFM per class coupled by softmax, exactly like the FM/FFM multiclass
kernels: per-class logits from frozen batch-start parameters, gradient
`sample_weight * (p_c - target_c)` with label smoothing, each class
accumulating and flushing independently. Parameters gain a leading class axis:
`w0 (C,)`, `w (C, n)`, `V (C, n, k)`, `R (C, F, F)`.

## Initialization

```
w0 = 0,  w = 0,  V ~ Normal(0, init_scale),  R = ones
```

(`V` follows FM's init, not FFM's uniform/sqrt(k); `R = ones` makes the
initial model a plain FM.)

## Early-stopping / partial_fit optimizer state

The per-epoch state hand-off extends the FM/FFM layout with an `R` group
appended:

```
adam_state = [m_w0, v_w0, t_w0, m_w, v_w, t_w, m_V, v_V, t_V, m_R, v_R, t_R]
ftrl_state = [z_w0, n_w0, z_w, n_w, z_V, n_V, z_R, n_R]
state      = [acc_w0, acc_w, acc_V, acc_R]              (AdaGrad accumulators)
```

Binary states carry scalar `w0` slots (written back by the trainer);
multiclass states are all per-class arrays mutated in place.
