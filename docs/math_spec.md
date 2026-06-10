# Mathematical Specification

This file fixes the exact formulas for v0.1. Implementations (Python reference,
Rust backend, future GPU backends) must match these definitions. Do not mix in
other FM variants (DeepFM, AFM, FwFM, FEFM); those are roadmap items with their
own future specs.

Notation: `n` features, `k` latent factors, input row `x ∈ R^n`,
`<a, b>` is the dot product.

## Factorization Machine (FM)

Parameters: bias `w0 ∈ R`, linear weights `w ∈ R^n`,
latent factors `V ∈ R^{n×k}` (row `v_i` for feature `i`).

Prediction (degree-2 FM, Rendle 2010):

```
y_hat(x) = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j
```

Efficient O(nk) computation of the pairwise term:

```
sum_{i<j} <v_i, v_j> x_i x_j
  = 0.5 * sum_f [ (sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2 ]
```

## Field-aware Factorization Machine (FFM)

Each feature `i` belongs to a field `f_i ∈ {0, ..., F-1}`.
Parameters: `w0`, `w ∈ R^n`, and per-(feature, field) latent vectors
`V ∈ R^{n×F×k}` where `V[i, g]` is the vector feature `i` uses against field `g`.

Interaction for a pair `(i, j)` with `i < j`:

```
interaction(i, j) = <V[i, f_j], V[j, f_i]> x_i x_j
```

Prediction (Juan et al. 2016):

```
y_hat(x) = w0 + sum_i w_i x_i + sum_{i<j} <V[i, f_j], V[j, f_i]> x_i x_j
```

Note: unlike FM there is no O(nk) factorization of the pairwise sum; complexity
is O(z^2 k) in the number of nonzeros `z` per row. This is why FFM defaults to
smaller `k` and why the linear term is sometimes omitted in other libraries —
**we keep w0 and the linear term** (libffm omits them; this is a deliberate
difference, controlled later by hyperparameters if needed).

## Binary logistic loss

Labels `y ∈ {0, 1}`, raw score `s = y_hat(x)`, `p = sigmoid(s)`:

```
loss = - y log(p) - (1 - y) log(1 - p)
```

Numerically stable form (no exp overflow):

```
loss = max(s, 0) - s * y + log1p(exp(-|s|))
```

With label smoothing `eps`:

```
y_smooth = y * (1 - eps) + 0.5 * eps
```

then use `y_smooth` in place of `y`.

## Multiclass softmax loss

One model (one set of FM/FFM parameters) per class `c` produces `logit_c`.

```
p_c = exp(logit_c) / sum_k exp(logit_k)        # compute via logsumexp
loss = - sum_c target_c * log(p_c)
```

With label smoothing `eps` and true class `y`:

```
target_c = 1 - eps              if c == y
target_c = eps / (n_classes-1)  otherwise
```

## Squared loss (regression)

```
loss = 0.5 * (y_hat - y)^2
```

## Reductions and weights

- Per-batch loss is the **weighted mean**: `sum_i sample_weight_i * loss_i / sum_i sample_weight_i`.
- `class_weight` multiplies into `sample_weight` per row before reduction.
- L2 regularization adds `0.5 * l2_linear * ||w||^2 + 0.5 * l2_factors * ||V||^2`
  (bias `w0` is not regularized).
