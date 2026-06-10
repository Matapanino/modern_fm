# Requirements

## v0.1 Scope

Implement high-performance FM and FFM estimators with a sklearn-like Python API.
Backend: Rust CPU (PyO3/maturin). Ground truth: pure-NumPy reference
implementations in `python/modern_fm/_reference.py`.

## Supported tasks

### Regression
- Squared loss
- Optional MAE metric in evaluation helper

### Binary classification
- Logistic loss
- `predict_proba`, `predict`, `decision_function`
- `class_weight`, `sample_weight`, `label_smoothing`

### Multiclass classification
- Softmax loss
- `predict_proba`, `predict`
- `class_weight`, `sample_weight`, `label_smoothing`

## Input types

Must support:
- `numpy.ndarray` dense, float32/float64
- `scipy.sparse.csr_matrix` / `csr_array`
- categorical integer arrays through a helper encoder
- explicit `field_ids` array for FFM (`fit(X, y, field_ids=...)`, required —
  no automatic field inference in v0.1)

Nice to have (v0.2+):
- pandas / polars DataFrame
- libffm text format loader

## Optimizers

- v0.1: SGD, AdaGrad
- v0.2: Adam, FTRL-Proximal

## Regularization

- L2 on linear weights (`l2_linear`)
- L2 on latent factors (`l2_factors`)
- v0.2+: optional L1 on linear weights, dropout on pairwise interactions

## Early stopping

- `eval_set`, `eval_metric`, `patience`, `min_delta`, `restore_best_weights`
- or internal split via `early_stopping=True` + `validation_fraction`

## Reproducibility

- `random_state` parameter
- deterministic initialization and data shuffling under a fixed seed

## Serialization

- `save_model(path)` / `load_model(path)`
- pickle-compatible Python wrapper

## Phase plan within v0.1

- Phase 0: docs + package skeleton  ← done
- Phase 1: Python reference predictions + losses + correctness tests  ← current
- Phase 2: Rust CPU backend (CSR bridge, FM/FFM predict + SGD/AdaGrad fit)
- Phase 3: sklearn API polish (mixins, check_is_fitted, validation)
- Phase 4: early_stopping, label_smoothing, class_weight, sample_weight, save/load
