# modern_fm

High-performance Factorization Machines (FM), Field-aware FM (FFM) and
Field-weighted FM (FwFM) for Python — a scikit-learn-compatible API on a Rust
CPU core (with an optional CUDA backend), built for Kaggle-style tabular data
and CTR-like sparse problems.

## Install

```bash
pip install modern-fm
```

Prebuilt wheels (abi3, Python ≥ 3.10) ship for Linux, macOS and Windows.
Building from source needs a Rust toolchain (1.74+).

## Quickstart

```python
import numpy as np
import scipy.sparse as sp
from modern_fm import FMClassifier, FFMClassifier, CategoricalEncoder

# dense or CSR input; sklearn conventions throughout
X, y = np.random.rand(1000, 50), np.random.randint(0, 2, 1000)
clf = FMClassifier(n_factors=8, max_iter=20, random_state=0).fit(X, y)
proba = clf.predict_proba(X)

# categorical CTR data: one-hot to CSR + field ids for FFM
enc = CategoricalEncoder()
X_onehot = enc.fit_transform(X_categorical)          # scipy CSR
ffm = FFMClassifier(n_factors=4, random_state=0)
ffm.fit(X_onehot, y, field_ids=enc.field_ids_)
```

All estimators support `fit` / `predict` / `predict_proba` /
`decision_function`, `get_params`/`set_params`, `save_model`/`load_model`,
`partial_fit` + `warm_start` for streaming, early stopping via
`early_stopping=True` or `eval_set=`, four optimizers
(SGD/AdaGrad/Adam/FTRL), mini-batch and multi-core training, and pass
scikit-learn's full `check_estimator` suite — so `Pipeline`, `GridSearchCV`
and `CalibratedClassifierCV` work directly.

## Models

| Estimator | Model | Tasks |
|---|---|---|
| `FMClassifier` | Factorization Machine | binary + multiclass (softmax) |
| `FMRegressor` | Factorization Machine | regression |
| `FFMClassifier` | Field-aware FM | binary + multiclass (softmax) |
| `FFMRegressor` | Field-aware FM | regression |
| `FwFMClassifier` | Field-weighted FM | binary + multiclass (softmax) |

The exact formulas live in the [math spec](math_spec.md) and the
[FwFM math spec](math_spec_fwfm.md); training updates in the
[optimization spec](optimization_spec.md).

## Where to go next

- [API reference](api_design.md) — constructors, contracts, calibration,
  `top_interactions`, partial_fit semantics.
- [Data format](data_format.md) — CSR expectations, `field_ids`, libffm I/O.
- [GPU backend](gpu_backend_plan.md) — the optional CUDA backend
  (`backend="cuda"`): FM/FFM prediction + FM/FFM binary/regression training.
- [Roadmap](roadmap.md) — shipped milestones and the v1.0 gate.
- Examples: [`examples/`](https://github.com/Matapanino/modern_fm/tree/main/examples)
  — basic usage, probability calibration, top interactions.
