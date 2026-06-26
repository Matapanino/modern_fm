# API Design

All estimators follow scikit-learn conventions: `__init__` stores parameters
only, learned attributes end with `_`, `fit` returns `self`,
`get_params`/`set_params` round-trip.

## FMClassifier / FMRegressor

```python
from modern_fm import FMClassifier

model = FMClassifier(
    n_factors=16,
    loss="logistic",          # classifier: "logistic" (binary) / "softmax" (auto for multiclass)
    optimizer="adagrad",      # "sgd" | "adagrad" | "adam" | "ftrl"
    learning_rate=0.05,       # also FTRL's alpha (step size)
    beta_1=0.9,               # Adam 1st-moment decay (optimizer="adam" only)
    beta_2=0.999,             # Adam 2nd-moment decay (optimizer="adam" only)
    epsilon=1e-8,             # Adam denominator epsilon (optimizer="adam" only)
    ftrl_beta=1.0,            # FTRL stabilizer beta (optimizer="ftrl" only)
    max_iter=100,
    batch_size=1,             # 1 = per-row SGD; >1 averages the batch gradient
    l2_linear=1e-5,
    l2_factors=1e-5,
    l1_linear=0.0,            # L1 on linear weights (FTRL only; yields exact zeros)
    l1_factors=0.0,           # L1 on latent factors (FTRL only)
    init_scale=0.01,          # stddev of latent factor init
    label_smoothing=0.0,
    class_weight=None,        # None | "balanced" | dict
    early_stopping=False,
    validation_fraction=0.1,
    patience=10,
    min_delta=0.0,
    dtype="float32",          # "float32" | "float64"
    backend="rust_cpu",       # fixed in v0.1; later "cuda", "torch"
    random_state=None,
    n_jobs=-1,
    verbose=0,
)

model.fit(X, y, sample_weight=None, eval_set=None)
model.predict(X)
model.predict_proba(X)        # classifier only; rows sum to 1
model.decision_function(X)    # classifier only; raw scores / logits
model.save_model(path)
FMClassifier.load_model(path)
```

`FMRegressor` is identical minus `loss`/`class_weight`/`label_smoothing`/
`predict_proba`/`decision_function` (loss is squared error).

## FFMClassifier

Field information is **explicit and required** in v0.1 — automatic field
inference hides bugs that silently degrade accuracy.

```python
from modern_fm import FFMClassifier

model = FFMClassifier(
    n_factors=8,
    optimizer="adagrad",
    learning_rate=0.05,
    max_iter=50,
    l2_linear=1e-5,
    l2_factors=1e-5,
    label_smoothing=0.0,
    random_state=42,
)

model.fit(X, y, field_ids=field_ids)        # field_ids: int array, shape (n_features,)
model.predict_proba(X)                       # field mapping is stored on the model at fit time
```

Binary (logistic) by default; pass a target with >2 classes (or `loss="softmax"`)
to train one FFM per class coupled by softmax — `predict_proba` rows then sum to 1
over `n_classes`.

`field_ids[i]` is the field of feature/column `i`; it is optional — when omitted,
each column becomes its own field, so `fit(X, y)` works under the plain sklearn
API. After `fit`, the model stores `field_ids_` and `n_fields_`; predict-time
calls do not take field_ids.

## Learned attributes (after fit)

- `w0_` (float), `w_` (n_features,), `V_`
  - FM: `V_` shape `(n_features, n_factors)`
  - FFM: `V_` shape `(n_features, n_fields, n_factors)`
- `classes_` (classifiers), `n_features_in_`, `n_iter_`
- FFM: `field_ids_`, `n_fields_`
- multiclass (one parameter set per class): `w0_` shape `(n_classes,)`, `w_` shape
  `(n_classes, n_features)`; FM `V_` shape `(n_classes, n_features, n_factors)`,
  FFM `V_` shape `(n_classes, n_features, n_fields, n_factors)`

## Errors and validation

- shape/dtype validation at `fit`/`predict`, raising `ValueError` with clear messages
- `predict` before `fit` raises `NotFittedError` (sklearn's)
- unknown optimizer/loss strings raise `ValueError` at `fit` time (not `__init__`)
