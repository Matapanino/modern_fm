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
    warm_start=False,         # fit() resumes from the previous solution + optimizer state
    dtype="float32",          # "float32" | "float64"
    backend="rust_cpu",       # or "cuda": requires a cuda-backend build + GPU and
                              # supports FM/FFM prediction only (fit with "rust_cpu",
                              # then set_params(backend="cuda") for inference);
                              # never a silent CPU fallback (gpu_backend_plan.md)
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

## FFMClassifier / FFMRegressor

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

`FFMRegressor` is the squared-loss counterpart (as `FMRegressor` is to
`FMClassifier`): the same constructor minus `loss` / `label_smoothing` /
`class_weight`, and no `predict_proba` / `decision_function` / `classes_`.
`fit(X, y, field_ids=…)` takes the same field mapping and stores `field_ids_` /
`n_fields_`; `predict(X)` returns the raw FFM score (squared-error loss).

## FwFMClassifier

Field-weighted FM (docs/math_spec_fwfm.md): FM-shaped factors `V (n, k)` plus
one learned scalar weight per field pair, `r_ (n_fields, n_fields)` (upper
triangle used), scaling each pairwise interaction. `r_` initializes to ones,
so a fresh FwFM is exactly a plain FM.

```python
from modern_fm import FwFMClassifier

model = FwFMClassifier(n_factors=8, random_state=42)
model.fit(X, y, field_ids=field_ids)   # same field plumbing as FFMClassifier
model.predict_proba(X)
```

The constructor, `fit(X, y, field_ids=…)`, binary/softmax dispatch,
early stopping / `eval_set`, `partial_fit(classes=…, field_ids=…)` and
`warm_start` all mirror `FFMClassifier`. Differences: training is serial in
v0.5 (`n_jobs` is accepted but does not parallelize FwFM), and there is one
extra learned attribute `r_` — `(n_fields, n_fields)` binary,
`(n_classes, n_fields, n_fields)` multiclass — regularized by
`l2_factors` / `l1_factors`.

## BiInteractionPooling (feature transform)

Bi-interaction pooling (He & Chua, SIGIR 2017) as an sklearn transformer — the
k-dim FM pairwise vector before its factor-sum, for downstream models. As a
*predictor* a linear head over it provably collapses to plain FM (NFM = this +
an MLP, which is out of scope), so it ships as a transform, not a model.

```python
from modern_fm import BiInteractionPooling, FMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(
    BiInteractionPooling(FMRegressor(n_factors=8, random_state=0)),
    LogisticRegression(),
).fit(X, y)
```

- `BiInteractionPooling(estimator=None)` clones and fits the given FM
  (`None` -> `FMRegressor(n_factors=8)`); `transform(X)` returns
  `(n_samples, n_factors)` pooled features (multiclass inner FMs pool per
  class, concatenated to `(n_samples, n_classes * n_factors)`);
  `get_feature_names_out()` follows the sklearn convention.
- The fitted FM estimators expose the same features directly via
  `model.bi_interaction(X)` (deliberately **not** named `transform`, so plain
  FMs keep plain-estimator semantics in sklearn tooling).

## Partial fit / warm start (incremental & streaming training)

All five estimators support incremental training:

```python
model.partial_fit(X, y, classes=None, sample_weight=None)                  # FM*
model.partial_fit(X, y, classes=None, field_ids=None, sample_weight=None)  # FFM*
# the regressors drop `classes`
```

- **One pass per call.** Each `partial_fit` runs a single epoch over its chunk in
  natural row order, continuing the persisted optimizer state, with no shuffle and no
  early stopping.
- **First call.** Classifiers require `classes=` (all labels) on the first call
  (sklearn convention); binary-vs-multiclass is frozen then. The FFM `field_ids` map
  is set on the first call and validated (or reused) thereafter.
  `class_weight="balanced"` is not supported by `partial_fit` (it cannot be computed
  from a stream).
- **Exactness contract.** N sequential `partial_fit` calls over consecutive chunks
  equal one `partial_fit` over the concatenation, bit-for-bit, given `dtype="float64"`,
  `n_jobs=1`, and `batch_size=1` (or chunk lengths that are multiples of
  `batch_size`). `dtype="float32"` truncates parameters between calls and `n_jobs>1`
  reorders float sums, so both relax bit-exactness.
- `n_iter_` accumulates the number of passes across calls.

`warm_start=True` makes `fit` resume from the current `w0_`/`w_`/`V_` (and the
persisted optimizer state) instead of re-initializing, then run `max_iter` more epochs
(honoring `early_stopping`); `warm_start=False` is a fresh fit. `save_model` /
`load_model` does **not** persist the streamed optimizer state (`pickle` does);
resuming after `load_model` restarts the optimizer accumulators from the loaded
parameters.

## Learned attributes (after fit)

- `w0_` (float), `w_` (n_features,), `V_`
  - FM: `V_` shape `(n_features, n_factors)`
  - FFM: `V_` shape `(n_features, n_fields, n_factors)`
  - FwFM: `V_` shape `(n_features, n_factors)` plus `r_` shape
    `(n_fields, n_fields)` (upper triangle used)
- `classes_` (classifiers), `n_features_in_`, `n_iter_`
- FFM / FwFM: `field_ids_`, `n_fields_`
- multiclass (one parameter set per class): `w0_` shape `(n_classes,)`, `w_` shape
  `(n_classes, n_features)`; FM `V_` shape `(n_classes, n_features, n_factors)`,
  FFM `V_` shape `(n_classes, n_features, n_fields, n_factors)`,
  FwFM `V_` shape `(n_classes, n_features, n_factors)` + `r_` shape
  `(n_classes, n_fields, n_fields)`

## Errors and validation

- shape/dtype validation at `fit`/`predict`, raising `ValueError` with clear messages
- `predict` before `fit` raises `NotFittedError` (sklearn's)
- unknown optimizer/loss strings raise `ValueError` at `fit` time (not `__init__`)
