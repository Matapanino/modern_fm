# Data Format

## Accepted inputs (v0.1)

| Input | FM | FFM | Notes |
|---|---|---|---|
| `numpy.ndarray` (2D, float32/float64) | yes | yes | C-contiguous preferred |
| `scipy.sparse.csr_matrix` / `csr_array` | yes | yes | canonical format; others converted with a warning |
| integer-encoded categoricals | via helper encoder | via helper encoder | one-hot → CSR |

Labels `y`: 1D array. Regression: float. Binary: {0, 1} (anything 2-class is
mapped via `classes_`). Multiclass: arbitrary labels mapped via `classes_`.

## field_ids (FFM)

- `field_ids`: integer array of shape `(n_features,)`; `field_ids[i]` is the
  field of column `i`.
- Fields must be `0..n_fields-1` contiguous; validation error otherwise.
- Typical construction: one-hot encode each raw categorical column → all
  resulting columns share that raw column's field id. The helper encoder
  (`preprocessing.py`, Phase 3) produces `(X_csr, field_ids)` pairs.

## Internal conventions

- Rust backend consumes CSR triples `(indptr, indices, data)` + shape;
  indices `i64`, data `f32`/`f64` matching the `dtype` parameter.
- Dense input is processed densely (no silent densify/sparsify conversions).
- v0.2+: libffm text format (`label field:feature:value ...`) loader/exporter.
