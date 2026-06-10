# Test Plan

Tests are written before (or together with) the code they cover. Every
optimized path must be proven equal to a naive reference.

## API tests (`tests/test_api.py`)

- model can be initialized; `__init__` only stores parameters (no validation)
- `get_params` returns constructor parameters; `set_params` round-trips
- (Phase 3) `fit` returns self; predict shapes; `predict_proba` rows sum to 1;
  `NotFittedError` before fit

## Correctness tests

- FM naive O(n^2 k) prediction equals optimized O(nk) prediction
  (`tests/test_fm_correctness.py`)
- FFM prediction matches a hand-computed tiny example and the naive loop
  (`tests/test_ffm_correctness.py`)
- Dense input and CSR input produce identical predictions for FM and FFM
  (`tests/test_sparse_dense_equivalence.py`)
- Losses match hand-computed / scipy values; label smoothing formula;
  numerical stability at extreme logits (`tests/test_losses.py`)
- (Phase 2) logistic/softmax loss decreases during training on synthetic data
- (Phase 2) Rust predictions equal Python reference predictions

## Reproducibility tests (Phase 2+)

- same `random_state` gives same predictions
- different `random_state` can produce different initialization

## Serialization tests (Phase 4)

- `save_model`/`load_model` preserves predictions
- pickle roundtrip preserves predictions

## Edge cases

- all-zero rows (prediction = w0 + 0)
- single nonzero feature (pairwise term = 0 for FM and FFM)
- empty interaction sets, k=1, single field
- (Phase 2+) unseen categories in encoder, class imbalance, sample_weight
