# Project Goal

Build `modern_fm`: a high-performance Factorization Machine (FM) / Field-aware
Factorization Machine (FFM) library for Python, aimed at Kaggle-style tabular
data and CTR-like sparse problems.

The Python API must feel like scikit-learn:

- `fit(X, y)` returns `self`
- `predict(X)`
- `predict_proba(X)` (classifiers)
- `decision_function(X)` (classifiers)
- `get_params()` / `set_params()`
- `save_model(path)` / `load_model(path)`

Primary backend: Rust CPU via PyO3/maturin, efficient on sparse CSR.
The pure-NumPy **reference implementations** in `python/modern_fm/_reference.py`
are the ground truth for all backends — never change them for speed.
Current state: v0.4.0 (API completeness & online learning) is the released
version; the v0.5 milestone (Rust early-stopping fast path, FwFM,
bi-interaction pooling, CUDA backend plumbing) is in progress — see
`docs/roadmap.md` and `docs/gpu_backend_plan.md`. Rust kernels in `rust/` cover
FM/FFM prediction and training — FM binary (logistic/squared) + multiclass softmax
and FFM binary (logistic/squared) + multiclass softmax, optimizers
SGD/AdaGrad/Adam/FTRL, mini-batch (`batch_size`) and `rayon` row-parallelism
(`n_jobs`) — dispatched through the private `modern_fm._backend` module (NumPy
reference fallback when the extension is not built). Early stopping now works for
every optimizer (incl. FTRL) across binary and multiclass FM/FFM; the
Adam/FTRL/multiclass optimizer-state hand-offs round-trip through the NumPy
reference path. `partial_fit` (sklearn first-call `classes`) and `warm_start` add
incremental/streaming training for all four estimators, with an exact
optimizer-state round-trip (`python/modern_fm/_partial.py`). Estimators:
`FMRegressor`, `FMClassifier`, `FFMClassifier`, `FFMRegressor`; scikit-learn is a
runtime dependency and `check_estimator`-clean. The path to a stable release — the
remaining v1.0 work (FwFM, calibration, top-interactions, docs site, real-data
benchmark, API freeze) and the `## v1.0 — criteria` gate — is fixed in
`docs/roadmap.md`; consult it before starting new work.

## Target models

- v0.1: `FMRegressor`, `FMClassifier`, `FFMClassifier`
- Later: `FFMRegressor`, `FwFMClassifier`, `AFMClassifier` (see `docs/roadmap.md`)

## Non-goals for v0.1

- Deep-learning CTR stacks (DeepFM, xDeepFM, ...)
- Distributed training
- GPU backend (design for it, do not implement it)

## Design priorities (in order)

1. Correctness
2. sklearn-like usability
3. Sparse-data speed
4. Reproducibility
5. Extensibility for GPU/backend variants
6. Good tests before extra features

## Authoritative specs — read before writing code

- `docs/math_spec.md` — the exact FM/FFM/loss formulas. Do NOT substitute
  other FM variants (DeepFM, AFM, FwFM, FEFM) for these.
- `docs/api_design.md` — constructor signatures and fit/predict contracts.
- `docs/requirements.md` — v0.1 feature scope.
- `docs/test_plan.md` — required tests; write tests before implementation.

## Coding rules

- scikit-learn conventions: `__init__` only stores parameters (no validation,
  no computation); all learned attributes end with `_`; `get_params`/`set_params`
  must round-trip.
- Deterministic behavior under a fixed `random_state`.
- Validate shapes and dtypes at `fit`/`predict` boundaries, not in `__init__`.
- Avoid silent data copying when possible.
- Every fast/optimized code path must have a test proving equivalence with the
  naive reference implementation (dense vs sparse, naive vs optimized).
- Numerically stable losses (logsumexp / log1p), no inf/nan on large logits.
- Include benchmark scripts but do not overfit to benchmarks.

## Repository layout

- `python/modern_fm/` — the Python package (maturin mixed layout)
- `rust/` — Rust backend crate (built as `modern_fm._rust`)
- `tests/` — pytest suite
- `docs/` — design documents (the source of truth)
- `benchmarks/`, `examples/` — non-library code

## Commands

- Tests: `.venv/bin/pytest -q`
- Lint: `.venv/bin/ruff check .`
- Rebuild Rust extension after editing `rust/`: `.venv/bin/pip install -e .`
- Rust checks: `cd rust && PYO3_PYTHON=$PWD/../.venv/bin/python3 cargo test`
  (same for `cargo clippy`; both must be warning-free)
