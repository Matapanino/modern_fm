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
Current state: v0.1 (Phases 0–4) is implemented and tested. Rust kernels in
`rust/` cover FM/FFM prediction and training — FM binary (logistic/squared) and
multiclass softmax, plus FFM logistic, all SGD/AdaGrad/Adam — dispatched through
the private `modern_fm._backend` module (NumPy reference fallback when the
extension is not built). Adam is per-parameter lazy Adam (`beta_1`/`beta_2`/
`epsilon`); it does not yet combine with early stopping. Remaining v0.2 work:
mini-batch, `rayon` parallelism, FTRL.

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
