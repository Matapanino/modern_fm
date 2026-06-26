# Changelog

All notable changes to `modern_fm` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Full scikit-learn `check_estimator` compatibility: `FMClassifier`,
  `FMRegressor`, and `FFMClassifier` subclass sklearn's `BaseEstimator` +
  `Classifier`/`RegressorMixin`, work in `Pipeline` / `GridSearchCV` / `clone`,
  and validate input via `validate_data` (pandas DataFrames carry
  `feature_names_in_`). `CategoricalEncoder` is now a `TransformerMixin`.
  `FFMClassifier.fit(X, y)` no longer requires `field_ids` (each column defaults
  to its own field). **scikit-learn (>=1.6) is now a runtime dependency.**
- pandas / polars DataFrame input: `fit`/`predict` accept DataFrames (columns
  taken in order, `feature_names_in_` recorded, column reorder rejected at
  predict), with ndarray-parity tests.
- libffm text-format I/O: `load_libffm` / `dump_libffm` for the
  `<label> field:feature:value ...` format, with round-trip tests.
- `benchmarks/bench_vs_baseline.py`: synthetic-CTR comparison (test AUC, fit
  time, predict throughput) vs scikit-learn `LogisticRegression`, with an
  `n_jobs` / `batch_size` sweep and a results table in the README.

## [0.2.1] - 2026-06-26

### Fixed
- Source distribution (sdist) now bundles the `LICENSE` file so it passes PyPI's
  `License-File` metadata validation. (The 0.2.0 sdist upload was rejected for
  this reason; the 0.2.0 wheels were unaffected and remain installable.)
- Release workflow publishes with `skip-existing` for idempotent re-runs.

## [0.2.0] - 2026-06-26

Training-quality & throughput release. The Rust backend remains parity-tested
against the pure-NumPy reference implementations in `python/modern_fm/`.

### Added
- **FTRL-Proximal optimizer** (`optimizer="ftrl"`) with L1/L2 folded into the
  update; `l1_linear` / `l1_factors` yield exact-zero weights. Covers FM
  binary/multiclass and FFM.
- **Mini-batch training** via `batch_size` (gradient averaging; `1` = per-row SGD).
- **Multi-core training** via `rayon` (`n_jobs`), using a deterministic
  parallel-accumulate / serial-apply scheme.
- **Adam + early stopping**: Adam moments round-trip across epochs through the
  NumPy reference path.
- **FFM multiclass softmax** (`FFMClassifier` with >2 classes or
  `loss="softmax"`), reaching feature parity with `FMClassifier`.
- Packaging & release infrastructure: MIT `LICENSE`; GitHub Actions CI
  (3 OS × CPython 3.10–3.13, plus `cargo test` / `clippy`); and `release.yml`,
  which builds abi3 wheels + sdist and publishes to PyPI via trusted
  publishing (OIDC) on a `v*` tag.

### Notes
- Remaining niche gaps (FTRL + early stopping; multiclass + early stopping for
  FFM) are tracked in `docs/roadmap.md`.

## [0.1.0]

- Initial FM/FFM estimators (`FMRegressor`, `FMClassifier`, `FFMClassifier`),
  SGD / AdaGrad optimizers, sparse-CSR Rust prediction & training kernels,
  `CategoricalEncoder`, `save_model` / `load_model`, and a scikit-learn-style API.
