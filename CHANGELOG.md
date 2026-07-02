# Changelog

All notable changes to `modern_fm` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [1.1.1] - 2026-07-02

### Fixed
- `modern_fm.__version__` (frozen public API) shipped as `"1.0.0"` in the
  1.1.0 wheels — the hardcoded string in `__init__.py` was missed by the
  release bump while the distribution metadata was correct. Now `"1.1.1"`,
  with a new test pinning `__version__` to
  `importlib.metadata.version("modern-fm")` so CI catches any future
  mismatch. No functional changes.

## [1.1.0] - 2026-07-02

Full CUDA coverage + CUDA-ready Linux wheels. Additive minor under
`docs/compat_policy.md` — no API changes; `backend="cuda"` simply stops
raising `NotImplementedError` anywhere.

### Added
- **CUDA multiclass (softmax) training for FM and FFM**
  (docs/gpu_backend_plan.md milestone 5, T4-validated): the GPU accumulates
  all C classes' batch gradients from the frozen batch-start parameters
  (stable softmax in CPU class order + label-smoothed targets computed
  in-kernel); the untouched CPU flush applies the optimizer per class via
  `McState::class_views`, so SGD/AdaGrad/Adam/FTRL, L1 exact zeros, early
  stopping and `partial_fit` state hand-offs are unchanged. FM stacks compact
  `(C, T)` gradient buffers over one shared slot map; FFM splits
  score/softmax from per-class pair accumulation so ONE class-sized dense
  slot-gradient buffer serves every class (no C× VRAM blowup). T4 quick
  bench: full-batch 4.0× (FM, C=3) / 5.3× (FFM, C=3) vs the serial CPU
  multiclass kernel.
- **FwFM CUDA — prediction, binary/regression and multiclass training**
  (milestone 6, T4-validated): prediction kernel with FFM geometry, FM-shaped
  V and the R field-pair weight; training kernels ride the FM compact
  feature-slot machinery (both pair endpoints are row nonzeros) with a dense
  `n_fields²` R-gradient buffer; the CPU flush including the R group
  (`GroupStateMut`/`McGroupState`) is reused verbatim. Every prediction and
  training cell (FM/FFM/FwFM × binary/regression/multiclass) is now
  CUDA-covered; the per-cell guards are gone.
- **CUDA-ready Linux wheels**: the published Linux wheels are built with the
  `cuda-backend` feature. cudarc's `dynamic-loading` is pinned, so nothing
  links libcuda/libnvrtc — the wheel stays manylinux-compliant, imports on
  CUDA-less machines exactly like a CPU build (`has_cuda()` is `False`), and
  `backend="cuda"` works wherever an NVIDIA driver (CUDA 12+) is present,
  e.g. Colab/Kaggle GPU runtimes, with no source build. New CI job
  (`cuda-wheel`) and release gate (`linux-wheel-check`: CPU-only
  import/tests/auditwheel) enforce this on every PR and every release.
  macOS/Windows wheels stay CPU-only.
- `benchmarks/bench_cuda.py`: FwFM prediction/training and FM/FFM multiclass
  training sections.

## [1.0.0] - 2026-07-02

First stable release. The public API is now frozen under the SemVer contract
in `docs/compat_policy.md`; every criterion of the roadmap's "v1.0 —
criteria" gate is met. Highlights of this release beyond v0.5.0: the CUDA
backend covers FM/FFM prediction **and** FM/FFM binary/regression training
(T4-validated, honest benchmarks recorded); calibrated probabilities and
`top_interactions` model inspection ship as production-CTR features; a real
click-data benchmark and the documentation site
(https://matapanino.github.io/modern_fm/) are live.

### Added
- **API freeze + backward-compatibility policy (v1.0 roadmap item)**:
  `docs/compat_policy.md` (SemVer contract: what is public, deprecation
  rules, numerical-reproducibility guarantees, dependency policy; in the
  docs-site nav). `load_model` now rejects files written by a newer
  modern_fm (`format_version` check) with a clear upgrade error;
  `CategoricalEncoder`/libffm I/O documented in api_design (the one gap an
  inspect-based parameter sweep found); a stale dead-code
  `NotImplementedError` claiming multiclass FM is unsupported was removed.
- **Documentation site (v1.0 roadmap item)**: mkdocs-material site over the
  existing design docs plus a new install/quickstart landing page
  (`docs/index.md`), auto-deployed to GitHub Pages by
  `.github/workflows/docs.yml` on doc changes; linked from the README
  (https://matapanino.github.io/modern_fm/).
- **Real-data CTR benchmark (v1.0 roadmap item)**:
  `benchmarks/bench_criteo_like.py` — the KDD Cup 2012 track-2 click sample
  from OpenML (zero-credential; the original Criteo/Avazu samples are gated
  now, documented in the script), 200k rows / 373k one-hot features / 9
  fields, fixed seed + stratified split + libFM-style fixed hyperparameters
  with built-in early stopping. README gains the results table (honest:
  FwFM 0.6891 ≈ LR 0.6908 on this singleton-heavy sample).
- **`top_interactions(n_top, class_idx=None)` (v1.0 roadmap "Model
  inspection")** on every estimator: the strongest learned pairwise
  interactions as `(i, j, strength)` tuples — `|<V_i, V_j>|` for FM, the
  r-weighted variant for FwFM, `|<V[i, f_j], V[j, f_i]>|` for FFM (exact
  blockwise upper-triangle scan, `python/modern_fm/_inspect.py`). Planted
  dominant-pair + blockwise-vs-naive tests; `examples/top_interactions.py`.
- **Probability calibration (v1.0 roadmap item)** via the recommended sklearn
  path: every public classifier works inside `CalibratedClassifierCV`
  (pinned by `tests/test_calibration.py`, including an ECE/Brier-improvement
  test on synthetically miscalibrated data); `examples/calibration.py` and an
  api_design "Probability calibration" section document the recipe. No
  library-specific calibration API — the estimators are sklearn-clean by
  design.
- **CUDA FFM training accumulation** (docs/gpu_backend_plan.md milestone 4):
  `FFMClassifier` (binary) and `FFMRegressor` now accept `backend="cuda"` at
  fit (`rust/src/cuda/ffm_train.rs`). V stays device-resident; per batch the
  GPU accumulates pair gradients into a dense slot buffer, a gather kernel
  packs (and re-zeroes) only the touched (feature, field) slots for the CPU
  optimizer flush, and a scatter kernel writes the flushed slots back — the
  host pays the touched-slot enumeration (the pair loop without the k-dot).
  All four optimizers, early stopping, `partial_fit`/`warm_start` and FTRL's
  exact L1 zeros ride through the unchanged CPU flush, like the FM path.
  Multiclass FFM and FwFM training still raise `NotImplementedError`; same
  nondeterminism and compute >= 6.0 caveats as CUDA FM training.
- **CUDA FM training accumulation** (docs/gpu_backend_plan.md milestone 3):
  `FMClassifier` (binary) and `FMRegressor` now accept `backend="cuda"` at
  fit — each mini-batch's data-gradient is accumulated on the GPU
  (`rust/src/cuda/fm_train.rs`; CSR/targets/row-order/parameters upload once
  per call and the parameters stay device-resident; per batch the transfers
  switch between compact touched-coordinate gradient buffers with
  touched-only parameter scatter-back, for small batches, and full dense
  buffers, for large ones) while the optimizer flush and **all** optimizer
  state stay on the CPU, so SGD/AdaGrad/Adam/FTRL, early stopping,
  `partial_fit` and `warm_start` ride through unchanged (FTRL's exact L1
  zeros included). Multiclass FM, FFM and FwFM training
  still raise `NotImplementedError`; there is never a silent fallback.
  Caveats: CUDA training is **nondeterministic run-to-run** (atomic gradient
  accumulation; the CPU backend keeps exact seeded reproducibility) and the
  CUDA backend now requires compute capability >= 6.0 (Pascal, 2016 — the
  shared module is compiled for `compute_60` because
  `atomicAdd(double*, double)` needs it). Parity is tolerance-based on final
  predictions (rtol 1e-7 / atol 1e-8, `tests/test_cuda_parity.py`).
- **CUDA FFM prediction** (docs/gpu_backend_plan.md milestone 2): an NVRTC
  kernel for FFM CSR prediction (`rust/src/cuda/ffm.rs`, one block per row,
  256 threads striding the O(z²) pair loop, no row-nnz or k limit). Usage
  matches FM: fit on `backend="rust_cpu"`, then `set_params(backend="cuda")`
  for inference on `FFMClassifier` (binary + multiclass) and `FFMRegressor`;
  FwFM prediction and all training still raise `NotImplementedError` under
  CUDA. Parity is tolerance-based (rtol/atol 1e-10, `tests/test_cuda_parity.py`)
  and validated on a real GPU per `docs/cuda_validation_runbook.md`.
- **CUDA context/module cache**: the device-0 context and the NVRTC-compiled
  module holding all prediction kernels are now created once per process and
  cached (`rust/src/cuda/mod.rs`); previously every predict call re-created
  the context and re-compiled the kernel. Only the first CUDA call pays
  initialization; calls stay transfer-inclusive (device-resident parameters
  are a later milestone). `benchmarks/bench_cuda.py` gained the FFM grid and
  a cold-start line separating first-call cost from steady-state timings.
  First recorded T4 numbers (`benchmarks/README.md`, full grid):
  transfer-inclusive prediction speedups of 1.4–14.1x (FM) and 1.9–14.5x
  (FFM, serial CPU baseline), one-time init ~0.4 s; FM training (1 epoch,
  100k×100k, nnz 32, k 8) 0.3x at batch 256 → 3.1x at full batch — per-batch
  parameter round-trips dominate small batches, as documented.

## [0.5.0] - 2026-07-02

### Added
- **CUDA FM prediction** (docs/gpu_backend_plan.md milestone 1): an NVRTC
  kernel for FM CSR prediction (`rust/src/cuda/fm.rs`, one block per row /
  one thread per factor, transfer-inclusive). Usage: fit on
  `backend="rust_cpu"`, then `set_params(backend="cuda")` for inference;
  FFM/FwFM prediction and all training still raise `NotImplementedError`
  under CUDA. Parity is tolerance-based (rtol/atol 1e-10,
  `tests/test_cuda_parity.py`, skipped without a GPU) and validated on a real
  GPU per `docs/cuda_validation_runbook.md`; `benchmarks/bench_cuda.py`
  reports transfer-inclusive CPU-vs-CUDA timings.
- **CUDA backend plumbing** (no kernels yet; docs/gpu_backend_plan.md): an
  optional `cuda-backend` Cargo feature (cudarc with runtime dynamic loading —
  builds need no CUDA toolkit; skipped on macOS), `_backend.has_cuda()`, and
  `backend="cuda"` accepted by every estimator with clear errors —
  `RuntimeError` without a CUDA build/driver/device, `NotImplementedError`
  while no CUDA kernels exist; never a silent CPU fallback. CI gained a
  `cuda-check` job compiling the feature on a GPU-less runner. CPU-only
  builds, wheels and imports are unchanged.
- **`BiInteractionPooling`** — bi-interaction pooling (He & Chua, SIGIR 2017)
  as an sklearn transformer: fits an FM and emits the k-dim second-order
  interaction vector `0.5 * ((Σᵢxᵢvᵢ)² − Σᵢ(xᵢvᵢ)²)` for downstream models
  (multiclass inner FMs pool per class, concatenated). Shipped as a feature
  transform because a linear head over it provably collapses to plain FM
  (identity pinned at 1e-12); the FM estimators also expose
  `bi_interaction(X)` directly. `check_estimator`-clean, Pipeline-tested.
- **`FwFMClassifier`** — Field-weighted Factorization Machine (Pan et al.,
  WWW 2018; `docs/math_spec_fwfm.md`): FM-shaped factors plus one learned
  scalar weight per field pair (`r_`, upper triangle used) scaling each
  pairwise interaction; `r_` initializes to ones so a fresh FwFM is exactly a
  plain FM (property-tested at 1e-12). Binary logistic + multiclass softmax,
  all four optimizers, mini-batch, early stopping / `eval_set` (bit-exact
  four-group state hand-off through the Rust kernel), `partial_fit` /
  `warm_start`, save/load + pickle, `check_estimator`-clean. Layered exactly
  like FM/FFM: NumPy reference (`fwfm_predict[_naive]`,
  `fwfm_fit[_multiclass]_reference`) → Rust kernel (`rust/src/fwfm.rs`) →
  `_backend` dispatch → estimator, with parity tests at each layer. Training
  is serial in v0.5 (`n_jobs` accepted, not used by FwFM).

### Changed
- **Rust early-stopping fast path**: every per-epoch optimizer-state hand-off —
  AdaGrad accumulators, Adam moments, FTRL `(z, n)`, and the per-class
  multiclass state — now round-trips through the Rust kernels (optional
  `state` / `adam_state` / `ftrl_state` arguments on the fit entry points).
  Previously, `early_stopping` / `eval_set` (and `partial_fit` / `warm_start`)
  with Adam, FTRL, or any multiclass model trained each epoch on the NumPy
  reference implementation. Results are unchanged — the epoch-driven loop is
  bit-identical to a single multi-epoch Rust call (new parity tests per
  optimizer × {FM, FFM} × {binary, multiclass}) — but ES fits are ~14–170x
  faster in the previously reference-bound cells (synthetic bench: FFM binary
  Adam ES 49.5 s → 0.86 s, FTRL 49.4 s → 0.29 s; FM multiclass AdaGrad ES
  3.08 s → 0.09 s). `benchmarks/bench_synthetic.py` gained
  `bench_early_stopping()`.

## [0.4.0] - 2026-07-02

### Added
- **`FFMRegressor`**: squared-loss Field-aware Factorization Machine, the
  regression counterpart to `FMRegressor` (`RegressorMixin`; `fit(X, y, field_ids=…)`
  and `predict`; SGD/AdaGrad/Adam/FTRL, mini-batch, `n_jobs`, early stopping). The
  FFM training kernel (Rust + NumPy reference) gained a `loss` parameter
  (`"logistic"` | `"squared"`); `check_estimator`-clean, `save_model`/`load_model`
  + pickle round-trip, exported in `__all__`. `FFMClassifier` and `FFMRegressor`
  now share a common `_FFMBase` (mirrors `_FMBase`).
- **FTRL + early stopping**: FTRL's per-coordinate `(z, n)` state now round-trips
  across epochs (a `ftrl_state` hand-off mirroring Adam's), so `early_stopping` /
  `eval_set` work with `optimizer="ftrl"` for FM (binary + multiclass) and FFM. The
  previous `NotImplementedError` guard is removed.
- **Multiclass FFM + early stopping**: per-class optimizer state (AdaGrad / Adam /
  FTRL) round-trips for multiclass `FFMClassifier`, evaluated with a softmax
  cross-entropy metric — `early_stopping` / `eval_set` now work for multiclass FFM.
- **`partial_fit` + `warm_start`**: incremental / streaming training for all four
  estimators (`partial_fit(X, y, classes=…)`, plus `field_ids=` for FFM). Each call
  runs one natural-order pass continuing the persisted optimizer state, so N chunked
  calls equal one pass over the concatenated data bit-for-bit (`dtype="float64"`,
  `n_jobs=1`, `batch_size` dividing the chunk lengths). `warm_start=True` makes `fit`
  resume from the previous solution + optimizer state. This closes the v0.4 milestone
  (no `NotImplementedError` left in the public surface).

## [0.3.0] - 2026-06-26

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
