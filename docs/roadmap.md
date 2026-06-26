# Roadmap

## v0.1 — Rust CPU core (current)

- [x] Phase 0: design docs + package skeleton
- [x] Phase 1: Python reference predictions (FM naive/fast, FFM naive/vectorized),
      losses (logistic/softmax + label smoothing), correctness tests
- [x] Phase 2A: Rust prediction backend (completed 2026-06-11)
  - toolchain updated (rustc 1.96), pyproject switched to maturin mixed layout
    (`python-source = "python"`, module `modern_fm._rust`, pyo3 0.25 abi3-py310)
  - `rust/src/{lib,data,fm,ffm}.rs`: FM fast + FFM predictions, dense and CSR,
    float64, GIL released during compute; input validation (CSR structure,
    field_ids range, shape mismatches -> ValueError)
  - `modern_fm._backend`: private dispatch — Rust when built, NumPy reference
    fallback otherwise; handles dtype/contiguity coercion
  - parity tests (`tests/test_rust_parity.py`): Rust vs reference at
    atol/rtol 1e-12, dense + CSR, multiple seeds, zero rows, single nonzero,
    hand-computed examples, bad-input rejection; Rust unit tests in-crate
- [x] Phase 2B: Rust training — SGD + AdaGrad for FM and FFM (NumPy reference
      trainer as ground truth, parity-tested), estimators wired to the backend
      (binary + regression), seeded reproducibility. `rayon` row-parallelism and
      mini-batch deferred to v0.2 (n_jobs=1, batch_size=1 in v0.1).
- [x] Phase 3: sklearn API polish (lightweight mixins, check_is_fitted,
      fit/predict validation) + `CategoricalEncoder`. Full sklearn
      `check_estimator` compliance deferred to v0.2 (no sklearn runtime dep).
- [x] Phase 4: early_stopping/eval_set, label_smoothing, class_weight,
      sample_weight, multiclass softmax (FM), save/load, examples + benchmark.

## v0.2 — Training quality & throughput

- [x] Adam optimizer (`optimizer="adam"`, per-parameter lazy Adam with
  `beta_1`/`beta_2`/`epsilon`); FM binary/multiclass + FFM, parity-tested vs the
  NumPy reference. Adam + early stopping is deferred (moments are not round-tripped).
- [x] FTRL-Proximal optimizer (`optimizer="ftrl"`, `l1_linear`/`l1_factors`/
  `ftrl_beta`): per-coordinate `(z, n)` state with L1/L2 folded into the update;
  FM binary/multiclass + FFM, parity-tested vs the NumPy reference; L1 yields exact
  zeros (composes with mini-batch + n_jobs; FTRL + early stopping deferred)
- [x] Rust multiclass-softmax training kernel (`fm_fit_multiclass_csr`),
  parity-tested vs the NumPy reference (done ahead of v0.2)
- [x] mini-batch (`batch_size > 1`): per-batch gradient averaging with one update
  per touched coordinate (FM binary/multiclass + FFM), parity-tested vs the NumPy
  reference at batch_size ∈ {1, 4, full}; batch_size=1 stays the per-row path
- [x] `rayon` row-parallelism (`n_jobs > 1`): deterministic parallel-accumulate /
  serial-apply per batch (FM binary + FFM; multiclass serial); `n_jobs=1` matches
  the reference, `n_jobs>1` reproducible per thread count. ~3x on 4 cores for FFM.
- [x] full sklearn `check_estimator` compatibility (estimators subclass sklearn
  `BaseEstimator` + `Classifier`/`RegressorMixin`; scikit-learn is now a runtime
  dependency). `FFMClassifier.fit(X, y)` defaults `field_ids` to per-column.
- `partial_fit`, `warm_start=True`
- pairwise dropout, interaction pruning
- calibration helper
- [x] libffm format loader/exporter (`load_libffm` / `dump_libffm`, round-trip tested)
- model inspection: top interactions
- [x] pandas/polars input (DataFrames via sklearn `validate_data`;
  `feature_names_in_` recorded, column reorder rejected at predict)
- [x] CI + release pipeline: `.github/workflows/ci.yml` (pytest + ruff across
  {Linux, macOS, Windows} × py3.10–3.13, plus cargo test/clippy) and
  `release.yml` (abi3 wheels via maturin-action + sdist, PyPI trusted publishing
  on a `v*` tag). Verified locally: maturin builds an abi3 wheel + sdist that
  install and run in a clean venv.
- [x] Adam + early stopping: moments round-tripped across epochs via the NumPy
  reference path (`fm_fit_reference`'s `adam_state`); FM binary/regression + FFM,
  per-epoch hand-off equals one multi-epoch call exactly.
- [x] multiclass + early stopping: per-class optimizer state (AdaGrad/Adam)
  round-tripped via the reference path; softmax cross-entropy eval metric,
  round-trip equals a single multi-epoch call.
- [x] FFM multiclass softmax (`ffm_fit_multiclass_csr`): one FFM per class coupled
  by softmax, all optimizers (SGD/AdaGrad/Adam/FTRL) + mini-batch, parity-tested
  vs the NumPy reference; `FFMClassifier` auto-detects >2 classes / `loss="softmax"`.
  FTRL + early stopping and multiclass FFM + early stopping remain (niche).

## v0.3+ — Model variants & GPU

- FwFM, AFM, FEFM/FmFM (each gets its own math spec before implementation)
- PyTorch-compatible backend prototype
- CUDA backend investigation (cuML-style `device=`/`backend=` switch)

## Distribution

- PyPI name: `modern-fm` (availability confirmed 2026-06-11)
- wheels via maturin + cibuildwheel once Rust backend lands
