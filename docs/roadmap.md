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
- [x] libffm format loader/exporter (`load_libffm` / `dump_libffm`, round-trip tested)
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
  FTRL + early stopping and multiclass FFM + early stopping are scheduled for v0.4 (see below).

## v0.4 — API completeness & online learning

Closes the (model × task × optimizer × early-stopping) matrix and adds
streaming / out-of-core training. No new model family, no release-infra work.

- [x] **FFMRegressor** — squared-loss FFM, the regression counterpart to
  `FMRegressor`. _Priority: P0._
  - DoD: `RegressorMixin` estimator; Rust kernel + NumPy-reference parity test
    (atol/rtol like existing FFM tests); dense+CSR equivalence; save/load +
    pickle round-trip; `check_estimator` passes; exported in `__all__`;
    `docs/api_design.md` + CHANGELOG updated. `_reference*.py` unchanged.
- [x] **FTRL + early stopping** — round-trip FTRL's per-coordinate `(z, n)`
  state across epochs (mirror Adam's `adam_state` hand-off). _Priority: P1._
  - DoD: remove the `_no_ftrl_early_stopping` guard (fm.py/ffm.py); per-epoch
    hand-off equals one multi-epoch call exactly (test); FM binary/multiclass +
    FFM; state carried via the reference path (no `_reference*.py` speed change).
- [x] **Multiclass FFM + early stopping** — per-class optimizer state
  round-tripped for FFM multiclass (mirror FM-multiclass + ES). _Priority: P1._
  - DoD: remove the multiclass guard in `ffm.py`; round-trip equals a single
    multi-epoch call (test); softmax cross-entropy eval metric.
- [x] **partial_fit + warm_start** — incremental/streaming training for FM & FFM.
  _Priority: P0._
  - DoD: `partial_fit(X, y, classes=...)` (sklearn first-call `classes`
    convention) and `warm_start=True` continue from existing params + optimizer
    state; `classes_` retained; N sequential `partial_fit` calls equal one `fit`
    on the concatenated data under a matched epoch/batch schedule (exact via
    optimizer-state round-trip); contract added to `docs/api_design.md`;
    reference parity preserved; CHANGELOG updated.

## v0.5 — Rust ES fast path, FwFM, pooling, CUDA plumbing

Performance completion of the early-stopping matrix, the v1.0 headline model
pulled forward, a research-honest `nfm_pooling`, and locally-testable CUDA
groundwork (kernels land separately, gated on real-GPU validation — see
`docs/gpu_backend_plan.md`).

- [x] **Rust early-stopping fast path** — every per-epoch optimizer-state
  hand-off (AdaGrad accumulators, Adam moments, FTRL `(z, n)`, per-class
  multiclass state) now round-trips through the Rust kernels via optional
  `state`/`adam_state`/`ftrl_state` PyO3 arguments; `_backend` dispatches to
  the reference only when the extension is missing. The epoch-driven ES loop
  is bit-identical to a single multi-epoch Rust call (tested per optimizer ×
  {FM, FFM} × {binary, multiclass}). ES fits sped up ~14–170x for the
  previously reference-bound cells (Adam/FTRL/multiclass; FFM+Adam ES
  49.5 s → 0.86 s on the synthetic bench); `partial_fit`/`warm_start` ride the
  same path. _Priority: P0._
- [x] **FwFM (`FwFMClassifier`)** — Field-weighted FM (moved up from v1.0; it
  remains the 1.0 headline). _Priority: P0._
  - DoD met: `docs/math_spec_fwfm.md` written FIRST (field-pair weights
    `r_{f(i),f(j)}` upper-triangle, exact prediction/gradients/updates, R=ones
    init = plain FM); NumPy reference → Rust kernel (`rust/src/fwfm.rs`) →
    `FwFMClassifier`, parity-tested at each layer (predict 1e-12, train
    RTOL=1e-9 × optimizer × loss × batch_size, multiclass, ES bit-exact
    hand-off); collapse-to-FM property test; existing FM/FFM formulas
    untouched; `check_estimator`, save/load, `partial_fit`/`warm_start`,
    `__all__` + api_design + CHANGELOG. Binary + multiclass; serial (rayon
    `n_jobs` for FwFM deferred). (AFM/FEFM/FmFM follow this template post-1.0;
    FmFM is the research-recommended next variant — one field-pair k×k matrix
    generalizes FM/FwFM/FvFM/FmFM.)
- [x] **Bi-interaction pooling (`BiInteractionPooling`)** — the honest
  "nfm_pooling": a sklearn transformer emitting the k-dim bi-interaction
  vector `0.5 * ((sum_i x_i v_i)^2 - sum_i (x_i v_i)^2)` from a fitted FM for
  downstream models. As a *predictor* a linear head over this provably
  collapses to plain FM (NFM = this + MLP, out of scope), so it ships as a
  feature transform, not a model. _Priority: P1._
  - DoD met: `_reference.fm_bi_interaction` (+ `_backend.fm_bi_interaction`
    wrapper; no Rust kernel — NumPy is two BLAS-grade sparse matmuls),
    `BiInteractionPooling` with `fit`/`transform`/`get_feature_names_out`
    (+ `bi_interaction(X)` on the FM estimators), collapse-to-FM identity
    test at 1e-12, Pipeline + `check_estimator` + pickle tests, api_design
    docs.
- [x] **CUDA plumbing (no kernels)** — `cuda-backend` Cargo feature (cudarc
  0.19, default `fallback-dynamic-loading` = no CUDA toolkit at build time;
  target-gated off on macOS), `rust/src/cuda/mod.rs` `available()`, always-
  registered `has_cuda()` pyfunction, `_backend.has_cuda()`,
  `backend="cuda"` accepted at fit with clear errors (RuntimeError without a
  CUDA build/device, NotImplementedError while no kernels exist — never a
  silent CPU fallback), CI `cuda-check` job (`cargo check/clippy --features
  cuda-backend`, in the `ci-success` gate). Kernels (FM CSR prediction
  first) follow in a separate PR that merges only after validation on a real
  GPU (runbook in the PR). _Priority: P2._

## v0.6 — in progress

- [x] **CUDA FFM prediction + context/module cache** (gpu_backend_plan
  milestone 2, pulled ahead of the post-1.0 GPU track): FFM CSR prediction
  kernel (`rust/src/cuda/ffm.rs`, one block/row, 256-thread pair-strided
  loop, no row-nnz/k limit; `FFMClassifier` binary+multiclass and
  `FFMRegressor` inference via `set_params(backend="cuda")`), plus a
  process-wide cache of the CUDA context + NVRTC module
  (`rust/src/cuda/mod.rs`) so only the first call pays initialization.
  Parity rtol/atol 1e-10, T4-validated per `docs/cuda_validation_runbook.md`;
  `bench_cuda.py` gained the FFM grid + a cold-start line. FwFM-CUDA and
  device-resident parameters remain out of scope.
- [x] **CUDA FM training accumulation** (gpu_backend_plan milestone 3):
  binary/regression FM fit with `backend="cuda"` — GPU accumulates each
  mini-batch's data-gradient (dense buffers, `atomicAdd`, CSR uploaded once
  per call), the untouched CPU flush applies SGD/AdaGrad/Adam/FTRL, so early
  stopping, `partial_fit`, `warm_start` and FTRL's exact L1 zeros work
  unchanged. Multiclass/FFM/FwFM training still raise. Nondeterministic
  run-to-run (atomics); parity on final predictions at rtol 1e-7/atol 1e-8;
  requires compute >= 6.0. Sparse gradient buffers + device-resident params
  are the follow-up before claiming training speed.

## v1.0 — stable release

Headline model variant + production-CTR features + docs/bench polish + API
freeze. Shipping this milestone = tagging v1.0.0.

- [ ] **FwFM** — pulled into v0.5 (see above); the v1.0 gate keeps its DoD.
- [ ] **Probability calibration** — calibrated `predict_proba` for CTR.
  _Priority: P1._
  - DoD: sklearn `CalibratedClassifierCV`-compatible (recommended) or built-in
    Platt/isotonic; ECE/reliability test shows improvement on synthetic
    miscalibrated data; example + docs.
- [ ] **Model inspection (top interactions)** — strongest learned pairwise
  interactions. _Priority: P1._
  - DoD: API returning top-`|<v_i, v_j>|` feature pairs for a fitted FM/FFM
    (e.g. `top_interactions(k)`); tested on a synthetic model with a known
    dominant pair; docs + example.
- [ ] **Real-data benchmark** — Criteo/Avazu *sample* (not full). _Priority: P1._
  - DoD: `benchmarks/bench_criteo_like.py` reporting test AUC + fit time on a
    small vendored/downloaded sample; README results table; fixed seeds + machine
    specs (benchmark_plan rules); no tuning-to-benchmark.
- [ ] **Documentation site** — published API/usage docs. _Priority: P1._
  - DoD: mkdocs-material (recommended) or Sphinx — install, quickstart, API
    reference, math specs, examples; GitHub Pages auto-deploy via CI; linked from
    README.
- [ ] **API freeze + backward-compat policy** — _Priority: P0._
  - DoD: audit public `__all__` (estimators, encoder, libffm I/O, errors); every
    public estimator + constructor param documented in `docs/api_design.md`;
    write a SemVer / backward-compatibility policy (`docs/compat_policy.md`);
    `save_model` format carries a version tag and stays forward-readable; no
    undocumented `NotImplementedError` in the public surface.
- [ ] **Release 1.0.0** — _Priority: P0._
  - DoD: bump version to `1.0.0` (`__init__.py`, `pyproject.toml`, `Cargo.toml`);
    CHANGELOG `1.0.0` entry; full CI matrix green (3 OS × py3.10–3.13 + cargo
    test/clippy); tag `v1.0.0` → trusted-publishing `release.yml`.

## v1.0 — criteria

The release is "stable" only when all of these hold (the global gate; the
per-item DoDs above are the local checks):

1. **Feature-matrix completeness** — every documented cell of
   (model {FM, FFM} × task {regression, binary, multiclass} × optimizer
   {sgd, adagrad, adam, ftrl} × early-stopping) works, or is *intentionally and
   visibly documented* as out-of-scope. No surprise `NotImplementedError` in the
   public surface (FFMRegressor, FTRL+ES, multiclass-FFM+ES all closed).
2. **Reference parity** — every fast/Rust path proven equal to the NumPy
   reference; `_reference*.py` never changed for speed (CLAUDE.md rule, now a
   release gate).
3. **Numerical stability** — no inf/nan at extreme logits (logsumexp/log1p),
   tested.
4. **Reproducibility** — identical results under a fixed `random_state` across
   the supported matrix.
5. **Serialization stability** — `save_model`/`load_model` + pickle round-trip
   preserve predictions; on-disk format carries a version tag.
6. **sklearn compatibility** — `check_estimator` passes for every public
   estimator; works in `Pipeline` / `GridSearchCV` / `clone`.
7. **API frozen & documented** — `__all__` audited; every public param in
   `docs/api_design.md`; a written backward-compatibility / SemVer policy.
8. **Quality gates green** — `pytest` green + `ruff` clean across the CI matrix
   (3 OS × py3.10–3.13); `cargo test` + `cargo clippy` warning-free.
9. **Docs site live** — GitHub Pages: install, quickstart, API reference, math
   specs, examples.
10. **Real-data evidence** — Criteo/Avazu-sample AUC + timing in the README.
11. **Production CTR features** — calibrated `predict_proba` + top-interaction
    inspection shipped.
12. **Released** — version `1.0.0`, CHANGELOG complete, `v1.0.0` tag published.

## Post-1.0 — model variants & GPU

- AFM, FEFM/FmFM (each gets its own math spec first, per the FwFM template)
- pairwise dropout, interaction pruning
- PyTorch-compatible backend prototype
- CUDA backend investigation (cuML-style `device=`/`backend=` switch)

## Distribution

- PyPI name: `modern-fm` (availability confirmed 2026-06-11)
- wheels via maturin + cibuildwheel once Rust backend lands
