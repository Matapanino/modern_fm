# Roadmap

## v0.1 — Rust CPU core (current)

- [x] Phase 0: design docs + package skeleton
- [ ] Phase 1: Python reference predictions (FM naive/fast, FFM naive/vectorized),
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
- [ ] Phase 2B: Rust training
  - SGD + AdaGrad for FM and FFM per docs/optimization_spec.md
    (gradients, init, epoch shuffling with seeded RNG)
  - Python reference SGD trainer first (ground truth for loss-decrease and
    parity-after-one-epoch tests)
  - wire estimators' `fit`/`predict` to the backend (binary + regression first)
  - reproducibility tests (same seed -> same predictions)
  - `rayon` row-parallelism can wait until correctness is locked (n_jobs=1 first)
- [ ] Phase 3: sklearn API polish (BaseEstimator/mixins, check_is_fitted, validation)
- [ ] Phase 4: early_stopping, label_smoothing, class_weight, sample_weight,
      save/load, docs

## v0.2 — Training quality & throughput

- FTRL optimizer, Adam
- `partial_fit`, `warm_start=True`
- pairwise dropout, interaction pruning
- calibration helper
- libffm format loader/exporter
- model inspection: top interactions
- pandas/polars input
- CI + cibuildwheel (Linux/macOS/Windows wheels)

## v0.3+ — Model variants & GPU

- FwFM, AFM, FEFM/FmFM (each gets its own math spec before implementation)
- PyTorch-compatible backend prototype
- CUDA backend investigation (cuML-style `device=`/`backend=` switch)

## Distribution

- PyPI name: `modern-fm` (availability confirmed 2026-06-11)
- wheels via maturin + cibuildwheel once Rust backend lands
