# Roadmap

## v0.1 — Rust CPU core (current)

- [x] Phase 0: design docs + package skeleton
- [ ] Phase 1: Python reference predictions (FM naive/fast, FFM naive/vectorized),
      losses (logistic/softmax + label smoothing), correctness tests
- [ ] Phase 2: Rust CPU backend
  - toolchain: `rustup update` (cargo 1.64 is too old for current PyO3),
    install `maturin`, switch pyproject build-backend to maturin (mixed layout,
    package stays at `python/modern_fm`)
  - CSR input bridge (zero-copy where possible)
  - FM predict / fit (SGD, AdaGrad), FFM predict / fit
  - `rayon` parallelism, equivalence tests vs Python reference
- [ ] Phase 3: sklearn API polish (BaseEstimator/mixins, check_is_fitted, validation)
- [ ] Phase 4: early_stopping, label_smoothing, class_weight, sample_weight,
      save/load, docs
- [ ] Python reference SGD trainer (ground truth for Rust fit)

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
