# modern_fm

[![PyPI](https://img.shields.io/pypi/v/modern-fm.svg)](https://pypi.org/project/modern-fm/)
[![Python versions](https://img.shields.io/pypi/pyversions/modern-fm.svg)](https://pypi.org/project/modern-fm/)
[![CI](https://github.com/Matapanino/modern_fm/actions/workflows/ci.yml/badge.svg)](https://github.com/Matapanino/modern_fm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Fast, sklearn-compatible Factorization Machines (FM) and Field-aware
Factorization Machines (FFM) for Python.

**Status: v0.2 (Beta).** A Rust CPU backend (parity-tested against pure-NumPy
reference implementations) drives sklearn-style estimators — `FMClassifier`
(binary + multiclass softmax), `FMRegressor`, and `FFMClassifier`
(binary + multiclass softmax) — with the SGD / AdaGrad / Adam /
**FTRL-Proximal** optimizers, **mini-batch**
gradient averaging (`batch_size`), **multi-core training** via `rayon`
(`n_jobs`), plus `sample_weight`/`class_weight`, `label_smoothing`, early
stopping, a `CategoricalEncoder`, and `save_model`/`load_model`. FTRL's L1
(`l1_linear`/`l1_factors`) yields exact-zero weights. See `docs/roadmap.md` for
remaining niche gaps (FTRL + early stopping, multiclass + early-stopping for FFM).

## Installation

```bash
pip install modern-fm        # prebuilt wheels for Linux/macOS/Windows, no Rust toolchain needed
```

To build from source instead (e.g. on a platform without a prebuilt wheel), see
Development below; it requires a Rust toolchain.

## Usage

```python
from modern_fm import FMClassifier, FFMClassifier

model = FMClassifier(
    n_factors=16,
    optimizer="adagrad",
    learning_rate=0.05,
    max_iter=100,
    batch_size=256,        # mini-batch gradient averaging (1 = per-row SGD)
    n_jobs=-1,             # train batches across all CPU cores
    l2_linear=1e-5,
    l2_factors=1e-5,
    random_state=42,
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)

# FTRL-Proximal with L1 for sparse linear weights (classic CTR setup)
sparse = FMClassifier(optimizer="ftrl", l1_linear=1.0, batch_size=256, random_state=42)
sparse.fit(X_train, y_train)

ffm = FFMClassifier(n_factors=8, n_jobs=-1, random_state=42)
ffm.fit(X_train, y_train, field_ids=field_ids)
```

`FMRegressor`, multiclass `FMClassifier` (just pass a target with >2 classes),
early stopping (`early_stopping=True` or `eval_set=(X_val, y_val)`), and the
`CategoricalEncoder` are demonstrated in `examples/basic_usage.py`.
`benchmarks/bench_synthetic.py` reports fit time and predict throughput against
the NumPy reference floor.

## Benchmarks

On synthetic CTR data (40k train / 20k test; 16 one-hot categorical fields →
256 features) with *planted pairwise interactions* between field pairs — signal
a linear model cannot represent — FM/FFM recover most of it. `n_jobs=-1` uses all
cores (8 here); absolute numbers vary by machine.

| Model | Test AUC | Fit (s) | Predict (rows/s) |
|---|---:|---:|---:|
| `LogisticRegression` (sklearn) | 0.694 | 0.01 | 60M |
| `FMClassifier` (batch=1) | 0.817 | 1.34 | 4.3M |
| `FMClassifier` (batch=512) | 0.816 | 0.45 | 4.8M |
| `FMClassifier` (batch=512, `n_jobs=-1`) | 0.816 | 0.33 | 5.0M |
| `FFMClassifier` (batch=512) | 0.846 | 1.68 | 2.3M |
| `FFMClassifier` (batch=512, `n_jobs=-1`) | 0.846 | 1.46 | 2.1M |

- **Interactions matter**: AUC climbs 0.69 → 0.82 (FM) → 0.85 (FFM) as the model
  captures the pairwise / field-aware structure the linear baseline misses.
- **Mini-batch**: `batch_size=512` trains ~3× faster than per-row SGD at equal AUC.
- **Multi-core**: `n_jobs=-1` adds a further ~1.2–1.4× here (more on larger/denser data).

Reproduce with `python benchmarks/bench_vs_baseline.py`. `xlearn` is auto-included
if importable, but it does not build on every platform (it failed to build here on
macOS/arm64 + CPython 3.11).

## Development

Requires Python >= 3.10 and a recent Rust toolchain (1.74+; `rustup update`).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"   # builds the Rust extension via maturin
.venv/bin/pytest -q
.venv/bin/ruff check .
```

`pip install -e .` compiles `rust/` and installs the extension as
`modern_fm._rust` (maturin mixed layout, config in `pyproject.toml`).
After editing Rust code, re-run `pip install -e .` to rebuild. Rust-only
checks:

```bash
cd rust
PYO3_PYTHON=$PWD/../.venv/bin/python3 cargo test
PYO3_PYTHON=$PWD/../.venv/bin/python3 cargo clippy
```

Without the extension built, the package still works: `modern_fm._backend`
falls back to the pure-NumPy reference implementations, and the parity tests
in `tests/test_rust_parity.py` are skipped.

Design documents live in `docs/` — start with `docs/requirements.md` and
`docs/math_spec.md`. The roadmap is in `docs/roadmap.md`.
