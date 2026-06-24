# modern_fm

Fast, sklearn-compatible Factorization Machines (FM) and Field-aware
Factorization Machines (FFM) for Python.

**Status: v0.1 in development.** Implemented: pure-NumPy reference
implementations, a Rust CPU backend for prediction and SGD/AdaGrad training
(parity-tested against the reference), and sklearn-style estimators —
`FMClassifier` (binary + multiclass softmax), `FMRegressor`, and
`FFMClassifier` (binary) — with `sample_weight`/`class_weight`,
`label_smoothing`, early stopping, a `CategoricalEncoder`, and
`save_model`/`load_model`. Multiclass training currently runs on the NumPy
reference path (a Rust multiclass kernel, mini-batch, and threaded training are
v0.2). See `docs/roadmap.md`.

## Usage

```python
from modern_fm import FMClassifier, FFMClassifier

model = FMClassifier(
    n_factors=16,
    optimizer="adagrad",
    learning_rate=0.05,
    max_iter=100,
    l2_linear=1e-5,
    l2_factors=1e-5,
    random_state=42,
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)

ffm = FFMClassifier(n_factors=8, random_state=42)
ffm.fit(X_train, y_train, field_ids=field_ids)
```

`FMRegressor`, multiclass `FMClassifier` (just pass a target with >2 classes),
early stopping (`early_stopping=True` or `eval_set=(X_val, y_val)`), and the
`CategoricalEncoder` are demonstrated in `examples/basic_usage.py`.
`benchmarks/bench_synthetic.py` reports fit time and predict throughput against
the NumPy reference floor.

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
