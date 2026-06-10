# modern_fm

Fast, sklearn-compatible Factorization Machines (FM) and Field-aware
Factorization Machines (FFM) for Python.

**Status: pre-alpha (v0.1 Phase 1).** Pure-NumPy reference implementations and
the public API skeleton exist; the high-performance Rust backend lands in
Phase 2. `fit` is not implemented yet.

## Planned API

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

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/ruff check .
```

Design documents live in `docs/` — start with `docs/requirements.md` and
`docs/math_spec.md`. The roadmap is in `docs/roadmap.md`.
