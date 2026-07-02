# Backward-compatibility & versioning policy

`modern_fm` follows [Semantic Versioning](https://semver.org/). From v1.0.0
on, this document is the contract for what may change when.

## What is public API

The public surface — covered by SemVer — is exactly:

1. **Names exported in `modern_fm.__all__`**: `FMClassifier`, `FMRegressor`,
   `FFMClassifier`, `FFMRegressor`, `FwFMClassifier`, `BiInteractionPooling`,
   `CategoricalEncoder`, `NotFittedError`, `load_libffm`, `dump_libffm`,
   `__version__`.
2. **Their documented methods and constructor parameters** as specified in
   `docs/api_design.md` — `fit` / `predict` / `predict_proba` /
   `decision_function` / `partial_fit` / `top_interactions` /
   `save_model` / `load_model`, `get_params`/`set_params` round-tripping, and
   every constructor keyword listed there (names, defaults, accepted values).
3. **Learned attributes** listed in api_design ("Learned attributes"):
   `w0_`, `w_`, `V_`, `r_` (FwFM), `classes_`, `field_ids_`, `n_iter_`,
   `n_features_in_`, `feature_names_in_`, encoder attributes.
4. **Exception types** for documented failure modes (`ValueError` /
   `RuntimeError` / `NotImplementedError` / sklearn's `NotFittedError`), as
   specified in api_design "Errors and validation". Exception *messages* are
   not part of the contract.
5. **The `save_model` on-disk format**: a pickle of
   `{format_version, class, params, attrs}`. Files written by an older
   modern_fm stay loadable by all newer versions within the same major
   version (forward-readable); loading a file with a *newer*
   `format_version` than the library understands raises a clear
   `ValueError`. The format is pickle-based — only load files you trust.

Everything else is private, most notably: every module or name with a
leading underscore (`modern_fm._backend`, `_reference*`, `_inspect`,
`_partial`, ...), the Rust crate and its PyO3 functions
(`modern_fm._rust.*`), kernel/benchmark internals, and the exact text of
error messages, logs and reprs. Private surface can change in any release.

## What each release number may change

- **Patch (1.0.x)** — bug fixes only. No public-API signature changes; no
  behavior changes except where behavior contradicted the documentation
  (the docs win).
- **Minor (1.x.0)** — additive: new estimators, parameters (with defaults
  preserving old behavior), methods, backends. Deprecations may be
  *announced* here (see below). Numerical output may shift only within the
  documented tolerance model (below).
- **Major (x.0.0)** — removals and breaking changes, each listed in the
  CHANGELOG with a migration note.

## Deprecation policy

A public name or parameter is removed only after: (1) at least one minor
release where using it emits a `FutureWarning` naming the replacement, and
(2) a CHANGELOG entry at both the deprecation and the removal.

## Numerical reproducibility

- **Within one version**: identical results for identical inputs under a
  fixed `random_state` on the CPU backend (`backend="rust_cpu"`), per
  machine and thread count (`n_jobs>1` fixes the reduction order per thread
  count; `backend="cuda"` training is documented as nondeterministic
  run-to-run).
- **Across versions**: bit-identical results are *not* guaranteed by SemVer.
  What is guaranteed: the fast paths stay parity-tested against the frozen
  NumPy reference implementations (`python/modern_fm/_reference*.py`), which
  never change for speed — so cross-version drift stays within the parity
  tolerances documented in the tests.

## Dependency & platform support

- Python: the versions in `pyproject.toml` classifiers (currently
  3.10–3.13). Dropping a Python version is a minor release, announced in the
  CHANGELOG, never a patch.
- scikit-learn / NumPy / SciPy: minimum versions in `pyproject.toml`; raising
  a minimum is a minor release.
- Wheels: abi3 for Linux/macOS/Windows. The optional CUDA backend
  (`cuda-backend` Cargo feature) is not part of published wheels; it is a
  source-build feature and its cell coverage is documented in api_design —
  expanding it is additive, shrinking it is breaking.
