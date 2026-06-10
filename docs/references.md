# References

## Core papers

- Rendle, S. (2010). *Factorization Machines.* ICDM.
  Key formula: `y = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j`,
  with the O(nk) pairwise reformulation used by all fast implementations.
- Juan, Y., Zhuang, Y., Chin, W.-S., Lin, C.-J. (2016).
  *Field-aware Factorization Machines for CTR Prediction.* RecSys.
  Key formula: `interaction(i, j) = <v_{i, field_j}, v_{j, field_i}> x_i x_j`.

## Roadmap-model papers (v0.3+)

- AFM — Xiao et al., *Attentional Factorization Machines* (arXiv:1708.04617).
  Learns per-interaction importance via attention.
- FEFM — Pande, *Field-Embedded Factorization Machines* (arXiv:2009.09931).
  Field-pair matrix embeddings, lower complexity than FFM.
- FmFM — Sun et al., *FM^2: Field-matrixed Factorization Machines*
  (arXiv:2102.12994).
- FwFM — Pan et al., *Field-weighted Factorization Machines* (WWW 2018).

## Existing libraries

- libffm — canonical C++ FFM implementation (logistic loss, OpenMP/SSE):
  https://www.csie.ntu.edu.tw/~cjlin/libffm/
  Note: libffm omits w0 and the linear term; we keep both (see math_spec.md).
- xLearn — C++ LR/FM/FFM library: https://github.com/aksnzhy/xlearn
- fastFM — existing FM library (name collision avoided: we are `modern-fm`)
- DeepCTR-Torch — PyTorch CTR models including FM-family

## Design inspirations

- scikit-learn estimator API: https://scikit-learn.org/stable/developers/develop.html
- cuML sklearn-like GPU API: https://docs.rapids.ai/api/cuml/stable/
- PyO3 / maturin for Rust-Python binding: https://pyo3.rs/
- cibuildwheel for wheel distribution: https://cibuildwheel.pypa.io/
