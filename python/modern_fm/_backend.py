"""Backend dispatch: Rust extension when built, NumPy reference otherwise.

Private module. The NumPy implementations in `_reference` remain the ground
truth; the Rust extension (`modern_fm._rust`, built via maturin) is an
optimized drop-in whose parity is enforced by tests/test_rust_parity.py.

Both prediction and training are dispatched here (FM/FFM predict, FM binary and
multiclass-softmax training, FFM training); training parity with the reference
trainers is enforced by tests/test_rust_train_parity.py.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import _reference, _reference_train

try:
    from . import _rust
except ImportError:  # extension not built — pure-Python install
    _rust = None


def has_rust():
    return _rust is not None


def has_cuda():
    """True when the extension was built with the `cuda-backend` feature AND a
    CUDA driver + device are present at runtime (docs/gpu_backend_plan.md).
    The getattr guard keeps older prebuilt extensions working."""
    return _rust is not None and getattr(_rust, "has_cuda", lambda: False)()


def _prep_dense(X):
    return np.ascontiguousarray(X, dtype=np.float64)


def _prep_vec(a, dtype=np.float64):
    return np.ascontiguousarray(a, dtype=dtype)


def _prep_csr(X):
    X = X.tocsr().astype(np.float64)
    X.sum_duplicates()
    return (
        np.ascontiguousarray(X.indptr, dtype=np.int64),
        np.ascontiguousarray(X.indices, dtype=np.int64),
        np.ascontiguousarray(X.data, dtype=np.float64),
        X.shape[1],
    )


def fm_predict_fast(X, w0, w, V, backend="rust_cpu"):
    """FM prediction (math_spec.md), Rust-accelerated when available.

    `backend="cuda"` runs the CUDA CSR kernel (dense X is converted;
    transfer-inclusive, tolerance-based parity vs the CPU paths — see
    docs/gpu_backend_plan.md); it requires `has_cuda()` and never falls back
    silently."""
    if backend == "cuda":
        if not has_cuda():
            raise RuntimeError(
                "backend='cuda' requires modern_fm built with the `cuda-backend` "
                "Cargo feature and an NVIDIA GPU + driver at runtime"
            )
        w = _prep_vec(w)
        V = _prep_dense(V)
        Xc = X if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float64))
        indptr, indices, data, n_features = _prep_csr(Xc)
        return _rust.fm_predict_cuda_csr(indptr, indices, data, n_features, float(w0), w, V)
    if _rust is None:
        return _reference.fm_predict_fast(X, w0, w, V)
    w = _prep_vec(w)
    V = _prep_dense(V)
    if sp.issparse(X):
        indptr, indices, data, n_features = _prep_csr(X)
        return _rust.fm_predict_fast_csr(indptr, indices, data, n_features, float(w0), w, V)
    return _rust.fm_predict_fast_dense(_prep_dense(X), float(w0), w, V)


def ffm_predict(X, field_ids, w0, w, V, backend="rust_cpu"):
    """FFM prediction (math_spec.md), Rust-accelerated when available.

    `backend="cuda"` runs the CUDA CSR kernel (dense X is converted;
    transfer-inclusive per call, context/module process-cached — see
    docs/gpu_backend_plan.md); it requires `has_cuda()` and never falls back
    silently."""
    if backend == "cuda":
        if not has_cuda():
            raise RuntimeError(
                "backend='cuda' requires modern_fm built with the `cuda-backend` "
                "Cargo feature and an NVIDIA GPU + driver at runtime"
            )
        field_ids = _prep_vec(field_ids, dtype=np.int64)
        w = _prep_vec(w)
        V = _prep_dense(V)
        Xc = X if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float64))
        indptr, indices, data, n_features = _prep_csr(Xc)
        return _rust.ffm_predict_cuda_csr(
            indptr, indices, data, n_features, field_ids, float(w0), w, V
        )
    if _rust is None:
        return _reference.ffm_predict(X, field_ids, w0, w, V)
    field_ids = _prep_vec(field_ids, dtype=np.int64)
    w = _prep_vec(w)
    V = _prep_dense(V)
    if sp.issparse(X):
        indptr, indices, data, n_features = _prep_csr(X)
        return _rust.ffm_predict_csr(
            indptr, indices, data, n_features, field_ids, float(w0), w, V
        )
    return _rust.ffm_predict_dense(_prep_dense(X), field_ids, float(w0), w, V)


def _prep_fit(X, y, params, row_orders):
    """Common coercion for the Rust fit entry points.

    Dense X is converted to CSR (exact zeros are skipped either way, matching
    the reference). Returns fresh float64 copies of w and V that the Rust
    kernel mutates in place; the caller's `params` are left untouched.
    """
    w0, w, V = params
    w = np.array(w, dtype=np.float64, order="C", copy=True)
    V = np.array(V, dtype=np.float64, order="C", copy=True)
    y = _prep_vec(y)
    row_orders = np.ascontiguousarray(row_orders, dtype=np.int64)
    if row_orders.ndim == 1:
        row_orders = row_orders[None, :]
    Xc = X if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float64))
    return _prep_csr(Xc), y, float(w0), w, V, row_orders


def _acc_arrays(state, w, V):
    """AdaGrad accumulators (acc_w0, acc_w, acc_v) from `state` or fresh zeros.

    `state` (a mutable list, for the epoch-driven early-stopping path) persists
    the accumulators across calls; None means a single all-epochs run.
    """
    if state is None:
        return 0.0, np.zeros(len(w)), np.zeros_like(V)
    acc_w0, acc_w, acc_v = state
    return float(acc_w0), _prep_vec(acc_w), _prep_dense(acc_v)


def _adam_arrays(adam_state):
    """Binary Adam moments prepped for the Rust `adam_state` kwarg.

    Layout mirrors `new_adam_state`: scalars (w0 moments) by value — the kernel
    returns them updated — and contiguous float64 arrays mutated in place.
    """
    m0, v0, t0 = (float(adam_state[i]) for i in range(3))
    arrs = tuple(np.ascontiguousarray(a, dtype=np.float64) for a in adam_state[3:])
    return (m0, v0, t0) + arrs


def _ftrl_arrays(ftrl_state):
    """Binary FTRL (z, n) state prepped for the Rust `ftrl_state` kwarg
    (scalars by value, arrays in place), mirroring `new_ftrl_state`."""
    z0, n0 = float(ftrl_state[0]), float(ftrl_state[1])
    arrs = tuple(np.ascontiguousarray(a, dtype=np.float64) for a in ftrl_state[2:])
    return (z0, n0) + arrs


def _state_arrays(state):
    """Multiclass optimizer state (all per-class arrays) prepped for Rust:
    contiguous float64, mutated in place by the kernel."""
    return tuple(np.ascontiguousarray(a, dtype=np.float64) for a in state)


def fm_fit(
    X, y, params, *, loss, optimizer, learning_rate, l2_linear, l2_factors, row_orders,
    l1_linear=0.0, l1_factors=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-8, ftrl_beta=1.0,
    batch_size=1, n_jobs=1, sample_weight=None, state=None, adam_state=None, ftrl_state=None,
    backend="rust_cpu",
):
    """Train an FM (docs/optimization_spec.md).

    `params` = (w0, w, V) initial values (unchanged); returns new float64
    (w0, w, V). `sample_weight` scales each row's gradient (None -> all ones).
    `batch_size` averages each batch's gradient (batch_size=1 is per-row).
    `n_jobs` (>= 1) splits each batch across that many rayon threads; n_jobs=1
    is the serial path matching the reference. `beta_1`/`beta_2`/`epsilon` apply
    only when optimizer == "adam". `state` carries AdaGrad accumulators in/out
    for epoch-by-epoch training (early stopping); `adam_state` / `ftrl_state`
    do the same for the Adam moments / FTRL (z, n) state — all three round-trip
    through the Rust kernel. The reference fallback is always serial (it is the
    n_jobs=1 ground truth).

    `backend="cuda"` accumulates each batch's data-gradient on the GPU and
    keeps the optimizer flush (and all optimizer state) on the CPU
    (docs/gpu_backend_plan.md milestone 3); it requires `has_cuda()`, ignores
    `n_jobs`, is nondeterministic run-to-run (atomic gradient accumulation)
    and never falls back silently.
    """
    if backend == "cuda" and not has_cuda():
        raise RuntimeError(
            "backend='cuda' requires modern_fm built with the `cuda-backend` "
            "Cargo feature and an NVIDIA GPU + driver at runtime"
        )
    if _rust is None:
        return _reference_train.fm_fit_reference(
            X, y, params, loss=loss, optimizer=optimizer, learning_rate=learning_rate,
            l2_linear=l2_linear, l2_factors=l2_factors, l1_linear=l1_linear,
            l1_factors=l1_factors, row_orders=row_orders, beta_1=beta_1, beta_2=beta_2,
            epsilon=epsilon, ftrl_beta=ftrl_beta, batch_size=batch_size,
            sample_weight=sample_weight, state=state, adam_state=adam_state,
            ftrl_state=ftrl_state,
        )
    (indptr, indices, data, n_features), y, w0, w, V, row_orders = _prep_fit(
        X, y, params, row_orders
    )
    sw = np.ones(len(y)) if sample_weight is None else _prep_vec(sample_weight)
    acc_w0, acc_w, acc_v = _acc_arrays(state, w, V)
    adam_t = None if adam_state is None else _adam_arrays(adam_state)
    ftrl_t = None if ftrl_state is None else _ftrl_arrays(ftrl_state)
    w0, acc_w0, adam_sc, ftrl_sc = _rust.fm_fit_csr(
        indptr, indices, data, n_features, y, sw, w0, acc_w0, w, V, acc_w, acc_v,
        loss, optimizer, learning_rate, l2_linear, l2_factors, beta_1, beta_2, epsilon,
        row_orders, batch_size, n_jobs, l1_linear, l1_factors, ftrl_beta,
        adam_state=adam_t, ftrl_state=ftrl_t, use_cuda=backend == "cuda",
    )
    if state is not None:
        state[0], state[1], state[2] = acc_w0, acc_w, acc_v
    if adam_state is not None:  # arrays mutated in place; write back scalars + preps
        adam_state[0], adam_state[1], adam_state[2] = adam_sc
        adam_state[3:] = adam_t[3:]
    if ftrl_state is not None:
        ftrl_state[0], ftrl_state[1] = ftrl_sc
        ftrl_state[2:] = ftrl_t[2:]
    return w0, w, V


def fm_fit_multiclass(
    X, y, params, *, optimizer, learning_rate, l2_linear, l2_factors, row_orders,
    label_smoothing=0.0, l1_linear=0.0, l1_factors=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
    ftrl_beta=1.0, batch_size=1, sample_weight=None, state=None, adam_state=None, ftrl_state=None,
):
    """Train a multiclass (softmax) FM (optimization_spec.md).

    `params` = (w0 (C,), w (C, n), V (C, n, k)) initial values (unchanged);
    `y` holds integer class indices in [0, C). Returns new float64 (w0, w, V).
    `batch_size` averages each batch's gradient (batch_size=1 is per-row).
    `beta_1`/`beta_2`/`epsilon` apply only when optimizer == "adam".
    `state` (per-class AdaGrad accumulators) / `adam_state` / `ftrl_state`
    round-trip the optimizer state across epoch-driven early-stopping calls,
    through the Rust kernel when available.
    """
    if _rust is None:
        return _reference_train.fm_fit_multiclass_reference(
            X, y, params, optimizer=optimizer, learning_rate=learning_rate,
            l2_linear=l2_linear, l2_factors=l2_factors, l1_linear=l1_linear,
            l1_factors=l1_factors, row_orders=row_orders, label_smoothing=label_smoothing,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, ftrl_beta=ftrl_beta,
            batch_size=batch_size, sample_weight=sample_weight, state=state,
            adam_state=adam_state, ftrl_state=ftrl_state,
        )
    w0, w, V = params
    w0 = np.array(w0, dtype=np.float64, order="C", copy=True)  # (C,), mutated in place
    w = np.array(w, dtype=np.float64, order="C", copy=True)  # (C, n)
    V = np.array(V, dtype=np.float64, order="C", copy=True)  # (C, n, k)
    y = _prep_vec(y, dtype=np.int64)
    row_orders = np.ascontiguousarray(row_orders, dtype=np.int64)
    if row_orders.ndim == 1:
        row_orders = row_orders[None, :]
    Xc = X if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float64))
    indptr, indices, data, n_features = _prep_csr(Xc)
    sw = np.ones(len(y)) if sample_weight is None else _prep_vec(sample_weight)
    st_t = None if state is None else _state_arrays(state)
    adam_t = None if adam_state is None else _state_arrays(adam_state)
    ftrl_t = None if ftrl_state is None else _state_arrays(ftrl_state)
    _rust.fm_fit_multiclass_csr(
        indptr, indices, data, n_features, y, sw, w0, w, V,
        optimizer, learning_rate, l2_linear, l2_factors, label_smoothing,
        beta_1, beta_2, epsilon, row_orders, batch_size, l1_linear, l1_factors, ftrl_beta,
        state=st_t, adam_state=adam_t, ftrl_state=ftrl_t,
    )
    # all state arrays are mutated in place; write the prepped arrays back into
    # the callers' lists in case ascontiguousarray re-allocated
    if state is not None:
        state[:] = st_t
    if adam_state is not None:
        adam_state[:] = adam_t
    if ftrl_state is not None:
        ftrl_state[:] = ftrl_t
    return w0, w, V


def ffm_fit(
    X, y, field_ids, params, *, loss, optimizer, learning_rate, l2_linear, l2_factors, row_orders,
    l1_linear=0.0, l1_factors=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-8, ftrl_beta=1.0,
    batch_size=1, n_jobs=1, sample_weight=None, state=None, adam_state=None, ftrl_state=None,
):
    """Train an FFM (loss "logistic" or "squared"); see fm_fit. `batch_size` averages each
    batch's gradient (batch_size=1 is per-row); `n_jobs` (>= 1) splits each batch
    across that many rayon threads (n_jobs=1 matches the serial reference).
    `state` / `adam_state` / `ftrl_state` round-trip the optimizer state across
    epoch-driven early-stopping calls through the Rust kernel, like fm_fit."""
    if _rust is None:
        return _reference_train.ffm_fit_reference(
            X, y, field_ids, params, loss=loss, optimizer=optimizer, learning_rate=learning_rate,
            l2_linear=l2_linear, l2_factors=l2_factors, l1_linear=l1_linear,
            l1_factors=l1_factors, row_orders=row_orders, beta_1=beta_1, beta_2=beta_2,
            epsilon=epsilon, ftrl_beta=ftrl_beta, batch_size=batch_size,
            sample_weight=sample_weight, state=state, adam_state=adam_state,
            ftrl_state=ftrl_state,
        )
    field_ids = _prep_vec(field_ids, dtype=np.int64)
    (indptr, indices, data, n_features), y, w0, w, V, row_orders = _prep_fit(
        X, y, params, row_orders
    )
    sw = np.ones(len(y)) if sample_weight is None else _prep_vec(sample_weight)
    acc_w0, acc_w, acc_v = _acc_arrays(state, w, V)
    adam_t = None if adam_state is None else _adam_arrays(adam_state)
    ftrl_t = None if ftrl_state is None else _ftrl_arrays(ftrl_state)
    w0, acc_w0, adam_sc, ftrl_sc = _rust.ffm_fit_csr(
        indptr, indices, data, n_features, y, sw, field_ids, w0, acc_w0, w, V, acc_w, acc_v,
        loss, optimizer, learning_rate, l2_linear, l2_factors, beta_1, beta_2, epsilon, row_orders,
        batch_size, n_jobs, l1_linear, l1_factors, ftrl_beta,
        adam_state=adam_t, ftrl_state=ftrl_t,
    )
    if state is not None:
        state[0], state[1], state[2] = acc_w0, acc_w, acc_v
    if adam_state is not None:  # arrays mutated in place; write back scalars + preps
        adam_state[0], adam_state[1], adam_state[2] = adam_sc
        adam_state[3:] = adam_t[3:]
    if ftrl_state is not None:
        ftrl_state[0], ftrl_state[1] = ftrl_sc
        ftrl_state[2:] = ftrl_t[2:]
    return w0, w, V


def ffm_fit_multiclass(
    X, y, field_ids, params, *, optimizer, learning_rate, l2_linear, l2_factors, row_orders,
    label_smoothing=0.0, l1_linear=0.0, l1_factors=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
    ftrl_beta=1.0, batch_size=1, sample_weight=None, state=None, adam_state=None, ftrl_state=None,
):
    """Train a multiclass (softmax) FFM (one FFM per class, coupled by softmax).

    `params` = (w0 (C,), w (C, n), V (C, n, n_fields, k)); `y` holds class indices
    in [0, C). Serial (no n_jobs), like FM multiclass. `state` / `adam_state` /
    `ftrl_state` round-trip the per-class optimizer state across epoch-driven
    early-stopping calls through the Rust kernel (see fm_fit_multiclass).
    """
    if _rust is None:
        return _reference_train.ffm_fit_multiclass_reference(
            X, y, field_ids, params, optimizer=optimizer, learning_rate=learning_rate,
            l2_linear=l2_linear, l2_factors=l2_factors, l1_linear=l1_linear,
            l1_factors=l1_factors, row_orders=row_orders, label_smoothing=label_smoothing,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, ftrl_beta=ftrl_beta,
            batch_size=batch_size, sample_weight=sample_weight,
            state=state, adam_state=adam_state, ftrl_state=ftrl_state,
        )
    w0, w, V = params
    w0 = np.array(w0, dtype=np.float64, order="C", copy=True)  # (C,), mutated in place
    w = np.array(w, dtype=np.float64, order="C", copy=True)  # (C, n)
    V = np.array(V, dtype=np.float64, order="C", copy=True)  # (C, n, n_fields, k)
    y = _prep_vec(y, dtype=np.int64)
    field_ids = _prep_vec(field_ids, dtype=np.int64)
    row_orders = np.ascontiguousarray(row_orders, dtype=np.int64)
    if row_orders.ndim == 1:
        row_orders = row_orders[None, :]
    Xc = X if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float64))
    indptr, indices, data, n_features = _prep_csr(Xc)
    sw = np.ones(len(y)) if sample_weight is None else _prep_vec(sample_weight)
    st_t = None if state is None else _state_arrays(state)
    adam_t = None if adam_state is None else _state_arrays(adam_state)
    ftrl_t = None if ftrl_state is None else _state_arrays(ftrl_state)
    _rust.ffm_fit_multiclass_csr(
        indptr, indices, data, n_features, y, sw, field_ids, w0, w, V,
        optimizer, learning_rate, l2_linear, l2_factors, label_smoothing,
        beta_1, beta_2, epsilon, row_orders, batch_size, l1_linear, l1_factors, ftrl_beta,
        state=st_t, adam_state=adam_t, ftrl_state=ftrl_t,
    )
    # all state arrays are mutated in place; write the prepped arrays back into
    # the callers' lists in case ascontiguousarray re-allocated
    if state is not None:
        state[:] = st_t
    if adam_state is not None:
        adam_state[:] = adam_t
    if ftrl_state is not None:
        ftrl_state[:] = ftrl_t
    return w0, w, V


def fwfm_predict(X, field_ids, w0, w, V, r):
    """FwFM prediction (math_spec_fwfm.md), Rust-accelerated when available."""
    if _rust is None:
        return _reference.fwfm_predict(X, field_ids, w0, w, V, r)
    field_ids = _prep_vec(field_ids, dtype=np.int64)
    w = _prep_vec(w)
    V = _prep_dense(V)
    r = _prep_dense(r)
    if sp.issparse(X):
        indptr, indices, data, n_features = _prep_csr(X)
        return _rust.fwfm_predict_csr(
            indptr, indices, data, n_features, field_ids, float(w0), w, V, r
        )
    return _rust.fwfm_predict_dense(_prep_dense(X), field_ids, float(w0), w, V, r)


def _fwfm_acc_arrays(state, w, V, r):
    """FwFM AdaGrad accumulators (acc_w0, acc_w, acc_v, acc_r) from `state`
    ([acc_w0, acc_w, acc_V, acc_R]) or fresh zeros; see `_acc_arrays`."""
    if state is None:
        return 0.0, np.zeros(len(w)), np.zeros_like(V), np.zeros_like(r)
    acc_w0, acc_w, acc_v, acc_r = state
    return float(acc_w0), _prep_vec(acc_w), _prep_dense(acc_v), _prep_dense(acc_r)


def fwfm_fit(
    X, y, field_ids, params, *, loss, optimizer, learning_rate, l2_linear, l2_factors,
    row_orders, l1_linear=0.0, l1_factors=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
    ftrl_beta=1.0, batch_size=1, sample_weight=None, state=None, adam_state=None,
    ftrl_state=None,
):
    """Train an FwFM (math_spec_fwfm.md); `params` = (w0, w, V, R), returns new
    float64 copies. Serial (no n_jobs in v0.5). `state` = [acc_w0, acc_w,
    acc_V, acc_R]; `adam_state` / `ftrl_state` follow `new_adam_state_fwfm` /
    `new_ftrl_state_fwfm` and round-trip through the Rust kernel, like fm_fit.
    """
    if _rust is None:
        return _reference_train.fwfm_fit_reference(
            X, y, field_ids, params, loss=loss, optimizer=optimizer,
            learning_rate=learning_rate, l2_linear=l2_linear, l2_factors=l2_factors,
            l1_linear=l1_linear, l1_factors=l1_factors, row_orders=row_orders,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, ftrl_beta=ftrl_beta,
            batch_size=batch_size, sample_weight=sample_weight, state=state,
            adam_state=adam_state, ftrl_state=ftrl_state,
        )
    w0, w, V, r = params
    field_ids = _prep_vec(field_ids, dtype=np.int64)
    r = np.array(r, dtype=np.float64, order="C", copy=True)
    (indptr, indices, data, n_features), y, w0, w, V, row_orders = _prep_fit(
        X, y, (w0, w, V), row_orders
    )
    sw = np.ones(len(y)) if sample_weight is None else _prep_vec(sample_weight)
    acc_w0, acc_w, acc_v, acc_r = _fwfm_acc_arrays(state, w, V, r)
    adam_t = None if adam_state is None else _adam_arrays(adam_state)
    ftrl_t = None if ftrl_state is None else _ftrl_arrays(ftrl_state)
    w0, acc_w0, adam_sc, ftrl_sc = _rust.fwfm_fit_csr(
        indptr, indices, data, n_features, y, sw, field_ids, w0, acc_w0, w, V, r,
        acc_w, acc_v, acc_r, loss, optimizer, learning_rate, l2_linear, l2_factors,
        beta_1, beta_2, epsilon, row_orders, batch_size, l1_linear, l1_factors, ftrl_beta,
        adam_state=adam_t, ftrl_state=ftrl_t,
    )
    if state is not None:
        state[0], state[1], state[2], state[3] = acc_w0, acc_w, acc_v, acc_r
    if adam_state is not None:  # arrays mutated in place; write back scalars + preps
        adam_state[0], adam_state[1], adam_state[2] = adam_sc
        adam_state[3:] = adam_t[3:]
    if ftrl_state is not None:
        ftrl_state[0], ftrl_state[1] = ftrl_sc
        ftrl_state[2:] = ftrl_t[2:]
    return w0, w, V, r


def fwfm_fit_multiclass(
    X, y, field_ids, params, *, optimizer, learning_rate, l2_linear, l2_factors,
    row_orders, label_smoothing=0.0, l1_linear=0.0, l1_factors=0.0, beta_1=0.9,
    beta_2=0.999, epsilon=1e-8, ftrl_beta=1.0, batch_size=1, sample_weight=None,
    state=None, adam_state=None, ftrl_state=None,
):
    """Train a multiclass (softmax) FwFM (one FwFM per class, coupled by softmax).

    `params` = (w0 (C,), w (C, n), V (C, n, k), R (C, F, F)); `y` holds class
    indices in [0, C). Serial. `state` / `adam_state` / `ftrl_state` round-trip
    the per-class optimizer state through the Rust kernel (see fm_fit_multiclass).
    """
    if _rust is None:
        return _reference_train.fwfm_fit_multiclass_reference(
            X, y, field_ids, params, optimizer=optimizer, learning_rate=learning_rate,
            l2_linear=l2_linear, l2_factors=l2_factors, l1_linear=l1_linear,
            l1_factors=l1_factors, row_orders=row_orders, label_smoothing=label_smoothing,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, ftrl_beta=ftrl_beta,
            batch_size=batch_size, sample_weight=sample_weight,
            state=state, adam_state=adam_state, ftrl_state=ftrl_state,
        )
    w0, w, V, r = params
    w0 = np.array(w0, dtype=np.float64, order="C", copy=True)  # (C,), mutated in place
    w = np.array(w, dtype=np.float64, order="C", copy=True)  # (C, n)
    V = np.array(V, dtype=np.float64, order="C", copy=True)  # (C, n, k)
    r = np.array(r, dtype=np.float64, order="C", copy=True)  # (C, F, F)
    y = _prep_vec(y, dtype=np.int64)
    field_ids = _prep_vec(field_ids, dtype=np.int64)
    row_orders = np.ascontiguousarray(row_orders, dtype=np.int64)
    if row_orders.ndim == 1:
        row_orders = row_orders[None, :]
    Xc = X if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float64))
    indptr, indices, data, n_features = _prep_csr(Xc)
    sw = np.ones(len(y)) if sample_weight is None else _prep_vec(sample_weight)
    st_t = None if state is None else _state_arrays(state)
    adam_t = None if adam_state is None else _state_arrays(adam_state)
    ftrl_t = None if ftrl_state is None else _state_arrays(ftrl_state)
    _rust.fwfm_fit_multiclass_csr(
        indptr, indices, data, n_features, y, sw, field_ids, w0, w, V, r,
        optimizer, learning_rate, l2_linear, l2_factors, label_smoothing,
        beta_1, beta_2, epsilon, row_orders, batch_size, l1_linear, l1_factors, ftrl_beta,
        state=st_t, adam_state=adam_t, ftrl_state=ftrl_t,
    )
    # all state arrays are mutated in place; write the prepped arrays back into
    # the callers' lists in case ascontiguousarray re-allocated
    if state is not None:
        state[:] = st_t
    if adam_state is not None:
        adam_state[:] = adam_t
    if ftrl_state is not None:
        ftrl_state[:] = ftrl_t
    return w0, w, V, r


def fm_bi_interaction(X, V):
    """Bi-interaction pooled features (n, k); NumPy is already two sparse
    matmuls (BLAS-grade, O(nnz k)), so there is no Rust kernel — this wrapper
    exists so one can slot in later without an API change."""
    return _reference.fm_bi_interaction(X, V)
