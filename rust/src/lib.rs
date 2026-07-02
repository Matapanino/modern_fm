//! PyO3 bindings for the modern_fm Rust backend, exposed as `modern_fm._rust`.
//!
//! Private API — called only through `modern_fm._backend`. Contract
//! (enforced by the Python wrapper, validated again here):
//! float64 C-contiguous arrays, int64 CSR indices/indptr and field_ids.
//! Shapes follow docs/math_spec.md: w (n_features,), FM V (n_features, k),
//! FFM V (n_features, n_fields, k).

use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3, PyReadwriteArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(all(feature = "cuda-backend", not(target_os = "macos")))]
mod cuda;
mod data;
mod ffm;
mod fm;
mod fwfm;
mod optimizer;

use data::CsrView;
use optimizer::{
    AdamStateMut, FtrlStateMut, GroupStateMut, Loss, McGroupState, McState, Optimizer,
};

fn val_err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

// Optional optimizer-state arguments for the epoch-driven early-stopping
// hand-off. Layouts mirror `_reference_train.new_adam_state` /
// `new_ftrl_state`: binary states carry the scalar w0 moments by value
// (returned updated); multiclass states are all (C, ·) arrays mutated in
// place. The FFM variants only differ in the factor-array dimensionality.
type Arr1<'py> = PyReadwriteArray1<'py, f64>;
type Arr2<'py> = PyReadwriteArray2<'py, f64>;
type Arr3<'py> = PyReadwriteArray3<'py, f64>;
type Arr4<'py> = PyReadwriteArray4<'py, f64>;
type FmAdamArg<'py> = (f64, f64, f64, Arr1<'py>, Arr1<'py>, Arr1<'py>, Arr2<'py>, Arr2<'py>, Arr2<'py>);
type FmFtrlArg<'py> = (f64, f64, Arr1<'py>, Arr1<'py>, Arr2<'py>, Arr2<'py>);
type FfmAdamArg<'py> = (f64, f64, f64, Arr1<'py>, Arr1<'py>, Arr1<'py>, Arr3<'py>, Arr3<'py>, Arr3<'py>);
type FfmFtrlArg<'py> = (f64, f64, Arr1<'py>, Arr1<'py>, Arr3<'py>, Arr3<'py>);
type FmMcStateArg<'py> = (Arr1<'py>, Arr2<'py>, Arr3<'py>);
type FmMcAdamArg<'py> = (Arr1<'py>, Arr1<'py>, Arr1<'py>, Arr2<'py>, Arr2<'py>, Arr2<'py>, Arr3<'py>, Arr3<'py>, Arr3<'py>);
type FmMcFtrlArg<'py> = (Arr1<'py>, Arr1<'py>, Arr2<'py>, Arr2<'py>, Arr3<'py>, Arr3<'py>);
type FfmMcStateArg<'py> = (Arr1<'py>, Arr2<'py>, Arr4<'py>);
// FwFM adds the field-pair matrix R as a fourth parameter group appended to
// the FM-shaped layouts (docs/math_spec_fwfm.md): binary V slots are (n, k),
// R slots (F, F); multiclass adds the class axis.
type FwfmAdamArg<'py> = (
    f64, f64, f64,
    Arr1<'py>, Arr1<'py>, Arr1<'py>,
    Arr2<'py>, Arr2<'py>, Arr2<'py>,
    Arr2<'py>, Arr2<'py>, Arr2<'py>,
);
type FwfmFtrlArg<'py> = (f64, f64, Arr1<'py>, Arr1<'py>, Arr2<'py>, Arr2<'py>, Arr2<'py>, Arr2<'py>);
type FwfmMcStateArg<'py> = (Arr1<'py>, Arr2<'py>, Arr3<'py>, Arr3<'py>);
type FwfmMcAdamArg<'py> = (
    Arr1<'py>, Arr1<'py>, Arr1<'py>,
    Arr2<'py>, Arr2<'py>, Arr2<'py>,
    Arr3<'py>, Arr3<'py>, Arr3<'py>,
    Arr3<'py>, Arr3<'py>, Arr3<'py>,
);
type FwfmMcFtrlArg<'py> = (
    Arr1<'py>, Arr1<'py>, Arr2<'py>, Arr2<'py>, Arr3<'py>, Arr3<'py>, Arr3<'py>, Arr3<'py>,
);
type FfmMcAdamArg<'py> = (Arr1<'py>, Arr1<'py>, Arr1<'py>, Arr2<'py>, Arr2<'py>, Arr2<'py>, Arr4<'py>, Arr4<'py>, Arr4<'py>);
type FfmMcFtrlArg<'py> = (Arr1<'py>, Arr1<'py>, Arr2<'py>, Arr2<'py>, Arr4<'py>, Arr4<'py>);

/// Scalar returns of the binary fit kernels: (w0, acc_w0, Adam w0 moments
/// (m, s, t), FTRL w0 state (z, n)) — the optimizer-state scalars are
/// meaningful only when the corresponding `*_state` argument was passed.
type FitScalars = (f64, f64, (f64, f64, f64), (f64, f64));

/// Zeroed local backing for one optimizer-state array when no external state
/// is passed (`on` = the owning optimizer is active; empty otherwise).
fn local_state(on: bool, len: usize) -> Vec<f64> {
    if on {
        vec![0.0; len]
    } else {
        Vec::new()
    }
}

fn parse_loss(s: &str) -> PyResult<Loss> {
    match s {
        "logistic" => Ok(Loss::Logistic),
        "squared" => Ok(Loss::Squared),
        _ => Err(val_err(format!("unknown loss {s:?}"))),
    }
}

fn parse_optimizer(
    s: &str,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    ftrl_beta: f64,
) -> PyResult<Optimizer> {
    match s {
        "sgd" => Ok(Optimizer::Sgd),
        "adagrad" => Ok(Optimizer::Adagrad),
        "adam" => Ok(Optimizer::Adam {
            beta1: beta_1,
            beta2: beta_2,
            eps: epsilon,
        }),
        "ftrl" => Ok(Optimizer::Ftrl { beta: ftrl_beta }),
        _ => Err(val_err(format!("unknown optimizer {s:?}"))),
    }
}

fn check_sw_ro(
    n_rows: usize,
    sample_weight: &[f64],
    row_orders_shape: &[usize],
    ro: &[i64],
) -> Result<(), String> {
    if sample_weight.len() != n_rows {
        return Err(format!(
            "sample_weight length {} != n_rows {}",
            sample_weight.len(),
            n_rows
        ));
    }
    if row_orders_shape[1] != n_rows {
        return Err(format!(
            "row_orders second dimension {} != n_rows {}",
            row_orders_shape[1], n_rows
        ));
    }
    if ro.iter().any(|&r| r < 0 || r as usize >= n_rows) {
        return Err(format!("row_orders entry out of range [0, {n_rows})"));
    }
    Ok(())
}

fn check_fit_args(
    csr: &CsrView,
    y: &[f64],
    sample_weight: &[f64],
    row_orders_shape: &[usize],
    ro: &[i64],
) -> Result<(), String> {
    let n_rows = csr.n_rows();
    if y.len() != n_rows {
        return Err(format!("y length {} != n_rows {}", y.len(), n_rows));
    }
    check_sw_ro(n_rows, sample_weight, row_orders_shape, ro)
}

fn check_fit_args_mc(
    csr: &CsrView,
    y: &[i64],
    n_classes: usize,
    sample_weight: &[f64],
    row_orders_shape: &[usize],
    ro: &[i64],
) -> Result<(), String> {
    let n_rows = csr.n_rows();
    if y.len() != n_rows {
        return Err(format!("y length {} != n_rows {}", y.len(), n_rows));
    }
    if y.iter().any(|&c| c < 0 || c as usize >= n_classes) {
        return Err(format!("class index out of range [0, {n_classes})"));
    }
    check_sw_ro(n_rows, sample_weight, row_orders_shape, ro)
}

fn check_w_v_fm(n_features: usize, w: &[f64], v_shape: &[usize]) -> PyResult<()> {
    if w.len() != n_features || v_shape[0] != n_features {
        return Err(val_err(format!(
            "shape mismatch: n_features={n_features}, w={}, V rows={}",
            w.len(),
            v_shape[0]
        )));
    }
    Ok(())
}

#[pyfunction]
fn fm_predict_fast_dense<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    w0: f64,
    w: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (n_rows, n_features) = (x.shape()[0], x.shape()[1]);
    let k = v.shape()[1];
    let x = x.as_slice()?;
    let w = w.as_slice()?;
    check_w_v_fm(n_features, w, v.shape())?;
    let v = v.as_slice()?;
    let out = py.allow_threads(|| fm::predict_dense(x, n_rows, n_features, w0, w, v, k));
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fm_predict_fast_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_features: usize,
    w0: f64,
    w: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let k = v.shape()[1];
    let (indptr, indices, data) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let w = w.as_slice()?;
    check_w_v_fm(n_features, w, v.shape())?;
    let v = v.as_slice()?;
    let out = py.allow_threads(|| -> Result<Vec<f64>, String> {
        let csr = CsrView::new(indptr, indices, data, n_features)?;
        Ok(fm::predict_csr(&csr, w0, w, v, k))
    });
    Ok(out.map_err(val_err)?.into_pyarray(py))
}

#[pyfunction]
fn ffm_predict_dense<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    field_ids: PyReadonlyArray1<'py, i64>,
    w0: f64,
    w: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (n_rows, n_features) = (x.shape()[0], x.shape()[1]);
    let (n_fields, k) = (v.shape()[1], v.shape()[2]);
    let x = x.as_slice()?;
    let w = w.as_slice()?;
    check_w_v_fm(n_features, w, v.shape())?;
    let field_ids = field_ids.as_slice()?;
    data::check_field_ids(field_ids, n_features, n_fields).map_err(val_err)?;
    let v = v.as_slice()?;
    let out = py.allow_threads(|| {
        ffm::predict_dense(x, n_rows, n_features, field_ids, w0, w, v, n_fields, k)
    });
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ffm_predict_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_features: usize,
    field_ids: PyReadonlyArray1<'py, i64>,
    w0: f64,
    w: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (n_fields, k) = (v.shape()[1], v.shape()[2]);
    let (indptr, indices, data_s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let w = w.as_slice()?;
    check_w_v_fm(n_features, w, v.shape())?;
    let field_ids = field_ids.as_slice()?;
    data::check_field_ids(field_ids, n_features, n_fields).map_err(val_err)?;
    let v = v.as_slice()?;
    let out = py.allow_threads(|| -> Result<Vec<f64>, String> {
        let csr = CsrView::new(indptr, indices, data_s, n_features)?;
        Ok(ffm::predict_csr(&csr, field_ids, w0, w, v, n_fields, k))
    });
    Ok(out.map_err(val_err)?.into_pyarray(py))
}

/// Train an FM in place (w, v, acc_*, adam/ftrl arrays mutated; new
/// (w0, acc_w0, (m_w0, s_w0, t_w0), (z_w0, n_w0)) returned — the scalar
/// optimizer state is zeros when its optimizer is off). Optimizer state is
/// passed in/out so the caller can drive epochs one at a time (early
/// stopping); a single all-epochs call passes zeros / None. See fm::fit_csr
/// and docs/optimization_spec.md.
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, data, n_features, y, sample_weight, w0, acc_w0, w, v, acc_w, acc_v,
    loss, optimizer, learning_rate, l2_linear, l2_factors, beta_1, beta_2, epsilon,
    row_orders, batch_size, n_jobs, l1_linear, l1_factors, ftrl_beta,
    adam_state=None, ftrl_state=None, use_cuda=false,
))]
#[allow(clippy::too_many_arguments)]
fn fm_fit_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_features: usize,
    y: PyReadonlyArray1<'py, f64>,
    sample_weight: PyReadonlyArray1<'py, f64>,
    w0: f64,
    acc_w0: f64,
    mut w: PyReadwriteArray1<'py, f64>,
    mut v: PyReadwriteArray2<'py, f64>,
    mut acc_w: PyReadwriteArray1<'py, f64>,
    mut acc_v: PyReadwriteArray2<'py, f64>,
    loss: &str,
    optimizer: &str,
    learning_rate: f64,
    l2_linear: f64,
    l2_factors: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    row_orders: PyReadonlyArray2<'py, i64>,
    batch_size: usize,
    n_jobs: usize,
    l1_linear: f64,
    l1_factors: f64,
    ftrl_beta: f64,
    mut adam_state: Option<FmAdamArg<'py>>,
    mut ftrl_state: Option<FmFtrlArg<'py>>,
    use_cuda: bool,
) -> PyResult<FitScalars> {
    let loss = parse_loss(loss)?;
    let opt = parse_optimizer(optimizer, beta_1, beta_2, epsilon, ftrl_beta)?;
    if batch_size == 0 {
        return Err(val_err("batch_size must be >= 1".to_string()));
    }
    let k = v.shape()[1];
    let v_rows = v.shape()[0];
    let ro_shape = [row_orders.shape()[0], row_orders.shape()[1]];
    let (indptr_s, indices_s, data_s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let y_s = y.as_slice()?;
    let sw_s = sample_weight.as_slice()?;
    let ro = row_orders.as_slice()?;
    let w_s = w.as_slice_mut()?;
    if w_s.len() != n_features || v_rows != n_features {
        return Err(val_err(format!(
            "shape mismatch: n_features={n_features}, w={}, V rows={v_rows}",
            w_s.len()
        )));
    }
    let v_s = v.as_slice_mut()?;
    let acc_w_s = acc_w.as_slice_mut()?;
    let acc_v_s = acc_v.as_slice_mut()?;
    if acc_w_s.len() != w_s.len() || acc_v_s.len() != v_s.len() {
        return Err(val_err("accumulator shapes must match w and V".to_string()));
    }
    let v_len = v_s.len();
    let adam_on = matches!(opt, Optimizer::Adam { .. });
    let ftrl_on = matches!(opt, Optimizer::Ftrl { .. });
    if adam_state.is_some() && !adam_on {
        return Err(val_err("adam_state is only valid for optimizer='adam'".to_string()));
    }
    if ftrl_state.is_some() && !ftrl_on {
        return Err(val_err("ftrl_state is only valid for optimizer='ftrl'".to_string()));
    }
    let (mut m_w0, mut s_w0, mut t_w0) =
        adam_state.as_ref().map_or((0.0, 0.0, 0.0), |t| (t.0, t.1, t.2));
    let (mut z_w0, mut n_w0) = ftrl_state.as_ref().map_or((0.0, 0.0), |t| (t.0, t.1));
    let ext_adam = adam_state.is_some();
    let ext_ftrl = ftrl_state.is_some();
    let need_adam = adam_on && !ext_adam;
    let need_ftrl = ftrl_on && !ext_ftrl;
    let (mut lm_w, mut ls_w, mut lt_w) = (
        local_state(need_adam, n_features),
        local_state(need_adam, n_features),
        local_state(need_adam, n_features),
    );
    let (mut lm_v, mut ls_v, mut lt_v) = (
        local_state(need_adam, v_len),
        local_state(need_adam, v_len),
        local_state(need_adam, v_len),
    );
    let (mut lz_w, mut ln_w) = (local_state(need_ftrl, n_features), local_state(need_ftrl, n_features));
    let (mut lz_v, mut ln_v) = (local_state(need_ftrl, v_len), local_state(need_ftrl, v_len));
    let (m_w_s, s_w_s, t_w_s, m_v_s, s_v_s, t_v_s) = match adam_state.as_mut() {
        Some(t) => (
            t.3.as_slice_mut()?, t.4.as_slice_mut()?, t.5.as_slice_mut()?,
            t.6.as_slice_mut()?, t.7.as_slice_mut()?, t.8.as_slice_mut()?,
        ),
        None => (
            lm_w.as_mut_slice(), ls_w.as_mut_slice(), lt_w.as_mut_slice(),
            lm_v.as_mut_slice(), ls_v.as_mut_slice(), lt_v.as_mut_slice(),
        ),
    };
    let (z_w_s, n_w_s, z_v_s, n_v_s) = match ftrl_state.as_mut() {
        Some(t) => (t.2.as_slice_mut()?, t.3.as_slice_mut()?, t.4.as_slice_mut()?, t.5.as_slice_mut()?),
        None => (lz_w.as_mut_slice(), ln_w.as_mut_slice(), lz_v.as_mut_slice(), ln_v.as_mut_slice()),
    };
    if ext_adam
        && (m_w_s.len() != n_features || s_w_s.len() != n_features || t_w_s.len() != n_features
            || m_v_s.len() != v_len || s_v_s.len() != v_len || t_v_s.len() != v_len)
    {
        return Err(val_err("adam_state shapes must match w and V".to_string()));
    }
    if ext_ftrl
        && (z_w_s.len() != n_features || n_w_s.len() != n_features
            || z_v_s.len() != v_len || n_v_s.len() != v_len)
    {
        return Err(val_err("ftrl_state shapes must match w and V".to_string()));
    }
    let adam_view = AdamStateMut {
        m_w0: &mut m_w0, s_w0: &mut s_w0, t_w0: &mut t_w0,
        m_w: m_w_s, s_w: s_w_s, t_w: t_w_s, m_v: m_v_s, s_v: s_v_s, t_v: t_v_s,
    };
    let ftrl_view = FtrlStateMut {
        z_w0: &mut z_w0, n_w0: &mut n_w0,
        z_w: z_w_s, n_w: n_w_s, z_v: z_v_s, n_v: n_v_s,
    };
    let out = py.allow_threads(|| -> Result<(f64, f64), String> {
        let csr = CsrView::new(indptr_s, indices_s, data_s, n_features)?;
        check_fit_args(&csr, y_s, sw_s, &ro_shape, ro)?;
        let mut w0 = w0;
        let mut acc_w0 = acc_w0;
        if use_cuda {
            #[cfg(all(feature = "cuda-backend", not(target_os = "macos")))]
            cuda::fm_train::fit_csr(
                &csr, y_s, sw_s, &mut w0, w_s, v_s, &mut acc_w0, acc_w_s, acc_v_s,
                adam_view, ftrl_view, k, loss, opt,
                learning_rate, l1_linear, l2_linear, l1_factors, l2_factors, ro_shape[1],
                batch_size, ro,
            )?;
            #[cfg(not(all(feature = "cuda-backend", not(target_os = "macos"))))]
            return Err(
                "use_cuda requires modern_fm built with the cuda-backend feature".to_string()
            );
        } else {
            fm::fit_csr(
                &csr, y_s, sw_s, &mut w0, w_s, v_s, &mut acc_w0, acc_w_s, acc_v_s,
                adam_view, ftrl_view, k, loss, opt,
                learning_rate, l1_linear, l2_linear, l1_factors, l2_factors, ro_shape[1],
                batch_size, n_jobs.max(1), ro,
            );
        }
        Ok((w0, acc_w0))
    });
    let (w0, acc_w0) = out.map_err(val_err)?;
    Ok((w0, acc_w0, (m_w0, s_w0, t_w0), (z_w0, n_w0)))
}

/// Train a multiclass (softmax) FM in place; w0 (C,), w (C, n_features) and
/// v (C, n_features, k) are all mutated. `y` holds class indices in [0, C).
/// `state` (AdaGrad accumulators) / `adam_state` / `ftrl_state` are optional
/// (C, ·) arrays mutated in place so the caller can drive epochs one at a time
/// (early stopping); None keeps the state internal (single all-epochs call).
/// batch_size=1, single-threaded; see fm::fit_multiclass_csr.
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, data, n_features, y, sample_weight, w0, w, v,
    optimizer, learning_rate, l2_linear, l2_factors, label_smoothing, beta_1, beta_2,
    epsilon, row_orders, batch_size, l1_linear, l1_factors, ftrl_beta,
    state=None, adam_state=None, ftrl_state=None,
))]
#[allow(clippy::too_many_arguments)]
fn fm_fit_multiclass_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_features: usize,
    y: PyReadonlyArray1<'py, i64>,
    sample_weight: PyReadonlyArray1<'py, f64>,
    mut w0: PyReadwriteArray1<'py, f64>,
    mut w: PyReadwriteArray2<'py, f64>,
    mut v: PyReadwriteArray3<'py, f64>,
    optimizer: &str,
    learning_rate: f64,
    l2_linear: f64,
    l2_factors: f64,
    label_smoothing: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    row_orders: PyReadonlyArray2<'py, i64>,
    batch_size: usize,
    l1_linear: f64,
    l1_factors: f64,
    ftrl_beta: f64,
    mut state: Option<FmMcStateArg<'py>>,
    mut adam_state: Option<FmMcAdamArg<'py>>,
    mut ftrl_state: Option<FmMcFtrlArg<'py>>,
) -> PyResult<()> {
    let opt = parse_optimizer(optimizer, beta_1, beta_2, epsilon, ftrl_beta)?;
    if batch_size == 0 {
        return Err(val_err("batch_size must be >= 1".to_string()));
    }
    let (n_classes, v_n, k) = (v.shape()[0], v.shape()[1], v.shape()[2]);
    let w_shape = [w.shape()[0], w.shape()[1]];
    let ro_shape = [row_orders.shape()[0], row_orders.shape()[1]];
    let (indptr_s, indices_s, data_s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let y_s = y.as_slice()?;
    let sw_s = sample_weight.as_slice()?;
    let ro = row_orders.as_slice()?;
    let w0_s = w0.as_slice_mut()?;
    let w_s = w.as_slice_mut()?;
    let v_s = v.as_slice_mut()?;
    if w0_s.len() != n_classes || w_shape != [n_classes, n_features] || v_n != n_features {
        return Err(val_err(format!(
            "shape mismatch: n_classes={n_classes}, n_features={n_features}, \
             w0={}, w={w_shape:?}, V dims=[{n_classes}, {v_n}, {k}]",
            w0_s.len()
        )));
    }
    let v_len = n_features * k; // per-class factor entries
    let adam_on = matches!(opt, Optimizer::Adam { .. });
    let ftrl_on = matches!(opt, Optimizer::Ftrl { .. });
    if adam_state.is_some() && !adam_on {
        return Err(val_err("adam_state is only valid for optimizer='adam'".to_string()));
    }
    if ftrl_state.is_some() && !ftrl_on {
        return Err(val_err("ftrl_state is only valid for optimizer='ftrl'".to_string()));
    }
    let (ext_state, ext_adam, ext_ftrl) = (state.is_some(), adam_state.is_some(), ftrl_state.is_some());
    let (need_adam, need_ftrl) = (adam_on && !ext_adam, ftrl_on && !ext_ftrl);
    // Local fallbacks: AdaGrad accumulators are always full-size; the w0-level
    // Adam/FTRL arrays are always (C,) (class_views indexes them); the large
    // per-coordinate arrays are allocated only for the active optimizer.
    let mut lacc_w0 = local_state(!ext_state, n_classes);
    let mut lacc_w = local_state(!ext_state, n_classes * n_features);
    let mut lacc_v = local_state(!ext_state, n_classes * v_len);
    let (mut lm_w0, mut ls_w0, mut lt_w0) = (
        local_state(!ext_adam, n_classes),
        local_state(!ext_adam, n_classes),
        local_state(!ext_adam, n_classes),
    );
    let (mut lm_w, mut ls_w, mut lt_w) = (
        local_state(need_adam, n_classes * n_features),
        local_state(need_adam, n_classes * n_features),
        local_state(need_adam, n_classes * n_features),
    );
    let (mut lm_v, mut ls_v, mut lt_v) = (
        local_state(need_adam, n_classes * v_len),
        local_state(need_adam, n_classes * v_len),
        local_state(need_adam, n_classes * v_len),
    );
    let (mut lz_w0, mut ln_w0) = (local_state(!ext_ftrl, n_classes), local_state(!ext_ftrl, n_classes));
    let (mut lz_w, mut ln_w) = (
        local_state(need_ftrl, n_classes * n_features),
        local_state(need_ftrl, n_classes * n_features),
    );
    let (mut lz_v, mut ln_v) = (
        local_state(need_ftrl, n_classes * v_len),
        local_state(need_ftrl, n_classes * v_len),
    );
    let (acc_w0_s, acc_w_s, acc_v_s) = match state.as_mut() {
        Some(t) => (t.0.as_slice_mut()?, t.1.as_slice_mut()?, t.2.as_slice_mut()?),
        None => (lacc_w0.as_mut_slice(), lacc_w.as_mut_slice(), lacc_v.as_mut_slice()),
    };
    let (m_w0_s, s_w0_s, t_w0_s, m_w_s, s_w_s, t_w_s, m_v_s, s_v_s, t_v_s) =
        match adam_state.as_mut() {
            Some(t) => (
                t.0.as_slice_mut()?, t.1.as_slice_mut()?, t.2.as_slice_mut()?,
                t.3.as_slice_mut()?, t.4.as_slice_mut()?, t.5.as_slice_mut()?,
                t.6.as_slice_mut()?, t.7.as_slice_mut()?, t.8.as_slice_mut()?,
            ),
            None => (
                lm_w0.as_mut_slice(), ls_w0.as_mut_slice(), lt_w0.as_mut_slice(),
                lm_w.as_mut_slice(), ls_w.as_mut_slice(), lt_w.as_mut_slice(),
                lm_v.as_mut_slice(), ls_v.as_mut_slice(), lt_v.as_mut_slice(),
            ),
        };
    let (z_w0_s, n_w0_s, z_w_s, n_w_s, z_v_s, n_v_s) = match ftrl_state.as_mut() {
        Some(t) => (
            t.0.as_slice_mut()?, t.1.as_slice_mut()?, t.2.as_slice_mut()?,
            t.3.as_slice_mut()?, t.4.as_slice_mut()?, t.5.as_slice_mut()?,
        ),
        None => (
            lz_w0.as_mut_slice(), ln_w0.as_mut_slice(), lz_w.as_mut_slice(),
            ln_w.as_mut_slice(), lz_v.as_mut_slice(), ln_v.as_mut_slice(),
        ),
    };
    if ext_state
        && (acc_w0_s.len() != n_classes
            || acc_w_s.len() != n_classes * n_features
            || acc_v_s.len() != n_classes * v_len)
    {
        return Err(val_err("state shapes must match (w0, w, V)".to_string()));
    }
    if ext_adam
        && (m_w0_s.len() != n_classes || s_w0_s.len() != n_classes || t_w0_s.len() != n_classes
            || m_w_s.len() != n_classes * n_features || s_w_s.len() != n_classes * n_features
            || t_w_s.len() != n_classes * n_features
            || m_v_s.len() != n_classes * v_len || s_v_s.len() != n_classes * v_len
            || t_v_s.len() != n_classes * v_len)
    {
        return Err(val_err("adam_state shapes must match (w0, w, V)".to_string()));
    }
    if ext_ftrl
        && (z_w0_s.len() != n_classes || n_w0_s.len() != n_classes
            || z_w_s.len() != n_classes * n_features || n_w_s.len() != n_classes * n_features
            || z_v_s.len() != n_classes * v_len || n_v_s.len() != n_classes * v_len)
    {
        return Err(val_err("ftrl_state shapes must match (w0, w, V)".to_string()));
    }
    let st = McState {
        acc_w0: acc_w0_s, acc_w: acc_w_s, acc_v: acc_v_s,
        m_w0: m_w0_s, s_w0: s_w0_s, t_w0: t_w0_s,
        m_w: m_w_s, s_w: s_w_s, t_w: t_w_s,
        m_v: m_v_s, s_v: s_v_s, t_v: t_v_s,
        z_w0: z_w0_s, n_w0: n_w0_s,
        z_w: z_w_s, n_w: n_w_s, z_v: z_v_s, n_v: n_v_s,
    };
    let out = py.allow_threads(|| -> Result<(), String> {
        let csr = CsrView::new(indptr_s, indices_s, data_s, n_features)?;
        check_fit_args_mc(&csr, y_s, n_classes, sw_s, &ro_shape, ro)?;
        fm::fit_multiclass_csr(
            &csr, y_s, sw_s, w0_s, w_s, v_s, st, n_classes, n_features, k, opt, learning_rate,
            l1_linear, l2_linear, l1_factors, l2_factors, label_smoothing, ro_shape[1], batch_size,
            ro,
        );
        Ok(())
    });
    out.map_err(val_err)
}

/// Train an FFM (squared or logistic loss) in place (w, v, acc_*, adam/ftrl
/// arrays mutated; new (w0, acc_w0, adam scalars, ftrl scalars) returned, like
/// fm_fit_csr). See ffm::fit_csr.
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, data, n_features, y, sample_weight, field_ids, w0, acc_w0, w, v,
    acc_w, acc_v, loss, optimizer, learning_rate, l2_linear, l2_factors, beta_1, beta_2,
    epsilon, row_orders, batch_size, n_jobs, l1_linear, l1_factors, ftrl_beta,
    adam_state=None, ftrl_state=None, use_cuda=false,
))]
#[allow(clippy::too_many_arguments)]
fn ffm_fit_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_features: usize,
    y: PyReadonlyArray1<'py, f64>,
    sample_weight: PyReadonlyArray1<'py, f64>,
    field_ids: PyReadonlyArray1<'py, i64>,
    w0: f64,
    acc_w0: f64,
    mut w: PyReadwriteArray1<'py, f64>,
    mut v: PyReadwriteArray3<'py, f64>,
    mut acc_w: PyReadwriteArray1<'py, f64>,
    mut acc_v: PyReadwriteArray3<'py, f64>,
    loss: &str,
    optimizer: &str,
    learning_rate: f64,
    l2_linear: f64,
    l2_factors: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    row_orders: PyReadonlyArray2<'py, i64>,
    batch_size: usize,
    n_jobs: usize,
    l1_linear: f64,
    l1_factors: f64,
    ftrl_beta: f64,
    mut adam_state: Option<FfmAdamArg<'py>>,
    mut ftrl_state: Option<FfmFtrlArg<'py>>,
    use_cuda: bool,
) -> PyResult<FitScalars> {
    let loss = parse_loss(loss)?;
    let opt = parse_optimizer(optimizer, beta_1, beta_2, epsilon, ftrl_beta)?;
    if batch_size == 0 {
        return Err(val_err("batch_size must be >= 1".to_string()));
    }
    let (v_rows, n_fields, k) = (v.shape()[0], v.shape()[1], v.shape()[2]);
    let ro_shape = [row_orders.shape()[0], row_orders.shape()[1]];
    let (indptr_s, indices_s, data_s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let y_s = y.as_slice()?;
    let sw_s = sample_weight.as_slice()?;
    let ro = row_orders.as_slice()?;
    let field_ids_s = field_ids.as_slice()?;
    let w_s = w.as_slice_mut()?;
    if w_s.len() != n_features || v_rows != n_features {
        return Err(val_err(format!(
            "shape mismatch: n_features={n_features}, w={}, V rows={v_rows}",
            w_s.len()
        )));
    }
    data::check_field_ids(field_ids_s, n_features, n_fields).map_err(val_err)?;
    let v_s = v.as_slice_mut()?;
    let acc_w_s = acc_w.as_slice_mut()?;
    let acc_v_s = acc_v.as_slice_mut()?;
    if acc_w_s.len() != w_s.len() || acc_v_s.len() != v_s.len() {
        return Err(val_err("accumulator shapes must match w and V".to_string()));
    }
    let v_len = v_s.len();
    let adam_on = matches!(opt, Optimizer::Adam { .. });
    let ftrl_on = matches!(opt, Optimizer::Ftrl { .. });
    if adam_state.is_some() && !adam_on {
        return Err(val_err("adam_state is only valid for optimizer='adam'".to_string()));
    }
    if ftrl_state.is_some() && !ftrl_on {
        return Err(val_err("ftrl_state is only valid for optimizer='ftrl'".to_string()));
    }
    let (mut m_w0, mut s_w0, mut t_w0) =
        adam_state.as_ref().map_or((0.0, 0.0, 0.0), |t| (t.0, t.1, t.2));
    let (mut z_w0, mut n_w0) = ftrl_state.as_ref().map_or((0.0, 0.0), |t| (t.0, t.1));
    let ext_adam = adam_state.is_some();
    let ext_ftrl = ftrl_state.is_some();
    let need_adam = adam_on && !ext_adam;
    let need_ftrl = ftrl_on && !ext_ftrl;
    let (mut lm_w, mut ls_w, mut lt_w) = (
        local_state(need_adam, n_features),
        local_state(need_adam, n_features),
        local_state(need_adam, n_features),
    );
    let (mut lm_v, mut ls_v, mut lt_v) = (
        local_state(need_adam, v_len),
        local_state(need_adam, v_len),
        local_state(need_adam, v_len),
    );
    let (mut lz_w, mut ln_w) = (local_state(need_ftrl, n_features), local_state(need_ftrl, n_features));
    let (mut lz_v, mut ln_v) = (local_state(need_ftrl, v_len), local_state(need_ftrl, v_len));
    let (m_w_s, s_w_s, t_w_s, m_v_s, s_v_s, t_v_s) = match adam_state.as_mut() {
        Some(t) => (
            t.3.as_slice_mut()?, t.4.as_slice_mut()?, t.5.as_slice_mut()?,
            t.6.as_slice_mut()?, t.7.as_slice_mut()?, t.8.as_slice_mut()?,
        ),
        None => (
            lm_w.as_mut_slice(), ls_w.as_mut_slice(), lt_w.as_mut_slice(),
            lm_v.as_mut_slice(), ls_v.as_mut_slice(), lt_v.as_mut_slice(),
        ),
    };
    let (z_w_s, n_w_s, z_v_s, n_v_s) = match ftrl_state.as_mut() {
        Some(t) => (t.2.as_slice_mut()?, t.3.as_slice_mut()?, t.4.as_slice_mut()?, t.5.as_slice_mut()?),
        None => (lz_w.as_mut_slice(), ln_w.as_mut_slice(), lz_v.as_mut_slice(), ln_v.as_mut_slice()),
    };
    if ext_adam
        && (m_w_s.len() != n_features || s_w_s.len() != n_features || t_w_s.len() != n_features
            || m_v_s.len() != v_len || s_v_s.len() != v_len || t_v_s.len() != v_len)
    {
        return Err(val_err("adam_state shapes must match w and V".to_string()));
    }
    if ext_ftrl
        && (z_w_s.len() != n_features || n_w_s.len() != n_features
            || z_v_s.len() != v_len || n_v_s.len() != v_len)
    {
        return Err(val_err("ftrl_state shapes must match w and V".to_string()));
    }
    let adam_view = AdamStateMut {
        m_w0: &mut m_w0, s_w0: &mut s_w0, t_w0: &mut t_w0,
        m_w: m_w_s, s_w: s_w_s, t_w: t_w_s, m_v: m_v_s, s_v: s_v_s, t_v: t_v_s,
    };
    let ftrl_view = FtrlStateMut {
        z_w0: &mut z_w0, n_w0: &mut n_w0,
        z_w: z_w_s, n_w: n_w_s, z_v: z_v_s, n_v: n_v_s,
    };
    let out = py.allow_threads(|| -> Result<(f64, f64), String> {
        let csr = CsrView::new(indptr_s, indices_s, data_s, n_features)?;
        check_fit_args(&csr, y_s, sw_s, &ro_shape, ro)?;
        let mut w0 = w0;
        let mut acc_w0 = acc_w0;
        if use_cuda {
            #[cfg(all(feature = "cuda-backend", not(target_os = "macos")))]
            cuda::ffm_train::fit_csr(
                &csr, y_s, sw_s, field_ids_s, &mut w0, w_s, v_s, &mut acc_w0, acc_w_s,
                acc_v_s, adam_view, ftrl_view, n_fields, k, loss, opt,
                learning_rate, l1_linear, l2_linear, l1_factors, l2_factors, ro_shape[1],
                batch_size, ro,
            )?;
            #[cfg(not(all(feature = "cuda-backend", not(target_os = "macos"))))]
            return Err(
                "use_cuda requires modern_fm built with the cuda-backend feature".to_string()
            );
        } else {
            ffm::fit_csr(
                &csr, y_s, sw_s, field_ids_s, &mut w0, w_s, v_s, &mut acc_w0, acc_w_s, acc_v_s,
                adam_view, ftrl_view,
                n_fields, k, loss, opt, learning_rate, l1_linear, l2_linear, l1_factors,
                l2_factors, ro_shape[1], batch_size, n_jobs.max(1), ro,
            );
        }
        Ok((w0, acc_w0))
    });
    let (w0, acc_w0) = out.map_err(val_err)?;
    Ok((w0, acc_w0, (m_w0, s_w0, t_w0), (z_w0, n_w0)))
}

/// Train a multiclass (softmax) FFM in place; w0 (C,), w (C, n_features) and
/// v (C, n_features, n_fields, k) are all mutated. `y` holds class indices in
/// [0, C). `state` / `adam_state` / `ftrl_state` behave as in
/// fm_fit_multiclass_csr. Serial (no n_jobs); see ffm::fit_multiclass_csr.
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, data, n_features, y, sample_weight, field_ids, w0, w, v,
    optimizer, learning_rate, l2_linear, l2_factors, label_smoothing, beta_1, beta_2,
    epsilon, row_orders, batch_size, l1_linear, l1_factors, ftrl_beta,
    state=None, adam_state=None, ftrl_state=None,
))]
#[allow(clippy::too_many_arguments)]
fn ffm_fit_multiclass_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_features: usize,
    y: PyReadonlyArray1<'py, i64>,
    sample_weight: PyReadonlyArray1<'py, f64>,
    field_ids: PyReadonlyArray1<'py, i64>,
    mut w0: PyReadwriteArray1<'py, f64>,
    mut w: PyReadwriteArray2<'py, f64>,
    mut v: PyReadwriteArray4<'py, f64>,
    optimizer: &str,
    learning_rate: f64,
    l2_linear: f64,
    l2_factors: f64,
    label_smoothing: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    row_orders: PyReadonlyArray2<'py, i64>,
    batch_size: usize,
    l1_linear: f64,
    l1_factors: f64,
    ftrl_beta: f64,
    mut state: Option<FfmMcStateArg<'py>>,
    mut adam_state: Option<FfmMcAdamArg<'py>>,
    mut ftrl_state: Option<FfmMcFtrlArg<'py>>,
) -> PyResult<()> {
    let opt = parse_optimizer(optimizer, beta_1, beta_2, epsilon, ftrl_beta)?;
    if batch_size == 0 {
        return Err(val_err("batch_size must be >= 1".to_string()));
    }
    let (n_classes, v_n, n_fields, k) = (v.shape()[0], v.shape()[1], v.shape()[2], v.shape()[3]);
    let w_shape = [w.shape()[0], w.shape()[1]];
    let ro_shape = [row_orders.shape()[0], row_orders.shape()[1]];
    let (indptr_s, indices_s, data_s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let y_s = y.as_slice()?;
    let sw_s = sample_weight.as_slice()?;
    let ro = row_orders.as_slice()?;
    let field_ids_s = field_ids.as_slice()?;
    let w0_s = w0.as_slice_mut()?;
    let w_s = w.as_slice_mut()?;
    let v_s = v.as_slice_mut()?;
    if w0_s.len() != n_classes || w_shape != [n_classes, n_features] || v_n != n_features {
        return Err(val_err(format!(
            "shape mismatch: n_classes={n_classes}, n_features={n_features}, \
             w0={}, w={w_shape:?}, V dims=[{n_classes}, {v_n}, {n_fields}, {k}]",
            w0_s.len()
        )));
    }
    data::check_field_ids(field_ids_s, n_features, n_fields).map_err(val_err)?;
    let v_len = n_features * n_fields * k; // per-class factor entries
    let adam_on = matches!(opt, Optimizer::Adam { .. });
    let ftrl_on = matches!(opt, Optimizer::Ftrl { .. });
    if adam_state.is_some() && !adam_on {
        return Err(val_err("adam_state is only valid for optimizer='adam'".to_string()));
    }
    if ftrl_state.is_some() && !ftrl_on {
        return Err(val_err("ftrl_state is only valid for optimizer='ftrl'".to_string()));
    }
    let (ext_state, ext_adam, ext_ftrl) = (state.is_some(), adam_state.is_some(), ftrl_state.is_some());
    let (need_adam, need_ftrl) = (adam_on && !ext_adam, ftrl_on && !ext_ftrl);
    let mut lacc_w0 = local_state(!ext_state, n_classes);
    let mut lacc_w = local_state(!ext_state, n_classes * n_features);
    let mut lacc_v = local_state(!ext_state, n_classes * v_len);
    let (mut lm_w0, mut ls_w0, mut lt_w0) = (
        local_state(!ext_adam, n_classes),
        local_state(!ext_adam, n_classes),
        local_state(!ext_adam, n_classes),
    );
    let (mut lm_w, mut ls_w, mut lt_w) = (
        local_state(need_adam, n_classes * n_features),
        local_state(need_adam, n_classes * n_features),
        local_state(need_adam, n_classes * n_features),
    );
    let (mut lm_v, mut ls_v, mut lt_v) = (
        local_state(need_adam, n_classes * v_len),
        local_state(need_adam, n_classes * v_len),
        local_state(need_adam, n_classes * v_len),
    );
    let (mut lz_w0, mut ln_w0) = (local_state(!ext_ftrl, n_classes), local_state(!ext_ftrl, n_classes));
    let (mut lz_w, mut ln_w) = (
        local_state(need_ftrl, n_classes * n_features),
        local_state(need_ftrl, n_classes * n_features),
    );
    let (mut lz_v, mut ln_v) = (
        local_state(need_ftrl, n_classes * v_len),
        local_state(need_ftrl, n_classes * v_len),
    );
    let (acc_w0_s, acc_w_s, acc_v_s) = match state.as_mut() {
        Some(t) => (t.0.as_slice_mut()?, t.1.as_slice_mut()?, t.2.as_slice_mut()?),
        None => (lacc_w0.as_mut_slice(), lacc_w.as_mut_slice(), lacc_v.as_mut_slice()),
    };
    let (m_w0_s, s_w0_s, t_w0_s, m_w_s, s_w_s, t_w_s, m_v_s, s_v_s, t_v_s) =
        match adam_state.as_mut() {
            Some(t) => (
                t.0.as_slice_mut()?, t.1.as_slice_mut()?, t.2.as_slice_mut()?,
                t.3.as_slice_mut()?, t.4.as_slice_mut()?, t.5.as_slice_mut()?,
                t.6.as_slice_mut()?, t.7.as_slice_mut()?, t.8.as_slice_mut()?,
            ),
            None => (
                lm_w0.as_mut_slice(), ls_w0.as_mut_slice(), lt_w0.as_mut_slice(),
                lm_w.as_mut_slice(), ls_w.as_mut_slice(), lt_w.as_mut_slice(),
                lm_v.as_mut_slice(), ls_v.as_mut_slice(), lt_v.as_mut_slice(),
            ),
        };
    let (z_w0_s, n_w0_s, z_w_s, n_w_s, z_v_s, n_v_s) = match ftrl_state.as_mut() {
        Some(t) => (
            t.0.as_slice_mut()?, t.1.as_slice_mut()?, t.2.as_slice_mut()?,
            t.3.as_slice_mut()?, t.4.as_slice_mut()?, t.5.as_slice_mut()?,
        ),
        None => (
            lz_w0.as_mut_slice(), ln_w0.as_mut_slice(), lz_w.as_mut_slice(),
            ln_w.as_mut_slice(), lz_v.as_mut_slice(), ln_v.as_mut_slice(),
        ),
    };
    if ext_state
        && (acc_w0_s.len() != n_classes
            || acc_w_s.len() != n_classes * n_features
            || acc_v_s.len() != n_classes * v_len)
    {
        return Err(val_err("state shapes must match (w0, w, V)".to_string()));
    }
    if ext_adam
        && (m_w0_s.len() != n_classes || s_w0_s.len() != n_classes || t_w0_s.len() != n_classes
            || m_w_s.len() != n_classes * n_features || s_w_s.len() != n_classes * n_features
            || t_w_s.len() != n_classes * n_features
            || m_v_s.len() != n_classes * v_len || s_v_s.len() != n_classes * v_len
            || t_v_s.len() != n_classes * v_len)
    {
        return Err(val_err("adam_state shapes must match (w0, w, V)".to_string()));
    }
    if ext_ftrl
        && (z_w0_s.len() != n_classes || n_w0_s.len() != n_classes
            || z_w_s.len() != n_classes * n_features || n_w_s.len() != n_classes * n_features
            || z_v_s.len() != n_classes * v_len || n_v_s.len() != n_classes * v_len)
    {
        return Err(val_err("ftrl_state shapes must match (w0, w, V)".to_string()));
    }
    let st = McState {
        acc_w0: acc_w0_s, acc_w: acc_w_s, acc_v: acc_v_s,
        m_w0: m_w0_s, s_w0: s_w0_s, t_w0: t_w0_s,
        m_w: m_w_s, s_w: s_w_s, t_w: t_w_s,
        m_v: m_v_s, s_v: s_v_s, t_v: t_v_s,
        z_w0: z_w0_s, n_w0: n_w0_s,
        z_w: z_w_s, n_w: n_w_s, z_v: z_v_s, n_v: n_v_s,
    };
    let out = py.allow_threads(|| -> Result<(), String> {
        let csr = CsrView::new(indptr_s, indices_s, data_s, n_features)?;
        check_fit_args_mc(&csr, y_s, n_classes, sw_s, &ro_shape, ro)?;
        ffm::fit_multiclass_csr(
            &csr, y_s, sw_s, field_ids_s, w0_s, w_s, v_s, st, n_classes, n_features, n_fields, k,
            opt, learning_rate, l1_linear, l2_linear, l1_factors, l2_factors, label_smoothing,
            ro_shape[1], batch_size, ro,
        );
        Ok(())
    });
    out.map_err(val_err)
}

/// FwFM prediction over a dense matrix (docs/math_spec_fwfm.md); `v` is
/// FM-shaped (n_features, k), `r` is (n_fields, n_fields), upper triangle used.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fwfm_predict_dense<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    field_ids: PyReadonlyArray1<'py, i64>,
    w0: f64,
    w: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
    r: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (n_rows, n_features) = (x.shape()[0], x.shape()[1]);
    let k = v.shape()[1];
    let n_fields = r.shape()[0];
    if r.shape()[1] != n_fields {
        return Err(val_err(format!("R must be square, got {:?}", r.shape())));
    }
    let x = x.as_slice()?;
    let w = w.as_slice()?;
    check_w_v_fm(n_features, w, v.shape())?;
    let field_ids = field_ids.as_slice()?;
    data::check_field_ids(field_ids, n_features, n_fields).map_err(val_err)?;
    let v = v.as_slice()?;
    let r = r.as_slice()?;
    let out = py.allow_threads(|| {
        fwfm::predict_dense(x, n_rows, n_features, field_ids, w0, w, v, r, n_fields, k)
    });
    Ok(out.into_pyarray(py))
}

/// FwFM prediction over a CSR matrix; see fwfm_predict_dense.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fwfm_predict_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_features: usize,
    field_ids: PyReadonlyArray1<'py, i64>,
    w0: f64,
    w: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
    r: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let k = v.shape()[1];
    let n_fields = r.shape()[0];
    if r.shape()[1] != n_fields {
        return Err(val_err(format!("R must be square, got {:?}", r.shape())));
    }
    let (indptr_s, indices_s, data_s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let w = w.as_slice()?;
    check_w_v_fm(n_features, w, v.shape())?;
    let field_ids = field_ids.as_slice()?;
    data::check_field_ids(field_ids, n_features, n_fields).map_err(val_err)?;
    let v = v.as_slice()?;
    let r = r.as_slice()?;
    let out = py.allow_threads(|| -> Result<Vec<f64>, String> {
        let csr = CsrView::new(indptr_s, indices_s, data_s, n_features)?;
        Ok(fwfm::predict_csr(&csr, field_ids, w0, w, v, r, n_fields, k))
    });
    Ok(out.map_err(val_err)?.into_pyarray(py))
}

/// Train an FwFM in place (w, v, r, acc_*, adam/ftrl arrays mutated; scalar
/// returns as in fm_fit_csr). `acc_r` is the R group's AdaGrad accumulator;
/// `adam_state`/`ftrl_state` append the R slots (docs/math_spec_fwfm.md).
/// Serial; see fwfm::fit_csr.
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, data, n_features, y, sample_weight, field_ids, w0, acc_w0, w, v, r,
    acc_w, acc_v, acc_r, loss, optimizer, learning_rate, l2_linear, l2_factors, beta_1,
    beta_2, epsilon, row_orders, batch_size, l1_linear, l1_factors, ftrl_beta,
    adam_state=None, ftrl_state=None,
))]
#[allow(clippy::too_many_arguments)]
fn fwfm_fit_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_features: usize,
    y: PyReadonlyArray1<'py, f64>,
    sample_weight: PyReadonlyArray1<'py, f64>,
    field_ids: PyReadonlyArray1<'py, i64>,
    w0: f64,
    acc_w0: f64,
    mut w: PyReadwriteArray1<'py, f64>,
    mut v: PyReadwriteArray2<'py, f64>,
    mut r: PyReadwriteArray2<'py, f64>,
    mut acc_w: PyReadwriteArray1<'py, f64>,
    mut acc_v: PyReadwriteArray2<'py, f64>,
    mut acc_r: PyReadwriteArray2<'py, f64>,
    loss: &str,
    optimizer: &str,
    learning_rate: f64,
    l2_linear: f64,
    l2_factors: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    row_orders: PyReadonlyArray2<'py, i64>,
    batch_size: usize,
    l1_linear: f64,
    l1_factors: f64,
    ftrl_beta: f64,
    mut adam_state: Option<FwfmAdamArg<'py>>,
    mut ftrl_state: Option<FwfmFtrlArg<'py>>,
) -> PyResult<FitScalars> {
    let loss = parse_loss(loss)?;
    let opt = parse_optimizer(optimizer, beta_1, beta_2, epsilon, ftrl_beta)?;
    if batch_size == 0 {
        return Err(val_err("batch_size must be >= 1".to_string()));
    }
    let k = v.shape()[1];
    let v_rows = v.shape()[0];
    let n_fields = r.shape()[0];
    if r.shape()[1] != n_fields {
        return Err(val_err(format!("R must be square, got {:?}", r.shape())));
    }
    let ro_shape = [row_orders.shape()[0], row_orders.shape()[1]];
    let (indptr_s, indices_s, data_s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let y_s = y.as_slice()?;
    let sw_s = sample_weight.as_slice()?;
    let ro = row_orders.as_slice()?;
    let field_ids_s = field_ids.as_slice()?;
    let w_s = w.as_slice_mut()?;
    if w_s.len() != n_features || v_rows != n_features {
        return Err(val_err(format!(
            "shape mismatch: n_features={n_features}, w={}, V rows={v_rows}",
            w_s.len()
        )));
    }
    data::check_field_ids(field_ids_s, n_features, n_fields).map_err(val_err)?;
    let v_s = v.as_slice_mut()?;
    let r_s = r.as_slice_mut()?;
    let acc_w_s = acc_w.as_slice_mut()?;
    let acc_v_s = acc_v.as_slice_mut()?;
    let acc_r_s = acc_r.as_slice_mut()?;
    if acc_w_s.len() != w_s.len() || acc_v_s.len() != v_s.len() || acc_r_s.len() != r_s.len() {
        return Err(val_err("accumulator shapes must match w, V and R".to_string()));
    }
    let v_len = v_s.len();
    let r_len = r_s.len();
    let adam_on = matches!(opt, Optimizer::Adam { .. });
    let ftrl_on = matches!(opt, Optimizer::Ftrl { .. });
    if adam_state.is_some() && !adam_on {
        return Err(val_err("adam_state is only valid for optimizer='adam'".to_string()));
    }
    if ftrl_state.is_some() && !ftrl_on {
        return Err(val_err("ftrl_state is only valid for optimizer='ftrl'".to_string()));
    }
    let (mut m_w0, mut s_w0, mut t_w0) =
        adam_state.as_ref().map_or((0.0, 0.0, 0.0), |t| (t.0, t.1, t.2));
    let (mut z_w0, mut n_w0) = ftrl_state.as_ref().map_or((0.0, 0.0), |t| (t.0, t.1));
    let ext_adam = adam_state.is_some();
    let ext_ftrl = ftrl_state.is_some();
    let need_adam = adam_on && !ext_adam;
    let need_ftrl = ftrl_on && !ext_ftrl;
    let (mut lm_w, mut ls_w, mut lt_w) = (
        local_state(need_adam, n_features),
        local_state(need_adam, n_features),
        local_state(need_adam, n_features),
    );
    let (mut lm_v, mut ls_v, mut lt_v) = (
        local_state(need_adam, v_len),
        local_state(need_adam, v_len),
        local_state(need_adam, v_len),
    );
    let (mut lm_r, mut ls_r, mut lt_r) = (
        local_state(need_adam, r_len),
        local_state(need_adam, r_len),
        local_state(need_adam, r_len),
    );
    let (mut lz_w, mut ln_w) = (local_state(need_ftrl, n_features), local_state(need_ftrl, n_features));
    let (mut lz_v, mut ln_v) = (local_state(need_ftrl, v_len), local_state(need_ftrl, v_len));
    let (mut lz_r, mut ln_r) = (local_state(need_ftrl, r_len), local_state(need_ftrl, r_len));
    let (m_w_s, s_w_s, t_w_s, m_v_s, s_v_s, t_v_s, m_r_s, s_r_s, t_r_s) =
        match adam_state.as_mut() {
            Some(t) => (
                t.3.as_slice_mut()?, t.4.as_slice_mut()?, t.5.as_slice_mut()?,
                t.6.as_slice_mut()?, t.7.as_slice_mut()?, t.8.as_slice_mut()?,
                t.9.as_slice_mut()?, t.10.as_slice_mut()?, t.11.as_slice_mut()?,
            ),
            None => (
                lm_w.as_mut_slice(), ls_w.as_mut_slice(), lt_w.as_mut_slice(),
                lm_v.as_mut_slice(), ls_v.as_mut_slice(), lt_v.as_mut_slice(),
                lm_r.as_mut_slice(), ls_r.as_mut_slice(), lt_r.as_mut_slice(),
            ),
        };
    let (z_w_s, n_w_s, z_v_s, n_v_s, z_r_s, n_r_s) = match ftrl_state.as_mut() {
        Some(t) => (
            t.2.as_slice_mut()?, t.3.as_slice_mut()?, t.4.as_slice_mut()?,
            t.5.as_slice_mut()?, t.6.as_slice_mut()?, t.7.as_slice_mut()?,
        ),
        None => (
            lz_w.as_mut_slice(), ln_w.as_mut_slice(), lz_v.as_mut_slice(),
            ln_v.as_mut_slice(), lz_r.as_mut_slice(), ln_r.as_mut_slice(),
        ),
    };
    if ext_adam
        && (m_w_s.len() != n_features || s_w_s.len() != n_features || t_w_s.len() != n_features
            || m_v_s.len() != v_len || s_v_s.len() != v_len || t_v_s.len() != v_len
            || m_r_s.len() != r_len || s_r_s.len() != r_len || t_r_s.len() != r_len)
    {
        return Err(val_err("adam_state shapes must match w, V and R".to_string()));
    }
    if ext_ftrl
        && (z_w_s.len() != n_features || n_w_s.len() != n_features
            || z_v_s.len() != v_len || n_v_s.len() != v_len
            || z_r_s.len() != r_len || n_r_s.len() != r_len)
    {
        return Err(val_err("ftrl_state shapes must match w, V and R".to_string()));
    }
    let adam_view = AdamStateMut {
        m_w0: &mut m_w0, s_w0: &mut s_w0, t_w0: &mut t_w0,
        m_w: m_w_s, s_w: s_w_s, t_w: t_w_s, m_v: m_v_s, s_v: s_v_s, t_v: t_v_s,
    };
    let ftrl_view = FtrlStateMut {
        z_w0: &mut z_w0, n_w0: &mut n_w0,
        z_w: z_w_s, n_w: n_w_s, z_v: z_v_s, n_v: n_v_s,
    };
    let r_view = GroupStateMut {
        acc: acc_r_s, m: m_r_s, s: s_r_s, t: t_r_s, z: z_r_s, n: n_r_s,
    };
    let out = py.allow_threads(|| -> Result<(f64, f64), String> {
        let csr = CsrView::new(indptr_s, indices_s, data_s, n_features)?;
        check_fit_args(&csr, y_s, sw_s, &ro_shape, ro)?;
        let mut w0 = w0;
        let mut acc_w0 = acc_w0;
        fwfm::fit_csr(
            &csr, y_s, sw_s, field_ids_s, &mut w0, w_s, v_s, r_s, &mut acc_w0, acc_w_s, acc_v_s,
            adam_view, ftrl_view, r_view, n_fields, k, loss, opt,
            learning_rate, l1_linear, l2_linear, l1_factors, l2_factors,
            ro_shape[1], batch_size, ro,
        );
        Ok((w0, acc_w0))
    });
    let (w0, acc_w0) = out.map_err(val_err)?;
    Ok((w0, acc_w0, (m_w0, s_w0, t_w0), (z_w0, n_w0)))
}

/// Train a multiclass (softmax) FwFM in place; w0 (C,), w (C, n),
/// v (C, n, k), r (C, F, F) all mutated. `state` = (acc_w0, acc_w, acc_V,
/// acc_R); `adam_state`/`ftrl_state` append the R slots. Serial; see
/// fwfm::fit_multiclass_csr.
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, data, n_features, y, sample_weight, field_ids, w0, w, v, r,
    optimizer, learning_rate, l2_linear, l2_factors, label_smoothing, beta_1, beta_2,
    epsilon, row_orders, batch_size, l1_linear, l1_factors, ftrl_beta,
    state=None, adam_state=None, ftrl_state=None,
))]
#[allow(clippy::too_many_arguments)]
fn fwfm_fit_multiclass_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_features: usize,
    y: PyReadonlyArray1<'py, i64>,
    sample_weight: PyReadonlyArray1<'py, f64>,
    field_ids: PyReadonlyArray1<'py, i64>,
    mut w0: PyReadwriteArray1<'py, f64>,
    mut w: PyReadwriteArray2<'py, f64>,
    mut v: PyReadwriteArray3<'py, f64>,
    mut r: PyReadwriteArray3<'py, f64>,
    optimizer: &str,
    learning_rate: f64,
    l2_linear: f64,
    l2_factors: f64,
    label_smoothing: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    row_orders: PyReadonlyArray2<'py, i64>,
    batch_size: usize,
    l1_linear: f64,
    l1_factors: f64,
    ftrl_beta: f64,
    mut state: Option<FwfmMcStateArg<'py>>,
    mut adam_state: Option<FwfmMcAdamArg<'py>>,
    mut ftrl_state: Option<FwfmMcFtrlArg<'py>>,
) -> PyResult<()> {
    let opt = parse_optimizer(optimizer, beta_1, beta_2, epsilon, ftrl_beta)?;
    if batch_size == 0 {
        return Err(val_err("batch_size must be >= 1".to_string()));
    }
    let (n_classes, v_n, k) = (v.shape()[0], v.shape()[1], v.shape()[2]);
    let n_fields = r.shape()[1];
    if r.shape()[0] != n_classes || r.shape()[2] != n_fields {
        return Err(val_err(format!(
            "R must be (n_classes, n_fields, n_fields), got {:?}",
            r.shape()
        )));
    }
    let w_shape = [w.shape()[0], w.shape()[1]];
    let ro_shape = [row_orders.shape()[0], row_orders.shape()[1]];
    let (indptr_s, indices_s, data_s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let y_s = y.as_slice()?;
    let sw_s = sample_weight.as_slice()?;
    let ro = row_orders.as_slice()?;
    let field_ids_s = field_ids.as_slice()?;
    let w0_s = w0.as_slice_mut()?;
    let w_s = w.as_slice_mut()?;
    let v_s = v.as_slice_mut()?;
    let r_s = r.as_slice_mut()?;
    if w0_s.len() != n_classes || w_shape != [n_classes, n_features] || v_n != n_features {
        return Err(val_err(format!(
            "shape mismatch: n_classes={n_classes}, n_features={n_features}, \
             w0={}, w={w_shape:?}, V dims=[{n_classes}, {v_n}, {k}]",
            w0_s.len()
        )));
    }
    data::check_field_ids(field_ids_s, n_features, n_fields).map_err(val_err)?;
    let v_len = n_features * k; // per-class factor entries
    let r_len = n_fields * n_fields; // per-class R entries
    let adam_on = matches!(opt, Optimizer::Adam { .. });
    let ftrl_on = matches!(opt, Optimizer::Ftrl { .. });
    if adam_state.is_some() && !adam_on {
        return Err(val_err("adam_state is only valid for optimizer='adam'".to_string()));
    }
    if ftrl_state.is_some() && !ftrl_on {
        return Err(val_err("ftrl_state is only valid for optimizer='ftrl'".to_string()));
    }
    let (ext_state, ext_adam, ext_ftrl) = (state.is_some(), adam_state.is_some(), ftrl_state.is_some());
    let (need_adam, need_ftrl) = (adam_on && !ext_adam, ftrl_on && !ext_ftrl);
    let mut lacc_w0 = local_state(!ext_state, n_classes);
    let mut lacc_w = local_state(!ext_state, n_classes * n_features);
    let mut lacc_v = local_state(!ext_state, n_classes * v_len);
    let mut lacc_r = local_state(!ext_state, n_classes * r_len);
    let (mut lm_w0, mut ls_w0, mut lt_w0) = (
        local_state(!ext_adam, n_classes),
        local_state(!ext_adam, n_classes),
        local_state(!ext_adam, n_classes),
    );
    let (mut lm_w, mut ls_w, mut lt_w) = (
        local_state(need_adam, n_classes * n_features),
        local_state(need_adam, n_classes * n_features),
        local_state(need_adam, n_classes * n_features),
    );
    let (mut lm_v, mut ls_v, mut lt_v) = (
        local_state(need_adam, n_classes * v_len),
        local_state(need_adam, n_classes * v_len),
        local_state(need_adam, n_classes * v_len),
    );
    let (mut lm_r, mut ls_r, mut lt_r) = (
        local_state(need_adam, n_classes * r_len),
        local_state(need_adam, n_classes * r_len),
        local_state(need_adam, n_classes * r_len),
    );
    let (mut lz_w0, mut ln_w0) = (local_state(!ext_ftrl, n_classes), local_state(!ext_ftrl, n_classes));
    let (mut lz_w, mut ln_w) = (
        local_state(need_ftrl, n_classes * n_features),
        local_state(need_ftrl, n_classes * n_features),
    );
    let (mut lz_v, mut ln_v) = (
        local_state(need_ftrl, n_classes * v_len),
        local_state(need_ftrl, n_classes * v_len),
    );
    let (mut lz_r, mut ln_r) = (
        local_state(need_ftrl, n_classes * r_len),
        local_state(need_ftrl, n_classes * r_len),
    );
    let (acc_w0_s, acc_w_s, acc_v_s, acc_r_s) = match state.as_mut() {
        Some(t) => (
            t.0.as_slice_mut()?, t.1.as_slice_mut()?, t.2.as_slice_mut()?, t.3.as_slice_mut()?,
        ),
        None => (
            lacc_w0.as_mut_slice(), lacc_w.as_mut_slice(), lacc_v.as_mut_slice(),
            lacc_r.as_mut_slice(),
        ),
    };
    let (m_w0_s, s_w0_s, t_w0_s, m_w_s, s_w_s, t_w_s, m_v_s, s_v_s, t_v_s, m_r_s, s_r_s, t_r_s) =
        match adam_state.as_mut() {
            Some(t) => (
                t.0.as_slice_mut()?, t.1.as_slice_mut()?, t.2.as_slice_mut()?,
                t.3.as_slice_mut()?, t.4.as_slice_mut()?, t.5.as_slice_mut()?,
                t.6.as_slice_mut()?, t.7.as_slice_mut()?, t.8.as_slice_mut()?,
                t.9.as_slice_mut()?, t.10.as_slice_mut()?, t.11.as_slice_mut()?,
            ),
            None => (
                lm_w0.as_mut_slice(), ls_w0.as_mut_slice(), lt_w0.as_mut_slice(),
                lm_w.as_mut_slice(), ls_w.as_mut_slice(), lt_w.as_mut_slice(),
                lm_v.as_mut_slice(), ls_v.as_mut_slice(), lt_v.as_mut_slice(),
                lm_r.as_mut_slice(), ls_r.as_mut_slice(), lt_r.as_mut_slice(),
            ),
        };
    let (z_w0_s, n_w0_s, z_w_s, n_w_s, z_v_s, n_v_s, z_r_s, n_r_s) = match ftrl_state.as_mut() {
        Some(t) => (
            t.0.as_slice_mut()?, t.1.as_slice_mut()?, t.2.as_slice_mut()?, t.3.as_slice_mut()?,
            t.4.as_slice_mut()?, t.5.as_slice_mut()?, t.6.as_slice_mut()?, t.7.as_slice_mut()?,
        ),
        None => (
            lz_w0.as_mut_slice(), ln_w0.as_mut_slice(), lz_w.as_mut_slice(), ln_w.as_mut_slice(),
            lz_v.as_mut_slice(), ln_v.as_mut_slice(), lz_r.as_mut_slice(), ln_r.as_mut_slice(),
        ),
    };
    if ext_state
        && (acc_w0_s.len() != n_classes
            || acc_w_s.len() != n_classes * n_features
            || acc_v_s.len() != n_classes * v_len
            || acc_r_s.len() != n_classes * r_len)
    {
        return Err(val_err("state shapes must match (w0, w, V, R)".to_string()));
    }
    if ext_adam
        && (m_w0_s.len() != n_classes || s_w0_s.len() != n_classes || t_w0_s.len() != n_classes
            || m_w_s.len() != n_classes * n_features || s_w_s.len() != n_classes * n_features
            || t_w_s.len() != n_classes * n_features
            || m_v_s.len() != n_classes * v_len || s_v_s.len() != n_classes * v_len
            || t_v_s.len() != n_classes * v_len
            || m_r_s.len() != n_classes * r_len || s_r_s.len() != n_classes * r_len
            || t_r_s.len() != n_classes * r_len)
    {
        return Err(val_err("adam_state shapes must match (w0, w, V, R)".to_string()));
    }
    if ext_ftrl
        && (z_w0_s.len() != n_classes || n_w0_s.len() != n_classes
            || z_w_s.len() != n_classes * n_features || n_w_s.len() != n_classes * n_features
            || z_v_s.len() != n_classes * v_len || n_v_s.len() != n_classes * v_len
            || z_r_s.len() != n_classes * r_len || n_r_s.len() != n_classes * r_len)
    {
        return Err(val_err("ftrl_state shapes must match (w0, w, V, R)".to_string()));
    }
    let st = McState {
        acc_w0: acc_w0_s, acc_w: acc_w_s, acc_v: acc_v_s,
        m_w0: m_w0_s, s_w0: s_w0_s, t_w0: t_w0_s,
        m_w: m_w_s, s_w: s_w_s, t_w: t_w_s,
        m_v: m_v_s, s_v: s_v_s, t_v: t_v_s,
        z_w0: z_w0_s, n_w0: n_w0_s,
        z_w: z_w_s, n_w: n_w_s, z_v: z_v_s, n_v: n_v_s,
    };
    let rst = McGroupState {
        acc: acc_r_s, m: m_r_s, s: s_r_s, t: t_r_s, z: z_r_s, n: n_r_s,
    };
    let out = py.allow_threads(|| -> Result<(), String> {
        let csr = CsrView::new(indptr_s, indices_s, data_s, n_features)?;
        check_fit_args_mc(&csr, y_s, n_classes, sw_s, &ro_shape, ro)?;
        fwfm::fit_multiclass_csr(
            &csr, y_s, sw_s, field_ids_s, w0_s, w_s, v_s, r_s, st, rst,
            n_classes, n_features, n_fields, k, opt, learning_rate,
            l1_linear, l2_linear, l1_factors, l2_factors, label_smoothing,
            ro_shape[1], batch_size, ro,
        );
        Ok(())
    });
    out.map_err(val_err)
}

/// True when the extension was built with the `cuda-backend` feature AND a
/// CUDA driver + device are present at runtime (docs/gpu_backend_plan.md).
/// Always registered so `_backend.has_cuda()` can call it unconditionally;
/// CPU-only builds simply return false.
#[pyfunction]
fn has_cuda() -> bool {
    #[cfg(all(feature = "cuda-backend", not(target_os = "macos")))]
    {
        cuda::available()
    }
    #[cfg(not(all(feature = "cuda-backend", not(target_os = "macos"))))]
    {
        false
    }
}

/// FM CSR prediction on CUDA (docs/gpu_backend_plan.md milestone 1). Same
/// input contract as fm_predict_fast_csr; CSR structure is validated on the
/// CPU first, CUDA errors become RuntimeError. Registered only in
/// cuda-backend builds; `_backend` guards calls with `has_cuda()`.
#[cfg(all(feature = "cuda-backend", not(target_os = "macos")))]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fm_predict_cuda_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_features: usize,
    w0: f64,
    w: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let k = v.shape()[1];
    let (indptr_s, indices_s, data_s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let w_s = w.as_slice()?;
    check_w_v_fm(n_features, w_s, v.shape())?;
    let v_s = v.as_slice()?;
    let out = py.allow_threads(|| -> Result<Vec<f64>, String> {
        CsrView::new(indptr_s, indices_s, data_s, n_features)?;
        cuda::fm::predict_csr(indptr_s, indices_s, data_s, w_s, v_s, w0, k)
    });
    Ok(out
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
        .into_pyarray(py))
}

/// FFM CSR prediction on CUDA (docs/gpu_backend_plan.md milestone 2). Same
/// input contract as ffm_predict_csr; CSR structure and field_ids are
/// validated on the CPU first, CUDA errors become RuntimeError. Registered
/// only in cuda-backend builds; `_backend` guards calls with `has_cuda()`.
#[cfg(all(feature = "cuda-backend", not(target_os = "macos")))]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ffm_predict_cuda_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_features: usize,
    field_ids: PyReadonlyArray1<'py, i64>,
    w0: f64,
    w: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (n_fields, k) = (v.shape()[1], v.shape()[2]);
    let (indptr_s, indices_s, data_s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let w_s = w.as_slice()?;
    check_w_v_fm(n_features, w_s, v.shape())?;
    let field_ids_s = field_ids.as_slice()?;
    data::check_field_ids(field_ids_s, n_features, n_fields).map_err(val_err)?;
    let v_s = v.as_slice()?;
    let out = py.allow_threads(|| -> Result<Vec<f64>, String> {
        CsrView::new(indptr_s, indices_s, data_s, n_features)?;
        cuda::ffm::predict_csr(indptr_s, indices_s, data_s, field_ids_s, w_s, v_s, w0, n_fields, k)
    });
    Ok(out
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
        .into_pyarray(py))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(has_cuda, m)?)?;
    #[cfg(all(feature = "cuda-backend", not(target_os = "macos")))]
    m.add_function(wrap_pyfunction!(fm_predict_cuda_csr, m)?)?;
    #[cfg(all(feature = "cuda-backend", not(target_os = "macos")))]
    m.add_function(wrap_pyfunction!(ffm_predict_cuda_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fm_predict_fast_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fm_predict_fast_csr, m)?)?;
    m.add_function(wrap_pyfunction!(ffm_predict_dense, m)?)?;
    m.add_function(wrap_pyfunction!(ffm_predict_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fm_fit_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fm_fit_multiclass_csr, m)?)?;
    m.add_function(wrap_pyfunction!(ffm_fit_csr, m)?)?;
    m.add_function(wrap_pyfunction!(ffm_fit_multiclass_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fwfm_predict_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fwfm_predict_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fwfm_fit_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fwfm_fit_multiclass_csr, m)?)?;
    Ok(())
}
