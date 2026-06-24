//! PyO3 bindings for the modern_fm Rust backend, exposed as `modern_fm._rust`.
//!
//! Private API — called only through `modern_fm._backend`. Contract
//! (enforced by the Python wrapper, validated again here):
//! float64 C-contiguous arrays, int64 CSR indices/indptr and field_ids.
//! Shapes follow docs/math_spec.md: w (n_features,), FM V (n_features, k),
//! FFM V (n_features, n_fields, k).

use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod data;
mod ffm;
mod fm;
mod optimizer;

use data::CsrView;
use optimizer::{Loss, Optimizer};

fn val_err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

fn parse_loss(s: &str) -> PyResult<Loss> {
    match s {
        "logistic" => Ok(Loss::Logistic),
        "squared" => Ok(Loss::Squared),
        _ => Err(val_err(format!("unknown loss {s:?}"))),
    }
}

fn parse_optimizer(s: &str) -> PyResult<Optimizer> {
    match s {
        "sgd" => Ok(Optimizer::Sgd),
        "adagrad" => Ok(Optimizer::Adagrad),
        _ => Err(val_err(format!("unknown optimizer {s:?}"))),
    }
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

/// Train an FM in place (w, v, acc_* mutated; new (w0, acc_w0) returned).
/// AdaGrad accumulators are passed in/out so the caller can drive epochs one
/// at a time (early stopping); a single all-epochs call passes zeros. See
/// fm::fit_csr and docs/optimization_spec.md.
#[pyfunction]
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
    row_orders: PyReadonlyArray2<'py, i64>,
) -> PyResult<(f64, f64)> {
    let loss = parse_loss(loss)?;
    let opt = parse_optimizer(optimizer)?;
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
    let out = py.allow_threads(|| -> Result<(f64, f64), String> {
        let csr = CsrView::new(indptr_s, indices_s, data_s, n_features)?;
        check_fit_args(&csr, y_s, sw_s, &ro_shape, ro)?;
        let mut w0 = w0;
        let mut acc_w0 = acc_w0;
        fm::fit_csr(
            &csr, y_s, sw_s, &mut w0, w_s, v_s, &mut acc_w0, acc_w_s, acc_v_s, k, loss, opt,
            learning_rate, l2_linear, l2_factors, ro,
        );
        Ok((w0, acc_w0))
    });
    out.map_err(val_err)
}

/// Train an FFM (logistic loss) in place (w, v mutated; new w0 returned).
/// batch_size=1, single-threaded; see ffm::fit_csr.
#[pyfunction]
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
    optimizer: &str,
    learning_rate: f64,
    l2_linear: f64,
    l2_factors: f64,
    row_orders: PyReadonlyArray2<'py, i64>,
) -> PyResult<(f64, f64)> {
    let opt = parse_optimizer(optimizer)?;
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
    let out = py.allow_threads(|| -> Result<(f64, f64), String> {
        let csr = CsrView::new(indptr_s, indices_s, data_s, n_features)?;
        check_fit_args(&csr, y_s, sw_s, &ro_shape, ro)?;
        let mut w0 = w0;
        let mut acc_w0 = acc_w0;
        ffm::fit_csr(
            &csr, y_s, sw_s, field_ids_s, &mut w0, w_s, v_s, &mut acc_w0, acc_w_s, acc_v_s,
            n_fields, k, opt, learning_rate, l2_linear, l2_factors, ro,
        );
        Ok((w0, acc_w0))
    });
    out.map_err(val_err)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fm_predict_fast_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fm_predict_fast_csr, m)?)?;
    m.add_function(wrap_pyfunction!(ffm_predict_dense, m)?)?;
    m.add_function(wrap_pyfunction!(ffm_predict_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fm_fit_csr, m)?)?;
    m.add_function(wrap_pyfunction!(ffm_fit_csr, m)?)?;
    Ok(())
}
