//! PyO3 bindings for the modern_fm Rust backend, exposed as `modern_fm._rust`.
//!
//! Private API — called only through `modern_fm._backend`. Contract
//! (enforced by the Python wrapper, validated again here):
//! float64 C-contiguous arrays, int64 CSR indices/indptr and field_ids.
//! Shapes follow docs/math_spec.md: w (n_features,), FM V (n_features, k),
//! FFM V (n_features, n_fields, k).

use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod data;
mod ffm;
mod fm;

use data::CsrView;

fn val_err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
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

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fm_predict_fast_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fm_predict_fast_csr, m)?)?;
    m.add_function(wrap_pyfunction!(ffm_predict_dense, m)?)?;
    m.add_function(wrap_pyfunction!(ffm_predict_csr, m)?)?;
    Ok(())
}
