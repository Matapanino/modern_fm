//! FM prediction kernels (docs/math_spec.md).
//!
//! y_hat = w0 + sum_i w_i x_i
//!       + 0.5 * sum_f [(sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2]

use crate::data::{dense_row_nonzeros, CsrView};
use crate::optimizer::{apply_update, loss_grad, Loss, Optimizer};

/// Score one row given its nonzero (index, value) pairs.
/// `v` is row-major (n_features, k).
#[allow(clippy::too_many_arguments)]
fn fm_score_row(
    indices: &[usize],
    values: &[f64],
    w0: f64,
    w: &[f64],
    v: &[f64],
    k: usize,
    sum: &mut [f64],
    sum_sq: &mut [f64],
) -> f64 {
    let mut s = w0;
    sum.fill(0.0);
    sum_sq.fill(0.0);
    for (&i, &x) in indices.iter().zip(values) {
        s += w[i] * x;
        let vi = &v[i * k..(i + 1) * k];
        for f in 0..k {
            let vx = vi[f] * x;
            sum[f] += vx;
            sum_sq[f] += vx * vx;
        }
    }
    let pairwise: f64 = (0..k).map(|f| sum[f] * sum[f] - sum_sq[f]).sum();
    s + 0.5 * pairwise
}

/// FM prediction over a dense C-contiguous (n_rows, n_features) matrix.
pub fn predict_dense(x: &[f64], n_rows: usize, n_features: usize, w0: f64, w: &[f64], v: &[f64], k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_rows);
    let mut sum = vec![0.0; k];
    let mut sum_sq = vec![0.0; k];
    let mut idx_buf = Vec::new();
    let mut val_buf = Vec::new();
    for r in 0..n_rows {
        let row = &x[r * n_features..(r + 1) * n_features];
        dense_row_nonzeros(row, &mut idx_buf, &mut val_buf);
        out.push(fm_score_row(&idx_buf, &val_buf, w0, w, v, k, &mut sum, &mut sum_sq));
    }
    out
}

/// FM prediction over a CSR matrix.
pub fn predict_csr(csr: &CsrView, w0: f64, w: &[f64], v: &[f64], k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(csr.n_rows());
    let mut sum = vec![0.0; k];
    let mut sum_sq = vec![0.0; k];
    let mut idx_buf = Vec::new();
    for r in 0..csr.n_rows() {
        let (indices, values) = csr.row(r);
        idx_buf.clear();
        idx_buf.extend(indices.iter().map(|&i| i as usize));
        out.push(fm_score_row(&idx_buf, values, w0, w, v, k, &mut sum, &mut sum_sq));
    }
    out
}

/// Train an FM in place with batch_size=1 (docs/optimization_spec.md).
///
/// Mirrors `_reference_train.fm_fit_reference`: per-row gradients from
/// pre-update parameters, lazy L2, update order w0 -> w -> V. Rows are
/// visited in the order given by `row_orders` (epochs * n_rows entries,
/// validated by the caller). For logistic loss, y must be 0/1.
#[allow(clippy::too_many_arguments)]
pub fn fit_csr(
    csr: &CsrView,
    y: &[f64],
    w0: &mut f64,
    w: &mut [f64],
    v: &mut [f64],
    k: usize,
    loss: Loss,
    opt: Optimizer,
    lr: f64,
    l2_linear: f64,
    l2_factors: f64,
    row_orders: &[i64],
) {
    let mut acc_w0 = 0.0;
    let mut acc_w = vec![0.0; w.len()];
    let mut acc_v = vec![0.0; v.len()];
    let mut cache = vec![0.0; k];
    for &r in row_orders {
        let (indices, values) = csr.row(r as usize);
        // score + factor cache from pre-update parameters
        cache.fill(0.0);
        let mut s = *w0;
        let mut sq = 0.0;
        for (&i, &x) in indices.iter().zip(values) {
            let i = i as usize;
            s += w[i] * x;
            let vi = &v[i * k..(i + 1) * k];
            for f in 0..k {
                let vx = vi[f] * x;
                cache[f] += vx;
                sq += vx * vx;
            }
        }
        let dot: f64 = cache.iter().map(|c| c * c).sum();
        s += 0.5 * (dot - sq);
        let g = loss_grad(loss, s, y[r as usize]);
        apply_update(w0, g, &mut acc_w0, lr, opt);
        for (&i, &x) in indices.iter().zip(values) {
            let i = i as usize;
            let grad = g * x + l2_linear * w[i];
            apply_update(&mut w[i], grad, &mut acc_w[i], lr, opt);
        }
        for (&i, &x) in indices.iter().zip(values) {
            let i = i as usize;
            for (f, &cache_f) in cache.iter().enumerate() {
                let vi = i * k + f;
                let grad = g * (x * cache_f - v[vi] * x * x) + l2_factors * v[vi];
                apply_update(&mut v[vi], grad, &mut acc_v[vi], lr, opt);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive O(z^2 k) pairwise sum, the math_spec definition.
    fn fm_naive(indices: &[usize], values: &[f64], w0: f64, w: &[f64], v: &[f64], k: usize) -> f64 {
        let mut s = w0;
        for (&i, &x) in indices.iter().zip(values) {
            s += w[i] * x;
        }
        for a in 0..indices.len() {
            for b in (a + 1)..indices.len() {
                let (i, j) = (indices[a], indices[b]);
                let dot: f64 = (0..k).map(|f| v[i * k + f] * v[j * k + f]).sum();
                s += dot * values[a] * values[b];
            }
        }
        s
    }

    #[test]
    fn fast_matches_naive() {
        // x = (2, 3), v_0 = (1, 0), v_1 = (1, 1), w = (0.5, -1), w0 = 0.25 -> 4.25
        let (w0, w, v, k) = (0.25, vec![0.5, -1.0], vec![1.0, 0.0, 1.0, 1.0], 2);
        let x = vec![2.0, 3.0];
        let out = predict_dense(&x, 1, 2, w0, &w, &v, k);
        assert!((out[0] - 4.25).abs() < 1e-12);
        let naive = fm_naive(&[0, 1], &[2.0, 3.0], w0, &w, &v, k);
        assert!((out[0] - naive).abs() < 1e-12);
    }

    #[test]
    fn zero_row_is_bias() {
        let out = predict_dense(&[0.0, 0.0, 0.0], 1, 3, 1.5, &[1.0; 3], &[1.0; 6], 2);
        assert_eq!(out[0], 1.5);
    }

    #[test]
    fn one_sgd_step_hand_computed() {
        // Single feature x=1, y=1, k=1, V=0, logistic SGD with lr=1:
        // s=0, g=sigmoid(0)-1=-0.5 -> w0=0.5, w=0.5; V grad is 0 (cache=0, v=0).
        let indptr = [0i64, 1];
        let indices = [0i64];
        let data = [1.0];
        let csr = CsrView::new(&indptr, &indices, &data, 1).unwrap();
        let (mut w0, mut w, mut v) = (0.0, vec![0.0], vec![0.0]);
        fit_csr(
            &csr, &[1.0], &mut w0, &mut w, &mut v, 1,
            Loss::Logistic, Optimizer::Sgd, 1.0, 0.0, 0.0, &[0],
        );
        assert!((w0 - 0.5).abs() < 1e-15);
        assert!((w[0] - 0.5).abs() < 1e-15);
        assert_eq!(v[0], 0.0);
    }
}
