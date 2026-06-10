//! FFM prediction kernels (docs/math_spec.md).
//!
//! y_hat = w0 + sum_i w_i x_i + sum_{i<j} <V[i, f_j], V[j, f_i]> x_i x_j
//!
//! O(z^2 k) per row in the number of nonzeros z; there is no FM-style
//! factorization for FFM.

use crate::data::{dense_row_nonzeros, CsrView};

/// Score one row given its nonzero (index, value) pairs.
/// `v` is row-major (n_features, n_fields, k).
#[allow(clippy::too_many_arguments)]
fn ffm_score_row(
    indices: &[usize],
    values: &[f64],
    field_ids: &[i64],
    w0: f64,
    w: &[f64],
    v: &[f64],
    n_fields: usize,
    k: usize,
) -> f64 {
    let mut s = w0;
    for (&i, &x) in indices.iter().zip(values) {
        s += w[i] * x;
    }
    for a in 0..indices.len() {
        let (i, xi) = (indices[a], values[a]);
        let fi = field_ids[i] as usize;
        for b in (a + 1)..indices.len() {
            let (j, xj) = (indices[b], values[b]);
            let fj = field_ids[j] as usize;
            let vi = &v[(i * n_fields + fj) * k..(i * n_fields + fj + 1) * k];
            let vj = &v[(j * n_fields + fi) * k..(j * n_fields + fi + 1) * k];
            let dot: f64 = vi.iter().zip(vj).map(|(p, q)| p * q).sum();
            s += dot * xi * xj;
        }
    }
    s
}

/// FFM prediction over a dense C-contiguous (n_rows, n_features) matrix.
#[allow(clippy::too_many_arguments)]
pub fn predict_dense(
    x: &[f64],
    n_rows: usize,
    n_features: usize,
    field_ids: &[i64],
    w0: f64,
    w: &[f64],
    v: &[f64],
    n_fields: usize,
    k: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_rows);
    let mut idx_buf = Vec::new();
    let mut val_buf = Vec::new();
    for r in 0..n_rows {
        let row = &x[r * n_features..(r + 1) * n_features];
        dense_row_nonzeros(row, &mut idx_buf, &mut val_buf);
        out.push(ffm_score_row(&idx_buf, &val_buf, field_ids, w0, w, v, n_fields, k));
    }
    out
}

/// FFM prediction over a CSR matrix.
#[allow(clippy::too_many_arguments)]
pub fn predict_csr(
    csr: &CsrView,
    field_ids: &[i64],
    w0: f64,
    w: &[f64],
    v: &[f64],
    n_fields: usize,
    k: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(csr.n_rows());
    let mut idx_buf = Vec::new();
    for r in 0..csr.n_rows() {
        let (indices, values) = csr.row(r);
        idx_buf.clear();
        idx_buf.extend(indices.iter().map(|&i| i as usize));
        out.push(ffm_score_row(&idx_buf, values, field_ids, w0, w, v, n_fields, k));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_hand_computed_example() {
        // Mirror of tests/test_ffm_correctness.py::test_tiny_hand_computed_example:
        // 3 features, 2 fields, k = 2, f = (0, 0, 1), x = (1, 2, 3), w0 = 0, w = 0.
        // pairs: (0,1) -> 2, (0,2) -> 1.5, (1,2) -> 6; total 9.5
        let mut v = vec![0.0; 3 * 2 * 2];
        let set = |v: &mut Vec<f64>, i: usize, g: usize, vals: [f64; 2]| {
            v[(i * 2 + g) * 2] = vals[0];
            v[(i * 2 + g) * 2 + 1] = vals[1];
        };
        set(&mut v, 0, 0, [1.0, 0.0]);
        set(&mut v, 0, 1, [0.0, 1.0]);
        set(&mut v, 1, 0, [1.0, 1.0]);
        set(&mut v, 1, 1, [2.0, 0.0]);
        set(&mut v, 2, 0, [0.5, 0.5]);
        set(&mut v, 2, 1, [1.0, -1.0]);
        let field_ids = [0i64, 0, 1];
        let x = vec![1.0, 2.0, 3.0];
        let out = predict_dense(&x, 1, 3, &field_ids, 0.0, &[0.0; 3], &v, 2, 2);
        assert!((out[0] - 9.5).abs() < 1e-12);
    }

    #[test]
    fn single_nonzero_has_no_pairwise() {
        let x = vec![0.0, -1.5, 0.0];
        let w = [0.0, 2.0, 0.0];
        let v = vec![7.0; 3 * 2 * 2]; // values must not matter
        let out = predict_dense(&x, 1, 3, &[0, 1, 1], 0.5, &w, &v, 2, 2);
        assert!((out[0] - (0.5 + 2.0 * -1.5)).abs() < 1e-12);
    }
}
