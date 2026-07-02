//! FwFM kernels (docs/math_spec_fwfm.md).
//!
//! y_hat = w0 + sum_i w_i x_i + sum_{i<j} r_{f_i f_j} <v_i, v_j> x_i x_j
//!
//! `v` is FM-shaped row-major (n_features, k); `r` is row-major
//! (n_fields, n_fields) with only the upper triangle (incl. diagonal) read via
//! (min(f_i,f_j), max(f_i,f_j)). O(z^2 k) per row via the explicit pair loop
//! (a ascending, b ascending) — the same operation order as the NumPy
//! reference; no FM-style factorization. Serial (no rayon) in v0.5.

use crate::data::{dense_row_nonzeros, CsrView};
use crate::optimizer::{
    loss_grad, step_coord, step_param, AdamStateMut, FtrlStateMut, GroupStateMut, Loss,
    McGroupState, McState, Optimizer,
};

#[inline]
fn pair_slot(fa: usize, fb: usize, n_fields: usize) -> usize {
    let (pa, pb) = if fa <= fb { (fa, fb) } else { (fb, fa) };
    pa * n_fields + pb
}

/// Score one row given its nonzero (index, value) pairs.
#[allow(clippy::too_many_arguments)]
fn fwfm_score_row(
    indices: &[usize],
    values: &[f64],
    field_ids: &[i64],
    w0: f64,
    w: &[f64],
    v: &[f64],
    r: &[f64],
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
            let vi = &v[i * k..(i + 1) * k];
            let vj = &v[j * k..(j + 1) * k];
            let dot: f64 = vi.iter().zip(vj).map(|(p, q)| p * q).sum();
            s += r[pair_slot(fi, fj, n_fields)] * dot * xi * xj;
        }
    }
    s
}

/// FwFM prediction over a dense C-contiguous (n_rows, n_features) matrix.
#[allow(clippy::too_many_arguments)]
pub fn predict_dense(
    x: &[f64],
    n_rows: usize,
    n_features: usize,
    field_ids: &[i64],
    w0: f64,
    w: &[f64],
    v: &[f64],
    r: &[f64],
    n_fields: usize,
    k: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_rows);
    let mut idx_buf = Vec::new();
    let mut val_buf = Vec::new();
    for row in 0..n_rows {
        let xr = &x[row * n_features..(row + 1) * n_features];
        dense_row_nonzeros(xr, &mut idx_buf, &mut val_buf);
        out.push(fwfm_score_row(&idx_buf, &val_buf, field_ids, w0, w, v, r, n_fields, k));
    }
    out
}

/// FwFM prediction over a CSR matrix.
#[allow(clippy::too_many_arguments)]
pub fn predict_csr(
    csr: &CsrView,
    field_ids: &[i64],
    w0: f64,
    w: &[f64],
    v: &[f64],
    r: &[f64],
    n_fields: usize,
    k: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(csr.n_rows());
    let mut idx_buf = Vec::new();
    for row in 0..csr.n_rows() {
        let (indices, values) = csr.row(row);
        idx_buf.clear();
        idx_buf.extend(indices.iter().map(|&i| i as usize));
        out.push(fwfm_score_row(&idx_buf, values, field_ids, w0, w, v, r, n_fields, k));
    }
    out
}

/// Per-FwFM mini-batch gradient scratch (docs/math_spec_fwfm.md): `gw`/`gv`
/// over touched features (FM-shaped) plus `gr` over touched upper-triangle
/// field pairs. Backing arrays stay zero between batches (`flush` clears
/// exactly the touched entries).
///
/// `pub(crate)` (fields included) so the CUDA training path can load a
/// device-accumulated batch into the same buffers and reuse `flush` — the
/// optimizer semantics live in exactly one place (like `FmGradAccum` /
/// `FfmGradAccum`).
pub(crate) struct FwfmGradAccum {
    pub(crate) g_w0: f64,
    pub(crate) gw: Vec<f64>,             // n_features
    pub(crate) gv: Vec<f64>,             // n_features * k
    pub(crate) gr: Vec<f64>,             // n_fields * n_fields (upper triangle used)
    pub(crate) touched: Vec<usize>,      // features touched this batch
    pub(crate) seen: Vec<bool>,          // n_features
    pub(crate) touched_pair: Vec<usize>, // pair slots (pa * n_fields + pb, pa <= pb)
    pub(crate) seen_pair: Vec<bool>,     // n_fields * n_fields
}

impl FwfmGradAccum {
    pub(crate) fn new(n_features: usize, n_fields: usize, k: usize) -> Self {
        Self {
            g_w0: 0.0,
            gw: vec![0.0; n_features],
            gv: vec![0.0; n_features * k],
            gr: vec![0.0; n_fields * n_fields],
            touched: Vec::new(),
            seen: vec![false; n_features],
            touched_pair: Vec::new(),
            seen_pair: vec![false; n_fields * n_fields],
        }
    }

    /// Add one row's data-gradient (`g` = dL/ds, already weighted), reading the
    /// frozen factors `v` and pair weights `r`.
    #[allow(clippy::too_many_arguments)]
    fn add_row(
        &mut self,
        idx: &[usize],
        values: &[f64],
        field_ids: &[i64],
        g: f64,
        v: &[f64],
        r: &[f64],
        n_fields: usize,
        k: usize,
    ) {
        self.g_w0 += g;
        for (&i, &x) in idx.iter().zip(values) {
            if !self.seen[i] {
                self.seen[i] = true;
                self.touched.push(i);
            }
            self.gw[i] += g * x;
        }
        let z = idx.len();
        for a in 0..z {
            let (i, xi) = (idx[a], values[a]);
            let fi = field_ids[i] as usize;
            for b in (a + 1)..z {
                let (j, xj) = (idx[b], values[b]);
                let fj = field_ids[j] as usize;
                let slot = pair_slot(fi, fj, n_fields);
                let coef = g * xi * xj;
                let rw = r[slot];
                let (vi, vj) = (i * k, j * k);
                let mut dot = 0.0;
                for t in 0..k {
                    dot += v[vi + t] * v[vj + t];
                    self.gv[vi + t] += coef * rw * v[vj + t];
                    self.gv[vj + t] += coef * rw * v[vi + t];
                }
                self.gr[slot] += coef * dot;
                if !self.seen_pair[slot] {
                    self.seen_pair[slot] = true;
                    self.touched_pair.push(slot);
                }
            }
        }
    }

    /// Apply one update per touched coordinate (w0, then w + V per touched
    /// feature, then R per touched pair), clearing the touched entries.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flush(
        &mut self,
        bsz: f64,
        w0: &mut f64,
        w: &mut [f64],
        v: &mut [f64],
        r: &mut [f64],
        acc_w0: &mut f64,
        acc_w: &mut [f64],
        acc_v: &mut [f64],
        adam: &mut AdamStateMut<'_>,
        ftrl: &mut FtrlStateMut<'_>,
        rst: &mut GroupStateMut<'_>,
        k: usize,
        opt: Optimizer,
        lr: f64,
        l1_linear: f64,
        l2_linear: f64,
        l1_factors: f64,
        l2_factors: f64,
    ) {
        step_param(
            w0, self.g_w0 / bsz, 0.0, 0.0, acc_w0,
            adam.m_w0, adam.s_w0, adam.t_w0, ftrl.z_w0, ftrl.n_w0, lr, opt,
        );
        self.g_w0 = 0.0;
        for &i in &self.touched {
            step_coord(
                &mut w[i], self.gw[i] / bsz, l1_linear, l2_linear, &mut acc_w[i],
                adam.m_w, adam.s_w, adam.t_w, ftrl.z_w, ftrl.n_w, i, lr, opt,
            );
            self.gw[i] = 0.0;
            let base = i * k;
            for f in 0..k {
                let vi = base + f;
                step_coord(
                    &mut v[vi], self.gv[vi] / bsz, l1_factors, l2_factors, &mut acc_v[vi],
                    adam.m_v, adam.s_v, adam.t_v, ftrl.z_v, ftrl.n_v, vi, lr, opt,
                );
                self.gv[vi] = 0.0;
            }
            self.seen[i] = false;
        }
        self.touched.clear();
        for &slot in &self.touched_pair {
            step_coord(
                &mut r[slot], self.gr[slot] / bsz, l1_factors, l2_factors, &mut rst.acc[slot],
                rst.m, rst.s, rst.t, rst.z, rst.n, slot, lr, opt,
            );
            self.gr[slot] = 0.0;
            self.seen_pair[slot] = false;
        }
        self.touched_pair.clear();
    }
}

/// Train an FwFM in place (docs/math_spec_fwfm.md). Mirrors
/// `_reference_train.fwfm_fit_reference`: per-batch data-gradients from the
/// frozen batch-start parameters, batch-mean update once per touched
/// coordinate (lazy L2; update order w0 -> w/V -> R; R is regularized with
/// l1/l2_factors). Serial. `adam`/`ftrl` carry the w0/w/V state, `rst` the R
/// group — kernel-local or caller-provided for the epoch-driven
/// early-stopping hand-off.
#[allow(clippy::too_many_arguments)]
pub fn fit_csr(
    csr: &CsrView,
    y: &[f64],
    sample_weight: &[f64],
    field_ids: &[i64],
    w0: &mut f64,
    w: &mut [f64],
    v: &mut [f64],
    r: &mut [f64],
    acc_w0: &mut f64,
    acc_w: &mut [f64],
    acc_v: &mut [f64],
    mut adam: AdamStateMut<'_>,
    mut ftrl: FtrlStateMut<'_>,
    mut rst: GroupStateMut<'_>,
    n_fields: usize,
    k: usize,
    loss: Loss,
    opt: Optimizer,
    lr: f64,
    l1_linear: f64,
    l2_linear: f64,
    l1_factors: f64,
    l2_factors: f64,
    n_rows: usize,
    batch_size: usize,
    row_orders: &[i64],
) {
    let n = w.len();
    let mut acc = FwfmGradAccum::new(n, n_fields, k);
    let mut idx_buf: Vec<usize> = Vec::new();
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            for &row in batch {
                let (indices, values) = csr.row(row as usize);
                idx_buf.clear();
                idx_buf.extend(indices.iter().map(|&i| i as usize));
                let s = fwfm_score_row(&idx_buf, values, field_ids, *w0, w, v, r, n_fields, k);
                let g = sample_weight[row as usize] * loss_grad(loss, s, y[row as usize]);
                acc.add_row(&idx_buf, values, field_ids, g, v, r, n_fields, k);
            }
            acc.flush(
                batch.len() as f64, w0, w, v, r, acc_w0, acc_w, acc_v,
                &mut adam, &mut ftrl, &mut rst, k, opt,
                lr, l1_linear, l2_linear, l1_factors, l2_factors,
            );
        }
    }
}

/// Train a multiclass (softmax) FwFM in place: one FwFM per class coupled by
/// the softmax gradient, mirroring `fwfm_fit_multiclass_reference`. `w0` (C,),
/// `w` (C, n), `v` (C, n, k), `r` (C, F, F) row-major; `st`/`rst` hold the
/// per-class optimizer state. Serial.
#[allow(clippy::too_many_arguments)]
pub fn fit_multiclass_csr(
    csr: &CsrView,
    y: &[i64],
    sample_weight: &[f64],
    field_ids: &[i64],
    w0: &mut [f64],
    w: &mut [f64],
    v: &mut [f64],
    r: &mut [f64],
    mut st: McState<'_>,
    mut rst: McGroupState<'_>,
    n_classes: usize,
    n_features: usize,
    n_fields: usize,
    k: usize,
    opt: Optimizer,
    lr: f64,
    l1_linear: f64,
    l2_linear: f64,
    l1_factors: f64,
    l2_factors: f64,
    label_smoothing: f64,
    n_rows: usize,
    batch_size: usize,
    row_orders: &[i64],
) {
    let n = n_features;
    let vc = n * k; // V entries per class
    let rc = n_fields * n_fields; // R entries per class
    let mut accums: Vec<FwfmGradAccum> =
        (0..n_classes).map(|_| FwfmGradAccum::new(n, n_fields, k)).collect();
    let off = if n_classes > 1 {
        label_smoothing / (n_classes as f64 - 1.0)
    } else {
        0.0
    };
    let mut probs = vec![0.0; n_classes];
    let mut idx_buf: Vec<usize> = Vec::new();
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            for &row in batch {
                let (indices, values) = csr.row(row as usize);
                idx_buf.clear();
                idx_buf.extend(indices.iter().map(|&i| i as usize));
                let yc = y[row as usize] as usize;
                let sw = sample_weight[row as usize];
                // pass 1: per-class logit from the frozen parameters
                for c in 0..n_classes {
                    let w_c = &w[c * n..(c + 1) * n];
                    let v_c = &v[c * vc..(c + 1) * vc];
                    let r_c = &r[c * rc..(c + 1) * rc];
                    probs[c] =
                        fwfm_score_row(&idx_buf, values, field_ids, w0[c], w_c, v_c, r_c, n_fields, k);
                }
                let maxl = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum_ex = 0.0;
                for p in probs.iter_mut() {
                    *p = (*p - maxl).exp();
                    sum_ex += *p;
                }
                for c in 0..n_classes {
                    let p = probs[c] / sum_ex;
                    let target = if c == yc { 1.0 - label_smoothing } else { off };
                    let g = sw * (p - target);
                    let v_c = &v[c * vc..(c + 1) * vc];
                    let r_c = &r[c * rc..(c + 1) * rc];
                    accums[c].add_row(&idx_buf, values, field_ids, g, v_c, r_c, n_fields, k);
                }
            }
            let bsz = batch.len() as f64;
            for c in 0..n_classes {
                let (wr, vr, rr) = (
                    c * n..(c + 1) * n,
                    c * vc..(c + 1) * vc,
                    c * rc..(c + 1) * rc,
                );
                let (acc_w0_c, acc_w_c, acc_v_c, mut adam_c, mut ftrl_c) =
                    st.class_views(c, n, vc);
                let mut rst_c = rst.class_views(c, rc);
                accums[c].flush(
                    bsz, &mut w0[c], &mut w[wr], &mut v[vr], &mut r[rr],
                    acc_w0_c, acc_w_c, acc_v_c, &mut adam_c, &mut ftrl_c, &mut rst_c, k,
                    opt, lr, l1_linear, l2_linear, l1_factors, l2_factors,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{sigmoid, AdamBuf, FtrlBuf};

    #[test]
    fn ones_r_matches_plain_fm() {
        // With R = ones, FwFM == FM: x = (2, 3), v_0 = (1, 0), v_1 = (1, 1),
        // w = (0.5, -1), w0 = 0.25 -> 4.25 (same case as fm::tests).
        let (w0, w, v, k) = (0.25, vec![0.5, -1.0], vec![1.0, 0.0, 1.0, 1.0], 2);
        let r = vec![1.0; 4]; // 2 fields
        let out = predict_dense(&[2.0, 3.0], 1, 2, &[0, 1], w0, &w, &v, &r, 2, k);
        assert!((out[0] - 4.25).abs() < 1e-12);
    }

    #[test]
    fn r_weights_scale_the_pairwise_term() {
        // Same as above but r_{01} = 2 doubles the pairwise part:
        // pairwise = <v0, v1> * 2 * 3 = 6 -> 0.25 + (1.0 - 3.0) + 2 * 6 = 10.25
        let (w0, w, v, k) = (0.25, vec![0.5, -1.0], vec![1.0, 0.0, 1.0, 1.0], 2);
        let mut r = vec![1.0; 4];
        r[1] = 2.0; // slot (0, 1)
        let out = predict_dense(&[2.0, 3.0], 1, 2, &[0, 1], w0, &w, &v, &r, 2, k);
        assert!((out[0] - 10.25).abs() < 1e-12);
    }

    #[test]
    fn training_decreases_logistic_loss() {
        let indptr = [0i64, 2, 4];
        let indices = [0i64, 1, 0, 1];
        let data = [1.0, 1.0, 1.0, -1.0];
        let csr = CsrView::new(&indptr, &indices, &data, 2).unwrap();
        let field_ids = [0i64, 1];
        let y = [1.0, 0.0];
        let (mut w0, mut w) = (0.0, vec![0.0; 2]);
        let mut v = vec![0.01; 2 * 2];
        let mut r = vec![1.0; 4];
        let loss = |w0: f64, w: &[f64], v: &[f64], r: &[f64]| -> f64 {
            let s = predict_csr(&csr, &field_ids, w0, w, v, r, 2, 2);
            -((sigmoid(s[0])).ln() + (1.0 - sigmoid(s[1])).ln())
        };
        let before = loss(w0, &w, &v, &r);
        let orders: Vec<i64> = (0..30).flat_map(|_| [0i64, 1]).collect();
        let (mut a0, mut aw, mut av) = (0.0, vec![0.0; 2], vec![0.0; 2 * 2]);
        let mut acc_r = vec![0.0; 4];
        let (mut ab, mut fb) = (AdamBuf::new(false, 2, 4), FtrlBuf::new(false, 2, 4));
        let mut empty: [Vec<f64>; 5] = Default::default();
        let [m, s, t, z, nn] = &mut empty;
        let rst = GroupStateMut { acc: &mut acc_r, m, s, t, z, n: nn };
        fit_csr(
            &csr, &y, &[1.0, 1.0], &field_ids, &mut w0, &mut w, &mut v, &mut r,
            &mut a0, &mut aw, &mut av, ab.view(), fb.view(), rst, 2, 2,
            Loss::Logistic, Optimizer::Adagrad, 0.1, 0.0, 0.0, 0.0, 0.0, 2, 1, &orders,
        );
        assert!(loss(w0, &w, &v, &r) < 0.5 * before);
    }
}
