//! FFM prediction kernels (docs/math_spec.md).
//!
//! y_hat = w0 + sum_i w_i x_i + sum_{i<j} <V[i, f_j], V[j, f_i]> x_i x_j
//!
//! O(z^2 k) per row in the number of nonzeros z; there is no FM-style
//! factorization for FFM.

use rayon::prelude::*;

use crate::data::{dense_row_nonzeros, CsrView};
use crate::optimizer::{sigmoid, step_coord, step_param, AdamState, FtrlState, Optimizer};

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

/// Per-FFM mini-batch gradient scratch (docs/optimization_spec.md, "Mini-batch").
///
/// Accumulates one batch's data-gradients from the frozen parameters — `g_w0`,
/// `gw[i]` (linear), and `gv` per touched (feature, field) slot — then `flush`
/// applies one update per touched coordinate with the batch-mean gradient plus
/// lazy L2. `gv` and the slot index are laid out exactly like `v`.
struct FfmGradAccum {
    g_w0: f64,
    gw: Vec<f64>,             // n_features
    gv: Vec<f64>,             // n_features * n_fields * k (indexed like v)
    touched_feat: Vec<usize>,
    seen_feat: Vec<bool>,     // n_features
    touched_slot: Vec<usize>, // slot = feature * n_fields + field
    seen_slot: Vec<bool>,     // n_features * n_fields
}

impl FfmGradAccum {
    fn new(n_features: usize, n_fields: usize, k: usize) -> Self {
        Self {
            g_w0: 0.0,
            gw: vec![0.0; n_features],
            gv: vec![0.0; n_features * n_fields * k],
            touched_feat: Vec::new(),
            seen_feat: vec![false; n_features],
            touched_slot: Vec::new(),
            seen_slot: vec![false; n_features * n_fields],
        }
    }

    /// Add one row's data-gradient (`g` = dL/ds, already weighted), reading the
    /// frozen factors `v`. `idx` are the row's (usize) feature ids.
    #[allow(clippy::too_many_arguments)]
    fn add_row(
        &mut self,
        idx: &[usize],
        values: &[f64],
        field_ids: &[i64],
        g: f64,
        v: &[f64],
        n_fields: usize,
        k: usize,
    ) {
        self.g_w0 += g;
        for (&i, &x) in idx.iter().zip(values) {
            if !self.seen_feat[i] {
                self.seen_feat[i] = true;
                self.touched_feat.push(i);
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
                let coef = g * xi * xj;
                let (slot_a, slot_b) = (i * n_fields + fj, j * n_fields + fi);
                let (va, vb) = (slot_a * k, slot_b * k);
                for t in 0..k {
                    self.gv[va + t] += coef * v[vb + t];
                    self.gv[vb + t] += coef * v[va + t];
                }
                if !self.seen_slot[slot_a] {
                    self.seen_slot[slot_a] = true;
                    self.touched_slot.push(slot_a);
                }
                if !self.seen_slot[slot_b] {
                    self.seen_slot[slot_b] = true;
                    self.touched_slot.push(slot_b);
                }
            }
        }
    }

    /// Fold another partial accumulator into self (clearing it), in a fixed order.
    fn merge_from(&mut self, other: &mut FfmGradAccum, k: usize) {
        self.g_w0 += other.g_w0;
        other.g_w0 = 0.0;
        for &i in &other.touched_feat {
            if !self.seen_feat[i] {
                self.seen_feat[i] = true;
                self.touched_feat.push(i);
            }
            self.gw[i] += other.gw[i];
            other.gw[i] = 0.0;
            other.seen_feat[i] = false;
        }
        other.touched_feat.clear();
        for &slot in &other.touched_slot {
            if !self.seen_slot[slot] {
                self.seen_slot[slot] = true;
                self.touched_slot.push(slot);
            }
            let base = slot * k;
            for t in 0..k {
                self.gv[base + t] += other.gv[base + t];
                other.gv[base + t] = 0.0;
            }
            other.seen_slot[slot] = false;
        }
        other.touched_slot.clear();
    }

    /// Apply one update per touched coordinate (w0, then w, then V slots), then
    /// clear the touched entries back to zero so the buffers are reusable.
    #[allow(clippy::too_many_arguments)]
    fn flush(
        &mut self,
        bsz: f64,
        w0: &mut f64,
        w: &mut [f64],
        v: &mut [f64],
        acc_w0: &mut f64,
        acc_w: &mut [f64],
        acc_v: &mut [f64],
        adam: &mut AdamState,
        ftrl: &mut FtrlState,
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
            &mut adam.m_w0, &mut adam.s_w0, &mut adam.t_w0, &mut ftrl.z_w0, &mut ftrl.n_w0, lr, opt,
        );
        self.g_w0 = 0.0;
        for &i in &self.touched_feat {
            step_coord(
                &mut w[i], self.gw[i] / bsz, l1_linear, l2_linear, &mut acc_w[i],
                &mut adam.m_w, &mut adam.s_w, &mut adam.t_w, &mut ftrl.z_w, &mut ftrl.n_w, i, lr, opt,
            );
            self.gw[i] = 0.0;
            self.seen_feat[i] = false;
        }
        self.touched_feat.clear();
        for &slot in &self.touched_slot {
            let base = slot * k;
            for t in 0..k {
                let vi = base + t;
                step_coord(
                    &mut v[vi], self.gv[vi] / bsz, l1_factors, l2_factors, &mut acc_v[vi],
                    &mut adam.m_v, &mut adam.s_v, &mut adam.t_v, &mut ftrl.z_v, &mut ftrl.n_v, vi, lr, opt,
                );
                self.gv[vi] = 0.0;
            }
            self.seen_slot[slot] = false;
        }
        self.touched_slot.clear();
    }
}

/// Accumulate a contiguous run of rows into `acc` from the frozen parameters.
/// One unit of work for the (optionally parallel) batch fill.
#[allow(clippy::too_many_arguments)]
fn accumulate_chunk(
    acc: &mut FfmGradAccum,
    chunk: &[i64],
    csr: &CsrView,
    y: &[f64],
    sample_weight: &[f64],
    field_ids: &[i64],
    w0: f64,
    w: &[f64],
    v: &[f64],
    n_fields: usize,
    k: usize,
) {
    let mut idx_buf: Vec<usize> = Vec::new();
    for &r in chunk {
        let (indices, values) = csr.row(r as usize);
        idx_buf.clear();
        idx_buf.extend(indices.iter().map(|&i| i as usize));
        let s = ffm_score_row(&idx_buf, values, field_ids, w0, w, v, n_fields, k);
        let g = sample_weight[r as usize] * (sigmoid(s) - y[r as usize]);
        acc.add_row(&idx_buf, values, field_ids, g, v, n_fields, k);
    }
}

/// Train an FFM (logistic loss) in place (docs/optimization_spec.md).
///
/// Mirrors `_reference_train.ffm_fit_reference`: each epoch's `n_rows` entries
/// in `row_orders` are consumed in `batch_size` chunks; scores come from the
/// frozen batch-start parameters, V gradients are accumulated per touched
/// (feature, field) slot over all pairs, averaged over the batch, and applied
/// once per slot; lazy L2; update order w0 -> w -> V. batch_size=1 is the
/// per-row path. `n_jobs` (>= 1) splits each batch across that many threads,
/// reduced in chunk order (n_jobs=1 is serial and matches the reference). y is 0/1.
#[allow(clippy::too_many_arguments)]
pub fn fit_csr(
    csr: &CsrView,
    y: &[f64],
    sample_weight: &[f64],
    field_ids: &[i64],
    w0: &mut f64,
    w: &mut [f64],
    v: &mut [f64],
    acc_w0: &mut f64,
    acc_w: &mut [f64],
    acc_v: &mut [f64],
    n_fields: usize,
    k: usize,
    opt: Optimizer,
    lr: f64,
    l1_linear: f64,
    l2_linear: f64,
    l1_factors: f64,
    l2_factors: f64,
    n_rows: usize,
    batch_size: usize,
    n_jobs: usize,
    row_orders: &[i64],
) {
    let n = w.len();
    let n_threads = n_jobs.max(1);
    let mut accs: Vec<FfmGradAccum> =
        (0..n_threads).map(|_| FfmGradAccum::new(n, n_fields, k)).collect();
    let mut adam = AdamState::new(matches!(opt, Optimizer::Adam { .. }), n, n * n_fields * k);
    let mut ftrl = FtrlState::new(matches!(opt, Optimizer::Ftrl { .. }), n, n * n_fields * k);
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            if n_threads == 1 {
                accumulate_chunk(
                    &mut accs[0], batch, csr, y, sample_weight, field_ids, *w0, w, v, n_fields, k,
                );
            } else {
                let chunk_len = batch.len().div_ceil(n_threads);
                let (w_ro, v_ro): (&[f64], &[f64]) = (w, v);
                accs.par_iter_mut()
                    .zip(batch.par_chunks(chunk_len))
                    .for_each(|(acc, chunk)| {
                        accumulate_chunk(
                            acc, chunk, csr, y, sample_weight, field_ids, *w0, w_ro, v_ro,
                            n_fields, k,
                        );
                    });
                let (head, tail) = accs.split_at_mut(1);
                for other in tail.iter_mut() {
                    head[0].merge_from(other, k);
                }
            }
            accs[0].flush(
                batch.len() as f64, w0, w, v, acc_w0, acc_w, acc_v, &mut adam, &mut ftrl, k, opt,
                lr, l1_linear, l2_linear, l1_factors, l2_factors,
            );
        }
    }
}

/// Train a multiclass (softmax) FFM in place (docs/optimization_spec.md).
///
/// Mirrors `_reference_train.ffm_fit_multiclass_reference`: one FFM per class,
/// coupled only through the softmax gradient `sample_weight * (p_c - target_c)`
/// (label smoothing as in the FM multiclass kernel). Per-class logits come from
/// the frozen batch-start parameters; each class accumulates and flushes
/// independently. `w0` is (C,), `w` is row-major (C, n_features), `v` is row-major
/// (C, n_features, n_fields, k); `y` holds class indices in [0, n_classes).
/// Serial (no `n_jobs`), like the FM multiclass kernel.
#[allow(clippy::too_many_arguments)]
pub fn fit_multiclass_csr(
    csr: &CsrView,
    y: &[i64],
    sample_weight: &[f64],
    field_ids: &[i64],
    w0: &mut [f64],
    w: &mut [f64],
    v: &mut [f64],
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
    let vc = n * n_fields * k; // V entries per class
    let mut acc_w0 = vec![0.0; n_classes];
    let mut acc_w = vec![0.0; n_classes * n];
    let mut acc_v = vec![0.0; n_classes * vc];
    let adam = matches!(opt, Optimizer::Adam { .. });
    let ftrl = matches!(opt, Optimizer::Ftrl { .. });
    let mut accums: Vec<FfmGradAccum> =
        (0..n_classes).map(|_| FfmGradAccum::new(n, n_fields, k)).collect();
    let mut adams: Vec<AdamState> = (0..n_classes).map(|_| AdamState::new(adam, n, vc)).collect();
    let mut ftrls: Vec<FtrlState> = (0..n_classes).map(|_| FtrlState::new(ftrl, n, vc)).collect();
    let off = if n_classes > 1 {
        label_smoothing / (n_classes as f64 - 1.0)
    } else {
        0.0
    };
    let mut probs = vec![0.0; n_classes];
    let mut idx_buf: Vec<usize> = Vec::new();
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            for &r in batch {
                let (indices, values) = csr.row(r as usize);
                idx_buf.clear();
                idx_buf.extend(indices.iter().map(|&i| i as usize));
                let yc = y[r as usize] as usize;
                let sw = sample_weight[r as usize];
                // pass 1: per-class FFM logit from the frozen parameters
                for c in 0..n_classes {
                    let w_c = &w[c * n..(c + 1) * n];
                    let v_c = &v[c * vc..(c + 1) * vc];
                    probs[c] = ffm_score_row(&idx_buf, values, field_ids, w0[c], w_c, v_c, n_fields, k);
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
                    accums[c].add_row(&idx_buf, values, field_ids, g, v_c, n_fields, k);
                }
            }
            let bsz = batch.len() as f64;
            for c in 0..n_classes {
                let (wr, vr) = (c * n..(c + 1) * n, c * vc..(c + 1) * vc);
                accums[c].flush(
                    bsz, &mut w0[c], &mut w[wr.clone()], &mut v[vr.clone()],
                    &mut acc_w0[c], &mut acc_w[wr], &mut acc_v[vr], &mut adams[c], &mut ftrls[c], k,
                    opt, lr, l1_linear, l2_linear, l1_factors, l2_factors,
                );
            }
        }
    }
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

    #[test]
    fn training_decreases_logistic_loss() {
        // Two rows, two features in different fields; y = (1, 0).
        let indptr = [0i64, 2, 4];
        let indices = [0i64, 1, 0, 1];
        let data = [1.0, 1.0, 1.0, -1.0];
        let csr = CsrView::new(&indptr, &indices, &data, 2).unwrap();
        let field_ids = [0i64, 1];
        let y = [1.0, 0.0];
        let (mut w0, mut w) = (0.0, vec![0.0; 2]);
        let mut v = vec![0.01; 2 * 2 * 2];
        let loss = |w0: f64, w: &[f64], v: &[f64]| -> f64 {
            let s = predict_csr(&csr, &field_ids, w0, w, v, 2, 2);
            -((sigmoid(s[0])).ln() + (1.0 - sigmoid(s[1])).ln())
        };
        let before = loss(w0, &w, &v);
        let orders: Vec<i64> = (0..30).flat_map(|_| [0i64, 1]).collect();
        let (mut a0, mut aw, mut av) = (0.0, vec![0.0; 2], vec![0.0; 2 * 2 * 2]);
        fit_csr(
            &csr, &y, &[1.0, 1.0], &field_ids, &mut w0, &mut w, &mut v,
            &mut a0, &mut aw, &mut av, 2, 2,
            Optimizer::Adagrad, 0.1, 0.0, 0.0, 0.0, 0.0, 2, 1, 1, &orders,
        );
        assert!(loss(w0, &w, &v) < 0.5 * before);
    }
}
