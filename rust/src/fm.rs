//! FM prediction kernels (docs/math_spec.md).
//!
//! y_hat = w0 + sum_i w_i x_i
//!       + 0.5 * sum_f [(sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2]

use rayon::prelude::*;

use crate::data::{dense_row_nonzeros, CsrView};
use crate::optimizer::{
    loss_grad, step_coord, step_param, AdamStateMut, FtrlStateMut, Loss, McState, Optimizer,
};

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

/// Score one CSR row from pre-update params, filling `cache[f] = sum_i v_{i,f} x_i`.
///
/// Uses the `(dot - sq)` accumulation order shared with the Python reference
/// (`cache @ cache - sum((Vi*val)^2)`), so binary and multiclass training stay
/// bit-parity with `_reference_train`. Indices are CSR (i64) column ids.
fn train_score_row(
    indices: &[i64],
    values: &[f64],
    w0: f64,
    w: &[f64],
    v: &[f64],
    k: usize,
    cache: &mut [f64],
) -> f64 {
    cache.fill(0.0);
    let mut s = w0;
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
    s + 0.5 * (dot - sq)
}

/// Per-FM mini-batch gradient scratch (docs/optimization_spec.md, "Mini-batch").
///
/// Accumulates one batch's data-gradients from the frozen batch-start
/// parameters — `g_w0` (bias), `gw[i]` (linear), `gv[i*k+f]` (factors) — over
/// the touched features, then `flush` applies one update per touched coordinate
/// with the batch-mean gradient plus lazy L2. Sized for one FM (one class in
/// the multiclass kernel); backing arrays stay zero between batches because
/// `flush` clears exactly the touched entries.
struct FmGradAccum {
    g_w0: f64,
    gw: Vec<f64>,        // n_features
    gv: Vec<f64>,        // n_features * k
    touched: Vec<usize>, // features touched this batch
    seen: Vec<bool>,     // membership flag for `touched`, n_features
}

impl FmGradAccum {
    fn new(n_features: usize, k: usize) -> Self {
        Self {
            g_w0: 0.0,
            gw: vec![0.0; n_features],
            gv: vec![0.0; n_features * k],
            touched: Vec::new(),
            seen: vec![false; n_features],
        }
    }

    /// Add one row's data-gradient (`g` = dL/ds, already weighted), reading the
    /// frozen factors `v` and the row's factor-sum `cache` from `train_score_row`.
    fn add_row(&mut self, indices: &[i64], values: &[f64], g: f64, cache: &[f64], v: &[f64], k: usize) {
        self.g_w0 += g;
        for (&i, &x) in indices.iter().zip(values) {
            let i = i as usize;
            if !self.seen[i] {
                self.seen[i] = true;
                self.touched.push(i);
            }
            self.gw[i] += g * x;
            let base = i * k;
            for f in 0..k {
                self.gv[base + f] += g * (x * cache[f] - v[base + f] * x * x);
            }
        }
    }

    /// Fold another partial accumulator's touched contributions into self,
    /// clearing `other` back to zero for reuse. Reduces per-thread partials in
    /// a fixed (thread-index) order so n_jobs>1 stays reproducible.
    fn merge_from(&mut self, other: &mut FmGradAccum, k: usize) {
        self.g_w0 += other.g_w0;
        other.g_w0 = 0.0;
        for &i in &other.touched {
            if !self.seen[i] {
                self.seen[i] = true;
                self.touched.push(i);
            }
            self.gw[i] += other.gw[i];
            other.gw[i] = 0.0;
            let base = i * k;
            for f in 0..k {
                self.gv[base + f] += other.gv[base + f];
                other.gv[base + f] = 0.0;
            }
            other.seen[i] = false;
        }
        other.touched.clear();
    }

    /// Apply one update per touched coordinate (bias always), then clear the
    /// touched entries back to zero so the buffers are reusable next batch.
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
        adam: &mut AdamStateMut<'_>,
        ftrl: &mut FtrlStateMut<'_>,
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
    }
}

/// Accumulate a contiguous run of rows into `acc` from the frozen parameters.
/// One unit of work for the (optionally parallel) batch fill; reads only shared
/// state besides `acc`, so chunks run on independent threads.
#[allow(clippy::too_many_arguments)]
fn accumulate_chunk(
    acc: &mut FmGradAccum,
    chunk: &[i64],
    csr: &CsrView,
    y: &[f64],
    sample_weight: &[f64],
    w0: f64,
    w: &[f64],
    v: &[f64],
    k: usize,
    loss: Loss,
) {
    let mut cache = vec![0.0; k];
    for &r in chunk {
        let (indices, values) = csr.row(r as usize);
        let s = train_score_row(indices, values, w0, w, v, k, &mut cache);
        let g = sample_weight[r as usize] * loss_grad(loss, s, y[r as usize]);
        acc.add_row(indices, values, g, &cache, v, k);
    }
}

/// Train an FM in place (docs/optimization_spec.md).
///
/// Mirrors `_reference_train.fm_fit_reference`: each epoch's `n_rows` entries in
/// `row_orders` are consumed in `batch_size` chunks; per-row gradients come from
/// the frozen batch-start parameters, are averaged over the batch, and applied
/// once per touched coordinate (lazy L2, update order w0 -> w -> V). batch_size=1
/// is the per-row path. For logistic loss, y must be 0/1.
///
/// `n_jobs` (>= 1) splits each batch into that many contiguous chunks accumulated
/// in parallel, then reduced in chunk order; n_jobs=1 is the serial path and is
/// bit-identical to the reference. n_jobs>1 differs only in float summation order
/// (reproducible for a fixed n_jobs).
///
/// `adam`/`ftrl` hold the per-coordinate optimizer state: kernel-local
/// (`AdamBuf`/`FtrlBuf`) for a single all-epochs call, or caller-provided
/// backing round-tripped across epoch-driven early-stopping calls.
#[allow(clippy::too_many_arguments)]
pub fn fit_csr(
    csr: &CsrView,
    y: &[f64],
    sample_weight: &[f64],
    w0: &mut f64,
    w: &mut [f64],
    v: &mut [f64],
    acc_w0: &mut f64,
    acc_w: &mut [f64],
    acc_v: &mut [f64],
    mut adam: AdamStateMut<'_>,
    mut ftrl: FtrlStateMut<'_>,
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
    n_jobs: usize,
    row_orders: &[i64],
) {
    let n = w.len();
    let n_threads = n_jobs.max(1);
    // One reusable partial accumulator per thread; thread 0 doubles as the
    // reduced batch accumulator.
    let mut accs: Vec<FmGradAccum> = (0..n_threads).map(|_| FmGradAccum::new(n, k)).collect();
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            if n_threads == 1 {
                accumulate_chunk(&mut accs[0], batch, csr, y, sample_weight, *w0, w, v, k, loss);
            } else {
                let chunk_len = batch.len().div_ceil(n_threads);
                let (w_ro, v_ro): (&[f64], &[f64]) = (w, v);
                accs.par_iter_mut()
                    .zip(batch.par_chunks(chunk_len))
                    .for_each(|(acc, chunk)| {
                        accumulate_chunk(acc, chunk, csr, y, sample_weight, *w0, w_ro, v_ro, k, loss);
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

/// Train a multiclass (softmax) FM in place (docs/optimization_spec.md).
///
/// Mirrors `_reference_train.fm_fit_multiclass_reference`: one FM per class,
/// coupled only through the softmax gradient `sample_weight * (p_c - target_c)`
/// (target uses label smoothing: `1 - eps` for the true class, `eps / (C - 1)`
/// otherwise). Per-class logits and factor caches come from the frozen
/// batch-start parameters; each class accumulates and flushes independently over
/// the shared touched-feature set. `w0` is (C,), `w` is row-major (C, n_features),
/// `v` is row-major (C, n_features, k); `y` holds class indices in [0, n_classes).
/// `st` holds the per-class optimizer state (AdaGrad accumulators + Adam/FTRL
/// per-coordinate state): kernel-local for a single all-epochs call, or
/// caller-provided backing round-tripped across epoch-driven early-stopping calls.
#[allow(clippy::too_many_arguments)]
pub fn fit_multiclass_csr(
    csr: &CsrView,
    y: &[i64],
    sample_weight: &[f64],
    w0: &mut [f64],
    w: &mut [f64],
    v: &mut [f64],
    mut st: McState<'_>,
    n_classes: usize,
    n_features: usize,
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
    let mut accums: Vec<FmGradAccum> = (0..n_classes).map(|_| FmGradAccum::new(n, k)).collect();
    let off = if n_classes > 1 {
        label_smoothing / (n_classes as f64 - 1.0)
    } else {
        0.0
    };
    let mut probs = vec![0.0; n_classes]; // logits, then exp(.), then probabilities
    let mut caches = vec![0.0; n_classes * k];
    let mut cache = vec![0.0; k];
    for epoch in row_orders.chunks(n_rows) {
        for batch in epoch.chunks(batch_size) {
            for &r in batch {
                let (indices, values) = csr.row(r as usize);
                let yc = y[r as usize] as usize;
                let sw = sample_weight[r as usize];
                // pass 1: per-class logit + factor cache from frozen parameters
                for c in 0..n_classes {
                    let w_c = &w[c * n..(c + 1) * n];
                    let v_c = &v[c * n * k..(c + 1) * n * k];
                    probs[c] = train_score_row(indices, values, w0[c], w_c, v_c, k, &mut cache);
                    caches[c * k..(c + 1) * k].copy_from_slice(&cache);
                }
                // stable softmax: subtract max, sum exp in class order
                let maxl = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum_ex = 0.0;
                for p in probs.iter_mut() {
                    *p = (*p - maxl).exp();
                    sum_ex += *p;
                }
                // accumulate the per-class softmax gradient into each class's scratch
                for c in 0..n_classes {
                    let p = probs[c] / sum_ex;
                    let target = if c == yc { 1.0 - label_smoothing } else { off };
                    let g = sw * (p - target);
                    let cache_c = &caches[c * k..(c + 1) * k];
                    let v_c = &v[c * n * k..(c + 1) * n * k];
                    accums[c].add_row(indices, values, g, cache_c, v_c, k);
                }
            }
            // flush each class over the shared touched-feature set
            let bsz = batch.len() as f64;
            for c in 0..n_classes {
                let (wr, vr) = (c * n..(c + 1) * n, c * n * k..(c + 1) * n * k);
                let (acc_w0_c, acc_w_c, acc_v_c, mut adam_c, mut ftrl_c) =
                    st.class_views(c, n, n * k);
                accums[c].flush(
                    bsz, &mut w0[c], &mut w[wr], &mut v[vr],
                    acc_w0_c, acc_w_c, acc_v_c, &mut adam_c, &mut ftrl_c, k,
                    opt, lr, l1_linear, l2_linear, l1_factors, l2_factors,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{AdamBuf, FtrlBuf, McBuf};

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
        let (mut a0, mut aw, mut av) = (0.0, vec![0.0], vec![0.0]);
        let (mut ab, mut fb) = (AdamBuf::new(false, 1, 1), FtrlBuf::new(false, 1, 1));
        fit_csr(
            &csr, &[1.0], &[1.0], &mut w0, &mut w, &mut v, &mut a0, &mut aw, &mut av,
            ab.view(), fb.view(), 1,
            Loss::Logistic, Optimizer::Sgd, 1.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, &[0],
        );
        assert!((w0 - 0.5).abs() < 1e-15);
        assert!((w[0] - 0.5).abs() < 1e-15);
        assert_eq!(v[0], 0.0);
    }

    #[test]
    fn adam_one_step_hand_computed() {
        // Same setup as one_sgd_step but Adam, lr=1: g=sigmoid(0)-1=-0.5.
        // t=1 -> m_hat=g, v_hat=g^2, step=lr*g/(|g|+eps) ~= -1, so w0,w ~= +1.
        // V gradient is 0 (cache=0), so V stays exactly 0.
        let indptr = [0i64, 1];
        let indices = [0i64];
        let data = [1.0];
        let csr = CsrView::new(&indptr, &indices, &data, 1).unwrap();
        let (mut w0, mut w, mut v) = (0.0, vec![0.0], vec![0.0]);
        let (mut a0, mut aw, mut av) = (0.0, vec![0.0], vec![0.0]);
        let (mut ab, mut fb) = (AdamBuf::new(true, 1, 1), FtrlBuf::new(false, 1, 1));
        fit_csr(
            &csr, &[1.0], &[1.0], &mut w0, &mut w, &mut v, &mut a0, &mut aw, &mut av,
            ab.view(), fb.view(), 1,
            Loss::Logistic, Optimizer::Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 }, 1.0,
            0.0, 0.0, 0.0, 0.0, 1, 1, 1, &[0],
        );
        assert!((w0 - 1.0).abs() < 1e-6);
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert_eq!(v[0], 0.0);
    }

    #[test]
    fn multiclass_one_sgd_step_hand_computed() {
        // 2 classes, 1 feature x=1, k=1, all params 0, y=class 0, eps=0, SGD lr=1.
        // logits=[0,0] -> p=[0.5,0.5]; g0=0.5-1=-0.5, g1=0.5-0=0.5.
        // w0 -= lr*g -> [0.5,-0.5]; w -= lr*(g*x) -> [0.5,-0.5]; V grad is 0 (cache=0).
        let indptr = [0i64, 1];
        let indices = [0i64];
        let data = [1.0];
        let csr = CsrView::new(&indptr, &indices, &data, 1).unwrap();
        let mut w0 = vec![0.0, 0.0];
        let mut w = vec![0.0, 0.0]; // (C=2, n=1)
        let mut v = vec![0.0, 0.0]; // (C=2, n=1, k=1)
        let mut mcb = McBuf::new(false, false, 2, 1, 1);
        fit_multiclass_csr(
            &csr, &[0], &[1.0], &mut w0, &mut w, &mut v, mcb.view(), 2, 1, 1, Optimizer::Sgd, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, &[0],
        );
        assert!((w0[0] - 0.5).abs() < 1e-15);
        assert!((w0[1] + 0.5).abs() < 1e-15);
        assert!((w[0] - 0.5).abs() < 1e-15);
        assert!((w[1] + 0.5).abs() < 1e-15);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
    }
}
