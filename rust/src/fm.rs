//! FM prediction kernels (docs/math_spec.md).
//!
//! y_hat = w0 + sum_i w_i x_i
//!       + 0.5 * sum_f [(sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2]

use crate::data::{dense_row_nonzeros, CsrView};
use crate::optimizer::{adam_step, apply_update, loss_grad, Loss, Optimizer};

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

/// Apply one row's FM gradient to (w0, w, V) with lazy L2 and update order
/// w0 -> w -> V. `g` is the loss derivative dL/ds (already weighted); `cache`
/// is the factor-sum vector from `train_score_row`. Shared by binary and
/// multiclass training.
#[allow(clippy::too_many_arguments)]
fn update_row(
    indices: &[i64],
    values: &[f64],
    g: f64,
    cache: &[f64],
    w0: &mut f64,
    w: &mut [f64],
    v: &mut [f64],
    acc_w0: &mut f64,
    acc_w: &mut [f64],
    acc_v: &mut [f64],
    k: usize,
    opt: Optimizer,
    lr: f64,
    l2_linear: f64,
    l2_factors: f64,
) {
    apply_update(w0, g, acc_w0, lr, opt);
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

/// Adam moment vectors (m, v, t) for one parameter group, or empty triples for
/// the SGD/AdaGrad paths (which never index them).
fn adam_vecs(adam: bool, len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if adam {
        (vec![0.0; len], vec![0.0; len], vec![0.0; len])
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    }
}

/// Adam counterpart of `update_row`: identical w0 -> w -> V order and lazy L2,
/// but every parameter steps through `adam_step` with its own (m, v, t) cell.
/// The slices are sized like (w, V) for the current FM (one class in the
/// multiclass kernel); w0 uses scalar cells. Shared by binary and multiclass.
#[allow(clippy::too_many_arguments)]
fn update_row_adam(
    indices: &[i64],
    values: &[f64],
    g: f64,
    cache: &[f64],
    w0: &mut f64,
    w: &mut [f64],
    v: &mut [f64],
    m_w0: &mut f64,
    v_w0: &mut f64,
    t_w0: &mut f64,
    m_w: &mut [f64],
    v_w: &mut [f64],
    t_w: &mut [f64],
    m_v: &mut [f64],
    v_v: &mut [f64],
    t_v: &mut [f64],
    k: usize,
    lr: f64,
    l2_linear: f64,
    l2_factors: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
) {
    adam_step(w0, g, m_w0, v_w0, t_w0, lr, beta1, beta2, eps);
    for (&i, &x) in indices.iter().zip(values) {
        let i = i as usize;
        let grad = g * x + l2_linear * w[i];
        adam_step(&mut w[i], grad, &mut m_w[i], &mut v_w[i], &mut t_w[i], lr, beta1, beta2, eps);
    }
    for (&i, &x) in indices.iter().zip(values) {
        let i = i as usize;
        for (f, &cache_f) in cache.iter().enumerate() {
            let vi = i * k + f;
            let grad = g * (x * cache_f - v[vi] * x * x) + l2_factors * v[vi];
            adam_step(&mut v[vi], grad, &mut m_v[vi], &mut v_v[vi], &mut t_v[vi], lr, beta1, beta2, eps);
        }
    }
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
    sample_weight: &[f64],
    w0: &mut f64,
    w: &mut [f64],
    v: &mut [f64],
    acc_w0: &mut f64,
    acc_w: &mut [f64],
    acc_v: &mut [f64],
    k: usize,
    loss: Loss,
    opt: Optimizer,
    lr: f64,
    l2_linear: f64,
    l2_factors: f64,
    row_orders: &[i64],
) {
    let mut cache = vec![0.0; k];
    // Adam moment state is internal and never round-tripped (optimization_spec.md).
    let adam = matches!(opt, Optimizer::Adam { .. });
    let (mut m_w0, mut v_w0, mut t_w0) = (0.0, 0.0, 0.0);
    let (mut m_w, mut vacc_w, mut t_w) = adam_vecs(adam, w.len());
    let (mut m_v, mut vacc_v, mut t_v) = adam_vecs(adam, v.len());
    for &r in row_orders {
        let (indices, values) = csr.row(r as usize);
        let s = train_score_row(indices, values, *w0, w, v, k, &mut cache);
        let g = sample_weight[r as usize] * loss_grad(loss, s, y[r as usize]);
        if let Optimizer::Adam { beta1, beta2, eps } = opt {
            update_row_adam(
                indices, values, g, &cache, w0, w, v, &mut m_w0, &mut v_w0, &mut t_w0,
                &mut m_w, &mut vacc_w, &mut t_w, &mut m_v, &mut vacc_v, &mut t_v, k, lr,
                l2_linear, l2_factors, beta1, beta2, eps,
            );
        } else {
            update_row(
                indices, values, g, &cache, w0, w, v, acc_w0, acc_w, acc_v, k, opt, lr,
                l2_linear, l2_factors,
            );
        }
    }
}

/// Train a multiclass (softmax) FM in place with batch_size=1.
///
/// Mirrors `_reference_train.fm_fit_multiclass_reference`: one FM per class,
/// coupled only through the softmax gradient `sample_weight * (p_c - target_c)`
/// (target uses label smoothing: `1 - eps` for the true class, `eps / (C - 1)`
/// otherwise). Per-class logits and factor caches are computed from pre-update
/// parameters, then every class is updated. `w0` is (C,), `w` is row-major
/// (C, n_features), `v` is row-major (C, n_features, k); `y` holds class indices
/// in [0, n_classes). AdaGrad accumulators are internal and span all epochs in
/// `row_orders` (no early-stopping state hand-off in v0.1).
#[allow(clippy::too_many_arguments)]
pub fn fit_multiclass_csr(
    csr: &CsrView,
    y: &[i64],
    sample_weight: &[f64],
    w0: &mut [f64],
    w: &mut [f64],
    v: &mut [f64],
    n_classes: usize,
    n_features: usize,
    k: usize,
    opt: Optimizer,
    lr: f64,
    l2_linear: f64,
    l2_factors: f64,
    label_smoothing: f64,
    row_orders: &[i64],
) {
    let n = n_features;
    let mut acc_w0 = vec![0.0; n_classes];
    let mut acc_w = vec![0.0; n_classes * n];
    let mut acc_v = vec![0.0; n_classes * n * k];
    let adam = matches!(opt, Optimizer::Adam { .. });
    let (mut m_w0, mut v_w0, mut t_w0) = adam_vecs(adam, n_classes);
    let (mut m_w, mut vacc_w, mut t_w) = adam_vecs(adam, n_classes * n);
    let (mut m_v, mut vacc_v, mut t_v) = adam_vecs(adam, n_classes * n * k);
    let off = if n_classes > 1 {
        label_smoothing / (n_classes as f64 - 1.0)
    } else {
        0.0
    };
    let mut probs = vec![0.0; n_classes]; // logits, then exp(.), then probabilities
    let mut caches = vec![0.0; n_classes * k];
    let mut cache = vec![0.0; k];
    for &r in row_orders {
        let (indices, values) = csr.row(r as usize);
        let yc = y[r as usize] as usize;
        let sw = sample_weight[r as usize];
        // pass 1: per-class logit + factor cache from pre-update parameters
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
        // pass 2: per-class softmax-gradient update
        for c in 0..n_classes {
            let p = probs[c] / sum_ex;
            let target = if c == yc { 1.0 - label_smoothing } else { off };
            let g = sw * (p - target);
            let cache_c = &caches[c * k..(c + 1) * k];
            let (wr, vr) = (c * n..(c + 1) * n, c * n * k..(c + 1) * n * k);
            if let Optimizer::Adam { beta1, beta2, eps } = opt {
                update_row_adam(
                    indices, values, g, cache_c, &mut w0[c], &mut w[wr.clone()], &mut v[vr.clone()],
                    &mut m_w0[c], &mut v_w0[c], &mut t_w0[c],
                    &mut m_w[wr.clone()], &mut vacc_w[wr.clone()], &mut t_w[wr],
                    &mut m_v[vr.clone()], &mut vacc_v[vr.clone()], &mut t_v[vr],
                    k, lr, l2_linear, l2_factors, beta1, beta2, eps,
                );
            } else {
                update_row(
                    indices, values, g, cache_c, &mut w0[c], &mut w[wr.clone()], &mut v[vr.clone()],
                    &mut acc_w0[c], &mut acc_w[wr], &mut acc_v[vr], k, opt, lr, l2_linear,
                    l2_factors,
                );
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
        let (mut a0, mut aw, mut av) = (0.0, vec![0.0], vec![0.0]);
        fit_csr(
            &csr, &[1.0], &[1.0], &mut w0, &mut w, &mut v, &mut a0, &mut aw, &mut av, 1,
            Loss::Logistic, Optimizer::Sgd, 1.0, 0.0, 0.0, &[0],
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
        fit_csr(
            &csr, &[1.0], &[1.0], &mut w0, &mut w, &mut v, &mut a0, &mut aw, &mut av, 1,
            Loss::Logistic, Optimizer::Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 }, 1.0,
            0.0, 0.0, &[0],
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
        fit_multiclass_csr(
            &csr, &[0], &[1.0], &mut w0, &mut w, &mut v, 2, 1, 1, Optimizer::Sgd, 1.0,
            0.0, 0.0, 0.0, &[0],
        );
        assert!((w0[0] - 0.5).abs() < 1e-15);
        assert!((w0[1] + 0.5).abs() < 1e-15);
        assert!((w[0] - 0.5).abs() < 1e-15);
        assert!((w[1] + 0.5).abs() < 1e-15);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
    }
}
