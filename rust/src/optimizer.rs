//! Optimizers and loss gradients (docs/optimization_spec.md).
//!
//! Must match `python/modern_fm/_reference_train.py` operation-for-operation;
//! parity is enforced by tests/test_rust_train_parity.py.

#[derive(Clone, Copy, PartialEq)]
pub enum Optimizer {
    Sgd,
    Adagrad,
    /// Per-parameter lazy Adam; hyperparameters ride in the variant so kernel
    /// signatures stay fixed. Handled by `adam_step`, not `apply_update`.
    Adam { beta1: f64, beta2: f64, eps: f64 },
    /// FTRL-Proximal; `alpha` is the step size (learning_rate, passed separately),
    /// `beta` the stabilizer. L1/L2 are folded into the update (not the gradient),
    /// so callers pass the pure data gradient. Handled by `ftrl_step`.
    Ftrl { beta: f64 },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    Logistic,
    Squared,
}

pub const ADAGRAD_EPS: f64 = 1e-10;

/// Numerically stable sigmoid; mirrors `_reference_train._sigmoid`.
pub fn sigmoid(s: f64) -> f64 {
    if s >= 0.0 {
        1.0 / (1.0 + (-s).exp())
    } else {
        let e = s.exp();
        e / (1.0 + e)
    }
}

/// dL/ds for one sample (logistic: y in {0, 1}).
pub fn loss_grad(loss: Loss, s: f64, y: f64) -> f64 {
    match loss {
        Loss::Logistic => sigmoid(s) - y,
        Loss::Squared => s - y,
    }
}

/// Apply one parameter update; `acc` is this parameter's AdaGrad accumulator
/// (ignored for SGD).
#[inline]
pub fn apply_update(theta: &mut f64, grad: f64, acc: &mut f64, lr: f64, opt: Optimizer) {
    match opt {
        Optimizer::Sgd => *theta -= lr * grad,
        Optimizer::Adagrad => {
            *acc += grad * grad;
            *theta -= lr * grad / (*acc + ADAGRAD_EPS).sqrt();
        }
        // Adam and FTRL carry extra per-parameter state; callers route them
        // through `adam_step` / `ftrl_step` and never reach these arms.
        Optimizer::Adam { .. } => unreachable!("Adam uses adam_step, not apply_update"),
        Optimizer::Ftrl { .. } => unreachable!("FTRL uses ftrl_step, not apply_update"),
    }
}

/// One lazy-Adam step for a single parameter (docs/optimization_spec.md), with
/// per-coordinate moment accumulators `m`, `v` and update count `t`. `t.powf`
/// via `beta.powf(*t)` matches the Python reference's `beta ** t`. Mirrors
/// `_reference_train._adam_scalar`/`_adam_array`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn adam_step(
    theta: &mut f64,
    grad: f64,
    m: &mut f64,
    v: &mut f64,
    t: &mut f64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
) {
    *t += 1.0;
    *m = beta1 * *m + (1.0 - beta1) * grad;
    *v = beta2 * *v + (1.0 - beta2) * grad * grad;
    let m_hat = *m / (1.0 - beta1.powf(*t));
    let v_hat = *v / (1.0 - beta2.powf(*t));
    *theta -= lr * m_hat / (v_hat.sqrt() + eps);
}

/// One FTRL-Proximal step for a single parameter (docs/optimization_spec.md).
/// Per-coordinate state `(z, n)`; `alpha` is the step size, `beta` the stabilizer,
/// `l1`/`l2` the regularization. The weight is reconstructed from `(z, n)` so
/// `theta` always holds the current FTRL weight. Mirrors `_ftrl_scalar`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn ftrl_step(theta: &mut f64, g: f64, z: &mut f64, n: &mut f64, alpha: f64, beta: f64, l1: f64, l2: f64) {
    let n_new = *n + g * g;
    let sigma = (n_new.sqrt() - n.sqrt()) / alpha;
    *z += g - sigma * *theta;
    *n = n_new;
    let zi = *z;
    *theta = if zi.abs() <= l1 {
        0.0
    } else {
        let sign = if zi < 0.0 { -1.0 } else { 1.0 };
        -(zi - sign * l1) / ((beta + n_new.sqrt()) / alpha + l2)
    };
}

/// Step one scalar parameter (e.g. the bias) from its (batch-mean) **data**
/// gradient `data_grad` plus regularization `l1`/`l2`. Each optimizer applies
/// its own regularization: SGD/AdaGrad/Adam fold `l2 * theta` into the gradient
/// (`l1` unused); FTRL uses `(z, n)` and folds `l1`/`l2` into its closed form.
/// Shared by the FM and FFM mini-batch flush paths.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn step_param(
    theta: &mut f64,
    data_grad: f64,
    l1: f64,
    l2: f64,
    acc: &mut f64,
    m: &mut f64,
    s: &mut f64,
    t: &mut f64,
    z: &mut f64,
    n: &mut f64,
    lr: f64,
    opt: Optimizer,
) {
    match opt {
        Optimizer::Ftrl { beta } => ftrl_step(theta, data_grad, z, n, lr, beta, l1, l2),
        Optimizer::Adam { beta1, beta2, eps } => {
            adam_step(theta, data_grad + l2 * *theta, m, s, t, lr, beta1, beta2, eps)
        }
        _ => apply_update(theta, data_grad + l2 * *theta, acc, lr, opt),
    }
}

/// Step one coordinate `theta[..]` from its (batch-mean) data gradient, indexing
/// the per-coordinate state arrays at `idx` only on the path that uses them — so
/// SGD/AdaGrad pass empty Adam/FTRL slices (moments/`z,n` are allocated only for
/// the active optimizer). See `step_param`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn step_coord(
    theta: &mut f64,
    data_grad: f64,
    l1: f64,
    l2: f64,
    acc: &mut f64,
    m: &mut [f64],
    s: &mut [f64],
    t: &mut [f64],
    z: &mut [f64],
    n: &mut [f64],
    idx: usize,
    lr: f64,
    opt: Optimizer,
) {
    match opt {
        Optimizer::Ftrl { beta } => {
            ftrl_step(theta, data_grad, &mut z[idx], &mut n[idx], lr, beta, l1, l2)
        }
        Optimizer::Adam { beta1, beta2, eps } => {
            adam_step(theta, data_grad + l2 * *theta, &mut m[idx], &mut s[idx], &mut t[idx], lr, beta1, beta2, eps)
        }
        _ => apply_update(theta, data_grad + l2 * *theta, acc, lr, opt),
    }
}

/// Per-coordinate Adam moment state (m, s, t) for one parameter set: scalar bias
/// plus linear (`w_len`) and factor (`v_len`) vectors. Empty vectors off the Adam
/// path (SGD/AdaGrad never index them; `step_coord` only touches them for Adam).
pub struct AdamState {
    pub m_w0: f64,
    pub s_w0: f64,
    pub t_w0: f64,
    pub m_w: Vec<f64>,
    pub s_w: Vec<f64>,
    pub t_w: Vec<f64>,
    pub m_v: Vec<f64>,
    pub s_v: Vec<f64>,
    pub t_v: Vec<f64>,
}

impl AdamState {
    pub fn new(adam: bool, w_len: usize, v_len: usize) -> Self {
        let z = |len: usize| if adam { vec![0.0; len] } else { Vec::new() };
        Self {
            m_w0: 0.0,
            s_w0: 0.0,
            t_w0: 0.0,
            m_w: z(w_len),
            s_w: z(w_len),
            t_w: z(w_len),
            m_v: z(v_len),
            s_v: z(v_len),
            t_v: z(v_len),
        }
    }
}

/// Per-coordinate FTRL state `(z, n)` for one parameter set: scalar bias plus
/// linear (`w_len`) and factor (`v_len`) vectors. Empty vectors off the FTRL path.
pub struct FtrlState {
    pub z_w0: f64,
    pub n_w0: f64,
    pub z_w: Vec<f64>,
    pub n_w: Vec<f64>,
    pub z_v: Vec<f64>,
    pub n_v: Vec<f64>,
}

impl FtrlState {
    pub fn new(ftrl: bool, w_len: usize, v_len: usize) -> Self {
        let z = |len: usize| if ftrl { vec![0.0; len] } else { Vec::new() };
        Self {
            z_w0: 0.0,
            n_w0: 0.0,
            z_w: z(w_len),
            n_w: z(w_len),
            z_v: z(v_len),
            n_v: z(v_len),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_basics() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-15);
        assert!(sigmoid(40.0) > 0.999_999);
        assert!(sigmoid(-40.0) < 1e-6);
    }

    #[test]
    fn adagrad_first_step_is_lr_times_sign() {
        // First update: theta -= lr * g / sqrt(g^2 + eps) ~= lr * sign(g)
        let mut theta = 0.0;
        let mut acc = 0.0;
        apply_update(&mut theta, 2.0, &mut acc, 0.1, Optimizer::Adagrad);
        assert!((theta + 0.1).abs() < 1e-6);
    }

    #[test]
    fn ftrl_first_step_no_reg() {
        // First step from theta=0, z=0, n=0, l1=l2=0, alpha=1, beta=1, g=2:
        // n_new=4, sigma=(2-0)/1=2, z += 2 - 2*0 = 2, theta = -2 / ((1+2)/1 + 0) = -2/3.
        let (mut theta, mut z, mut n) = (0.0, 0.0, 0.0);
        ftrl_step(&mut theta, 2.0, &mut z, &mut n, 1.0, 1.0, 0.0, 0.0);
        assert!((theta + 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(n, 4.0);
        assert!((z - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ftrl_l1_zeros_small_z() {
        // |z| <= l1 forces theta to exactly 0: g=0.5, l1=1.0 -> z=0.5 <= 1 -> 0.
        let (mut theta, mut z, mut n) = (0.0, 0.0, 0.0);
        ftrl_step(&mut theta, 0.5, &mut z, &mut n, 1.0, 1.0, 1.0, 0.0);
        assert_eq!(theta, 0.0);
    }

    #[test]
    fn adam_first_step_is_lr_times_sign() {
        // t=1: m_hat = m/(1-b1) = g, v_hat = v/(1-b2) = g^2, so
        // theta -= lr * g / (sqrt(g^2) + eps) ~= lr * sign(g).
        let (mut theta, mut m, mut v, mut t) = (0.0, 0.0, 0.0, 0.0);
        adam_step(&mut theta, 2.0, &mut m, &mut v, &mut t, 0.1, 0.9, 0.999, 1e-8);
        assert!((theta + 0.1).abs() < 1e-6);
        assert_eq!(t, 1.0);
    }
}
