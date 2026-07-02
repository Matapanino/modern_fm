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

/// Mutable per-coordinate Adam moment state (m, s, t) for one parameter set:
/// scalar bias plus linear (`w_len`) and factor (`v_len`) slices. The backing
/// storage is either kernel-local (`AdamBuf`, single all-epochs call) or NumPy
/// arrays handed across the PyO3 boundary so the epoch-driven early-stopping
/// loop round-trips the moments (layout mirrors `_reference_train.new_adam_state`;
/// the reference's `v_*` second moment is `s_*` here to avoid clashing with the
/// factor matrix). Empty slices off the Adam path (SGD/AdaGrad never index them;
/// `step_coord` only touches them for Adam).
pub struct AdamStateMut<'a> {
    pub m_w0: &'a mut f64,
    pub s_w0: &'a mut f64,
    pub t_w0: &'a mut f64,
    pub m_w: &'a mut [f64],
    pub s_w: &'a mut [f64],
    pub t_w: &'a mut [f64],
    pub m_v: &'a mut [f64],
    pub s_v: &'a mut [f64],
    pub t_v: &'a mut [f64],
}

/// Owned backing for `AdamStateMut` in in-crate tests (the PyO3 wrappers
/// assemble their own backing from NumPy arrays or local vectors): zeroed
/// vectors when Adam is the active optimizer, empty otherwise.
#[cfg(test)]
pub struct AdamBuf {
    m_w0: f64,
    s_w0: f64,
    t_w0: f64,
    m_w: Vec<f64>,
    s_w: Vec<f64>,
    t_w: Vec<f64>,
    m_v: Vec<f64>,
    s_v: Vec<f64>,
    t_v: Vec<f64>,
}

#[cfg(test)]
impl AdamBuf {
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

    pub fn view(&mut self) -> AdamStateMut<'_> {
        AdamStateMut {
            m_w0: &mut self.m_w0,
            s_w0: &mut self.s_w0,
            t_w0: &mut self.t_w0,
            m_w: &mut self.m_w,
            s_w: &mut self.s_w,
            t_w: &mut self.t_w,
            m_v: &mut self.m_v,
            s_v: &mut self.s_v,
            t_v: &mut self.t_v,
        }
    }
}

/// Mutable per-coordinate FTRL state `(z, n)` for one parameter set: scalar bias
/// plus linear (`w_len`) and factor (`v_len`) slices. Backing storage as in
/// `AdamStateMut` (layout mirrors `_reference_train.new_ftrl_state`). Empty
/// slices off the FTRL path.
pub struct FtrlStateMut<'a> {
    pub z_w0: &'a mut f64,
    pub n_w0: &'a mut f64,
    pub z_w: &'a mut [f64],
    pub n_w: &'a mut [f64],
    pub z_v: &'a mut [f64],
    pub n_v: &'a mut [f64],
}

/// Owned backing for `FtrlStateMut` in in-crate tests (see `AdamBuf`).
#[cfg(test)]
pub struct FtrlBuf {
    z_w0: f64,
    n_w0: f64,
    z_w: Vec<f64>,
    n_w: Vec<f64>,
    z_v: Vec<f64>,
    n_v: Vec<f64>,
}

#[cfg(test)]
impl FtrlBuf {
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

    pub fn view(&mut self) -> FtrlStateMut<'_> {
        FtrlStateMut {
            z_w0: &mut self.z_w0,
            n_w0: &mut self.n_w0,
            z_w: &mut self.z_w,
            n_w: &mut self.n_w,
            z_v: &mut self.z_v,
            n_v: &mut self.n_v,
        }
    }
}

/// View one class's slice of a (C, `len`)-row-major backing array; an empty
/// backing (inactive optimizer) yields an empty slice the update path never reads.
#[inline]
pub fn class_slice(buf: &mut [f64], c: usize, len: usize) -> &mut [f64] {
    if buf.is_empty() {
        Default::default()
    } else {
        &mut buf[c * len..(c + 1) * len]
    }
}

/// Mutable optimizer state for one extra array-shaped parameter group (FwFM's
/// field-pair matrix R): AdaGrad accumulator (always full-size) plus Adam
/// `(m, s, t)` and FTRL `(z, n)` slices (empty when their optimizer is off).
/// Coordinates step through `step_coord` exactly like `w`/`V` coordinates.
pub struct GroupStateMut<'a> {
    pub acc: &'a mut [f64],
    pub m: &'a mut [f64],
    pub s: &'a mut [f64],
    pub t: &'a mut [f64],
    pub z: &'a mut [f64],
    pub n: &'a mut [f64],
}

/// Multiclass counterpart of `GroupStateMut`: (C, ·)-row-major backing with
/// per-class views (`acc` always full-size; Adam/FTRL arrays empty-safe).
pub struct McGroupState<'a> {
    pub acc: &'a mut [f64],
    pub m: &'a mut [f64],
    pub s: &'a mut [f64],
    pub t: &'a mut [f64],
    pub z: &'a mut [f64],
    pub n: &'a mut [f64],
}

impl McGroupState<'_> {
    pub fn class_views(&mut self, c: usize, len: usize) -> GroupStateMut<'_> {
        GroupStateMut {
            acc: &mut self.acc[c * len..(c + 1) * len],
            m: class_slice(self.m, c, len),
            s: class_slice(self.s, c, len),
            t: class_slice(self.t, c, len),
            z: class_slice(self.z, c, len),
            n: class_slice(self.n, c, len),
        }
    }
}

/// Multiclass optimizer-state backing, all (C, ·) row-major: AdaGrad
/// accumulators plus Adam / FTRL per-coordinate state. The `acc_*` and `*_w0`
/// arrays are always full-size (w0-level state is only C floats); the large
/// `*_w` / `*_v` arrays are empty when their optimizer is inactive. The backing
/// is kernel-local or NumPy arrays round-tripped for early stopping (layouts
/// mirror the multiclass `new_adam_state` / `new_ftrl_state`).
pub struct McState<'a> {
    pub acc_w0: &'a mut [f64],
    pub acc_w: &'a mut [f64],
    pub acc_v: &'a mut [f64],
    pub m_w0: &'a mut [f64],
    pub s_w0: &'a mut [f64],
    pub t_w0: &'a mut [f64],
    pub m_w: &'a mut [f64],
    pub s_w: &'a mut [f64],
    pub t_w: &'a mut [f64],
    pub m_v: &'a mut [f64],
    pub s_v: &'a mut [f64],
    pub t_v: &'a mut [f64],
    pub z_w0: &'a mut [f64],
    pub n_w0: &'a mut [f64],
    pub z_w: &'a mut [f64],
    pub n_w: &'a mut [f64],
    pub z_v: &'a mut [f64],
    pub n_v: &'a mut [f64],
}

/// Owned backing for `McState` in in-crate tests: full-size AdaGrad
/// accumulators and w0-level arrays, large Adam/FTRL arrays zeroed when active
/// and empty otherwise.
#[cfg(test)]
pub struct McBuf {
    acc_w0: Vec<f64>,
    acc_w: Vec<f64>,
    acc_v: Vec<f64>,
    adam_w0: [Vec<f64>; 3],
    adam_w: [Vec<f64>; 3],
    adam_v: [Vec<f64>; 3],
    ftrl_w0: [Vec<f64>; 2],
    ftrl_w: [Vec<f64>; 2],
    ftrl_v: [Vec<f64>; 2],
}

#[cfg(test)]
impl McBuf {
    pub fn new(adam: bool, ftrl: bool, n_classes: usize, n: usize, v_len: usize) -> Self {
        let big = |on: bool, len: usize| if on { vec![0.0; n_classes * len] } else { Vec::new() };
        Self {
            acc_w0: vec![0.0; n_classes],
            acc_w: vec![0.0; n_classes * n],
            acc_v: vec![0.0; n_classes * v_len],
            adam_w0: std::array::from_fn(|_| vec![0.0; n_classes]),
            adam_w: std::array::from_fn(|_| big(adam, n)),
            adam_v: std::array::from_fn(|_| big(adam, v_len)),
            ftrl_w0: std::array::from_fn(|_| vec![0.0; n_classes]),
            ftrl_w: std::array::from_fn(|_| big(ftrl, n)),
            ftrl_v: std::array::from_fn(|_| big(ftrl, v_len)),
        }
    }

    pub fn view(&mut self) -> McState<'_> {
        let [m_w0, s_w0, t_w0] = &mut self.adam_w0;
        let [m_w, s_w, t_w] = &mut self.adam_w;
        let [m_v, s_v, t_v] = &mut self.adam_v;
        let [z_w0, n_w0] = &mut self.ftrl_w0;
        let [z_w, n_w] = &mut self.ftrl_w;
        let [z_v, n_v] = &mut self.ftrl_v;
        McState {
            acc_w0: &mut self.acc_w0,
            acc_w: &mut self.acc_w,
            acc_v: &mut self.acc_v,
            m_w0,
            s_w0,
            t_w0,
            m_w,
            s_w,
            t_w,
            m_v,
            s_v,
            t_v,
            z_w0,
            n_w0,
            z_w,
            n_w,
            z_v,
            n_v,
        }
    }
}

impl McState<'_> {
    /// Per-class mutable views for the flush loop: (acc_w0, acc_w, acc_v, adam,
    /// ftrl) for class `c` with `n` linear and `v_len` factor entries per class.
    pub fn class_views(
        &mut self,
        c: usize,
        n: usize,
        v_len: usize,
    ) -> (&mut f64, &mut [f64], &mut [f64], AdamStateMut<'_>, FtrlStateMut<'_>) {
        (
            &mut self.acc_w0[c],
            &mut self.acc_w[c * n..(c + 1) * n],
            &mut self.acc_v[c * v_len..(c + 1) * v_len],
            AdamStateMut {
                m_w0: &mut self.m_w0[c],
                s_w0: &mut self.s_w0[c],
                t_w0: &mut self.t_w0[c],
                m_w: class_slice(self.m_w, c, n),
                s_w: class_slice(self.s_w, c, n),
                t_w: class_slice(self.t_w, c, n),
                m_v: class_slice(self.m_v, c, v_len),
                s_v: class_slice(self.s_v, c, v_len),
                t_v: class_slice(self.t_v, c, v_len),
            },
            FtrlStateMut {
                z_w0: &mut self.z_w0[c],
                n_w0: &mut self.n_w0[c],
                z_w: class_slice(self.z_w, c, n),
                n_w: class_slice(self.n_w, c, n),
                z_v: class_slice(self.z_v, c, v_len),
                n_v: class_slice(self.n_v, c, v_len),
            },
        )
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
