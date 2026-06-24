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
        // Adam carries extra per-parameter state (m, v, t); callers route it
        // through `adam_step` and never reach this arm.
        Optimizer::Adam { .. } => unreachable!("Adam uses adam_step, not apply_update"),
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
    fn adam_first_step_is_lr_times_sign() {
        // t=1: m_hat = m/(1-b1) = g, v_hat = v/(1-b2) = g^2, so
        // theta -= lr * g / (sqrt(g^2) + eps) ~= lr * sign(g).
        let (mut theta, mut m, mut v, mut t) = (0.0, 0.0, 0.0, 0.0);
        adam_step(&mut theta, 2.0, &mut m, &mut v, &mut t, 0.1, 0.9, 0.999, 1e-8);
        assert!((theta + 0.1).abs() < 1e-6);
        assert_eq!(t, 1.0);
    }
}
