//! Optimizers and loss gradients (docs/optimization_spec.md).
//!
//! Must match `python/modern_fm/_reference_train.py` operation-for-operation;
//! parity is enforced by tests/test_rust_train_parity.py.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    Sgd,
    Adagrad,
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
}
