import numpy as np
from modern_fm.losses import logistic_loss, sigmoid, softmax, softmax_loss, squared_loss
from scipy.special import expit


def test_sigmoid_matches_scipy():
    s = np.linspace(-30, 30, 101)
    np.testing.assert_allclose(sigmoid(s), expit(s), atol=1e-12)


def test_sigmoid_extreme_values_no_overflow():
    out = sigmoid(np.array([-1e4, 1e4]))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-12)


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)
    p = softmax(rng.normal(size=(50, 7)) * 10)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(p >= 0)


def test_logistic_loss_hand_computed():
    # s = 0 -> p = 0.5 -> loss = log(2) regardless of y
    np.testing.assert_allclose(
        logistic_loss(np.array([0.0, 1.0]), np.array([0.0, 0.0])), np.log(2.0)
    )


def test_logistic_loss_matches_direct_formula():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=200).astype(float)
    s = rng.normal(size=200) * 3
    p = expit(s)
    direct = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    np.testing.assert_allclose(logistic_loss(y, s), direct, atol=1e-10)


def test_logistic_loss_stable_at_extreme_logits():
    y = np.array([1.0, 0.0])
    s = np.array([1e4, -1e4])
    assert np.isfinite(logistic_loss(y, s))
    np.testing.assert_allclose(logistic_loss(y, s), 0.0, atol=1e-12)


def test_logistic_label_smoothing():
    # eps = 0.2 -> y_smooth in {0.1, 0.9}
    y, s = np.array([1.0]), np.array([2.0])
    eps = 0.2
    y_smooth = 0.9
    p = expit(s)
    expected = -(y_smooth * np.log(p) + (1 - y_smooth) * np.log(1 - p))
    np.testing.assert_allclose(logistic_loss(y, s, label_smoothing=eps), expected[0], atol=1e-12)


def test_softmax_loss_hand_computed():
    # uniform logits over 4 classes -> loss = log(4)
    logits = np.zeros((3, 4))
    y = np.array([0, 1, 3])
    np.testing.assert_allclose(softmax_loss(y, logits), np.log(4.0), atol=1e-12)


def test_softmax_loss_label_smoothing():
    logits = np.array([[2.0, 0.0, -1.0]])
    y = np.array([0])
    eps = 0.3
    log_p = logits - np.log(np.exp(logits).sum())
    targets = np.array([[1 - eps, eps / 2, eps / 2]])
    expected = -(targets * log_p).sum()
    np.testing.assert_allclose(softmax_loss(y, logits, label_smoothing=eps), expected, atol=1e-12)


def test_softmax_loss_stable_at_extreme_logits():
    logits = np.array([[1e4, 0.0], [-1e4, 0.0]])
    y = np.array([0, 1])
    out = softmax_loss(y, logits)
    assert np.isfinite(out)
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_sample_weight():
    y = np.array([1.0, 0.0])
    s = np.array([0.0, 5.0])
    w = np.array([1.0, 0.0])  # second sample fully down-weighted
    np.testing.assert_allclose(logistic_loss(y, s, sample_weight=w), np.log(2.0), atol=1e-12)


def test_squared_loss():
    np.testing.assert_allclose(
        squared_loss(np.array([1.0, 2.0]), np.array([2.0, 0.0])), 0.5 * (1 + 4) / 2
    )
