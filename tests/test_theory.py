from nvf.theory.sample_complexity import hoeffding_sample_size
from nvf.theory.threshold import expected_effect, optimal_threshold


def test_optimal_threshold_basic():
    # q=0.8 (80% fix rate), r=0.2 (20% regression rate)
    tau = optimal_threshold(q=0.8, r=0.2)
    assert abs(tau - 0.2) < 1e-10


def test_optimal_threshold_equal():
    tau = optimal_threshold(q=0.5, r=0.5)
    assert abs(tau - 0.5) < 1e-10


def test_expected_effect_positive():
    # High precision (p=0.9), good fix rate, low regression
    effect = expected_effect(p=0.9, q=0.8, r=0.1)
    assert effect > 0


def test_expected_effect_negative():
    # Low precision (p=0.1), so mostly false positives causing regressions
    effect = expected_effect(p=0.1, q=0.8, r=0.3)
    assert effect < 0


def test_expected_effect_at_threshold():
    # At the optimal threshold, expected effect should be ~0
    q, r = 0.7, 0.3
    tau = optimal_threshold(q, r)
    effect = expected_effect(p=tau, q=q, r=r)
    assert abs(effect) < 1e-10


def test_hoeffding_sample_size():
    n = hoeffding_sample_size(epsilon=0.1, delta=0.05)
    assert n > 0
    # Should be roughly 185 for these parameters
    assert 180 <= n <= 200


def test_hoeffding_tighter_epsilon_needs_more():
    n1 = hoeffding_sample_size(epsilon=0.1, delta=0.05)
    n2 = hoeffding_sample_size(epsilon=0.05, delta=0.05)
    assert n2 > n1
