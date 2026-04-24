from nvf.theory.multi_finding import (
    multi_finding_expected_effect,
    optimal_subset,
    single_finding_threshold_is_conservative,
)
from nvf.theory.adaptive import AdaptiveThreshold, simulate_adaptive_policy


def test_single_finding_matches_original():
    """Multi-finding with k=1 should match single-finding formula."""
    result = multi_finding_expected_effect([0.8], q=0.5, r=0.2)
    # p*q - (1-p)*r = 0.8*0.5 - 0.2*0.2 = 0.4 - 0.04 = 0.36
    assert abs(result.expected_delta_jp - 0.36) < 0.01


def test_empty_findings():
    result = multi_finding_expected_effect([], q=0.5, r=0.2)
    assert result.expected_delta_jp == 0.0
    assert result.n_surfaced == 0


def test_more_findings_increases_benefit():
    r1 = multi_finding_expected_effect([0.8], q=0.5, r=0.0)
    r2 = multi_finding_expected_effect([0.8, 0.8], q=0.5, r=0.0)
    # With r=0, more findings should help (more chances to surface a TP)
    assert r2.expected_benefit >= r1.expected_benefit


def test_low_precision_findings_add_harm():
    r1 = multi_finding_expected_effect([0.8], q=0.5, r=0.3)
    r2 = multi_finding_expected_effect([0.8, 0.1], q=0.5, r=0.3)
    # Adding a low-precision finding increases harm
    assert r2.expected_harm > r1.expected_harm


def test_optimal_subset_excludes_low_precision():
    precisions = [0.9, 0.8, 0.1, 0.05]
    selected, delta = optimal_subset(precisions, q=0.5, r=0.3)
    # Should not include the 0.1 and 0.05 precision findings
    selected_p = [precisions[i] for i in selected]
    assert all(p >= 0.5 for p in selected_p)


def test_adaptive_threshold_convergence():
    at = AdaptiveThreshold()
    # Simulate Haiku: q=0.25, r=0.0
    for _ in range(50):
        at.update(was_tp=True, was_fixed=True, had_regression=False)
    for _ in range(150):
        at.update(was_tp=True, was_fixed=False, had_regression=False)
    for _ in range(58):
        at.update(was_tp=False, was_fixed=False, had_regression=False)

    assert at.tau_star < 0.1  # Should converge near 0


def test_adaptive_threshold_qwen_like():
    at = AdaptiveThreshold()
    # Simulate Qwen: q=0.42, r=0.43
    for _ in range(42):
        at.update(was_tp=True, was_fixed=True, had_regression=False)
    for _ in range(58):
        at.update(was_tp=True, was_fixed=False, had_regression=False)
    for _ in range(43):
        at.update(was_tp=False, was_fixed=False, had_regression=True)
    for _ in range(57):
        at.update(was_tp=False, was_fixed=False, had_regression=False)

    assert 0.4 < at.tau_star < 0.6  # Should be near 0.5


def test_simulate_adaptive():
    history = simulate_adaptive_policy(
        precisions=[0.3, 0.5, 0.7, 0.9],
        q_true=0.25, r_true=0.0,
        n_iterations=100,
    )
    assert len(history) == 100
    # Should converge toward 0
    assert history[-1] < 0.3
