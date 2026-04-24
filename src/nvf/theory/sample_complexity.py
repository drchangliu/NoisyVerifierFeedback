from __future__ import annotations

import math


def hoeffding_sample_size(epsilon: float, delta: float) -> int:
    """Minimum samples per rule to estimate precision within epsilon w.p. >= 1-delta.

    By Hoeffding's inequality: n >= ln(2/delta) / (2 * epsilon^2)
    """
    return math.ceil(math.log(2.0 / delta) / (2.0 * epsilon**2))


def bernstein_sample_size(epsilon: float, delta: float, p_hat: float) -> int:
    """Tighter bound using Bernstein's inequality for Bernoulli(p) with estimated p_hat.

    n >= (2 * p_hat * (1 - p_hat) * ln(2/delta)) / epsilon^2
       + (2/3) * ln(2/delta) / epsilon

    Useful when p_hat is far from 0.5 (low-precision or high-precision rules).
    """
    log_term = math.log(2.0 / delta)
    variance_term = 2.0 * p_hat * (1.0 - p_hat) * log_term / (epsilon**2)
    range_term = (2.0 / 3.0) * log_term / epsilon
    return math.ceil(variance_term + range_term)


def total_sample_complexity(
    num_rules: int, epsilon: float, delta: float, use_union_bound: bool = True
) -> int:
    """Total calibration samples needed across all rules.

    With union bound: use delta/R per rule so that simultaneous coverage holds.
    Returns O(log(R) / epsilon^2) per rule.
    """
    per_rule_delta = delta / num_rules if use_union_bound else delta
    per_rule_n = hoeffding_sample_size(epsilon, per_rule_delta)
    return per_rule_n * num_rules
