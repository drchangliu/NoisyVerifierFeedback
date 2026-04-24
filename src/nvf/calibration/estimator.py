from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from nvf.analyzers.finding import Finding


@dataclass
class RulePrecision:
    rule_id: str
    tp_count: int
    fp_count: int
    precision: float
    ci_lower: float  # Hoeffding lower bound
    ci_upper: float  # Hoeffding upper bound
    n_samples: int


def estimate_precision(
    findings_with_labels: list[tuple[Finding, bool]],
    delta: float = 0.05,
) -> dict[str, RulePrecision]:
    """Estimate per-rule precision from labeled findings.

    Args:
        findings_with_labels: List of (finding, is_true_positive) pairs.
        delta: Confidence level for Hoeffding interval (default 95%).

    Returns:
        Map from rule_id to RulePrecision.
    """
    rule_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0})

    for finding, is_tp in findings_with_labels:
        key = "tp" if is_tp else "fp"
        rule_counts[finding.rule_id][key] += 1

    results = {}
    for rule_id, counts in rule_counts.items():
        tp = counts["tp"]
        fp = counts["fp"]
        n = tp + fp
        precision = tp / n if n > 0 else 0.0

        # Hoeffding confidence interval
        epsilon = math.sqrt(math.log(2.0 / delta) / (2.0 * n)) if n > 0 else 1.0
        ci_lower = max(0.0, precision - epsilon)
        ci_upper = min(1.0, precision + epsilon)

        results[rule_id] = RulePrecision(
            rule_id=rule_id,
            tp_count=tp,
            fp_count=fp,
            precision=precision,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=n,
        )

    return results
