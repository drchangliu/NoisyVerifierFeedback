from __future__ import annotations

from nvf.analyzers.finding import Finding


def filter_by_precision(
    findings: list[Finding],
    precision_map: dict[str, float],
    threshold_tau: float,
) -> list[Finding]:
    """Return only findings whose rule has estimated precision >= tau.

    Rules not in the precision map are suppressed (conservative default).
    """
    return [
        f
        for f in findings
        if precision_map.get(f.rule_id, 0.0) >= threshold_tau
    ]
