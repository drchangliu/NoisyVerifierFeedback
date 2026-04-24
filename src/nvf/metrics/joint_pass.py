from __future__ import annotations

from nvf.agents.base import AgentTrace


def joint_pass_at_k(trace: AgentTrace, k: int) -> bool:
    """True iff code at iteration k passes all unit tests AND has no ground-truth vulnerabilities."""
    if k >= len(trace.iterations):
        return False
    record = trace.iterations[k]
    return bool(record.tests_passed and not record.has_vulnerability)


def aggregate_joint_pass(traces: list[AgentTrace], k: int) -> float:
    """Fraction of traces that achieve joint pass at iteration k."""
    if not traces:
        return 0.0
    return sum(joint_pass_at_k(t, k) for t in traces) / len(traces)
