from __future__ import annotations

from nvf.agents.base import AgentTrace


def security_pass_at_k(trace: AgentTrace, k: int) -> bool:
    """True iff code at iteration k has no ground-truth vulnerabilities."""
    if k >= len(trace.iterations):
        return False
    return not trace.iterations[k].has_vulnerability


def aggregate_security_pass(traces: list[AgentTrace], k: int) -> float:
    if not traces:
        return 0.0
    return sum(security_pass_at_k(t, k) for t in traces) / len(traces)
