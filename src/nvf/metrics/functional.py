from __future__ import annotations

from nvf.agents.base import AgentTrace


def functional_pass_at_k(trace: AgentTrace, k: int) -> bool:
    """True iff code at iteration k passes all unit tests."""
    if k >= len(trace.iterations):
        return False
    return bool(trace.iterations[k].tests_passed)


def aggregate_functional_pass(traces: list[AgentTrace], k: int) -> float:
    if not traces:
        return 0.0
    return sum(functional_pass_at_k(t, k) for t in traces) / len(traces)
