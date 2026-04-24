from __future__ import annotations

from nvf.agents.base import AgentTrace


def total_cost(traces: list[AgentTrace]) -> float:
    """Total API cost across all traces."""
    return sum(t.total_cost_usd for t in traces)


def mean_cost_per_item(traces: list[AgentTrace]) -> float:
    if not traces:
        return 0.0
    return total_cost(traces) / len(traces)
