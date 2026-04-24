from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    pass


@dataclass
class CostTracker:
    """Tracks cumulative API spend and enforces a hard budget cap."""

    max_cost_usd: float = 100.0
    total_cost_usd: float = 0.0
    calls: list[dict] = field(default_factory=list)

    def record(self, model: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        self.total_cost_usd += cost_usd
        self.calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "cumulative_usd": self.total_cost_usd,
        })

    def check_budget(self) -> None:
        if self.total_cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.total_cost_usd:.2f} >= ${self.max_cost_usd:.2f}"
            )
