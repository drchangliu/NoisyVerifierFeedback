from __future__ import annotations

from nvf.agents.base import CodeAgent
from nvf.analyzers.finding import Finding


class NaiveAgent(CodeAgent):
    """Condition (a): Surface all analyzer findings each iteration."""

    @property
    def condition_name(self) -> str:
        return "naive"

    def filter_findings(self, findings: list[Finding], code: str) -> list[Finding]:
        return findings  # No filtering
