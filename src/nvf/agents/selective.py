from __future__ import annotations

from nvf.agents.base import CodeAgent
from nvf.analyzers.base import Analyzer
from nvf.analyzers.finding import Finding
from nvf.feedback.filter import filter_by_precision
from nvf.llm.client import LLMClient


class SelectiveAgent(CodeAgent):
    """Condition (b): Surface only findings with per-rule precision >= tau."""

    def __init__(
        self,
        llm: LLMClient,
        analyzer: Analyzer,
        precision_map: dict[str, float],
        threshold_tau: float = 0.5,
        max_iterations: int = 5,
        feedback_format: str = "natural_language",
    ):
        super().__init__(llm, analyzer, max_iterations, feedback_format)
        self.precision_map = precision_map
        self.threshold_tau = threshold_tau

    @property
    def condition_name(self) -> str:
        return "selective"

    def filter_findings(self, findings: list[Finding], code: str) -> list[Finding]:
        return filter_by_precision(findings, self.precision_map, self.threshold_tau)
