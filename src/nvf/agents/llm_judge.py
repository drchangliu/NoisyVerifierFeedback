from __future__ import annotations

from nvf.agents.base import CodeAgent
from nvf.analyzers.base import Analyzer
from nvf.analyzers.finding import Finding
from nvf.feedback.triage import triage_findings
from nvf.llm.client import LLMClient


class LLMJudgeAgent(CodeAgent):
    """Condition (c): Use a cheap model to pre-triage findings."""

    def __init__(
        self,
        llm: LLMClient,
        analyzer: Analyzer,
        judge_llm: LLMClient,
        max_iterations: int = 5,
        feedback_format: str = "natural_language",
        use_cot: bool = False,
    ):
        super().__init__(llm, analyzer, max_iterations, feedback_format)
        self.judge_llm = judge_llm
        self.use_cot = use_cot

    @property
    def condition_name(self) -> str:
        return "llm_judge_cot" if self.use_cot else "llm_judge"

    def filter_findings(self, findings: list[Finding], code: str) -> list[Finding]:
        return triage_findings(findings, code, self.judge_llm, use_cot=self.use_cot)
