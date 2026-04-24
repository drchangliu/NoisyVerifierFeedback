from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from nvf.analyzers.base import Analyzer
from nvf.analyzers.finding import Finding
from nvf.benchmark.schema import BenchmarkItem
from nvf.feedback.formatter import format_findings
from nvf.llm.client import LLMClient


@dataclass
class IterationRecord:
    """Record of a single iteration in the agent loop."""

    iteration: int
    code: str
    findings: list[Finding]
    feedback_shown: list[Finding]
    tests_passed: bool | None = None
    has_vulnerability: bool | None = None
    cost_usd: float = 0.0


@dataclass
class AgentTrace:
    """Full trace of an agent run on a benchmark item."""

    item_id: str
    condition: str
    model: str = ""
    iterations: list[IterationRecord] = field(default_factory=list)
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        d = {
            "item_id": self.item_id,
            "condition": self.condition,
            "model": self.model,
            "total_cost_usd": self.total_cost_usd,
            "iterations": [],
        }
        for rec in self.iterations:
            d["iterations"].append({
                "iteration": rec.iteration,
                "code": rec.code,
                "n_findings": len(rec.findings),
                "n_feedback_shown": len(rec.feedback_shown),
                "finding_rules": [f.rule_id for f in rec.findings],
                "feedback_rules": [f.rule_id for f in rec.feedback_shown],
                "tests_passed": rec.tests_passed,
                "has_vulnerability": rec.has_vulnerability,
                "cost_usd": rec.cost_usd,
            })
        return d


def extract_code(response: str) -> str:
    """Extract Python code from LLM response, stripping markdown fences."""
    # Try to find ```python ... ``` block
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try generic ``` ... ``` block
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Return as-is (might already be raw code)
    return response.strip()


class CodeAgent(ABC):
    """Abstract base for the three experimental conditions."""

    def __init__(
        self,
        llm: LLMClient,
        analyzer: Analyzer,
        max_iterations: int = 5,
        feedback_format: str = "natural_language",
    ):
        self.llm = llm
        self.analyzer = analyzer
        self.max_iterations = max_iterations
        self.feedback_format = feedback_format

    @abstractmethod
    def filter_findings(self, findings: list[Finding], code: str) -> list[Finding]:
        """Apply the condition-specific filtering policy."""
        ...

    def run(self, item: BenchmarkItem) -> AgentTrace:
        """Run the iterative code generation loop and return the full trace."""
        from nvf.agents.prompts import FEEDBACK_PROMPT, GENERATION_PROMPT, SYSTEM_PROMPT

        trace = AgentTrace(
            item_id=item.item_id,
            condition=self.condition_name,
            model=self.llm.model,
        )

        # Initial generation
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": GENERATION_PROMPT.format(prompt=item.prompt)},
            ]
        )
        code = extract_code(response.content)
        trace.total_cost_usd += response.cost_usd

        for i in range(self.max_iterations + 1):
            findings = self.analyzer.analyze(code)
            feedback_shown = self.filter_findings(findings, code)

            record = IterationRecord(
                iteration=i,
                code=code,
                findings=findings,
                feedback_shown=feedback_shown,
                cost_usd=response.cost_usd if i == 0 else 0.0,
            )
            trace.iterations.append(record)

            # Stop if no feedback to give or last iteration
            if i == self.max_iterations or not feedback_shown:
                break

            # Generate fix
            feedback_text = format_findings(feedback_shown, fmt=self.feedback_format)
            response = self.llm.generate(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": FEEDBACK_PROMPT.format(
                        code=code, feedback=feedback_text
                    )},
                ]
            )
            code = extract_code(response.content)
            trace.total_cost_usd += response.cost_usd
            record.cost_usd = response.cost_usd

        return trace

    @property
    @abstractmethod
    def condition_name(self) -> str:
        ...
