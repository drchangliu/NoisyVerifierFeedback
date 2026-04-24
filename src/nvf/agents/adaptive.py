from __future__ import annotations

from nvf.agents.base import AgentTrace, CodeAgent, IterationRecord, extract_code
from nvf.analyzers.base import Analyzer
from nvf.analyzers.finding import Finding
from nvf.benchmark.schema import BenchmarkItem
from nvf.feedback.filter import filter_by_precision
from nvf.feedback.formatter import format_findings
from nvf.llm.client import LLMClient
from nvf.theory.adaptive import AdaptiveThreshold


class AdaptiveAgent(CodeAgent):
    """Condition (d): Adaptive threshold that learns q and r online.

    Starts with a configurable prior (default: uniform Beta(1,1) giving
    tau*=0.5). After each item's feedback loop, updates q and r estimates
    based on whether the code improved or regressed. The threshold tau*
    adjusts automatically across items within a run.
    """

    def __init__(
        self,
        llm: LLMClient,
        analyzer: Analyzer,
        precision_map: dict[str, float],
        prior_strength: float = 1.0,
        max_iterations: int = 5,
        feedback_format: str = "natural_language",
    ):
        super().__init__(llm, analyzer, max_iterations, feedback_format)
        self.precision_map = precision_map
        self.adaptive = AdaptiveThreshold(
            q_alpha=prior_strength,
            q_beta=prior_strength,
            r_alpha=prior_strength,
            r_beta=prior_strength,
        )
        self.tau_history: list[float] = []

    @property
    def condition_name(self) -> str:
        return "adaptive"

    def filter_findings(self, findings: list[Finding], code: str) -> list[Finding]:
        tau = self.adaptive.tau_star
        return filter_by_precision(findings, self.precision_map, tau)

    def run(self, item: BenchmarkItem) -> AgentTrace:
        """Run the loop, then update q/r estimates from the outcome."""
        # Record tau before this item
        self.tau_history.append(self.adaptive.tau_star)

        # Run the standard loop
        trace = super().run(item)

        # Update q and r from the outcome
        if len(trace.iterations) >= 2:
            k0 = trace.iterations[0]
            k1 = trace.iterations[1]

            vuln_k0 = k0.has_vulnerability
            vuln_k1 = k1.has_vulnerability
            func_k1 = k1.tests_passed

            if vuln_k0 is not None and vuln_k1 is not None:
                if vuln_k0:
                    # Was vulnerable — did feedback help?
                    fixed = not vuln_k1 and (func_k1 is True)
                    self.adaptive.update(
                        was_tp=True, was_fixed=fixed, had_regression=False
                    )
                else:
                    # Was not vulnerable — did feedback cause regression?
                    regressed = vuln_k1 or (func_k1 is False)
                    self.adaptive.update(
                        was_tp=False, was_fixed=False, had_regression=regressed
                    )

        return trace
