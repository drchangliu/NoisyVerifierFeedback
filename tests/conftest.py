from __future__ import annotations

from nvf.analyzers.finding import Finding
from nvf.llm.client import LLMResponse


class MockLLMClient:
    """Mock LLM client for testing without API calls."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["def hello(): return 'world'"]
        self._call_count = 0
        self.model = "mock"

    def generate(self, messages: list[dict]) -> LLMResponse:
        content = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return LLMResponse(content=content, model="mock")


class MockAnalyzer:
    """Mock analyzer that returns predetermined findings."""

    def __init__(self, findings_per_call: list[list[Finding]] | None = None):
        self.findings_per_call = findings_per_call or [[]]
        self._call_count = 0

    def analyze(self, code: str, filename: str = "target.py") -> list[Finding]:
        findings = self.findings_per_call[
            min(self._call_count, len(self.findings_per_call) - 1)
        ]
        self._call_count += 1
        return findings
