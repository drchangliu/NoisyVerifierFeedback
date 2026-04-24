from __future__ import annotations

from nvf.analyzers.base import Analyzer
from nvf.analyzers.finding import Finding


class CombinedAnalyzer(Analyzer):
    """Runs multiple analyzers and merges their findings."""

    def __init__(self, analyzers: list[Analyzer]):
        self.analyzers = analyzers

    def analyze(self, code: str, filename: str = "target.py") -> list[Finding]:
        all_findings = []
        for analyzer in self.analyzers:
            try:
                findings = analyzer.analyze(code, filename)
                all_findings.extend(findings)
            except Exception:
                continue  # Skip failed analyzers
        return all_findings
