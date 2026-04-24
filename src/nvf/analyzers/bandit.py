from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from nvf.analyzers.base import Analyzer
from nvf.analyzers.finding import Finding


class BanditAnalyzer(Analyzer):
    """Bandit static analyzer wrapper."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def analyze(self, code: str, filename: str = "target.py") -> list[Finding]:
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / filename
            code_path.write_text(code)

            try:
                result = subprocess.run(
                    ["bandit", "-f", "json", str(code_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return []

            if not result.stdout:
                return []

            data = json.loads(result.stdout)
            return self._parse_results(data)

    def _parse_results(self, data: dict) -> list[Finding]:
        findings = []
        for r in data.get("results", []):
            cwe_id = r.get("issue_cwe", {}).get("id")
            findings.append(
                Finding(
                    analyzer="bandit",
                    rule_id=r.get("test_id", "unknown"),
                    cwe_ids=[f"CWE-{cwe_id}"] if cwe_id else [],
                    message=r.get("issue_text", ""),
                    severity=r.get("issue_severity", "MEDIUM").upper(),
                    confidence=r.get("issue_confidence", "MEDIUM").upper(),
                    line_start=r.get("line_number", 0),
                    line_end=r.get("end_col_offset", 0),
                    code_snippet=r.get("code", ""),
                    raw=r,
                )
            )
        return findings
