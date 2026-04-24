from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from nvf.analyzers.base import Analyzer
from nvf.analyzers.finding import Finding


class SemgrepAnalyzer(Analyzer):
    """Semgrep static analyzer wrapper."""

    def __init__(self, config: str = "auto", timeout: int = 30):
        self.config = config
        self.timeout = timeout

    def analyze(self, code: str, filename: str = "target.py") -> list[Finding]:
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / filename
            code_path.write_text(code)
            sarif_path = Path(tmpdir) / "output.sarif"

            try:
                subprocess.run(
                    [
                        "semgrep",
                        "--config",
                        self.config,
                        "--sarif",
                        "-o",
                        str(sarif_path),
                        str(code_path),
                    ],
                    capture_output=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return []

            if not sarif_path.exists():
                return []

            sarif = json.loads(sarif_path.read_text())
            return self._parse_sarif(sarif)

    def _parse_sarif(self, sarif: dict) -> list[Finding]:
        findings = []
        for run in sarif.get("runs", []):
            rules = {r["id"]: r for r in run.get("tool", {}).get("driver", {}).get("rules", [])}
            for result in run.get("results", []):
                rule_id = result.get("ruleId", "unknown")
                rule_meta = rules.get(rule_id, {})
                cwe_ids = self._extract_cwes(rule_meta)
                region = (
                    result.get("locations", [{}])[0]
                    .get("physicalLocation", {})
                    .get("region", {})
                )
                findings.append(
                    Finding(
                        analyzer="semgrep",
                        rule_id=rule_id,
                        cwe_ids=cwe_ids,
                        message=result.get("message", {}).get("text", ""),
                        severity=result.get("level", "warning").upper(),
                        confidence=rule_meta.get("properties", {}).get("confidence", "MEDIUM"),
                        line_start=region.get("startLine", 0),
                        line_end=region.get("endLine", 0),
                        code_snippet=region.get("snippet", {}).get("text", ""),
                        raw=result,
                    )
                )
        return findings

    def _extract_cwes(self, rule_meta: dict) -> list[str]:
        tags = rule_meta.get("properties", {}).get("tags", [])
        return [t for t in tags if t.startswith("CWE-")]
