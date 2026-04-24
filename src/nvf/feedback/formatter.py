from __future__ import annotations

import json

from nvf.analyzers.finding import Finding


def format_findings(findings: list[Finding], fmt: str = "natural_language") -> str:
    """Format findings for LLM feedback in the specified format."""
    if fmt == "raw_sarif":
        return _format_raw_sarif(findings)
    elif fmt == "natural_language":
        return _format_natural_language(findings)
    elif fmt == "minimal":
        return _format_minimal(findings)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _format_raw_sarif(findings: list[Finding]) -> str:
    return json.dumps([f.raw for f in findings], indent=2)


def _format_natural_language(findings: list[Finding]) -> str:
    parts = []
    for f in findings:
        cwes = ", ".join(f.cwe_ids) if f.cwe_ids else "unknown CWE"
        parts.append(
            f"Line {f.line_start}: {f.message} ({cwes}, severity: {f.severity})"
        )
    return "\n".join(parts)


def _format_minimal(findings: list[Finding]) -> str:
    parts = []
    for f in findings:
        cwes = ", ".join(f.cwe_ids) if f.cwe_ids else "?"
        parts.append(f"Line {f.line_start}: {cwes}. {f.rule_id}")
    return "\n".join(parts)
