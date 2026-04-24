from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Finding:
    """Unified representation of a static analyzer finding."""

    analyzer: str  # "semgrep" | "bandit"
    rule_id: str
    cwe_ids: list[str] = field(default_factory=list)
    message: str = ""
    severity: str = "WARNING"
    confidence: str = "MEDIUM"
    line_start: int = 0
    line_end: int = 0
    code_snippet: str = ""
    raw: dict = field(default_factory=dict)
