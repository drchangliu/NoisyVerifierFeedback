from __future__ import annotations

import json
from pathlib import Path

from nvf.analyzers.finding import Finding


def save_labeled_set(
    labeled: list[tuple[Finding, bool]],
    path: str | Path,
) -> None:
    """Save labeled findings to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for finding, is_tp in labeled:
            record = {
                "rule_id": finding.rule_id,
                "cwe_ids": finding.cwe_ids,
                "message": finding.message,
                "line_start": finding.line_start,
                "is_true_positive": is_tp,
            }
            f.write(json.dumps(record) + "\n")


def load_labeled_set(path: str | Path) -> list[tuple[Finding, bool]]:
    """Load labeled findings from JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            finding = Finding(
                analyzer="labeled",
                rule_id=record["rule_id"],
                cwe_ids=record.get("cwe_ids", []),
                message=record.get("message", ""),
                line_start=record.get("line_start", 0),
            )
            results.append((finding, record["is_true_positive"]))
    return results
