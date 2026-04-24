"""Auto-labeling logic for calibration findings.

A finding is labeled as a true positive (TP) if:
1. The finding's CWE tags overlap with the benchmark item's known CWE, OR
2. The finding was raised on known-insecure code (and the rule category is relevant)

A finding is labeled as a false positive (FP) if:
1. It was raised on known-secure code, OR
2. It was raised on insecure code but the CWE doesn't match the known vulnerability
"""
from __future__ import annotations

import re

from nvf.analyzers.finding import Finding
from nvf.benchmark.schema import BenchmarkItem


def _normalize_cwe(cwe: str) -> str:
    """Extract numeric CWE ID: 'CWE-78: OS Command Injection' -> 'CWE-78'"""
    match = re.match(r"(CWE-\d+)", cwe)
    return match.group(1) if match else cwe


def label_finding(
    finding: Finding,
    item: BenchmarkItem,
    code_is_insecure: bool,
) -> bool:
    """Determine if a finding is a true positive.

    Args:
        finding: The static analyzer finding.
        item: The benchmark item (has ground-truth CWE).
        code_is_insecure: Whether the analyzed code is the known-insecure version.

    Returns:
        True if the finding is a true positive, False if false positive.
    """
    if not code_is_insecure:
        # Finding on secure code is always a false positive
        return False

    # On insecure code: TP if finding CWE matches the item's known CWE
    item_cwe = _normalize_cwe(item.cwe_id)
    finding_cwes = {_normalize_cwe(c) for c in finding.cwe_ids}

    return item_cwe in finding_cwes
