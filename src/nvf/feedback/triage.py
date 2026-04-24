from __future__ import annotations

from nvf.analyzers.finding import Finding
from nvf.llm.client import LLMClient

TRIAGE_PROMPT = """\
You are a security expert. Analyze whether this static analysis finding is a true positive.

Code context:
```python
{code}
```

Finding:
- Rule: {rule_id}
- Line {line_start}: {message}

Is this a genuine security vulnerability in the code above? Reply with ONLY "YES" or "NO".\
"""


def triage_findings(
    findings: list[Finding],
    code: str,
    judge_llm: LLMClient,
) -> list[Finding]:
    """Use a cheap LLM to triage findings, keeping only likely true positives."""
    kept = []
    for f in findings:
        prompt = TRIAGE_PROMPT.format(
            code=code,
            rule_id=f.rule_id,
            line_start=f.line_start,
            message=f.message,
        )
        response = judge_llm.generate(
            messages=[{"role": "user", "content": prompt}]
        )
        if response.content.strip().upper().startswith("YES"):
            kept.append(f)
    return kept
