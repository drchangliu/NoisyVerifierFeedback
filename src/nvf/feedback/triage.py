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

TRIAGE_COT_PROMPT = """\
You are a security expert triaging a static analysis finding. Decide whether it
represents a genuine vulnerability in the code as written.

Code context:
```python
{code}
```

Finding:
- Rule: {rule_id}
- Line {line_start}: {message}

Think step by step:

1. What does this rule pattern-match on? Is it a broad / over-approximating
   rule (e.g. flags every subprocess call) or a narrow rule that already
   considers context?
2. What does the code at line {line_start} actually do? Identify the exact
   API, arguments, and surrounding control flow.
3. Are there mitigations in the code that defuse the rule's concern (list
   arguments instead of shell strings, parameterized SQL, input validation,
   safe key sizes, allowlist checks, etc.)?
4. Is the data at this point attacker-controlled, or does it come from a
   trusted source?
5. Final verdict: is this a true positive (genuine vulnerability that needs
   fixing) or a false positive (rule fires but the code is safe)?

End your response with a single line in exactly this format:

VERDICT: YES

or

VERDICT: NO
"""


def _parse_cot_verdict(text: str) -> bool:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            return "YES" in line.upper()
    return text.strip().upper().startswith("YES")


def triage_findings(
    findings: list[Finding],
    code: str,
    judge_llm: LLMClient,
    use_cot: bool = False,
) -> list[Finding]:
    """Use a cheap LLM to triage findings, keeping only likely true positives.

    When `use_cot` is True, the judge is prompted to reason step-by-step
    before emitting a `VERDICT: YES/NO` line. This corresponds to the
    SAST-Genius / ZeroFalse-style chain-of-thought triage baseline.
    """
    template = TRIAGE_COT_PROMPT if use_cot else TRIAGE_PROMPT
    kept = []
    for f in findings:
        prompt = template.format(
            code=code,
            rule_id=f.rule_id,
            line_start=f.line_start,
            message=f.message,
        )
        response = judge_llm.generate(
            messages=[{"role": "user", "content": prompt}]
        )
        if use_cot:
            keep = _parse_cot_verdict(response.content)
        else:
            keep = response.content.strip().upper().startswith("YES")
        if keep:
            kept.append(f)
    return kept
