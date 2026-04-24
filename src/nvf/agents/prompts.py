SYSTEM_PROMPT = """\
You are an expert Python developer. Write secure, correct code that passes all tests.
Pay attention to security vulnerabilities (CWE categories) and avoid common pitfalls.\
"""

GENERATION_PROMPT = """\
Write a Python function that satisfies the following specification.
Return ONLY the code, no explanations.

{prompt}\
"""

FEEDBACK_PROMPT = """\
Your code has the following security issues detected by static analysis.
Please fix these issues while maintaining functional correctness.
Return ONLY the corrected code, no explanations.

Current code:
```python
{code}
```

Issues found:
{feedback}\
"""
