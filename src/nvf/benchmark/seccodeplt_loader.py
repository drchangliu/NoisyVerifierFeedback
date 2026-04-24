"""Loader for SecCodePLT benchmark (SeCodePLT).

Parses the generate_dataset/data/{CWE}/ directories from the GitHub repo.
Each item has: metadata, setup code, code template, and testcases with
"capability" (functional) and "safety" (security) tests.

Source: https://github.com/ucsb-mlsec/SeCodePLT
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from nvf.benchmark.schema import BenchmarkItem

SECCODEPLT_DIR = Path("data/raw/SeCodePLT")


def load_seccodeplt(repo_dir: Path | str | None = None) -> list[BenchmarkItem]:
    """Load SecCodePLT Python items from local clone."""
    repo_dir = Path(repo_dir) if repo_dir else SECCODEPLT_DIR
    _ensure_clone(repo_dir)

    data_dir = repo_dir / "generate_dataset" / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"SecCodePLT data dir not found: {data_dir}")

    items = []
    for cwe_dir in sorted(data_dir.iterdir()):
        if not cwe_dir.is_dir():
            continue
        py_file = cwe_dir / "succeed_python_list.json"
        if not py_file.exists():
            continue

        cwe_num = cwe_dir.name
        with open(py_file) as f:
            raw_items = json.load(f)

        for i, raw in enumerate(raw_items):
            item = _parse_item(raw, cwe_num, i)
            if item:
                items.append(item)

    return items


def _ensure_clone(repo_dir: Path) -> None:
    if (repo_dir / "generate_dataset").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ucsb-mlsec/SeCodePLT.git", str(repo_dir)],
        check=True, capture_output=True,
    )


def _parse_item(raw: str, cwe_num: str, index: int) -> BenchmarkItem | None:
    """Parse a single SecCodePLT item string into a BenchmarkItem."""
    sections = _extract_sections(raw)

    if "TESTCASES" not in sections or not sections["TESTCASES"].strip():
        return None  # Skip items without test cases

    # Parse metadata
    meta = _parse_metadata(sections.get("METADATA", ""))
    if not meta:
        return None

    task_desc = meta.get("task_description", {})
    func_name = task_desc.get("function_name", f"func_{index}")
    description = task_desc.get("description", "")
    security_policy = task_desc.get("security_policy", "")

    # Build prompt from metadata
    prompt = _build_prompt(sections, task_desc)
    if not prompt:
        return None

    # Build test code
    setup = sections.get("SETUP", "")
    testcases = sections.get("TESTCASES", "")
    test_code = _build_test_code(setup, testcases, func_name)

    # Build insecure reference (code with vulnerable pattern)
    code_before = sections.get("CODE BEFORE", "")
    code = sections.get("CODE", "")
    code_after = sections.get("CODE AFTER", "")
    insecure_code = f"{setup}\n\n{code_before}\n{code}\n{code_after}".strip()

    item_id = f"seccodeplt-{cwe_num}-{index}"
    cwe_id = f"CWE-{cwe_num}"

    return BenchmarkItem(
        item_id=item_id,
        cwe_id=cwe_id,
        prompt=prompt,
        insecure_code=insecure_code if insecure_code else "",
        test_code=test_code,
        source="seccodeplt",
    )


def _extract_sections(raw: str) -> dict[str, str]:
    """Extract ## START X ## ... ## END X ## sections."""
    sections = {}
    for section in ["METADATA", "SETUP", "CODE BEFORE", "CODE", "CODE AFTER", "TESTCASES"]:
        pattern = rf"## START {section} ##\s*(.*?)\s*## END {section} ##"
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            sections[section] = match.group(1).strip()
    return sections


def _parse_metadata(meta_str: str) -> dict | None:
    """Parse JSON metadata, handling trailing commas."""
    if not meta_str:
        return None
    # Fix trailing commas
    meta_str = re.sub(r",\s*}", "}", meta_str)
    meta_str = re.sub(r",\s*]", "]", meta_str)
    try:
        return json.loads(meta_str)
    except json.JSONDecodeError:
        try:
            import ast
            return ast.literal_eval(meta_str)
        except Exception:
            return None


def _build_prompt(sections: dict, task_desc: dict) -> str:
    """Build an LLM prompt from SecCodePLT metadata and code template."""
    func_name = task_desc.get("function_name", "solution")
    description = task_desc.get("description", "")
    security_policy = task_desc.get("security_policy", "")
    params = task_desc.get("parameters", [])
    return_type = task_desc.get("return_type", "")

    setup = sections.get("SETUP", "")
    code_before = sections.get("CODE BEFORE", "")
    code_after = sections.get("CODE AFTER", "")

    # Build function signature
    param_str = ", ".join(
        f"{p.get('name', 'x')}: {p.get('type', 'str')}" for p in params
    ) if isinstance(params, list) else ""

    prompt = f"{setup}\n\n{code_before}\n"
    prompt += f"def {func_name}({param_str})"
    if return_type:
        prompt += f" -> {return_type}"
    prompt += ":\n"
    prompt += f"    '''\n    {description}\n"
    if security_policy:
        prompt += f"    Security: {security_policy}\n"
    prompt += "    '''\n"

    return prompt.strip()


def _build_test_code(setup: str, testcases_code: str, func_name: str) -> str:
    """Build executable test code from SecCodePLT testcases dict.

    SecCodePLT tests use a dict format:
        testcases = {"capability": [...], "safety": [...]}

    We convert this to pytest functions.
    """
    test_code = f"""
{setup}

{testcases_code}

import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)

def test_capability():
    \"\"\"Functional tests from SecCodePLT capability test cases.\"\"\"
    for case in testcases.get("capability", []):
        kwargs = case[0]
        expected = case[1]
        signal.alarm(10)
        try:
            result = {func_name}(**kwargs)
            if isinstance(expected, type) and issubclass(expected, Exception):
                assert False, f"Expected {{expected.__name__}} but got {{result}}"
            else:
                assert str(result) == str(expected), f"Expected {{expected}}, got {{result}}"
        except Exception as e:
            if isinstance(expected, type) and issubclass(expected, Exception):
                assert isinstance(e, expected), f"Expected {{expected.__name__}}, got {{type(e).__name__}}"
            else:
                raise
        finally:
            signal.alarm(0)

def test_safety():
    \"\"\"Security tests from SecCodePLT safety test cases.\"\"\"
    for case in testcases.get("safety", []):
        kwargs = case[0]
        expected = case[1]
        signal.alarm(10)
        try:
            result = {func_name}(**kwargs)
            if isinstance(expected, type) and issubclass(expected, Exception):
                assert False, f"Expected security exception {{expected.__name__}} but function succeeded with {{result}}"
            else:
                assert str(result) == str(expected), f"Expected {{expected}}, got {{result}}"
        except Exception as e:
            if isinstance(expected, type) and issubclass(expected, Exception):
                assert isinstance(e, expected), f"Expected {{expected.__name__}}, got {{type(e).__name__}}: {{e}}"
            else:
                raise
        finally:
            signal.alarm(0)
"""
    return test_code
