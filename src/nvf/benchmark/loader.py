from __future__ import annotations

import re
import subprocess
from pathlib import Path

from nvf.benchmark.schema import BenchmarkItem

# Default local paths for cloned repos / cached data
CWEVAL_DIR = Path("data/raw/CWEval")
SECURITYEVAL_CACHE = Path("data/raw/securityeval")


def load_benchmark(name: str = "cweval") -> list[BenchmarkItem]:
    """Load benchmark dataset by name.

    Supported benchmarks:
    - cweval: CWEval with functional + security tests (primary)
    - securityeval: SecurityEval with CWE labels (all 121 items)
    - securityeval_tested: SecurityEval items with hand-written test oracles only
    - combined: CWEval + SecurityEval items with test oracles (largest set)
    """
    if name == "cweval":
        return load_cweval()
    elif name == "securityeval":
        return load_securityeval()
    elif name == "securityeval_tested":
        return load_securityeval(with_tests_only=True)
    elif name == "seccodeplt":
        from nvf.benchmark.seccodeplt_loader import load_seccodeplt
        return load_seccodeplt()
    elif name == "combined":
        cweval = load_cweval()
        seceval = load_securityeval(with_tests_only=True)
        return cweval + seceval
    elif name == "combined_all":
        cweval = load_cweval()
        seceval = load_securityeval(with_tests_only=True)
        from nvf.benchmark.seccodeplt_loader import load_seccodeplt
        seccodeplt = [i for i in load_seccodeplt() if i.item_id != "seccodeplt-77-0"]
        return cweval + seceval + seccodeplt
    else:
        raise ValueError(f"Unknown benchmark: {name}")


# ---------------------------------------------------------------------------
# CWEval
# ---------------------------------------------------------------------------

def load_cweval(repo_dir: Path | str | None = None) -> list[BenchmarkItem]:
    """Load CWEval Python tasks from local clone.

    Clones the repo to data/raw/CWEval if not already present.
    Each task is a pair: cwe_*_task.py + cwe_*_test.py under benchmark/core/py/.
    """
    repo_dir = Path(repo_dir) if repo_dir else CWEVAL_DIR
    _ensure_cweval_clone(repo_dir)

    py_dir = repo_dir / "benchmark" / "core" / "py"
    if not py_dir.exists():
        raise FileNotFoundError(f"CWEval Python benchmark dir not found: {py_dir}")

    # Find all task files and pair with test files
    task_files = sorted(py_dir.glob("cwe_*_task.py"))
    items = []
    for task_file in task_files:
        test_file = task_file.with_name(task_file.name.replace("_task.py", "_test.py"))
        item = _parse_cweval_task(task_file, test_file)
        if item:
            items.append(item)

    return items


def _ensure_cweval_clone(repo_dir: Path) -> None:
    """Clone CWEval repo if not already present."""
    if (repo_dir / "benchmark").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/Co1lin/CWEval.git", str(repo_dir)],
        check=True,
        capture_output=True,
    )


def _parse_cweval_task(task_file: Path, test_file: Path) -> BenchmarkItem | None:
    """Parse a CWEval task/test file pair into a BenchmarkItem."""
    task_content = task_file.read_text()

    # Extract CWE ID from filename: cwe_078_0_task.py -> CWE-078
    match = re.match(r"cwe_(\d+)_(\d+)_task\.py", task_file.name)
    if not match:
        return None
    cwe_num = match.group(1)
    cwe_id = f"CWE-{cwe_num}"
    task_index = match.group(2)
    item_id = f"{cwe_id}_{task_index}"

    # Extract prompt (before BEGIN SOLUTION) and secure code (after BEGIN SOLUTION)
    prompt, secure_code = _split_task_content(task_content)

    # Extract insecure code from test file if available
    insecure_code = ""
    test_code = ""
    if test_file.exists():
        test_content = test_file.read_text()
        test_code = test_content
        insecure_code = _extract_insecure_reference(test_content)

    return BenchmarkItem(
        item_id=item_id,
        cwe_id=cwe_id,
        prompt=prompt,
        insecure_code=insecure_code,
        secure_code=secure_code,
        test_code=test_code,
        source="cweval",
    )


def _split_task_content(content: str) -> tuple[str, str]:
    """Split task file into prompt and secure reference solution.

    CWEval uses markers like '# BEGIN SOLUTION' to separate prompt from solution.
    """
    # Try BEGIN PROMPT / BEGIN SOLUTION markers
    prompt_match = re.search(r"# BEGIN PROMPT", content)
    solution_match = re.search(r"# BEGIN SOLUTION", content)

    if solution_match:
        # Find the full line containing BEGIN SOLUTION
        line_start = content.rfind("\n", 0, solution_match.start())
        line_end = content.find("\n", solution_match.end())
        prompt_end = line_start + 1 if line_start >= 0 else solution_match.start()
        solution_start = line_end + 1 if line_end >= 0 else solution_match.end()

        prompt = content[:prompt_end].rstrip("\n") + "\n"
        secure_code = content[solution_start:]
        # Remove the marker line from prompt if BEGIN PROMPT exists
        if prompt_match:
            bp_line_end = content.find("\n", prompt_match.end())
            bp_start = bp_line_end + 1 if bp_line_end >= 0 else prompt_match.end()
            prompt = content[bp_start:prompt_end].rstrip("\n") + "\n"
        return prompt, secure_code

    # Fallback: treat entire file as prompt, no solution
    return content, ""


def _extract_insecure_reference(test_content: str) -> str:
    """Extract the insecure reference implementation from a CWEval test file.

    CWEval test files contain an unsafe implementation used to validate that
    the security tests correctly catch vulnerabilities.
    """
    # Look for common patterns: function definitions marked as unsafe/insecure
    # The pattern varies but typically there's a function defined before the test class
    lines = test_content.split("\n")
    in_unsafe = False
    unsafe_lines = []
    for line in lines:
        if re.match(r"^def\s+\w+.*unsafe|^def\s+\w+.*insecure", line, re.IGNORECASE):
            in_unsafe = True
        if in_unsafe:
            if line and not line[0].isspace() and unsafe_lines:
                break
            unsafe_lines.append(line)

    if unsafe_lines:
        return "\n".join(unsafe_lines)

    # Fallback: return empty
    return ""


# ---------------------------------------------------------------------------
# SecurityEval
# ---------------------------------------------------------------------------

def load_securityeval(with_tests_only: bool = False) -> list[BenchmarkItem]:
    """Load SecurityEval from HuggingFace (s2e-lab/SecurityEval).

    121 Python items with CWE labels and insecure reference code.
    Hand-written test oracles available for ~30 items.

    Args:
        with_tests_only: If True, only return items that have test oracles.
    """
    from datasets import load_dataset
    from nvf.benchmark.securityeval_tests import SECURITYEVAL_TESTS

    ds = load_dataset("s2e-lab/SecurityEval", split="train")
    items = []
    for row in ds:
        # ID format: CWE-078_codeql_1.py
        item_id = row["ID"]
        cwe_match = re.match(r"(CWE-\d+)", item_id)
        cwe_id = cwe_match.group(1) if cwe_match else "unknown"

        # Attach hand-written tests if available
        tests = SECURITYEVAL_TESTS.get(item_id)
        test_code = None
        if tests:
            test_code = (
                "import pytest\n\n"
                + tests.get("functional_tests", "")
                + "\n\n"
                + tests.get("security_tests", "")
            )

        if with_tests_only and not test_code:
            continue

        items.append(
            BenchmarkItem(
                item_id=item_id,
                cwe_id=cwe_id,
                prompt=row["Prompt"],
                insecure_code=row["Insecure_code"],
                test_code=test_code,
                source="securityeval",
            )
        )

    return items
