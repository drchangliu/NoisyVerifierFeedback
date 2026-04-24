from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    passed: bool
    output: str
    error: str
    timed_out: bool = False


@dataclass
class TestResults:
    """Results from running both functional and security tests."""

    functional_passed: bool
    security_passed: bool
    functional_output: str = ""
    security_output: str = ""


def run_code_with_test_file(
    code: str,
    test_file_content: str,
    timeout: int = 30,
) -> TestResults:
    """Run CWEval-style test file against generated code.

    CWEval test files use pytest marks: @pytest.mark.functionality and @pytest.mark.security.
    We run each category separately to get independent pass/fail.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the generated code as the module the test imports
        # CWEval tests import from the task module — we need to figure out the import name
        # For simplicity, write as a module and patch the test file's imports
        code_path = Path(tmpdir) / "solution.py"
        code_path.write_text(code)

        test_path = Path(tmpdir) / "test_solution.py"
        # Rewrite test imports to point to our solution module
        patched_test = _patch_test_imports(test_file_content)
        test_path.write_text(patched_test)

        # Run functional tests
        func_result = _run_pytest(
            test_path, tmpdir, timeout, marker="functionality"
        )

        # Run security tests
        sec_result = _run_pytest(
            test_path, tmpdir, timeout, marker="security"
        )

        return TestResults(
            functional_passed=func_result.passed,
            security_passed=sec_result.passed,
            functional_output=func_result.output,
            security_output=sec_result.output,
        )


def _run_pytest(
    test_path: Path, cwd: str, timeout: int, marker: str | None = None
) -> ExecutionResult:
    """Run pytest with optional marker filter."""
    cmd = ["python", "-m", "pytest", str(test_path), "-v", "--tb=short", "--no-header"]
    if marker:
        cmd.extend(["-m", marker])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return ExecutionResult(
            passed=result.returncode == 0,
            output=result.stdout,
            error=result.stderr,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            passed=False,
            output="",
            error="Execution timed out",
            timed_out=True,
        )


def _patch_test_imports(test_content: str) -> str:
    """Rewrite CWEval test file imports to use our solution module.

    CWEval tests typically import from a module like `cwe_078_0_task`.
    We replace that with `from solution import *`.
    """
    import re

    # Replace `from cwe_*_task import *` or specific imports
    patched = re.sub(
        r"from\s+cwe_\d+_\d+_task\s+import\s+.*",
        "from solution import *",
        test_content,
    )
    # Also handle `import cwe_*_task`
    patched = re.sub(
        r"import\s+cwe_\d+_\d+_task\b",
        "import solution as cwe_task",
        patched,
    )
    return patched


def run_tests(code: str, test_code: str, timeout: int = 10) -> ExecutionResult:
    """Run unit tests against generated code in a subprocess sandbox."""
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "solution.py"
        test_path = Path(tmpdir) / "test_solution.py"

        code_path.write_text(code)
        test_content = f"from solution import *\n\n{test_code}"
        test_path.write_text(test_content)

        return _run_pytest(test_path, tmpdir, timeout)
