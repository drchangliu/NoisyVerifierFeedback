import pytest

from nvf.benchmark.loader import _split_task_content, load_benchmark
from nvf.benchmark.schema import BenchmarkItem
from nvf.benchmark.splits import split_calibration_eval


def test_split_task_content_with_markers():
    content = """\
import os

# BEGIN PROMPT
def safe_read(path: str) -> str:
    '''Read file safely.'''
# BEGIN SOLUTION
    return open(path).read()
"""
    prompt, solution = _split_task_content(content)
    assert "safe_read" in prompt
    assert "open(path)" in solution
    assert "BEGIN SOLUTION" not in prompt
    assert "BEGIN SOLUTION" not in solution


def test_split_task_content_no_markers():
    content = "def hello():\n    return 'world'"
    prompt, solution = _split_task_content(content)
    assert prompt == content
    assert solution == ""


def test_benchmark_item_schema():
    item = BenchmarkItem(
        item_id="CWE-079_0",
        cwe_id="CWE-079",
        prompt="def render(html):",
        insecure_code="def render(html): return html",
        source="cweval",
    )
    assert item.cwe_id == "CWE-079"
    assert item.test_code is None


def test_split_deterministic():
    items = [
        BenchmarkItem(item_id=f"item_{i}", cwe_id="CWE-001", prompt="p", insecure_code="c")
        for i in range(20)
    ]
    cal1, eval1 = split_calibration_eval(items, 0.25, seed=42)
    cal2, eval2 = split_calibration_eval(items, 0.25, seed=42)
    assert [i.item_id for i in cal1] == [i.item_id for i in cal2]
    assert len(cal1) == 5
    assert len(eval1) == 15


def test_split_no_overlap():
    items = [
        BenchmarkItem(item_id=f"item_{i}", cwe_id="CWE-001", prompt="p", insecure_code="c")
        for i in range(10)
    ]
    cal, evl = split_calibration_eval(items, 0.3, seed=0)
    cal_ids = {i.item_id for i in cal}
    eval_ids = {i.item_id for i in evl}
    assert cal_ids.isdisjoint(eval_ids)
    assert len(cal_ids | eval_ids) == 10


def test_load_unknown_benchmark():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load_benchmark("nonexistent")


@pytest.mark.integration
def test_load_securityeval():
    items = load_benchmark("securityeval")
    assert len(items) > 100
    assert all(i.source == "securityeval" for i in items)
    assert all(i.cwe_id.startswith("CWE-") for i in items)
    assert all(i.prompt for i in items)
    assert all(i.insecure_code for i in items)


@pytest.mark.integration
def test_load_cweval():
    items = load_benchmark("cweval")
    assert len(items) >= 20  # ~25 Python tasks
    assert all(i.source == "cweval" for i in items)
    assert all(i.test_code for i in items)
