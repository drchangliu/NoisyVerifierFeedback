from nvf.agents.base import extract_code
from nvf.agents.naive import NaiveAgent
from nvf.agents.selective import SelectiveAgent
from nvf.analyzers.finding import Finding
from nvf.benchmark.schema import BenchmarkItem
from conftest import MockAnalyzer, MockLLMClient


def _make_item() -> BenchmarkItem:
    return BenchmarkItem(
        item_id="test-001",
        cwe_id="CWE-079",
        prompt="def sanitize(html): ...",
        insecure_code="def sanitize(html): return html",
    )


def _make_finding(rule_id: str = "xss-rule") -> Finding:
    return Finding(
        analyzer="semgrep",
        rule_id=rule_id,
        cwe_ids=["CWE-079"],
        message="Potential XSS",
        line_start=1,
    )


def test_extract_code_markdown():
    text = "Here is the code:\n```python\ndef foo():\n    return 1\n```\nDone."
    assert extract_code(text) == "def foo():\n    return 1"


def test_extract_code_raw():
    text = "def foo():\n    return 1"
    assert extract_code(text) == text


def test_naive_agent_no_findings():
    llm = MockLLMClient(["def sanitize(html): return escape(html)"])
    analyzer = MockAnalyzer([[]])  # No findings
    agent = NaiveAgent(llm, analyzer, max_iterations=3)

    trace = agent.run(_make_item())
    assert trace.condition == "naive"
    assert len(trace.iterations) == 1  # Stops immediately with no findings
    assert trace.iterations[0].code == "def sanitize(html): return escape(html)"


def test_naive_agent_iterates_on_findings():
    finding = _make_finding()
    llm = MockLLMClient([
        "def sanitize(html): return html",       # Initial (vulnerable)
        "def sanitize(html): return escape(html)",  # Fixed
    ])
    analyzer = MockAnalyzer([
        [finding],  # First analysis finds issue
        [],         # Second analysis clean
    ])
    agent = NaiveAgent(llm, analyzer, max_iterations=3)

    trace = agent.run(_make_item())
    assert len(trace.iterations) >= 2
    assert trace.iterations[0].feedback_shown == [finding]


def test_selective_agent_filters():
    finding_high = _make_finding("high-precision-rule")
    finding_low = _make_finding("low-precision-rule")

    llm = MockLLMClient([
        "def sanitize(html): return html",
        "def sanitize(html): return escape(html)",
    ])
    analyzer = MockAnalyzer([
        [finding_high, finding_low],  # Both found
        [],
    ])
    precision_map = {"high-precision-rule": 0.9, "low-precision-rule": 0.2}
    agent = SelectiveAgent(
        llm, analyzer, precision_map, threshold_tau=0.5, max_iterations=3
    )

    trace = agent.run(_make_item())
    # Only high-precision finding should be shown
    assert len(trace.iterations[0].feedback_shown) == 1
    assert trace.iterations[0].feedback_shown[0].rule_id == "high-precision-rule"
    # But both were detected
    assert len(trace.iterations[0].findings) == 2


def test_naive_agent_respects_max_iterations():
    finding = _make_finding()
    llm = MockLLMClient(["def bad(): pass"])
    analyzer = MockAnalyzer([[finding]])  # Always returns findings
    agent = NaiveAgent(llm, analyzer, max_iterations=2)

    trace = agent.run(_make_item())
    # Should stop at max_iterations + 1 records (0, 1, 2)
    assert len(trace.iterations) <= 3


def test_trace_serialization():
    llm = MockLLMClient(["def foo(): pass"])
    analyzer = MockAnalyzer([[_make_finding()], []])
    agent = NaiveAgent(llm, analyzer, max_iterations=3)

    trace = agent.run(_make_item())
    d = trace.to_dict()
    assert d["item_id"] == "test-001"
    assert d["condition"] == "naive"
    assert isinstance(d["iterations"], list)
    assert "finding_rules" in d["iterations"][0]

    # Should be JSON-serializable
    import json
    json.dumps(d)
