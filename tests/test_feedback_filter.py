from nvf.analyzers.finding import Finding
from nvf.feedback.filter import filter_by_precision
from nvf.feedback.formatter import format_findings


def _make_finding(rule_id: str, cwe: str = "CWE-79") -> Finding:
    return Finding(
        analyzer="semgrep",
        rule_id=rule_id,
        cwe_ids=[cwe],
        message=f"Test finding for {rule_id}",
        line_start=10,
    )


def test_filter_keeps_high_precision():
    findings = [_make_finding("rule-a"), _make_finding("rule-b")]
    precision_map = {"rule-a": 0.9, "rule-b": 0.3}
    result = filter_by_precision(findings, precision_map, threshold_tau=0.5)
    assert len(result) == 1
    assert result[0].rule_id == "rule-a"


def test_filter_suppresses_unknown_rules():
    findings = [_make_finding("unknown-rule")]
    precision_map = {}
    result = filter_by_precision(findings, precision_map, threshold_tau=0.5)
    assert len(result) == 0


def test_filter_boundary():
    findings = [_make_finding("rule-a")]
    precision_map = {"rule-a": 0.5}
    result = filter_by_precision(findings, precision_map, threshold_tau=0.5)
    assert len(result) == 1  # >= threshold


def test_format_natural_language():
    findings = [_make_finding("rule-a")]
    text = format_findings(findings, fmt="natural_language")
    assert "CWE-79" in text
    assert "Line 10" in text


def test_format_minimal():
    findings = [_make_finding("rule-a")]
    text = format_findings(findings, fmt="minimal")
    assert "rule-a" in text
