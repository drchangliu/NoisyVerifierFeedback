from nvf.analyzers.finding import Finding
from nvf.benchmark.schema import BenchmarkItem
from nvf.calibration.estimator import estimate_precision
from nvf.calibration.labeler import _normalize_cwe, label_finding


def _make_finding(rule_id: str, cwes: list[str]) -> Finding:
    return Finding(analyzer="semgrep", rule_id=rule_id, cwe_ids=cwes, message="test")


def _make_item(cwe_id: str) -> BenchmarkItem:
    return BenchmarkItem(item_id="test", cwe_id=cwe_id, prompt="p", insecure_code="c")


def test_normalize_cwe():
    assert _normalize_cwe("CWE-78") == "CWE-78"
    assert _normalize_cwe("CWE-78: OS Command Injection") == "CWE-78"
    assert _normalize_cwe("CWE-078") == "CWE-078"


def test_label_finding_tp_on_insecure():
    finding = _make_finding("rule-1", ["CWE-78: OS Command Injection"])
    item = _make_item("CWE-78")
    assert label_finding(finding, item, code_is_insecure=True) is True


def test_label_finding_fp_cwe_mismatch():
    finding = _make_finding("rule-1", ["CWE-502: Deserialization"])
    item = _make_item("CWE-78")
    assert label_finding(finding, item, code_is_insecure=True) is False


def test_label_finding_fp_on_secure():
    finding = _make_finding("rule-1", ["CWE-78"])
    item = _make_item("CWE-78")
    # Even matching CWE — if code is secure, it's FP
    assert label_finding(finding, item, code_is_insecure=False) is False


def test_label_finding_no_cwes():
    finding = _make_finding("rule-1", [])
    item = _make_item("CWE-78")
    assert label_finding(finding, item, code_is_insecure=True) is False


def test_estimate_precision_basic():
    f1 = _make_finding("rule-a", ["CWE-78"])
    f2 = _make_finding("rule-a", ["CWE-502"])
    f3 = _make_finding("rule-b", ["CWE-79"])

    labeled = [
        (f1, True),   # rule-a TP
        (f2, False),  # rule-a FP
        (f3, True),   # rule-b TP
    ]

    results = estimate_precision(labeled)
    assert results["rule-a"].precision == 0.5
    assert results["rule-a"].tp_count == 1
    assert results["rule-a"].fp_count == 1
    assert results["rule-b"].precision == 1.0
    assert results["rule-b"].tp_count == 1


def test_estimate_precision_confidence_interval():
    f = _make_finding("rule-a", ["CWE-78"])
    # 10 samples, all TP
    labeled = [(f, True)] * 10
    results = estimate_precision(labeled)
    rp = results["rule-a"]
    assert rp.precision == 1.0
    assert rp.ci_lower > 0.5  # With 10 samples, lower bound should be well above 0.5
    assert rp.ci_upper == 1.0
