from nvf.analyzers.finding import Finding
from nvf.benchmark.schema import BenchmarkItem
from nvf.execution.security_eval_runner import (
    _check_known_patterns,
    _check_syntax,
    evaluate_securityeval_code,
)


def _make_item(cwe: str) -> BenchmarkItem:
    return BenchmarkItem(
        item_id="test", cwe_id=cwe, prompt="p", insecure_code="c", source="securityeval"
    )


def test_check_syntax_valid():
    assert _check_syntax("def foo(): return 1") is True


def test_check_syntax_invalid():
    assert _check_syntax("def foo(: return") is False


def test_pattern_cwe078_shell_true():
    code = "subprocess.call(cmd, shell=True)"
    assert _check_known_patterns(code, "CWE-078") is True


def test_pattern_cwe078_safe():
    code = "subprocess.call(['ls', '-l'])"
    assert _check_known_patterns(code, "CWE-078") is False


def test_pattern_cwe089_sql_injection():
    code = 'cursor.execute(f"SELECT * FROM users WHERE id={user_id}")'
    assert _check_known_patterns(code, "CWE-089") is True


def test_pattern_cwe502_pickle():
    code = "data = pickle.loads(raw_data)"
    assert _check_known_patterns(code, "CWE-502") is True


def test_pattern_unknown_cwe():
    code = "print('hello')"
    assert _check_known_patterns(code, "CWE-999") is None


def test_evaluate_securityeval_code():
    item = _make_item("CWE-078")
    code = "import subprocess\nsubprocess.call(cmd, shell=True)"
    findings = [
        Finding(analyzer="bandit", rule_id="B602", cwe_ids=["CWE-78"], message="shell=True")
    ]
    result = evaluate_securityeval_code(code, item, findings)
    assert result.syntactically_valid is True
    assert result.has_known_vulnerability is True
    assert result.analyzer_findings_match_cwe is True


def test_evaluate_securityeval_safe_code():
    item = _make_item("CWE-078")
    code = "import subprocess\nsubprocess.call(['ls', '-l'])"
    findings = []
    result = evaluate_securityeval_code(code, item, findings)
    assert result.syntactically_valid is True
    assert result.has_known_vulnerability is False
    assert result.analyzer_findings_match_cwe is False
