from nvf.agents.base import AgentTrace, IterationRecord
from nvf.metrics.joint_pass import aggregate_joint_pass, joint_pass_at_k


def _make_trace(iterations: list[tuple[bool, bool]]) -> AgentTrace:
    """Create a trace from (tests_passed, has_vulnerability) tuples."""
    trace = AgentTrace(item_id="test", condition="test")
    for i, (tp, hv) in enumerate(iterations):
        trace.iterations.append(
            IterationRecord(
                iteration=i,
                code="pass",
                findings=[],
                feedback_shown=[],
                tests_passed=tp,
                has_vulnerability=hv,
            )
        )
    return trace


def test_joint_pass_both_true():
    trace = _make_trace([(True, False)])  # passes tests, no vulnerability
    assert joint_pass_at_k(trace, 0) is True


def test_joint_pass_fails_tests():
    trace = _make_trace([(False, False)])
    assert joint_pass_at_k(trace, 0) is False


def test_joint_pass_has_vulnerability():
    trace = _make_trace([(True, True)])
    assert joint_pass_at_k(trace, 0) is False


def test_joint_pass_out_of_range():
    trace = _make_trace([(True, False)])
    assert joint_pass_at_k(trace, 5) is False


def test_aggregate():
    traces = [
        _make_trace([(True, False)]),   # joint pass
        _make_trace([(True, True)]),    # fails security
        _make_trace([(False, False)]),  # fails functional
    ]
    rate = aggregate_joint_pass(traces, k=0)
    assert abs(rate - 1 / 3) < 1e-10
