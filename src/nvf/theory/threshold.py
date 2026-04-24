from __future__ import annotations


def optimal_threshold(q: float, r: float) -> float:
    """Compute optimal precision threshold tau*.

    A finding should be surfaced iff per-rule precision p >= tau*, where:
      tau* = r / (q + r)

    Args:
        q: Probability the LLM correctly fixes a true positive.
        r: Probability that "fixing" a false positive introduces a regression.

    Returns:
        Optimal threshold tau* in [0, 1].
    """
    if q + r == 0:
        return 0.0
    return r / (q + r)


def expected_effect(p: float, q: float, r: float) -> float:
    """Expected effect on JointPass of surfacing a finding with TP probability p.

    Positive means surfacing helps; negative means it hurts.
      E[effect] = p * q - (1 - p) * r
    """
    return p * q - (1 - p) * r
