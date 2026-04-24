"""Multi-finding interaction model for the noisy-verifier framework.

Extends the single-finding model (Eq. 1) to handle multiple findings
surfaced simultaneously in a single iteration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MultiFindingResult:
    """Expected effect of surfacing a set of findings simultaneously."""
    expected_delta_jp: float
    expected_benefit: float
    expected_harm: float
    n_surfaced: int
    effective_q: float  # Probability at least one TP is fixed
    effective_r: float  # Probability at least one FP causes regression


def multi_finding_expected_effect(
    precisions: list[float],
    q: float,
    r: float,
    independent: bool = True,
) -> MultiFindingResult:
    """Compute expected JointPass change from surfacing multiple findings.

    With k findings surfaced simultaneously:
    - Each finding i is TP with probability p_i (independent)
    - If any TP is surfaced, the LLM fixes it with probability q
    - If any FP is surfaced, the LLM may regress with probability r per FP

    Under independence:
        P(at least one TP) = 1 - prod(1 - p_i)
        P(at least one FP) = 1 - prod(p_i)
        E[benefit] = P(≥1 TP) * q
        E[harm] = 1 - (1-r)^(expected FPs)

    Args:
        precisions: List of per-rule precisions for surfaced findings.
        q: Fix probability per iteration (model fixes at least one TP).
        r: Per-FP regression probability.
        independent: Assume findings are independent (default True).

    Returns:
        MultiFindingResult with expected effect decomposition.
    """
    if not precisions:
        return MultiFindingResult(0.0, 0.0, 0.0, 0, 0.0, 0.0)

    k = len(precisions)

    # Probability that at least one finding is a TP
    prob_no_tp = math.prod(1 - p for p in precisions)
    prob_at_least_one_tp = 1 - prob_no_tp

    # Expected number of FPs
    expected_fps = sum(1 - p for p in precisions)

    # Probability of at least one regression from FPs
    # Each FP independently causes regression with probability r
    prob_no_regression = (1 - r) ** expected_fps
    prob_regression = 1 - prob_no_regression

    # Expected benefit: at least one TP surfaced AND model fixes it
    benefit = prob_at_least_one_tp * q

    # Expected harm: at least one FP causes regression
    harm = prob_regression

    delta_jp = benefit - harm

    return MultiFindingResult(
        expected_delta_jp=delta_jp,
        expected_benefit=benefit,
        expected_harm=harm,
        n_surfaced=k,
        effective_q=prob_at_least_one_tp * q,
        effective_r=prob_regression,
    )


def optimal_subset(
    precisions: list[float],
    q: float,
    r: float,
) -> tuple[list[int], float]:
    """Find the optimal subset of findings to surface.

    Greedy approach: sort by precision descending, add findings while
    the marginal expected effect is positive.

    Returns:
        (indices of selected findings, expected delta JP)
    """
    if not precisions:
        return [], 0.0

    # Sort by precision descending
    indexed = sorted(enumerate(precisions), key=lambda x: -x[1])

    best_subset = []
    best_delta = 0.0

    current_subset = []
    for idx, p in indexed:
        # Try adding this finding
        candidate = current_subset + [p]
        result = multi_finding_expected_effect(candidate, q, r)

        if result.expected_delta_jp > best_delta:
            best_subset.append(idx)
            best_delta = result.expected_delta_jp
            current_subset = candidate
        else:
            break  # Diminishing returns

    return best_subset, best_delta


def single_finding_threshold_is_conservative(
    precisions: list[float],
    q: float,
    r: float,
) -> bool:
    """Show that the single-finding threshold τ* is conservative.

    When multiple findings are surfaced, the threshold for the marginal
    finding is LOWER than τ* = r/(q+r) because:
    - Adding a finding to a set that already contains TPs has less
      marginal harm (regression may already be caused by other FPs)
    - Adding a finding to a set with FPs has more marginal benefit
      (an additional TP provides another chance to fix)

    Returns True if the multi-finding optimal includes findings
    below the single-finding threshold.
    """
    tau_star = r / (q + r) if (q + r) > 0 else 0.0
    selected, _ = optimal_subset(precisions, q, r)
    selected_precisions = [precisions[i] for i in selected]

    # Check if any selected finding has precision below τ*
    return any(p < tau_star for p in selected_precisions)
