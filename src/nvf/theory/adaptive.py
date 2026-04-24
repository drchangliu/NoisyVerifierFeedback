"""Adaptive threshold policy that updates τ online.

Instead of using a fixed τ from calibration, the adaptive policy
estimates q and r from the agent's own feedback-loop interactions
and adjusts τ* = r/(q+r) after each iteration.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdaptiveThreshold:
    """Online threshold estimator using Bayesian updating.

    Maintains running estimates of q (fix probability) and r (regression
    probability) from the agent's own interactions, with Beta priors.
    """

    # Beta prior parameters (1,1 = uniform prior)
    q_alpha: float = 1.0  # Prior successes for q
    q_beta: float = 1.0   # Prior failures for q
    r_alpha: float = 1.0  # Prior successes for r
    r_beta: float = 1.0   # Prior failures for r

    # Observed counts
    tp_fixed: int = 0
    tp_not_fixed: int = 0
    fp_regressed: int = 0
    fp_ok: int = 0

    def update(self, was_tp: bool, was_fixed: bool, had_regression: bool) -> None:
        """Update estimates after observing one feedback interaction.

        Args:
            was_tp: Whether the surfaced finding was a true positive.
            was_fixed: Whether the TP was successfully fixed.
            had_regression: Whether the fix attempt caused a regression.
        """
        if was_tp:
            if was_fixed:
                self.tp_fixed += 1
            else:
                self.tp_not_fixed += 1
        else:
            if had_regression:
                self.fp_regressed += 1
            else:
                self.fp_ok += 1

    @property
    def q_estimate(self) -> float:
        """Posterior mean of q (Beta posterior)."""
        alpha = self.q_alpha + self.tp_fixed
        beta = self.q_beta + self.tp_not_fixed
        return alpha / (alpha + beta)

    @property
    def r_estimate(self) -> float:
        """Posterior mean of r (Beta posterior)."""
        alpha = self.r_alpha + self.fp_regressed
        beta = self.r_beta + self.fp_ok
        return alpha / (alpha + beta)

    @property
    def tau_star(self) -> float:
        """Current optimal threshold estimate."""
        q = self.q_estimate
        r = self.r_estimate
        if q + r == 0:
            return 0.5  # Uncertain, use moderate threshold
        return r / (q + r)

    @property
    def n_observations(self) -> int:
        return self.tp_fixed + self.tp_not_fixed + self.fp_regressed + self.fp_ok

    def confidence_width(self) -> float:
        """Approximate 95% CI width for τ* using delta method."""
        n_q = self.tp_fixed + self.tp_not_fixed
        n_r = self.fp_regressed + self.fp_ok
        if n_q < 2 or n_r < 2:
            return 1.0  # Very uncertain

        q = self.q_estimate
        r = self.r_estimate

        # Variance of q and r (Beta posterior variance)
        var_q = (q * (1 - q)) / (n_q + 2)
        var_r = (r * (1 - r)) / (n_r + 2)

        # Delta method for τ* = r/(q+r):
        # dτ*/dq = -r/(q+r)^2, dτ*/dr = q/(q+r)^2
        denom = (q + r) ** 2 if (q + r) > 0 else 1.0
        var_tau = (r / denom) ** 2 * var_q + (q / denom) ** 2 * var_r

        return 1.96 * var_tau ** 0.5


def simulate_adaptive_policy(
    precisions: list[float],
    q_true: float,
    r_true: float,
    n_iterations: int = 50,
    prior_strength: float = 1.0,
) -> list[float]:
    """Simulate the adaptive threshold converging to τ*.

    Returns list of τ* estimates at each iteration.
    """
    import random

    at = AdaptiveThreshold(
        q_alpha=prior_strength,
        q_beta=prior_strength,
        r_alpha=prior_strength,
        r_beta=prior_strength,
    )

    tau_history = []
    for _ in range(n_iterations):
        # Pick a random finding
        p = random.choice(precisions)
        is_tp = random.random() < p

        if is_tp:
            fixed = random.random() < q_true
            at.update(was_tp=True, was_fixed=fixed, had_regression=False)
        else:
            regressed = random.random() < r_true
            at.update(was_tp=False, was_fixed=False, had_regression=regressed)

        tau_history.append(at.tau_star)

    return tau_history
