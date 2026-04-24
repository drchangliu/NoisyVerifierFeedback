from __future__ import annotations

import numpy as np


def compute_pareto_frontier(
    functional_rates: list[float],
    security_rates: list[float],
) -> tuple[list[float], list[float]]:
    """Compute the Pareto frontier from (functional, security) rate pairs.

    Returns the subset of points that are Pareto-optimal (no other point
    dominates on both axes).
    """
    points = sorted(zip(functional_rates, security_rates), reverse=True)
    frontier_f, frontier_s = [], []
    max_s = -1.0
    for f, s in points:
        if s > max_s:
            frontier_f.append(f)
            frontier_s.append(s)
            max_s = s
    return frontier_f, frontier_s


def sweep_threshold(
    tau_values: np.ndarray,
    precision_map: dict[str, float],
    q: float,
    r: float,
) -> dict[str, list[float]]:
    """Sweep threshold tau and compute expected functional/security rates.

    This is a simplified theoretical model; empirical results come from
    the actual experiment runs.
    """
    results: dict[str, list[float]] = {
        "tau": tau_values.tolist(),
        "expected_functional": [],
        "expected_security": [],
    }

    for tau in tau_values:
        surfaced = [p for p in precision_map.values() if p >= tau]
        suppressed = [p for p in precision_map.values() if p < tau]

        # Expected benefit from surfacing true positives
        benefit = sum(p * q for p in surfaced)
        # Expected harm from surfacing false positives
        harm = sum((1 - p) * r for p in surfaced)

        results["expected_functional"].append(1.0 - harm)
        results["expected_security"].append(benefit)

    return results
