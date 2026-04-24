#!/usr/bin/env python3
"""Generate synthetic experiment results for testing the analysis pipeline.

Simulates three conditions (naive, selective, llm_judge) with realistic
JointPass curves based on the calibration data and theoretical model.

Usage:
    python scripts/generate_synthetic_results.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def generate_synthetic_traces(
    condition: str,
    n_items: int = 20,
    max_iterations: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic traces with realistic behavior per condition."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Base probabilities by condition (modeling the theory)
    # naive: many FP fixes cause regressions, some TP fixes help
    # selective: fewer findings surfaced, but higher quality
    # llm_judge: moderate filtering, some judge errors
    params = {
        "naive": {
            "initial_func_pass": 0.60,
            "initial_sec_pass": 0.25,
            "func_improvement_per_iter": -0.02,  # Slight regression from FP fixes
            "sec_improvement_per_iter": 0.08,
            "n_findings_mean": 4.0,
            "n_feedback_ratio": 1.0,  # Show all
        },
        "selective": {
            "initial_func_pass": 0.60,
            "initial_sec_pass": 0.25,
            "func_improvement_per_iter": 0.03,  # Less regression
            "sec_improvement_per_iter": 0.12,
            "n_findings_mean": 4.0,
            "n_feedback_ratio": 0.3,  # Only high-precision rules
        },
        "llm_judge": {
            "initial_func_pass": 0.60,
            "initial_sec_pass": 0.25,
            "func_improvement_per_iter": 0.01,
            "sec_improvement_per_iter": 0.10,
            "n_findings_mean": 4.0,
            "n_feedback_ratio": 0.5,
        },
    }

    p = params[condition]
    cwe_ids = ["CWE-078", "CWE-079", "CWE-022", "CWE-502", "CWE-089",
               "CWE-095", "CWE-117", "CWE-327", "CWE-918", "CWE-020"]

    traces = []
    for i in range(n_items):
        item_id = f"{rng.choice(cwe_ids)}_{i}"
        iterations = []
        cost = 0.0

        for k in range(max_iterations + 1):
            # Probability of passing at this iteration
            func_prob = min(1.0, max(0.0,
                p["initial_func_pass"] + k * p["func_improvement_per_iter"]
                + np_rng.normal(0, 0.05)
            ))
            sec_prob = min(1.0, max(0.0,
                p["initial_sec_pass"] + k * p["sec_improvement_per_iter"]
                + np_rng.normal(0, 0.05)
            ))

            tests_passed = rng.random() < func_prob
            has_vulnerability = rng.random() > sec_prob

            n_findings = max(0, int(np_rng.poisson(p["n_findings_mean"] * max(0.5, 1 - k * 0.15))))
            n_feedback = int(n_findings * p["n_feedback_ratio"])

            iter_cost = rng.uniform(0.001, 0.008)
            cost += iter_cost

            iterations.append({
                "iteration": k,
                "code": f"# synthetic code for {item_id} iter {k}",
                "n_findings": n_findings,
                "n_feedback_shown": n_feedback,
                "finding_rules": [f"rule-{j}" for j in range(n_findings)],
                "feedback_rules": [f"rule-{j}" for j in range(n_feedback)],
                "tests_passed": tests_passed,
                "has_vulnerability": has_vulnerability,
                "cost_usd": iter_cost,
            })

            # Stop early if no findings to show
            if n_feedback == 0 and k > 0:
                break

        traces.append({
            "item_id": item_id,
            "condition": condition,
            "model": "synthetic",
            "total_cost_usd": cost,
            "iterations": iterations,
        })

    return traces


def main():
    output_base = Path("data/results")

    for condition in ["naive", "selective", "llm_judge"]:
        run_dir = output_base / f"synthetic_{condition}"
        run_dir.mkdir(parents=True, exist_ok=True)

        traces = generate_synthetic_traces(condition, n_items=20, max_iterations=5)

        traces_path = run_dir / "traces.jsonl"
        with open(traces_path, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")

        # Save config
        config = {
            "feedback": {"condition": condition},
            "agent": {"model": "synthetic", "max_iterations": 5},
            "benchmark": {"name": "synthetic"},
            "synthetic": True,
        }
        (run_dir / "config.yaml").write_text(
            json.dumps(config, indent=2)
        )

        print(f"Generated {len(traces)} traces for {condition} -> {traces_path}")


if __name__ == "__main__":
    main()
