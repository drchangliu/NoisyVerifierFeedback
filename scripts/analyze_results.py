#!/usr/bin/env python3
"""Post-hoc analysis of experiment results.

Loads traces from one or more experiment runs, computes metrics,
and outputs summary tables.

Usage:
    python scripts/analyze_results.py data/results/synthetic_naive data/results/synthetic_selective data/results/synthetic_llm_judge
    python scripts/analyze_results.py data/results/naive_* data/results/selective_*
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def load_traces(run_dir: Path) -> tuple[list[dict], dict]:
    """Load traces and config from a run directory."""
    traces = []
    traces_path = run_dir / "traces.jsonl"
    if not traces_path.exists():
        return [], {}

    with open(traces_path) as f:
        for line in f:
            data = json.loads(line)
            if "error" not in data:
                traces.append(data)

    config = {}
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        import yaml
        config = yaml.safe_load(config_path.read_text())
        if config is None:
            # Might be JSON
            config = json.loads(config_path.read_text())

    return traces, config


def compute_metrics(traces: list[dict]) -> pd.DataFrame:
    """Compute per-iteration metrics across all traces."""
    if not traces:
        return pd.DataFrame()

    max_k = max(len(t["iterations"]) for t in traces)
    rows = []

    for k in range(max_k):
        joint_pass = 0
        func_pass = 0
        sec_pass = 0
        total = 0
        findings_total = 0
        feedback_total = 0

        for t in traces:
            if k < len(t["iterations"]):
                total += 1
                it = t["iterations"][k]
                tp = it.get("tests_passed", False)
                hv = it.get("has_vulnerability", True)

                if tp:
                    func_pass += 1
                if not hv:
                    sec_pass += 1
                if tp and not hv:
                    joint_pass += 1

                findings_total += it.get("n_findings", 0)
                feedback_total += it.get("n_feedback_shown", 0)

        rows.append({
            "k": k,
            "n": total,
            "joint_pass": joint_pass,
            "joint_pass_rate": joint_pass / total if total > 0 else 0,
            "functional_pass": func_pass,
            "functional_rate": func_pass / total if total > 0 else 0,
            "security_pass": sec_pass,
            "security_rate": sec_pass / total if total > 0 else 0,
            "mean_findings": findings_total / total if total > 0 else 0,
            "mean_feedback": feedback_total / total if total > 0 else 0,
        })

    return pd.DataFrame(rows)


def compute_confidence_interval(values: list[bool], confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    p_hat = sum(values) / n
    z = 1.96  # 95% CI
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


def compute_cost_summary(traces: list[dict]) -> dict:
    """Compute cost statistics."""
    costs = [t.get("total_cost_usd", 0) for t in traces]
    return {
        "total_cost": sum(costs),
        "mean_cost": np.mean(costs) if costs else 0,
        "median_cost": np.median(costs) if costs else 0,
        "max_cost": max(costs) if costs else 0,
    }


def print_comparison_table(all_metrics: dict[str, pd.DataFrame]) -> None:
    """Print a comparison table across conditions."""
    table = Table(title="JointPass@k by Condition")
    table.add_column("k", style="bold", justify="right")

    for condition in all_metrics:
        table.add_column(condition, justify="right")

    # Find max k across all conditions
    max_k = max(len(df) for df in all_metrics.values())

    for k in range(max_k):
        row = [str(k)]
        for condition, df in all_metrics.items():
            if k < len(df):
                r = df.iloc[k]
                row.append(f"{r['joint_pass_rate']:.1%} ({int(r['joint_pass'])}/{int(r['n'])})")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)


def print_detailed_table(condition: str, df: pd.DataFrame) -> None:
    """Print detailed metrics for a single condition."""
    table = Table(title=f"Detailed Metrics: {condition}")
    table.add_column("k", justify="right")
    table.add_column("N", justify="right")
    table.add_column("JointPass", justify="right")
    table.add_column("Functional", justify="right")
    table.add_column("Secure", justify="right")
    table.add_column("Avg Findings", justify="right")
    table.add_column("Avg Feedback", justify="right")

    for _, r in df.iterrows():
        table.add_row(
            str(int(r["k"])),
            str(int(r["n"])),
            f"{r['joint_pass_rate']:.1%}",
            f"{r['functional_rate']:.1%}",
            f"{r['security_rate']:.1%}",
            f"{r['mean_findings']:.1f}",
            f"{r['mean_feedback']:.1f}",
        )

    console.print(table)


def print_pareto_analysis(all_metrics: dict[str, pd.DataFrame]) -> None:
    """Print Pareto analysis: functional vs security tradeoff."""
    table = Table(title="Pareto Analysis (Final Iteration)")
    table.add_column("Condition", style="bold")
    table.add_column("Functional", justify="right")
    table.add_column("Secure", justify="right")
    table.add_column("JointPass", justify="right")
    table.add_column("Dominated?", justify="center")

    points = []
    for condition, df in all_metrics.items():
        if len(df) == 0:
            continue
        # Use last iteration with data
        last = df.iloc[-1]
        points.append((condition, last["functional_rate"], last["security_rate"], last["joint_pass_rate"]))

    # Check Pareto dominance
    for i, (cond, func, sec, jp) in enumerate(points):
        dominated = any(
            f2 >= func and s2 >= sec and (f2 > func or s2 > sec)
            for j, (_, f2, s2, _) in enumerate(points) if i != j
        )
        table.add_row(
            cond,
            f"{func:.1%}",
            f"{sec:.1%}",
            f"{jp:.1%}",
            "Yes" if dominated else "[green]No[/green]",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("run_dirs", nargs="+", help="Paths to run directories")
    parser.add_argument("--output", default=None, help="Save metrics CSV to this path")
    args = parser.parse_args()

    all_metrics: dict[str, pd.DataFrame] = {}
    all_costs: dict[str, dict] = {}

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            console.print(f"[yellow]Skipping {run_dir} (not found)[/yellow]")
            continue

        traces, config = load_traces(run_dir)
        if not traces:
            console.print(f"[yellow]Skipping {run_dir} (no traces)[/yellow]")
            continue

        condition = traces[0].get("condition", run_dir.name)
        label = f"{condition} ({run_dir.name})"

        df = compute_metrics(traces)
        all_metrics[label] = df
        all_costs[label] = compute_cost_summary(traces)

        console.print(f"\n[bold]Loaded:[/bold] {run_dir} -> {len(traces)} traces, condition={condition}")
        print_detailed_table(label, df)

    if len(all_metrics) > 1:
        console.print("\n")
        print_comparison_table(all_metrics)
        console.print("\n")
        print_pareto_analysis(all_metrics)

    # Cost summary
    cost_table = Table(title="Cost Summary")
    cost_table.add_column("Condition", style="bold")
    cost_table.add_column("Total $", justify="right")
    cost_table.add_column("Mean $/item", justify="right")
    cost_table.add_column("Median $/item", justify="right")

    for label, costs in all_costs.items():
        cost_table.add_row(
            label,
            f"${costs['total_cost']:.4f}",
            f"${costs['mean_cost']:.4f}",
            f"${costs['median_cost']:.4f}",
        )

    console.print(cost_table)

    # Save combined metrics
    if args.output:
        combined = []
        for label, df in all_metrics.items():
            df_copy = df.copy()
            df_copy["condition"] = label
            combined.append(df_copy)
        if combined:
            pd.concat(combined).to_csv(args.output, index=False)
            console.print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
