#!/usr/bin/env python3
"""Aggregate results across multiple seeds and compute confidence intervals.

Groups runs by (model, condition, analyzer) and computes mean ± 95% CI
for JointPass@k across seeds.

Usage:
    python scripts/aggregate_seeds.py
    python scripts/aggregate_seeds.py --filter "combined"
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


def load_all_runs(results_dir: Path, filter_str: str | None = None) -> dict[str, list[dict]]:
    """Load all runs, grouped by (model_short, condition, analyzer)."""
    grouped: dict[str, list[dict]] = defaultdict(list)

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir() or "synthetic" in run_dir.name:
            continue

        traces_path = run_dir / "traces.jsonl"
        if not traces_path.exists():
            continue

        traces = []
        with open(traces_path) as f:
            for line in f:
                d = json.loads(line)
                if "error" not in d:
                    traces.append(d)
        if not traces:
            continue

        # Get config
        config_path = run_dir / "config.yaml"
        analyzer = "semgrep"
        feedback_format = "natural_language"
        if config_path.exists():
            cfg = yaml.safe_load(config_path.read_text())
            if cfg:
                analyzer = cfg.get("analyzer", {}).get("name", "semgrep")
                feedback_format = cfg.get("feedback", {}).get("format", "natural_language")

        condition = traces[0].get("condition", "?")
        model = traces[0].get("model", "?")

        # Shorten model name
        if "haiku" in model:
            model_short = "haiku"
        elif "sonnet" in model:
            model_short = "sonnet"
        elif "qwen" in model:
            model_short = "qwen3-8b"
        else:
            model_short = model[:15]

        # Extract tau for selective
        tau = ""
        if condition == "selective":
            if cfg:
                tau = str(cfg.get("selective", {}).get("threshold_tau", "0.5"))

        key = f"{model_short}|{condition}|{analyzer}|{feedback_format}"
        if tau and condition == "selective":
            key = f"{model_short}|{condition}(τ={tau})|{analyzer}|{feedback_format}"

        if filter_str and filter_str not in key:
            continue

        # Compute per-run JointPass@0 and final
        n_items = len(traces)
        if n_items < 10:  # Skip small test runs
            continue

        jp_k0 = 0
        jp_final = 0
        for t in traces:
            iters = t.get("iterations", [])
            if iters:
                it0 = iters[0]
                if it0.get("tests_passed") and not it0.get("has_vulnerability"):
                    jp_k0 += 1
                # Final iteration
                it_final = iters[-1]
                if it_final.get("tests_passed") and not it_final.get("has_vulnerability"):
                    jp_final += 1

        grouped[key].append({
            "run_dir": run_dir.name,
            "n_items": n_items,
            "jp_k0": jp_k0 / n_items,
            "jp_final": jp_final / n_items,
            "cost": sum(t.get("total_cost_usd", 0) for t in traces),
        })

    return dict(grouped)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a single run."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default=None, help="Filter keys containing this string")
    parser.add_argument("--results-dir", default="data/results", help="Results directory")
    args = parser.parse_args()

    grouped = load_all_runs(Path(args.results_dir), args.filter)

    if not grouped:
        print("No matching runs found.")
        return

    # Summary table
    table = Table(title="Multi-Seed Results (mean ± 95% CI)")
    table.add_column("Configuration", style="cyan")
    table.add_column("Seeds", justify="right")
    table.add_column("JointPass@0", justify="right")
    table.add_column("JP Final", justify="right")
    table.add_column("Avg Cost", justify="right")

    rows = []
    for key in sorted(grouped.keys()):
        runs = grouped[key]
        n_seeds = len(runs)

        jp0_values = [r["jp_k0"] for r in runs]
        jpf_values = [r["jp_final"] for r in runs]
        costs = [r["cost"] for r in runs]

        jp0_mean = np.mean(jp0_values)
        jp0_std = np.std(jp0_values, ddof=1) if n_seeds > 1 else 0
        jp0_ci = 1.96 * jp0_std / np.sqrt(n_seeds) if n_seeds > 1 else 0

        jpf_mean = np.mean(jpf_values)
        jpf_std = np.std(jpf_values, ddof=1) if n_seeds > 1 else 0
        jpf_ci = 1.96 * jpf_std / np.sqrt(n_seeds) if n_seeds > 1 else 0

        cost_mean = np.mean(costs)

        parts = key.split("|")
        label = f"{parts[0]} / {parts[1]} / {parts[2]}"
        if parts[3] != "natural_language":
            label += f" / {parts[3]}"

        table.add_row(
            label,
            str(n_seeds),
            f"{jp0_mean:.1%} ± {jp0_ci:.1%}",
            f"{jpf_mean:.1%} ± {jpf_ci:.1%}",
            f"${cost_mean:.4f}",
        )

        rows.append({
            "key": key,
            "label": label,
            "n_seeds": n_seeds,
            "jp0_mean": jp0_mean,
            "jp0_ci": jp0_ci,
            "jpf_mean": jpf_mean,
            "jpf_ci": jpf_ci,
            "cost_mean": cost_mean,
        })

    console.print(table)

    # Save as CSV
    import csv
    output_path = Path(args.results_dir) / "aggregated_results.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
