#!/usr/bin/env python3
"""Generate publication-ready figures from experiment results.

Usage:
    python scripts/generate_figures.py data/results/synthetic_naive data/results/synthetic_selective data/results/synthetic_llm_judge
    python scripts/generate_figures.py data/results/naive_* --output-dir figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Publication-quality defaults
plt.rcParams.update({
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

CONDITION_STYLES = {
    "naive": {"color": "#d62728", "marker": "o", "label": "Naive (all findings)"},
    "selective": {"color": "#2ca02c", "marker": "s", "label": "Selective (τ-filtered)"},
    "llm_judge": {"color": "#1f77b4", "marker": "^", "label": "LLM Judge"},
}


def load_run(run_dir: Path) -> tuple[list[dict], str]:
    """Load traces and determine condition name."""
    traces = []
    with open(run_dir / "traces.jsonl") as f:
        for line in f:
            data = json.loads(line)
            if "error" not in data:
                traces.append(data)
    condition = traces[0]["condition"] if traces else run_dir.name
    return traces, condition


def compute_rates_with_ci(traces: list[dict], max_k: int = 6) -> pd.DataFrame:
    """Compute pass rates with Wilson confidence intervals at each k."""
    rows = []
    for k in range(max_k):
        joint_vals = []
        func_vals = []
        sec_vals = []

        for t in traces:
            if k < len(t["iterations"]):
                it = t["iterations"][k]
                tp = it.get("tests_passed", False)
                hv = it.get("has_vulnerability", True)
                joint_vals.append(tp and not hv)
                func_vals.append(bool(tp))
                sec_vals.append(not hv)

        if not joint_vals:
            continue

        def wilson_ci(vals):
            n = len(vals)
            p = sum(vals) / n
            z = 1.96
            denom = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denom
            margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
            return p, max(0, center - margin), min(1, center + margin)

        jp, jp_lo, jp_hi = wilson_ci(joint_vals)
        fp, fp_lo, fp_hi = wilson_ci(func_vals)
        sp, sp_lo, sp_hi = wilson_ci(sec_vals)

        rows.append({
            "k": k, "n": len(joint_vals),
            "joint_pass": jp, "joint_lo": jp_lo, "joint_hi": jp_hi,
            "functional": fp, "func_lo": fp_lo, "func_hi": fp_hi,
            "security": sp, "sec_lo": sp_lo, "sec_hi": sp_hi,
        })

    return pd.DataFrame(rows)


def plot_joint_pass_curves(all_data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Figure 1: JointPass@k curves across conditions."""
    fig, ax = plt.subplots()

    for condition, df in all_data.items():
        style = CONDITION_STYLES.get(condition, {"color": "gray", "marker": "x", "label": condition})
        ax.plot(df["k"], df["joint_pass"], marker=style["marker"],
                color=style["color"], label=style["label"])
        ax.fill_between(df["k"], df["joint_lo"], df["joint_hi"],
                        color=style["color"], alpha=0.15)

    ax.set_xlabel("Iteration k")
    ax.set_ylabel("JointPass@k")
    ax.set_title("Joint Functional + Security Pass Rate")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    path = output_dir / "joint_pass_curves.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close()
    print(f"Saved: {path}")


def plot_functional_vs_security(all_data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Figure 2: Functional vs Security tradeoff (Pareto frontier)."""
    fig, ax = plt.subplots()

    for condition, df in all_data.items():
        style = CONDITION_STYLES.get(condition, {"color": "gray", "marker": "x", "label": condition})
        ax.plot(df["functional"], df["security"], marker=style["marker"],
                color=style["color"], label=style["label"], alpha=0.8)

        # Label iteration numbers
        for _, row in df.iterrows():
            ax.annotate(f'k={int(row["k"])}', (row["functional"], row["security"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7)

    ax.set_xlabel("Functional Pass Rate")
    ax.set_ylabel("Security Pass Rate")
    ax.set_title("Functional vs. Security: Pareto Analysis")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Draw ideal point
    ax.plot(1.0, 1.0, "*", color="gold", markersize=15, zorder=5, label="Ideal")
    ax.grid(True, alpha=0.3)

    path = output_dir / "pareto_frontier.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close()
    print(f"Saved: {path}")


def plot_findings_feedback(all_data: dict[str, list[dict]], output_dir: Path) -> None:
    """Figure 3: Average findings vs feedback shown per iteration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for condition, traces in all_data.items():
        style = CONDITION_STYLES.get(condition, {"color": "gray", "marker": "x", "label": condition})

        max_k = max(len(t["iterations"]) for t in traces)
        findings_by_k = []
        feedback_by_k = []

        for k in range(max_k):
            findings = [t["iterations"][k].get("n_findings", 0) for t in traces if k < len(t["iterations"])]
            feedback = [t["iterations"][k].get("n_feedback_shown", 0) for t in traces if k < len(t["iterations"])]
            findings_by_k.append(np.mean(findings) if findings else 0)
            feedback_by_k.append(np.mean(feedback) if feedback else 0)

        ks = list(range(max_k))
        axes[0].plot(ks, findings_by_k, marker=style["marker"], color=style["color"], label=style["label"])
        axes[1].plot(ks, feedback_by_k, marker=style["marker"], color=style["color"], label=style["label"])

    axes[0].set_xlabel("Iteration k")
    axes[0].set_ylabel("Avg. Findings (Raw)")
    axes[0].set_title("Analyzer Findings per Iteration")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Iteration k")
    axes[1].set_ylabel("Avg. Feedback Shown")
    axes[1].set_title("Feedback Surfaced per Iteration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    path = output_dir / "findings_feedback.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close()
    print(f"Saved: {path}")


def plot_cost_comparison(all_data: dict[str, list[dict]], output_dir: Path) -> None:
    """Figure 4: Cost distribution by condition."""
    fig, ax = plt.subplots()

    data_for_box = []
    labels = []
    for condition, traces in all_data.items():
        costs = [t.get("total_cost_usd", 0) for t in traces]
        data_for_box.append(costs)
        style = CONDITION_STYLES.get(condition, {"label": condition})
        labels.append(style.get("label", condition))

    bp = ax.boxplot(data_for_box, tick_labels=labels, patch_artist=True)
    colors = [CONDITION_STYLES.get(c, {}).get("color", "gray") for c in all_data]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    ax.set_ylabel("Cost per Item (USD)")
    ax.set_title("API Cost Distribution by Condition")
    ax.grid(True, alpha=0.3, axis="y")

    path = output_dir / "cost_comparison.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close()
    print(f"Saved: {path}")


def plot_calibration_precision(output_dir: Path) -> None:
    """Figure 5: Per-rule precision from calibration data."""
    precision_path = Path("data/calibration/precision_map.json")
    if not precision_path.exists():
        print("Skipping calibration plot (no precision_map.json)")
        return

    data = json.loads(precision_path.read_text())

    # Sort by precision
    rules = sorted(data.items(), key=lambda x: x[1]["precision"], reverse=True)
    rule_names = [r[0].split(".")[-1][:30] for r, _ in rules]
    precisions = [v["precision"] for _, v in rules]
    ci_lo = [v["ci_lower"] for _, v in rules]
    ci_hi = [v["ci_upper"] for _, v in rules]
    n_samples = [v["n_samples"] for _, v in rules]

    fig, ax = plt.subplots(figsize=(10, max(4, len(rules) * 0.3)))
    y_pos = range(len(rules))

    # Color bars by precision level
    colors = ["#2ca02c" if p >= 0.5 else "#ff7f0e" if p > 0 else "#d62728" for p in precisions]

    ax.barh(y_pos, precisions, color=colors, alpha=0.7)
    ax.errorbar(precisions, y_pos,
                xerr=[np.array(precisions) - np.array(ci_lo),
                      np.array(ci_hi) - np.array(precisions)],
                fmt="none", color="black", alpha=0.5, capsize=3)

    # Annotate with sample sizes
    for i, (p, n) in enumerate(zip(precisions, n_samples)):
        ax.text(min(p + 0.02, 0.95), i, f"n={n}", va="center", fontsize=8, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(rule_names, fontsize=8)
    ax.set_xlabel("Estimated Precision")
    ax.set_title("Per-Rule Semgrep Precision (Calibration)")
    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.3, label="τ=0.5 threshold")
    ax.legend()
    ax.set_xlim(-0.05, 1.15)
    ax.invert_yaxis()

    path = output_dir / "calibration_precision.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready figures")
    parser.add_argument("run_dirs", nargs="+", help="Paths to run directories")
    parser.add_argument("--output-dir", default="figures", help="Output directory for figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs
    all_rates: dict[str, pd.DataFrame] = {}
    all_traces: dict[str, list[dict]] = {}

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        if not (run_dir / "traces.jsonl").exists():
            print(f"Skipping {run_dir} (no traces.jsonl)")
            continue

        traces, condition = load_run(run_dir)
        if not traces:
            continue

        rates = compute_rates_with_ci(traces)
        all_rates[condition] = rates
        all_traces[condition] = traces
        print(f"Loaded: {run_dir} ({len(traces)} traces, condition={condition})")

    if not all_rates:
        print("No data to plot.")
        return

    # Generate all figures
    plot_joint_pass_curves(all_rates, output_dir)
    plot_functional_vs_security(all_rates, output_dir)
    plot_findings_feedback(all_traces, output_dir)
    plot_cost_comparison(all_traces, output_dir)
    plot_calibration_precision(output_dir)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
