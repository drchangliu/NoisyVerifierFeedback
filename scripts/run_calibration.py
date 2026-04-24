#!/usr/bin/env python3
"""Build per-rule precision calibration map.

Runs Semgrep on both insecure and secure code from the calibration split,
labels each finding as TP/FP based on CWE overlap with ground truth,
then estimates per-rule precision with confidence intervals.

Usage:
    python scripts/run_calibration.py
    python scripts/run_calibration.py --benchmark securityeval --output data/calibration/seceval_precision.json
    python scripts/run_calibration.py --benchmark both
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from nvf.analyzers.bandit import BanditAnalyzer
from nvf.analyzers.combined import CombinedAnalyzer
from nvf.analyzers.semgrep import SemgrepAnalyzer
from nvf.benchmark.loader import load_benchmark
from nvf.benchmark.splits import split_calibration_eval
from nvf.calibration.estimator import estimate_precision
from nvf.calibration.labeler import label_finding
from nvf.calibration.labeled_set import save_labeled_set

console = Console()


def _build_analyzer(name: str = "semgrep"):
    if name == "semgrep":
        return SemgrepAnalyzer(config="auto", timeout=60)
    elif name == "bandit":
        return BanditAnalyzer(timeout=30)
    elif name == "combined":
        return CombinedAnalyzer([
            SemgrepAnalyzer(config="auto", timeout=60),
            BanditAnalyzer(timeout=30),
        ])
    else:
        raise ValueError(f"Unknown analyzer: {name}")


def run_calibration(
    benchmark_name: str,
    output_path: Path,
    seed: int = 42,
    calibration_fraction: float = 0.25,
    use_full_dataset: bool = False,
    analyzer_name: str = "semgrep",
) -> None:
    """Run the full calibration pipeline."""

    console.print(f"[bold]Loading benchmark:[/bold] {benchmark_name}")
    items = load_benchmark(benchmark_name)

    if use_full_dataset:
        cal_items = items
        console.print(f"Using full dataset: {len(cal_items)} items")
    else:
        cal_items, _ = split_calibration_eval(items, calibration_fraction, seed)
        console.print(f"Calibration split: {len(cal_items)} items (of {len(items)} total)")

    analyzer = _build_analyzer(analyzer_name)

    # Collect labeled findings
    all_labeled: list[tuple] = []
    items_with_findings = 0

    for item in tqdm(cal_items, desc="Analyzing"):
        item_findings = []

        # Analyze insecure code (if available)
        if item.insecure_code and item.insecure_code.strip():
            findings = analyzer.analyze(item.insecure_code)
            for f in findings:
                is_tp = label_finding(f, item, code_is_insecure=True)
                item_findings.append((f, is_tp))

        # Analyze secure code (if available) — findings here are always FP
        if item.secure_code and item.secure_code.strip():
            # Need full function for valid Python
            full_secure = item.prompt + item.secure_code
            findings = analyzer.analyze(full_secure)
            for f in findings:
                is_tp = label_finding(f, item, code_is_insecure=False)
                item_findings.append((f, is_tp))

        if item_findings:
            items_with_findings += 1
        all_labeled.extend(item_findings)

    console.print(f"\n[bold]Findings:[/bold] {len(all_labeled)} total from {items_with_findings}/{len(cal_items)} items")

    tp_count = sum(1 for _, is_tp in all_labeled if is_tp)
    fp_count = len(all_labeled) - tp_count
    console.print(f"  True positives: {tp_count}")
    console.print(f"  False positives: {fp_count}")

    if not all_labeled:
        console.print("[yellow]No findings to calibrate. Exiting.[/yellow]")
        return

    # Save labeled set
    labeled_path = output_path.parent / "labeled_findings.jsonl"
    save_labeled_set(all_labeled, labeled_path)
    console.print(f"Labeled findings saved to: {labeled_path}")

    # Estimate per-rule precision
    precision_results = estimate_precision(all_labeled, delta=0.05)

    # Save precision map
    output_path.parent.mkdir(parents=True, exist_ok=True)
    precision_map = {}
    for rule_id, rp in precision_results.items():
        precision_map[rule_id] = {
            "precision": rp.precision,
            "tp_count": rp.tp_count,
            "fp_count": rp.fp_count,
            "n_samples": rp.n_samples,
            "ci_lower": rp.ci_lower,
            "ci_upper": rp.ci_upper,
        }

    output_path.write_text(json.dumps(precision_map, indent=2))
    console.print(f"Precision map saved to: {output_path}")

    # Print summary table
    table = Table(title="Per-Rule Precision Estimates")
    table.add_column("Rule ID", style="cyan", max_width=60)
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("N", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("95% CI", justify="right")

    for rule_id in sorted(precision_results, key=lambda r: precision_results[r].precision, reverse=True):
        rp = precision_results[rule_id]
        table.add_row(
            rule_id[-60:],
            str(rp.tp_count),
            str(rp.fp_count),
            str(rp.n_samples),
            f"{rp.precision:.2f}",
            f"[{rp.ci_lower:.2f}, {rp.ci_upper:.2f}]",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Run calibration to estimate per-rule precision")
    parser.add_argument(
        "--output",
        default="data/calibration/precision_map.json",
        help="Output path for precision map",
    )
    parser.add_argument(
        "--benchmark",
        default="both",
        choices=["cweval", "securityeval", "both"],
        help="Which benchmark to use for calibration",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.25,
        help="Fraction of data to use for calibration (rest reserved for evaluation)",
    )
    parser.add_argument(
        "--use-full-dataset",
        action="store_true",
        help="Use full dataset for calibration (no train/eval split)",
    )
    parser.add_argument(
        "--analyzer",
        default="semgrep",
        choices=["semgrep", "bandit", "combined"],
        help="Which analyzer to calibrate",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.benchmark == "both":
        # Run calibration on both benchmarks, merge results
        from collections import defaultdict

        all_labeled = []

        for bench in ["cweval", "securityeval"]:
            console.print(f"\n{'='*60}")
            items = load_benchmark(bench)
            if args.use_full_dataset:
                cal_items = items
            else:
                cal_items, _ = split_calibration_eval(items, args.calibration_fraction, args.seed)

            console.print(f"[bold]{bench}:[/bold] {len(cal_items)} calibration items")
            analyzer = _build_analyzer(args.analyzer)

            for item in tqdm(cal_items, desc=f"Analyzing {bench}"):
                if item.insecure_code and item.insecure_code.strip():
                    findings = analyzer.analyze(item.insecure_code)
                    for f in findings:
                        is_tp = label_finding(f, item, code_is_insecure=True)
                        all_labeled.append((f, is_tp))

                if item.secure_code and item.secure_code.strip():
                    full_secure = item.prompt + item.secure_code
                    findings = analyzer.analyze(full_secure)
                    for f in findings:
                        is_tp = label_finding(f, item, code_is_insecure=False)
                        all_labeled.append((f, is_tp))

        tp_count = sum(1 for _, is_tp in all_labeled if is_tp)
        console.print(f"\n[bold]Combined:[/bold] {len(all_labeled)} findings ({tp_count} TP, {len(all_labeled)-tp_count} FP)")

        # Save and estimate
        labeled_path = output_path.parent / "labeled_findings.jsonl"
        labeled_path.parent.mkdir(parents=True, exist_ok=True)
        save_labeled_set(all_labeled, labeled_path)

        precision_results = estimate_precision(all_labeled, delta=0.05)
        precision_map = {}
        for rule_id, rp in precision_results.items():
            precision_map[rule_id] = {
                "precision": rp.precision,
                "tp_count": rp.tp_count,
                "fp_count": rp.fp_count,
                "n_samples": rp.n_samples,
                "ci_lower": rp.ci_lower,
                "ci_upper": rp.ci_upper,
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(precision_map, indent=2))
        console.print(f"\nPrecision map saved to: {output_path}")

        # Summary table
        table = Table(title="Per-Rule Precision Estimates (Combined)")
        table.add_column("Rule ID", style="cyan", max_width=60)
        table.add_column("TP", justify="right")
        table.add_column("FP", justify="right")
        table.add_column("N", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("95% CI", justify="right")

        for rule_id in sorted(precision_results, key=lambda r: precision_results[r].precision, reverse=True):
            rp = precision_results[rule_id]
            table.add_row(
                rule_id[-60:],
                str(rp.tp_count),
                str(rp.fp_count),
                str(rp.n_samples),
                f"{rp.precision:.2f}",
                f"[{rp.ci_lower:.2f}, {rp.ci_upper:.2f}]",
            )
        console.print(table)

    else:
        run_calibration(
            benchmark_name=args.benchmark,
            output_path=output_path,
            seed=args.seed,
            calibration_fraction=args.calibration_fraction,
            use_full_dataset=args.use_full_dataset,
            analyzer_name=args.analyzer,
        )


if __name__ == "__main__":
    main()
