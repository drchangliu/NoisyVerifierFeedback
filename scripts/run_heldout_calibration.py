#!/usr/bin/env python3
"""Recompute per-rule precision on the held-out SecurityEval subset
(audit #9).

Wraps the existing calibration pipeline (analyzers + label_finding +
estimate_precision) without modifying it. The held-out subset is the
SecurityEval items NOT present in the 51-item core benchmark
(combined = CWEval + securityeval_tested). The held-out IDs are
persisted at data/calibration/heldout_securityeval_ids.json.

This produces a held-out precision map that we compare against the
original 146-item precision map to verify that policy conclusions
(specifically the τ = 0.5 surfaced-rule set) are stable under a strict
calibration / evaluation split.

Usage:
    python scripts/run_heldout_calibration.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_calibration import _build_analyzer  # type: ignore

from nvf.benchmark.loader import load_securityeval
from nvf.calibration.estimator import estimate_precision
from nvf.calibration.labeled_set import save_labeled_set
from nvf.calibration.labeler import label_finding

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ids-file",
        default="data/calibration/heldout_securityeval_ids.json",
        help="JSON file with 'held_out_securityeval_ids' list",
    )
    ap.add_argument(
        "--output",
        default="data/calibration/heldout_precision_map.json",
    )
    ap.add_argument(
        "--analyzer",
        default="combined",
        choices=["semgrep", "bandit", "combined"],
    )
    args = ap.parse_args()

    ids_data = json.loads(Path(args.ids_file).read_text())
    held_out_ids = set(ids_data["held_out_securityeval_ids"])
    console.print(f"[bold]Held-out IDs:[/bold] {len(held_out_ids)} items")

    all_items = load_securityeval(with_tests_only=False)
    held_items = [it for it in all_items if it.item_id in held_out_ids]
    console.print(f"[bold]Loaded held-out items:[/bold] {len(held_items)}")
    if len(held_items) != len(held_out_ids):
        console.print(
            f"[red]Warning:[/red] expected {len(held_out_ids)} items but found "
            f"{len(held_items)}; some IDs did not match the loader output."
        )

    analyzer = _build_analyzer(args.analyzer)

    all_labeled = []
    for item in tqdm(held_items, desc="Analyzing held-out items"):
        if item.insecure_code and item.insecure_code.strip():
            for f in analyzer.analyze(item.insecure_code):
                is_tp = label_finding(f, item, code_is_insecure=True)
                all_labeled.append((f, is_tp))
        if item.secure_code and item.secure_code.strip():
            full_secure = item.prompt + item.secure_code
            for f in analyzer.analyze(full_secure):
                is_tp = label_finding(f, item, code_is_insecure=False)
                all_labeled.append((f, is_tp))

    tp_count = sum(1 for _, is_tp in all_labeled if is_tp)
    console.print(
        f"\n[bold]Held-out calibration:[/bold] {len(all_labeled)} findings "
        f"({tp_count} TP, {len(all_labeled) - tp_count} FP)"
    )

    labeled_path = Path(args.output).parent / "labeled_findings_heldout.jsonl"
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    save_labeled_set(all_labeled, labeled_path)

    precision_results = estimate_precision(all_labeled, delta=0.05)
    precision_map = {}
    for rule_id, rp in precision_results.items():
        precision_map[rule_id] = dict(
            precision=rp.precision, tp_count=rp.tp_count, fp_count=rp.fp_count,
            n_samples=rp.n_samples, ci_lower=rp.ci_lower, ci_upper=rp.ci_upper,
        )

    Path(args.output).write_text(json.dumps(precision_map, indent=2))
    console.print(f"\nHeld-out precision map saved to: {args.output}")


if __name__ == "__main__":
    main()
