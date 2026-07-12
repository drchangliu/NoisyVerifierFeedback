#!/usr/bin/env python3
"""Aggregate v2 batch results (Sonnet 4.6, Gemma v2) into a reproducible CSV.

The original aggregate_seeds.py collapses Sonnet 4 and Sonnet 4.6 into the
same "sonnet" bucket because it strips model versions. This script preserves
version distinctions and the broken-batch cutoff dates so the headline cells
that informed the Sonnet-version-gap and Gemma F2-validation analyses are
trivially reproducible.

Usage:
  python scripts/aggregate_v2.py                  # writes data/aggregated_v2.csv
  python scripts/aggregate_v2.py --print          # also print to stdout
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent

# Cohorts of valid runs (excludes pre-fix batches with broken analyzer pipeline).
# See logs/{sonnet46,gemma,qwen}_driver_v2.log for the corresponding driver runs.
COHORTS = {
    # (model_match_in_trace_field, cutoff_unix, glob_subpattern, label)
    "sonnet-4.6": (
        "sonnet-4-6",
        datetime(2026, 5, 30, 10, 0).timestamp(),   # before the May 30 10:59 v2 batch
        "*_sonnet_*",
        "Sonnet 4.6 v2 batch (n=20 per cell, 51-item)",
    ),
    "gemma-v2": (
        "gemma",
        datetime(2026, 5, 29, 12, 0).timestamp(),   # before the May 29-31 v2 batch
        "*_gemma4:31b_*",
        "Gemma4-31B v2 batch (n=11-12 per cell, 51-item)",
    ),
    "qwen-v2": (
        "qwen",
        datetime(2026, 6, 1, 6, 0).timestamp(),     # before the June 1 Qwen v2 batch
        "*_qwen3:8b_*",
        "Qwen3-8B v2 batch (n=8-12 per cell, 51-item)",
    ),
    "haiku-v2": (
        "haiku",
        datetime(2026, 6, 12, 0, 0).timestamp(),    # June 12 batch; last 2 runs died on credits
        "*_haiku_*",
        "Haiku v2 batch (n=7-8 per cell, 51-item)",
    ),
}


def jp_pass(it: dict) -> bool:
    return bool(it.get("tests_passed")) and not bool(it.get("has_vulnerability"))


def aggregate_cohort(model_match: str, cutoff: float, pattern: str) -> dict[str, list[dict]]:
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for run_dir in sorted(glob.glob(str(ROOT / "data" / "results" / pattern))):
        tp = Path(run_dir) / "traces.jsonl"
        if not tp.exists() or tp.stat().st_mtime < cutoff:
            continue
        with tp.open() as f:
            first_line = f.readline()
        if not first_line:
            continue
        try:
            first = json.loads(first_line)
        except json.JSONDecodeError:
            continue
        if model_match not in first.get("model", ""):
            continue
        traces = []
        with tp.open() as f:
            for line in f:
                t = json.loads(line)
                if "error" not in t:
                    traces.append(t)
        if len(traces) < 40:
            continue
        cond = first["condition"]
        # tau-sweep ablation runs share condition == "selective"; break them
        # out per threshold so they don't pollute the tau = 0.5 cell.
        if cond == "selective":
            cfg_path = Path(run_dir) / "config.yaml"
            if cfg_path.exists():
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text()) or {}
                tau = (cfg.get("selective") or {}).get("threshold_tau", 0.5)
                if tau != 0.5:
                    cond = f"selective_tau{tau}"
        n_items = len(traces)
        jp0 = sum(1 for t in traces if t["iterations"] and jp_pass(t["iterations"][0]))
        jpf = sum(1 for t in traces if t["iterations"] and jp_pass(t["iterations"][-1]))
        engaged = sum(1 for t in traces if len(t["iterations"]) > 1)
        # r proxy: P(secure@0 turned vulnerable@final | engaged)
        r_num = sum(1 for t in traces if len(t["iterations"]) > 1
                    and jp_pass(t["iterations"][0]) and not jp_pass(t["iterations"][-1]))
        r_den = sum(1 for t in traces if len(t["iterations"]) > 1
                    and jp_pass(t["iterations"][0]))
        by_policy[cond].append({
            "jp0": jp0 / n_items,
            "jpf": jpf / n_items,
            "engaged": engaged,
            "r_num": r_num,
            "r_den": r_den,
            "cost": sum(t.get("total_cost_usd", 0) for t in traces),
        })
    return by_policy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--print", action="store_true", help="also print summary to stdout")
    args = p.parse_args()

    out_path = ROOT / "data" / "aggregated_v2.csv"
    rows = []
    for cohort_key, (model_match, cutoff, pattern, label) in COHORTS.items():
        by_policy = aggregate_cohort(model_match, cutoff, pattern)
        for cond in sorted(by_policy):
            runs = by_policy[cond]
            n = len(runs)
            jp0_vals = [r["jp0"] for r in runs]
            jpf_vals = [r["jpf"] for r in runs]
            r_num_total = sum(r["r_num"] for r in runs)
            r_den_total = sum(r["r_den"] for r in runs)
            rows.append({
                "cohort": cohort_key,
                "label": label,
                "condition": cond,
                "n_seeds": n,
                "jp0_mean": round(mean(jp0_vals), 4),
                "jp0_ci_95": round(1.96 * stdev(jp0_vals) / sqrt(n), 4) if n > 1 else 0,
                "jpf_mean": round(mean(jpf_vals), 4),
                "jpf_ci_95": round(1.96 * stdev(jpf_vals) / sqrt(n), 4) if n > 1 else 0,
                "gain_pp": round((mean(jpf_vals) - mean(jp0_vals)) * 100, 2),
                "engaged_mean": round(mean([r["engaged"] for r in runs]), 2),
                "r_proxy_num": r_num_total,
                "r_proxy_den": r_den_total,
                "r_proxy_pct": round(100 * r_num_total / r_den_total, 1) if r_den_total else None,
                "cost_total_usd": round(sum(r["cost"] for r in runs), 4),
            })

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    if args.print:
        print()
        hdr = f"{'cohort':<12}{'condition':<12}{'n':>4}{'JP@0':>13}{'JP@final':>15}{'gain':>10}{'r%':>8}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            jp0_str = f"{r['jp0_mean']*100:.1f}±{r['jp0_ci_95']*100:.1f}"
            jpf_str = f"{r['jpf_mean']*100:.1f}±{r['jpf_ci_95']*100:.1f}"
            rpct = f"{r['r_proxy_pct']}%" if r['r_proxy_pct'] is not None else "—"
            print(f"{r['cohort']:<12}{r['condition']:<12}{r['n_seeds']:>4}{jp0_str:>13}{jpf_str:>15}{r['gain_pp']:+6.1f}pp{rpct:>8}")


if __name__ == "__main__":
    main()
