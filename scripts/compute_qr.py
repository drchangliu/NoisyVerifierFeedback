#!/usr/bin/env python3
"""Compute empirical q (fix probability) and r (regression probability)
with 95% Wilson CIs from feedback-loop traces.

Definition (canonical in this repo, matching AdaptiveAgent.run in
src/nvf/agents/adaptive.py): per-item, iter 0 -> iter 1 transition only,
restricted to items that entered the loop (>=2 iterations). Trial label
is derived from has_vulnerability at iter 0:

  - v0 = True  -> TP trial; counted as "fixed" iff
                  v1 = False AND tests_passed at iter 1 = True
  - v0 = False -> FP trial; counted as "regressed" iff
                  v1 = True  OR  tests_passed at iter 1 = False

Pooled across naive, selective, and llm_judge conditions on 51-item core
benchmark runs (n_items in {50, 51}). adaptive runs are excluded so the
table reflects fixed-policy behavior under known feedback.

Usage:
    python scripts/compute_qr.py                 # all 4 models
    python scripts/compute_qr.py --model sonnet  # one model
    python scripts/compute_qr.py --json /tmp/qr.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

INCLUDED_CONDITIONS = {"naive", "selective", "llm_judge"}
CORE_ITEM_SIZES = {50, 51}


def model_short(model_str: str) -> str:
    s = model_str.lower()
    if "haiku" in s:
        return "haiku"
    if "sonnet" in s:
        return "sonnet"
    if "qwen" in s:
        return "qwen3-8b"
    if "gemma" in s:
        return "gemma4-31b"
    return model_str


def wilson_ci(s: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = s / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, c - m), min(1.0, c + m)


def count_qr_for_runs(results_dir: Path,
                      group_by_policy: bool = False) -> dict[str, dict]:
    """Walk run dirs and tally per-(model[, policy]) q/r counts.

    When `group_by_policy` is False (default), counts are pooled across
    the three fixed-policy conditions (naive, selective, llm_judge);
    keys are model_short strings.

    When `group_by_policy` is True, counts are kept per (model, policy);
    keys are "model|policy" strings. This is used to check the pooling
    assumption: if per-policy q,r differ materially, the pooled estimate
    can be misleading.
    """
    counts: dict[str, dict] = defaultdict(
        lambda: dict(tp_fixed=0, tp_total=0, fp_regressed=0, fp_total=0,
                     n_runs=0, n_loop_items=0)
    )
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
        if len(traces) not in CORE_ITEM_SIZES:
            continue
        cond = traces[0].get("condition")
        if cond not in INCLUDED_CONDITIONS:
            continue
        m = model_short(traces[0].get("model", ""))
        key = f"{m}|{cond}" if group_by_policy else m
        counts[key]["n_runs"] += 1
        for t in traces:
            iters = t.get("iterations", [])
            if len(iters) < 2:
                continue
            i0, i1 = iters[0], iters[1]
            if not i0.get("feedback_rules"):
                continue  # nothing was surfaced -> item did not enter loop
            counts[key]["n_loop_items"] += 1
            v0, v1 = i0.get("has_vulnerability"), i1.get("has_vulnerability")
            tp0, tp1 = i0.get("tests_passed"), i1.get("tests_passed")
            if v0 is None or v1 is None:
                continue
            if v0:
                counts[key]["tp_total"] += 1
                if (v1 is False) and (tp1 is True):
                    counts[key]["tp_fixed"] += 1
            else:
                counts[key]["fp_total"] += 1
                if (v1 is True) or (tp1 is False):
                    counts[key]["fp_regressed"] += 1
    return dict(counts)


def summarize(model: str, c: dict) -> dict:
    tpf, tpt = c["tp_fixed"], c["tp_total"]
    fpr, fpt = c["fp_regressed"], c["fp_total"]
    q = tpf / tpt if tpt else 0.0
    r = fpr / fpt if fpt else 0.0
    q_lo, q_hi = wilson_ci(tpf, tpt)
    r_lo, r_hi = wilson_ci(fpr, fpt)
    tau = r / (q + r) if (q + r) > 0 else 0.0
    return dict(
        model=model, tp_fixed=tpf, tp_total=tpt, q=q, q_lo=q_lo, q_hi=q_hi,
        fp_regressed=fpr, fp_total=fpt, r=r, r_lo=r_lo, r_hi=r_hi,
        tau_star=tau, n_runs=c.get("n_runs", 0),
        n_loop_items=c.get("n_loop_items", 0),
    )


def print_summary(s: dict) -> None:
    print(f"Model: {s['model']}")
    print(f"  TP fixed:    {s['tp_fixed']:>3} / {s['tp_total']:<3}  "
          f"-> q = {s['q']:.3f} [Wilson 95% CI: {s['q_lo']:.3f}, {s['q_hi']:.3f}]")
    print(f"  FP regress:  {s['fp_regressed']:>3} / {s['fp_total']:<3}  "
          f"-> r = {s['r']:.3f} [Wilson 95% CI: {s['r_lo']:.3f}, {s['r_hi']:.3f}]")
    print(f"  tau* = r/(q+r) = {s['tau_star']:.3f}")
    print(f"  ({s['n_runs']} runs, {s['n_loop_items']} items entered loop)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument(
        "--model", default=None,
        help="One of {haiku,sonnet,gemma4-31b,qwen3-8b}. Default: all four.",
    )
    ap.add_argument("--json", default=None, help="Optional path to dump JSON summary.")
    ap.add_argument(
        "--per-policy", action="store_true",
        help="Report per-(model, policy) breakdown instead of pooled.",
    )
    args = ap.parse_args()

    counts = count_qr_for_runs(Path(args.results_dir),
                               group_by_policy=args.per_policy)
    models = (
        [args.model] if args.model
        else ["haiku", "sonnet", "gemma4-31b", "qwen3-8b"]
    )

    out = []
    if args.per_policy:
        # Print per-(model, policy) grid
        print(f"{'model':<12} {'policy':<10} {'TP fixed':>9} {'TP total':>9}"
              f" {'q':>6} {'95% CI':>16} {'FP regr':>8} {'FP total':>9}"
              f" {'r':>6} {'95% CI':>16} {'tau*':>6}")
        print("-" * 124)
        for m in models:
            for cond in ("naive", "selective", "llm_judge"):
                key = f"{m}|{cond}"
                c = counts.get(key, dict(tp_fixed=0, tp_total=0, fp_regressed=0,
                                          fp_total=0, n_runs=0, n_loop_items=0))
                s = summarize(m, c)
                s["policy"] = cond
                out.append(s)
                if s["tp_total"] + s["fp_total"] == 0:
                    continue
                print(f"{m:<12} {cond:<10} {s['tp_fixed']:>9d} {s['tp_total']:>9d}"
                      f" {s['q']:>6.3f} [{s['q_lo']:.2f},{s['q_hi']:.2f}]"
                      f" {s['fp_regressed']:>8d} {s['fp_total']:>9d}"
                      f" {s['r']:>6.3f} [{s['r_lo']:.2f},{s['r_hi']:.2f}]"
                      f" {s['tau_star']:>6.3f}")
            print()
    else:
        for m in models:
            c = counts.get(m, dict(tp_fixed=0, tp_total=0, fp_regressed=0,
                                    fp_total=0, n_runs=0, n_loop_items=0))
            s = summarize(m, c)
            print_summary(s)
            print()
            out.append(s)

        total_trials = sum(s["tp_total"] + s["fp_total"] for s in out)
        print(f"Total feedback-loop trials across listed models: {total_trials}")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"Wrote JSON summary to {args.json}")


if __name__ == "__main__":
    main()
