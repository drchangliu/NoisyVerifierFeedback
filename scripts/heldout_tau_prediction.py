#!/usr/bin/env python3
"""Prospective (held-out) version of the tau* policy-prediction test.

Two analyses, both per cohort, addressing the circularity in the
retrospective test (scripts/tau_star_prediction.py), where q/r were
estimated from the same traces whose outcomes they predict:

1. Item-split prediction: repeatedly split the 51 core items into a
   calibration half and an evaluation half. Estimate tau* = r/(q+r)
   from loop trials on calibration items only (naive runs only, so the
   trial mix is selection-free), predict the naive-vs-selective winner,
   and score the prediction against the actual sel-naive dJP gap
   computed on the held-out items only.

2. Sample-complexity curve: bootstrap m loop-trials from the cohort's
   naive trials and measure how often tau-hat* lands on the same side
   of 0.5 as the full-sample estimate. This is the empirical companion
   to the paper's Hoeffding/Bernstein calibration bound: how many
   calibration trials buy a reliable policy choice.

Usage:
    python scripts/heldout_tau_prediction.py
    python scripts/heldout_tau_prediction.py --splits 500 --boot 2000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tau_star_prediction import COHORT_ORDER, load_runs, jp

ROOT = Path(__file__).resolve().parent.parent


def collect(results_dir: Path):
    """Per cohort: naive loop-trials by item, and per-run per-item JP deltas."""
    # trials[coh][item_id] = list of (is_tp, success) from naive runs
    trials = defaultdict(lambda: defaultdict(list))
    # runs[coh][cond] = list of {item_id: dJP} dicts (one per run)
    runs = defaultdict(lambda: defaultdict(list))
    for coh, cond, traces in load_runs(results_dir):
        if cond not in ("naive", "selective"):
            continue
        deltas = {}
        for t in traces:
            its = t["iterations"]
            deltas[t["item_id"]] = int(jp(its[-1])) - int(jp(its[0]))
            if cond != "naive" or len(its) < 2 or not its[0].get("feedback_rules"):
                continue
            v0, v1 = its[0].get("has_vulnerability"), its[1].get("has_vulnerability")
            tp1 = its[1].get("tests_passed")
            if v0 is None or v1 is None:
                continue
            if v0:
                trials[coh][t["item_id"]].append(("tp", v1 is False and tp1 is True))
            else:
                trials[coh][t["item_id"]].append(("fp", v1 is True or tp1 is False))
        runs[coh][cond].append(deltas)
    return trials, runs


def tau_from_trials(ts: list[tuple[str, bool]]) -> float | None:
    tpt = [s for k, s in ts if k == "tp"]
    fpt = [s for k, s in ts if k == "fp"]
    if not tpt or not fpt:
        return None
    q = mean(tpt)
    r = mean(fpt)
    return r / (q + r) if q + r else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "data" / "results"))
    ap.add_argument("--splits", type=int, default=500)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    trials, runs = collect(Path(args.results_dir))

    # --- 1. item-split prospective prediction ---
    print("Item-split prospective test "
          f"({args.splits} random half-splits per cohort)\n")
    print("Regret = held-out JP (pp) lost vs the oracle winner; "
          "'always-X' = regret of that fixed choice.\n")
    print(f"{'cohort':<11}{'trials':>7}{'pred=actual':>13}{'gap(pp)':>9}"
          f"{'regret':>9}{'alw-naive':>11}{'alw-sel':>9}")
    for coh in COHORT_ORDER:
        if coh not in trials or not runs[coh]["selective"]:
            continue
        items = sorted(trials[coh])
        n_trials = sum(len(v) for v in trials[coh].values())
        ok = correct = 0
        gaps, regret, reg_naive, reg_sel = [], [], [], []
        for _ in range(args.splits):
            rng.shuffle(items)
            half = len(items) // 2
            cal, ev = set(items[:half]), set(items[half:])
            tau = tau_from_trials(
                [t for i in cal for t in trials[coh][i]])
            if tau is None:
                continue
            ok += 1

            def cond_delta(cond):
                per_run = [mean(d[i] for i in ev if i in d)
                           for d in runs[coh][cond]]
                return mean(per_run) * 100

            gap = cond_delta("selective") - cond_delta("naive")
            gaps.append(gap)
            correct += (tau > 0.5) == (gap > 0)
            regret.append(abs(gap) if (tau > 0.5) != (gap > 0) else 0.0)
            reg_naive.append(max(0.0, gap))
            reg_sel.append(max(0.0, -gap))
        if not ok:
            continue
        print(f"{coh:<11}{n_trials:>7}{correct / ok:>13.2f}{mean(gaps):>9.2f}"
              f"{mean(regret):>9.2f}{mean(reg_naive):>11.2f}{mean(reg_sel):>9.2f}")

    # --- 2. sample-complexity of policy identification ---
    print("\nSample complexity: P(tau-hat* on same side of 0.5 as full-sample)"
          f"  ({args.boot} bootstraps)\n")
    budgets = [10, 20, 40, 80, 160]
    print(f"{'cohort':<11}{'full tau*':>10}" +
          "".join(f"{'m=' + str(m):>8}" for m in budgets))
    for coh in COHORT_ORDER:
        pool = [t for v in trials.get(coh, {}).values() for t in v]
        full = tau_from_trials(pool)
        if full is None:
            continue
        side = full > 0.5
        row = f"{coh:<11}{full:>10.3f}"
        for m in budgets:
            hits = tries = 0
            for _ in range(args.boot):
                tau = tau_from_trials([rng.choice(pool) for _ in range(m)])
                if tau is None:
                    continue
                tries += 1
                hits += (tau > 0.5) == side
            row += f"{hits / tries:>8.2f}" if tries else f"{'--':>8}"
        print(row)


if __name__ == "__main__":
    main()
