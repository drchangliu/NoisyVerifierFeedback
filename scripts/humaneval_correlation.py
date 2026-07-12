#!/usr/bin/env python3
"""Correlate the counterfactual regression rate r with an EXTERNAL capability
axis: HumanEval pass@1 (scripts/run_humaneval.py).

The within-benchmark proxies (JP@0, sec@0, CWEval-JP@0) share the
`has_vulnerability` oracle with r, so their correlations are partly mechanical.
HumanEval pass@1 is functional correctness on disjoint items with no security
signal -- independent of everything r is computed from -- so this is the
non-circular test of the r-law.

Models with no valid HumanEval measurement are excluded and listed:
sonnet4 (claude-sonnet-4-20250514 retired from the Anthropic API) and
glm-4.6 (retired from Ollama Cloud 2026-06-16) could not be measured.

Usage:
    python scripts/humaneval_correlation.py
    python scripts/humaneval_correlation.py --figure figures/r_vs_humaneval.png
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
from tau_star_prediction import load_runs, LABELS
from r_capability import pearson, ranks, accumulate_qr, MIN_TRIALS

ROOT = Path(__file__).resolve().parent.parent
HE_DIR = ROOT / "data" / "humaneval"
N_TASKS = 164  # HumanEval is a fixed 164-problem set


def humaneval_pass1(label: str) -> float | None:
    """pass@1 from the saved jsonl; None if absent or the run is invalid:
    incomplete (fewer than all 164 tasks, e.g. a --limit smoke test),
    generation errors, or token-cap truncations (explicitly marked by
    run_humaneval.py) -- any of these would understate the model. Fix with
    run_humaneval.py --redo-capped / --redo-errors."""
    path = HE_DIR / f"{label}.jsonl"
    if not path.exists():
        return None
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    bad = any(str(r.get("error", "")).startswith("gen-error")
              or r.get("error") == "token-cap"
              for r in recs)
    if len(recs) != N_TASKS or bad:
        return None
    return sum(r["passed"] for r in recs) / len(recs)


def spearman_perm_p(a, b, n_perm: int = 100_000, seed: int = 0) -> float:
    """Two-sided permutation p-value for the Spearman correlation."""
    obs = abs(pearson(ranks(a), ranks(b)))
    rng = random.Random(seed)
    b = list(b)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(b)
        if abs(pearson(ranks(a), ranks(b))) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "data" / "results"))
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()

    # q/r per distinct model, identical to r_capability.py --collapse-eras
    # (same accumulate_qr helper, so the canonical labeling cannot diverge).
    qr = defaultdict(lambda: dict(tpf=0, tpt=0, fpr=0, fpt=0))
    for coh, cond, traces in load_runs(Path(args.results_dir), collapse=True):
        accumulate_qr(qr[coh], traces)

    rows, missing = [], []
    for coh, d in sorted(qr.items()):
        if d["fpt"] < MIN_TRIALS or d["tpt"] < MIN_TRIALS:
            continue
        he = humaneval_pass1(coh.replace("(cloud)", ""))
        if he is None:
            missing.append(coh)
            continue
        rows.append((coh, he * 100, d["fpr"] / d["fpt"], d["tpf"] / d["tpt"]))
    rows.sort(key=lambda x: x[1])

    print(f"{'model':<26}{'HumanEval':>10}{'r':>7}{'q':>7}")
    for coh, he, r, q in rows:
        print(f"{coh:<26}{he:>9.1f}%{r:>7.2f}{q:>7.2f}")
    if missing:
        print(f"\nexcluded (no valid HumanEval run): {', '.join(missing)}")

    hes = [x[1] for x in rows]
    rs = [x[2] for x in rows]
    qs = [x[3] for x in rows]
    p = spearman_perm_p(hes, rs)
    print(f"\nn={len(rows)} models")
    print(f"corr(HumanEval, r): Pearson {pearson(hes, rs):+.3f} "
          f" Spearman {pearson(ranks(hes), ranks(rs)):+.3f} "
          f" (permutation p={p:.4f})")
    print(f"corr(HumanEval, q): Pearson {pearson(hes, qs):+.3f} "
          f" Spearman {pearson(ranks(hes), ranks(qs)):+.3f}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fam = {"haiku": "#1f77b4", "sonnet": "#d62728", "gemma": "#2ca02c",
               "qwen": "#9467bd", "llama": "#ff7f0e", "deepseek": "#8c564b",
               "glm": "#17becf"}
        # HumanEval is saturated (10/13 models >= 94%), so a linear pass@1 axis
        # piles the cluster onto one column. Plot the log ERROR rate instead,
        # inverted so capability still grows rightward. Spearman is invariant
        # to this monotone relabeling; a perfect score is clamped to half the
        # smallest resolvable error (0.5 tasks / 164) for the log axis only.
        floor = 100 * 0.5 / 164
        errs = [max(100 - he, floor) for he in hes]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        # hand-tuned label offsets for the crowded r~0.28 trio
        nudge = {"qwen3-32b": (6, 10), "deepseek-v4-flash(cloud)": (6, -14),
                 "qwen3-coder-480b(cloud)": (-6, 4)}
        for (coh, he, r, q), e in zip(rows, errs):
            color = next((v for k, v in fam.items() if k in coh), "#7f7f7f")
            ax.scatter(e, r, s=70, color=color, zorder=3)
            dx, dy = nudge.get(coh, (6, 4))
            ax.annotate(LABELS.get(coh, coh), (e, r), textcoords="offset points",
                        xytext=(dx, dy), fontsize=7.5,
                        ha="right" if dx < 0 else "left")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ticks = [50, 20, 10, 5, 2, 1, floor]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["50", "80", "90", "95", "98", "99", "100"])
        ax.set_xlabel("HumanEval pass@1 (%, log-error axis)  --  external capability axis")
        ax.set_ylabel(r"counterfactual regression rate $r$")
        rho = pearson(ranks(hes), ranks(rs))
        ax.set_title("The r-law on an external capability axis\n"
                     "(HumanEval shares no items or oracles with the r measurement)")
        ax.plot([], [], " ", label=f"Spearman rho = {rho:+.2f}")
        ax.legend(loc="upper right", handlelength=0)
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
