#!/usr/bin/env python3
"""Retrospective test: does calibrated tau* = r/(q+r) predict the
naive-vs-selective winner across model cohorts?

Cohorts split by model version AND era (Haiku Apr vs May acts as a
same-model control; Sonnet 4 vs 4.6 is the clean version pair; Qwen
v1 vs v2 is confounded with the Ollama serving fix in dfe3f5e).
Broken-analyzer batches (May 27-30) are excluded per run by the
zero-findings-at-iter-0 signature rather than by time cutoff.

q/r follow the canonical definition in scripts/compute_qr.py
(iter 0 -> iter 1, items that entered the loop). By default they pool
naive/selective/llm_judge; --naive-only restricts to naive runs, which
surface every finding and are therefore free of the selection bias
that selective/llm_judge runs introduce into the FP-trial mix.

Usage:
    python scripts/tau_star_prediction.py
    python scripts/tau_star_prediction.py --naive-only
    python scripts/tau_star_prediction.py --figure figures/tau_star_prediction.png
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
CORE_ITEM_SIZES = {50, 51}
FIXED_CONDITIONS = ("naive", "selective", "llm_judge")
COHORT_ORDER = ["haiku-apr", "haiku-may", "haiku-v2", "sonnet4", "sonnet46",
                "gemma-v1", "gemma-v2", "qwen-v1", "qwen-v2"]
LABELS = {"haiku-apr": "Haiku (Apr)", "haiku-may": "Haiku (May)",
          "haiku-v2": "Haiku (Jun)",
          "sonnet4": "Sonnet 4", "sonnet46": "Sonnet 4.6",
          "gemma-v1": "Gemma4-31B (v1)", "gemma-v2": "Gemma4-31B (v2)",
          "qwen-v1": "Qwen3-8B (v1)", "qwen-v2": "Qwen3-8B (v2)"}
V2_SPLIT = datetime(2026, 5, 27)

# Prospective out-of-sample cohorts (scripts/prospective_cohorts.sh): NEW
# model snapshots not used to establish the law. Keyed by exact Ollama tag
# so they never collide with the qwen3:8b / gemma4:31b in-sample cohorts.
PROSPECTIVE_MODELS = {
    "llama3.1:8b": "llama3.1-8b",
    "qwen3:14b": "qwen3-14b",
    "deepseek-v2:16b": "deepseek-v2-16b",
    "gemma3:27b": "gemma3-27b",
    "qwen3.5:27b": "qwen3.5-27b",
    "qwen3.6:27b": "qwen3.6-27b",
    "qwen3:32b": "qwen3-32b",
    # cloud-hosted (Ollama Cloud) -- high-capability / new-lineage points.
    "glm-4.6:cloud": "glm-4.6(cloud)",
    "qwen3-coder:480b-cloud": "qwen3-coder-480b(cloud)",
    "deepseek-v4-flash:cloud": "deepseek-v4-flash(cloud)",
    # Anthropic API frontier cohorts (July 2026): further out-of-sample points
    # at the high-capability end. Keyed by exact API model ID.
    "claude-opus-4-8": "opus-4.8",
    "claude-fable-5": "fable-5",
}
LABELS.update({v: v for v in PROSPECTIVE_MODELS.values()})

# Data-driven additions: configs/model_registry.json lets new cohorts join
# without code edits (see scripts/model_registry.py).
from model_registry import load_registry
for _tag, _e in load_registry().items():
    PROSPECTIVE_MODELS[_tag] = _e["cohort"]
    LABELS[_e["cohort"]] = _e.get("display", _e["cohort"])

# Option-2 collapse: the same model ID re-measured on different dates is ONE
# model, not several. The era splits (Haiku Apr/May/Jun; Gemma/Qwen v1/v2) are
# temporal replicates of a single model ID, so for the headline capability->r
# law each DISTINCT model must contribute exactly one point (avoids
# pseudo-replication). Sonnet 4 vs 4.6 are genuinely different IDs and are NOT
# collapsed. The within-model spread across eras is reported separately as
# temporal drift.
COLLAPSE = {
    "haiku-apr": "haiku-4.5", "haiku-may": "haiku-4.5", "haiku-v2": "haiku-4.5",
    "gemma-v1": "gemma4-31b", "gemma-v2": "gemma4-31b",
    "qwen-v1": "qwen3-8b", "qwen-v2": "qwen3-8b",
}
LABELS.update({"haiku-4.5": "Haiku 4.5", "gemma4-31b": "Gemma4-31B",
               "qwen3-8b": "Qwen3-8B"})


def cohort_of(model: str, mtime: float) -> str | None:
    # Prospective snapshots first: several contain "qwen"/"gemma" substrings
    # and must not fall into the in-sample qwen3:8b / gemma4:31b cohorts.
    if model in PROSPECTIVE_MODELS:
        return PROSPECTIVE_MODELS[model]
    d = datetime.fromtimestamp(mtime)
    if "haiku" in model:
        if d.month == 4:
            return "haiku-apr"
        return "haiku-may" if d < datetime(2026, 6, 1) else "haiku-v2"
    if model == "claude-sonnet-4-20250514":
        return "sonnet4"
    if model == "claude-sonnet-4-6":
        return "sonnet46"
    if model == "gemma4:31b":
        return "gemma-v1" if d < V2_SPLIT else "gemma-v2"
    if model == "qwen3:8b":
        return "qwen-v1" if d < V2_SPLIT else "qwen-v2"
    return None


def jp(it: dict) -> bool:
    return bool(it.get("tests_passed")) and not bool(it.get("has_vulnerability"))


def wilson_ci(s: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = s / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, c - m), min(1.0, c + m)


def load_runs(results_dir: Path, collapse: bool = False):
    """Yield (cohort, condition, traces) for valid 51-item core runs.

    Selective runs from the tau-sweep ablations (threshold_tau != 0.5)
    are excluded: the 'selective' cell means the tau = 0.5 policy.

    collapse=True merges same-model-ID era cohorts into one distinct model
    (Option 2: one point per model for the headline correlation).
    """
    import yaml
    for run_dir in sorted(results_dir.iterdir()):
        tp = run_dir / "traces.jsonl"
        if not run_dir.is_dir() or "synthetic" in run_dir.name or not tp.exists():
            continue
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(open(cfg_path)) or {}
            tau = (cfg.get("selective") or {}).get("threshold_tau", 0.5)
            if cfg.get("feedback", {}).get("condition") == "selective" and tau != 0.5:
                continue
        traces = []
        with open(tp) as f:
            for line in f:
                t = json.loads(line)
                if "error" not in t and t.get("iterations"):
                    traces.append(t)
        if len(traces) not in CORE_ITEM_SIZES:
            continue
        cond = traces[0].get("condition")
        coh = cohort_of(traces[0].get("model", ""), tp.stat().st_mtime)
        if coh is None:
            continue
        if collapse:
            coh = COLLAPSE.get(coh, coh)
        # broken-analyzer signature: no item has any finding at iter 0
        if not any(t["iterations"][0].get("n_findings", 0) > 0 for t in traces):
            continue
        yield coh, cond, traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "data" / "results"))
    ap.add_argument("--naive-only", action="store_true",
                    help="Estimate q/r from naive runs only (selection-free).")
    ap.add_argument("--figure", default=None,
                    help="Optional path to write the scatter figure PNG.")
    ap.add_argument("--collapse-eras", action="store_true",
                    help="Collapse same-model-ID era cohorts into one distinct "
                         "model (Option 2: distinct-model headline).")
    ap.add_argument("--drop", default=None,
                    help="Comma-separated cohort labels to exclude "
                         "(sensitivity analyses, e.g. --drop sonnet4,sonnet46).")
    args = ap.parse_args()
    dropped = set(args.drop.split(",")) if args.drop else set()

    qr_conditions = ("naive",) if args.naive_only else FIXED_CONDITIONS
    qr = defaultdict(lambda: dict(tpf=0, tpt=0, fpr=0, fpt=0))
    deltas = defaultdict(list)  # (cohort, cond) -> per-run dJP

    for coh, cond, traces in load_runs(Path(args.results_dir), collapse=args.collapse_eras):
        if coh in dropped:
            continue
        if cond in FIXED_CONDITIONS:
            jp0 = mean(jp(t["iterations"][0]) for t in traces)
            jpf = mean(jp(t["iterations"][-1]) for t in traces)
            deltas[(coh, cond)].append(jpf - jp0)
        if cond not in qr_conditions:
            continue
        for t in traces:
            its = t["iterations"]
            if len(its) < 2 or not its[0].get("feedback_rules"):
                continue
            v0, v1 = its[0].get("has_vulnerability"), its[1].get("has_vulnerability")
            tp1 = its[1].get("tests_passed")
            if v0 is None or v1 is None:
                continue
            if v0:
                qr[coh]["tpt"] += 1
                if v1 is False and tp1 is True:
                    qr[coh]["tpf"] += 1
            else:
                qr[coh]["fpt"] += 1
                if v1 is True or tp1 is False:
                    qr[coh]["fpr"] += 1

    print(f"{'cohort':<16}{'q':>20}{'r':>20}{'tau*':>7}{'tau* CI':>14}"
          f"{'d_naive':>9}{'d_sel':>8}{'sel-naive':>11}{'pred':>11}{'match':>7}")
    prospective_keys = set(PROSPECTIVE_MODELS.values())
    present = {c for c in qr if qr[c]["tpt"] + qr[c]["fpt"] > 0}
    ordered = ([c for c in COHORT_ORDER if c in present]
               + sorted(present - set(COHORT_ORDER)))
    points = []
    for coh in ordered:
        d = qr[coh]
        if d["tpt"] + d["fpt"] == 0:
            continue
        q = d["tpf"] / d["tpt"] if d["tpt"] else 0.0
        r = d["fpr"] / d["fpt"] if d["fpt"] else 0.0
        q_lo, q_hi = wilson_ci(d["tpf"], d["tpt"])
        r_lo, r_hi = wilson_ci(d["fpr"], d["fpt"])
        tau = r / (q + r) if q + r else 0.0
        t_lo = r_lo / (q_hi + r_lo) if q_hi + r_lo else 0.0
        t_hi = r_hi / (q_lo + r_hi) if q_lo + r_hi else 1.0
        dn = mean(deltas[(coh, "naive")]) * 100
        ds = mean(deltas[(coh, "selective")]) * 100
        pred = "selective" if tau > 0.5 else "naive"
        match = (tau > 0.5) == (ds > dn)
        is_prosp = coh in prospective_keys
        points.append((coh, tau, ds - dn, match, is_prosp))
        flag = "*" if is_prosp else " "
        print(f"{flag}{coh:<15}  {q:.3f} ({d['tpf']:>3}/{d['tpt']:<3})"
              f"   {r:.3f} ({d['fpr']:>3}/{d['fpt']:<3})"
              f" {tau:>6.3f}  [{t_lo:.2f},{t_hi:.2f}]"
              f" {dn:>+8.2f}{ds:>+8.2f}{ds - dn:>+11.2f}{pred:>11}"
              f"{'YES' if match else 'no':>7}")

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]

    def pearson(a, b):
        ma, mb = mean(a), mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den

    def ranks(v):
        # midranks (tie-corrected), matching scripts/r_capability.py
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            mid = (i + j) / 2
            for k in range(i, j + 1):
                out[order[k]] = mid
            i = j + 1
        return out

    def sign_report(pts, label):
        if not pts:
            return
        k = sum(1 for p in pts if p[3])
        m = len(pts)
        p = sum(math.comb(m, j) for j in range(k, m + 1)) / 2 ** m
        line = f"{label:<22} sign agreement {k}/{m}, one-sided binomial p = {p:.4f}"
        if m >= 2:
            xx = [q[1] for q in pts]
            yy = [q[2] for q in pts]
            line += f"  | Spearman rho = {pearson(ranks(xx), ranks(yy)):+.3f}"
        print(line)

    print()
    if len(xs) >= 2:
        print(f"Pearson(tau*, sel-naive)  = {pearson(xs, ys):+.3f}")
        print(f"Spearman(tau*, sel-naive) = {pearson(ranks(xs), ranks(ys)):+.3f}")
    insample = [p for p in points if not p[4]]
    prosp = [p for p in points if p[4]]
    sign_report(points, "all cohorts:")
    sign_report(insample, "in-sample (law fit):")
    sign_report(prosp, "out-of-sample (new):")

    # Regret: sign-agreement weights a wrong call on a 1pp coin-flip the same
    # as a wrong call on a 6pp effect. Regret (pp of JointPass lost vs the
    # oracle policy choice) weights by effect size, which is what actually
    # matters operationally. gap = sel - naive (per point p[2]).
    def regret_report(pts, label):
        if not pts:
            return
        # follow tau*: lose |gap| only when the prediction is wrong
        follow = mean(0.0 if p[3] else abs(p[2]) for p in pts)
        # always-naive loses when selective was better (gap > 0)
        alw_naive = mean(max(0.0, p[2]) for p in pts)
        # always-selective loses when naive was better (gap < 0)
        alw_sel = mean(max(0.0, -p[2]) for p in pts)
        print(f"{label:<22} follow tau* = {follow:.2f}pp | "
              f"always-naive = {alw_naive:.2f}pp | always-selective = {alw_sel:.2f}pp")

    print("\nMean regret (pp JointPass lost vs oracle policy):")
    regret_report(points, "all cohorts:")
    regret_report(insample, "in-sample:")
    regret_report(prosp, "out-of-sample:")

    # --- Baseline comparison (review M1): the sign test above uses a 50%
    # coin-flip null, but the class distribution is skewed (naive wins most
    # models), so the informative baseline is 'always predict naive'.
    import random
    def baseline_block(pts, label):
        if not pts:
            return
        base_ok = sum(1 for p in pts if p[2] < 0)          # naive actually won
        tau_ok = sum(1 for p in pts if p[3])
        b = sum(1 for p in pts if p[3] and p[2] > 0)       # tau* right, baseline wrong
        c = sum(1 for p in pts if p[2] < 0 and not p[3])   # baseline right, tau* wrong
        nd = b + c
        p_mcn = (sum(math.comb(nd, k) for k in range(b, nd + 1)) / 2 ** nd
                 if nd else 1.0)
        print(f"{label:<22} tau* {tau_ok}/{len(pts)} vs always-naive {base_ok}/{len(pts)}"
              f"  (discordant {b} vs {c}, one-sided exact McNemar p = {p_mcn:.3f})")

    print("\nSign accuracy vs the always-naive baseline:")
    baseline_block(points, "all cohorts:")
    baseline_block(prosp, "out-of-sample:")

    # --- Bootstrap uncertainty over models for the mean regrets and for the
    # (always-naive - follow-tau*) regret difference (review M1).
    def regret_of(p):
        return (0.0 if p[3] else abs(p[2]),   # follow tau*
                max(0.0, p[2]),               # always-naive
                max(0.0, -p[2]))              # always-selective
    rng = random.Random(0)
    B = 10_000
    boots = {k: [] for k in ("follow", "naive", "sel", "diff")}
    for _ in range(B):
        smp = [points[rng.randrange(len(points))] for _ in points]
        f = mean(regret_of(p)[0] for p in smp)
        nv = mean(regret_of(p)[1] for p in smp)
        sl = mean(regret_of(p)[2] for p in smp)
        boots["follow"].append(f)
        boots["naive"].append(nv)
        boots["sel"].append(sl)
        boots["diff"].append(nv - f)
    def ci(v):
        s = sorted(v)
        return s[int(0.025 * B)], s[int(0.975 * B)]
    print("\nBootstrap 95% CIs over models (all cohorts, B=10000):")
    for key, name in (("follow", "follow tau*"), ("naive", "always-naive"),
                      ("sel", "always-selective")):
        lo, hi = ci(boots[key])
        print(f"  {name:<17} mean {mean(boots[key]):.2f}pp  [{lo:.2f}, {hi:.2f}]")
    lo, hi = ci(boots["diff"])
    frac_pos = sum(1 for v in boots["diff"] if v > 0) / B
    print(f"  naive - follow    mean {mean(boots['diff']):.2f}pp  [{lo:.2f}, {hi:.2f}]"
          f"  P(diff>0) = {frac_pos:.3f}")

    n_match = sum(1 for p in points if p[3])
    n = len(points)

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fam = {"haiku": "#1f77b4", "sonnet": "#d62728",
               "gemma": "#2ca02c", "qwen": "#9467bd",
               "llama": "#ff7f0e", "deepseek": "#8c564b",
               "glm": "#17becf"}
        by = {p[0]: p for p in points}
        fig, ax = plt.subplots(figsize=(8.5, 6))
        for coh, tau, diff, _, is_prosp in points:
            color = next((v for k, v in fam.items() if k in coh), "#7f7f7f")
            # open diamonds = out-of-sample prospective cohorts
            kw = (dict(marker="D", facecolors="none", edgecolors=color,
                       linewidths=1.8) if is_prosp
                  else dict(marker="o", color=color))
            ax.scatter(tau, diff, s=75, zorder=3, **kw)
            off = (8, 6) if coh not in ("sonnet46", "haiku-may") else (8, -12)
            ax.annotate(LABELS[coh], (tau, diff),
                        textcoords="offset points", xytext=off, fontsize=8)
        for a, b in [("sonnet4", "sonnet46"), ("qwen-v1", "qwen-v2"),
                     ("gemma-v1", "gemma-v2"), ("haiku-apr", "haiku-may"),
                     ("haiku-may", "haiku-v2")]:
            if a in by and b in by:
                ax.annotate("", xy=by[b][1:3], xytext=by[a][1:3],
                            arrowprops=dict(arrowstyle="->", color="gray",
                                            lw=1, alpha=0.6))
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xlabel(r"calibrated $\tau^* = \hat{r}/(\hat{q}+\hat{r})$")
        ax.set_ylabel(r"$\Delta$JP(selective) $-$ $\Delta$JP(naive)  (pp)")
        ax.set_title(r"$\tau^*$ predicts the naive-vs-selective winner"
                     f"\nsign agreement {n_match}/{n}, "
                     f"Spearman $\\rho={pearson(ranks(xs), ranks(ys)):+.2f}$")
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
