#!/usr/bin/env python3
"""Compute formal statistical tests for the seven main findings.

For each (finding, model) cell we run:
  1. McNemar's exact (mid-P) test on paired binary per-item JointPass
     outcomes. Each (item, seed-index) is one matched pair across the two
     conditions being contrasted.
  2. Wilcoxon signed-rank on per-seed JointPass proportions (each seed
     yields one 51-item proportion; pair across conditions on the seed
     index). Exact mode where n is small.
  3. Cliff's delta on the per-seed differences as a non-parametric
     effect size, with Romano et al. magnitude labels.
  4. Holm-Bonferroni correction across the family of cells within each
     finding.

Per-finding family definitions:
  F1  improvement under naive: k=final vs k=0, family = 4 models.
  F2  improvement under selective: k=final vs k=0, family = 4 models.
  F3  regression under naive: k=final vs k=0, family = 4 models (Sonnet
      and Qwen3-8B are the claimed regressors; reporting the same data
      keeps Holm honest).
  F4  capability gradient: best-fixed-policy JP@final, family = 3
      adjacent contrasts (Gemma>Haiku, Haiku>Sonnet, Sonnet>Qwen).
  F5  Haiku tau-insensitivity: naive vs selective at JP@final, family
      = 1 cell.
  F6  Qwen3-8B naive reversal: naive JP@final vs naive JP@0, family
      = 1 cell (also reported under F1/F3).
  F7  format ablation on Haiku naive: NL vs raw_sarif, NL vs minimal at
      JP@final, family = 2 cells.

Usage:
    python scripts/compute_significance.py
    python scripts/compute_significance.py --json /tmp/sig.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

INCLUDED_CORE_SIZES = {50, 51}


def model_short(s: str) -> str:
    s = s.lower()
    if "haiku" in s:
        return "haiku"
    if "sonnet" in s:
        return "sonnet"
    if "qwen" in s:
        return "qwen3-8b"
    if "gemma" in s:
        return "gemma4-31b"
    return s


def jp(it: dict) -> int | None:
    t = it.get("tests_passed")
    v = it.get("has_vulnerability")
    if t is None or v is None:
        return None
    return int(bool(t) and not bool(v))


def load_runs(results_dir: Path) -> dict[tuple, list[dict]]:
    """Return {(model_short, condition, format, tau): [run_dict, ...]}.

    Each run_dict carries `items`: list of dicts {item_id, jp0, jp_final}.
    Runs are sorted by directory name (timestamp) so the index serves as
    a stable seed identifier within a (model, condition, format, tau)
    group.
    """
    out: dict[tuple, list[dict]] = defaultdict(list)
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or "synthetic" in d.name:
            continue
        traces_path = d / "traces.jsonl"
        if not traces_path.exists():
            continue
        traces = []
        with open(traces_path) as f:
            for line in f:
                obj = json.loads(line)
                if "error" not in obj:
                    traces.append(obj)
        if len(traces) not in INCLUDED_CORE_SIZES:
            continue
        cfg_path = d / "config.yaml"
        fmt = "natural_language"
        tau = 0.5
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            fmt = cfg.get("feedback", {}).get("format", "natural_language")
            tau = cfg.get("selective", {}).get("threshold_tau", 0.5)
        condition = traces[0].get("condition", "?")
        m = model_short(traces[0].get("model", ""))
        items = []
        for t in traces:
            iters = t.get("iterations", [])
            if not iters:
                continue
            items.append(dict(
                item_id=t["item_id"],
                jp0=jp(iters[0]),
                jp_final=jp(iters[-1]),
            ))
        out[(m, condition, fmt, tau)].append(dict(run_dir=d.name, items=items))
    return out


def pair_items_seed_index(runs_a: list[dict], runs_b: list[dict],
                          field_a: str, field_b: str
                          ) -> list[tuple[int, int]]:
    """Build paired (a, b) outcome list across the seed-index dimension.

    Aligns runs_a[i] with runs_b[i] for i in 0..min(len_a, len_b). Within
    each aligned pair, items are matched by item_id.
    """
    pairs: list[tuple[int, int]] = []
    n = min(len(runs_a), len(runs_b))
    for i in range(n):
        idx_b = {it["item_id"]: it for it in runs_b[i]["items"]}
        for it_a in runs_a[i]["items"]:
            jt_a = it_a[field_a]
            mate = idx_b.get(it_a["item_id"])
            if mate is None:
                continue
            jt_b = mate[field_b]
            if jt_a is None or jt_b is None:
                continue
            pairs.append((jt_a, jt_b))
    return pairs


def per_seed_proportions(runs_a: list[dict], runs_b: list[dict],
                         field_a: str, field_b: str
                         ) -> tuple[list[float], list[float]]:
    """One proportion per aligned seed-index for the two conditions."""
    a, b = [], []
    n = min(len(runs_a), len(runs_b))
    for i in range(n):
        items_a = [it[field_a] for it in runs_a[i]["items"] if it[field_a] is not None]
        items_b = [it[field_b] for it in runs_b[i]["items"] if it[field_b] is not None]
        if not items_a or not items_b:
            continue
        a.append(sum(items_a) / len(items_a))
        b.append(sum(items_b) / len(items_b))
    return a, b


def mcnemar_exact_midp(b: int, c: int) -> float:
    """Exact mid-P McNemar test on discordant counts b and c.

    Returns a two-sided p-value. mid-P avoids the conservatism of the
    standard exact test by giving half-weight to the observed point.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Two-sided exact tail (binomial with p=0.5)
    # mid-P = 2 * (sum_{i<k} C(n,i) p^i (1-p)^(n-i)) + C(n,k) p^k (1-p)^(n-k)
    log_half_n = -n * math.log(2)
    tail = 0.0
    for i in range(k):
        log_term = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n
        tail += math.exp(log_term)
    log_point = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + log_half_n
    mid = 2 * tail + math.exp(log_point)
    return min(1.0, mid)


def cliffs_delta(x: list[float], y: list[float]) -> float:
    if not x or not y:
        return 0.0
    gt = lt = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                gt += 1
            elif xi < yi:
                lt += 1
    return (gt - lt) / (len(x) * len(y))


def cliffs_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def one_cell(runs_a, runs_b, field_a, field_b) -> dict:
    """Run McNemar (exact mid-P), Wilcoxon (exact when feasible), and
    Cliff's delta for one comparison cell.
    """
    pairs = pair_items_seed_index(runs_a, runs_b, field_a, field_b)
    n_total = len(pairs)
    b = sum(1 for a, c in pairs if a == 0 and c == 1)  # 0 -> 1 (improvement)
    c = sum(1 for a, c in pairs if a == 1 and c == 0)  # 1 -> 0 (regression)
    n_disc = b + c
    p_mcnemar = mcnemar_exact_midp(b, c) if n_disc > 0 else 1.0

    # Per-seed Wilcoxon and Cliff's delta on the deltas
    pa, pb = per_seed_proportions(runs_a, runs_b, field_a, field_b)
    diffs = [pb_i - pa_i for pa_i, pb_i in zip(pa, pb)]
    n_seed = len(diffs)
    delta_mean = float(np.mean(diffs)) if diffs else 0.0
    if n_seed >= 2 and any(d != 0 for d in diffs):
        try:
            wres = stats.wilcoxon(pb, pa, zero_method="wilcox",
                                  alternative="two-sided",
                                  method="exact" if n_seed <= 25 else "auto")
            p_wilcoxon = float(wres.pvalue)
        except ValueError:
            p_wilcoxon = 1.0
    else:
        p_wilcoxon = 1.0
    cd = cliffs_delta(pb, pa)
    return dict(
        n_pairs=n_total, b_0_to_1=b, c_1_to_0=c, n_discordant=n_disc,
        p_mcnemar=p_mcnemar, p_wilcoxon=p_wilcoxon, cliffs_delta=cd,
        cliffs_magnitude=cliffs_magnitude(cd), n_seeds=n_seed,
        delta_mean_jp=delta_mean,
    )


def best_fixed_for(model: str, runs: dict) -> tuple[str, float, list[dict]]:
    """Return (condition, mean_final_jp, runs_list) for the model's best
    fixed (i.e. non-adaptive) policy at default tau=0.5, format=NL.
    """
    candidates = []
    for cond in ("naive", "selective", "llm_judge"):
        key = (model, cond, "natural_language", 0.5)
        rs = runs.get(key, [])
        if not rs:
            continue
        per_seed = []
        for r in rs:
            vals = [it["jp_final"] for it in r["items"] if it["jp_final"] is not None]
            if vals:
                per_seed.append(sum(vals) / len(vals))
        if per_seed:
            candidates.append((cond, float(np.mean(per_seed)), rs))
    if not candidates:
        return "?", 0.0, []
    return max(candidates, key=lambda x: x[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    runs = load_runs(Path(args.results_dir))

    MODELS = ["haiku", "sonnet", "gemma4-31b", "qwen3-8b"]

    def naive_key(m): return (m, "naive", "natural_language", 0.5)
    def selective_key(m): return (m, "selective", "natural_language", 0.5)

    families: list[dict] = []

    # ── F1: improvement under naive (final vs k=0), 4-model family
    cells = []
    for m in MODELS:
        rs = runs.get(naive_key(m), [])
        cell = dict(finding="F1", model=m, contrast="naive: JP@final - JP@0",
                    **one_cell(rs, rs, "jp0", "jp_final"))
        cells.append(cell)
    families.append(dict(name="F1: feedback improves strong models (naive)",
                         cells=cells))

    # ── F2: improvement under selective, 4-model family
    cells = []
    for m in MODELS:
        rs = runs.get(selective_key(m), [])
        cell = dict(finding="F2", model=m,
                    contrast="selective: JP@final - JP@0",
                    **one_cell(rs, rs, "jp0", "jp_final"))
        cells.append(cell)
    families.append(dict(name="F2: selective preserves or improves JP",
                         cells=cells))

    # ── F3: regression under naive, 4-model family (uses same data as F1
    #       but is tested as a separate family because the claim asymmetry
    #       matters and Holm corrects within each finding).
    cells = []
    for m in MODELS:
        rs = runs.get(naive_key(m), [])
        cell = dict(finding="F3", model=m, contrast="naive: JP@final - JP@0",
                    **one_cell(rs, rs, "jp0", "jp_final"))
        cells.append(cell)
    families.append(dict(name="F3: naive harms mid-tier models",
                         cells=cells))

    # ── F4: capability gradient, 3 adjacent contrasts on best-fixed JP@final.
    cells = []
    best = {m: best_fixed_for(m, runs) for m in MODELS}
    ladder = ["gemma4-31b", "haiku", "sonnet", "qwen3-8b"]
    for hi, lo in zip(ladder, ladder[1:]):
        hi_cond, hi_mean, rs_hi = best[hi]
        lo_cond, lo_mean, rs_lo = best[lo]
        cell = dict(
            finding="F4",
            model=f"{hi}({hi_cond}) vs {lo}({lo_cond})",
            contrast=f"JP@final: {hi}-{lo}",
            **one_cell(rs_lo, rs_hi, "jp_final", "jp_final"),
        )
        cell["hi_mean_jp"] = hi_mean
        cell["lo_mean_jp"] = lo_mean
        cells.append(cell)
    families.append(dict(name="F4: capability gradient at best fixed policy",
                         cells=cells))

    # ── F5: Haiku tau-insensitivity: naive vs selective at JP@final.
    rs_n = runs.get(naive_key("haiku"), [])
    rs_s = runs.get(selective_key("haiku"), [])
    cells = [dict(finding="F5", model="haiku",
                  contrast="naive vs selective: JP@final",
                  **one_cell(rs_n, rs_s, "jp_final", "jp_final"))]
    families.append(dict(name="F5: Haiku tau-insensitivity", cells=cells))

    # ── F6: Qwen3-8B naive reversal (same data as F1/F3 Qwen cell).
    rs = runs.get(naive_key("qwen3-8b"), [])
    cells = [dict(finding="F6", model="qwen3-8b",
                  contrast="naive: JP@final - JP@0",
                  **one_cell(rs, rs, "jp0", "jp_final"))]
    families.append(dict(name="F6: Qwen3-8B naive reversal", cells=cells))

    # ── F7: format ablation on Haiku naive.
    rs_nl = runs.get(("haiku", "naive", "natural_language", 0.5), [])
    rs_sarif = runs.get(("haiku", "naive", "raw_sarif", 0.5), [])
    rs_minimal = runs.get(("haiku", "naive", "minimal", 0.5), [])
    cells = [
        dict(finding="F7", model="haiku-NL-vs-SARIF",
             contrast="JP@final: NL vs raw_sarif",
             **one_cell(rs_sarif, rs_nl, "jp_final", "jp_final")),
        dict(finding="F7", model="haiku-NL-vs-minimal",
             contrast="JP@final: NL vs minimal",
             **one_cell(rs_minimal, rs_nl, "jp_final", "jp_final")),
    ]
    families.append(dict(name="F7: NL format vs raw_sarif/minimal", cells=cells))

    # ── F8: adaptive vs best fixed policy per model.
    cells = []
    for m in MODELS:
        rs_ad = runs.get((m, "adaptive", "natural_language", 0.5), [])
        best_cond, best_mean, rs_best = best_fixed_for(m, runs)
        if not rs_ad or not rs_best:
            # Adaptive runs not available for this model — record an
            # empty cell with NaN-ish placeholders so Holm correction
            # ignores it.
            cells.append(dict(
                finding="F8", model=m,
                contrast=f"adaptive vs {best_cond}: JP@final",
                n_pairs=0, b_0_to_1=0, c_1_to_0=0, n_discordant=0,
                p_mcnemar=1.0, p_wilcoxon=1.0, cliffs_delta=0.0,
                cliffs_magnitude="negligible", n_seeds=0,
                delta_mean_jp=0.0, hi_mean_jp=0.0, lo_mean_jp=best_mean,
            ))
            continue
        # Order matters: best_fixed is the baseline (a), adaptive is the new (b)
        cell = dict(
            finding="F8", model=m,
            contrast=f"adaptive vs {best_cond}: JP@final",
            **one_cell(rs_best, rs_ad, "jp_final", "jp_final"),
        )
        # Annotate adaptive mean for reporting
        per_seed_ad = []
        for r in rs_ad:
            vals = [it["jp_final"] for it in r["items"] if it["jp_final"] is not None]
            if vals:
                per_seed_ad.append(sum(vals) / len(vals))
        cell["hi_mean_jp"] = float(np.mean(per_seed_ad)) if per_seed_ad else 0.0
        cell["lo_mean_jp"] = best_mean
        cells.append(cell)
    families.append(dict(name="F8: adaptive vs best fixed policy", cells=cells))

    # ── Apply Holm-Bonferroni within each family on McNemar p-values.
    for fam in families:
        ps = [c["p_mcnemar"] for c in fam["cells"]]
        if not ps:
            continue
        reject, p_holm, _, _ = multipletests(ps, alpha=0.05, method="holm")
        for c, r, pa in zip(fam["cells"], reject, p_holm):
            c["p_holm"] = float(pa)
            c["survives"] = bool(r)

    # ── Print summary
    print(f"{'Finding':<6} {'Model':<32} {'Δ JP':>8} {'n_disc':>6} {'McNemar':>10} {'Wilcoxon':>10} {'Cliff δ':>10} {'Holm':>10} {'✓':>3}")
    print("-" * 102)
    for fam in families:
        for c in fam["cells"]:
            d = c["delta_mean_jp"]
            print(f"{c['finding']:<6} {c['model']:<32} {d:>+8.3f} {c['n_discordant']:>6d} "
                  f"{c['p_mcnemar']:>10.4g} {c['p_wilcoxon']:>10.4g} "
                  f"{c['cliffs_delta']:>+7.3f} ({c['cliffs_magnitude'][:3]}) "
                  f"{c['p_holm']:>10.4g} {'✓' if c['survives'] else '✗':>3}")
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(families, indent=2))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
