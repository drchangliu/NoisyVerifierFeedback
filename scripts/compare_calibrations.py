#!/usr/bin/env python3
"""Compare the original 146-item precision map against the held-out
95/96-item precision map (audit #9).

For each rule that triggered at least once in either calibration,
report side-by-side:
    n_orig, p_orig, n_held, p_held, |Δ|, threshold flip at τ = 0.5?

Also emits a LaTeX-ready table fragment.

Usage:
    python scripts/compare_calibrations.py
    python scripts/compare_calibrations.py --tex /tmp/heldout_tab.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def load_map(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--orig", default="data/calibration/precision_map_combined.json",
    )
    ap.add_argument(
        "--held", default="data/calibration/heldout_precision_map.json",
    )
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--tex", default=None, help="Optional path for LaTeX fragment")
    ap.add_argument("--min-n", type=int, default=2,
                    help="Only report rules with n >= min_n in either set")
    args = ap.parse_args()

    orig = load_map(Path(args.orig))
    held = load_map(Path(args.held))

    all_rules = sorted(set(orig) | set(held))

    rows = []
    for rule in all_rules:
        o = orig.get(rule, {})
        h = held.get(rule, {})
        n_o = o.get("n_samples", 0)
        n_h = h.get("n_samples", 0)
        if n_o < args.min_n and n_h < args.min_n:
            continue
        p_o = o.get("precision", 0.0)
        p_h = h.get("precision", 0.0)
        d = abs(p_o - p_h)
        # Threshold flip: original above τ but held below, or vice versa
        passes_o = p_o >= args.tau and n_o > 0
        passes_h = p_h >= args.tau and n_h > 0
        flip = passes_o != passes_h
        rows.append(dict(
            rule=rule, n_orig=n_o, p_orig=p_o, n_held=n_h, p_held=p_h,
            abs_delta=d, passes_orig=passes_o, passes_held=passes_h,
            threshold_flip=flip,
        ))

    rows.sort(key=lambda r: (-r["abs_delta"], r["rule"]))

    # ── Pretty table for the terminal
    tab = Table(title=f"Precision comparison (rules with n≥{args.min_n} in either set)")
    tab.add_column("Rule", max_width=58)
    tab.add_column("n_o", justify="right")
    tab.add_column("p_o", justify="right")
    tab.add_column("n_h", justify="right")
    tab.add_column("p_h", justify="right")
    tab.add_column("|Δ|", justify="right")
    tab.add_column("τ-flip", justify="center")
    for r in rows:
        tab.add_row(
            r["rule"][-58:],
            str(r["n_orig"]), f"{r['p_orig']:.2f}",
            str(r["n_held"]), f"{r['p_held']:.2f}",
            f"{r['abs_delta']:.2f}",
            "FLIP" if r["threshold_flip"] else "",
        )
    console.print(tab)

    # ── Headline summary
    flips = [r for r in rows if r["threshold_flip"]]
    big_shifts = [r for r in rows if r["abs_delta"] >= 0.15
                  and r["n_orig"] >= 4 and r["n_held"] >= 4]
    max_delta_robust = max((r["abs_delta"] for r in rows
                            if r["n_orig"] >= 4 and r["n_held"] >= 4),
                           default=0.0)

    print()
    print(f"Rules reported:                              {len(rows)}")
    print(f"Threshold-flips at τ = {args.tau}:             {len(flips)}")
    for r in flips:
        print(f"  FLIP {r['rule'][-60:]}  p_o={r['p_orig']:.2f} (n={r['n_orig']}) "
              f"vs p_h={r['p_held']:.2f} (n={r['n_held']})")
    print(f"Big shifts (|Δ| ≥ 0.15 with n ≥ 4 each):     {len(big_shifts)}")
    for r in big_shifts:
        print(f"  shift {r['rule'][-60:]}  p_o={r['p_orig']:.2f} (n={r['n_orig']}) "
              f"vs p_h={r['p_held']:.2f} (n={r['n_held']})")
    print(f"Max |Δ| for n≥4 rules:                       {max_delta_robust:.3f}")
    if max_delta_robust < 0.15 and not flips:
        print("==> calibration is robust to held-out evaluation")
    else:
        print("==> calibration shifts materially; see flips/big_shifts above")

    if args.tex:
        lines = []
        lines.append("\\begin{tabular}{lrrrrrc}")
        lines.append("\\toprule")
        lines.append("Rule & $n_{\\text{orig}}$ & $\\hat p_{\\text{orig}}$ "
                     "& $n_{\\text{held}}$ & $\\hat p_{\\text{held}}$ "
                     "& $|\\Delta|$ & $\\tau$-flip \\\\")
        lines.append("\\midrule")
        for r in rows:
            # Escape underscores in rule IDs and truncate for legibility
            r_short = r["rule"].replace("_", "\\_")
            if len(r_short) > 50:
                r_short = r_short[-50:]
            flip_mark = "$\\checkmark$" if r["threshold_flip"] else ""
            lines.append(
                f"\\texttt{{{r_short}}} & {r['n_orig']} & {r['p_orig']:.2f} "
                f"& {r['n_held']} & {r['p_held']:.2f} "
                f"& {r['abs_delta']:.2f} & {flip_mark} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        Path(args.tex).write_text("\n".join(lines))
        print(f"\nWrote LaTeX fragment to {args.tex}")


if __name__ == "__main__":
    main()
