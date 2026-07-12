#!/usr/bin/env python3
"""Generate the noisy verifier model diagram (Figure 3).

Usage:
    python scripts/plot_noisy_verifier_model.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 9,
})


def main():
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Colors ────────────────────────────────────────────────────────────
    green_bg = "#d4edda"
    green_brd = "#28a745"
    yellow_bg = "#fff3cd"
    yellow_brd = "#ffc107"
    blue_bg = "#cce5ff"
    blue_brd = "#0d6efd"
    red_bg = "#f8d7da"
    red_brd = "#dc3545"
    orange_bg = "#fff8e1"
    orange_brd = "#ff9800"

    # ── Box dimensions ────────────────────────────────────────────────────
    bw, bh = 1.8, 0.9  # box width, height
    row_y = 4.8         # top row center y
    gap = 1.9           # gap between boxes

    # Box centers (x positions, evenly spaced)
    x1 = 1.8
    x2 = x1 + bw + gap
    x3 = x2 + bw + gap
    x4 = x3 + bw + gap

    def draw_box(x, y, w, h, text, bg, border, fontsize=8.5):
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=bg, edgecolor=border, linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", linespacing=1.4)

    # ── Top row: pipeline boxes ───────────────────────────────────────────
    draw_box(x1, row_y, bw, bh,
             "LLM Agent\ngenerates $C_k$", green_bg, green_brd)
    draw_box(x2, row_y, bw, bh,
             "Analyzers\nSemgrep + Bandit\n$F = V(C_k)$", yellow_bg, yellow_brd, fontsize=8)
    draw_box(x3, row_y, bw, bh,
             "Feedback Policy\nselect $S \\subseteq F$\nwhere $\\hat{p}_i \\geq \\tau$",
             blue_bg, blue_brd, fontsize=8)
    draw_box(x4, row_y, bw, bh,
             "LLM Agent\nrevises → $C_{k+1}$", green_bg, green_brd)

    # ── Arrows between boxes (horizontal) ─────────────────────────────────
    arrow_kw = dict(arrowstyle="-|>", color="#333333", linewidth=1.5,
                    mutation_scale=14)
    arr_y = row_y

    # Arrow 1→2: "code"
    ax.annotate("", xy=(x2 - bw / 2 - 0.05, arr_y),
                xytext=(x1 + bw / 2 + 0.05, arr_y),
                arrowprops=arrow_kw)
    ax.text((x1 + x2) / 2, arr_y + 0.22, "code",
            ha="center", va="bottom", fontsize=8, style="italic")

    # Arrow 2→3: "75 rules\n23 findings"
    ax.text((x2 + x3) / 2, arr_y + 0.22, "75 rules\n23 findings",
            ha="center", va="bottom", fontsize=7.5, style="italic",
            linespacing=1.2)
    ax.annotate("", xy=(x3 - bw / 2 - 0.05, arr_y),
                xytext=(x2 + bw / 2 + 0.05, arr_y),
                arrowprops=arrow_kw)

    # Arrow 3→4: "filtered\nfindings"
    ax.text((x3 + x4) / 2, arr_y + 0.22, "filtered\nfindings",
            ha="center", va="bottom", fontsize=7.5, style="italic",
            linespacing=1.2)
    ax.annotate("", xy=(x4 - bw / 2 - 0.05, arr_y),
                xytext=(x3 + bw / 2 + 0.05, arr_y),
                arrowprops=arrow_kw)

    # ── Curved "iterate" arrow (box4 top → box1 top) ─────────────────────
    iterate_y = row_y + bh / 2 + 0.08
    ax.annotate(
        "", xy=(x1, iterate_y),
        xytext=(x4, iterate_y),
        arrowprops=dict(
            arrowstyle="-|>", color=blue_brd, linewidth=2,
            connectionstyle="arc3,rad=0.35", mutation_scale=16,
        ),
    )
    ax.text((x1 + x4) / 2, row_y + bh / 2 + 0.95,
            "iterate (max 5 rounds)",
            ha="center", va="bottom", fontsize=9, color=blue_brd,
            fontweight="bold")

    # ── Calibration line (above the iterate arrow) ────────────────────────
    ax.text((x1 + x4) / 2, row_y + bh / 2 + 1.35,
            "Calibration: 23.8% precision (36 TP / 115 FP across 75 rules)",
            ha="center", va="bottom", fontsize=8, color="#555555",
            style="italic")

    # ── Orange decision banner ────────────────────────────────────────────
    banner_y = 3.35
    banner_w = 8.2
    banner_h = 0.45
    banner_x = (x1 + x4) / 2
    banner = FancyBboxPatch(
        (banner_x - banner_w / 2, banner_y - banner_h / 2),
        banner_w, banner_h,
        boxstyle="round,pad=0.08",
        facecolor=orange_bg, edgecolor=orange_brd, linewidth=1.5,
    )
    ax.add_patch(banner)
    ax.text(banner_x, banner_y,
            "Per-finding decision: surface finding $i$ only if "
            "$\\hat{p}_i \\geq \\tau^* = r/(q+r)$",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="#e65100")

    # ── TP and FP boxes ───────────────────────────────────────────────────
    tp_x = banner_x - 2.2
    fp_x = banner_x + 2.2
    outcome_y = 1.85
    outcome_w = 3.6
    outcome_h = 1.1

    draw_box(tp_x, outcome_y, outcome_w, outcome_h,
             "True Positive surfaced\n"
             "prob $p_i$: finding IS a vulnerability\n"
             "LLM fixes with prob $q$\n"
             "Haiku: $q$=0.23  |  Qwen: $q$=0.28",
             green_bg, green_brd, fontsize=7.5)

    draw_box(fp_x, outcome_y, outcome_w, outcome_h,
             "False Positive surfaced\n"
             "prob $1-p_i$: finding is NOT a vulnerability\n"
             'LLM "fix" causes regression with prob $r$\n'
             "Haiku: $r$=0.08  |  Qwen: $r$=0.53",
             red_bg, red_brd, fontsize=7.5)

    # ── JointPass labels (below outcome boxes) ───────────────────────────
    jp_y = outcome_y - outcome_h / 2 - 0.3
    ax.text(tp_x, jp_y, "→ JointPass improves",
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            color=green_brd)
    ax.text(fp_x, jp_y, "→ JointPass degrades",
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            color=red_brd)

    # ── Optimal threshold footer ──────────────────────────────────────────
    ax.text((x1 + x4) / 2, 0.35,
            "Optimal threshold:  "
            "$\\tau^*_{\\mathrm{Haiku}} = 0.08/(0.23+0.08) = 0.26$ (low threshold)"
            "      "
            "$\\tau^*_{\\mathrm{Qwen}} = 0.53/(0.28+0.53) = 0.66$ "
            "(filter low-precision rules)",
            ha="center", va="center", fontsize=8, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8eaf6",
                      edgecolor="#5c6bc0", linewidth=1))

    # ── Save ──────────────────────────────────────────────────────────────
    out_png = "figures/noisy_verifier_model.png"
    out_pdf = "figures/noisy_verifier_model.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"Saved: {out_png}, {out_pdf}")


if __name__ == "__main__":
    main()
