#!/usr/bin/env python3
"""Magazine-styled r-vs-capability figure for the IEEE Software piece.

Same data as scripts/r_capability.py --collapse-eras (15 distinct models),
restyled for print: large type, plain-language axes, vendor-colored points
with direct labels in text ink, a recessive trend line, and one annotation
on the Sonnet 4 outlier. Colors are the dataviz reference categorical
palette (validated for CVD separation on a light surface); identity is
never color-alone because every point carries a visible label.

Usage:
    python scripts/magazine_figure.py            # writes PNG (300dpi) + PDF
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tau_star_prediction import load_runs, jp
from r_capability import accumulate_qr, pearson, ranks, MIN_TRIALS

ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = ROOT / "paper" / "figures" / "magazine_r_vs_capability.png"
OUT_PDF = ROOT / "paper" / "figures" / "magazine_r_vs_capability.pdf"

# Validated categorical palette (light surface), fixed slot order; vendors
# assigned by panel size so adjacent-slot CVD separation does the most work.
VENDOR_COLOR = {
    "Alibaba":   "#2a78d6",  # slot 1 blue  (6 models)
    "Anthropic": "#1baf7a",  # slot 2 aqua  (3)
    "Google":    "#eda100",  # slot 3 yellow(2)
    "DeepSeek":  "#008300",  # slot 4 green (2)
    "Meta":      "#4a3aa7",  # slot 5 violet(1)
    "Zhipu":     "#e34948",  # slot 6 red   (1)
}
INK, INK2, GRID = "#1a1a19", "#5f5e56", "#e4e3dd"

DISPLAY = {
    "haiku-4.5": ("Haiku 4.5", "Anthropic"), "sonnet4": ("Sonnet 4", "Anthropic"),
    "sonnet46": ("Sonnet 4.6", "Anthropic"), "gemma4-31b": ("Gemma4-31B", "Google"),
    "gemma3-27b": ("Gemma3-27B", "Google"), "qwen3-8b": ("Qwen3-8B", "Alibaba"),
    "qwen3-14b": ("Qwen3-14B", "Alibaba"), "qwen3-32b": ("Qwen3-32B", "Alibaba"),
    "qwen3.5-27b": ("Qwen3.5-27B", "Alibaba"), "qwen3.6-27b": ("Qwen3.6-27B", "Alibaba"),
    "qwen3-coder-480b(cloud)": ("Qwen3-coder-480B", "Alibaba"),
    "llama3.1-8b": ("Llama3.1-8B", "Meta"),
    "deepseek-v2-16b": ("DeepSeek-V2-16B", "DeepSeek"),
    "deepseek-v4-flash(cloud)": ("DeepSeek-V4-flash", "DeepSeek"),
    "glm-4.6(cloud)": ("GLM-4.6", "Zhipu"),
    "opus-4.8": ("Opus 4.8", "Anthropic"), "fable-5": ("Fable 5", "Anthropic"),
}
from model_registry import load_registry, EXTRA_VENDOR_COLORS
for _tag, _e in load_registry().items():
    DISPLAY[_e["cohort"]] = (_e.get("display", _e["cohort"]), _e.get("vendor", "?"))
VENDOR_COLOR.update(EXTRA_VENDOR_COLORS)

# Hand-tuned label offsets (points) after visual inspection.
NUDGE = {
    "Qwen3-coder-480B": (0, 13), "DeepSeek-V4-flash": (-6, -13),
    "Haiku 4.5": (8, -4), "GLM-4.6": (10, 8), "Qwen3.5-27B": (-8, -16),
    "Sonnet 4.6": (10, 2), "Gemma4-31B": (-8, -8), "Qwen3-32B": (8, 6),
    "Sonnet 4": (12, -14), "Opus 4.8": (7, 7),
    "Kimi-K2.7-code": (-8, -14), "GPT-OSS-120B": (0, 14),
    "MiniMax-M2.7": (-10, -13), "Nemotron-3-Super": (7, -12),
    "DeepSeek-V4-pro": (-10, -13),
}


def collect():
    cap = defaultdict(list)
    qr = defaultdict(lambda: dict(tpf=0, tpt=0, fpr=0, fpt=0))
    for coh, cond, traces in load_runs(ROOT / "data" / "results", collapse=True):
        cap[coh].append(mean(jp(t["iterations"][0]) for t in traces))
        accumulate_qr(qr[coh], traces)
    rows = []
    for coh, d in qr.items():
        if d["fpt"] < MIN_TRIALS or d["tpt"] < MIN_TRIALS:
            continue
        name, vendor = DISPLAY[coh]
        rows.append((name, vendor, mean(cap[coh]) * 100, d["fpr"] / d["fpt"]))
    return sorted(rows, key=lambda x: x[2])


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = collect()
    caps = [x[2] for x in rows]
    rs = [x[3] for x in rows]
    rho = pearson(ranks(caps), ranks(rs))

    plt.rcParams.update({
        "font.size": 12, "font.family": "sans-serif", "text.color": INK,
        "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    fig.patch.set_facecolor("white")

    # Recessive grid + trend line first, marks on top.
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    mx, my = mean(caps), mean(rs)
    b = sum((x - mx) * (y - my) for x, y in zip(caps, rs)) / \
        sum((x - mx) ** 2 for x in caps)
    a = my - b * mx
    xs = [min(caps) - 2, max(caps) + 2]
    ax.plot(xs, [a + b * x for x in xs], ls=(0, (5, 4)), lw=1.4,
            color=INK2, alpha=0.55, zorder=1)

    for name, vendor, c, r in rows:
        ax.scatter(c, r, s=120, color=VENDOR_COLOR[vendor], zorder=3,
                   edgecolors="white", linewidths=1.6)
        dx, dy = NUDGE.get(name, (8, 4))
        ha = "center" if dx == 0 else ("right" if dx < 0 else "left")
        ax.annotate(name, (c, r), textcoords="offset points", xytext=(dx, dy),
                    fontsize=9.5, color=INK, ha=ha, zorder=4)

    # The one annotation the story needs: Sonnet 4 sits above the trend.
    s4 = next(x for x in rows if x[0] == "Sonnet 4")
    ax.annotate("Sonnet 4: unusually high $r$\nfor its capability — one\n"
                "version later, $r$ halved",
                xy=(s4[2] + 0.4, s4[3] + 0.012), xytext=(24, 46),
                textcoords="offset points", fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"), zorder=5)

    ax.set_xlabel("Model capability (baseline task success, %)", fontsize=12.5)
    ax.set_ylabel("Regression rate $r$\n(share of false alarms that break working code)",
                  fontsize=12.5)
    ax.set_ylim(0.08, 0.82)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [plt.Line2D([], [], marker="o", ls="", markersize=8.5,
                          markerfacecolor=col, markeredgecolor="white",
                          markeredgewidth=1.2, label=v)
               for v, col in VENDOR_COLOR.items()]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9.5,
              ncol=2, handletextpad=0.2, columnspacing=0.9, borderaxespad=0.2,
              labelcolor=INK2)

    ax.text(0.99, 1.02, f"rank correlation −{abs(rho):.2f} across {len(rows)} models",
            transform=ax.transAxes, ha="right", fontsize=10, color=INK2)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    print(f"wrote {OUT_PNG} and {OUT_PDF}  (Spearman {rho:+.2f}, n={len(rows)})")


if __name__ == "__main__":
    main()
