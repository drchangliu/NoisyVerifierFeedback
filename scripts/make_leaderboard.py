#!/usr/bin/env python3
"""Generate the public trust leaderboard from the released traces.

Ranks models by the counterfactual regression rate r (ascending): how safely
each model can consume UNFILTERED static-analyzer feedback. Low r = the model
recognizes false alarms and leaves working code alone.

Outputs (all repo-root relative, single source of truth):
  LEADERBOARD.md            full table + methodology
  README.md                 compact table injected between LEADERBOARD markers
  docs/index.html           polished GitHub Pages site
  docs/r_vs_capability.png  hero figure (copied from figures/leaderboard_*.png
                            if present, else figures/r_vs_capability.png)

Usage:  python scripts/make_leaderboard.py
"""
from __future__ import annotations

import datetime
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tau_star_prediction import load_runs, jp, wilson_ci
from r_capability import accumulate_qr, MIN_TRIALS, pearson, ranks
from humaneval_correlation import humaneval_pass1

ROOT = Path(__file__).resolve().parent.parent

VENDOR = {
    "haiku-4.5": "Anthropic", "sonnet4": "Anthropic", "sonnet46": "Anthropic",
    "opus-4.8": "Anthropic", "fable-5": "Anthropic",
    "gemma4-31b": "Google", "gemma3-27b": "Google",
    "qwen3-8b": "Alibaba", "qwen3-14b": "Alibaba", "qwen3-32b": "Alibaba",
    "qwen3.5-27b": "Alibaba", "qwen3.6-27b": "Alibaba",
    "qwen3-coder-480b(cloud)": "Alibaba",
    "llama3.1-8b": "Meta",
    "deepseek-v2-16b": "DeepSeek", "deepseek-v4-flash(cloud)": "DeepSeek",
    "glm-4.6(cloud)": "Zhipu",
}
DISPLAY = {
    "haiku-4.5": "Claude Haiku 4.5", "sonnet4": "Claude Sonnet 4",
    "sonnet46": "Claude Sonnet 4.6", "opus-4.8": "Claude Opus 4.8",
    "gemma4-31b": "Gemma4-31B", "gemma3-27b": "Gemma3-27B",
    "qwen3-8b": "Qwen3-8B", "qwen3-14b": "Qwen3-14B", "qwen3-32b": "Qwen3-32B",
    "qwen3.5-27b": "Qwen3.5-27B", "qwen3.6-27b": "Qwen3.6-27B",
    "qwen3-coder-480b(cloud)": "Qwen3-coder-480B",
    "llama3.1-8b": "Llama3.1-8B",
    "deepseek-v2-16b": "DeepSeek-V2-16B",
    "deepseek-v4-flash(cloud)": "DeepSeek-V4-flash",
    "glm-4.6(cloud)": "GLM-4.6",
}
from model_registry import load_registry, EXTRA_VENDOR_COLORS, FALLBACK_VENDOR_COLOR
ADDED = {}
for _tag, _e in load_registry().items():
    VENDOR[_e["cohort"]] = _e.get("vendor", "?")
    DISPLAY[_e["cohort"]] = _e.get("display", _e["cohort"])
    if _e.get("added"):
        ADDED[_e["cohort"]] = _e["added"]

IN_SAMPLE = {"haiku-4.5", "sonnet4", "sonnet46", "gemma4-31b", "qwen3-8b"}
RETIRED = {"sonnet4", "glm-4.6(cloud)"}  # retired by their providers mid-study

VENDOR_COLOR = {"Alibaba": "#2a78d6", "Anthropic": "#1baf7a",
                "Google": "#eda100", "DeepSeek": "#008300",
                "Meta": "#4a3aa7", "Zhipu": "#e34948"}
VENDOR_COLOR.update(EXTRA_VENDOR_COLORS)

# Disclosed parameter counts (total; "MoE" = mixture-of-experts, far fewer
# active per token). Em-dash where the developer does not disclose a size --
# true of most frontier API models. Capability, not size, predicts r.
PARAMS = {
    "haiku-4.5": "\u2014", "sonnet4": "\u2014", "sonnet46": "\u2014",
    "opus-4.8": "\u2014",
    "gemma4-31b": "31B", "gemma3-27b": "27B",
    "qwen3-8b": "8B", "qwen3-14b": "14B", "qwen3-32b": "32B",
    "qwen3.5-27b": "27B", "qwen3.6-27b": "27B",
    "qwen3-coder-480b(cloud)": "480B (MoE)",
    "llama3.1-8b": "8B",
    "deepseek-v2-16b": "16B (MoE)",
    "deepseek-v4-flash(cloud)": "\u2014",
    "glm-4.6(cloud)": "\u2014",
}
for _tag, _e in load_registry().items():
    if _e.get("params"):
        PARAMS[_e["cohort"]] = _e["params"]


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
        q = d["tpf"] / d["tpt"]
        r = d["fpr"] / d["fpt"]
        lo, hi = wilson_ci(d["fpr"], d["fpt"])
        he = humaneval_pass1(coh.replace("(cloud)", ""))
        rows.append(dict(
            coh=coh, name=DISPLAY.get(coh, coh), vendor=VENDOR.get(coh, "?"),
            cap=mean(cap[coh]) * 100, q=q, r=r, r_lo=lo, r_hi=hi,
            tau=r / (q + r), policy="selective" if r / (q + r) > 0.5 else "naive",
            he=None if he is None else he * 100,
            tpt=d["tpt"], fpt=d["fpt"],
            insample=coh in IN_SAMPLE, retired=coh in RETIRED,
            params=PARAMS.get(coh, "\u2014"),
        ))
    rows.sort(key=lambda x: x["r"])
    return rows


def notes(m):
    parts = []
    if m["insample"]:
        parts.append("in-sample")
    if m["retired"]:
        parts.append("retired by provider")
    if m["coh"] in ADDED:
        parts.append(f"added {ADDED[m['coh']]}")
    return ", ".join(parts)


def md_full(rows, today):
    rho = pearson(ranks([m['cap'] for m in rows]), ranks([m['r'] for m in rows]))
    L = []
    L.append("# Trust Leaderboard: which models can safely consume unfiltered "
             "static-analyzer feedback?\n")
    L.append(f"*Updated {today} · {len(rows)} models · ranked by the "
             "**counterfactual regression rate r** (lower = better): the "
             "probability that the model, handed a **false-positive** security "
             "finding on its own working code, \"fixes\" it and breaks the "
             "code.*\n")
    L.append("![r falls with capability](docs/r_vs_capability.png)\n")
    L.append("| # | Model | Developer | Params | r (95% CI) | q | τ\\* | Better fixed "
             "policy | JP@0 | HumanEval | FP/TP trials | Notes |")
    L.append("|--:|---|---|---|---|--:|--:|---|--:|--:|--:|---|")
    for i, m in enumerate(rows, 1):
        he = "—" if m["he"] is None else f"{m['he']:.1f}%"
        L.append(
            f"| {i} | **{m['name']}** | {m['vendor']} | {m['params']} "
            f"| {m['r']:.2f} [{m['r_lo']:.2f}, {m['r_hi']:.2f}] "
            f"| {m['q']:.2f} | {m['tau']:.2f} | {m['policy']} "
            f"| {m['cap']:.1f}% | {he} | {m['fpt']}/{m['tpt']} "
            f"| {notes(m)} |")
    L.append("""
## How to read this

- **r (regression rate)** — of the loop interactions where the analyzer's
  finding was a *false alarm* on already-clean code, the fraction where the
  model's "fix" broke the code (introduced a vulnerability or failed tests).
  This is the leaderboard's ranking key: it measures whether a model can
  *recognize* a false alarm instead of blindly obeying it.
- **q (fix rate)** — of the interactions where the finding was real, the
  fraction the model actually fixed (secure *and* still passing tests).
- **τ\\* = r/(q+r)** — the minimum per-rule precision at which surfacing a
  finding helps this model more than it hurts. Feed the model a finding only
  if the rule's historical precision exceeds its τ\\*.
- **Better fixed policy** — the better of the two deployed fixed policies:
  *naive* (surface everything) vs *selective* (surface only rules with >50%
  precision). "Naive" does **not** mean surfacing everything is optimal — the
  optimum filters at the model's own τ\\*, which is above zero for every model.
- **JP@0** — baseline JointPass (functionally correct AND vulnerability-free
  before any feedback) on the 51-item core benchmark; the capability axis.
- **HumanEval** — external capability axis (pass@1, 164 tasks); missing for
  models retired before that experiment.
- *in-sample* — one of the five models used to formulate the r-law; all
  others were measured prospectively, after the law and decision rule were
  frozen.

## The headline finding

Across these models, r falls steeply as capability rises (Spearman RHOVAL) —
**better models are less gullible** — while q stays flat. So the optimal
feedback policy is a property of the *model*, slides from "filter
aggressively" toward "surface everything" as models improve, and expires with
every model update. Full study: *Better Models Are Less Gullible: Selective Feedback
for LLM Code Agents under Noisy Static Analysis* (under review at Empirical Software Engineering; the submitted
state is preserved as tag
[`emse-2026-07`](../../releases/tag/emse-2026-07)).

## Measurement notes

- Estimates pool the naive+selective runs on the 51-item core benchmark
  (CWEval + SecurityEval), combined Semgrep+Bandit analyzer, multi-seed;
  models re-measured on several dates contribute one pooled row.
- Claude Fable 5 could not be measured: its safety classifiers refused 9/10
  generation requests on this benchmark (see `data/smoke_tests/`).
- Regenerate this file: `python scripts/make_leaderboard.py`.
""")
    return "\n".join(L).replace("RHOVAL", f"{rho:+.2f}".replace("+", "\u2212" if rho < 0 else "+").replace("-", "\u2212"))


def md_compact(rows, today):
    L = []
    L.append(f"*Ranked by the regression rate **r** — the probability the "
             f"model breaks working code when handed a false-alarm security "
             f"finding (lower = harder to mislead). Updated {today}. Full "
             f"table, CIs, and methodology: [LEADERBOARD.md](LEADERBOARD.md) "
             f"· [web version](https://drchangliu.github.io/NoisyVerifierFeedback/).*\n")
    L.append("![Regression rate r falls as model capability rises "
             "(16 models, rank correlation -0.89)](docs/r_vs_capability.png)\n")
    L.append("| # | Model | r | τ\\* | Better fixed policy | HumanEval |")
    L.append("|--:|---|--:|--:|---|--:|")
    for i, m in enumerate(rows, 1):
        he = "—" if m["he"] is None else f"{m['he']:.1f}%"
        L.append(f"| {i} | {m['name']} | {m['r']:.2f} | {m['tau']:.2f} "
                 f"| {m['policy']} | {he} |")
    return "\n".join(L)


START, END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"


def inject_readme(compact):
    p = ROOT / "README.md"
    txt = p.read_text()
    block = f"{START}\n{compact}\n{END}"
    if START in txt and END in txt:
        pre = txt.split(START)[0]
        post = txt.split(END)[1]
        p.write_text(pre + block + post)
    else:
        # First run: insert a titled section after the intro (first '## ').
        idx = txt.find("\n## ")
        section = f"\n## Trust leaderboard\n\n{block}\n"
        p.write_text(txt[:idx] + section + txt[idx:])


def _interactive_svg(rows):
    """Inline SVG scatter (capability vs r), vendor-colored, with per-point
    click-to-toggle labels. Muted points keep a gray dot and drop the name."""
    W, H = 960, 580
    X0, X1, Y0, Y1 = 78, 902, 40, 502
    xmin, xmax, ymin, ymax = 10.0, 66.0, 0.05, 0.80
    def px(c): return X0 + (c - xmin) / (xmax - xmin) * (X1 - X0)
    def py(r): return Y0 + (ymax - r) / (ymax - ymin) * (Y1 - Y0)

    caps = [m["cap"] for m in rows]
    rs = [m["r"] for m in rows]
    rho = pearson(ranks(caps), ranks(rs))
    # OLS trend
    mx, my = mean(caps), mean(rs)
    b = sum((x - mx) * (y - my) for x, y in zip(caps, rs)) / \
        sum((x - mx) ** 2 for x in caps)
    a = my - b * mx
    tx0, tx1 = min(caps), max(caps)

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" '
             f'role="img" aria-label="Regression rate r vs model capability, '
             f'{len(rows)} models, rank correlation {rho:+.2f}. Click a point '
             f'to toggle its name.">']
    # gridlines + axes
    for gx in range(10, 61, 10):
        X = px(gx)
        parts.append(f'<line x1="{X:.1f}" y1="{Y0}" x2="{X:.1f}" y2="{Y1}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{X:.1f}" y="{Y1 + 22}" class="axl" '
                     f'text-anchor="middle">{gx}</text>')
    for gy in (0.2, 0.4, 0.6, 0.8):
        Y = py(gy)
        parts.append(f'<line x1="{X0}" y1="{Y:.1f}" x2="{X1}" y2="{Y:.1f}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{X0 - 12}" y="{Y + 4:.1f}" class="axl" '
                     f'text-anchor="end">{int(gy * 100)}%</text>')
    parts.append(f'<text x="{(X0 + X1) / 2:.0f}" y="{H - 8}" class="axt" '
                 f'text-anchor="middle">Model capability (baseline task '
                 f'success, %)</text>')
    parts.append(f'<text transform="translate(20 {(Y0 + Y1) / 2:.0f}) '
                 f'rotate(-90)" class="axt" text-anchor="middle">Regression '
                 f'rate r (share of false alarms that break working '
                 f'code)</text>')
    parts.append(f'<line x1="{px(tx0):.1f}" y1="{py(a + b * tx0):.1f}" '
                 f'x2="{px(tx1):.1f}" y2="{py(a + b * tx1):.1f}" '
                 f'class="trend"/>')
    parts.append(f'<text x="{X1}" y="{Y0 - 14}" class="rho" '
                 f'text-anchor="end">rank correlation '
                 f'{rho:+.2f} across {len(rows)} models</text>')

    # label anti-collision: side by x, push apart vertically per side
    mid = (X0 + X1) / 2
    pts = []
    for m in rows:
        X, Y = px(m["cap"]), py(m["r"])
        pts.append({"m": m, "X": X, "Y": Y, "left": X > mid})
    for side in (True, False):
        col = sorted([p for p in pts if p["left"] == side], key=lambda p: p["Y"])
        last = -1e9
        for p in col:
            ly = p["Y"] + 4
            if ly - last < 15:
                ly = last + 15
            p["ly"] = ly
            last = ly

    for p in pts:
        m = p["m"]
        color = VENDOR_COLOR.get(m["vendor"], FALLBACK_VENDOR_COLOR)
        left = p["left"]
        lx = p["X"] - 11 if left else p["X"] + 11
        anchor = "end" if left else "start"
        tip = (f'{m["name"]}: r={m["r"]:.2f}, tau*={m["tau"]:.2f}, '
               f'JP@0={m["cap"]:.1f}%, better fixed = {m["policy"]}')
        parts.append(
            f'<g class="pt" tabindex="0" data-name="{m["name"]}">'
            f'<title>{tip}</title>'
            f'<line class="conn" x1="{p["X"]:.1f}" y1="{p["Y"]:.1f}" '
            f'x2="{lx:.1f}" y2="{p["ly"] - 4:.1f}"/>'
            f'<circle cx="{p["X"]:.1f}" cy="{p["Y"]:.1f}" r="7" '
            f'style="fill:{color}"/>'
            f'<text class="lbl" x="{lx:.1f}" y="{p["ly"]:.1f}" '
            f'text-anchor="{anchor}">{m["name"]}</text></g>')
    parts.append('</svg>')
    return "".join(parts)


_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Trust Leaderboard — Noisy Verifier Feedback</title>
<style>
  :root {
    --ink:#17171d; --ink2:#5b5b66; --ink3:#8a8a93; --line:#e7e6e1;
    --bg:#faf9f7; --card:#fff; --accent:#215d8c; --accent2:#2f7fb5;
    --soft:#eef3f7; --naive:#e9f5ef; --sel:#fbf1e0;
  }
  * { box-sizing:border-box; margin:0; }
  html { -webkit-text-size-adjust:100%; }
  body { font:16px/1.62 ui-sans-serif,-apple-system,"Segoe UI",Roboto,
         Helvetica,Arial,sans-serif; color:var(--ink); background:var(--bg);
         padding:0 1.1rem 5rem; -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1060px; margin:0 auto; }
  a { color:var(--accent); text-decoration:none; }
  a:hover { text-decoration:underline; }
  em { font-style:italic; }

  .mast { padding:3.2rem 0 1.8rem; border-bottom:1px solid var(--line); }
  .eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.71rem;
             font-weight:700; color:var(--accent); }
  h1 { font-size:2.1rem; line-height:1.14; letter-spacing:-.022em;
       margin:.55rem 0 .75rem; max-width:24ch; font-weight:700; }
  .lede { color:var(--ink2); font-size:1.07rem; max-width:62ch; }
  .lede b { color:var(--ink); }
  .stats { display:flex; flex-wrap:wrap; gap:.35rem 1.3rem; margin-top:1.4rem;
           font-size:.83rem; color:var(--ink2); }
  .stats span b { color:var(--ink); font-variant-numeric:tabular-nums; }

  section { margin:2.4rem 0; }
  .sec-label { text-transform:uppercase; letter-spacing:.1em; font-size:.71rem;
               font-weight:700; color:var(--ink3); margin-bottom:.85rem; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:14px; box-shadow:0 1px 3px rgba(20,20,30,.04); }

  /* rates card with real stacked-fraction formulas */
  .rates { padding:.4rem 1.7rem 1.4rem; }
  .rate { padding:1.25rem 0; border-top:1px solid var(--line);
          display:grid; grid-template-columns:1fr auto; gap:.4rem 2rem;
          align-items:center; }
  .rate:first-child { border-top:none; }
  .rate h3 { font-size:1.02rem; font-weight:650; }
  .rate .tag { font-size:.7rem; font-weight:700; text-transform:uppercase;
               letter-spacing:.05em; padding:.12rem .5rem; border-radius:99px;
               margin-left:.5rem; vertical-align:.08em; }
  .rate .tag.up { background:var(--naive); color:#1c6b45; }
  .rate .tag.down { background:var(--sel); color:#9a5b09; }
  .rate .tag.key { background:var(--soft); color:var(--accent); }
  .rate p { color:var(--ink2); font-size:.93rem; margin-top:.3rem;
            max-width:52ch; }
  .eq { display:flex; align-items:center; justify-content:flex-end; gap:.5rem;
        font-size:1.16rem; color:var(--ink); white-space:nowrap; }
  .v { font-family:"Cambria Math","STIX Two Math","Times New Roman",Georgia,
       serif; font-style:italic; font-size:1.08em; }
  .frac { display:inline-flex; flex-direction:column; text-align:center;
          line-height:1.22; font-size:.82em; }
  .frac .n { border-bottom:1.6px solid currentColor; padding:0 .55em .12em; }
  .frac .d { padding:.12em .55em 0; }
  .rule { margin-top:1.1rem; padding-top:1rem; border-top:1px dashed var(--line);
          color:var(--ink2); font-size:.95rem; display:flex; gap:.6rem;
          align-items:center; flex-wrap:wrap; }
  .rule .eq { font-size:1.05rem; justify-content:flex-start; }

  .chartcard { padding:1rem 1.1rem .4rem; }
  .chart { width:100%; height:auto; display:block; touch-action:manipulation; }
  .chart .grid { stroke:#eceCE6; stroke-width:1; }
  .chart .trend { stroke:#9a998f; stroke-width:1.6; stroke-dasharray:6 5; opacity:.7; }
  .chart .axl { fill:var(--ink2); font-size:15px; }
  .chart .axt { fill:var(--ink); font-size:16px; }
  .chart .rho { fill:var(--ink2); font-size:15px; }
  .chart .pt { cursor:pointer; }
  .chart .pt circle { stroke:#fff; stroke-width:1.6; transition:fill .1s; }
  .chart .lbl { fill:var(--ink); font-size:14px; paint-order:stroke;
                stroke:#fff; stroke-width:3px; }
  .chart .conn { stroke:#bdbcb4; stroke-width:.8; opacity:0; }
  .chart .pt:hover circle { stroke:var(--ink); }
  .chart .pt.muted circle { fill:#cbcac4 !important; }
  .chart .pt.muted .lbl { display:none; }
  .chart .pt:focus { outline:none; }
  .chart .pt:focus circle { stroke:var(--accent2); stroke-width:2.4; }
  .controls { display:flex; gap:.55rem; align-items:center; flex-wrap:wrap;
              color:var(--ink2); font-size:.88rem; padding:.3rem .3rem 1rem; }
  .controls button { font:inherit; font-size:.84rem; padding:.3rem .8rem;
      border:1px solid var(--line); border-radius:99px; background:#fff;
      color:var(--ink); cursor:pointer; }
  .controls button:hover { border-color:var(--accent); color:var(--accent); }
  .controls .grow { flex:1; }
  .controls a.dl { color:var(--accent); font-weight:600; }

  .tablewrap { overflow-x:auto; border:1px solid var(--line);
               border-radius:14px; background:#fff; }
  table { border-collapse:collapse; width:100%; font-size:.9rem;
          white-space:nowrap; }
  thead th { text-align:left; font-weight:600; color:var(--ink2);
        background:#f7f6f2; border-bottom:1.5px solid #dcdbd4;
        padding:.6rem .8rem; font-size:.72rem; text-transform:uppercase;
        }
  thead th .noup { text-transform:none;
        letter-spacing:.03em; vertical-align:bottom; position:sticky; top:0; }
  tbody td { padding:.55rem .8rem; border-bottom:1px solid var(--line); }
  tbody tr:last-child td { border-bottom:none; }
  tbody tr:nth-child(even) { background:#fbfaf7; }
  tbody tr:hover { background:var(--soft); }
  td.rank { color:var(--ink3); font-variant-numeric:tabular-nums; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  td.model { font-weight:600; }
  td.vend { color:var(--ink2); }
  .dot { display:inline-block; width:.7em; height:.7em; border-radius:50%;
         margin-right:.5em; vertical-align:.02em; }
  .ci { color:var(--ink3); font-size:.85em; }
  .note { color:var(--ink3); font-weight:400; font-size:.8em; margin-left:.55em; }
  abbr.tip { text-decoration:underline dotted; cursor:help; }
  .chip { font-size:.77rem; padding:.12rem .6rem; border-radius:99px;
          border:1px solid var(--line); }
  .chip.naive { background:var(--naive); }
  .chip.selective { background:var(--sel); }

  .how { color:var(--ink2); font-size:.93rem; max-width:60ch; }
  .how b { color:var(--ink); }
  .how ul { margin:.5rem 0 .9rem 1.1rem; padding:0; }
  .how li { margin:.38rem 0; }
  .headline { border-left:3px solid var(--accent); padding:.2rem 0 .2rem 1rem;
              margin-top:1.4rem; color:var(--ink); font-size:1rem; }
  footer { margin-top:2.6rem; padding-top:1.4rem; border-top:1px solid var(--line);
           color:var(--ink3); font-size:.83rem; }
  @media (max-width:620px) {
    h1 { font-size:1.7rem; } .rate { grid-template-columns:1fr; }
    .eq { justify-content:flex-start; }
  }
</style></head><body><div class="wrap">

<header class="mast">
  <div class="eyebrow">LLM Trust Leaderboard</div>
  <h1>How much should each LLM trust its static analyzer &mdash; and how should
  you filter its findings?</h1>
  <p class="lede">AI coding agents fix static-analyzer findings in a loop, but
  most findings are false alarms (combined Semgrep+Bandit precision is
  <b>23.8%</b> on LLM-generated Python). The answer is neither to trust every
  finding nor to ignore them all, but to <b>filter per model</b> &mdash; from
  real fix-loop interactions we measure two rates and derive one threshold that
  says which findings a given model should even see.</p>
  <div class="stats">
    <span><b>__N__</b> models</span>
    <span><b>__NV__</b> developers</span>
    <span>ranked by regression rate <span class="v">r</span></span>
    <span>updated <b>__TODAY__</b></span>
  </div>
</header>

<section>
  <div class="sec-label">The three rates</div>
  <div class="card rates">
    <div class="rate">
      <div>
        <h3>Fix rate <span class="v">q</span>
          <span class="tag up">higher is better</span></h3>
        <p>Of the <em>real</em> findings surfaced to the model, the fraction it
        correctly fixes &mdash; secure <em>and</em> tests still pass.</p>
      </div>
      <div class="eq"><span class="v">q</span> =
        <span class="frac"><span class="n">true findings the model fixed</span>
        <span class="d">true findings shown</span></span></div>
    </div>
    <div class="rate">
      <div>
        <h3>Regression rate <span class="v">r</span>
          <span class="tag key">the ranking key</span>
          <span class="tag down">lower is better</span></h3>
        <p>Of the <em>false</em> findings surfaced, the fraction where the
        model&rsquo;s &ldquo;fix&rdquo; breaks working code. A low-<span
        class="v">r</span> model recognizes false alarms instead of obeying
        them.</p>
      </div>
      <div class="eq"><span class="v">r</span> =
        <span class="frac"><span class="n">working code the &ldquo;fix&rdquo; broke</span>
        <span class="d">false alarms shown</span></span></div>
    </div>
    <div class="rate">
      <div>
        <h3>Surfacing threshold <span class="v">&tau;</span><sup>*</sup></h3>
        <p>Feed the model a finding only if its rule&rsquo;s historical
        precision <span class="v">p</span> exceeds <span class="v">&tau;</span><sup>*</sup>;
        filter everything below. Low <span class="v">r</span> &rArr; low
        threshold &rArr; surface almost everything; high <span class="v">r</span>
        &rArr; filter aggressively.</p>
      </div>
      <div class="eq"><span class="v">&tau;</span><sup>*</sup> =
        <span class="frac"><span class="n"><span class="v">r</span></span>
        <span class="d"><span class="v">q</span> + <span class="v">r</span></span></span></div>
    </div>
    <div class="rule">
      <span>Surfacing rule:</span>
      <span class="eq">surface a finding &nbsp;&hArr;&nbsp;
        <span class="v">p</span> &gt; <span class="v">&tau;</span><sup>*</sup></span>
    </div>
  </div>
</section>

<section>
  <div class="sec-label">Better models are less gullible</div>
  <div class="card chartcard">__CHART__</div>
  <div class="controls">
    <span><b>Tip:</b> click (or tab + Enter) any point to hide its name and
      gray its dot &mdash; declutter to see the trend.</span>
    <span class="grow"></span>
    <button id="showAll" type="button">Show all names</button>
    <button id="hideAll" type="button">Hide all names</button>
    <a class="dl" href="r_vs_capability.png" download>Download original figure (PNG)</a>
  </div>
</section>

<section>
  <div class="sec-label">The leaderboard &mdash; ranked by regression rate <span class="v">r</span> (lower is better)</div>
  <div class="tablewrap"><table>
  <thead><tr><th>#</th><th>Model</th><th>Developer</th>
  <th class="num">Params</th>
  <th class="num">Regression rate r<br>(95% CI)</th>
  <th class="num">Fix rate q</th>
  <th class="num">Surfacing<br>threshold <span class="noup v">&tau;</span><span class="noup">*</span></th>
  <th>Better fixed policy</th>
  <th class="num">Baseline capability<br>(JointPass@0)</th>
  <th class="num">HumanEval<br>pass@1</th>
  <th class="num">FP / TP<br>trials</th></tr></thead>
  <tbody>__ROWS__</tbody></table></div>
</section>

<section class="how">
  <div class="sec-label">More on the columns</div>
  <ul>
  <li><b>Params</b> &mdash; total parameters where the developer discloses them (<em>MoE</em> = mixture-of-experts, where far fewer are active per token); &ldquo;&mdash;&rdquo; = undisclosed, as for most frontier API models. <b>Note: capability, not size, predicts <span class="v">r</span></b> &mdash; the x-axis (JointPass@0) and HumanEval are the capability measures, and several small models resist false feedback better than larger ones.</li>
  <li><b>Better fixed policy</b> &mdash; the better of the two deployed fixed
    policies, <em>naive</em> (surface every finding) vs <em>selective</em>
    (only rules above 50% precision). &ldquo;Naive&rdquo; does not mean
    surfacing everything is optimal &mdash; the optimum filters at the
    model&rsquo;s own <span class="v">&tau;</span><sup>*</sup>, which is above
    zero for every model.</li>
  <li><b>Baseline capability (JointPass@0)</b> &mdash; fraction of items whose
    pre-feedback code is both functionally correct and vulnerability-free, on
    the 51-item core benchmark (combined Semgrep+Bandit analyzer, multi-seed).</li>
  <li><b>HumanEval pass@1</b> &mdash; external functional-capability axis
    (disjoint tasks, no security signal).</li>
  <li><b>in-sample</b> (Notes) &mdash; one of the five models used to formulate
    the r-law; all others were measured prospectively.</li>
  </ul>
  <p class="headline">The headline: <span class="v">r</span> falls steeply with
  capability (Spearman &minus;0.90) while <span class="v">q</span> stays flat
  &mdash; <b>better models are less gullible</b> &mdash; so the right feedback
  policy is a property of the model and expires with every model update. One
  frontier model (Claude Fable&nbsp;5) could not be measured at all: its safety
  layer refused 9/10 benchmark requests.</p>
</section>

<footer>Updated __TODAY__ &middot;
<a href="https://github.com/drchangliu/NoisyVerifierFeedback">code, raw traces
&amp; methodology</a> &middot; paper: <em>Better Models Are Less Gullible:
Selective Feedback for LLM Code Agents under Noisy Static Analysis</em> (under
review at Empirical Software Engineering; submitted state:
<a href="https://github.com/drchangliu/NoisyVerifierFeedback/releases/tag/emse-2026-07">tag
emse-2026-07</a>) &middot; Chang Liu, Ohio University</footer>

<script>
(function() {
  var pts = document.querySelectorAll('.chart .pt');
  function toggle(g) { g.classList.toggle('muted'); }
  pts.forEach(function(g) {
    g.addEventListener('click', function() { toggle(g); });
    g.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(g); }
    });
  });
  document.getElementById('showAll').onclick = function() {
    pts.forEach(function(g) { g.classList.remove('muted'); }); };
  document.getElementById('hideAll').onclick = function() {
    pts.forEach(function(g) { g.classList.add('muted'); }); };
})();
</script>
</div></body></html>
"""


def html_page(rows, today):
    n = len(rows)
    nv = len({m["vendor"] for m in rows})
    chart = _interactive_svg(rows)

    trs = []
    for i, m in enumerate(rows, 1):
        dot = (f'<span class="dot" '
               f'style="background:{VENDOR_COLOR.get(m["vendor"], FALLBACK_VENDOR_COLOR)}"></span>')
        pol = f'<span class="chip {m["policy"]}">{m["policy"]}</span>'
        he = "&mdash;" if m["he"] is None else f"{m['he']:.1f}%"
        note = notes(m).replace(
            "in-sample",
            '<abbr class="tip" title="One of the five models used to '
            'formulate the r-law. All other models were measured '
            'prospectively, after the law and decision rule were frozen '
            '&mdash; so they are genuine out-of-sample tests.">in-sample</abbr>')
        trs.append(
            f"<tr><td class='rank'>{i}</td>"
            f"<td class='model'>{dot}<span>{m['name']}</span>"
            + (f"<span class='note'>{note}</span>" if note else "") + "</td>"
            f"<td class='vend'>{m['vendor']}</td>"
            f"<td class='num params'>{m['params']}</td>"
            f"<td class='num'><b>{m['r']:.2f}</b> <span class='ci'>"
            f"[{m['r_lo']:.2f}, {m['r_hi']:.2f}]</span></td>"
            f"<td class='num'>{m['q']:.2f}</td>"
            f"<td class='num'>{m['tau']:.2f}</td>"
            f"<td>{pol}</td>"
            f"<td class='num'>{m['cap']:.1f}%</td>"
            f"<td class='num'>{he}</td>"
            f"<td class='num'>{m['fpt']}/{m['tpt']}</td></tr>")

    html = _TEMPLATE
    for k, v in {"__N__": str(n), "__NV__": str(nv), "__TODAY__": today,
                 "__CHART__": chart, "__ROWS__": "".join(trs)}.items():
        html = html.replace(k, v)
    return html


def main() -> None:
    today = datetime.date.today().isoformat()
    rows = collect()
    (ROOT / "LEADERBOARD.md").write_text(md_full(rows, today))
    inject_readme(md_compact(rows, today))
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(html_page(rows, today))
    src = ROOT / "figures" / "leaderboard_r_vs_capability.png"
    if not src.exists():
        src = ROOT / "figures" / "r_vs_capability.png"
    shutil.copy(src, docs / "r_vs_capability.png")
    print(f"wrote LEADERBOARD.md, README block, docs/index.html "
          f"({len(rows)} models, figure: {src.name})")


if __name__ == "__main__":
    main()
