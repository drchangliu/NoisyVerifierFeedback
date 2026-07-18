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
from r_capability import accumulate_qr, MIN_TRIALS
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
    L = []
    L.append("# Trust Leaderboard: which models can safely consume unfiltered "
             "static-analyzer feedback?\n")
    L.append(f"*Updated {today} · {len(rows)} models · ranked by the "
             "**counterfactual regression rate r** (lower = better): the "
             "probability that the model, handed a **false-positive** security "
             "finding on its own working code, \"fixes\" it and breaks the "
             "code.*\n")
    L.append("![r falls with capability](docs/r_vs_capability.png)\n")
    L.append("| # | Model | Developer | r (95% CI) | q | τ\\* | Best fixed "
             "policy | JP@0 | HumanEval | FP/TP trials | Notes |")
    L.append("|--:|---|---|---|--:|--:|---|--:|--:|--:|---|")
    for i, m in enumerate(rows, 1):
        he = "—" if m["he"] is None else f"{m['he']:.1f}%"
        L.append(
            f"| {i} | **{m['name']}** | {m['vendor']} "
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
- **Best fixed policy** — the better of the two deployed fixed policies:
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

Across these models, r falls steeply as capability rises (Spearman −0.89) —
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
    return "\n".join(L)


def md_compact(rows, today):
    L = []
    L.append(f"*Ranked by the regression rate **r** — the probability the "
             f"model breaks working code when handed a false-alarm security "
             f"finding (lower = harder to mislead). Updated {today}. Full "
             f"table, CIs, and methodology: [LEADERBOARD.md](LEADERBOARD.md) "
             f"· [web version](https://drchangliu.github.io/NoisyVerifierFeedback/).*\n")
    L.append("![Regression rate r falls as model capability rises "
             "(16 models, rank correlation -0.89)](docs/r_vs_capability.png)\n")
    L.append("| # | Model | r | τ\\* | Best fixed policy | HumanEval |")
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


def html_page(rows, today):
    trs = []
    for i, m in enumerate(rows, 1):
        dot = (f'<span class="dot" '
               f'style="background:{VENDOR_COLOR.get(m["vendor"], FALLBACK_VENDOR_COLOR)}"></span>')
        pol = (f'<span class="chip {m["policy"]}">{m["policy"]}</span>')
        he = "&mdash;" if m["he"] is None else f"{m['he']:.1f}%"
        note = notes(m)
        trs.append(
            f"<tr><td class='rank'>{i}</td>"
            f"<td class='model'>{dot}{m['name']}"
            + (f"<span class='note'>{note}</span>" if note else "") + "</td>"
            f"<td>{m['vendor']}</td>"
            f"<td class='num'><b>{m['r']:.2f}</b> <span class='ci'>"
            f"[{m['r_lo']:.2f}, {m['r_hi']:.2f}]</span></td>"
            f"<td class='num'>{m['q']:.2f}</td>"
            f"<td class='num'>{m['tau']:.2f}</td>"
            f"<td>{pol}</td>"
            f"<td class='num'>{m['cap']:.1f}%</td>"
            f"<td class='num'>{he}</td>"
            f"<td class='num'>{m['fpt']}/{m['tpt']}</td></tr>")
    n = len(rows)
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Trust Leaderboard &mdash; Noisy Verifier Feedback</title>
<style>
  :root {{ --ink:#1a1a19; --ink2:#5f5e56; --line:#e4e3dd; --bg:#fdfdfc; }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ font:16px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,
         sans-serif; color:var(--ink); background:var(--bg);
         padding:2.5rem 1rem 4rem; }}
  main {{ max-width:1000px; margin:0 auto; }}
  h1 {{ font-size:1.7rem; line-height:1.25; letter-spacing:-0.01em; }}
  .sub {{ color:var(--ink2); margin:0.6rem 0 1.6rem; max-width:46em; }}
  .figure {{ margin:1.6rem 0; }}
  .figure img {{ max-width:100%; height:auto; border:1px solid var(--line);
                 border-radius:8px; }}
  .tablewrap {{ overflow-x:auto; margin:1.6rem 0 0.8rem; }}
  table {{ border-collapse:collapse; width:100%; font-size:0.92rem;
           white-space:nowrap; }}
  th {{ text-align:left; font-weight:600; color:var(--ink2);
        border-bottom:2px solid var(--ink); padding:0.45rem 0.7rem;
        font-size:0.8rem; text-transform:uppercase; letter-spacing:0.04em; }}
  td {{ padding:0.5rem 0.7rem; border-bottom:1px solid var(--line); }}
  td.rank {{ color:var(--ink2); }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.model {{ font-weight:600; }}
  .dot {{ display:inline-block; width:0.65em; height:0.65em;
          border-radius:50%; margin-right:0.5em; }}
  .ci {{ color:var(--ink2); font-size:0.85em; }}
  .note {{ color:var(--ink2); font-weight:400; font-size:0.82em;
           margin-left:0.6em; }}
  .chip {{ font-size:0.78rem; padding:0.1rem 0.55rem; border-radius:99px;
           border:1px solid var(--line); }}
  .chip.naive {{ background:#eef7f2; }}
  .chip.selective {{ background:#fdf3e2; }}
  .how {{ color:var(--ink2); font-size:0.92rem; max-width:52em;
          margin-top:1.4rem; }}
  .how b {{ color:var(--ink); }}
  .how p {{ margin:0.5rem 0; }}
  a {{ color:#2a78d6; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  footer {{ margin-top:2rem; color:var(--ink2); font-size:0.85rem; }}
</style></head><body><main>
<h1>Which LLMs can you trust with unfiltered static-analyzer feedback?</h1>
<p class="sub">A live trust leaderboard of {n} models, ranked by the
<b>counterfactual regression rate r</b>: the probability that the model,
handed a <em>false-alarm</em> security finding about its own working code,
&ldquo;fixes&rdquo; the non-problem and breaks the code. Lower is better
&mdash; a low-r model recognizes false alarms instead of obeying them.</p>

<div class="figure"><img src="r_vs_capability.png"
  alt="Scatter plot: regression rate r falls as model capability rises
       across {n} models (rank correlation -0.89)"></div>

<div class="tablewrap"><table>
<thead><tr><th>#</th><th>Model</th><th>Developer</th>
<th class="num">r (95% CI)</th><th class="num">q</th>
<th class="num">&tau;*</th><th>Best fixed policy</th>
<th class="num">JP@0</th><th class="num">HumanEval</th>
<th class="num">FP/TP trials</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>

<div class="how">
<p><b>r</b> &mdash; share of false-positive findings whose &ldquo;fix&rdquo;
broke working code. <b>q</b> &mdash; share of true positives actually fixed
(secure <em>and</em> tests still pass). <b>&tau;* = r/(q+r)</b> &mdash; feed
the model a finding only if the rule&rsquo;s historical precision exceeds
this threshold. <b>Best fixed policy</b> &mdash; the better of the two
deployed fixed policies, <em>naive</em> (surface everything) vs
<em>selective</em> (only rules with &gt;50% precision); &ldquo;naive&rdquo;
does not mean surfacing everything is optimal &mdash; the optimum filters at
the model&rsquo;s own &tau;*. <b>JP@0</b> &mdash; baseline JointPass
(functionally correct <em>and</em> vulnerability-free) on the 51-item core
benchmark. Combined Semgrep+Bandit analyzer, multi-seed; <em>in-sample</em>
marks the five models used to formulate the r-law, all others measured
prospectively.</p>
<p>The headline: r falls steeply with capability (Spearman &minus;0.89) while
q stays flat &mdash; <b>better models are less gullible</b> &mdash; so the
right feedback policy is a property of the model and expires with every model
update. One frontier model (Claude Fable&nbsp;5) could not be measured at
all: its safety layer refused 9/10 benchmark requests.</p>
</div>

<footer>Updated {today} &middot;
<a href="https://github.com/drchangliu/NoisyVerifierFeedback">code, raw
traces &amp; methodology</a> &middot; paper: <em>Better Models Are Less Gullible:
Selective Feedback for LLM Code Agents under Noisy Static Analysis</em> (under review at Empirical Software
Engineering; submitted state:
<a href="https://github.com/drchangliu/NoisyVerifierFeedback/releases/tag/emse-2026-07">
tag emse-2026-07</a>) &middot; Chang Liu, Ohio University</footer>
</main></body></html>
"""


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
