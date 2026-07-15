#!/usr/bin/env python3
"""Per-source construct-validity check (paper Sec. 'SecurityEval Oracle Quality').

Splits within-run Delta-JP under the naive policy by benchmark source
(CWEval professional oracles vs hand-authored SecurityEval oracles) for the
four-model panel on the 51-item core.

Inclusion rules (the paper's own):
  - naive condition, natural_language format only (format-ablation runs use
    the same condition with different formats and are excluded);
  - 51-item core runs (45..51 items present; missing items = missing cells);
  - runs where the analyzer returned zero findings on every item are excluded
    (the mechanically broken 2026-05-28 batch documented in R2).

Delta-JP is within-run (JP@final - JP@0 per run, pooled per (item, seed)),
so pooling across collection batches is valid under the batch-drift rule.

Usage: python scripts/per_source_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from nvf.benchmark.loader import load_benchmark  # noqa: E402

MODELS = {
    "claude-haiku-4-5-20251001": "Haiku",
    "claude-sonnet-4-20250514": "Sonnet",
    "gemma4:31b": "Gemma4-31B",
    "qwen3:8b": "Qwen3-8B",
}


def jp(it) -> bool:
    return bool(it.get("tests_passed")) and not bool(it.get("has_vulnerability"))


def main() -> None:
    src_of = {it.item_id: it.source for it in load_benchmark("combined")}
    n_items = {"cweval": sum(1 for s in src_of.values() if s == "cweval"),
               "securityeval": sum(1 for s in src_of.values() if s == "securityeval")}

    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # model->src->[n,jp0,jpF]
    runs = defaultdict(int)
    dropped_broken = defaultdict(int)
    for d in sorted((ROOT / "data" / "results").iterdir()):
        cfgf, trf = d / "config.yaml", d / "traces.jsonl"
        if not (cfgf.exists() and trf.exists()):
            continue
        cfg = yaml.safe_load(cfgf.read_text())
        m = cfg.get("agent", {}).get("model")
        if m not in MODELS:
            continue
        if cfg.get("feedback", {}).get("condition") != "naive":
            continue
        if cfg.get("feedback", {}).get("format", "natural_language") != "natural_language":
            continue
        traces = [json.loads(l) for l in trf.read_text().splitlines() if l.strip()]
        traces = [t for t in traces if t.get("iterations")]
        if not (45 <= len(traces) <= 51):
            continue
        if all(t["iterations"][0].get("n_findings", 0) == 0 for t in traces):
            dropped_broken[MODELS[m]] += 1  # mechanically broken batch (R2)
            continue
        runs[MODELS[m]] += 1
        for t in traces:
            s = src_of.get(t["item_id"])
            if s is None:
                continue
            a = agg[MODELS[m]][s]
            a[0] += 1
            a[1] += jp(t["iterations"][0])
            a[2] += jp(t["iterations"][-1])

    print(f"{'model':<12}{'runs':>5}{'src':>14}{'pairs':>7}{'max':>6}"
          f"{'JP@0':>8}{'JP@f':>8}{'dJP':>8}")
    for mdl in ["Haiku", "Sonnet", "Gemma4-31B", "Qwen3-8B"]:
        pooled = [0, 0, 0]
        for s in ["cweval", "securityeval"]:
            n, j0, jf = agg[mdl][s]
            for i, v in enumerate((n, j0, jf)):
                pooled[i] += v
            print(f"{mdl:<12}{runs[mdl]:>5}{s:>14}{n:>7}{n_items[s]*runs[mdl]:>6}"
                  f"{100*j0/n:>7.1f}%{100*jf/n:>7.1f}%{100*(jf-j0)/n:>+7.1f}pp")
        n, j0, jf = pooled
        print(f"{mdl:<12}{runs[mdl]:>5}{'pooled':>14}{n:>7}{51*runs[mdl]:>6}"
              f"{100*j0/n:>7.1f}%{100*jf/n:>7.1f}%{100*(jf-j0)/n:>+7.1f}pp")
        if dropped_broken[mdl]:
            print(f"{'':<12}{'':>5}  (excluded {dropped_broken[mdl]} zero-finding broken runs)")


if __name__ == "__main__":
    main()
