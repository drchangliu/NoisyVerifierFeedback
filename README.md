# Selective Feedback for Noisy Verifiers in LLM Code Agents

Ongoing home of the noisy-verifier feedback project (Chang Liu, Ohio
University): framework, benchmarks, raw traces, and analysis for studying how
LLM code agents should consume noisy static-analyzer findings. The project
derives the optimal precision threshold τ\* = r/(q+r) for surfacing findings
to an agent, and shows empirically — currently across 16 distinct models from
six developers — that the counterfactual regression rate *r* falls with model
capability while the fix rate *q* stays flat, so fixed feedback policies do
not transfer across model snapshots.

This repository is **live**: numbers are refreshed as new model cohorts are
measured and the benchmark expands over time. The first full study is described
in the manuscript *"Better Models Are Less Gullible: Selective Feedback for LLM Code
Agents under Noisy Static Analysis"* (under review at *Empirical
Software Engineering*, submitted July 2026); the exact repository state that
paper reports on is preserved as the git tag
[`emse-2026-07`](../../releases/tag/emse-2026-07), so the paper's numbers stay
reproducible while the main branch moves on.

## Trust leaderboard

<!-- LEADERBOARD:START -->
*Ranked by the regression rate **r** — the probability the model breaks working code when handed a false-alarm security finding (lower = harder to mislead). Updated 2026-07-20. Full table, CIs, and methodology: [LEADERBOARD.md](LEADERBOARD.md) · [web version](https://drchangliu.github.io/NoisyVerifierFeedback/).*

![Regression rate r falls as model capability rises (16 models, rank correlation -0.89)](docs/r_vs_capability.png)

| # | Model | r | τ\* | Better fixed policy | HumanEval |
|--:|---|--:|--:|---|--:|
| 1 | Kimi-K2.7-code | 0.13 | 0.27 | naive | 98.8% |
| 2 | Nemotron-3-Super | 0.15 | 0.29 | naive | 97.0% |
| 3 | GLM-5.2 | 0.15 | 0.32 | naive | 98.8% |
| 4 | Claude Opus 4.8 | 0.16 | 0.36 | naive | 99.4% |
| 5 | Qwen3.5-27B | 0.17 | 0.33 | naive | 96.3% |
| 6 | Claude Sonnet 4.6 | 0.19 | 0.30 | naive | 98.8% |
| 7 | Gemma4-31B | 0.19 | 0.33 | naive | 100.0% |
| 8 | GPT-OSS-120B | 0.21 | 0.29 | naive | 97.6% |
| 9 | Claude Haiku 4.5 | 0.22 | 0.40 | naive | 96.3% |
| 10 | Qwen3-coder-480B | 0.28 | 0.40 | naive | 96.3% |
| 11 | DeepSeek-V4-flash | 0.28 | 0.31 | naive | 97.0% |
| 12 | Qwen3-32B | 0.29 | 0.32 | naive | 97.6% |
| 13 | GLM-4.6 | 0.29 | 0.36 | naive | — |
| 14 | MiniMax-M2.7 | 0.35 | 0.40 | naive | 98.8% |
| 15 | Llama3.1-8B | 0.37 | 0.47 | naive | 59.1% |
| 16 | Claude Sonnet 4 | 0.41 | 0.64 | selective | — |
| 17 | Qwen3-14B | 0.43 | 0.43 | naive | 96.3% |
| 18 | Qwen3.6-27B | 0.48 | 0.49 | naive | 98.8% |
| 19 | Qwen3-8B | 0.57 | 0.56 | selective | 93.9% |
| 20 | DeepSeek-V2-16B | 0.57 | 0.62 | selective | 42.1% |
| 21 | Gemma3-27B | 0.72 | 0.67 | selective | 87.8% |
<!-- LEADERBOARD:END -->

## Contents

| Path | Contents |
|---|---|
| `src/nvf/` | The feedback-loop framework: agents (naive/selective/LLM-judge/adaptive), analyzer integration (Semgrep + Bandit), calibration, unified LLM client |
| `scripts/` | Every experiment and analysis script; each result in the paper maps to one script (table below) |
| `configs/` | Exact run configurations for all reported experiments |
| `data/calibration/` | Per-rule precision calibration data (151 findings, 75 rules) |
| `data/results/` | Raw traces for every experiment run to date (798 as of July 2026; `traces.jsonl` + `config.yaml` per run) — the project's primary data |
| `data/humaneval/` | HumanEval pass@1 records for the external capability axis (per-model JSONL) |
| `data/smoke_tests/` | Pilot traces, including the Claude Fable 5 refusal probe cited in the paper (9/10 generations refused) |
| `data/raw/` | Benchmark items (CWEval, SecurityEval incl. our hand-authored oracles, SecCodePLT) |
| `figures/` | All paper figures as generated |
| `tests/` | Unit and integration tests (`make test`, `make test-integration`) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install pycryptodome cryptography lxml defusedxml pyjwt flask  # CWEval test deps
cp .env.example .env   # add ANTHROPIC_API_KEY only if re-running API models
```

Analysis scripts below run **entirely from the released traces** — no API key
or GPU needed to reproduce every number and figure in the paper.

## Reproducing the paper's results

Every number and figure in the EMSE manuscript regenerates from the released
traces with the commands below. To match the paper exactly, check out the
submission tag first (`git checkout emse-2026-07`); on the main branch the
same commands reflect the latest data.

| Paper result | Command |
|---|---|
| R3/R4/R5 — the r-law, q-flatness, proxy robustness, Table 7 (q, r, τ\*) | `python scripts/r_capability.py --collapse-eras` (add `--figure figures/r_vs_capability.png`) |
| R5 external axis — HumanEval vs r (Fig. 5) | `python scripts/humaneval_correlation.py --figure figures/r_vs_humaneval.png` |
| R6/R7 — τ\* sign prediction, regret, baselines, bootstrap CIs (Tables 5–6, Fig. 6) | `python scripts/tau_star_prediction.py --collapse-eras --naive-only` |
| R6 sensitivity (dropping the Sonnet pair) | `python scripts/tau_star_prediction.py --collapse-eras --naive-only --drop sonnet4,sonnet46` |
| Estimator comparison (pooled vs naive-only; Sec. 6.1) | run the above with and without `--naive-only` |
| Per-policy tables, CIs (Tables 4, 8–10) | `python scripts/aggregate_seeds.py`, `python scripts/aggregate_v2.py` |
| Significance tests (Table 11) | `python scripts/compute_significance.py` |
| q/r canonical labeling (Sec. 6.1) | `python scripts/compute_qr.py` |
| Per-rule precision calibration (Sec. 4.5, Appendix) | `python scripts/run_calibration.py --benchmark both --use-full-dataset` |
| Held-out calibration (Sec. 8.2, Appendix) | `python scripts/run_heldout_calibration.py`, `python scripts/compare_calibrations.py` |
| Adaptive-policy convergence (Fig. 8) | `python scripts/plot_adaptive_empirical.py` |
| HumanEval measurement (re-collection; needs models) | `python scripts/run_humaneval.py` |
| New feedback-loop runs (needs models) | `python scripts/run_experiment.py --config configs/<name>.yaml` |

Batch-drift note (Sec. 6.2 of the paper): all headline policy gaps are
computed from within-run ΔJP; when comparing across runs, match collection
batches or difference against each run's iteration-0 baseline.

### Model availability caveat

Two of the 16 evaluated models were retired by their providers during the
study (Claude Sonnet 4 on the Anthropic API; GLM-4.6 on Ollama Cloud), so
their cells cannot be re-collected — analysis of their released traces remains
fully reproducible. This is, itself, an instance of the paper's premise.

## License

Code and original data: MIT (see `LICENSE`). Redistributed benchmark items
under `data/raw/` remain under their upstream licenses (CWEval, SecurityEval,
SecCodePLT).

## Citation

```bibtex
@article{liu2026noisyverifier,
  author = {Liu, Chang},
  title  = {Better Models Are Less Gullible: Selective Feedback for
            LLM Code Agents under Noisy Static Analysis},
  note   = {Under review at Empirical Software Engineering},
  year   = {2026}
}
```
