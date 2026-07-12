# Replication Package: Selective Feedback for Noisy Verifiers in LLM Code Agents

Replication package for the manuscript *"Selective Feedback for Noisy Verifiers
in LLM Code Agents: When Static Analyzers Mislead Language Models"* (Chang Liu,
Ohio University; under review at *Empirical Software Engineering*, submitted
July 2026).

The paper derives the optimal precision threshold τ\* = r/(q+r) for surfacing
noisy static-analyzer findings to an LLM code agent, and shows empirically —
across 16 distinct models from six developers — that the counterfactual
regression rate *r* falls with model capability while the fix rate *q* stays
flat, so fixed feedback policies do not transfer across model snapshots.

## Contents

| Path | Contents |
|---|---|
| `src/nvf/` | The feedback-loop framework: agents (naive/selective/LLM-judge/adaptive), analyzer integration (Semgrep + Bandit), calibration, unified LLM client |
| `scripts/` | Every experiment and analysis script; each result in the paper maps to one script (table below) |
| `configs/` | Exact run configurations for all reported experiments |
| `data/calibration/` | Per-rule precision calibration data (151 findings, 75 rules) |
| `data/results/` | Raw traces for all 798 experiment runs (`traces.jsonl` + `config.yaml` per run) — the paper's primary data |
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

Two of the 15 evaluated models were retired by their providers during the
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
  title  = {Selective Feedback for Noisy Verifiers in LLM Code Agents:
            When Static Analyzers Mislead Language Models},
  note   = {Under review at Empirical Software Engineering},
  year   = {2026}
}
```
