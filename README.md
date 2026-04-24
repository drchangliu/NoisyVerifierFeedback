# Selective Feedback for Noisy Verifiers in LLM Code Agents

Code and data for the paper:

> **Selective Feedback for Noisy Verifiers in LLM Code Agents: When Static Analyzers Mislead Language Models**
>
> Chang Liu et al. — *Submitted to Empirical Software Engineering (EMSE), 2026*

## Overview

LLM code agents use static analyzers (Semgrep, Bandit) in feedback loops to fix security vulnerabilities. But these analyzers are **noisy**: combined precision is only 23.8%. Surfacing false positives causes regressions — the LLM "fixes" nonexistent issues and breaks working code.

We formalize this as the **noisy-verifier problem** and study four feedback policies:

| Policy | Description |
|--------|-------------|
| **Naive** | Surface all findings (status quo) |
| **Selective** | Filter by per-rule precision threshold |
| **LLM-Judge** | Cheap model triages findings |
| **Adaptive** | Online Bayesian learning of optimal threshold |

Key finding: naive feedback **harms** mid-tier models (Qwen3-8B: -2.5pp) while the adaptive policy **outperforms all fixed policies** (+4.1pp for Haiku, +11.6pp for Qwen3-8B).

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install pycryptodome cryptography lxml defusedxml pyjwt flask

# Configure API key (for Haiku/Sonnet experiments)
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run calibration (no API key needed, ~15 min)
python scripts/run_calibration.py --benchmark both --use-full-dataset --analyzer combined

# Run a small live test (requires API key)
python scripts/live_test.py --model claude-haiku-4-5-20251001 --items 3

# Run full experiment (51 items, ~$0.05 per run)
python scripts/run_experiment.py --config configs/naive_combined51_haiku.yaml
```

## Reproducing Key Results

### Core experiments (51 items, multi-seed)

```bash
# Naive baseline
python scripts/run_experiment.py --config configs/naive_combined51_haiku.yaml

# Selective (tau=0.5)
python scripts/run_experiment.py --config configs/selective_combined51_haiku.yaml

# Adaptive threshold
python scripts/run_experiment.py --config configs/adaptive_combined51_haiku.yaml

# LLM Judge
python scripts/run_experiment.py --config configs/llm_judge_combined51_haiku.yaml
```

For Qwen3-8B (free, requires [Ollama](https://ollama.ai)):
```bash
ollama pull qwen3:8b
python scripts/run_experiment.py --config configs/naive_combined51_qwen3.yaml
python scripts/run_experiment.py --config configs/selective_combined51_qwen3.yaml
python scripts/run_experiment.py --config configs/adaptive_combined51_qwen3.yaml
```

### Multi-seed runs

```bash
python scripts/run_multi_seed.py --config configs/naive_combined51_haiku.yaml --seeds 8
```

### Threshold sweep

```bash
for tau in 0.1 0.3 0.5 0.7 0.9; do
  python scripts/run_experiment.py \
    --config configs/selective_combined51_haiku_tau${tau}.yaml
done
```

### Analyze results and generate figures

```bash
python scripts/analyze_results.py data/results/*
python scripts/generate_figures.py data/results/* --output-dir figures/
python scripts/plot_adaptive_empirical.py
```

## Project Structure

```
NoisyVerifierFeedback/
├── src/nvf/                    # Core library
│   ├── agents/                 # Feedback policies (naive, selective, llm_judge, adaptive)
│   ├── analyzers/              # Semgrep + Bandit wrappers
│   ├── benchmark/              # CWEval, SecurityEval, SecCodePLT loaders
│   ├── calibration/            # Per-rule precision estimation
│   ├── execution/              # Test runner + sandbox
│   ├── feedback/               # Finding filter + formatter
│   ├── llm/                    # LLM client + cost tracker
│   ├── metrics/                # JointPass@k computation
│   └── theory/                 # Optimal threshold + adaptive policy
├── configs/                    # Experiment configurations (YAML)
├── scripts/                    # Experiment runners + analysis
├── data/
│   ├── calibration/            # Pre-computed precision maps
│   └── results/                # Experiment traces (168 runs)
├── figures/                    # Publication figures
└── tests/                      # Unit tests
```

## Pre-computed Results

The `data/results/` directory contains traces from all 168 experiment runs reported in the paper. Each run directory contains:
- `config.yaml` — resolved experiment configuration
- `traces.jsonl` — per-item iteration traces with findings, code, and test results

The `data/calibration/` directory contains:
- `precision_map_combined.json` — per-rule precision estimates (75 rules)
- `precision_map.json` — Semgrep-only precision (41 rules)
- `labeled_findings.jsonl` — ground-truth labels for 146 calibration items

## Requirements

- Python 3.11+
- Semgrep (`pip install semgrep`)
- Bandit (`pip install bandit`)
- For local models: [Ollama](https://ollama.ai) with 8+ GB VRAM (Qwen3-8B) or 24+ GB (Gemma4-31B)
- For API models: Anthropic API key (~$0.05 per 51-item run with Haiku)

## Cost

A complete reproduction of core experiments (51 items, 4 policies, 2 API models, 4 seeds) costs **~$3** in API fees. Local model experiments are free. The full extended benchmark (136 items) costs ~$5 additional.

## Citation

```bibtex
@article{liu2026noisy,
  title={Selective Feedback for Noisy Verifiers in {LLM} Code Agents: When Static Analyzers Mislead Language Models},
  author={Liu, Chang},
  journal={Empirical Software Engineering},
  year={2026},
  note={Under review}
}
```

## License

MIT
