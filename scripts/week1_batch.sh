#!/usr/bin/env bash
# Week-1 batch driver: run the additional seeds needed to bring each
# (model, policy) cell on the 51-item core to 16 runs.
#
# Usage:
#   scripts/week1_batch.sh anthropic   # Haiku + Sonnet (~$10, ~4-5h)
#   scripts/week1_batch.sh gemma       # Gemma4-31B (~60h GPU)
#   scripts/week1_batch.sh qwen        # Qwen3-8B (~13h GPU)
#   scripts/week1_batch.sh sonnet46     # Sonnet 4.6 4 fixed policies (~$15, ~6h)
#   scripts/week1_batch.sh sonnet46-cot # Sonnet 4.6 CoT-LLM-Judge add-on (~$10, ~4h)
#
# Each run appends a header to logs/week1_<bucket>.log so progress is
# tail-able. Designed to be invoked under nohup or Bash run_in_background.
set -u
mkdir -p logs

# Resolve project root from this script's location so we can find .venv even
# if the script is launched from somewhere else (e.g., nohup driver). The repo
# stores deps in .venv at the project root and the original interactive
# invocations relied on the user having activated it.
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PROJ_ROOT}/.venv/bin/python"
if [ ! -x "$PYBIN" ]; then
  echo "FATAL: ${PYBIN} not found. Activate or rebuild the venv first." >&2
  exit 3
fi

# Make .venv/bin/{semgrep,bandit} visible to subprocess.run inside the analyzer
# wrappers; otherwise the loop completes silently with 0 findings per item and
# burns API credits on what looks like a successful run. (Diagnosed 2026-05-30
# after an 80-run Sonnet 4.6 batch returned all-1-iter results.)
export PATH="${PROJ_ROOT}/.venv/bin:${PATH}"

# Pre-flight: confirm the analyzer pipeline finds vulnerabilities in canonical
# snippets before spending any API budget. Exits 0 on success, 1 if any snippet
# returns 0 findings (pipeline regression), 2 if a binary is missing.
"$PYBIN" "${PROJ_ROOT}/scripts/analyzer_sanity.py" --quiet || {
  echo "FATAL: analyzer_sanity.py failed. Run it without --quiet for details." >&2
  exit 4
}

run_n () {
  local cfg="$1"; local n="$2"
  local log="logs/week1_$(basename "${cfg%.yaml}").log"
  echo "[$(date -Is)] BEGIN $cfg x $n" | tee -a "$log"
  for i in $(seq 1 "$n"); do
    echo "[$(date -Is)] -- run $i/$n --" | tee -a "$log"
    "$PYBIN" scripts/run_experiment.py --config "$cfg" >>"$log" 2>&1 \
      || echo "[$(date -Is)] run $i FAILED for $cfg (continuing)" | tee -a "$log"
  done
  echo "[$(date -Is)] END $cfg" | tee -a "$log"
}

bucket="${1:-}"

case "$bucket" in
  anthropic)
    # Haiku: need +12 naive, +8 selective, +8 llm_judge
    run_n configs/naive_combined51_haiku.yaml     12
    run_n configs/selective_combined51_haiku.yaml  8
    run_n configs/llm_judge_combined51_haiku.yaml  8
    # Sonnet: need +12 of each
    run_n configs/naive_combined51_sonnet.yaml    12
    run_n configs/selective_combined51_sonnet.yaml 12
    run_n configs/llm_judge_combined51_sonnet.yaml 12
    ;;
  gemma)
    # Gemma: need +11 naive, +12 selective, +12 llm_judge
    run_n configs/naive_combined51_gemma4.yaml    11
    run_n configs/selective_combined51_gemma4.yaml 12
    run_n configs/llm_judge_combined51_gemma4.yaml 12
    ;;
  qwen)
    # Qwen3-8B: need +12 naive, +8 selective, +12 llm_judge
    run_n configs/naive_combined51_qwen3.yaml     12
    run_n configs/selective_combined51_qwen3.yaml  8
    run_n configs/llm_judge_combined51_qwen3.yaml 12
    ;;
  sonnet46)
    # Sonnet 4.6 four-fixed-policy re-run for M3 (version-gap check).
    # Target n=20 per cell to match the F2 polish effort. ~$15 total at sticker.
    # 1 sanity-check seed of naive already verified API + pricing.
    run_n configs/naive_combined51_sonnet46.yaml     20
    run_n configs/selective_combined51_sonnet46.yaml 20
    run_n configs/llm_judge_combined51_sonnet46.yaml 20
    run_n configs/adaptive_combined51_sonnet46.yaml  20
    ;;
  sonnet46-cot)
    # Sonnet 4.6 CoT-LLM-Judge add-on for R5 (cross-model CoT coverage).
    # Higher per-run cost (CoT reasoning tokens); ~$10 total at sticker.
    run_n configs/llm_judge_cot_combined51_sonnet46.yaml 20
    ;;
  *)
    echo "Usage: $0 {anthropic|gemma|qwen|sonnet46|sonnet46-cot}" >&2
    exit 2
    ;;
esac
