#!/usr/bin/env bash
# Prospective out-of-sample cohorts for the tau* prediction law.
#
# Runs NEW local model snapshots (not in the original 9-cohort table)
# through naive + selective(tau=0.5) on the 51-item core, so each model
# yields an out-of-sample test point: estimate (q,r) from its naive runs,
# compute tau* = r/(q+r), predict the naive-vs-selective winner, verify.
#
# Spans 4 families (Llama, DeepSeek, Qwen, Gemma) plus Qwen
# generation (3 -> 3.5 -> 3.6) and size (14b/32b) sweeps.
#
# Self-sequencing: waits for any in-flight tau-sweep on the shared Ollama
# endpoint (11435) to finish before starting, so it will not evict the
# running Gemma sweep's model from VRAM.
#
# Usage: scripts/prospective_cohorts.sh   (designed for run_in_background)
set -u
mkdir -p logs

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PROJ_ROOT}/.venv/bin/python"
[ -x "$PYBIN" ] || { echo "FATAL: ${PYBIN} not found." >&2; exit 3; }
export PATH="${PROJ_ROOT}/.venv/bin:${PATH}"

"$PYBIN" "${PROJ_ROOT}/scripts/analyzer_sanity.py" --quiet || {
  echo "FATAL: analyzer_sanity.py failed." >&2; exit 4; }

GLOG=logs/prospective_cohorts.log

# 1) Wait for any running tau-sweep to clear the shared GPU endpoint.
while pgrep -f 'selective_combined51_(gemma4|qwen3)_tau' >/dev/null 2>&1; do
  echo "[$(date -Is)] waiting for in-flight tau-sweep to finish..." >>"$GLOG"
  sleep 180
done
echo "[$(date -Is)] GPU clear; starting prospective cohorts" | tee -a "$GLOG"

# Small -> large so fast cohorts bank first and can be validated early.
MODELS=(llama31_8b qwen3_14b deepseek-v2_16b gemma3_27b qwen35_27b qwen36_27b qwen3_32b)
SEEDS=4

run_one () {
  local cfg="$1"; local tag="$2"
  echo "[$(date -Is)] RUN $cfg" | tee -a "logs/prospective_${tag}.log"
  "$PYBIN" scripts/run_experiment.py --config "$cfg" >>"logs/prospective_${tag}.log" 2>&1 \
    || echo "[$(date -Is)] FAILED $cfg (continuing)" | tee -a "logs/prospective_${tag}.log"
}

for m in "${MODELS[@]}"; do
  echo "[$(date -Is)] === COHORT ${m} (${SEEDS} seeds x naive+selective) ===" | tee -a "$GLOG"
  for seed in $(seq 1 "$SEEDS"); do
    run_one "configs/prospective/naive_${m}.yaml"     "$m"
    run_one "configs/prospective/selective_${m}.yaml" "$m"
  done
  echo "[$(date -Is)] === COHORT ${m} DONE ===" | tee -a "$GLOG"
done
echo "[$(date -Is)] ALL PROSPECTIVE COHORTS COMPLETE" | tee -a "$GLOG"
