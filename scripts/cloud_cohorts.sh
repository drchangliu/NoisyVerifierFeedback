#!/usr/bin/env bash
# Cloud-hosted cohorts (Ollama Cloud, flat-rate) for the r-vs-capability law.
#
# Adds high-capability / new-lineage points that don't fit on the local GPU:
#   glm-4.6:cloud            -- NEW lineage (Zhipu GLM)
#   qwen3-coder:480b-cloud   -- 480B code model, extends the high-cap end
#   deepseek-v4-flash:cloud  -- newer/bigger DeepSeek than the local v2:16b
#
# naive + selective(tau=0.5), 6 seeds each on the 51-item core. 6 (not 4)
# because high-capability models trigger few findings, so r's FP-trial
# denominator is small -- extra seeds keep r credible at the high end.
#
# Cloud runs are remote (no local VRAM), so this does NOT contend with the
# local GPU boosts; it can run concurrently. GLM first (new lineage banks
# first); seeds are the outer loop so partial completion stays balanced.
#
# Usage: scripts/cloud_cohorts.sh   (designed for run_in_background)
set -u
mkdir -p logs
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PROJ_ROOT}/.venv/bin/python"
[ -x "$PYBIN" ] || { echo "FATAL: ${PYBIN} not found." >&2; exit 3; }
export PATH="${PROJ_ROOT}/.venv/bin:${PATH}"
"$PYBIN" "${PROJ_ROOT}/scripts/analyzer_sanity.py" --quiet || {
  echo "FATAL: analyzer_sanity.py failed." >&2; exit 4; }

GLOG=logs/cloud_cohorts.log
MODELS=(glm_46_cloud qwen3_coder_480b_cloud deepseek_v4_flash_cloud)
SEEDS=6

run_one () {
  local cfg="$1"; local tag="$2"
  echo "[$(date -Is)] RUN $cfg" | tee -a "logs/cloud_${tag}.log"
  "$PYBIN" scripts/run_experiment.py --config "$cfg" >>"logs/cloud_${tag}.log" 2>&1 \
    || echo "[$(date -Is)] FAILED $cfg (continuing)" | tee -a "logs/cloud_${tag}.log"
}

echo "[$(date -Is)] cloud cohorts START" | tee -a "$GLOG"
for m in "${MODELS[@]}"; do
  echo "[$(date -Is)] === COHORT ${m} (${SEEDS} seeds x naive+selective) ===" | tee -a "$GLOG"
  for seed in $(seq 1 "$SEEDS"); do
    run_one "configs/prospective/naive_${m}.yaml"     "$m"
    run_one "configs/prospective/selective_${m}.yaml" "$m"
  done
  echo "[$(date -Is)] === COHORT ${m} DONE ===" | tee -a "$GLOG"
done
echo "[$(date -Is)] ALL CLOUD COHORTS COMPLETE" | tee -a "$GLOG"
