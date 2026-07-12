#!/usr/bin/env bash
# Crash-resume for cloud_cohorts.sh (interrupted 2026-06-15 ~16:50).
#
# State at resume (complete 51-line runs counted; truncated dirs already removed):
#   glm_46_cloud            naive 6/6, selective 6/6   -> DONE, skip
#   qwen3_coder_480b_cloud  naive 3/6, selective 3/6   -> need +3 naive, +3 selective
#   deepseek_v4_flash_cloud naive 0/6, selective 0/6   -> need 6 naive, 6 selective
#
# Targets match the original driver (6 seeds each). Cloud models verified
# reachable on the localhost:11435 Ollama Cloud proxy before launch.
set -u
mkdir -p logs
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PROJ_ROOT}/.venv/bin/python"
[ -x "$PYBIN" ] || { echo "FATAL: ${PYBIN} not found." >&2; exit 3; }
export PATH="${PROJ_ROOT}/.venv/bin:${PATH}"
"$PYBIN" "${PROJ_ROOT}/scripts/analyzer_sanity.py" --quiet || {
  echo "FATAL: analyzer_sanity.py failed." >&2; exit 4; }

GLOG=logs/cloud_cohorts.log

run_n () {  # cfg n tag
  local cfg="$1"; local n="$2"; local tag="$3"
  for i in $(seq 1 "$n"); do
    echo "[$(date -Is)] RUN $cfg ($i/$n)" | tee -a "logs/cloud_${tag}.log"
    "$PYBIN" scripts/run_experiment.py --config "$cfg" >>"logs/cloud_${tag}.log" 2>&1 \
      || echo "[$(date -Is)] FAILED $cfg (continuing)" | tee -a "logs/cloud_${tag}.log"
  done
}

echo "[$(date -Is)] CLOUD RESUME START" | tee -a "$GLOG"

echo "[$(date -Is)] === RESUME qwen3_coder_480b_cloud (+3 naive, +3 selective) ===" | tee -a "$GLOG"
run_n configs/prospective/naive_qwen3_coder_480b_cloud.yaml     3 qwen3_coder_480b_cloud
run_n configs/prospective/selective_qwen3_coder_480b_cloud.yaml 3 qwen3_coder_480b_cloud
echo "[$(date -Is)] === qwen3_coder_480b_cloud COMPLETE ===" | tee -a "$GLOG"

echo "[$(date -Is)] === COHORT deepseek_v4_flash_cloud (6 seeds x naive+selective) ===" | tee -a "$GLOG"
run_n configs/prospective/naive_deepseek_v4_flash_cloud.yaml     6 deepseek_v4_flash_cloud
run_n configs/prospective/selective_deepseek_v4_flash_cloud.yaml 6 deepseek_v4_flash_cloud
echo "[$(date -Is)] === COHORT deepseek_v4_flash_cloud DONE ===" | tee -a "$GLOG"

echo "[$(date -Is)] ALL CLOUD COHORTS COMPLETE (resume)" | tee -a "$GLOG"
