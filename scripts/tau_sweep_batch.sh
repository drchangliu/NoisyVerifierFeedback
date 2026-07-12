#!/usr/bin/env bash
# Tau-sweep + Haiku-v2 batch driver (post tau*-prediction analysis).
#
# Usage:
#   scripts/tau_sweep_batch.sh local     # Qwen3-8B tau sweep then Gemma4-31B tau sweep (~33h GPU)
#   scripts/tau_sweep_batch.sh haiku-v2  # Haiku naive/selective/llm_judge x 8 seeds (~$4, ~5h)
#
# Sweeps run 4 seeds x tau in {0.1,0.3,0.7,0.9} on the 51-item core.
# tau=0.5 and naive (tau=0) cells already exist at n>=8 in the v2 cohorts.
# Seeds are the OUTER loop so an interrupted sweep leaves equal seed counts
# across tau values.
set -u
mkdir -p logs

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PROJ_ROOT}/.venv/bin/python"
if [ ! -x "$PYBIN" ]; then
  echo "FATAL: ${PYBIN} not found. Activate or rebuild the venv first." >&2
  exit 3
fi

# Same guard as week1_batch.sh: make .venv/bin/{semgrep,bandit} visible to the
# analyzer wrappers, then pre-flight the pipeline before spending compute.
export PATH="${PROJ_ROOT}/.venv/bin:${PATH}"
"$PYBIN" "${PROJ_ROOT}/scripts/analyzer_sanity.py" --quiet || {
  echo "FATAL: analyzer_sanity.py failed. Run it without --quiet for details." >&2
  exit 4
}

run_one () {
  local cfg="$1"; local tag="$2"
  local log="logs/tausweep_${tag}.log"
  echo "[$(date -Is)] RUN $cfg" | tee -a "$log"
  "$PYBIN" scripts/run_experiment.py --config "$cfg" >>"$log" 2>&1 \
    || echo "[$(date -Is)] FAILED $cfg (continuing)" | tee -a "$log"
}

sweep_model () {
  local model="$1"; local n_seeds="$2"
  for seed in $(seq 1 "$n_seeds"); do
    echo "[$(date -Is)] === ${model} sweep: seed round ${seed}/${n_seeds} ===" \
      | tee -a "logs/tausweep_${model}.log"
    for tag in tau01 tau03 tau07 tau09; do
      run_one "configs/ablations/selective_combined51_${model}_${tag}.yaml" "$model"
    done
  done
}

bucket="${1:-}"

case "$bucket" in
  local)
    sweep_model qwen3 4
    sweep_model gemma4 4
    ;;
  haiku-v2)
    for seed in $(seq 1 8); do
      echo "[$(date -Is)] === haiku-v2: seed round ${seed}/8 ===" \
        | tee -a logs/tausweep_haiku_v2.log
      run_one configs/naive_combined51_haiku.yaml     haiku_v2
      run_one configs/selective_combined51_haiku.yaml haiku_v2
      run_one configs/llm_judge_combined51_haiku.yaml haiku_v2
    done
    ;;
  *)
    echo "Usage: $0 {local|haiku-v2}" >&2
    exit 2
    ;;
esac
echo "[$(date -Is)] BUCKET ${bucket} COMPLETE" | tee -a "logs/tausweep_${bucket}.log"
