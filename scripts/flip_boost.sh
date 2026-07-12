#!/usr/bin/env bash
# Seed-boost to firm up the noisy legs of candidate policy-winner flips.
#
#   scripts/flip_boost.sh sonnet  # Sonnet 4 selective leg (API, ~$1, ~1h)
#   scripts/flip_boost.sh qwen    # Qwen3-14b/32b legs (GPU, free, self-sequenced)
#
# Sonnet 4 (claude-sonnet-4-20250514) is on borrowed time (its sibling
# Opus 4 hits EOL 2026-06-15), so the selective leg -- only n=5 usable
# after the broken May-28 batch is excluded -- is grabbed first.
set -u
mkdir -p logs

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PROJ_ROOT}/.venv/bin/python"
[ -x "$PYBIN" ] || { echo "FATAL: ${PYBIN} not found." >&2; exit 3; }
export PATH="${PROJ_ROOT}/.venv/bin:${PATH}"
"$PYBIN" "${PROJ_ROOT}/scripts/analyzer_sanity.py" --quiet || {
  echo "FATAL: analyzer_sanity.py failed." >&2; exit 4; }

run_n () {
  local cfg="$1"; local n="$2"; local tag="$3"
  for i in $(seq 1 "$n"); do
    echo "[$(date -Is)] RUN $cfg ($i/$n)" | tee -a "logs/flipboost_${tag}.log"
    "$PYBIN" scripts/run_experiment.py --config "$cfg" >>"logs/flipboost_${tag}.log" 2>&1 \
      || echo "[$(date -Is)] FAILED $cfg" | tee -a "logs/flipboost_${tag}.log"
  done
}

case "${1:-}" in
  sonnet)
    # Weak leg is selective (n=5 usable); naive is already n=16 usable.
    run_n configs/selective_combined51_sonnet.yaml 12 sonnet4
    echo "[$(date -Is)] sonnet boost DONE" | tee -a logs/flipboost_sonnet4.log
    ;;
  qwen)
    # Wait for the prospective batch to release the shared GPU endpoint.
    while pgrep -f 'prospective_cohorts.sh' >/dev/null 2>&1; do
      echo "[$(date -Is)] waiting for prospective batch to finish..." >>logs/flipboost_qwen.log
      sleep 180
    done
    echo "[$(date -Is)] GPU clear; boosting Qwen flip legs" | tee -a logs/flipboost_qwen.log
    # Bring each leg to ~12 seeds (current: 14b 4/4, 32b 4/3).
    run_n configs/prospective/naive_qwen3_14b.yaml     8 qwen
    run_n configs/prospective/selective_qwen3_14b.yaml 8 qwen
    run_n configs/prospective/naive_qwen3_32b.yaml     8 qwen
    run_n configs/prospective/selective_qwen3_32b.yaml 9 qwen
    echo "[$(date -Is)] qwen boost DONE" | tee -a logs/flipboost_qwen.log
    ;;
  *)
    echo "Usage: $0 {sonnet|qwen}" >&2; exit 2 ;;
esac
