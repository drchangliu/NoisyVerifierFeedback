#!/usr/bin/env bash
# Frontier-cohort batch driver: Opus 4.8 and Fable 5 on the 51-item core,
# 4 naive + 4 selective runs each -- new out-of-sample points for the
# capability->r law at the high-capability end (July 2026).
#
# Usage:
#   scripts/frontier_cohorts.sh opus48    # Opus 4.8 only (~$4)
#   scripts/frontier_cohorts.sh fable5    # Fable 5 only (~$15-25, thinking billed)
#   scripts/frontier_cohorts.sh all       # both, Opus first
#
# Notes:
#  - Both models reject temperature; run-to-run variation comes from
#    nondeterministic serving (see config comments).
#  - Fable 5 may refuse security-adjacent items (stop_reason "refusal" in the
#    trace); the loop keeps the previous code on an empty response, and
#    refusal counts must be reported alongside q/r for this cohort.
#  - NEVER add a fallback model: mixing models within a cohort invalidates q/r.
set -u
mkdir -p logs

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PROJ_ROOT}/.venv/bin/python"
if [ ! -x "$PYBIN" ]; then
  echo "FATAL: ${PYBIN} not found. Activate or rebuild the venv first." >&2
  exit 3
fi

# Make .venv/bin/{semgrep,bandit} visible to the analyzer subprocesses
# (same failure mode as week1_batch.sh, diagnosed 2026-05-30).
export PATH="${PROJ_ROOT}/.venv/bin:${PATH}"

"$PYBIN" "${PROJ_ROOT}/scripts/analyzer_sanity.py" --quiet || {
  echo "FATAL: analyzer_sanity.py failed. Run it without --quiet for details." >&2
  exit 4
}

run_n () {
  local cfg="$1"; local n="$2"
  local log="logs/frontier_$(basename "${cfg%.yaml}").log"
  echo "[$(date -Is)] BEGIN $cfg x $n" | tee -a "$log"
  for i in $(seq 1 "$n"); do
    echo "[$(date -Is)] -- run $i/$n --" | tee -a "$log"
    "$PYBIN" scripts/run_experiment.py --config "$cfg" >>"$log" 2>&1 \
      || echo "[$(date -Is)] run $i FAILED for $cfg (continuing)" | tee -a "$log"
  done
  echo "[$(date -Is)] END $cfg" | tee -a "$log"
}

bucket="${1:-all}"

case "$bucket" in
  opus48)
    run_n configs/naive_combined51_opus48.yaml     4
    run_n configs/selective_combined51_opus48.yaml 4
    ;;
  fable5)
    run_n configs/naive_combined51_fable5.yaml     4
    run_n configs/selective_combined51_fable5.yaml 4
    ;;
  all)
    run_n configs/naive_combined51_opus48.yaml     4
    run_n configs/selective_combined51_opus48.yaml 4
    run_n configs/naive_combined51_fable5.yaml     4
    run_n configs/selective_combined51_fable5.yaml 4
    ;;
  *)
    echo "usage: $0 {opus48|fable5|all}" >&2
    exit 2
    ;;
esac
echo "[$(date -Is)] ALL DONE ($bucket)"
