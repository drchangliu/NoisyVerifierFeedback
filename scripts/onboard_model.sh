#!/usr/bin/env bash
# Onboard a new model into the benchmark: generate configs, run the cohort,
# and print the remaining steps (HumanEval + leaderboard refresh).
#
# Prereq: an entry for the model tag in configs/model_registry.json
# (cohort label, display name, vendor, added date, ollama_base_url if
# applicable, max_tokens). No code edits are needed anywhere else.
#
# Usage:
#   scripts/onboard_model.sh <model-tag> <slug> [runs-per-policy=4]
#   e.g. scripts/onboard_model.sh kimi-k2.7-code:cloud kimi27 4
#
# After completion:
#   python scripts/run_humaneval.py --models <humaneval_label>
#   python scripts/r_capability.py --collapse-eras          # verify numbers
#   python scripts/make_leaderboard.py                      # refresh leaderboard
set -u
mkdir -p logs

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PROJ_ROOT}/.venv/bin/python"
[ -x "$PYBIN" ] || { echo "FATAL: ${PYBIN} not found." >&2; exit 3; }
export PATH="${PROJ_ROOT}/.venv/bin:${PATH}"

TAG="${1:?model tag, e.g. kimi-k2.7-code:cloud}"
SLUG="${2:?short slug for config filenames, e.g. kimi27}"
N="${3:-4}"

# Generate configs from the registry entry (idempotent).
"$PYBIN" - "$TAG" "$SLUG" <<'PYEOF'
import json, sys
from pathlib import Path
tag, slug = sys.argv[1], sys.argv[2]
root = Path(__file__).resolve()  # unused; cwd is project root via PATH setup
reg = json.loads(Path("configs/model_registry.json").read_text())
e = reg.get(tag) or sys.exit(f"FATAL: {tag} not in configs/model_registry.json")
base_url = e.get("ollama_base_url")
max_tokens = e.get("max_tokens", 4096)
for cond in ("naive", "selective"):
    p = Path(f"configs/{cond}_combined51_{slug}.yaml")
    if p.exists():
        print(f"exists: {p}"); continue
    lines = [
        'benchmark:', '  name: "combined"', '  calibration_fraction: 0.0',
        'analyzer:', '  name: "combined"', '  timeout: 30',
        'feedback:', f'  condition: "{cond}"',
        'agent:', f'  model: "{tag}"', f'  max_tokens: {max_tokens}',
    ]
    if base_url:
        lines.append(f'  ollama_base_url: "{base_url}"')
    lines += ['selective:', '  threshold_tau: 0.5',
              'llm_judge:', '  model: "claude-haiku-4-5-20251001"']
    p.write_text("\n".join(lines) + "\n")
    print(f"wrote: {p}")
PYEOF
[ $? -eq 0 ] || exit 1

# Analyzer preflight (same guard as week1_batch.sh).
"$PYBIN" "${PROJ_ROOT}/scripts/analyzer_sanity.py" --quiet || {
  echo "FATAL: analyzer_sanity.py failed." >&2; exit 4; }

run_n () {
  local cfg="$1"; local n="$2"
  local log="logs/onboard_$(basename "${cfg%.yaml}").log"
  echo "[$(date -Is)] BEGIN $cfg x $n" | tee -a "$log"
  for i in $(seq 1 "$n"); do
    echo "[$(date -Is)] -- run $i/$n --" | tee -a "$log"
    "$PYBIN" scripts/run_experiment.py --config "$cfg" >>"$log" 2>&1 \
      || echo "[$(date -Is)] run $i FAILED for $cfg (continuing)" | tee -a "$log"
  done
  echo "[$(date -Is)] END $cfg" | tee -a "$log"
}

run_n "configs/naive_combined51_${SLUG}.yaml"     "$N"
run_n "configs/selective_combined51_${SLUG}.yaml" "$N"
echo "[$(date -Is)] COHORT DONE for $TAG"
echo "Next: python scripts/run_humaneval.py --models \$(python -c \"import json;print(json.load(open('configs/model_registry.json'))['$TAG']['humaneval_label'])\")"
echo "Then: python scripts/make_leaderboard.py"
