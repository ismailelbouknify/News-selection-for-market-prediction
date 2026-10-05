# Shared helpers for the reproduction scripts. Source from the repository root.
set -euo pipefail
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

run_config() {  # run_config <experiment-config> [extra args...]
  local cfg="$1"; shift
  echo ">>> $cfg"
  "$PYTHON" scripts/experiments/run_experiment.py --experiment-config "$cfg" "$@"
}
