#!/usr/bin/env bash
# Optional: submit one SLURM job per experiment config of a paper table.
#   bash scripts/hpc/submit_grid.sh architecture   # Table 3
#   bash scripts/hpc/submit_grid.sh selectors      # Table 4
#   bash scripts/hpc/submit_grid.sh lookback       # Table 6
# Adapt the #SBATCH resources in scripts/hpc/run_experiment.sbatch to your cluster first.
set -euo pipefail
cd "$(dirname "$0")/../.."
group="${1:?usage: submit_grid.sh <architecture|selectors|lookback>}"
case "$group" in
  architecture)  # Table 3 also contains GreenFin-Selected (farthest, k=10), which lives in selectors/
    configs=(configs/experiments/architecture/*.yaml configs/experiments/selectors/farthest_k10.yaml) ;;
  selectors|lookback)
    configs=(configs/experiments/"$group"/*.yaml) ;;
  *)
    echo "unknown group '$group'; expected architecture, selectors or lookback" >&2; exit 1 ;;
esac
mkdir -p outputs/logs  # SLURM does not create the log directory itself
for cfg in "${configs[@]}"; do
  sbatch --job-name="gf_$(basename "$cfg" .yaml)" --export=ALL,CONFIG="$cfg" scripts/hpc/run_experiment.sbatch
done
