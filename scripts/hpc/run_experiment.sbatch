#!/usr/bin/env bash
# Optional generic SLURM template for one experiment. Adapt partition/resources to your cluster.
# Submit from the repository root; SLURM needs the log directory to exist beforehand:
#
#   mkdir -p outputs/logs
#   sbatch --export=ALL,CONFIG=configs/experiments/selectors/farthest_k10.yaml scripts/hpc/run_experiment.sbatch
#
# Environment variables:
#   CONFIG      experiment YAML (required)
#   CONDA_ENV   optional conda environment to activate
#   EXTRA_ARGS  optional extra arguments for run_experiment.py (e.g. "--seeds 0")
#
# Memory note: the full embedding file is loaded into host RAM (~22 GB in fp16)
# and Input_t20.jsonl is ~16 GB on disk; request enough memory for both.
#SBATCH --job-name=greenfin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
set -euo pipefail
: "${CONFIG:?Set CONFIG to an experiment YAML, e.g. --export=ALL,CONFIG=configs/...yaml}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p outputs/logs
if [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
python scripts/experiments/run_experiment.py --experiment-config "$CONFIG" ${EXTRA_ARGS:-}
