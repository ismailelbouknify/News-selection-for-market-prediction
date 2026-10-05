#!/usr/bin/env bash
# Table 3 -- architecture ablation at T=5:
# FININ-Full, FININ-Selected (farthest k=10/50/100), GreenFin-Full,
# GreenFin-Selected (farthest k=10) and GreenFin-MarketOnly. 5 seeds x 5 windows each.
# Requires a GPU; FININ-Full is by far the most expensive run.
cd "$(dirname "$0")/../.."
source scripts/reproduce/_common.sh
for cfg in \
  configs/experiments/architecture/finin_full.yaml \
  configs/experiments/architecture/finin_selected_farthest_k10.yaml \
  configs/experiments/architecture/finin_selected_farthest_k50.yaml \
  configs/experiments/architecture/finin_selected_farthest_k100.yaml \
  configs/experiments/architecture/greenfin_full.yaml \
  configs/experiments/selectors/farthest_k10.yaml \
  configs/experiments/architecture/greenfin_market_only.yaml; do
  run_config "$cfg" "$@"
done
"$PYTHON" scripts/evaluation/aggregate_results.py --runs-root outputs/runs --output results/table3_architecture.csv \
  --experiments finin_full finin_selected_farthest_k10 finin_selected_farthest_k50 finin_selected_farthest_k100 greenfin_full greenfin_farthest_k10 greenfin_market_only
