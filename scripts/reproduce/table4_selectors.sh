#!/usr/bin/env bash
# Table 4 -- selector x budget at T=5: random / topconf / kmeans / farthest, k = 10, 50, 100.
cd "$(dirname "$0")/../.."
source scripts/reproduce/_common.sh
for sel in random topconf kmeans farthest; do
  for k in 10 50 100; do
    run_config "configs/experiments/selectors/${sel}_k${k}.yaml" "$@"
  done
done
"$PYTHON" scripts/evaluation/aggregate_results.py --runs-root outputs/runs --output results/table4_selectors.csv \
  --experiments $(for s in random topconf kmeans farthest; do for k in 10 50 100; do echo greenfin_${s}_k${k}; done; done)
