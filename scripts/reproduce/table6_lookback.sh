#!/usr/bin/env bash
# Table 6 -- look-back x budget: farthest selector, T = 1, 3, 5, 10, 20 and k = 10, 50, 100.
# (T=5 cells are configured identically to configs/experiments/selectors/farthest_k*.yaml.)
# With W=2770 and step=340, T=1..10 give 5 CV windows; T=20 (~4,120 samples) gives 4.
cd "$(dirname "$0")/../.."
source scripts/reproduce/_common.sh
for T in 1 3 5 10 20; do
  for k in 10 50 100; do
    run_config "configs/experiments/lookback/farthest_T${T}_k${k}.yaml" "$@"
  done
done
"$PYTHON" scripts/evaluation/aggregate_results.py --runs-root outputs/runs --output results/table6_lookback.csv \
  --experiments $(for T in 1 3 5 10 20; do for k in 10 50 100; do echo greenfin_farthest_T${T}_k${k}; done; done)
