#!/usr/bin/env bash
# End-to-end CPU smoke test of every entry point on tiny synthetic data (~5 minutes).
# Does not download anything and writes only under outputs/smoke/.
cd "$(dirname "$0")/../.."
source scripts/reproduce/_common.sh
"$PYTHON" scripts/data/make_synthetic_data.py --root outputs/smoke
for cfg in \
  configs/experiments/architecture/finin_full.yaml \
  configs/experiments/architecture/greenfin_market_only.yaml \
  configs/experiments/selectors/farthest_k10.yaml \
  configs/experiments/selectors/topconf_k10.yaml \
  configs/experiments/selectors/kmeans_k10.yaml \
  configs/experiments/selectors/random_k10.yaml \
  configs/experiments/lookback/farthest_T20_k50.yaml; do
  run_config "$cfg" --base-config configs/smoke/base.yaml --data-dir outputs/smoke
done
"$PYTHON" scripts/evaluation/aggregate_results.py --runs-root outputs/smoke/runs --output outputs/smoke/results/summary.csv
"$PYTHON" scripts/evaluation/practical_trading.py --config configs/smoke/trading.yaml
"$PYTHON" scripts/evaluation/bootstrap_significance.py --config configs/smoke/bootstrap.yaml
echo "Smoke test finished successfully."
