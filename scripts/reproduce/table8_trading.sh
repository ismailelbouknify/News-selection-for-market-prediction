#!/usr/bin/env bash
# Table 8 -- practical trading evaluation (T=5): buy-and-hold / Always-Up, previous-day,
# validation-selected AR, GreenFin k=10 selectors (long/short) and farthest k=10 (long/cash),
# transaction costs 0/5/10/20 bps. Needs the predictions.csv of the k=10 selector runs. CPU only.
cd "$(dirname "$0")/../.."
source scripts/reproduce/_common.sh
"$PYTHON" scripts/evaluation/practical_trading.py --config configs/evaluation/trading.yaml "$@"
