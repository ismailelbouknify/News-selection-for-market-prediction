#!/usr/bin/env bash
# Table 7 -- paired block bootstrap (10,000 replicates, block length 5, per-window resampling).
# Needs the predictions.csv files of the runs listed in configs/evaluation/bootstrap.yaml
# (produced by table3_architecture.sh and table4_selectors.sh). CPU only.
cd "$(dirname "$0")/../.."
source scripts/reproduce/_common.sh
"$PYTHON" scripts/evaluation/bootstrap_significance.py --config configs/evaluation/bootstrap.yaml "$@"
