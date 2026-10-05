# Results

Small, human-readable result tables are written here by the reproduction
scripts. Large artifacts (checkpoints, per-day predictions, raw CodeCarbon
logs) stay in `outputs/`, which is not tracked by git.

| File | Produced by | Paper |
|---|---|---|
| `table3_architecture.csv` | `scripts/reproduce/table3_architecture.sh` | Table 3 |
| `table4_selectors.csv` | `scripts/reproduce/table4_selectors.sh` | Table 4 |
| `table6_lookback.csv` | `scripts/reproduce/table6_lookback.sh` | Table 6 |
| `significance/bootstrap_results.csv` | `scripts/reproduce/table7_bootstrap.sh` | Table 7 |
| `trading/trading_summary.csv`, `trading/trading_per_window_seed.csv`, `trading/ar_validation_metrics.csv` | `scripts/reproduce/table8_trading.sh` | Table 8 |
| `carbon/upstream_sentiment.json`, `carbon/upstream_embeddings.json` | `scripts/data/build_sentiment.py`, `scripts/data/build_embeddings.py` | upstream encoding cost |

Columns of the `table*.csv` files (from `scripts/evaluation/aggregate_results.py`):

* `DA_*`, `PnL_*`, `SR_*`: mean and standard deviation across seeds of the
  per-seed mean over the 5 test windows;
* `train_time_min`, `energy_kWh`, `co2_kg`: downstream training cost of one
  seed (sum over the 5 windows), averaged over seeds; `*_all_seeds` are summed
  over all seeds;
* `inference_s_per_1000`: downstream inference seconds per 1,000 predictions;
* `end_to_end_*` (only with `--upstream-carbon`): downstream cost plus the
  common upstream RoBERTa + FinBERT encoding cost.

No result files from the original research runs are included: those runs
printed their metrics to the console without saving them (see
`docs/REPRODUCIBILITY_AUDIT.md`, section 7).
