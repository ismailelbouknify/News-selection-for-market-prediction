#!/usr/bin/env python
"""Practical trading evaluation (paper Table 8).

Evaluates, on the identical out-of-sample test windows used for training
(see ``greenfin.windows``):

* buy-and-hold and Always-Up (identical positions; Always-Up also has a
  directional accuracy);
* previous-day direction (sign of the last realised return);
* AR(p) return forecasts, ``p`` chosen on the validation splits only;
* every model listed in the config, from the ``predictions.csv`` written by
  ``scripts/experiments/run_experiment.py``, as a long/short or long/cash
  strategy.

For every (strategy, window, seed) and every transaction-cost level the
script computes gross/net compounded return, annualised return and
volatility, Sharpe ratio (daily excess returns, rf = 2%/yr by default),
maximum drawdown, turnover and directional accuracy; results are then
averaged over windows and seeds.

Usage::

    python scripts/evaluation/practical_trading.py --config configs/evaluation/trading.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from greenfin.baselines import fit_ar, predict_ar, select_ar_lag
from greenfin.metrics import get_returns_map
from greenfin.trading import (
    aggregate_runs,
    compute_trading_metrics,
    positions_buy_and_hold,
    positions_from_forecasts,
    positions_long_cash,
    positions_long_short,
    positions_previous_day_direction,
    strategy_returns_long_cash,
    strategy_returns_long_short,
)
from greenfin.windows import fold_date_splits

STRATEGIES = {"long_short", "long_cash"}


def load_dates_and_labels(path: str) -> Tuple[List[str], Dict[str, int]]:
    """Sorted sample dates and ``{date: label}`` from a dataset JSONL (streams the file)."""
    labels: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            labels[rec["date"]] = int(rec["label"])
    return sorted(labels), labels


def _rows_for(
    name: str,
    window: int,
    seed: int,
    positions: np.ndarray,
    gross: np.ndarray,
    rf: np.ndarray,
    bps_grid: List[float],
    actual=None,
    predicted=None,
) -> List[Dict]:
    return [
        {"strategy": name, "window": window, "seed": seed,
         **compute_trading_metrics(positions, gross, rf, bps, actual, predicted)}
        for bps in bps_grid
    ]


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/evaluation/trading.yaml")
    args = p.parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg.get("output_dir", "results/trading"))
    out_dir.mkdir(parents=True, exist_ok=True)
    bps_grid = [float(b) for b in cfg.get("transaction_costs_bps", [0, 5, 10, 20])]
    tdays = int(cfg.get("trading_days_per_year", 252))
    rf_daily = float(cfg.get("risk_free_annual", 0.02)) / tdays
    cv = cfg.get("cv", {})

    dates, label_map = load_dates_and_labels(cfg["data"]["input_jsonl"])
    returns_map = get_returns_map(cfg["data"]["market_csv"])
    folds = fold_date_splits(dates, int(cv.get("train_days", 2430)), int(cv.get("step_days", 340)), cv.get("max_windows"))
    print(f"{len(dates)} samples, {len(folds)} windows, cost grid {bps_grid} bps")

    rows: List[Dict] = []
    ar_validation: List[Dict] = []
    ar_test: Dict[int, List[Dict]] = {}

    for fold in folds:
        w = fold["window"]
        test_dates = fold["test_dates"]
        r_test = np.array([returns_map[d] for d in test_dates], dtype=np.float64)
        if not np.all(np.isfinite(r_test)):
            raise ValueError(f"Missing next-day returns in test window {w}.")
        y_test = np.array([label_map[d] for d in test_dates], dtype=np.float64)
        rf = np.full(len(test_dates), rf_daily)

        bh = positions_buy_and_hold(len(test_dates))
        rows += _rows_for("Buy-and-hold", w, 0, bh, strategy_returns_long_short(bh, r_test), rf, bps_grid)
        rows += _rows_for("Always-Up", w, 0, bh, strategy_returns_long_short(bh, r_test), rf, bps_grid,
                          y_test, np.ones_like(y_test))

        prev = positions_previous_day_direction(test_dates, dates, returns_map)
        rows += _rows_for("Previous-day direction", w, 0, prev, strategy_returns_long_short(prev, r_test), rf,
                          bps_grid, y_test, (prev > 0).astype(float))

        # AR(p): fit on the training split of this window only.
        window_dates = fold["window_dates"]
        r_window = np.array([returns_map[d] for d in window_dates], dtype=np.float64)
        n_tr, n_va = len(fold["train_dates"]), len(fold["val_dates"])
        val_idx = list(range(n_tr, n_tr + n_va))
        test_idx = list(range(n_tr + n_va, len(window_dates)))
        y_val = np.array([label_map[d] for d in fold["val_dates"]], dtype=np.float64)
        for lag in cfg.get("ar_lags", [1, 2, 3, 4, 5]):
            coefs = fit_ar(r_window[:n_tr], lag)
            val_pos = positions_from_forecasts(predict_ar(r_window, coefs, lag, val_idx))
            val_metrics = compute_trading_metrics(
                val_pos, strategy_returns_long_short(val_pos, r_window[val_idx]), np.full(n_va, rf_daily), 0.0,
                y_val, (val_pos > 0).astype(float),
            )
            ar_validation.append({"lag": lag, "window": w, **val_metrics})
            test_pos = positions_from_forecasts(predict_ar(r_window, coefs, lag, test_idx))
            ar_test.setdefault(lag, []).extend(
                _rows_for(f"AR({lag})", w, 0, test_pos, strategy_returns_long_short(test_pos, r_test), rf,
                          bps_grid, y_test, (test_pos > 0).astype(float))
            )

    val_df = pd.DataFrame(ar_validation)
    best_lag = select_ar_lag(val_df)
    val_df.to_csv(out_dir / "ar_validation_metrics.csv", index=False)
    for rec in ar_test[best_lag]:
        rows.append({**rec, "strategy": f"AR (validation-selected, p={best_lag})"})
    print(f"AR lag selected on validation Sharpe: p={best_lag}")

    for model in cfg.get("models", []):
        name, path, strategy = model["name"], Path(model["predictions"]), model.get("strategy", "long_short")
        if strategy not in STRATEGIES:
            raise ValueError(f"{name}: strategy must be one of {sorted(STRATEGIES)}, got {strategy!r}")
        if not path.exists():
            raise FileNotFoundError(f"{name}: predictions file not found: {path}")
        preds = pd.read_csv(path, dtype={"date": str})
        model_windows = sorted(int(w) for w in preds["window"].unique())
        fold_windows = [int(f["window"]) for f in folds]
        if model_windows != fold_windows:
            raise ValueError(
                f"{name}: predictions cover windows {model_windows} but the evaluation CV defines "
                f"{fold_windows}; the 'cv' section must match the one used for training."
            )
        for (seed, w), g in preds.groupby(["seed", "window"]):
            g = g.sort_values("date")
            fold = folds[int(w) - 1]
            if g["date"].tolist() != fold["test_dates"]:
                raise ValueError(f"{name}: test dates of seed={seed}, window={w} do not match the CV test window.")
            r_test = np.array([returns_map[d] for d in fold["test_dates"]], dtype=np.float64)
            rf = np.full(len(r_test), rf_daily)
            predicted = g["predicted_class"].to_numpy(dtype=np.float64)
            if strategy == "long_short":
                pos = positions_long_short(predicted)
                gross = strategy_returns_long_short(pos, r_test)
            else:
                pos = positions_long_cash(predicted)
                gross = strategy_returns_long_cash(pos, r_test, rf)
            rows += _rows_for(name, int(w), int(seed), pos, gross, rf, bps_grid,
                              g["y_true"].to_numpy(dtype=np.float64), predicted)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "trading_per_window_seed.csv", index=False)
    summary = aggregate_runs(per_run, ["strategy", "transaction_cost_bps"])
    summary.to_csv(out_dir / "trading_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(summary[[
            "strategy", "transaction_cost_bps", "directional_accuracy_mean", "gross_compounded_return_mean",
            "net_compounded_return_mean", "sharpe_ratio_mean", "max_drawdown_mean", "turnover_mean", "n_runs",
        ]].to_string(index=False))
    print(f"\nWrote {out_dir}/trading_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
