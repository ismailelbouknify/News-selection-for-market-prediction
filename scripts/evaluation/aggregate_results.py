#!/usr/bin/env python
"""Collect ``summary.json`` files of finished runs into one results table.

Accuracy, PnL and Sharpe are mean +/- std across seeds of the per-seed window
means. Resource columns describe the DOWNSTREAM forecasting stage only
(training and inference of the forecaster on precomputed RoBERTa embeddings
and FinBERT sentiment):

* ``train_time_min`` / ``energy_kWh`` / ``co2_kg``: sum over the 5 windows of
  one seed, averaged across seeds (``*_all_seeds`` = summed over all seeds);
* ``inference_s_per_1000``: test-time inference seconds per 1,000 predictions.

``--upstream-carbon`` adds the one-off encoding footprint written by
``scripts/data/build_sentiment.py`` / ``build_embeddings.py`` to obtain the
conservative end-to-end accounting, in which the SAME upstream cost is added
to every configuration (selection happens after encoding).

Usage::

    python scripts/evaluation/aggregate_results.py --runs-root outputs/runs --output results/summary.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _nansum(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.nansum(arr)) if np.isfinite(arr).any() else float("nan")


def _row(name: str, summary: Dict) -> Dict:
    per_seed = summary.get("per_seed_summaries", [summary])

    def stat(key: str):
        vals = np.asarray([s.get(key, np.nan) for s in per_seed], dtype=np.float64)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    acc, acc_sd = stat("acc_mean")
    pnl, pnl_sd = stat("pnl_mean")
    sr, sr_sd = stat("sr_mean")
    train, _ = stat("train_time_min_total")
    energy, _ = stat("energy_kWh_total")
    co2_g, _ = stat("CO2_g_total")
    infer_ms, _ = stat("infer_time_per_day_ms_total")
    return {
        "experiment": name,
        "n_seeds": len(per_seed),
        "DA_mean": acc, "DA_std": acc_sd,
        "PnL_mean": pnl, "PnL_std": pnl_sd,
        "SR_mean": sr, "SR_std": sr_sd,
        "train_time_min": train,
        "energy_kWh": energy,
        "co2_kg": co2_g / 1000.0,
        "train_time_min_all_seeds": _nansum(s.get("train_time_min_total", np.nan) for s in per_seed),
        "energy_kWh_all_seeds": _nansum(s.get("energy_kWh_total", np.nan) for s in per_seed),
        "co2_kg_all_seeds": _nansum(s.get("CO2_g_total", np.nan) for s in per_seed) / 1000.0,
        "inference_s_per_1000": infer_ms,  # ms per prediction == s per 1,000 predictions
    }


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="outputs/runs")
    p.add_argument("--output", default="results/summary.csv")
    p.add_argument("--experiments", nargs="*", help="Only these experiment names (default: all runs found).")
    p.add_argument("--upstream-carbon", nargs="*", default=[],
                   help="JSON files with upstream encoding energy_kWh / co2_kg (added to every row).")
    args = p.parse_args(argv)

    rows = []
    paths = sorted(Path(args.runs_root).glob("*/summary.json"))
    if args.experiments:
        found = {pth.parent.name: pth for pth in paths}
        missing = [e for e in args.experiments if e not in found]
        if missing:
            raise SystemExit(f"No summary.json under {args.runs_root} for: {missing}")
        paths = [found[e] for e in args.experiments]
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("cv_method") != "kfold_time_cv":
            print(f"skipping {path} (cv_method={payload.get('cv_method')})")
            continue
        rows.append(_row(payload["experiment_name"], payload["summary"]))
    if not rows:
        raise SystemExit(f"No kfold_time_cv summaries found under {args.runs_root}")
    df = pd.DataFrame(rows)

    if args.upstream_carbon:
        up_energy = up_co2 = 0.0
        for path in args.upstream_carbon:
            with open(path, "r", encoding="utf-8") as f:
                up = json.load(f)
            up_energy += float(up["energy_kWh"])
            up_co2 += float(up["co2_kg"])
        df["upstream_energy_kWh"] = up_energy
        df["upstream_co2_kg"] = up_co2
        df["end_to_end_energy_kWh"] = df["energy_kWh"] + up_energy
        df["end_to_end_co2_kg"] = df["co2_kg"] + up_co2

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
