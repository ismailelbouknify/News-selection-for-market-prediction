from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def get_returns_map(market_csv: str) -> Dict[str, float]:
    df = pd.read_csv(market_csv)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        dups = int(df["Date"].duplicated().sum())
        print(f"[get_returns_map] deduped {dups} rows by Date using last Close.")
        df = df.groupby("Date", as_index=False).agg({"Close": "last"})

    df["ret_next"] = df["Close"].shift(-1) / df["Close"] - 1.0
    return {d.strftime("%Y-%m-%d"): float(r) for d, r in zip(df["Date"], df["ret_next"])}


def compute_pnl_sharpe(R: List[float], rf_annual: float, tdays: int = 252) -> Tuple[float, float]:
    if len(R) <= 2:
        return float("nan"), float("nan")

    arr = np.asarray(R, dtype=np.float64)
    pnl = float(arr.sum())
    vol = float(arr.std(ddof=1))
    mean = float(arr.mean())
    if vol <= 0:
        return pnl, float("nan")

    rf_daily = rf_annual / tdays
    sharpe_daily = (mean - rf_daily) / vol
    sharpe_annual = sharpe_daily * np.sqrt(tdays)
    return pnl, sharpe_annual
