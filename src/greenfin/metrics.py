"""Headline financial metrics (directional accuracy, PnL, Sharpe ratio).

Primary (paper) definition, used by :func:`greenfin.evaluate.evaluate` and
the cross-validation routines:

* next-day market return ``r_d = Close_{d+1} / Close_d - 1``;
* the predicted class is mapped to a tradable position
  ``s_d = +1`` (predicted up, long) or ``s_d = -1`` (predicted down, short);
* daily strategy return ``R_d = s_d * r_d``;
* ``PnL = sum_d R_d`` (cumulative, non-compounded) over a test window;
* annualised Sharpe ``(mean(R) - rf/252) / std(R, ddof=1) * sqrt(252)`` with
  an annual risk-free rate ``rf = 0.02``.

Metrics are computed per test window, averaged over windows within a seed and
then over seeds.

The *legacy* FININ-style metric (:func:`legacy_finin_daily_returns`) is kept
only to reproduce the original FININ protocol. It multiplies the realised
return by ``+1`` if the prediction was correct and ``-1`` otherwise, which is
not a tradable strategy (a correct "down" call on a falling day books a
*loss*). It must not be reported as the primary PnL.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def get_returns_map(market_csv: str) -> Dict[str, float]:
    """``{date: Close_{next trading day} / Close_date - 1}`` keyed by ``YYYY-MM-DD``."""
    df = pd.read_csv(market_csv)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        dups = int(df["Date"].duplicated().sum())
        print(f"[get_returns_map] deduped {dups} rows by Date using last Close.")
        df = df.groupby("Date", as_index=False).agg({"Close": "last"})

    df["ret_next"] = df["Close"].shift(-1) / df["Close"] - 1.0
    return {d.strftime("%Y-%m-%d"): float(r) for d, r in zip(df["Date"], df["ret_next"])}


def positions_from_predictions(predicted_class: Sequence[float]) -> np.ndarray:
    """Predicted class 1 -> +1 (long), 0 -> -1 (short)."""
    p = np.asarray(predicted_class, dtype=np.float64)
    return np.where(p >= 1.0, 1.0, -1.0)


def compute_pnl_sharpe(R: List[float], rf_annual: float, tdays: int = 252) -> Tuple[float, float]:
    """Summed PnL and annualised Sharpe of daily strategy returns ``R``.

    Returns ``(nan, nan)`` for fewer than three observations and a NaN Sharpe
    when the returns have zero variance.
    """
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


def legacy_finin_daily_returns(
    predicted_class: Sequence[float],
    labels: Sequence[float],
    market_returns: Sequence[float],
) -> np.ndarray:
    """LEGACY FININ metric: ``(+1 if correct else -1) * r_d``. Not a tradable strategy."""
    p = np.asarray(predicted_class, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    r = np.asarray(market_returns, dtype=np.float64)
    return np.where(p == y, 1.0, -1.0) * r
