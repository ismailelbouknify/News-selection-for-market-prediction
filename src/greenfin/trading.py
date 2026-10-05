"""Practical trading evaluation: positions, transaction costs and risk metrics.

Used for the paper's practical trading analysis (long/short and long/cash
strategies, buy-and-hold / Always-Up, previous-day direction and AR
benchmarks). This is a *different* convention from the headline PnL/Sharpe in
:mod:`greenfin.metrics`:

* returns are **compounded** (``prod(1 + r) - 1``), not summed;
* the Sharpe ratio is computed on daily *excess* returns
  ``r_t - rf_t`` with ``rf_t = 0.02 / 252`` by default;
* every metric is computed on one (window, seed) test series at a time and
  then averaged -- daily series are never concatenated across windows.

Position convention: ``+1`` = long, ``-1`` = short, ``0`` = cash.

Transaction costs are proportional to the absolute change in position::

    cost_t = bps / 10_000 * |position_t - position_{t-1}|

with ``position_{-1} = 0`` (each test window starts from cash). A long<->short
flip therefore costs twice as much as entering or leaving the market.
Turnover and position changes use the same convention so that they reconcile
with the costs actually charged.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
DEFAULT_RF_ANNUAL = 0.02


# ---------------------------------------------------------------------------
# Positions and strategy returns
# ---------------------------------------------------------------------------
def positions_long_short(predicted_label: Sequence[float]) -> np.ndarray:
    """Predicted up (1) -> +1 (long); predicted down (0) -> -1 (short)."""
    lab = np.asarray(predicted_label, dtype=np.float64)
    return np.where(lab >= 0.5, 1.0, -1.0)


def positions_long_cash(predicted_label: Sequence[float]) -> np.ndarray:
    """Predicted up (1) -> +1 (invested); predicted down (0) -> 0 (cash)."""
    lab = np.asarray(predicted_label, dtype=np.float64)
    return np.where(lab >= 0.5, 1.0, 0.0)


def positions_buy_and_hold(n_days: int) -> np.ndarray:
    """Long on every test day (identical positions to an Always-Up classifier)."""
    return np.ones(int(n_days), dtype=np.float64)


def sign_to_position(x: float) -> float:
    """``+1`` for ``x >= 0`` and ``-1`` otherwise."""
    return 1.0 if x >= 0 else -1.0


def positions_from_forecasts(predicted_returns: Sequence[float]) -> np.ndarray:
    """Sign of a point forecast of the next-day return (AR benchmarks)."""
    return np.asarray([sign_to_position(float(x)) for x in predicted_returns], dtype=np.float64)


def positions_previous_day_direction(
    test_dates: Sequence[str],
    all_dates: Sequence[str],
    returns_map: Dict[str, float],
) -> np.ndarray:
    """Hold the sign of the most recent *realised* daily return.

    ``returns_map[x]`` is the return from ``x`` to the next trading day, so
    ``returns_map[previous_date]`` is the return realised *on* ``d`` -- known
    at the close of ``d`` and therefore usable to position for ``d -> d+1``.
    """
    index_of = {d: i for i, d in enumerate(all_dates)}
    out = np.empty(len(test_dates), dtype=np.float64)
    for j, d in enumerate(test_dates):
        i = index_of[d]
        if i == 0:
            raise ValueError(f"{d} is the first date; no previous-day return is available.")
        r_prev = returns_map.get(all_dates[i - 1])
        if r_prev is None or not np.isfinite(r_prev):
            raise ValueError(f"Missing or non-finite realised return for {all_dates[i - 1]}.")
        out[j] = sign_to_position(r_prev)
    return out


def strategy_returns_long_short(positions: Sequence[float], market_returns: Sequence[float]) -> np.ndarray:
    """Daily strategy return ``position_t * r_{t -> t+1}``."""
    return np.asarray(positions, dtype=np.float64) * np.asarray(market_returns, dtype=np.float64)


def strategy_returns_long_cash(
    positions: Sequence[float],
    market_returns: Sequence[float],
    daily_risk_free: Sequence[float],
) -> np.ndarray:
    """Invested days earn the market return; cash days earn the daily risk-free rate."""
    p = np.asarray(positions, dtype=np.float64)
    return np.where(p >= 0.5, np.asarray(market_returns, dtype=np.float64), np.asarray(daily_risk_free, dtype=np.float64))


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------
def previous_positions(positions: Sequence[float], initial_position: float = 0.0) -> np.ndarray:
    p = np.asarray(positions, dtype=np.float64)
    prev = np.empty_like(p)
    if p.size:
        prev[0] = initial_position
        prev[1:] = p[:-1]
    return prev


def transaction_costs(positions: Sequence[float], bps: float, initial_position: float = 0.0) -> np.ndarray:
    """``bps / 10_000 * |position_t - position_{t-1}|`` per day."""
    if bps < 0:
        raise ValueError(f"Transaction cost must be non-negative, got {bps} bps.")
    p = np.asarray(positions, dtype=np.float64)
    return (bps / 10_000.0) * np.abs(p - previous_positions(p, initial_position))


def net_returns(gross_returns: Sequence[float], costs: Sequence[float]) -> np.ndarray:
    g = np.asarray(gross_returns, dtype=np.float64)
    c = np.asarray(costs, dtype=np.float64)
    if g.shape != c.shape:
        raise ValueError(f"gross_returns shape {g.shape} != costs shape {c.shape}")
    return g - c


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compounded_return(daily_returns: Sequence[float]) -> float:
    r = np.asarray(daily_returns, dtype=np.float64)
    return float(np.prod(1.0 + r) - 1.0) if r.size else float("nan")


def wealth_series(daily_returns: Sequence[float], start_wealth: float = 1.0) -> np.ndarray:
    r = np.asarray(daily_returns, dtype=np.float64)
    return start_wealth * np.cumprod(1.0 + r) if r.size else np.asarray([], dtype=np.float64)


def max_drawdown(wealth: Sequence[float]) -> float:
    """Most negative ``wealth_t / max_{s<=t} wealth_s - 1`` (0 if never below a peak)."""
    w = np.asarray(wealth, dtype=np.float64)
    if w.size == 0:
        return float("nan")
    return float((w / np.maximum.accumulate(w) - 1.0).min())


def annualized_return(daily_returns: Sequence[float], tdays: int = TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(daily_returns, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    base = 1.0 + compounded_return(r)
    if base <= 0:
        return -1.0
    return float(base ** (tdays / r.size) - 1.0)


def annualized_volatility(daily_returns: Sequence[float], tdays: int = TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(daily_returns, dtype=np.float64)
    return float(r.std(ddof=1) * np.sqrt(tdays)) if r.size >= 2 else float("nan")


def sharpe_ratio(
    daily_returns: Sequence[float],
    daily_risk_free: Sequence[float],
    tdays: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio of daily excess returns ``r_t - rf_t``."""
    r = np.asarray(daily_returns, dtype=np.float64)
    rf = np.asarray(daily_risk_free, dtype=np.float64)
    if rf.size != r.size:
        raise ValueError(f"daily_risk_free length ({rf.size}) must match daily_returns length ({r.size}).")
    if r.size < 2:
        return float("nan")
    excess = r - rf
    vol = float(excess.std(ddof=1))
    if vol <= 0:
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(tdays))


def turnover(positions: Sequence[float], initial_position: float = 0.0) -> float:
    """Mean absolute daily position change, including the entry from cash."""
    p = np.asarray(positions, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - previous_positions(p, initial_position))))


def position_changes(positions: Sequence[float], initial_position: float = 0.0) -> int:
    """Number of days on which the position changes, including the entry from cash."""
    p = np.asarray(positions, dtype=np.float64)
    if p.size == 0:
        return 0
    return int(np.sum(p != previous_positions(p, initial_position)))


def directional_accuracy(actual_label: Sequence[float], predicted_label: Sequence[float]) -> float:
    a = np.asarray(actual_label, dtype=np.float64)
    p = np.asarray(predicted_label, dtype=np.float64)
    mask = ~np.isnan(a) & ~np.isnan(p)
    return float(np.mean(a[mask] == p[mask])) if mask.any() else float("nan")


def compute_trading_metrics(
    positions: Sequence[float],
    gross_returns: Sequence[float],
    daily_risk_free: Sequence[float],
    bps: float,
    actual_label: Optional[Sequence[float]] = None,
    predicted_label: Optional[Sequence[float]] = None,
    tdays: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, float]:
    """All practical-trading metrics for ONE (window, seed) test series."""
    gross = np.asarray(gross_returns, dtype=np.float64)
    net = net_returns(gross, transaction_costs(positions, bps))
    return {
        "directional_accuracy": (
            directional_accuracy(actual_label, predicted_label)
            if actual_label is not None and predicted_label is not None
            else float("nan")
        ),
        "gross_compounded_return": compounded_return(gross),
        "net_compounded_return": compounded_return(net),
        "annualized_return": annualized_return(gross, tdays),
        "annualized_volatility": annualized_volatility(gross, tdays),
        "sharpe_ratio": sharpe_ratio(gross, daily_risk_free, tdays),
        "net_sharpe_ratio": sharpe_ratio(net, daily_risk_free, tdays),
        "max_drawdown": max_drawdown(wealth_series(gross)),
        "net_max_drawdown": max_drawdown(wealth_series(net)),
        "turnover": turnover(positions),
        "position_changes": float(position_changes(positions)),
        "transaction_cost_bps": float(bps),
    }


def aggregate_runs(per_run: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Mean and standard deviation (ddof=1) of every metric across (window, seed) runs."""
    id_cols = {"window", "seed", *group_cols}
    metric_cols = [c for c in per_run.columns if c not in id_cols and pd.api.types.is_numeric_dtype(per_run[c])]
    agg = per_run.groupby(list(group_cols), sort=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg["n_runs"] = per_run.groupby(list(group_cols), sort=False).size()
    return agg.reset_index()
