"""Autoregressive return benchmark for the practical trading evaluation.

An AR(p) model of daily S&P 500 returns is fitted by ordinary least squares on
the *training* part of each sliding window only. Validation and test
predictions are one-step-ahead forecasts that use realised (never forecast)
lagged returns, all strictly earlier than the forecast target, so the
benchmark has no look-ahead.

The lag order ``p`` is chosen per experiment from the validation windows only
(:func:`select_ar_lag`); test performance is never used for model selection.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def build_lagged_design(returns: Sequence[float], lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Design matrix ``X = [1, r_{t-1}, ..., r_{t-lag}]`` and target ``y = r_t``."""
    if lag < 1:
        raise ValueError(f"AR lag must be >= 1, got {lag}.")
    r = np.asarray(returns, dtype=np.float64)
    n = r.size
    if n <= lag:
        raise ValueError(f"Need more than {lag} observations to fit AR({lag}), got {n}.")
    X = np.ones((n - lag, lag + 1), dtype=np.float64)
    for i in range(lag, n):
        X[i - lag, 1:] = r[i - lag : i][::-1]
    return X, r[lag:].copy()


def fit_ar(train_returns: Sequence[float], lag: int) -> np.ndarray:
    """OLS coefficients ``[intercept, phi_1, ..., phi_lag]`` from training returns only."""
    X, y = build_lagged_design(train_returns, lag)
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coefs


def predict_ar(returns_full: Sequence[float], coefs: np.ndarray, lag: int, indices: Sequence[int]) -> np.ndarray:
    """One-step-ahead forecasts of ``returns_full[i]`` for each ``i`` in ``indices``.

    Only ``returns_full[i - lag : i]`` (already realised values) enter the
    forecast for index ``i``.
    """
    r = np.asarray(returns_full, dtype=np.float64)
    out = np.empty(len(indices), dtype=np.float64)
    for j, i in enumerate(indices):
        if i < lag:
            raise ValueError(f"Cannot forecast index {i} with AR({lag}): not enough history.")
        out[j] = float(np.concatenate([[1.0], r[i - lag : i][::-1]]) @ coefs)
    return out


def select_ar_lag(validation_metrics: pd.DataFrame) -> int:
    """Lag with the best mean *validation* Sharpe ratio (ties: validation accuracy).

    ``validation_metrics`` has one row per (lag, window) with columns
    ``lag``, ``sharpe_ratio`` and ``directional_accuracy``.
    """
    required = {"lag", "sharpe_ratio", "directional_accuracy"}
    missing = required - set(validation_metrics.columns)
    if missing:
        raise ValueError(f"validation_metrics is missing columns {sorted(missing)}")
    summary = (
        validation_metrics.groupby("lag")[["sharpe_ratio", "directional_accuracy"]]
        .mean()
        .sort_values(["sharpe_ratio", "directional_accuracy"], ascending=False)
    )
    return int(summary.index[0])
