"""Mapping headline timestamps to the trading day whose information set they join.

Two policies are supported by ``scripts/data/build_dataset.py``:

``calendar`` (default; reproduces the released ``Input_t*.jsonl`` files)
    A headline belongs to its calendar date as stored in the cleaned news CSV
    (FNSPID timestamps truncated to the UTC date). Headlines dated on a
    non-trading day (weekend/holiday) are not attached to any sample.

``market_close``
    Timestamps are converted to ``America/New_York``. A headline published at
    or before the 16:00 close belongs to that trading day; a headline published
    after the close, or on a weekend/holiday, is rolled forward to the next
    available trading day. Headlines after the last trading day are dropped.

In both cases the information set of trading day ``d`` is used to predict the
direction of the close-to-close move ``d -> d+1`` (label
``Close_{d+1} > Close_d``). Because every headline assigned to ``d`` is
published before the close of ``d+1``, neither policy introduces look-ahead
relative to the label; ``market_close`` additionally guarantees that only
information available *before the close of d* is used for day ``d``.

Note: roughly 45% of raw FNSPID timestamps carry no intraday time
(``00:00:00 UTC``). Under ``market_close`` such a timestamp is 19:00/20:00 New
York time on the previous calendar day, i.e. after that day's close, so it is
rolled forward to the stamped calendar date (or the next trading day if that
date is not a trading day).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

MARKET_TZ = "America/New_York"
MARKET_CLOSE = "16:00"


def to_market_time(timestamps: Iterable, tz: str = MARKET_TZ) -> pd.Series:
    """Parse timestamps (naive values are interpreted as UTC) and convert them to ``tz``."""
    ts = pd.to_datetime(pd.Series(list(timestamps)), utc=True, errors="coerce")
    return ts.dt.tz_convert(tz)


def assign_trading_day(
    timestamps: Iterable,
    trading_days: Iterable,
    close_time: str = MARKET_CLOSE,
    tz: str = MARKET_TZ,
) -> pd.Series:
    """Trading day (``datetime64[ns]``, normalised) for each timestamp under the ``market_close`` policy.

    ``trading_days`` should be the dates of the market data (e.g. the ``Date``
    column of ``sp500.csv``). Unparseable timestamps and headlines after the
    last trading day map to ``NaT``.
    """
    days = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(trading_days)))).normalize().unique().sort_values()
    if len(days) == 0:
        raise ValueError("trading_days is empty.")
    local = to_market_time(timestamps, tz)
    close = pd.Timestamp(close_time)
    close_seconds = close.hour * 3600 + close.minute * 60 + close.second

    local_date = local.dt.tz_localize(None).dt.normalize()
    seconds = local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second
    after_close = (seconds > close_seconds).fillna(False).astype(int)  # exactly 16:00:00 counts as same day
    candidate = local_date + pd.to_timedelta(after_close, unit="D")

    pos = days.searchsorted(candidate.to_numpy(), side="left")
    valid = local.notna().to_numpy() & (pos < len(days))
    out = np.full(len(candidate), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[valid] = days.to_numpy()[pos[valid]]
    return pd.Series(out, index=candidate.index, name="trading_day")


def assign_calendar_day(dates: Iterable, trading_days: Iterable) -> pd.Series:
    """``calendar`` policy: the stored date if it is a trading day, else ``NaT`` (headline unused)."""
    days = set(pd.to_datetime(pd.Series(list(trading_days))).dt.normalize())
    d = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dt.normalize()
    return d.where(d.isin(days)).rename("trading_day")
