from __future__ import annotations

import pandas as pd

from greenfin.alignment import assign_calendar_day, assign_trading_day

# Week of 2023-07-03: Monday 3rd is a trading day, Tuesday 4th (Independence Day) is a holiday.
TRADING_DAYS = ["2023-06-30", "2023-07-03", "2023-07-05", "2023-07-06", "2023-07-07", "2023-07-10"]


def _assign(ts):
    return [None if pd.isna(x) else str(pd.Timestamp(x).date()) for x in assign_trading_day(ts, TRADING_DAYS)]


def test_headline_before_close_belongs_to_same_trading_day():
    # 15:59 New York (EDT, UTC-4) = 19:59 UTC
    assert _assign(["2023-07-05 19:59:00+00:00"]) == ["2023-07-05"]


def test_headline_exactly_at_close_belongs_to_same_day():
    assert _assign(["2023-07-05 20:00:00+00:00"]) == ["2023-07-05"]


def test_headline_after_close_moves_to_next_trading_day():
    # 16:01 New York
    assert _assign(["2023-07-05 20:01:00+00:00"]) == ["2023-07-06"]


def test_friday_after_close_and_weekend_move_to_monday():
    assert _assign([
        "2023-07-07 21:00:00+00:00",  # Friday 17:00 NY
        "2023-07-08 15:00:00+00:00",  # Saturday
        "2023-07-09 23:00:00+00:00",  # Sunday 19:00 NY
    ]) == ["2023-07-10", "2023-07-10", "2023-07-10"]


def test_holiday_moves_to_next_trading_day():
    assert _assign(["2023-07-04 14:00:00+00:00", "2023-07-03 21:00:00+00:00"]) == ["2023-07-05", "2023-07-05"]


def test_winter_time_offset_is_handled():
    trading = ["2023-01-05", "2023-01-06"]
    # EST is UTC-5: 20:30 UTC = 15:30 NY (same day); 21:30 UTC = 16:30 NY (next day).
    out = assign_trading_day(["2023-01-05 20:30:00+00:00", "2023-01-05 21:30:00+00:00"], trading)
    assert [str(x.date()) for x in out] == ["2023-01-05", "2023-01-06"]


def test_date_only_utc_midnight_stays_on_stamped_trading_day():
    # 00:00 UTC is the previous evening in New York (after that day's close).
    assert _assign(["2023-07-06 00:00:00 UTC"]) == ["2023-07-06"]


def test_naive_timestamps_are_utc_and_late_headlines_are_dropped():
    assert _assign(["2023-07-06 13:00:00", "2023-07-10 21:00:00+00:00", "not a date"]) == ["2023-07-06", None, None]


def test_calendar_policy_drops_non_trading_days():
    out = assign_calendar_day(["2023-07-03", "2023-07-04", "2023-07-08", "2023-07-05"], TRADING_DAYS)
    assert [None if pd.isna(x) else str(x.date()) for x in out] == ["2023-07-03", None, None, "2023-07-05"]
