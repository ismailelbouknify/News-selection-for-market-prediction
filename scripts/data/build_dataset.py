"""Build the look-back datasets ``Input_t{T}.jsonl`` used by every experiment.

Each line is one trading day ``d`` (the last day of a ``T``-day window)::

    {"date": d, "markets": [[Open, High, Low, Close, Volume] x T],
     "headline_ids": [[ids of day d-T+1], ..., [ids of day d]],
     "sentiments":   [[[p_pos, p_neg, p_neu] per headline] per day],
     "label": int(Close_{d+1} > Close_d)}

Sentiment vectors are FinBERT softmax probabilities in the order
``[positive, negative, neutral]`` (FinBERT's native ``id2label`` order).
Market features are raw OHLCV levels; standardisation is done later, per CV
window, with statistics from the training split only.

News-to-day alignment (see ``greenfin.alignment``):

* ``--alignment calendar`` (default, used for the released datasets): a
  headline joins its stored calendar date (UTC); headlines dated on
  weekends/holidays are not used.
* ``--alignment market_close``: 16:00 New York cutoff, after-close and
  non-trading-day headlines roll forward to the next trading day. Requires the
  full timestamp column written by ``build_cleaned_news.py --keep-timestamp``.

Usage::

    python scripts/data/build_dataset.py --lookbacks 1 3 5 10 20
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from scipy.special import softmax
from tqdm.auto import tqdm

from greenfin.alignment import assign_calendar_day, assign_trading_day


# =======================
# Config
# =======================
NEWS_CSV = "data/interim/cleaned_news_sentiment.csv"
MARKET_CSV = "data/raw/market/sp500.csv"
OUTPUT_DIR = "data/processed"

START_DATE = "2007-01-01"
END_DATE = "2024-01-01"

WINDOW_SIZES = [1, 3, 5, 10, 20]

NEWS_DATE_COL = "Date"
MARKET_DATE_COL = "Date"
SENTIMENT_COLS = ["positive", "negative", "neutral"]
MARKET_FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]
# =======================


@dataclass(frozen=True)
class Config:
    news_csv: str = NEWS_CSV
    market_csv: str = MARKET_CSV
    output_dir: str = OUTPUT_DIR
    start_date: str = START_DATE
    end_date: str = END_DATE
    window_sizes: tuple[int, ...] = tuple(WINDOW_SIZES)


def load_news_data(path: str, start_date: str, end_date: str, timestamp_col: str | None = None) -> pd.DataFrame:
    """Load sentiment-enriched news and prepare row-aligned headline IDs."""
    print("Loading news data...")
    df = pd.read_csv(path)

    required_cols = {"Date", "Article_title", "positive", "negative", "neutral"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    # Keep headline_id consistent with previous pipeline:
    # row 1..N in the CSV after sentiment generation.
    df = df.copy().reset_index(drop=True)
    df["headline_id"] = range(1, len(df) + 1)

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()

    keep = ["Date", "Article_title", "positive", "negative", "neutral", "headline_id"]
    if timestamp_col and timestamp_col in df.columns:
        keep.append(timestamp_col)
    df = df[keep]
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Loaded {len(df):,} filtered news rows from {path}")
    return df


def apply_sentiment_softmax(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw sentiment logits into probabilities row-wise."""
    print("Applying softmax to sentiment columns...")
    df = df.copy()
    df[SENTIMENT_COLS] = df[SENTIMENT_COLS].apply(
        lambda row: pd.Series(softmax(row.values), index=SENTIMENT_COLS),
        axis=1,
    )
    return df


def load_market_data(path: str) -> pd.DataFrame:
    """Load market data and build next-day-vs-today binary labels."""
    print("Loading market data...")
    df = pd.read_csv(path)

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    # label_i = 1 if Close_{i+1} > Close_i else 0
    df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Keep only rows that have a valid "next day" label target
    df = df.iloc[:-1].copy().reset_index(drop=True)

    print(f"Loaded {len(df):,} market rows from {path}")
    return df


def create_training_dataset(
    df_news: pd.DataFrame,
    df_market: pd.DataFrame,
    t: int,
    output_dir: str,
) -> List[Dict]:
    """
    Build the FININ-style JSONL dataset for a lookback window of size t.

    For each sample:
      - markets use days [i - t + 1, ..., i]
      - news is grouped per day over the same window
      - label is for day i: whether Close_{i+1} > Close_i
    """
    print(f"Creating dataset for t={t}...")

    news_by_date = df_news.groupby("TradingDay")
    structured_samples: List[Dict] = []

    for i in tqdm(range(t - 1, len(df_market)), desc=f"Building Input_t{t}.jsonl"):
        window = df_market.iloc[i - t + 1 : i + 1]
        target_row = df_market.iloc[i]

        market_data = window[MARKET_FEATURE_COLS].values.tolist()
        market_dates = window["Date"].tolist()

        news_ids_per_day: List[List[int]] = []
        news_sentiments_per_day: List[List[List[float]]] = []

        for date in market_dates:
            if date in news_by_date.groups:
                daily_news = news_by_date.get_group(date)
                ids = daily_news["headline_id"].astype(int).tolist()
                sentiments = daily_news[SENTIMENT_COLS].values.tolist()
            else:
                ids = []
                sentiments = []

            news_ids_per_day.append(ids)
            news_sentiments_per_day.append(sentiments)

        sample = {
            "date": str(window.iloc[-1]["Date"].date()),
            "markets": market_data,
            "headline_ids": news_ids_per_day,
            "sentiments": news_sentiments_per_day,
            "label": int(target_row["label"]),
        }
        structured_samples.append(sample)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"Input_t{t}.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in structured_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(structured_samples):,} samples to {output_path}")
    return structured_samples


def attach_trading_day(
    df_news: pd.DataFrame, df_market: pd.DataFrame, alignment: str, timestamp_col: str
) -> pd.DataFrame:
    """Add the ``TradingDay`` each headline contributes to; unassigned headlines are dropped."""
    df_news = df_news.copy()
    if alignment == "calendar":
        df_news["TradingDay"] = assign_calendar_day(df_news["Date"], df_market["Date"]).to_numpy()
    elif alignment == "market_close":
        if timestamp_col not in df_news.columns:
            raise ValueError(
                f"--alignment market_close needs the full timestamp column {timestamp_col!r}; "
                "re-run build_cleaned_news.py with --keep-timestamp and the sentiment step."
            )
        df_news["TradingDay"] = assign_trading_day(df_news[timestamp_col], df_market["Date"]).to_numpy()
        # Keep chronological order of the assigned day (stable within a day).
        df_news = df_news.sort_values("TradingDay", kind="stable")
    else:
        raise ValueError(f"Unknown alignment {alignment!r}")
    n_before = len(df_news)
    df_news = df_news.dropna(subset=["TradingDay"])
    print(f"Alignment '{alignment}': {len(df_news):,} of {n_before:,} headlines assigned to a trading day.")
    return df_news


def parse_args() -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--news-csv", default=d.news_csv)
    p.add_argument("--market-csv", default=d.market_csv)
    p.add_argument("--output-dir", default=d.output_dir)
    p.add_argument("--start-date", default=d.start_date, help="First news date kept (default: %(default)s)")
    p.add_argument("--end-date", default=d.end_date, help="Last news date kept (default: %(default)s)")
    p.add_argument("--lookbacks", type=int, nargs="+", default=list(d.window_sizes))
    p.add_argument("--alignment", choices=["calendar", "market_close"], default="calendar")
    p.add_argument("--timestamp-col", default="Datetime", help="Full-timestamp column for market_close alignment")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    news_df = load_news_data(args.news_csv, args.start_date, args.end_date, timestamp_col=args.timestamp_col)
    news_df = apply_sentiment_softmax(news_df)
    market_df = load_market_data(args.market_csv)
    news_df = attach_trading_day(news_df, market_df, args.alignment, args.timestamp_col)

    for t in args.lookbacks:
        create_training_dataset(df_news=news_df, df_market=market_df, t=t, output_dir=args.output_dir)

    print("All datasets created successfully.")


if __name__ == "__main__":
    main()