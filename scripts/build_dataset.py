from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from scipy.special import softmax
from tqdm.auto import tqdm


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


def load_news_data(path: str, start_date: str, end_date: str) -> pd.DataFrame:
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

    df = df[["Date", "Article_title", "positive", "negative", "neutral", "headline_id"]]
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

    news_by_date = df_news.groupby("Date")
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


def main() -> None:
    cfg = Config()

    news_df = load_news_data(
        path=cfg.news_csv,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
    )
    news_df = apply_sentiment_softmax(news_df)

    market_df = load_market_data(cfg.market_csv)

    for t in cfg.window_sizes:
        create_training_dataset(
            df_news=news_df,
            df_market=market_df,
            t=t,
            output_dir=cfg.output_dir,
        )

    print("All datasets created successfully.")


if __name__ == "__main__":
    main()