from __future__ import annotations

import argparse

from build_cleaned_news import clean_news_file
from download_sp500 import download_sp500, save_dataframe


DEFAULT_RAW_NEWS_INPUT = "data/raw/news/nasdaq_external_data.csv"
DEFAULT_CLEANED_NEWS_OUTPUT = "data/interim//cleaned_news.csv"
DEFAULT_SP500_OUTPUT = "data/raw/market/sp500.csv"

DEFAULT_TICKER = "^GSPC"
DEFAULT_START_DATE = "2007-07-23"
DEFAULT_END_DATE = "2024-01-01"
DEFAULT_INTERVAL = "1d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full preprocessing pipeline: clean news data and download S&P 500 data."
    )

    parser.add_argument(
        "--news-input",
        default=DEFAULT_RAW_NEWS_INPUT,
        help=f"Path to raw news CSV. Default: {DEFAULT_RAW_NEWS_INPUT}",
    )
    parser.add_argument(
        "--news-output",
        default=DEFAULT_CLEANED_NEWS_OUTPUT,
        help=f"Path to save cleaned news CSV. Default: {DEFAULT_CLEANED_NEWS_OUTPUT}",
    )
    parser.add_argument(
        "--market-output",
        default=DEFAULT_SP500_OUTPUT,
        help=f"Path to save S&P 500 CSV. Default: {DEFAULT_SP500_OUTPUT}",
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help=f"Ticker symbol to download. Default: {DEFAULT_TICKER}",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help=f"Start date in YYYY-MM-DD format. Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END_DATE,
        help=f"End date in YYYY-MM-DD format. Default: {DEFAULT_END_DATE}",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help=f"Download interval. Default: {DEFAULT_INTERVAL}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Step 1/2: Cleaning news data...")
    cleaned_df = clean_news_file(
        input_path=args.news_input,
        output_path=args.news_output,
    )
    print(f"Cleaned news saved to: {args.news_output}")
    print(f"News rows: {len(cleaned_df):,}")

    print("\nStep 2/2: Downloading S&P 500 data...")
    market_df = download_sp500(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
    )
    save_dataframe(market_df, args.market_output)
    print(f"Market data saved to: {args.market_output}")
    print(f"Market shape: {market_df.shape}")

    print("\nPreprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()