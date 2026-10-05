from __future__ import annotations

import argparse
import os

import yfinance as yf


DEFAULT_TICKER = "^GSPC"
DEFAULT_START_DATE = "2007-07-23"
DEFAULT_END_DATE = "2024-01-01"
DEFAULT_INTERVAL = "1d"
DEFAULT_OUTPUT_PATH = "data/raw/market/sp500.csv"


def download_sp500(
    ticker: str = DEFAULT_TICKER,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    interval: str = DEFAULT_INTERVAL,
):
    """
    Download daily S&P 500 data from Yahoo Finance.
    """
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for ticker={ticker}, start={start_date}, end={end_date}, interval={interval}."
        )

    # Flatten multi-index columns if present
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df.columns.name = None
    df = df.reset_index()

    return df


def save_dataframe(df, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily S&P 500 data from Yahoo Finance."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker symbol.")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="End date YYYY-MM-DD.")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Download interval.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = download_sp500(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
    )
    save_dataframe(df, args.output)

    print("Download completed.")
    print(f"Saved market data to: {args.output}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()