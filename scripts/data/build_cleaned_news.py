"""Clean the raw FNSPID news CSV.

Keeps ``Date``, ``Article_title``, ``Article`` and ``Lsa_summary``; truncates the
timestamp to its UTC calendar date; drops unparseable dates, exact duplicates,
rows with Cyrillic text in the title/article and rows before 1999-01-01.

``--keep-timestamp`` additionally stores the full UTC timestamp in a
``Datetime`` column (needed for ``build_dataset.py --alignment market_close``).
Duplicates are still identified on the original columns, so the retained rows
-- and therefore the 1-based ``headline_id`` of every row -- are unchanged.
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from typing import Iterable

import pandas as pd


KEEP_COLUMNS = ["Date", "Article_title", "Article", "Lsa_summary"]
CYRILLIC_PATTERN = r"[А-Яа-яЁё]"
MIN_DATE = date(1999, 1, 1)
DEFAULT_OUTPUT_PATH = "data/interim/cleaned_news.csv"


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def clean_news_dataframe(
    df: pd.DataFrame,
    output_path: str = DEFAULT_OUTPUT_PATH,
    keep_timestamp: bool = False,
) -> pd.DataFrame:
    """
    Clean the raw news dataframe and save it to CSV.

    Steps:
    - keep only selected columns
    - parse Date and drop invalid dates
    - remove duplicate rows
    - remove rows containing Cyrillic characters in title/article
    - keep only rows from 1999-01-01 onward
    - fill missing values with empty strings
    - save cleaned CSV
    """
    existing_cols = _existing_columns(df, KEEP_COLUMNS)
    if "Date" not in existing_cols:
        raise ValueError("Input dataframe must contain a 'Date' column.")

    df1 = df.loc[:, existing_cols].copy()

    # Parse date and drop invalid values
    parsed = pd.to_datetime(df1["Date"], errors="coerce")
    if keep_timestamp:
        df1["Datetime"] = pd.to_datetime(df1["Date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    df1["Date"] = parsed.dt.date
    df1 = df1.dropna(subset=["Date"])

    # Remove exact duplicate rows (judged on the original columns only)
    df1 = df1.drop_duplicates(subset=existing_cols).reset_index(drop=True)

    # Remove Cyrillic text rows where relevant columns exist
    if "Article_title" in df1.columns:
        df1 = df1[
            ~df1["Article_title"].astype(str).str.contains(
                CYRILLIC_PATTERN, regex=True, na=False
            )
        ]

    if "Article" in df1.columns:
        df1 = df1[
            ~df1["Article"].astype(str).str.contains(
                CYRILLIC_PATTERN, regex=True, na=False
            )
        ]

    # Keep only rows from 1999-01-01 onward
    df1 = df1[df1["Date"] >= MIN_DATE].reset_index(drop=True)

    # Fill missing values
    df1 = df1.fillna("")

    # Save
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df1.to_csv(output_path, index=False)
    return df1


def clean_news_file(
    input_path: str,
    output_path: str = DEFAULT_OUTPUT_PATH,
    keep_timestamp: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    return clean_news_dataframe(df, output_path=output_path, keep_timestamp=keep_timestamp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw news CSV for the GreenFin pipeline.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw input CSV.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save the cleaned CSV. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--keep-timestamp",
        action="store_true",
        help="Also store the full UTC timestamp in a 'Datetime' column (for market-close alignment).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned_df = clean_news_file(args.input, args.output, keep_timestamp=args.keep_timestamp)

    print("Cleaning completed.")
    print(f"Saved cleaned news to: {args.output}")
    print(f"Rows: {len(cleaned_df):,}")
    print(f"Columns: {list(cleaned_df.columns)}")


if __name__ == "__main__":
    main()