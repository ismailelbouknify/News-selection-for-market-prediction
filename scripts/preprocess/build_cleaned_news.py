from __future__ import annotations

import argparse
import os
from datetime import date
from typing import Iterable

import pandas as pd


KEEP_COLUMNS = ["Date", "Article_title", "Article", "Lsa_summary"]
CYRILLIC_PATTERN = r"[А-Яа-яЁё]"
MIN_DATE = date(1999, 1, 1)
DEFAULT_OUTPUT_PATH = "Cleaned_dataset/cleaned_news.csv"


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def clean_news_dataframe(
    df: pd.DataFrame,
    output_path: str = DEFAULT_OUTPUT_PATH,
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
    df1["Date"] = pd.to_datetime(df1["Date"], errors="coerce").dt.date
    df1 = df1.dropna(subset=["Date"])

    # Remove exact duplicate rows
    df1 = df1.drop_duplicates().reset_index(drop=True)

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
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    return clean_news_dataframe(df, output_path=output_path)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned_df = clean_news_file(args.input, args.output)

    print("Cleaning completed.")
    print(f"Saved cleaned news to: {args.output}")
    print(f"Rows: {len(cleaned_df):,}")
    print(f"Columns: {list(cleaned_df.columns)}")


if __name__ == "__main__":
    main()