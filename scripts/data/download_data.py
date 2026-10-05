"""Download the preprocessed GreenFin data from Hugging Face.

Dataset: https://huggingface.co/datasets/ismail-ELBOUKNIFY/news-selection-for-market-prediction

Default download (into ``data/``)::

    raw/market/sp500.csv                    S&P 500 daily OHLCV (2007-07-23 .. 2023-12-29)
    processed/Input_t{1,3,5,10,20}.jsonl    look-back datasets (~31 GB in total)
    interim/headline_embeddings_fp16.pt     RoBERTa embeddings (~22 GB, needed by every news model)

Usage::

    python scripts/data/download_data.py                       # everything needed for the paper
    python scripts/data/download_data.py --lookbacks 5         # only T=5 (Tables 3, 4, 7, 8)
    python scripts/data/download_data.py --no-embeddings --lookbacks 5   # market-only baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "ismail-ELBOUKNIFY/news-selection-for-market-prediction"
LOOKBACKS = [1, 3, 5, 10, 20]
EMBEDDINGS = "interim/headline_embeddings_fp16.pt"
SENTIMENT_CSV = "interim/cleaned_news_sentiment.csv"
MARKET_CSV = "raw/market/sp500.csv"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-dir", type=Path, default=Path("data"), help="Local data folder (default: data).")
    p.add_argument("--lookbacks", type=int, nargs="+", default=LOOKBACKS, choices=LOOKBACKS,
                   help="Look-back datasets to download (default: all).")
    p.add_argument("--no-embeddings", action="store_true", help="Skip the ~22 GB embedding file.")
    p.add_argument("--with-sentiment-csv", action="store_true",
                   help=f"Also download {SENTIMENT_CSV} (~12 GB; only needed to rebuild the JSONL files).")
    p.add_argument("--revision", default=None, help="Optional dataset revision (branch, tag or commit).")
    p.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    args = p.parse_args()

    patterns = [MARKET_CSV] + [f"processed/Input_t{t}.jsonl" for t in args.lookbacks]
    if not args.no_embeddings:
        patterns.append(EMBEDDINGS)
    if args.with_sentiment_csv:
        patterns.append(SENTIMENT_CSV)

    args.target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {REPO_ID} into {args.target_dir.resolve()}:")
    for pattern in patterns:
        print(f"  - {pattern}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(args.target_dir),
        allow_patterns=patterns,
        revision=args.revision,
        force_download=args.force,
    )
    missing = [pt for pt in patterns if not (args.target_dir / pt).exists()]
    if missing:
        raise SystemExit(f"Download finished but these files are missing: {missing}")
    print("Download completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
