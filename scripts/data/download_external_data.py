"""Download the raw FNSPID news file (~23 GB) needed to rebuild the dataset from scratch.

Source: https://huggingface.co/datasets/Zihan1004/FNSPID (file
``Stock_news/nasdaq_exteral_data.csv`` -- the misspelling is upstream). Please
follow the FNSPID license and citation requirements.

Usage::

    python scripts/data/download_external_data.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

SOURCE_REPO_ID = "Zihan1004/FNSPID"
SOURCE_FILENAME = "Stock_news/nasdaq_exteral_data.csv"
DEFAULT_TARGET = Path("data/raw/news/nasdaq_external_data.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Local file path (default: %(default)s)")
    p.add_argument("--revision", default=None, help="Optional dataset revision (branch, tag or commit).")
    args = p.parse_args()

    download_dir = args.target.parent / ".hf_download"
    print(f"Downloading {SOURCE_REPO_ID}:{SOURCE_FILENAME} ...")
    # local_dir avoids keeping a second 23 GB copy in the Hugging Face cache.
    path = Path(
        hf_hub_download(
            repo_id=SOURCE_REPO_ID,
            repo_type="dataset",
            filename=SOURCE_FILENAME,
            revision=args.revision,
            local_dir=str(download_dir),
        )
    )
    args.target.parent.mkdir(parents=True, exist_ok=True)
    path.replace(args.target)
    print(f"Saved to {args.target.resolve()} ({args.target.stat().st_size / 1024**3:.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
