from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "ismail-ELBOUKNIFY/news-selection-for-market-prediction"
REPO_TYPE = "dataset"

# These match the files you currently have in your Hugging Face dataset repo.
DEFAULT_ALLOW_PATTERNS = [
    "processed/Input_t1.jsonl",
    "processed/Input_t3.jsonl",
    "processed/Input_t5.jsonl",
    "processed/Input_t10.jsonl",
    "processed/Input_t20.jsonl",
    "raw/market/sp500.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download project data from Hugging Face into a local folder."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data"),
        help="Local folder where the dataset files will be placed (default: data).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional branch, tag, or commit hash to download.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download the entire dataset repo instead of only the known project files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh download even if files already exist in the local cache.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    target_dir = args.target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = None if args.all_files else DEFAULT_ALLOW_PATTERNS

    print(f"Downloading dataset from: {REPO_ID}")
    print(f"Target directory: {target_dir}")
    if args.revision:
        print(f"Revision: {args.revision}")
    if allow_patterns is None:
        print("Mode: download all files from the dataset repo")
    else:
        print("Mode: download only the project data files")
        for pattern in allow_patterns:
            print(f"  - {pattern}")

    try:
        local_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=str(target_dir),
            allow_patterns=allow_patterns,
            revision=args.revision,
            force_download=args.force,
        )

        print("\nDownload completed successfully.")
        print(f"Snapshot path: {local_path}")
        print(f"Project data available under: {target_dir}")
        print("\nExpected local files:")
        if allow_patterns is None:
            print("  - Entire dataset repo downloaded")
        else:
            for pattern in allow_patterns:
                print(f"  - {target_dir / pattern}")

        return 0

    except Exception as exc:
        print("\nDownload failed.")
        print(f"Error: {exc}")
        print(
            "\nChecks:\n"
            "1. Verify internet access on this machine.\n"
            "2. Verify 'huggingface_hub' is installed.\n"
            "3. Verify the dataset repo and file paths still exist.\n"
            "4. If needed, try again with --all-files."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())