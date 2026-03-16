from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


# Source dataset on Hugging Face
SOURCE_REPO_ID = "Zihan1004/FNSPID"
SOURCE_REPO_TYPE = "dataset"

# https://huggingface.co/datasets/Zihan1004/FNSPID
SOURCE_FILENAME = "data/raw/news/nasdaq_external_data.csv"

# Where to save it in your local project
LOCAL_TARGET = Path("data/raw/news/nasdaq_external_data.csv")


def main() -> int:
    try:
        print(f"Downloading '{SOURCE_FILENAME}' from '{SOURCE_REPO_ID}'...")

        cached_path = hf_hub_download(
            repo_id=SOURCE_REPO_ID,
            repo_type=SOURCE_REPO_TYPE,
            filename=SOURCE_FILENAME,
        )

        LOCAL_TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, LOCAL_TARGET)

        size_mb = LOCAL_TARGET.stat().st_size / (1024 * 1024)

        print("Done.")
        print(f"Saved to: {LOCAL_TARGET.resolve()}")
        print(f"Size: {size_mb:.2f} MB")
        return 0

    except Exception as e:
        print("Download failed.")
        print(f"Error: {e}")
        print(
            "\nCheck these things:\n"
            "1. The dataset repo exists and is public.\n"
            "2. The file path inside the dataset is correct.\n"
            "3. You have internet access from the machine.\n"
            "4. 'huggingface_hub' is installed."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())