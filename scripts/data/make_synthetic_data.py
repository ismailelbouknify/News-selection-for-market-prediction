#!/usr/bin/env python
"""Create a tiny synthetic dataset with the same file layout as the real data.

Used for CPU smoke tests of the full pipeline (no FNSPID download, no
transformer models). Writes, under ``--root``::

    data/raw/market/sp500.csv
    data/interim/headline_embeddings_fp16.pt
    data/processed/Input_t{1,3,5,10,20}.jsonl

The numbers are random and carry no financial meaning.

Usage::

    python scripts/data/make_synthetic_data.py --root outputs/smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs/smoke", help="Output root directory.")
    p.add_argument("--n-days", type=int, default=600, help="Number of trading days.")
    p.add_argument("--emb-dim", type=int, default=16, help="Embedding dimension (RoBERTa uses 768).")
    p.add_argument("--max-headlines", type=int, default=25, help="Maximum headlines per day.")
    p.add_argument("--lookbacks", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    root = Path(args.root)
    dates = pd.bdate_range("2020-01-02", periods=args.n_days)
    close = 3000.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, args.n_days)))
    open_ = close * (1 + rng.normal(0, 0.002, args.n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, args.n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, args.n_days)))
    volume = rng.integers(2_000_000_000, 5_000_000_000, args.n_days).astype(float)
    market = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Close": close, "High": high,
                           "Low": low, "Open": open_, "Volume": volume})
    (root / "data/raw/market").mkdir(parents=True, exist_ok=True)
    market.to_csv(root / "data/raw/market/sp500.csv", index=False)

    # Headlines per day with embeddings and FinBERT-style probabilities
    # (order: positive, negative, neutral -- see README "Dataset format").
    emb_dict = {}
    day_ids, day_sents = [], []
    next_id = 1
    for _ in range(args.n_days):
        h = int(rng.integers(0, args.max_headlines + 1))
        ids = list(range(next_id, next_id + h))
        next_id += h
        logits = rng.normal(0, 1.5, (h, 3))
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        for hid in ids:
            emb_dict[hid] = torch.from_numpy(rng.normal(0, 1, args.emb_dim).astype(np.float32)).half()
        day_ids.append(ids)
        day_sents.append(probs.round(6).tolist())
    (root / "data/interim").mkdir(parents=True, exist_ok=True)
    torch.save(emb_dict, root / "data/interim/headline_embeddings_fp16.pt")

    # Label of day i: Close_{i+1} > Close_i; the last day has no label and is dropped.
    labels = (market["Close"].shift(-1) > market["Close"]).astype(int).to_numpy()
    feats = market[["Open", "High", "Low", "Close", "Volume"]].to_numpy().tolist()
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    for t in args.lookbacks:
        path = root / f"data/processed/Input_t{t}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in range(t - 1, args.n_days - 1):
                f.write(json.dumps({
                    "date": market["Date"].iloc[i],
                    "markets": feats[i - t + 1 : i + 1],
                    "headline_ids": day_ids[i - t + 1 : i + 1],
                    "sentiments": day_sents[i - t + 1 : i + 1],
                    "label": int(labels[i]),
                }) + "\n")
    print(f"Synthetic data written under {root}/data ({args.n_days} days, {next_id - 1} headlines).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
