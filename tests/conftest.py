from __future__ import annotations

import pandas as pd
import pytest
import torch

from greenfin.dataset import GreenFinDataset


@pytest.fixture
def tiny_records():
    """
    Three tiny chronological samples with t=2.
    Dates are intentionally unsorted so we can verify GreenFinDataset sorting.
    """
    return [
        {
            "date": "2024-01-03",
            "markets": [
                [1.0, 2.0, 0.5, 1.5, 100.0],
                [1.1, 2.1, 0.6, 1.6, 110.0],
            ],
            "headline_ids": [[1, 2, 3], [4]],
            "sentiments": [
                [[0.90, 0.05, 0.05], [0.34, 0.33, 0.33], [0.10, 0.80, 0.10]],
                [[0.20, 0.20, 0.60]],
            ],
            "label": 1,
        },
        {
            "date": "2024-01-01",
            "markets": [
                [0.8, 1.8, 0.4, 1.2, 80.0],
                [0.9, 1.9, 0.5, 1.3, 90.0],
            ],
            "headline_ids": [[5], []],
            "sentiments": [
                [[0.70, 0.10, 0.20]],
                [],
            ],
            "label": 0,
        },
        {
            "date": "2024-01-02",
            "markets": [
                [0.95, 1.95, 0.45, 1.25, 95.0],
                [1.0, 2.0, 0.5, 1.4, 105.0],
            ],
            "headline_ids": [[6, 7], [8, 9, 10]],
            "sentiments": [
                [[0.60, 0.20, 0.20], [0.40, 0.30, 0.30]],
                [[0.10, 0.10, 0.80], [0.33, 0.33, 0.34], [0.85, 0.10, 0.05]],
            ],
            "label": 1,
        },
    ]


@pytest.fixture
def tiny_dataset(tiny_records):
    return GreenFinDataset(tiny_records)


@pytest.fixture
def tiny_emb_dict():
    """
    Small deterministic embedding dictionary with 8-dimensional vectors.
    Enough for selection tests and forward-pass tests.
    """
    out = {}
    for hid in range(1, 21):
        base = torch.arange(8, dtype=torch.float32)
        out[hid] = base + float(hid)
    return out


@pytest.fixture
def tmp_market_csv(tmp_path):
    """
    Tiny market file for get_returns_map tests.
    Close series:
      2024-01-01 -> 100
      2024-01-02 -> 110
      2024-01-03 -> 99
      2024-01-04 -> 120
    """
    path = tmp_path / "market.csv"
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "Close": [100.0, 110.0, 99.0, 120.0],
        }
    )
    df.to_csv(path, index=False)
    return path
