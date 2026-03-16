from __future__ import annotations

import math

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from greenfin.evaluate import evaluate_always_buy
from greenfin.metrics import get_returns_map

try:
    from greenfin.metrics import compute_pnl_sharpe
except ImportError:
    # If you left this helper in train.py instead of metrics.py
    from greenfin.train import compute_pnl_sharpe


class TinyEvalDataset(Dataset):
    def __init__(self):
        self.rows = [
            {"date": "2024-01-01", "label": 1.0},
            {"date": "2024-01-02", "label": 0.0},
            {"date": "2024-01-03", "label": 1.0},
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "dates": row["date"],
            "labels": torch.tensor([row["label"]], dtype=torch.float32),
        }


def test_get_returns_map_computes_next_day_returns(tmp_market_csv):
    returns_map = get_returns_map(str(tmp_market_csv))

    assert returns_map["2024-01-01"] == pytest.approx(0.10, rel=1e-6)
    assert returns_map["2024-01-02"] == pytest.approx((99.0 / 110.0) - 1.0, rel=1e-6)
    assert returns_map["2024-01-03"] == pytest.approx((120.0 / 99.0) - 1.0, rel=1e-6)

    # Last day should usually map to NaN because there is no next close
    assert math.isnan(returns_map["2024-01-04"])


def test_compute_pnl_sharpe_returns_expected_pnl():
    R = [0.01, -0.005, 0.02, 0.0]
    pnl, sharpe = compute_pnl_sharpe(R, rf_annual=0.02, tdays=252)

    assert pnl == pytest.approx(0.025, rel=1e-6)
    assert math.isfinite(sharpe)


def test_compute_pnl_sharpe_returns_nan_when_too_short():
    pnl, sharpe = compute_pnl_sharpe([0.01, -0.01], rf_annual=0.02, tdays=252)

    assert math.isnan(pnl)
    assert math.isnan(sharpe)


def test_evaluate_always_buy_on_tiny_loader():
    dataset = TinyEvalDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    returns_map = {
        "2024-01-01": 0.10,
        "2024-01-02": -0.05,
        "2024-01-03": 0.20,
    }

    acc, pnl, sharpe = evaluate_always_buy(
        loader,
        returns_map,
        rf_annual=0.02,
        tdays=252,
    )

    # Always buy predicts "up" every day, so accuracy is fraction of labels==1
    assert acc == pytest.approx(2 / 3, rel=1e-6)

    # PnL is the sum of raw market returns
    assert pnl == pytest.approx(0.10 - 0.05 + 0.20, rel=1e-6)

    # There are 3 returns, so sharpe should be finite
    assert math.isfinite(sharpe)
