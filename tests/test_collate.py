from __future__ import annotations

import pytest
import torch

from greenfin.collate import collate_fn
from greenfin.dataset import GreenFinDataset


def test_collate_without_news_returns_only_markets_and_labels(tiny_records, tiny_emb_dict):
    ds = GreenFinDataset(tiny_records)
    batch = [ds[0], ds[1]]

    out = collate_fn(
        batch=batch,
        emb_dict=tiny_emb_dict,
        cap_per_day=None,
        use_news=False,
        selection_mode="kmeans",
    )

    assert set(out.keys()) == {"dates", "markets", "labels"}
    assert out["markets"].shape == (2, 2, 5)
    assert out["labels"].shape == (2, 1)


def test_collate_with_news_builds_expected_shapes(tiny_records, tiny_emb_dict):
    ds = GreenFinDataset(tiny_records)
    batch = [ds[0], ds[1]]

    out = collate_fn(
        batch=batch,
        emb_dict=tiny_emb_dict,
        cap_per_day=None,
        use_news=True,
        selection_mode="kmeans",
    )

    assert "news_emb" in out
    assert "sentiments" in out
    assert "pad_mask" in out

    assert out["markets"].shape == (2, 2, 5)
    assert out["news_emb"].ndim == 4
    assert out["sentiments"].ndim == 4
    assert out["pad_mask"].ndim == 3

    # B=2, t=2, e_in=8
    assert out["news_emb"].shape[0] == 2
    assert out["news_emb"].shape[1] == 2
    assert out["news_emb"].shape[-1] == 8

    # Some non-pad positions should exist
    assert (~out["pad_mask"]).sum().item() > 0


def test_collate_handles_batch_with_no_news_days(tiny_emb_dict):
    records = [
        {
            "date": "2024-01-01",
            "markets": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
            "headline_ids": [[], []],
            "sentiments": [[], []],
            "label": 1,
        },
        {
            "date": "2024-01-02",
            "markets": [[2, 3, 4, 5, 6], [3, 4, 5, 6, 7]],
            "headline_ids": [[], []],
            "sentiments": [[], []],
            "label": 0,
        },
    ]
    ds = GreenFinDataset(records)
    batch = [ds[0], ds[1]]

    out = collate_fn(
        batch=batch,
        emb_dict=tiny_emb_dict,
        cap_per_day=None,
        use_news=True,
        selection_mode="kmeans",
    )

    # If Hmax == 0, collate returns only dates, markets, labels
    assert set(out.keys()) == {"dates", "markets", "labels"}
    assert out["markets"].shape == (2, 2, 5)
    assert out["labels"].shape == (2, 1)


def test_collate_rejects_cap_per_day_argument_when_selection_is_offline(tiny_records, tiny_emb_dict):
    ds = GreenFinDataset(tiny_records)
    batch = [ds[0], ds[1]]

    with pytest.raises(ValueError, match="must NOT cap"):
        collate_fn(
            batch=batch,
            emb_dict=tiny_emb_dict,
            cap_per_day=2,
            use_news=True,
            selection_mode="kmeans",
        )
