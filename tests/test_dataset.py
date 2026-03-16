from __future__ import annotations

import pytest
import torch

from greenfin.dataset import GreenFinDataset, Sample, make_splits


def test_GreenFinDataset_sorts_by_date_and_returns_expected_item(tiny_records):
    ds = GreenFinDataset(tiny_records)

    assert len(ds) == 3
    assert ds.t == 2

    # GreenFinDataset sorts by date internally
    assert [s.date for s in ds.samples] == ["2024-01-01", "2024-01-02", "2024-01-03"]

    item = ds[0]
    assert item["date"] == "2024-01-01"
    assert isinstance(item["markets"], torch.Tensor)
    assert item["markets"].shape == (2, 5)
    assert item["label"].shape == (1,)
    assert item["headline_ids"] == [[5], []]
    assert item["sentiments"] == [[[0.70, 0.10, 0.20]], []]


def test_GreenFinDataset_accepts_sample_objects():
    samples = [
        Sample(
            date="2024-01-01",
            markets=[[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
            headline_ids=[[1], [2]],
            sentiments=[[[0.3, 0.3, 0.4]], [[0.2, 0.5, 0.3]]],
            label=1,
        ),
        Sample(
            date="2024-01-02",
            markets=[[2, 3, 4, 5, 6], [3, 4, 5, 6, 7]],
            headline_ids=[[3], [4]],
            sentiments=[[[0.1, 0.7, 0.2]], [[0.4, 0.4, 0.2]]],
            label=0,
        ),
    ]

    ds = GreenFinDataset(samples)
    assert len(ds) == 2
    assert ds.t == 2
    assert ds[1]["date"] == "2024-01-02"


def test_GreenFinDataset_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        GreenFinDataset([])


def test_GreenFinDataset_rejects_mixed_window_sizes():
    bad_records = [
        {
            "date": "2024-01-01",
            "markets": [[1, 2, 3, 4, 5]],
            "headline_ids": [[]],
            "sentiments": [[]],
            "label": 1,
        },
        {
            "date": "2024-01-02",
            "markets": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
            "headline_ids": [[], []],
            "sentiments": [[], []],
            "label": 0,
        },
    ]

    with pytest.raises(ValueError, match="Mixed window sizes"):
        GreenFinDataset(bad_records)


def test_GreenFinDataset_rejects_misaligned_headline_days():
    bad_records = [
        {
            "date": "2024-01-01",
            "markets": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
            "headline_ids": [[1]],   # only 1 day instead of 2
            "sentiments": [[[0.2, 0.3, 0.5]]],
            "label": 1,
        }
    ]

    with pytest.raises(AssertionError):
        GreenFinDataset(bad_records)


def test_make_splits_cover_all_indices_without_overlap(tiny_dataset):
    train_idx, val_idx, test_idx = make_splits(tiny_dataset)

    all_idx = train_idx + val_idx + test_idx
    assert sorted(all_idx) == list(range(len(tiny_dataset)))
    assert len(set(all_idx)) == len(all_idx)

    # Default split with n=3 should usually be train=2, val=0, test=1
    assert len(train_idx) + len(val_idx) + len(test_idx) == 3
