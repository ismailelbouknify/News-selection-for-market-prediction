from __future__ import annotations

import pytest

from greenfin.windows import build_fold_windows, fold_date_splits, split_indices


def test_paper_geometry_for_t5_dataset():
    # Input_t5.jsonl has 4,136 samples; the paper uses W=2770, step=340.
    bounds = build_fold_windows(4136, train_days=2430, step_days=340)
    assert bounds == [(0, 2770), (340, 3110), (680, 3450), (1020, 3790), (1360, 4130)]
    tr, va, te = split_indices(2770)
    assert (len(tr), len(va), len(te)) == (2216, 277, 277)


def test_split_is_chronological_and_disjoint():
    tr, va, te = split_indices(100)
    assert tr == list(range(0, 80)) and va == list(range(80, 90)) and te == list(range(90, 100))


def test_test_windows_are_contiguous_and_non_overlapping():
    dates = [f"d{i:05d}" for i in range(4136)]
    folds = fold_date_splits(dates)
    test_sets = [set(f["test_dates"]) for f in folds]
    for a in range(len(test_sets)):
        for b in range(a + 1, len(test_sets)):
            assert not test_sets[a] & test_sets[b]
    for f in folds:
        # Every training date precedes every validation date, which precedes every test date.
        assert max(f["train_dates"]) < min(f["val_dates"]) <= max(f["val_dates"]) < min(f["test_dates"])


def test_max_windows_and_too_small_input():
    assert len(build_fold_windows(1000, 60, 20, max_windows=3)) == 3
    with pytest.raises(ValueError):
        build_fold_windows(100, train_days=90, step_days=20)


def test_fold_date_splits_requires_sorted_dates():
    with pytest.raises(ValueError):
        fold_date_splits(["2020-01-03", "2020-01-02"] * 100, train_days=60, step_days=20)
