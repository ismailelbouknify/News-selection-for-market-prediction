"""Sliding-window time-series cross-validation geometry.

This is the single source of truth for the evaluation windows used by every
experiment in the paper (neural models in :mod:`greenfin.cv` and the
non-neural baselines in ``scripts/evaluation/practical_trading.py``), so that
all of them are scored on exactly the same out-of-sample days.

Protocol (paper defaults):

* window length ``W = train_days + step_days = 2430 + 340 = 2770`` samples,
* windows start every ``step_days = 340`` samples,
* inside each window: chronological 80% / 10% / 10% train / validation / test
  split (no shuffling across the split boundaries),
* for the T=5 dataset (4,136 samples) this yields exactly 5 windows with
  2,216 / 277 / 277 train / validation / test samples each.

Samples are indexed in chronological order, so a window never uses future
observations for training relative to its own validation and test days.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_TRAIN_DAYS = 2430
DEFAULT_STEP_DAYS = 340
DEFAULT_TRAIN_FRAC = 0.8
DEFAULT_VAL_FRAC = 0.1


def split_indices(
    n: int,
    train: float = DEFAULT_TRAIN_FRAC,
    val: float = DEFAULT_VAL_FRAC,
) -> Tuple[List[int], List[int], List[int]]:
    """Chronological train/validation/test indices for a window of ``n`` samples.

    Uses ``int()`` truncation for the train and validation sizes; the test
    split receives the remainder (277 samples for ``n = 2770``).
    """
    if n <= 0:
        raise ValueError(f"split_indices needs n > 0, got n={n}.")
    if not (0.0 < train < 1.0 and 0.0 <= val < 1.0 and train + val < 1.0):
        raise ValueError(f"Invalid split fractions train={train}, val={val}.")
    n_train = int(n * train)
    n_val = int(n * val)
    return (
        list(range(0, n_train)),
        list(range(n_train, n_train + n_val)),
        list(range(n_train + n_val, n)),
    )


def build_fold_windows(
    n_samples: int,
    train_days: int = DEFAULT_TRAIN_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    max_windows: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Half-open ``(lo, hi)`` sample-index bounds of every sliding window."""
    if train_days <= 0 or step_days <= 0:
        raise ValueError(f"train_days and step_days must be positive, got {train_days}, {step_days}.")
    window_size = train_days + step_days
    if n_samples < window_size:
        raise ValueError(
            f"Need at least window_size={window_size} samples for one window, got n_samples={n_samples}."
        )
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start + window_size <= n_samples and (max_windows is None or len(bounds) < max_windows):
        bounds.append((start, start + window_size))
        start += step_days
    return bounds


def fold_date_splits(
    dates: Sequence[str],
    train_days: int = DEFAULT_TRAIN_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    max_windows: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Train/validation/test date lists for every window.

    ``dates`` must be the chronologically sorted sample dates of the dataset
    (as produced by :class:`greenfin.dataset.GreenFinDataset`).
    """
    dates = list(dates)
    if dates != sorted(dates):
        raise ValueError("fold_date_splits expects chronologically sorted dates.")
    out: List[Dict[str, object]] = []
    for window, (lo, hi) in enumerate(build_fold_windows(len(dates), train_days, step_days, max_windows), start=1):
        window_dates = dates[lo:hi]
        idx_tr, idx_va, idx_te = split_indices(len(window_dates))
        out.append(
            {
                "window": window,
                "window_lo": lo,
                "window_hi": hi,
                "window_dates": window_dates,
                "train_dates": [window_dates[i] for i in idx_tr],
                "val_dates": [window_dates[i] for i in idx_va],
                "test_dates": [window_dates[i] for i in idx_te],
            }
        )
    return out
