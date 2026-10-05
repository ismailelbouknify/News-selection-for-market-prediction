"""Paired moving-block bootstrap for out-of-sample model comparisons.

Ported from the statistical-validation analysis of the paper. For two models
A and B evaluated on the *same* out-of-sample test days:

* every replicate draws, independently for each test window, one set of
  moving-block indices (block length 5 trading days by default) inside that
  window only -- blocks never cross a window boundary;
* the identical index draw is applied to every seed of A and every seed of B
  for that window, which keeps the comparison paired;
* per replicate the metric is aggregated exactly like the headline results
  (mean over windows within a seed, then mean over seeds) and the difference
  ``Delta = A - B`` is recorded;
* the 95% CI is the 2.5/97.5 percentile interval of the replicate deltas and
  the two-sided p-value is ``min(1, 2 * min(P(Delta* <= 0), P(Delta* >= 0)))``
  with the ``(count + 1) / (n + 1)`` finite-sample correction.

Metrics follow :mod:`greenfin.metrics` (accuracy, summed PnL, annualised
Sharpe with a 2% annual risk-free rate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import compute_pnl_sharpe

METRICS = ("accuracy", "pnl", "sharpe")
REQUIRED_COLUMNS = ("date", "window", "seed", "y_true", "predicted_class", "strategy_return")


def compute_metrics(
    y_true: np.ndarray,
    predicted_class: np.ndarray,
    strategy_return: np.ndarray,
    rf_annual: float = 0.02,
    tdays: int = 252,
) -> Dict[str, float]:
    pnl, sharpe = compute_pnl_sharpe(list(strategy_return), rf_annual, tdays)
    return {"accuracy": float(np.mean(y_true == predicted_class)), "pnl": pnl, "sharpe": sharpe}


@dataclass
class Cell:
    """Daily test-day arrays of one (seed, window), sorted by date."""

    dates: np.ndarray
    y_true: np.ndarray
    predicted_class: np.ndarray
    strategy_return: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Cell":
        d = df.sort_values("date")
        if d["date"].duplicated().any():
            raise ValueError("Duplicate dates inside one (seed, window) cell.")
        return cls(
            dates=d["date"].astype(str).to_numpy(),
            y_true=d["y_true"].to_numpy(dtype=np.float64),
            predicted_class=d["predicted_class"].to_numpy(dtype=np.float64),
            strategy_return=d["strategy_return"].to_numpy(dtype=np.float64),
        )

    def __len__(self) -> int:
        return len(self.dates)

    def metrics(self, idx: np.ndarray | None = None) -> Dict[str, float]:
        sl = slice(None) if idx is None else idx
        return compute_metrics(self.y_true[sl], self.predicted_class[sl], self.strategy_return[sl])


def cells_from_predictions(df: pd.DataFrame) -> Dict[Tuple[int, int], Cell]:
    """Group a predictions table (see ``scripts/experiments/run_experiment.py``) into cells."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Predictions table is missing columns {sorted(missing)}")
    return {(int(s), int(w)): Cell.from_frame(g) for (s, w), g in df.groupby(["seed", "window"])}


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Non-circular moving-block bootstrap indices in ``[0, n)``."""
    if n <= 0:
        raise ValueError("Cannot resample an empty window.")
    L = max(1, min(int(block_length), n))
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + L) for s in starts])[:n]


def _aggregate(per_seed_window: Dict[int, Dict[int, Dict[str, float]]]) -> Dict[str, float]:
    """Mean over windows within each seed, then mean over seeds."""
    seed_means = [
        {m: float(np.mean([v[m] for v in per_window.values()])) for m in METRICS}
        for per_window in per_seed_window.values()
    ]
    return {m: float(np.mean([s[m] for s in seed_means])) for m in METRICS}


def check_matched(
    cells_a: Dict[Tuple[int, int], Cell],
    cells_b: Dict[Tuple[int, int], Cell],
    seeds_a: Iterable[int],
    seeds_b: Iterable[int],
    windows: Iterable[int],
) -> None:
    """Every compared cell must cover exactly the same test dates."""
    for w in windows:
        reference = None
        for tag, cells, seeds in (("A", cells_a, seeds_a), ("B", cells_b, seeds_b)):
            for s in seeds:
                if (s, w) not in cells:
                    raise KeyError(f"Model {tag} has no predictions for seed={s}, window={w}.")
                dates = cells[(s, w)].dates
                if reference is None:
                    reference = dates
                elif not np.array_equal(reference, dates):
                    raise ValueError(f"Test dates differ between cells in window {w} (model {tag}, seed {s}).")


@dataclass
class BootstrapResult:
    metric: str
    observed_a: float
    observed_b: float
    observed_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_boot: int
    block_length: int
    windows: List[int]
    seeds_a: List[int]
    seeds_b: List[int]
    rng_seed: int


def paired_block_bootstrap(
    cells_a: Dict[Tuple[int, int], Cell],
    cells_b: Dict[Tuple[int, int], Cell],
    seeds_a: Sequence[int],
    seeds_b: Sequence[int],
    windows: Sequence[int],
    block_length: int = 5,
    n_boot: int = 10_000,
    rng_seed: int = 20260908,
) -> Dict[str, BootstrapResult]:
    """Paired, window-respecting moving-block bootstrap of ``A - B`` (see module docstring)."""
    seeds_a, seeds_b, windows = list(seeds_a), list(seeds_b), list(windows)
    if not (seeds_a and seeds_b and windows):
        raise ValueError("seeds_a, seeds_b and windows must all be non-empty.")
    check_matched(cells_a, cells_b, seeds_a, seeds_b, windows)
    rng = np.random.default_rng(rng_seed)

    obs_a = _aggregate({s: {w: cells_a[(s, w)].metrics() for w in windows} for s in seeds_a})
    obs_b = _aggregate({s: {w: cells_b[(s, w)].metrics() for w in windows} for s in seeds_b})

    deltas = {m: np.empty(n_boot, dtype=np.float64) for m in METRICS}
    for rep in range(n_boot):
        rep_a: Dict[int, Dict[int, Dict[str, float]]] = {s: {} for s in seeds_a}
        rep_b: Dict[int, Dict[int, Dict[str, float]]] = {s: {} for s in seeds_b}
        for w in windows:
            # One draw per window, shared by every seed of both models; indices
            # stay inside this window, so no block crosses a window boundary.
            idx = moving_block_indices(len(cells_a[(seeds_a[0], w)]), block_length, rng)
            for s in seeds_a:
                rep_a[s][w] = cells_a[(s, w)].metrics(idx)
            for s in seeds_b:
                rep_b[s][w] = cells_b[(s, w)].metrics(idx)
        agg_a, agg_b = _aggregate(rep_a), _aggregate(rep_b)
        for m in METRICS:
            deltas[m][rep] = agg_a[m] - agg_b[m]

    results: Dict[str, BootstrapResult] = {}
    for m in METRICS:
        d = deltas[m][~np.isnan(deltas[m])]
        if d.size == 0:
            raise ValueError(f"All bootstrap replicates are NaN for metric {m!r}.")
        ci_low, ci_high = np.percentile(d, [2.5, 97.5])
        p_le = (np.sum(d <= 0) + 1) / (d.size + 1)
        p_ge = (np.sum(d >= 0) + 1) / (d.size + 1)
        results[m] = BootstrapResult(
            metric=m,
            observed_a=obs_a[m],
            observed_b=obs_b[m],
            observed_delta=obs_a[m] - obs_b[m],
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            p_value=float(min(1.0, 2.0 * min(p_le, p_ge))),
            n_boot=int(n_boot),
            block_length=int(block_length),
            windows=windows,
            seeds_a=seeds_a,
            seeds_b=seeds_b,
            rng_seed=int(rng_seed),
        )
    return results
