from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greenfin import bootstrap as bs
from greenfin.baselines import fit_ar, predict_ar, select_ar_lag


# --------------------------------------------------------------------------- AR
def test_ar_recovers_coefficients():
    rng = np.random.default_rng(0)
    r = np.zeros(5000)
    for t in range(1, len(r)):
        r[t] = 0.001 + 0.3 * r[t - 1] + rng.normal(0, 0.01)
    coefs = fit_ar(r, lag=1)
    assert coefs[0] == pytest.approx(0.001, abs=5e-4)
    assert coefs[1] == pytest.approx(0.3, abs=0.05)


def test_ar_forecast_uses_only_past_values():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 50)
    coefs = fit_ar(r[:30], lag=3)
    before = predict_ar(r, coefs, 3, [35, 40])
    r_future_changed = r.copy()
    r_future_changed[40:] = 99.0  # values at/after the forecast index must not matter for index 40
    after = predict_ar(r_future_changed, coefs, 3, [35, 40])
    np.testing.assert_allclose(before, after)


def test_ar_lag_selected_on_validation_sharpe_then_accuracy():
    val = pd.DataFrame({
        "lag": [1, 1, 2, 2, 3, 3],
        "window": [1, 2, 1, 2, 1, 2],
        "sharpe_ratio": [0.1, 0.3, 0.5, 0.1, 0.5, 0.1],
        "directional_accuracy": [0.5, 0.5, 0.51, 0.51, 0.52, 0.52],
    })
    assert select_ar_lag(val) == 3


# --------------------------------------------------------------------------- bootstrap
def _cells(n_per_window, seeds, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    cells = {}
    for w, n in n_per_window.items():
        dates = np.array([f"w{w}-{i:03d}" for i in range(n)])
        y = rng.integers(0, 2, n).astype(float)
        for s in seeds:
            pred = np.where(rng.random(n) < 0.6, y, 1 - y)
            ret = rng.normal(shift, 0.01, n)
            cells[(s, w)] = bs.Cell(dates, y, pred, np.where(pred >= 1, 1.0, -1.0) * ret)
    return cells


def test_moving_block_indices_are_contiguous_blocks_inside_range():
    rng = np.random.default_rng(0)
    idx = bs.moving_block_indices(23, 5, rng)
    assert len(idx) == 23 and idx.min() >= 0 and idx.max() < 23
    for start in range(0, 20, 5):
        block = idx[start : start + 5]
        np.testing.assert_array_equal(np.diff(block), np.ones(len(block) - 1))


def test_blocks_never_cross_window_boundaries(monkeypatch):
    lengths = {1: 12, 2: 7, 3: 9}
    cells = _cells(lengths, seeds=[0, 1])
    calls = []
    real = bs.moving_block_indices

    def spy(n, block_length, rng):
        idx = real(n, block_length, rng)
        calls.append((n, int(idx.max())))
        return idx

    monkeypatch.setattr(bs, "moving_block_indices", spy)
    bs.paired_block_bootstrap(cells, cells, [0, 1], [0, 1], [1, 2, 3], block_length=5, n_boot=20, rng_seed=1)
    # One draw per window per replicate, sized to that window, indices inside it.
    assert [n for n, _ in calls] == [12, 7, 9] * 20
    assert all(mx < n for n, mx in calls)


def test_identical_models_have_zero_delta_and_p_one():
    cells = _cells({1: 30, 2: 30}, seeds=[0])
    res = bs.paired_block_bootstrap(cells, cells, [0], [0], [1, 2], block_length=5, n_boot=200, rng_seed=3)
    for r in res.values():
        assert r.observed_delta == 0.0 and r.ci_low == 0.0 and r.ci_high == 0.0 and r.p_value == 1.0


def test_bootstrap_is_deterministic_and_detects_large_difference():
    a = _cells({1: 200, 2: 200}, seeds=[0, 1], seed=5)
    b = {k: bs.Cell(c.dates, c.y_true, c.predicted_class, c.strategy_return - 0.01) for k, c in a.items()}
    r1 = bs.paired_block_bootstrap(a, b, [0, 1], [0, 1], [1, 2], n_boot=300, rng_seed=7)
    r2 = bs.paired_block_bootstrap(a, b, [0, 1], [0, 1], [1, 2], n_boot=300, rng_seed=7)
    assert r1["pnl"].p_value == r2["pnl"].p_value and r1["pnl"].ci_low == r2["pnl"].ci_low
    assert r1["pnl"].observed_delta == pytest.approx(2.0)  # 0.01 * 200 days, averaged over windows
    assert r1["pnl"].p_value < 0.01 and r1["pnl"].ci_low > 0


def test_unmatched_test_dates_are_rejected():
    a = _cells({1: 10}, seeds=[0])
    b = {k: bs.Cell(np.array([d + "x" for d in c.dates]), c.y_true, c.predicted_class, c.strategy_return)
         for k, c in a.items()}
    with pytest.raises(ValueError):
        bs.paired_block_bootstrap(a, b, [0], [0], [1], n_boot=5)
    with pytest.raises(KeyError):
        bs.paired_block_bootstrap(a, a, [0], [9], [1], n_boot=5)


def test_cells_from_predictions_table():
    df = pd.DataFrame({
        "date": ["2020-01-02", "2020-01-01", "2020-01-02"],
        "window": [1, 1, 1], "seed": [0, 0, 1],
        "y_true": [1, 0, 1], "predicted_class": [1, 1, 0], "strategy_return": [0.01, -0.02, -0.01],
    })
    cells = bs.cells_from_predictions(df)
    assert set(cells) == {(0, 1), (1, 1)}
    np.testing.assert_array_equal(cells[(0, 1)].dates, ["2020-01-01", "2020-01-02"])
