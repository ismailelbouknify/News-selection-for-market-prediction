from __future__ import annotations

import math

import numpy as np
import pytest

from greenfin import trading as tr
from greenfin.metrics import compute_pnl_sharpe, legacy_finin_daily_returns, positions_from_predictions


def test_pnl_sign_logic_long_up_and_short_down_both_earn():
    r = np.array([0.01, -0.02])  # up day, down day
    pos = positions_from_predictions([1, 0])  # long, short
    np.testing.assert_allclose(pos * r, [0.01, 0.02])


def test_legacy_finin_metric_differs_from_tradable_metric():
    # Correct "down" call on a falling day: tradable strategy earns +2%, legacy books -2%.
    np.testing.assert_allclose(legacy_finin_daily_returns([0], [0], [-0.02]), [-0.02])
    np.testing.assert_allclose(positions_from_predictions([0]) * np.array([-0.02]), [0.02])


def test_headline_sharpe_matches_hand_computation():
    R = [0.01, -0.005, 0.02, 0.0]
    pnl, sr = compute_pnl_sharpe(R, rf_annual=0.02, tdays=252)
    arr = np.asarray(R)
    expected = (arr.mean() - 0.02 / 252) / arr.std(ddof=1) * math.sqrt(252)
    assert pnl == pytest.approx(0.025)
    assert sr == pytest.approx(expected)


def test_trading_sharpe_uses_excess_returns():
    r = np.array([0.01, 0.0, 0.02, -0.01])
    rf = np.full(4, 0.02 / 252)
    excess = r - rf
    assert tr.sharpe_ratio(r, rf) == pytest.approx(excess.mean() / excess.std(ddof=1) * math.sqrt(252))
    with pytest.raises(ValueError):
        tr.sharpe_ratio(r, rf[:3])


def test_compounded_return_and_wealth():
    r = [0.1, -0.1]
    assert tr.compounded_return(r) == pytest.approx(1.1 * 0.9 - 1)
    np.testing.assert_allclose(tr.wealth_series(r), [1.1, 0.99])


def test_max_drawdown():
    assert tr.max_drawdown([1.0, 1.2, 0.9, 1.3, 1.04]) == pytest.approx(0.9 / 1.2 - 1)
    assert tr.max_drawdown([1.0, 1.1, 1.2]) == 0.0


def test_transaction_costs_flip_costs_twice_an_entry():
    pos = [1, -1, -1, 0]
    costs = tr.transaction_costs(pos, bps=10)
    # entry from cash (1), flip (2), hold (0), exit to cash (1) units of turnover
    np.testing.assert_allclose(costs, np.array([1, 2, 0, 1]) * 10 / 10_000)
    np.testing.assert_allclose(tr.net_returns([0.01, 0.0, 0.0, 0.0], costs), [0.009, -0.002, 0.0, -0.001])
    with pytest.raises(ValueError):
        tr.transaction_costs(pos, bps=-1)


def test_turnover_and_position_changes_include_initial_entry():
    pos = [1, 1, -1, -1]
    assert tr.turnover(pos) == pytest.approx((1 + 0 + 2 + 0) / 4)
    assert tr.position_changes(pos) == 2
    # Turnover reconciles with the charged costs.
    assert tr.transaction_costs(pos, bps=10_000).mean() == pytest.approx(tr.turnover(pos))


def test_long_cash_earns_risk_free_when_out_of_market():
    pos = tr.positions_long_cash([1, 0, 1])
    np.testing.assert_allclose(pos, [1, 0, 1])
    np.testing.assert_allclose(tr.strategy_returns_long_cash(pos, [0.01, -0.03, 0.02], [0.001] * 3), [0.01, 0.001, 0.02])


def test_previous_day_direction_uses_only_realised_return():
    dates = ["d0", "d1", "d2", "d3"]
    returns_map = {"d0": 0.01, "d1": -0.02, "d2": 0.03, "d3": 0.5}
    # Position on d is the sign of returns_map[d-1] (the move realised ON d).
    np.testing.assert_allclose(tr.positions_previous_day_direction(["d1", "d2", "d3"], dates, returns_map), [1, -1, 1])
    with pytest.raises(ValueError):
        tr.positions_previous_day_direction(["d0"], dates, returns_map)


def test_buy_and_hold_metrics_and_cost_grid():
    r = np.array([0.01, -0.005, 0.002])
    pos = tr.positions_buy_and_hold(3)
    m0 = tr.compute_trading_metrics(pos, r, np.zeros(3), bps=0)
    m20 = tr.compute_trading_metrics(pos, r, np.zeros(3), bps=20)
    assert m0["gross_compounded_return"] == pytest.approx(np.prod(1 + r) - 1)
    assert m0["net_compounded_return"] == pytest.approx(m0["gross_compounded_return"])
    # Only the initial entry is charged for buy-and-hold.
    expected_net = (1 + r[0] - 0.002) * (1 + r[1]) * (1 + r[2]) - 1
    assert m20["net_compounded_return"] == pytest.approx(expected_net)
    assert m20["turnover"] == pytest.approx(1 / 3)
