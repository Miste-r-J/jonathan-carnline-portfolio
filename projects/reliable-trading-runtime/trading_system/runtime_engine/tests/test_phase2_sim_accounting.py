from __future__ import annotations

import pandas as pd
import pytest

from trading_system.runtime_engine.modeling.phase2_sim import Phase2SimConfig, simulate_trades


def _config(*, commission: float = 2.0, slippage_ticks: float = 1.0) -> Phase2SimConfig:
    return Phase2SimConfig(
        tz="America/Denver",
        trade_window_start="00:00",
        trade_window_end="23:59",
        point_value=50.0,
        tick_value=12.5,
        contracts=1,
        max_hold_bars=100,
        commission_per_contract=commission,
        slippage_ticks=slippage_ticks,
    )


def _frame(prices: list[float], signals: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Datetime": pd.date_range("2026-06-20 09:00", periods=len(prices), freq="5min"),
            "Close": prices,
            "phase2_direction_signal": signals,
        }
    )


def test_single_trade_accounting_charges_costs_once_and_equity_reconciles() -> None:
    result = simulate_trades(
        _frame([100.0, 101.0, 101.0], [1, 1, 0]),
        {},
        cfg=_config(),
    )

    trade = result["trades"][0]
    assert trade["gross_pnl_usd"] == pytest.approx(50.0)
    assert trade["costs_usd"] == pytest.approx(29.0)
    assert trade["pnl_usd"] == pytest.approx(21.0)
    assert result["gross_pnl_usd"] == pytest.approx(50.0)
    assert result["total_costs_usd"] == pytest.approx(29.0)
    assert result["total_pnl_usd"] == pytest.approx(21.0)
    assert result["gross_pnl_usd"] - result["total_costs_usd"] == pytest.approx(
        result["total_pnl_usd"]
    )
    assert result["equity"][-1] == pytest.approx(result["total_pnl_usd"])
    assert result["max_drawdown"] == pytest.approx(14.5)


def test_long_short_metrics_reconcile_to_account_totals() -> None:
    result = simulate_trades(
        _frame([100.0, 101.0, 100.0, 100.0], [1, -1, -1, 0]),
        {},
        cfg=_config(commission=1.0, slippage_ticks=0.0),
    )

    long_metrics = result["side_metrics"]["long"]
    short_metrics = result["side_metrics"]["short"]
    assert long_metrics["trade_count"] == 1
    assert short_metrics["trade_count"] == 1
    assert long_metrics["gross_pnl_usd"] == pytest.approx(50.0)
    assert short_metrics["gross_pnl_usd"] == pytest.approx(50.0)
    assert long_metrics["costs_usd"] == pytest.approx(2.0)
    assert short_metrics["costs_usd"] == pytest.approx(2.0)
    assert long_metrics["realized_pnl_usd"] == pytest.approx(48.0)
    assert short_metrics["realized_pnl_usd"] == pytest.approx(48.0)

    assert (
        long_metrics["gross_pnl_usd"] + short_metrics["gross_pnl_usd"]
    ) == pytest.approx(result["gross_pnl_usd"])
    assert long_metrics["costs_usd"] + short_metrics["costs_usd"] == pytest.approx(
        result["total_costs_usd"]
    )
    assert (
        long_metrics["realized_pnl_usd"] + short_metrics["realized_pnl_usd"]
    ) == pytest.approx(result["total_pnl_usd"])
    assert sum(trade["pnl_usd"] for trade in result["trades"]) == pytest.approx(
        result["total_pnl_usd"]
    )
    assert result["equity"][-1] == pytest.approx(result["total_pnl_usd"])


def test_gap_flatten_uses_pre_gap_price_instead_of_reopen_price() -> None:
    frame = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                ["2026-02-27T14:55:00-07:00", "2026-02-27T15:00:00-07:00", "2026-03-01T16:05:00-07:00"],
                utc=True,
            ),
            "Close": [100.0, 101.0, 50.0],
            "phase2_direction_signal": [1, 1, 1],
        }
    )

    result = simulate_trades(frame, {}, cfg=_config())

    trade = result["trades"][0]
    assert trade["reason"] == "gap_flatten"
    assert trade["exit_price"] == pytest.approx(101.0)
    assert trade["exit_ts"].startswith("2026-02-27")
    assert trade["pnl_usd"] == pytest.approx(21.0)
