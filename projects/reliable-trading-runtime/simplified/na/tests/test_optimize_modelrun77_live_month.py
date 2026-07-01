from __future__ import annotations

from tools.optimize_modelrun77_live_month import _apply_risk_rails, _metrics


def test_risk_rails_cap_trade_and_stop_day() -> None:
    trades = [
        {"entry_ts": "2026-06-01T10:00:00-06:00", "pnl_usd": -900.0, "costs_usd": 29.0},
        {"entry_ts": "2026-06-01T11:00:00-06:00", "pnl_usd": -100.0, "costs_usd": 29.0},
        {"entry_ts": "2026-06-01T12:00:00-06:00", "pnl_usd": 500.0, "costs_usd": 29.0},
    ]

    accepted = _apply_risk_rails(
        trades,
        max_risk_per_trade=400.0,
        daily_loss_stop=500.0,
        max_trades_per_day=6,
        max_losses_per_day=2,
    )

    assert [row["pnl_usd"] for row in accepted] == [-429.0, -100.0]
    assert _metrics(accepted)["worst_day_usd"] == -529.0


def test_metrics_keep_contract_segments_separate() -> None:
    metrics = _metrics(
        [
            {"entry_ts": "2026-06-01T10:00:00-06:00", "pnl_usd": 100.0, "_segment": "june"},
            {"entry_ts": "2026-06-02T10:00:00-06:00", "pnl_usd": -25.0, "_segment": "september"},
        ]
    )

    assert metrics["segment_pnl"] == {"june": 100.0, "september": -25.0}
