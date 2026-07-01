from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trading_system.runtime_engine.modeling.config import PRESETS
import trading_system.runtime_engine.integrations.cli.live_trading_runtime as live_trading_runtime_module
from trading_system.runtime_engine.integrations.cli.live_trading_runtime import LiveCSVStreamer, StreamState


def _make_streamer() -> LiveCSVStreamer:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.state = StreamState()
    streamer.instrument = SimpleNamespace(alias="ES", point_value=50.0, tick_size=0.25)
    streamer.phase2_enabled = True
    streamer.phase2_close_enabled = True
    streamer.phase2_close_model = object()
    streamer.phase2_close_meta = {}
    streamer.phase2_close_model_path = "close.joblib"
    streamer.phase2_close_calibrator = None
    streamer.phase2_close_expected_features = ["feature_a"]
    streamer.phase2_close_threshold = 0.85
    streamer.allow_feature_mismatch = True
    streamer.pnl_overlay_enabled = True
    streamer.pnl_giveback_activate_r = 1.0
    streamer.pnl_giveback_close_r = 0.40
    streamer.pnl_stall_bars = 4
    streamer.pnl_stall_min_mfe_r = 0.25
    streamer.pnl_stall_close_below_r = 0.0
    streamer.pnl_runner_enabled = False
    streamer.pnl_runner_ignore_close_before_arm = True
    streamer._last_phase2_close_prob = None
    streamer._last_prob = 0.70
    streamer._last_close = None
    streamer._bars_in_trade = 6
    streamer._entry_time = None
    streamer._entry_price = None
    streamer._pos_side = ""
    streamer._trend_direction = "up"
    streamer._pos = 1
    streamer._log_exec_event_rows = []
    streamer._log_exec_event = lambda payload: streamer._log_exec_event_rows.append(payload)
    streamer._ensure_compat_defaults = lambda: None
    return streamer


def test_prop_safe_close_hotfix_preset_values() -> None:
    cfg = PRESETS["es_maxpack_10_full_send_prop_safe_pnl"]
    assert float(cfg["phase2_close_threshold"]) == pytest.approx(0.85)
    assert float(cfg["pnl_giveback_activate_r"]) == pytest.approx(1.0)
    assert float(cfg["pnl_giveback_close_r"]) == pytest.approx(0.40)
    assert int(cfg["pnl_stall_bars"]) == 4
    assert float(cfg["pnl_stall_close_below_r"]) == pytest.approx(0.0)


def test_phase2_close_requires_085_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    streamer = _make_streamer()
    streamer._pos = 1
    streamer._open_trade = {
        "side": "LONG",
        "entry_price": 100.0,
        "planned_stop": 95.0,
        "planned_target": 110.0,
        "contracts": 1.0,
    }

    monkeypatch.setattr(live_trading_runtime_module, "_predict_proba_safely", lambda *args, **kwargs: [0.84])
    row = pd.Series(
        {
            "Datetime": pd.Timestamp("2026-04-21T11:00:00-06:00"),
            "Close": 101.0,
            "High": 101.5,
            "Low": 100.5,
            "feature_a": 1.0,
        }
    )

    event = streamer._maybe_phase2_close_event(row, pd.Timestamp("2026-04-21T11:00:00-06:00"))

    assert event is None
    assert bool(row["phase2_close_signal"]) is False
    assert float(row["phase2_close_prob"]) == pytest.approx(0.84)


def test_phase2_close_suppressed_when_profit_overlay_is_managing_trade(monkeypatch: pytest.MonkeyPatch) -> None:
    streamer = _make_streamer()
    streamer._pos = 1
    streamer._open_trade = {
        "side": "LONG",
        "entry_price": 100.0,
        "planned_stop": 95.0,
        "planned_target": 110.0,
        "contracts": 1.0,
        "max_favorable_r": 1.30,
    }

    monkeypatch.setattr(live_trading_runtime_module, "_predict_proba_safely", lambda *args, **kwargs: [0.90])
    row = pd.Series(
        {
            "Datetime": pd.Timestamp("2026-04-21T11:05:00-06:00"),
            "Close": 106.0,
            "High": 106.25,
            "Low": 105.5,
            "feature_a": 1.0,
            "trend_score": 0.10,
        }
    )

    event = streamer._maybe_phase2_close_event(row, pd.Timestamp("2026-04-21T11:05:00-06:00"))

    assert event is None
    assert row["phase2_close_reason"] == "close_model_suppressed_profit_protected"
    assert streamer._log_exec_event_rows[-1]["reason"] == "close_model_suppressed_profit_protected"


def test_phase2_close_suppressed_near_stop_after_meaningful_mfe(monkeypatch: pytest.MonkeyPatch) -> None:
    streamer = _make_streamer()
    streamer._pos = -1
    streamer._open_trade = {
        "side": "SHORT",
        "entry_price": 100.0,
        "planned_stop": 105.0,
        "planned_target": 92.0,
        "contracts": 1.0,
        "max_favorable_r": 0.80,
    }

    monkeypatch.setattr(live_trading_runtime_module, "_predict_proba_safely", lambda *args, **kwargs: [0.90])
    row = pd.Series(
        {
            "Datetime": pd.Timestamp("2026-04-21T11:10:00-06:00"),
            "Close": 104.0,
            "High": 104.25,
            "Low": 99.5,
            "feature_a": 1.0,
            "trend_score": 1.20,
        }
    )

    event = streamer._maybe_phase2_close_event(row, pd.Timestamp("2026-04-21T11:10:00-06:00"))

    assert event is None
    assert row["phase2_close_reason"] == "close_model_suppressed_near_stop_context"


def test_pnl_overlay_giveback_preserves_overlay_exit_reason() -> None:
    streamer = _make_streamer()
    streamer._pos = 1
    streamer._last_close = 102.5
    streamer._open_trade = {
        "side": "LONG",
        "entry_price": 100.0,
        "planned_stop": 95.0,
        "planned_target": 110.0,
        "contracts": 1.0,
        "max_favorable_r": 1.30,
    }
    row = pd.Series(
        {
            "Close": 102.5,
            "High": 103.0,
            "Low": 102.0,
        }
    )

    event = streamer._maybe_pnl_overlay_event(row, pd.Timestamp("2026-04-21T11:15:00-06:00"))

    assert event is not None
    assert event["ctx"]["exit_reason"] == "pnl_overlay_giveback"
    assert event["ctx"]["pnl_overlay"] is True


def test_phase2_close_accepts_dir_prob_aliases_from_phase2_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamer = _make_streamer()
    streamer._pos = 1
    streamer.allow_feature_mismatch = False
    streamer.phase2_close_expected_features = ["feature_a", "dir_prob_raw", "dir_prob_effective"]
    streamer._open_trade = {
        "side": "LONG",
        "entry_price": 100.0,
        "planned_stop": 95.0,
        "planned_target": 110.0,
        "contracts": 1.0,
    }

    monkeypatch.setattr(live_trading_runtime_module, "_predict_proba_safely", lambda *args, **kwargs: [0.90])
    row = pd.Series(
        {
            "Datetime": pd.Timestamp("2026-04-21T11:20:00-06:00"),
            "Close": 101.0,
            "High": 101.5,
            "Low": 100.5,
            "feature_a": 1.0,
            "proba": 0.63,
            "phase2_dir_prob_raw": 0.61,
            "phase2_dir_prob_effective": 0.62,
        }
    )

    _ = streamer._maybe_phase2_close_event(row, pd.Timestamp("2026-04-21T11:20:00-06:00"))

    # Primary assertion: strict feature align no longer crashes when only
    # phase2_dir_prob_* aliases are present on the row.
    assert float(row["phase2_close_prob"]) == pytest.approx(0.90)
