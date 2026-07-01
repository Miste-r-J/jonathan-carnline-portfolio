from __future__ import annotations

import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer, _build_arg_parser


def _make_stub_streamer() -> LiveCSVStreamer:
    streamer = object.__new__(LiveCSVStreamer)
    streamer.session_tz = "America/Denver"
    streamer.bar_interval_sec = 300
    streamer.max_entry_age_sec = None
    streamer.max_entry_bars_since_signal = None
    streamer.max_entry_drift_points = None
    streamer._snapshot_price_state = lambda: (False, None, None, None, None)
    return streamer


def test_cli_parser_accepts_entry_guard_arguments():
    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "--csv",
            "dummy.csv",
            "--model",
            "dummy",
            "--max_entry_age_sec",
            "3",
            "--max_entry_bars_since_signal",
            "1.5",
            "--max_entry_drift_points",
            "2.0",
        ]
    )
    assert args.max_entry_age_sec == 3.0
    assert args.max_entry_bars_since_signal == 1.5
    assert args.max_entry_drift_points == 2.0


def test_pre_send_entry_guard_blocks_stale_signal():
    streamer = _make_stub_streamer()
    streamer.max_entry_age_sec = 3.0
    old_bar = pd.Timestamp.now(tz="America/Denver") - timedelta(seconds=15)
    intent = SimpleNamespace(
        bar_ts=old_bar.isoformat(),
        entry_price=5000.25,
        model_price=5000.25,
    )

    violation = streamer._pre_send_entry_guard_violation(intent)

    assert violation is not None
    assert violation["reason"] == "stale_entry_pre_send"


def test_pre_send_entry_guard_blocks_snapshot_price_drift():
    streamer = _make_stub_streamer()
    streamer.max_entry_drift_points = 2.0
    now_bar = pd.Timestamp.now(tz="America/Denver")
    streamer._snapshot_price_state = lambda: (True, 5004.75, None, None, now_bar)
    intent = SimpleNamespace(
        bar_ts=now_bar.isoformat(),
        entry_price=5000.25,
        model_price=5000.25,
    )

    violation = streamer._pre_send_entry_guard_violation(intent)

    assert violation is not None
    assert violation["reason"] == "entry_price_drift_pre_send"


def test_maybe_emit_signal_suppresses_executable_outside_live_but_allows_mark():
    streamer = _make_stub_streamer()
    streamer.phase_state_machine_enabled = True
    streamer.run_mode = "live"
    streamer._phase = "CATCHUP"
    streamer._phase_current_confirmations = 0
    streamer.phase_currentness_confirmations_required = 3
    streamer.phase_currentness_lag_sec_threshold = 30.0
    streamer._phase_last_lag_sec = 5.0
    now = pd.Timestamp.now(tz="America/Denver")
    streamer._phase_last_bar_ts = now.isoformat()
    streamer._get_gate_now = lambda: now
    streamer._suppressed_due_to_phase_count = 0
    streamer._executable_signals_outside_live_count = 0
    streamer._last_suppressed_phase_bar_ts = None
    streamer._log_exec_event = MagicMock()
    streamer._append_signal = MagicMock()

    streamer._maybe_emit_signal({"type": "OPEN", "datetime": now.isoformat()})
    streamer._maybe_emit_signal({"type": "MARK", "datetime": now.isoformat()})

    assert streamer._suppressed_due_to_phase_count == 1
    assert streamer._executable_signals_outside_live_count == 1
    streamer._append_signal.assert_called_once()
    appended_ev = streamer._append_signal.call_args.args[0]
    assert appended_ev["_phase_non_executable"] is True
    assert appended_ev["_emit_allowed"] is False


def test_missing_stop_repair_is_suppressed_after_confirmed_exit_fill():
    streamer = _make_stub_streamer()
    streamer._ensure_compat_defaults = lambda: None
    streamer._expected_unprotected_reason = lambda inst_key: None
    streamer.protection_repair_enabled = True
    streamer.state = SimpleNamespace(
        position_state="IN_POSITION_PROTECTED",
        last_confirmed_exit_order_id=None,
        protection_cleanup_state=None,
    )
    streamer._require_fresh_snapshot = lambda **kwargs: True
    streamer._nt_repair_state_by_instrument = {}
    streamer._protection_repairs_suppressed_confirmed_exit = 0
    active_state = {
        "client_order_id": "cid-1",
        "signal_id": "sig-1",
        "exit_fill_ts": time.time(),
        "stop_order_id": "stop-1",
        "last_exit_order_id": "stop-1",
    }
    streamer._active_order_state_for_instrument = lambda inst_key: ("cid-1", active_state)
    streamer._set_position_state = MagicMock()
    streamer._log_exec_event = MagicMock()
    streamer._sync_position_flat = MagicMock()
    streamer._set_hard_lockout = MagicMock()

    streamer._maybe_repair_protection(
        inst_key="ES",
        pos_qty=-1.0,
        stop_price=None,
        target_price=None,
        reason="snapshot_no_orders",
    )

    streamer._set_hard_lockout.assert_not_called()
    streamer._sync_position_flat.assert_called_once()
    assert streamer._protection_repairs_suppressed_confirmed_exit == 1
    assert streamer.state.last_confirmed_exit_order_id == "stop-1"


def test_trade_pnl_state_updates_mfe_mae_timestamps():
    streamer = _make_stub_streamer()
    streamer.instrument = SimpleNamespace(point_value=50.0)
    streamer._pos = -1
    streamer._bars_in_trade = 2
    streamer._last_close = 7060.0
    streamer._entry_time = None
    streamer.state = SimpleNamespace(nt_protected=True)
    streamer._canonical_protection_snapshot = lambda open_trade=None, state=None: {"stop": 7077.0, "target": 7049.0}
    streamer._open_trade = {
        "side": "SHORT",
        "entry_price": 7068.75,
        "actual_entry_price": 7068.75,
        "contracts": 1.0,
        "protection_status": "protected_confirmed",
        "mfe_points": 0.0,
        "mae_points": 0.0,
    }

    ts_dt = pd.Timestamp("2026-04-16T07:55:00-06:00")
    pnl_state = streamer._compute_trade_pnl_state(
        mark_price=7052.0,
        high_price=7070.0,
        low_price=7049.0,
        mark_ts=ts_dt,
        update_extrema=True,
    )

    assert round(pnl_state["unrealized_points"], 2) == 16.75
    assert streamer._open_trade["mfe_points"] == 19.75
    assert streamer._open_trade["mae_points"] == 1.25
    assert streamer._open_trade["best_mark_ts"] == ts_dt.isoformat()
    assert streamer._open_trade["worst_mark_ts"] == ts_dt.isoformat()
