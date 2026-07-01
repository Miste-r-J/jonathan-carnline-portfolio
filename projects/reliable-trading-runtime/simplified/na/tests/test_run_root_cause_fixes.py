from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer, StreamState, _gate_failure_detail


class _FakeNTBridge:
    def __init__(self) -> None:
        self.sent = []

    def send(self, payload):
        self.sent.append(dict(payload))
        return True


def _make_streamer(tmp_path: Path) -> LiveCSVStreamer:
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    s.state = StreamState()
    s.out_dir = tmp_path
    s.state_path = tmp_path / "stream_state.json"
    s.status_path = tmp_path / "status.json"
    s.run_id = "test"
    s._csv_time_shift_auto = False
    s._csv_time_shift_sec = None
    s._log_exec_event = lambda *args, **kwargs: None
    s._log_order_event = lambda *args, **kwargs: None
    s._emit_block_event = lambda *args, **kwargs: None
    s._set_entries_disarmed = lambda *args, **kwargs: None
    s._set_hard_lockout = lambda *args, **kwargs: None
    s._set_position_state = lambda *args, **kwargs: None
    s._resolve_entry_protection_mode = lambda *args, **kwargs: "abs"
    s._nt_instrument_for_tx = lambda: "ES MAR26"
    s._should_block_nt_send_policy = lambda *_args, **_kwargs: None
    s._has_real_chosen_account = lambda: False
    s.nt_account_mode = "none"
    s.nt_account = None
    s.nt_enabled = True
    s.nt_bridge = _FakeNTBridge()
    s._nt_repair_state_by_instrument = {}
    s._pending_until = None
    s._pending_client_order_id = None
    s._active_close_correlation_id = None
    s._closing_fill_in_flight = False
    s._open_trade = None
    s._pos = 0
    s._pos_side = None
    s._cooldown_left = 0
    s._post_stop_cooldown_left = 0
    s.cooldown_bars = 2
    s.contracts_per_trade = 1
    s.bar_interval_sec = 300
    s.max_daily_loss_usd = None
    s.max_daily_drawdown_usd = None
    s._cooldown_started_ts = None
    s._cooldown_started_bar_ts = None
    s._cooldown_started_side = None
    s._cooldown_last_update_ts = None
    s._last_bar_ts_guard = pd.Timestamp("2026-04-15T19:45:00-06:00")
    return s


def test_load_state_canonicalizes_short_side(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.state_path.write_text(
        json.dumps(
            {
                "state": {"position_state": "IN_POSITION_UNPROTECTED"},
                "position": {
                    "pos": 1,
                    "side": "SHORT",
                    "entry_price": 7000.0,
                    "entry_stop": 7005.0,
                    "entry_target": 6990.0,
                    "cooldown_left": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    LiveCSVStreamer._load_state(streamer)

    assert streamer._pos == -1
    assert streamer._open_trade["side"] == "SHORT"


def test_current_position_side_prefers_canonical_short_side(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 1
    streamer._pos_side = "SHORT"
    streamer._open_trade = {"side": "SHORT"}

    assert LiveCSVStreamer._current_position_side(streamer) == "SHORT"


def test_generic_cooldown_blocks_same_side_only(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._cooldown_left = 2
    streamer._cooldown_started_side = "LONG"
    streamer._flat_without_pending_execution = lambda *_args, **_kwargs: True
    streamer._is_stale_flat_startup_cooldown = lambda *args, **kwargs: False
    streamer._cooldown_age_exceeded = lambda *_args, **_kwargs: False
    streamer._get_gate_now = lambda: pd.Timestamp("2026-04-15T19:45:00-06:00")
    streamer._maybe_reset_hard_lockout = lambda *_args, **_kwargs: None
    streamer.disable_safety_gates = False
    streamer.state.day_stopped = False
    streamer.state.hard_lockout_until = None
    streamer._instrument_lockouts = {}
    streamer.instrument_alias = "ES"
    streamer.state.active_client_order_id = None
    streamer.trade_window_start = "07:30"
    streamer.trade_window_end = "14:00"

    same_side = LiveCSVStreamer._hard_gate_reason(streamer, pd.Timestamp("2026-04-15T19:45:00-06:00"), "OPEN", "LONG", {})
    opp_side = LiveCSVStreamer._hard_gate_reason(streamer, pd.Timestamp("2026-04-15T19:45:00-06:00"), "OPEN", "SHORT", {})

    assert same_side == "cooldown_active"
    assert opp_side is None


def test_repair_payload_marks_optional_target(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)

    LiveCSVStreamer._send_nt_protection_repair(
        streamer,
        inst_key="ES MAR26",
        pos_qty=-1.0,
        stop_price=7005.0,
        target_price=None,
        reason="repair_test",
    )

    payload = streamer.nt_bridge.sent[-1]
    assert payload["context"] == "repair"
    assert payload["repair_target_required"] is False
    assert payload["target_price"] is None


def test_short_prob_gate_detail_is_neutral_about_comparison_direction() -> None:
    failed_gate, compare = _gate_failure_detail(
        gate_state={"prob": False, "vwap": True, "ema": True, "tod": True},
        prob=0.55,
        p_buy=0.7,
        p_sell=0.4,
        close=1.0,
        vwap=1.0,
        ema20=1.0,
        ema50=1.0,
        ts_dt=pd.Timestamp("2026-04-15T19:45:00-06:00"),
        trade_window_start="07:30",
        trade_window_end="14:00",
        phase2_setup_prob=0.0,
        phase2_p_setup=0.35,
        risk_reason=None,
    )

    assert failed_gate == "prob"
    assert "p_short_required" in compare
    assert "< p_short_required" not in compare


