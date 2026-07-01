from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

from na.discord_addons.cli.stream_live_csv import (
    InstrumentRiskParams,
    LiveCSVStreamer,
    StreamState,
    _compute_vwap_gate_raw,
    _apply_preset_to_args,
    _gate_failure_detail,
)


def _make_min_streamer() -> LiveCSVStreamer:
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    s.state = StreamState()
    s.state.day_stopped = False
    s.disable_safety_gates = False
    s._pending_until = None
    s._pending_client_order_id = None
    s._instrument_lockouts = {}
    s.session_tz = "America/Denver"
    s.trade_window_start = "07:30"
    s.trade_window_end = "14:00"
    s.pre_unlock_start = "06:30"
    s.instrument_alias = "ES"
    s.instrument = SimpleNamespace(tick_size=0.25, point_value=50.0)
    s.sim_mode = True
    s.nt_enabled = False
    s.safe_mode = True
    s.run_mode = "replay"
    s.leak_guard_strict = False
    s._pos = 1
    s._bars_in_trade = 0
    s._entry_time = None
    s._entry_price = 105.0
    s._entry_stop = 99.0
    s._entry_target = 110.0
    s._open_trade = {}
    s.qa_emit_signals = False
    s._qa_emit_state = None
    s._last_prob = 0.5
    s._cooldown_left = 0
    s.cooldown_bars = 0
    s.post_stop_cooldown_bars = 3
    s._post_stop_cooldown_left = 0
    s._scratch_cooldown_left = 0
    s.max_daily_loss_usd = None
    s.max_daily_drawdown_usd = None
    s.max_position_contracts = None
    s.max_risk_usd_per_trade = None
    s.contracts_per_trade = 1
    s._get_gate_now = lambda: pd.Timestamp("2026-03-12T08:00:00-06:00")
    s._maybe_reset_hard_lockout = lambda *args, **kwargs: None
    s.min_target_r_multiple = 1.0
    s.instrument_risk_params = InstrumentRiskParams(
        min_stop_ticks=6,
        max_stop_points=20.0,
        max_target_points=80.0,
        max_r_multiple=3.0,
        base_risk_usd=0.0,
    )
    return s


def test_sim_bar_stop_enforcement_caps_at_stop() -> None:
    s = _make_min_streamer()
    s.sim_bar_stop_enforcement = True

    row = pd.Series({"Open": 95.0, "High": 96.0, "Low": 94.0, "Close": 95.5})
    ts_dt = pd.Timestamp("2026-03-12T08:00:00")

    ev = LiveCSVStreamer._maybe_forced_exit(s, row, ts_dt)
    assert ev is not None
    assert ev["type"] == "CLOSE"
    assert ev["price"] == 99.0
    assert (ev.get("ctx") or {}).get("exit_reason") == "stop_hit"

    s.sim_bar_stop_enforcement = False
    ev2 = LiveCSVStreamer._maybe_forced_exit(s, row, ts_dt)
    assert ev2 is not None
    assert ev2["price"] == 95.0


def test_post_stop_cooldown_blocks_entries() -> None:
    s = _make_min_streamer()
    s._pos = 0
    s._post_stop_cooldown_left = 2

    reason = LiveCSVStreamer._hard_gate_reason(
        s,
        pd.Timestamp("2026-03-12T08:00:00"),
        "OPEN",
        "LONG",
        {},
    )
    assert reason == "post_stop_cooldown_active"

    s._post_stop_cooldown_left = 0
    reason2 = LiveCSVStreamer._hard_gate_reason(
        s,
        pd.Timestamp("2026-03-12T08:00:00"),
        "OPEN",
        "LONG",
        {},
    )
    assert reason2 is None


def test_post_stop_cooldown_armed_and_decrements(tmp_path) -> None:
    s = _make_min_streamer()
    s._phase = "LIVE"
    s.online_update_enabled = False
    s.online_learner = None
    s.policy = None
    s.max_losses_per_day = None
    s._fx_state = SimpleNamespace(active_trades={})

    s._set_position_state = lambda *args, **kwargs: None
    s._set_policy_flag = lambda *args, **kwargs: None
    s._set_hard_lockout = lambda *args, **kwargs: None
    s._update_trade_csv_row = lambda *args, **kwargs: None
    s._log_order_event = lambda *args, **kwargs: None
    s._append_sim_close_row = lambda *args, **kwargs: None

    s.out_dir = tmp_path
    s.trades_csv = tmp_path / "trades.csv"

    s._open_trade = {
        "side": "LONG",
        "entry_price": 105.0,
        "contracts": 1.0,
        "risk": 6.0,
        "datetime": "2026-03-12T07:55:00-06:00",
        "client_order_id": "CID",
    }

    close_ev = {
        "type": "CLOSE",
        "datetime": "2026-03-12T08:00:00-06:00",
        "price": 99.0,
        "ctx": {"exit_reason": "stop"},
    }

    LiveCSVStreamer._record_exit_event(s, close_ev)
    assert s._post_stop_cooldown_left == 3

    LiveCSVStreamer._decrement_cooldowns(s)
    assert s._post_stop_cooldown_left == 2


def test_position_tracking_allows_forced_exit_after_fallback_open() -> None:
    s = _make_min_streamer()
    s._pos = 0
    s._bars_in_trade = 4
    s._entry_time = None
    s._entry_price = None
    s._entry_stop = None
    s._entry_target = None
    s.cooldown_bars = 2
    s.sim_bar_stop_enforcement = True

    open_ev = {
        "type": "OPEN",
        "datetime": "2026-03-12T08:00:00-06:00",
        "price": 105.0,
        "side": "LONG",
        "risk": {"stop": 99.0, "target": 111.0},
    }

    LiveCSVStreamer._apply_event_position_tracking(
        s,
        typ="OPEN",
        side="LONG",
        ev=open_ev,
        prev_pos=0,
        prev_bars=4,
    )

    assert s._pos == 1
    assert s._entry_stop == 99.0
    assert s._entry_target == 111.0
    assert s._cooldown_left == 2

    row = pd.Series({"Open": 98.0, "High": 100.0, "Low": 97.0, "Close": 98.5})
    ev = LiveCSVStreamer._maybe_forced_exit(s, row, pd.Timestamp("2026-03-12T08:05:00"))
    assert ev is not None
    assert ev["price"] == 99.0
    assert (ev.get("ctx") or {}).get("exit_reason") == "stop_hit"


def test_clear_position_tracking_after_close_resets_state() -> None:
    s = _make_min_streamer()
    s._pos = -1
    s._bars_in_trade = 3
    s._entry_time = pd.Timestamp("2026-03-12T08:00:00")
    s._entry_price = 105.0
    s._entry_stop = 109.0
    s._entry_target = 97.0

    LiveCSVStreamer._clear_position_tracking_after_close(s)

    assert s._pos == 0
    assert s._bars_in_trade == 0
    assert s._entry_time is None
    assert s._entry_price is None
    assert s._entry_stop is None
    assert s._entry_target is None


def test_vwap_gate_fade_mode_allows_short_above_vwap() -> None:
    assert _compute_vwap_gate_raw(
        close=6800.75,
        vwap=6795.926,
        longish=False,
        prob=0.2882711559391614,
        mode="fade",
    )


def test_vwap_gate_trend_mode_blocks_short_above_vwap() -> None:
    assert not _compute_vwap_gate_raw(
        close=6800.75,
        vwap=6795.926,
        longish=False,
        prob=0.2882711559391614,
        mode="trend",
    )


def test_vwap_failure_detail_respects_fade_mode() -> None:
    failed_gate, compare = _gate_failure_detail(
        gate_state={"prob": True, "vwap": False, "ema": True, "tod": True},
        prob=0.72,
        p_buy=0.53,
        p_sell=0.47,
        close=6781.25,
        vwap=6780.34,
        ema20=1.0,
        ema50=0.0,
        ts_dt=pd.Timestamp("2026-03-11T11:30:00-06:00"),
        trade_window_start="07:30",
        trade_window_end="14:30",
        phase2_setup_prob=0.5,
        phase2_p_setup=0.4,
        risk_reason=None,
        vwap_gate_mode="fade",
    )
    assert failed_gate == "vwap"
    assert compare == "close=6781.250 >= vwap=6780.340"


def test_stop_target_sanity_expands_target_to_min_r_multiple() -> None:
    s = _make_min_streamer()
    s.min_target_r_multiple = 1.5

    ev = {
        "type": "OPEN",
        "side": "LONG",
        "price": 100.0,
        "risk": {"stop": 92.0, "target": 108.0},
    }

    ok, reason, meta = LiveCSVStreamer._stop_target_sanity_check(s, ev)
    assert ok
    assert reason is None
    assert ev["risk"]["target"] == 112.0
    assert meta["target_floor_applied"] is True
    assert meta["r_multiple"] == 1.5


def test_apply_preset_to_args_loads_vwap_gate_mode() -> None:
    args = argparse.Namespace(
        preset="es_maxpack_10_full_send_prop_safe_pnl",
        _preset_fields=set(),
        instrument=None,
        session_tz=None,
        csv_tz=None,
        trade_window_start=None,
        trade_window_end=None,
        rth_start=None,
        rth_end=None,
        p_buy=0.5,
        p_sell=0.5,
        gate_vwap=True,
        vwap_gate_mode="trend",
        gate_ema=True,
        gate_tod=True,
        gate_mode="prob_and_one",
        protection_mode=None,
        close_on_bracket_only=True,
        override_prob_min=0.78,
        override_hold_conf_min=0.70,
        phase2=False,
        phase2_tag=None,
        phase2_use_manifest_thresholds=False,
        publish_min_grade="B+",
        allow_shorts=True,
        min_execute_grade=None,
        max_trades_per_session=0,
        max_trades_per_day=0,
        max_daily_loss=None,
        max_risk_per_trade_usd=None,
        cooldown_after_loss=0,
        loss_streak_limit=0,
        max_losses_per_day=0,
        min_hold_bars=1,
        hold_threshold_long=0.6,
        hold_threshold_short=0.4,
        max_hold_bars=0,
        proba_cut_bad=0.55,
        exit_prob_confirm_bars=1,
        persist_prob_bars=1,
        max_r_giveback=0.0,
        max_position_contracts=None,
        auto_contracts=False,
        scratch_block_bars_max=0,
        scratch_reentry_cooldown_bars=0,
        vwap_fade_override_pts=0.0,
        vwap_fade_prob_threshold=0.3,
        min_target_r_multiple=1.0,
        extreme_proximity_pts=0.0,
        extreme_proximity_prob_boost=0.05,
        sim_bar_stop_enforcement=False,
        post_stop_cooldown_bars=0,
    )

    updated = _apply_preset_to_args(args, [])
    assert updated.trade_window_start == "08:10"
    assert updated.gate_vwap is True
    assert updated.vwap_gate_mode == "any"
    assert updated.min_target_r_multiple == pytest.approx(1.0)
    assert updated.protection_mode == "stop_and_target"
    assert updated.override_prob_min == pytest.approx(1.0)
    assert updated.override_hold_conf_min == pytest.approx(1.0)


def test_sync_fallback_gate_config_applies_vwap_mode() -> None:
    s = _make_min_streamer()
    s.fallback = SimpleNamespace(
        cfg=SimpleNamespace(
            trade_window_start="08:00",
            trade_window_end="14:30",
            use_vwap_gate=False,
            vwap_gate_mode="trend",
            use_ema_gate=True,
            use_tod_gate=True,
            gate_mode="prob_and_one",
        )
    )

    LiveCSVStreamer._sync_fallback_gate_config(
        s,
        {
            "trade_window_start": "07:30",
            "gate_vwap": True,
            "vwap_gate_mode": "fade",
            "gate_ema": False,
            "gate_tod": True,
            "gate_mode": "prob_and_both",
        },
    )

    assert s.fallback.cfg.trade_window_start == "07:30"
    assert s.fallback.cfg.use_vwap_gate is True
    assert s.fallback.cfg.vwap_gate_mode == "fade"
    assert s.fallback.cfg.use_ema_gate is False
    assert s.fallback.cfg.gate_mode == "prob_and_both"
