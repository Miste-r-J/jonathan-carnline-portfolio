from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from na.bot.EnhancedGuardrails import EnhancedGuardRailConfig, EnhancedGuardrails
from na.bot.PropRiskManager import PropRiskConfig
from na.bot.VolatilityFilter import VolatilityConfig
from na.discord_addons.addons.risk_guard import Context, RiskGuard
from na.discord_addons.cli.stream_live_csv import (
    ExecutionIntent,
    LiveCSVStreamer,
    StreamState,
    _load_prop_guardrails,
)


class _FakeEmissionLedger:
    def seen(self, _client_order_id: str) -> bool:
        return False

    def append(self, _client_order_id: str, _payload: dict) -> None:
        return None


def test_risk_guard_counts_completed_fills_not_entry_candidates(tmp_path: Path) -> None:
    cfg = tmp_path / "risk_guard.yaml"
    cfg.write_text(
        """
profiles:
  test:
    tz: America/Denver
    session: {rth_start: "00:00", rth_end: "23:59"}
    loss_limits: {max_trades_per_day: 1}
""",
        encoding="utf-8",
    )
    guard = RiskGuard(profile="test", yaml_path=cfg, instrument="ES", default_tz="America/Denver")
    now = datetime(2026, 6, 4, 19, 55, tzinfo=ZoneInfo("America/Denver"))
    ctx = Context(instrument="ES", tier_name=None, strategy_name=None, now=now, price=7563.5)

    decision = guard.evaluate_entry(ctx, "short")

    assert decision.action == "allow"
    assert guard.last_policy_snapshot("ES")["trades_today"] == 0

    guard.record_fill(result="flat", r=0.0, usd=0.0, when=now, side="short", instrument="ES")

    assert guard.should_trade_now(now, instrument="ES").reason == "max_trades_reached"


def test_risk_guard_zero_max_trades_means_disabled(tmp_path: Path) -> None:
    cfg = tmp_path / "risk_guard.yaml"
    cfg.write_text(
        """
profiles:
  test:
    tz: America/Denver
    session: {rth_start: "00:00", rth_end: "23:59"}
    loss_limits: {max_trades_per_day: 0}
""",
        encoding="utf-8",
    )
    guard = RiskGuard(profile="test", yaml_path=cfg, instrument="ES", default_tz="America/Denver")
    now = datetime(2026, 6, 4, 20, 55, tzinfo=ZoneInfo("America/Denver"))

    decision = guard.should_trade_now(now, instrument="ES")

    assert decision.action == "allow"
    assert decision.reason == "within_limits"


def test_risk_guard_zero_loss_limits_mean_disabled(tmp_path: Path) -> None:
    cfg = tmp_path / "risk_guard.yaml"
    cfg.write_text(
        """
profiles:
  test:
    tz: America/Denver
    session: {rth_start: "00:00", rth_end: "23:59"}
    loss_limits:
      max_losses_per_day: 0
      max_consecutive_losses: 0
      max_dollar_loss: 0
      max_R_per_day: 0
""",
        encoding="utf-8",
    )
    guard = RiskGuard(profile="test", yaml_path=cfg, instrument="ES", default_tz="America/Denver")
    now = datetime(2026, 6, 4, 20, 55, tzinfo=ZoneInfo("America/Denver"))

    decision = guard.should_trade_now(now, instrument="ES")

    assert decision.action == "allow"
    assert decision.reason == "within_limits"


def _make_streamer_for_execute():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.run_id = "RUNID"
    streamer.instrument_alias = "ES"
    streamer._last_executor_decision = {}
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=False, handshake_ok=lambda: False)
    streamer.nt_exec_state = "ARMED"
    streamer._hard_lockout_active = False
    streamer._entries_disarmed_reason = None
    streamer.nt_require_snapshot = False
    streamer._nt_snapshot_seen = True
    streamer.nt_snapshot_fresh_sec = 9999.0
    streamer.nt_account_mode = "none"
    streamer.live_dry_run = False
    streamer.nt_exec_policy = "paper"
    streamer.execution_ledger = {}
    streamer.state = StreamState()
    streamer.state.last_nt_client_order_id = None
    streamer.state.position_state = "FLAT"
    streamer._pos = 0
    streamer.exec_instrument = "ES MAR26"
    streamer.nt_instrument = "ES MAR26"
    streamer.instrument_alias = "ES"
    streamer.protection_price_mode = "offset"
    streamer.protection_mode = "stop_and_target"
    streamer._nt_last_price_by_instrument = {}
    streamer._last_close = 6700.0
    streamer._log_exec_event = lambda payload: None
    streamer._send_intent_order = lambda intent: (_ for _ in ()).throw(AssertionError("send path should not run"))
    streamer._block_active = {}
    streamer._block_last_detail = {}
    streamer._current_block_code = None
    streamer._current_block_ts = None
    streamer._first_trade_block_code = None
    streamer._first_trade_block_ts = None
    streamer._first_trade_block_detail = None
    return streamer


def _make_streamer_for_append(tmp_path: Path):
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.run_id = "RUNID"
    streamer.instrument_alias = "ES"
    streamer.active_preset_name = "preset"
    streamer.active_strategy_tag = "strategy"
    streamer.model_run_id = "model"
    streamer.model_sha = "sha"
    streamer.session_tz = "America/Denver"
    streamer.run_mode = "live"
    streamer.execution_mode = "model_master"
    streamer.protection_mode = "stop_and_target"
    streamer.protection_price_mode = "offset"
    streamer.allow_replay_exec_to_nt = False
    streamer.replay_exec_pacing = None
    streamer.contract_version = "v1"
    streamer._nt_addon_version = "test"
    streamer._nt_addon_flags = {}
    streamer._phase = "LIVE"
    streamer._phase_allows_execution = lambda: True
    streamer._emission_eval_calls = {}
    streamer._executor_stats = {}
    streamer._ensure_close_intent_id = lambda ev: None
    streamer._ensure_signal_id = lambda ev: "sig|prop"
    streamer._append_signal_jsonl = lambda ev: None
    streamer._build_nt_client_order_id = lambda ev: "cid|prop"
    streamer._prop_guardrails_pre_signal_reason = lambda ev: "prop_pre_signal:consecutive_loss_limit"
    streamer._log_exec_event = lambda payload: None
    streamer._log_signal_to_order = lambda **kwargs: None
    streamer._maybe_send_nt_order = lambda ev: (_ for _ in ()).throw(AssertionError("NT send should not run"))
    streamer._emission_ledger = _FakeEmissionLedger()
    streamer.contracts_per_trade = 1.0
    streamer.signals_csv = tmp_path / "signals.csv"
    streamer.out_dir = tmp_path
    streamer.status_path = tmp_path / "status.json"
    streamer.events_jsonl = tmp_path / "events.jsonl"
    streamer.event_ledger_path = tmp_path / "event_ledger.jsonl"
    streamer.blocked_candidates_path = tmp_path / "blocked_candidates.jsonl"
    streamer.order_events_path = tmp_path / "order_events.jsonl"
    streamer.policy_guard_enabled = False
    streamer.state = StreamState()
    streamer.state.position_state = "FLAT"
    streamer.nt_enabled = False
    streamer.nt_bridge = None
    streamer.nt_exec_state = "ARMED"
    streamer._hard_lockout_active = False
    streamer._entries_disarmed_reason = None
    streamer._last_csv_bar = pd.Timestamp("2026-03-23T08:00:00-06:00")
    streamer._last_bar_ts_guard = streamer._last_csv_bar
    streamer._last_suppressed_phase_bar_ts = None
    streamer._block_active = {}
    streamer._block_last_detail = {}
    streamer._current_block_code = None
    streamer._current_block_ts = None
    streamer._first_trade_block_code = None
    streamer._first_trade_block_ts = None
    streamer._first_trade_block_detail = None
    return streamer


def _make_streamer_for_sync_flat(lockouts: list[str]):
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.state = StreamState()
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer._phase = "LIVE"
    streamer.run_mode = "live"
    streamer.nt_exec_policy = "live"
    streamer.fallback = SimpleNamespace(pos=1, bars_in_trade=1, entry_time=None, entry_price=6700.0)
    streamer._pos = 1
    streamer._bars_in_trade = 1
    streamer._entry_time = None
    streamer._entry_price = 6700.0
    streamer._entry_stop = 6699.0
    streamer._entry_target = 6708.0
    streamer._active_close_correlation_id = None
    streamer._active_close_started_ts = None
    streamer._close_intents = {}
    streamer._exit_stuck_flatten_attempted = set()
    streamer._close_watchdog_resend_attempted = set()
    streamer._exit_started_ts = None
    streamer._exit_stuck_detail = None
    streamer.execution_state = SimpleNamespace(update_position=lambda qty, side: None)
    streamer._set_position_state = lambda new_state, cid=None, signal_id=None: setattr(
        streamer.state, "position_state", new_state
    )
    streamer._log_exec_event = lambda payload: None
    streamer._set_hard_lockout = lambda reason, evidence=None, until=None: lockouts.append(str(reason))
    streamer.prop_guardrails = EnhancedGuardrails(
        EnhancedGuardRailConfig(max_daily_loss=10.0),
        VolatilityConfig(),
        PropRiskConfig(max_daily_loss=10.0, max_drawdown=10000.0),
    )
    streamer.prop_guardrails.start_day(50000.0)
    streamer._open_trade = {
        "side": "LONG",
        "contracts": 1.0,
        "client_order_id": "cid|trade",
        "signal_id": "sig|trade",
        "entry_price": 6700.0,
        "entry_fill_price": 6700.0,
        "actual_entry_price": 6700.0,
        "planned_stop": 6699.0,
        "actual_exit_price": 6698.0,
        "exit_fill_price": 6698.0,
        "exit_reason": "stop_hit",
        "prop_trade_recorded": False,
    }
    return streamer


def test_execute_intent_open_blocks_when_prop_guardrail_rejects_signal_validation():
    streamer = _make_streamer_for_execute()
    streamer._prop_guardrails_execution_validation = (
        lambda **kwargs: (False, "slippage_too_high:3.0pts>max=2.0pts", {"slippage_points": 3.0})
    )

    intent = ExecutionIntent(
        intent_id="cid|1",
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES",
        exec_instrument="ES MAR26",
        account=None,
        bar_ts="2026-03-23T08:00:00-06:00",
        model_price=6700.0,
        model_stop_price=6692.0,
        model_target_price=6708.0,
        signal_id="sig|1",
    )

    res = streamer.execute_intent(intent)

    assert res.decision == "BLOCKED_SAFETY"
    assert res.reason_code == "slippage_too_high:3.0pts>max=2.0pts"


def test_append_signal_blocks_open_when_prop_pre_signal_guardrail_disallows_entry(tmp_path: Path):
    streamer = _make_streamer_for_append(tmp_path)
    ev = {
        "type": "OPEN",
        "side": "LONG",
        "datetime": "2026-03-23T08:00:00-06:00",
        "price": 6700.0,
        "prob": 0.76,
        "grade": "A+",
        "risk": {"stop": 6692.0, "target": 6708.0},
        "contracts": 1,
    }

    streamer._append_signal(ev)

    assert streamer.signals_csv.exists()
    content = streamer.signals_csv.read_text(encoding="utf-8")
    assert "prop_pre_signal:consecutive_loss_limit" in content


def test_load_prop_guardrails_returns_initialized_bundle():
    config_path = Path(__file__).resolve().parents[1] / "config" / "prop_config.yaml"

    guardrails, loaded_path = _load_prop_guardrails(str(config_path))

    assert guardrails is not None
    assert loaded_path == config_path
    summary = guardrails.get_state_summary()
    assert "risk" in summary
    assert "guardrails" in summary
    assert guardrails.config.signal_max_age_sec == 180.0
    assert guardrails.config.max_slippage_points == 2.0
    assert guardrails.config.fill_slippage_max == 2.0


def test_sync_position_flat_records_trade_in_prop_risk_manager_and_locks_after_daily_loss():
    lockouts: list[str] = []
    streamer = _make_streamer_for_sync_flat(lockouts)

    streamer._sync_position_flat("close_fill")

    risk_summary = streamer.prop_guardrails.get_state_summary()["risk"]
    assert risk_summary["trades_count"] == 1
    assert risk_summary["realized_pnl"] == -100.0
    assert risk_summary["is_locked"] is True
    assert lockouts == ["daily_loss_limit_breach"]


def test_sync_position_flat_records_trade_using_last_close_when_exit_fill_missing():
    lockouts: list[str] = []
    streamer = _make_streamer_for_sync_flat(lockouts)
    streamer._last_close = 6698.5
    streamer._open_trade["actual_exit_price"] = None
    streamer._open_trade["exit_fill_price"] = None

    streamer._sync_position_flat("manual_flatten")

    risk_summary = streamer.prop_guardrails.get_state_summary()["risk"]
    assert risk_summary["trades_count"] == 1
    assert risk_summary["realized_pnl"] == -75.0
    assert lockouts == ["daily_loss_limit_breach"]


def test_prop_guardrails_update_volatility_blocks_extreme_crash_bar():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.prop_guardrails = EnhancedGuardrails(
        EnhancedGuardRailConfig(),
        VolatilityConfig(price_velocity_threshold=5.0, velocity_lookback_bars=5),
        PropRiskConfig(),
    )
    streamer._prop_last_volatility_state = None
    streamer._prop_last_volatility_log_ts = None
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(payload)

    ts0 = pd.Timestamp("2026-03-23T08:00:00-06:00")
    ts1 = pd.Timestamp("2026-03-23T08:01:00-06:00")
    streamer._prop_guardrails_update_volatility(atr=5.0, price=6700.0, ts_dt=ts0)
    state = streamer._prop_guardrails_update_volatility(atr=20.0, price=6680.0, ts_dt=ts1)

    assert state is not None
    assert state.should_block is True
    assert state.regime in {"elevated", "high", "extreme"}
    assert any(event.get("event") == "prop_volatility_state" for event in events)
