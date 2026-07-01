from types import SimpleNamespace

import pandas as pd

from trading_system.runtime_engine.integrations.execution_state import ExecutionState
from trading_system.runtime_engine.integrations.cli.live_trading_runtime import LiveCSVStreamer, StreamState
from trading_system.runtime_engine.integrations.cli.live_trading_runtime import ExecutionIntent, ExecutionResult


def _make_streamer():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.state = StreamState()
    streamer.state.last_open_bar_ts_by_side = {}
    streamer.state.day_stopped = False
    streamer.state.daily_usd = 0.0
    streamer.state.daily_peak_usd = 0.0
    streamer.state.hard_lockout_until = None
    streamer.state.hard_lockout_reason = None
    streamer.state.hard_lockout_set_ts = None
    streamer.state.entries_disarmed_until = None
    streamer.state.last_session_reset_ts = None
    streamer.session_tz = "America/Denver"
    streamer.session_controller_tz = "America/Denver"
    streamer.instrument_alias = "ES"
    streamer.trade_window_start = "07:30"
    streamer.trade_window_end = "14:30"
    streamer.pre_arm_until = "07:30"
    streamer.pre_unlock_start = "06:30"
    streamer.session_start = "07:30"
    streamer.session_end = "14:30"
    streamer.use_tod_gate = False
    streamer.fallback = None
    streamer.disable_safety_gates = False
    streamer._pending_until = None
    streamer._pending_client_order_id = None
    streamer._instrument_lockouts = {}
    streamer._cooldown_left = 0
    streamer._post_stop_cooldown_left = 0
    streamer._scratch_cooldown_left = 0
    streamer._scratch_cooldown_side = None
    streamer._pos = 0
    streamer._strategy_trade_limit = 0
    streamer._entries_disarmed_reason = None
    streamer._bootstrap_active = False
    streamer._in_soft_lockout_window = False
    streamer.phase_state_machine_enabled = False
    streamer.time_mode = "wall_clock"
    streamer._sim_now_initialized = True
    streamer.run_mode = "live"
    streamer.lockout_policy = "standard"
    streamer.skip_weekdays = set()
    streamer._session_high = None
    streamer._session_low = None
    streamer._last_hold_side = None
    streamer._last_hold_grade = None
    streamer._last_hold_ts = None
    streamer.max_daily_loss_usd = None
    streamer.max_daily_drawdown_usd = None
    streamer.max_position_contracts = None
    streamer.max_risk_usd_per_trade = None
    streamer.max_consec_errors = 0
    streamer._consec_errors = 0
    streamer.min_execute_grade = None
    streamer.contracts_per_trade = 1
    streamer.instrument = SimpleNamespace(point_value=50.0)
    streamer.p_buy = 0.75
    streamer.p_sell = 0.50
    streamer._emit_unblocked = lambda code: None
    streamer._log_exec_event = lambda payload: None
    streamer._force_reset_day_state = lambda ts, reason="session_open": setattr(streamer.state, "last_session_reset_ts", ts)
    streamer._set_live_state = lambda state, reason=None: setattr(streamer, "_live_state", state)
    streamer._set_entries_disarmed = lambda reason, detail=None: setattr(streamer, "_entries_disarmed_reason", reason)
    streamer._clear_entries_disarmed = lambda: setattr(streamer, "_entries_disarmed_reason", None)
    streamer._clear_hard_lockout = lambda reason, evidence=None: setattr(streamer.state, "hard_lockout_reason", None)
    streamer._get_gate_now = lambda: pd.Timestamp("2026-03-16T23:10:00-06:00")
    return streamer


def _make_entry_streamer():
    streamer = _make_streamer()
    streamer._phase = "BACKFILL"
    streamer._ensure_compat_defaults = lambda: None
    streamer._ensure_day_state = lambda ts: None
    streamer._ensure_signal_id = lambda ev: "sig|1"
    streamer._build_nt_client_order_id = lambda ev: "cid|1"
    streamer._apply_contract_size = lambda ev: None
    streamer._append_trade_open_row = lambda row: None
    streamer.active_preset_name = "preset"
    streamer.active_strategy_tag = None
    streamer._last_prediction_id = None
    streamer.nt_enabled = False
    streamer.nt_bridge = None
    streamer._last_row_features = None
    streamer._last_base_context = {}
    streamer._strategy_last_entry_ts = None
    streamer._strategy_entries_today = 0
    streamer._entry_intents_today = 0
    streamer._current_bar_index = None
    streamer.policy = None
    streamer._dedupe_until = {}
    streamer._update_dedupe_flag = lambda ts: None
    streamer.state.last_signal_ts = {}
    return streamer


def test_hard_gate_skips_trade_window_when_tod_disabled():
    streamer = _make_streamer()
    streamer._maybe_reset_hard_lockout = lambda now: None

    ts = pd.Timestamp("2026-03-16T23:10:00-06:00")
    reason = streamer._hard_gate_reason(ts, "OPEN", "LONG", {"price": 6691.5, "prob": 0.9})

    assert reason is None


def test_hard_gate_blocks_offhours_when_tod_enabled():
    streamer = _make_streamer()
    streamer.use_tod_gate = True
    streamer._maybe_reset_hard_lockout = lambda now: None

    ts = pd.Timestamp("2026-03-16T23:10:00-06:00")
    reason = streamer._hard_gate_reason(ts, "OPEN", "LONG", {"price": 6691.5, "prob": 0.9})

    assert reason == "outside_trade_window"


def test_session_controller_stays_trading_offhours_when_tod_disabled():
    streamer = _make_streamer()
    streamer._entries_disarmed_reason = "session_closed"

    streamer._session_controller_tick(pd.Timestamp("2026-03-16T23:10:00-06:00"), has_new_bars=True)

    assert streamer._live_state == "TRADING_SESSION"
    assert streamer._entries_disarmed_reason is None


def test_session_lockout_clears_when_tod_disabled():
    streamer = _make_streamer()
    streamer.state.hard_lockout_reason = "session_closed"
    streamer._entries_disarmed_reason = "session_closed"

    streamer._maybe_reset_hard_lockout(pd.Timestamp("2026-03-16T23:10:00-06:00"))

    assert streamer.state.hard_lockout_reason is None
    assert streamer._entries_disarmed_reason is None


def test_backfill_entry_events_do_not_consume_live_trade_limit():
    streamer = _make_entry_streamer()

    streamer._record_entry_event(
        {
            "type": "OPEN",
            "side": "LONG",
            "datetime": "2026-03-16T23:45:00-06:00",
            "price": 6687.5,
            "risk": {"stop": 6680.0, "target": 6695.0},
        }
    )

    assert streamer._entry_intents_today == 0
    assert streamer.state.entry_intents_today == 0


def test_execute_intent_open_uses_broker_state_not_speculative_model_pos():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.state = StreamState()
    streamer.state.position_state = "FLAT"
    streamer.state.last_nt_client_order_id = None
    streamer.execution_state = ExecutionState()
    streamer._pos = 1
    streamer.live_dry_run = False
    streamer.nt_exec_policy = "paper"
    streamer.strict_entries_require_signal_id = False
    streamer._hard_lockout_active = False
    streamer._entries_disarmed_reason = None
    streamer.nt_exec_state = "ARMED"
    streamer.nt_require_snapshot = False
    streamer.nt_account_mode = "none"
    streamer.nt_enabled = True
    streamer.nt_bridge = object()
    streamer.execution_ledger = {}
    streamer._send_intent_order = lambda intent: ExecutionResult("SENT", "stub_sent")

    intent = ExecutionIntent(
        intent_id="intent|1",
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="DEMO6927902",
        order_type="MARKET",
        stop_price=6673.0,
        target_price=6689.0,
        entry_price=6681.0,
        bar_ts="2026-03-17T02:45:00-06:00",
        model_price=6681.0,
        model_stop_price=6673.0,
        model_target_price=6689.0,
        signal_id="sig|1",
    )

    result = streamer.execute_intent(intent)

    assert result.decision == "SENT"
    assert result.reason_code == "stub_sent"


def test_fill_price_out_of_bounds_accepts_market_fill_matching_fresh_snapshot():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.max_fill_slippage_ticks = 37
    streamer.tick_size = 0.25
    streamer.nt_snapshot_fresh_sec = 450.0
    streamer.guardrail_config = SimpleNamespace(snapshot_age_max_sec=450.0)
    streamer._nt_last_price_by_instrument = {"ES 06-26": 6745.75}
    streamer._snapshot_age_sec = lambda inst_key=None: 0.5
    streamer._record_guardrail_event = lambda kind: None
    streamer._log_exec_event = lambda payload: None
    streamer._exec_instrument_key = lambda: "ES JUN26"

    blocked = streamer._fill_price_out_of_bounds(
        {
            "instrument": "ES JUN26",
            "order_type": "MARKET",
            "expected_entry_ref": 6695.5,
        },
        6745.75,
    )

    assert blocked is False


def test_fill_price_out_of_bounds_still_blocks_without_snapshot_reconcile():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.max_fill_slippage_ticks = 37
    streamer.tick_size = 0.25
    streamer.nt_snapshot_fresh_sec = 450.0
    streamer.guardrail_config = SimpleNamespace(snapshot_age_max_sec=450.0)
    streamer._nt_last_price_by_instrument = {}
    streamer._snapshot_age_sec = lambda inst_key=None: None
    streamer._record_guardrail_event = lambda kind: None
    streamer._log_exec_event = lambda payload: None
    streamer._exec_instrument_key = lambda: "ES JUN26"

    blocked = streamer._fill_price_out_of_bounds(
        {
            "instrument": "ES JUN26",
            "order_type": "MARKET",
            "expected_entry_ref": 6695.5,
        },
        6745.75,
    )

    assert blocked is True


def test_reset_state_ignores_fill_truth_from_prior_run():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.reset_state = True
    streamer.run_id = "new-run"
    streamer.state = StreamState()
    streamer._pos = 0
    streamer._open_trade = None
    streamer._ensure_compat_defaults = lambda: None
    streamer._load_fill_truth_index = lambda fresh=False: {"old": {}}
    streamer._summarize_fill_truth = lambda fill_index: {
        "fills_present": True,
        "net_qty": 1.0,
        "last_cid": "old-run|entry",
        "last_entry": None,
        "last_exit": None,
    }
    streamer._cid_matches_run = lambda cid: False
    streamer._rebuild_trades_from_fill_truth = lambda fill_index: (_ for _ in ()).throw(AssertionError("should not rebuild"))
    streamer._log_exec_event = lambda payload: None

    streamer._apply_fill_truth_reconciliation(source="startup")

    assert streamer._pos == 0
    assert streamer.state.active_client_order_id is None
