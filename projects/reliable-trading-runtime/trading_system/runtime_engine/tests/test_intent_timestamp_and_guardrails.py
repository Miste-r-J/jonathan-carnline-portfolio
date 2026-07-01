import pytest

from trading_system.runtime_engine.integrations.cli.live_trading_runtime import ExecutionIntent, LiveCSVStreamer


def _make_streamer_for_intents():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.nt_enabled = True
    streamer.nt_bridge = None
    streamer.nt_exec_state = "ARMED"
    streamer._hard_lockout_active = False
    streamer._entries_disarmed_reason = None
    streamer.nt_require_snapshot = False
    streamer._nt_snapshot_seen = True
    streamer.nt_snapshot_fresh_sec = 9999.0
    streamer._nt_last_snapshot_orders_ts_by_instrument = {}
    streamer._nt_last_snapshot_ts = "2026-02-09T07:30:00-07:00"
    streamer.nt_account_mode = "none"
    streamer._pos = 0
    streamer.exec_instrument = "ES MAR26"
    streamer.nt_instrument = "ES MAR26"
    streamer.instrument_alias = "ES"
    streamer.protection_price_mode = "offset"
    streamer.protection_mode = "stop_and_target"
    return streamer


def test_build_execution_intent_normalizes_timezone_naive_strings(tmp_path):
    streamer = _make_streamer_for_intents()
    ev = {
        "type": "OPEN",
        "side": "SHORT",
        "qty": 1,
        "instrument": "ES MAR26",
        # tz-naive string (this was showing up in order_events.jsonl today)
        "datetime": "2026-02-09T07:45:00",
        "price": 6953.0,
        "stop": 6965.5,
        "target": 6940.5,
        "signal_id": "sig|1",
    }
    intent, err = streamer._build_execution_intent(ev)
    assert err is None
    assert intent is not None
    assert intent.bar_ts is not None
    # Feb 9, 2026 is MST in America/Denver (UTC-07:00)
    assert intent.bar_ts.endswith("-07:00")


def test_execute_intent_blocks_entries_when_lockout_active():
    streamer = _make_streamer_for_intents()
    streamer._hard_lockout_active = True
    intent = ExecutionIntent(
        intent_id="cid|1",
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES",
        exec_instrument="ES MAR26",
        account=None,
        bar_ts="2026-02-09T07:30:00-07:00",
        model_price=100.0,
        model_stop_price=99.0,
        model_target_price=101.0,
        signal_id="sig|1",
    )
    res = streamer.execute_intent(intent)
    assert res.decision == "BLOCKED_SAFETY"
    assert res.reason_code == "hard_lockout_active"


def test_execute_intent_rejects_entries_without_signal_id():
    streamer = _make_streamer_for_intents()
    streamer.strict_entries_require_signal_id = True
    intent = ExecutionIntent(
        intent_id="cid|2",
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES",
        exec_instrument="ES MAR26",
        account=None,
        bar_ts="2026-02-09T07:30:00-07:00",
        model_price=100.0,
        model_stop_price=99.0,
        model_target_price=101.0,
        signal_id=None,
    )
    res = streamer.execute_intent(intent)
    assert res.decision == "REJECTED"
    assert res.reason_code == "missing_signal_id"
