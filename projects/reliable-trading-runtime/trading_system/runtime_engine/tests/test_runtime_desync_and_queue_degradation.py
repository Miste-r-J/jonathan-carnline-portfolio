from __future__ import annotations

import queue
from types import SimpleNamespace

from trading_system.runtime_engine.integrations.cli.live_trading_runtime import ExecutionIntent, LiveCSVStreamer


def _mk_streamer() -> LiveCSVStreamer:
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    s.state = SimpleNamespace(position_state="FLAT")
    s.exec_instrument = "ES 06-26"
    s.nt_instrument = "ES 06-26"
    s.instrument_alias = "ES"
    s._emit_event = lambda *args, **kwargs: None
    s._flush_inflight_buffer = lambda: None
    s._log_exec_event = lambda *args, **kwargs: None
    s._send_nt_flatten = lambda *args, **kwargs: None
    s._set_entries_disarmed = lambda *args, **kwargs: None
    s._close_intents = {}
    s._exit_stuck_flatten_attempted = set()
    s._close_watchdog_resend_attempted = set()
    s._open_trade = None
    return s


def test_lineage_missing_signal_id_uses_client_order_fallback() -> None:
    s = _mk_streamer()
    s.enforce_integrity_gate = True
    s.run_id = "RUNID"
    s._open_intents_sent = set()
    s._signal_lineage_by_signal_id = {}
    s._signal_lineage_by_client_order_id = {
        "RUNID|OPEN|1": {
            "run_id": "RUNID",
            "bar_ts": "2026-04-01T08:30:00-06:00",
            "action": "OPEN",
            "side": "LONG",
            "gate_pass": True,
            "signal_id": "sig|fallback",
            "client_order_id": "RUNID|OPEN|1",
        }
    }
    intent = ExecutionIntent(
        intent_id="RUNID|OPEN|1",
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES",
        exec_instrument="ES 06-26",
        account=None,
        bar_ts="2026-04-01T08:30:00-06:00",
        model_price=6200.0,
        model_stop_price=6192.0,
        model_target_price=6208.0,
        signal_id=None,
    )
    reason = s._validate_open_lineage(intent)
    assert reason is None
    assert s._lineage_fallback_used is True


def test_desync_toggle_triggers_kill_switch() -> None:
    s = _mk_streamer()
    events = []
    disarms = []
    flattens = []
    s._log_exec_event = lambda payload: events.append(payload)
    s._set_entries_disarmed = lambda reason, detail=None: disarms.append((reason, detail))
    s._send_nt_flatten = lambda cid, reason=None: flattens.append((cid, reason))
    s._desync_toggle_threshold = 4
    s._desync_toggle_window_sec = 60.0

    s._set_position_state("IN_POSITION_UNPROTECTED", cid="CID-1", signal_id="SIG-1")
    s._set_position_state("FLAT", cid="CID-1", signal_id="SIG-1")
    s._set_position_state("IN_POSITION_UNPROTECTED", cid="CID-1", signal_id="SIG-1")
    s._set_position_state("FLAT", cid="CID-1", signal_id="SIG-1")

    assert s._desync_latch_active is True
    assert any(r == "desync_kill_switch" for r, _ in disarms)
    assert flattens


def test_desync_latch_blocks_reopen_after_flat() -> None:
    s = _mk_streamer()
    s._desync_latch_active = True
    s.state.position_state = "FLAT"
    s._set_position_state("IN_POSITION_UNPROTECTED", cid="CID-1", signal_id="SIG-1")
    assert s.state.position_state == "FLAT"


def test_nt_queue_overflow_sets_degraded_disarm() -> None:
    s = _mk_streamer()
    disarms = []
    s._set_entries_disarmed = lambda reason, detail=None: disarms.append((reason, detail))
    s._nt_dispatcher_enabled = True
    s._nt_event_queue = queue.Queue(maxsize=1)
    s._nt_event_queue_max = 1
    s._nt_event_queue_degrade_threshold = 1
    s._nt_event_queue.put_nowait({"type": "HEARTBEAT"})

    s._handle_nt_message({"type": "POSITION_SNAPSHOT"})

    assert s._nt_event_queue_degraded is True
    assert any(r == "nt_event_queue_degraded" for r, _ in disarms)
