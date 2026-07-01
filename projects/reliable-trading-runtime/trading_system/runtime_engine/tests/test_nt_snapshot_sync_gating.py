import threading
import time
from typing import Any, Dict, List

import pytest

from trading_system.runtime_engine.integrations.cli.live_trading_runtime import LiveCSVStreamer, StreamState, utc_ts


class _FakeNTBridge:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    def send(self, payload: Dict[str, Any]) -> bool:
        self.sent.append(dict(payload))
        return True

    def handshake_ok(self) -> bool:
        return True


def _make_streamer(tmp_path):
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.nt_enabled = True
    streamer.nt_bridge = _FakeNTBridge()
    streamer.nt_adapter = "nt"
    streamer.nt_account_mode = "auto"
    streamer.nt_account = "auto"
    streamer.nt_account_autoresolve = True
    streamer.nt_account_autoresolve_policy = "use_first_valid"
    streamer.nt_instrument = "ES MAR26"
    streamer.instrument_alias = "ES"
    streamer.exec_instrument = "ES MAR26"
    streamer.exec_instrument_source = "cli"
    streamer.nt_port = 5016
    streamer.nt_require_snapshot = True
    streamer.nt_snapshot_timeout_sec = 1.0
    streamer.nt_strict_protocol = False
    streamer.nt_reset_lockout_on_connect = False
    streamer._nt_lockout_reminder_sent = False
    streamer._nt_dispatcher_enabled = False
    streamer._nt_snapshot_processing_disabled = False
    streamer._nt_snapshot_seen = False
    streamer._nt_account_chosen = None
    streamer._nt_account_configured = "auto"
    streamer._nt_account_detected = None
    streamer._nt_account_detected_source = None
    streamer.state = StreamState()
    streamer.out_dir = tmp_path
    streamer.exec_events_path = tmp_path / "exec_events.jsonl"
    streamer.order_events_path = tmp_path / "order_events.jsonl"
    streamer.signal_to_order_path = tmp_path / "signal_to_order.jsonl"
    streamer.entry_blocks_path = tmp_path / "entry_blocks.jsonl"
    streamer.emission_ledger_path = tmp_path / "emission_ledger.jsonl"
    streamer.gating_events_path = tmp_path / "gating_events.jsonl"
    streamer._log_order_event = lambda *args, **kwargs: None
    streamer._log_exec_event = lambda *args, **kwargs: None
    streamer._emit_block_event = lambda *args, **kwargs: None
    streamer._emit_event = lambda *args, **kwargs: None
    streamer._set_hard_lockout = lambda *args, **kwargs: None
    streamer._nt_order_state = {}
    streamer._update_nt_order_state = lambda *args, **kwargs: None
    return streamer


def test_nt_sync_emits_global_and_instrument(tmp_path):
    streamer = _make_streamer(tmp_path)

    streamer._wait_for_nt_snapshot(reason="test_sync", timeout_sec=0.4)

    sent = [msg for msg in streamer.nt_bridge.sent if msg.get("type") == "SYNC"]
    assert sent, "Expected SYNC requests to be sent"
    assert streamer._nt_snapshot_sync_sent_global is True
    assert streamer._nt_snapshot_sync_sent_inst is True
    assert any("instrument" not in msg or not msg.get("instrument") for msg in sent)
    assert any(msg.get("instrument") for msg in sent)


def test_nt_snapshot_arrives_after_sync(tmp_path):
    streamer = _make_streamer(tmp_path)

    def _inject_snapshot():
        time.sleep(0.2)
        snapshot = {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "instrument": "ES MAR26",
            "account": "Sim101",
            "timestamp": utc_ts(),
            "pos_qty": 0,
            "qty": 0,
            "side": "FLAT",
            "orders": [],
            "positions": [],
        }
        streamer._handle_nt_message(snapshot)

    t = threading.Thread(target=_inject_snapshot, daemon=True)
    t.start()
    streamer._wait_for_nt_snapshot(reason="test_snapshot", timeout_sec=1.0)
    t.join(timeout=1.0)

    assert streamer._nt_snapshot_seen is True
    assert int(getattr(streamer, "_nt_snapshot_received_count", 0) or 0) >= 1


def test_nt_snapshot_wait_drains_dispatcher_queue(tmp_path):
    streamer = _make_streamer(tmp_path)
    # Match production: inbound messages go to the queue.
    streamer._nt_dispatcher_enabled = True
    streamer._nt_event_queue_max = 100
    import queue as _queue

    streamer._nt_event_queue = _queue.Queue(maxsize=streamer._nt_event_queue_max)

    # Enqueue a snapshot via the public handler (this should NOT be processed yet).
    snapshot = {
        "type": "POSITION_SNAPSHOT",
        "protocol_version": 1,
        "instrument": "ES MAR26",
        "account": "Sim101",
        "timestamp": utc_ts(),
        "pos_qty": 0,
        "qty": 0,
        "side": "FLAT",
        "orders": [],
        "positions": [],
    }
    streamer._handle_nt_message(snapshot)
    assert streamer._nt_snapshot_seen is False

    # The wait loop should drain the queue and observe the snapshot.
    streamer._wait_for_nt_snapshot(reason="test_dispatcher_drain", timeout_sec=0.5)

    assert streamer._nt_snapshot_seen is True
    assert int(getattr(streamer, "_nt_snapshot_received_count", 0) or 0) >= 1


def test_nt_snapshot_timeout_with_ack_diagnostics(tmp_path):
    streamer = _make_streamer(tmp_path)

    def _inject_acks():
        time.sleep(0.2)
        ack_global = {
            "type": "ACK",
            "protocol_version": 1,
            "client_order_id": "SAFETY|UNKNOWN|20260206000000|SYNC",
            "timestamp": utc_ts(),
        }
        ack_inst = {
            "type": "ACK",
            "protocol_version": 1,
            "client_order_id": "SAFETY|ES MAR26|20260206000001|SYNC",
            "timestamp": utc_ts(),
        }
        streamer._handle_nt_message(ack_global)
        streamer._handle_nt_message(ack_inst)

    t = threading.Thread(target=_inject_acks, daemon=True)
    t.start()
    streamer._wait_for_nt_snapshot(reason="test_timeout", timeout_sec=0.6)
    t.join(timeout=1.0)

    assert streamer._nt_snapshot_seen is False
    assert streamer._nt_snapshot_sync_acked_global is True
    assert streamer._nt_snapshot_sync_acked_inst is True
    rx_counts = dict(getattr(streamer, "_nt_snapshot_rx_counts", {}) or {})
    assert rx_counts.get("ACK", 0) >= 2
    last_msgs = list(getattr(streamer, "_nt_snapshot_last_msgs", []) or [])
    assert any(item.get("type") == "ACK" for item in last_msgs)


def test_nt_snapshot_timestamp_numeric_string(tmp_path):
    streamer = _make_streamer(tmp_path)
    snapshot = {
        "type": "POSITION_SNAPSHOT",
        "protocol_version": 1,
        "instrument": "ES MAR26",
        "account": "Sim101",
        "timestamp": "1738860000",
        "pos_qty": 0,
        "qty": 0,
        "side": "FLAT",
        "orders": [],
        "positions": [],
    }
    streamer._handle_nt_message(snapshot)
    assert streamer._nt_snapshot_seen is True
    assert streamer._nt_last_snapshot_ts is not None


def test_global_snapshot_marks_seen_when_flat(tmp_path):
    streamer = _make_streamer(tmp_path)
    snapshot = {
        "type": "POSITION_SNAPSHOT",
        "protocol_version": 1,
        "timestamp": utc_ts(),
        "pos_qty": 0,
        "qty": 0,
        "side": "FLAT",
        "orders": [],
        "positions": [],
    }
    streamer._handle_nt_message(snapshot)
    assert streamer._nt_snapshot_seen is True


def test_snapshot_instrument_stays_on_target_when_global_feed_has_other_instruments(tmp_path):
    streamer = _make_streamer(tmp_path)

    def _snapshot(inst: str) -> Dict[str, Any]:
        return {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "instrument": inst,
            "account": "Sim101",
            "timestamp": utc_ts(),
            "pos_qty": 0,
            "qty": 0,
            "side": "FLAT",
            "orders": [],
            "positions": [],
        }

    # First snapshot can seed instrument context.
    streamer._handle_nt_message(_snapshot("MES MAR26"))
    assert streamer._nt_last_snapshot_instrument == "MES MAR26"

    # Matching snapshot should take over readiness instrument.
    streamer._handle_nt_message(_snapshot("ES MAR26"))
    assert streamer._nt_last_snapshot_instrument == "ES MAR26"

    # Subsequent non-target snapshots must not overwrite readiness instrument.
    streamer._handle_nt_message(_snapshot("MES MAR26"))
    assert streamer._nt_last_snapshot_instrument == "ES MAR26"


def test_nt_hello_applies_auto_flatten_flag(tmp_path):
    streamer = _make_streamer(tmp_path)
    streamer.auto_flatten_enabled = True
    streamer.protection_repair_enabled = True

    hello = {
        "type": "HELLO",
        "protocol_version": 1,
        "platform": "NinjaTrader8",
        "flags": {
            "protection_repair_enabled": False,
            "stop_update_enabled": False,
            "auto_flatten_enabled": False,
        },
        "timestamp": utc_ts(),
    }
    streamer._handle_nt_message(hello)

    assert streamer.protection_repair_enabled is False
    assert streamer.stop_update_enabled is False
    assert streamer.auto_flatten_enabled is False


def test_nt_hello_blocks_live_money_when_required_safety_flags_missing(tmp_path):
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer.nt_exec_policy = "live"
    streamer._nt_account_chosen = "DEMO6730705"
    streamer.protection_repair_requested = True
    streamer.auto_flatten_requested = True
    streamer.stop_update_requested = True
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, evidence=None, **_kwargs: lockouts.append((code, evidence or {}))

    streamer._handle_nt_message(
        {
            "type": "HELLO",
            "protocol_version": 1,
            "platform": "NinjaTrader8",
            "flags": {
                "protection_repair_enabled": False,
                "stop_update_enabled": False,
                "auto_flatten_enabled": False,
            },
            "timestamp": utc_ts(),
        }
    )

    assert streamer._nt_safety_capabilities_ok is False
    assert {w["flag"] for w in streamer._nt_safety_capability_warnings} == {
        "protection_repair_enabled",
        "stop_update_enabled",
        "auto_flatten_enabled",
    }
    assert lockouts and lockouts[-1][0] == "nt_safety_capabilities_unavailable"
