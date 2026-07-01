import types

import pandas as pd


def _make_streamer_base(tmp_path):
    from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer, StreamState

    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer._ensure_compat_defaults = lambda: None
    streamer.nt_enabled = True
    streamer.run_mode = "live"
    streamer.time_mode = "wall_clock"
    streamer._sim_clock_ts = None
    streamer.session_tz = "America/Denver"
    streamer.state = StreamState()
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer.execution_state = types.SimpleNamespace(position_qty=1, position_side="LONG")
    streamer.execution_ledger = {}
    streamer.nt_account_mode = "none"
    streamer.nt_account = None
    streamer._last_csv_bar = "2026-02-17T08:00:00-07:00"
    streamer.bar_interval_sec = 300.0
    streamer.max_bar_age_seconds_for_exec = 1.0
    streamer._tail_mode = False

    streamer.out_dir = tmp_path
    streamer.exec_events_path = tmp_path / "exec_events.jsonl"
    streamer.order_events_path = tmp_path / "order_events.jsonl"

    streamer._emit_execution_decision_record = lambda *args, **kwargs: None
    streamer._record_executor_decision = lambda *args, **kwargs: None
    streamer._emit_executor_decision = lambda *args, **kwargs: None
    streamer._emit_block_event = lambda *args, **kwargs: None
    streamer._emit_unblocked = lambda *args, **kwargs: None
    streamer._set_hard_lockout = lambda *args, **kwargs: None
    streamer._build_nt_client_order_id = lambda ev: ev.get("client_order_id") or "cid|1"
    streamer._should_block_nt_send_policy = lambda *args, **kwargs: None
    streamer._log_order_event_records = []
    streamer._log_exec_event_records = []
    streamer._log_order_event = lambda cid, status, payload=None: streamer._log_order_event_records.append((cid, status, payload))
    streamer._log_exec_event = lambda payload: streamer._log_exec_event_records.append(dict(payload))

    streamer._nt_send_with_wait = lambda *args, **kwargs: True
    streamer._resolve_exec_instrument = lambda *args, **kwargs: ("ES MAR26", None)
    streamer._round_to_tick = lambda x: x
    streamer.tick_size = 0.25
    streamer.protection_price_mode = "offset"
    streamer.nt_instrument = "ES MAR26"
    streamer.exec_instrument = "ES MAR26"
    streamer.instrument_alias = "ES"
    streamer.nt_adapter = "nt"
    streamer.nt_bridge = types.SimpleNamespace(send=lambda payload: True, handshake_ok=lambda: True)
    streamer.nt_exec_state = "ARMED"

    return streamer


def test_stale_bar_evidence_uses_bar_ts_when_datetime_missing(tmp_path):
    from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer

    streamer = _make_streamer_base(tmp_path)
    # Fix "now" so evidence is deterministic.
    streamer._now_utc = lambda: pd.Timestamp("2026-02-17T15:00:00Z")
    streamer._last_csv_bar = "2026-02-17T08:00:00-07:00"  # effectively "now-ish" local
    streamer.max_bar_age_seconds_for_exec = 60.0

    # Missing datetime, but bar_ts is old.
    ev = {"type": "OPEN", "bar_ts": "2026-02-17T07:30:00-07:00"}
    evidence = LiveCSVStreamer._stale_bar_evidence(streamer, ev)
    assert evidence is not None
    assert evidence["bar_ts"].startswith("2026-02-17T07:30:00")


def test_close_is_not_stale_blocked_and_emits_override_audit(tmp_path, monkeypatch):
    from na.discord_addons.cli.stream_live_csv import ExecutionIntent, LiveCSVStreamer

    streamer = _make_streamer_base(tmp_path)
    streamer._inflight_buffer_enabled = True
    streamer.state.position_state = "EXITING"
    streamer.max_bar_age_seconds_for_exec = 1.0
    streamer._now_utc = lambda: pd.Timestamp("2026-02-17T15:00:00Z")

    # Ensure CLOSE is not buffered just because we're EXITING.
    streamer._buffer_inflight_intent = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CLOSE should not be buffered when EXITING"))

    # Ensure stale-block is not invoked for CLOSE.
    streamer._should_block_stale_nt_send = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CLOSE should bypass stale-bar blocking"))

    captured = {}

    def _execute_intent(intent):
        captured["intent"] = intent
        return types.SimpleNamespace(decision="SENT", reason_code="sent", nt_order_ids=())

    streamer.execute_intent = _execute_intent

    streamer._build_execution_intent = lambda ev: (
        ExecutionIntent(
            intent_id=ev.get("client_order_id") or "cid|close",
            action="CLOSE",
            side="NONE",
            qty=1,
            instrument_raw="ES",
            exec_instrument="ES MAR26",
            account=None,
            bar_ts=str(ev.get("datetime") or ev.get("bar_ts") or ""),
            model_price=None,
            model_stop_price=None,
            model_target_price=None,
            signal_id=ev.get("signal_id"),
        ),
        None,
    )

    ev = {
        "type": "CLOSE",
        "client_order_id": "cid|close",
        # Very stale bar timestamp should still allow exit transmit.
        "bar_ts": "2026-02-17T07:30:00-07:00",
    }
    LiveCSVStreamer._maybe_send_nt_order(streamer, ev)

    assert captured.get("intent") is not None
    assert any(rec.get("event") == "stale_exit_override" for rec in streamer._log_exec_event_records)
    assert not any(status == "blocked_stale_bar" for _, status, _ in streamer._log_order_event_records)


def test_inflight_flip_ttl_drop(tmp_path, monkeypatch):
    from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer

    streamer = _make_streamer_base(tmp_path)
    streamer._inflight_buffer_enabled = True
    streamer.state.position_state = "EXITING"

    # Buffer a FLIP intent.
    now = 2000.0
    monkeypatch.setattr("time.time", lambda: now)
    flip_ev = {"type": "FLIP", "client_order_id": "cid|flip", "bar_ts": "2026-02-17T08:05:00-07:00"}
    LiveCSVStreamer._buffer_inflight_intent(streamer, flip_ev, reason="inflight_pending")

    # Make it stale past TTL (2*300 + 30 = 630s).
    now = 2000.0 + 631.0
    streamer.state.position_state = "FLAT"

    streamer._maybe_send_nt_order = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Stale FLIP should be dropped, not sent"))

    LiveCSVStreamer._flush_inflight_buffer(streamer)

    assert any(rec.get("event") == "inflight_dropped_stale" for rec in streamer._log_exec_event_records)
