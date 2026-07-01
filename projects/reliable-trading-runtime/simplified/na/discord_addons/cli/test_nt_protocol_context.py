from discord_addons.cli.stream_live_csv import LiveCSVStreamer


def test_ensure_nt_protocol_context_recovers_session_id() -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.nt_instrument = "MES"
    streamer._nt_instrument_for_tx = lambda: "MES 06-26"
    streamer._log_exec_event = lambda payload: None
    streamer._send_nt_sync_request = lambda **kwargs: True
    streamer._require_fresh_snapshot = lambda **kwargs: True
    state = {"session": ""}
    streamer._current_nt_session_id = lambda: state["session"]

    order = {"type": "ORDER", "instrument": "MES 06-26", "session_id": ""}

    def _refresh(**kwargs):
        state["session"] = "sess-123"
        return True

    streamer._send_nt_sync_request = _refresh
    ok = streamer._ensure_nt_protocol_context(
        order,
        context="unit_test",
        client_order_id="cid-1",
        max_refresh_attempts=2,
    )
    assert ok is True
    assert order["session_id"] == "sess-123"


def test_build_reject_diagnostics_flags_missing_session_id() -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.nt_instrument = "MES"
    detail = streamer._build_reject_diagnostics(
        cid="cid-2",
        status="REJECTED",
        reason="",
        reject_code="nt_broker_reject",
        state={"intent_action": "CLOSE", "instrument": "MES 06-26"},
        msg={"schema_version": 1, "session_id": "", "instrument": "MES 06-26"},
    )
    assert detail["reject_class"] == "nt_protocol_context_missing_session_id"
    assert detail["session_id_present"] is False


def test_build_reject_diagnostics_preserves_explicit_busy_reason() -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.nt_instrument = "MES"
    detail = streamer._build_reject_diagnostics(
        cid="cid-3",
        status="REJECTED",
        reason="instrument_busy_recovered",
        reject_code="nt_broker_reject",
        state={"intent_action": "BUY", "instrument": "MES 06-26"},
        msg={"schema_version": 1, "session_id": "", "instrument": "MES 06-26"},
    )
    assert detail["reject_class"] == "nt_instrument_busy_recovered"
    assert detail["session_id_present"] is False
