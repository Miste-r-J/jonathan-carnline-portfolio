from trading_system.runtime_engine.integrations.cli.live_trading_runtime import LiveCSVStreamer


def _streamer_stub() -> LiveCSVStreamer:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer._nt_repair_state_by_instrument = {}
    return streamer


def test_repair_state_for_instrument_initializes_defaults() -> None:
    streamer = _streamer_stub()
    state = LiveCSVStreamer._repair_state_for_instrument(streamer, "MES 06-26", now_ts=100.0)
    assert state["start_ts"] == 100.0
    assert state["attempts"] == 0
    assert state["cooldown_until_ts"] is None
    assert state["repair_in_progress_until_ts"] is None
    assert state["stale_snapshot_confirmations"] == 0


def test_repair_snapshot_stale_for_protection_uses_freshness_threshold() -> None:
    streamer = _streamer_stub()
    streamer.nt_snapshot_fresh_sec = 30.0
    streamer.nt_snapshot_operational_stale_sec = 15.0
    streamer._snapshot_age_sec = lambda inst_key=None: 20.0

    stale, detail = LiveCSVStreamer._repair_snapshot_stale_for_protection(streamer, "MES 06-26")
    assert stale is True
    assert detail["fresh_threshold_sec"] == 15.0
    assert detail["reason"] == "snapshot_stale_for_repair"


def test_should_clear_repair_latch_after_flat_only_for_safety_and_lockout() -> None:
    streamer = _streamer_stub()
    streamer._hard_lockout_code = "protection_repair_failed"
    streamer._is_safety_close_reason = lambda reason: reason == "protection_repair_failed"

    assert LiveCSVStreamer._should_clear_repair_latch_after_flat(
        streamer,
        reason="protection_repair_failed",
        cid="SAFETY|RUN|MES|protection_repair_failed|x",
    )
    streamer._hard_lockout_code = "other_lockout"
    assert not LiveCSVStreamer._should_clear_repair_latch_after_flat(
        streamer,
        reason="protection_repair_failed",
        cid="SAFETY|RUN|MES|protection_repair_failed|x",
    )


def test_snapshot_sync_coherence_requires_position_and_order_windows() -> None:
    streamer = _streamer_stub()
    streamer.snapshot_age_max_sec = 450.0
    streamer.snapshot_sync_coherence_max_skew_sec = 3.0
    import time
    now = time.time()
    state = {
        "last_position_ts_epoch": now - 1.0,
        "last_order_ts_epoch": now - 0.2,
        "last_snapshot_ts_epoch": now - 0.5,
        "generation": 2,
        "last_seq": 12,
    }
    coherent, detail = LiveCSVStreamer._snapshot_sync_coherence(streamer, inst_key="MES 06-26", state=state)
    assert coherent is True
    assert detail["has_position_snapshot"] is True
    assert detail["has_order_snapshot"] is True


def test_snapshot_sync_mismatch_classifier_prefers_stale_then_lineage_then_qty() -> None:
    streamer = _streamer_stub()
    streamer.state = type("State", (), {"active_client_order_id": "CID-A"})()
    streamer._run_id = "RUN-A"
    assert (
        LiveCSVStreamer._snapshot_sync_mismatch_reason(
            streamer,
            missing_stop=True,
            missing_target=False,
            qty_mismatch=False,
            over_covered=False,
            snapshot_stale_for_repair=True,
            pos_qty=1.0,
            inst_key="MES 06-26",
            state={"last_run_id": "RUN-A", "last_sync_cid": "CID-A"},
        )
        == "snapshot_stale_for_reconcile"
    )
    assert (
        LiveCSVStreamer._snapshot_sync_mismatch_reason(
            streamer,
            missing_stop=False,
            missing_target=False,
            qty_mismatch=True,
            over_covered=False,
            snapshot_stale_for_repair=False,
            pos_qty=1.0,
            inst_key="MES 06-26",
            state={"last_run_id": "RUN-A", "last_sync_cid": "CID-A"},
        )
        == "position_qty_disagrees_with_fill_truth"
    )
