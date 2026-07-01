from discord_addons.cli.stream_live_csv import (
    LiveCSVStreamer,
    _resolve_execution_intended_mode,
    _safety_escalation_state,
    _strict_intent_parity_is_hard_block,
)


def test_live_mode_requires_nt_enabled_for_execution_intended() -> None:
    assert _resolve_execution_intended_mode(
        run_mode="live",
        nt_enabled=True,
        replay_execution_intended=False,
    )
    assert not _resolve_execution_intended_mode(
        run_mode="live",
        nt_enabled=False,
        replay_execution_intended=True,
    )


def test_replay_mode_respects_intent_only_when_nt_enabled() -> None:
    assert _resolve_execution_intended_mode(
        run_mode="replay",
        nt_enabled=True,
        replay_execution_intended=True,
    )
    assert not _resolve_execution_intended_mode(
        run_mode="replay",
        nt_enabled=True,
        replay_execution_intended=False,
    )
    assert not _resolve_execution_intended_mode(
        run_mode="replay",
        nt_enabled=False,
        replay_execution_intended=True,
    )


def test_strict_parity_marks_contract_violation_pre_send_as_hard_block() -> None:
    assert _strict_intent_parity_is_hard_block("contract_violation_pre_send")


def test_reject_diagnostics_exposes_scope_and_recoverability() -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.nt_instrument = "ES JUN26"
    diag = LiveCSVStreamer._build_reject_diagnostics(
        streamer,
        cid="SAFETY|RUN|ES|missing_stop|x",
        status="REJECTED",
        reason="missing_stop_price",
        reject_code="nt_missing_stop_price",
        state={"intent_action": "OPEN", "side": "LONG", "qty": 1, "instrument": "ES JUN26"},
        msg={},
    )
    assert diag["reject_scope"] == "protection"
    assert diag["reject_recoverability"] == "recoverable"
    assert diag["reject_class"] == "nt_protection_repair_recoverable"


def test_effective_protection_timeout_adaptive_extends_under_queue_pressure() -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.protection_timeout_sec = 30.0
    streamer.protection_timeout_policy = "adaptive"
    streamer.protection_timeout_adaptive_cap_sec = 45.0
    streamer._nt_event_queue_depth_peak = 1200
    streamer._nt_event_queue_coalesce_count = 20
    streamer._nt_event_queue_backpressure_active = True
    assert LiveCSVStreamer._effective_protection_timeout_sec(streamer) == 44.0


def test_pre_send_parity_contract_allows_flatten_without_stop_target() -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)

    class _Intent:
        action = "FLATTEN"
        intent_id = "CID-1"

    violation = LiveCSVStreamer._pre_send_parity_contract_violation(
        streamer,
        intent=_Intent(),
        order={
            "schema_version": 1,
            "session_id": "RUN-1",
            "client_order_id": "CID-1",
            "instrument": "ES JUN26",
            "qty": 1,
        },
    )
    assert violation is None


def test_safety_escalation_state_normalization() -> None:
    assert _safety_escalation_state("missing_stop_soft_failed") == "soft_failed"
    assert _safety_escalation_state("protection_timeout_retry") == "retrying"
    assert _safety_escalation_state("repair_exhausted") == "hard_failed"


def test_runtime_safety_rows_are_excluded_from_strict_parity_counts() -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.strict_intent_parity_enabled = True
    streamer._strict_intent_parity_counts = {"intents_total": 7, "mismatches_total": 2, "mismatches_by_reason": {}}

    payload = {
        "run_mode": "live",
        "phase": "LIVE",
        "execution_phase_allows": True,
        "intent_origin": "runtime_safety",
        "client_order_id": "SAFETY|RUN|ES|protection_repair_failed|x|a1",
        "decision": "REJECTED",
        "reason_code": "nt_broker_reject",
    }
    LiveCSVStreamer._record_strict_intent_parity(streamer, payload)
    assert streamer._strict_intent_parity_counts["intents_total"] == 7
    assert streamer._strict_intent_parity_counts["mismatches_total"] == 2
