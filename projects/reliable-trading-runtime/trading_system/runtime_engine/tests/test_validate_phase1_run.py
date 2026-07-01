from __future__ import annotations

import json
from pathlib import Path

from trading_system.development_tools.validate_phase1_run import validate_run


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_ok_run(run_dir: Path) -> None:
    run_dir.mkdir()
    _write_json(run_dir / "status.json", {"verdict": "ok", "position_state": "FLAT", "working_order_count": 0})
    _write_json(run_dir / "run_health_summary.json", {"verdict": "ok"})
    _write_json(run_dir / "stream_state.json", {"position": {"pos": 0}})
    _write_json(
        run_dir / "resolved_config.json",
        {"force_open_policy": {"enabled": False}, "legacy_gate_bypass_allowed": False},
    )
    (run_dir / "diagnostics.log").write_text("ok\n", encoding="utf-8")
    (run_dir / "exec_events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "execution_ledger.jsonl").write_text("", encoding="utf-8")
    (run_dir / "trades.csv").write_text("client_order_id,entry_ts\n", encoding="utf-8")


def test_validator_old_artifact_entry_id_missing_after_fill_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "diagnostics.log").write_text("ENTRY_ID_MISSING_AFTER_FILL\n", encoding="utf-8")
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "ENTRY_ID_MISSING_AFTER_FILL" in codes


def test_validator_setup_bypass_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    _write_json(run_dir / "status.json", {"verdict": "ok", "position_state": "FLAT", "working_order_count": 0, "legacy_gate_bypassed": True})
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "LEGACY_SETUP_BYPASS_ACTIVE" in codes


def test_validator_unsafe_health_verdict_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    _write_json(run_dir / "run_health_summary.json", {"verdict": "unsafe"})
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "HEALTH_VERDICT_UNSAFE" in codes


def test_validator_open_missing_target_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "execution_ledger.jsonl").write_text(
        json.dumps({"intent_action": "OPEN", "stop_price": 7200.0, "target_price": None}) + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "OPEN_MISSING_TARGET_PRICE" in codes


def test_validator_open_missing_stop_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "execution_ledger.jsonl").write_text(
        json.dumps({"intent_action": "OPEN", "stop_price": None, "target_price": 7210.0}) + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "OPEN_MISSING_STOP_PRICE" in codes


def test_validator_bad_time_in_trade_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(json.dumps({"event": "BAD_TIME_IN_TRADE"}) + "\n", encoding="utf-8")
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "BAD_TIME_IN_TRADE" in codes


def test_validator_desync_without_resync_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        json.dumps({"event": "BROKER_INTERNAL_POSITION_DESYNC"}) + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "DESYNC_WITHOUT_RESYNC_OR_DISARM" in codes


def test_validator_flat_with_working_orders_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    _write_json(run_dir / "status.json", {"verdict": "ok", "position_state": "FLAT", "working_order_count": 2})
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "FLAT_WITH_WORKING_ORDERS" in codes


def test_validator_valid_protected_in_position_passes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    _write_json(run_dir / "status.json", {"verdict": "ok", "position_state": "IN_POSITION_PROTECTED", "working_order_count": 2})
    _write_json(run_dir / "stream_state.json", {"position": {"pos": 1}})
    (run_dir / "execution_ledger.jsonl").write_text(
        json.dumps({"intent_action": "OPEN", "stop_price": 7200.0, "target_price": 7210.0}) + "\n",
        encoding="utf-8",
    )
    failures = validate_run(run_dir)
    codes = {f["code"] for f in failures}
    assert "IN_POSITION_UNPROTECTED" not in codes
    assert "OPEN_MISSING_STOP_PRICE" not in codes
    assert "OPEN_MISSING_TARGET_PRICE" not in codes


def test_validator_valid_clean_flat_passes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    failures = validate_run(run_dir)
    assert failures == []


def test_validator_transport_mode_fails_clean_flat_no_trade(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    failures = validate_run(run_dir, require_order_lifecycle=True)
    codes = {f["code"] for f in failures}
    assert "TRANSPORT_MISSING_OPEN" in codes
    assert "TRANSPORT_MISSING_FILL" in codes


def test_validator_transport_mode_passes_full_lifecycle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    _write_json(run_dir / "status.json", {"verdict": "ok", "position_state": "FLAT", "working_order_count": 0})
    (run_dir / "execution_ledger.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"intent_action": "OPEN", "status": "SENT", "stop_price": 7200.0, "target_price": 7210.0}),
                json.dumps({"intent_action": "OPEN", "status": "ENTRY_FILLED", "stop_price": 7200.0, "target_price": 7210.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "POSITION_SNAPSHOT"}),
                json.dumps({"event": "PROTECTION_CONFIRMED_BY_BROKER_SNAPSHOT"}),
                json.dumps({"type": "ORDER_ACK"}),
                json.dumps({"type": "FILL"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    failures = validate_run(run_dir, require_order_lifecycle=True)
    assert failures == []


def test_validator_transport_mode_accepts_open_in_client_order_id_and_nested_payload_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "execution_ledger.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "client_order_id": "rid|preset|ES|2026-05-12T00:20:00-06:00|OPEN|LONG|abc",
                        "status": "sent",
                        "stop_price": 7200.0,
                        "target_price": 7210.0,
                    }
                ),
                json.dumps(
                    {
                        "client_order_id": "rid|preset|ES|2026-05-12T00:20:00-06:00|OPEN|LONG|abc",
                        "status": "entry_filled",
                        "stop_price": 7200.0,
                        "target_price": 7210.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"payload": {"type": "ORDER_ACK"}}),
                json.dumps({"payload": {"event_type": "FILL"}}),
                json.dumps({"payload": {"type": "POSITION_SNAPSHOT"}}),
                json.dumps({"event": "PROTECTION_CONFIRMED_BY_BROKER_SNAPSHOT"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    failures = validate_run(run_dir, require_order_lifecycle=True)
    assert failures == []


def test_validator_transport_mode_accepts_protected_flag_and_flat_from_health(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    _write_json(run_dir / "status.json", {"verdict": "ok", "working_order_count": 0})
    _write_json(run_dir / "run_health_summary.json", {"verdict": "ok", "position_state": "FLAT"})
    (run_dir / "stream_state.json").write_text(json.dumps({"position": {"pos": 0}}), encoding="utf-8")
    (run_dir / "execution_ledger.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"client_order_id": "rid|OPEN|x", "status": "sent", "stop_price": 1.0, "target_price": 2.0}),
                json.dumps({"client_order_id": "rid|OPEN|x", "status": "entry_filled", "stop_price": 1.0, "target_price": 2.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"payload": {"type": "ORDER_ACK"}}),
                json.dumps({"payload": {"type": "FILL"}}),
                json.dumps({"payload": {"type": "POSITION_SNAPSHOT"}}),
                json.dumps({"trade_pnl_state": {"protected_confirmed": True}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    failures = validate_run(run_dir, require_order_lifecycle=True)
    assert failures == []


def test_validator_ignores_trace_only_setup_fail(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        json.dumps(
            {
                "event": "ORDER_DECISION_TRACE",
                "stage": "before_append_signal",
                "setup_pass": False,
                "final_action": "OPEN",
                "order_sent": False,
                "order_send_attempted": False,
                "final_send_guard_passed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "SETUP_FAIL_ORDER_SENT" not in codes
    assert "SETUP_BLOCKED_BUT_SENT" not in codes


def test_validator_restored_mode_requires_restored_entry_event(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    _write_json(
        run_dir / "resolved_config.json",
        {
            "force_open_policy": {"enabled": False},
            "legacy_gate_bypass_allowed": False,
            "phase2_force_open_policy": {
                "enabled": True,
                "mode": "directional_bridge_restored",
                "legacy_gate_bypass_allowed": True,
                "allow_setup_fail_entries": True,
            },
        },
    )
    (run_dir / "exec_events.jsonl").write_text(
        json.dumps({"event": "ORDER_DECISION_TRACE", "setup_pass": False, "final_action": "OPEN", "order_sent": True, "side": "LONG", "bar_ts": "2026-05-12T12:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "SETUP_FAIL_ORDER_SENT" in codes


def test_validator_rejects_planned_only_counted_as_executed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "execution_ledger.jsonl").write_text(
        json.dumps({"intent_action": "OPEN", "stop_price": 7200.0, "target_price": 7210.0, "is_executed": True, "protection_status": "planned_only"}) + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "PLANNED_ONLY_COUNTED_AS_EXECUTED" in codes


def test_validator_fails_real_setup_fail_open(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        json.dumps({"event": "ORDER_INTENT_CREATED", "setup_pass": False, "final_action": "OPEN", "nt_adapter": "mock", "run_mode": "replay"})
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "SETUP_FAIL_ORDER_SENT" in codes


def test_validator_fails_signal_to_order_setup_fail_to_open(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "signal_to_order.jsonl").write_text(
        json.dumps(
            {
                "setup_pass": False,
                "final_action": "OPEN",
                "blocked_by": ["setup"],
                "decision": "SENT",
                "client_order_id": "cid-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "S2O_SETUP_FAIL_TO_OPEN" in codes
    assert "S2O_SETUP_BLOCKED_BUT_OPEN" in codes


def test_validator_fails_missing_terminal_classification_for_missing_stop_reject_chain(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "signal_to_order.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"client_order_id": "cid-2", "final_action": "SENT", "decision": "SENT"}),
                json.dumps({"client_order_id": "cid-2", "decision": "FILLED"}),
                json.dumps({"client_order_id": "cid-2", "decision": "REJECTED", "reason": "nt_missing_stop_price"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "MISSING_TERMINAL_CLASSIFIED_CAUSE_AFTER_SENT_REJECT_CHAIN" in codes


def test_validator_allows_recovered_reconcile_noise_for_run_owned_terminal_chain(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "signal_to_order.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"client_order_id": "rid|OPEN|runowned", "final_action": "SENT", "decision": "SENT"}),
                json.dumps({"client_order_id": "rid|OPEN|runowned", "decision": "FILLED"}),
                json.dumps({"client_order_id": "rid|OPEN|runowned", "decision": "REJECTED", "reason": "nt_missing_stop_price"}),
                json.dumps({"client_order_id": "rid|OPEN|runowned", "decision": "LOCKOUT", "reason": "cleanup_while_armed"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "recovered_fill_recorded_for_forensics", "client_order_id": "UNKNOWNCID|ES JUN26|POSITION", "fill_origin": "recovered_reconcile", "lifecycle_eligible": False}),
                json.dumps({"event": "FILL", "client_order_id": "rid|OPEN|runowned", "fill_origin": "run_owned", "lifecycle_eligible": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "MISSING_TERMINAL_CLASSIFIED_CAUSE_AFTER_SENT_REJECT_CHAIN" not in codes


def test_validator_allows_deterministic_fixture_setup_bypass_when_fully_scoped(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "QA_LIFECYCLE_SETUP_GATE_BYPASS_APPLIED"}),
                json.dumps(
                    {
                        "event": "ORDER_INTENT_CREATED",
                        "setup_pass": False,
                        "final_action": "OPEN",
                        "nt_adapter": "mock",
                        "run_mode": "replay",
                        "is_deterministic_order_lifecycle_fixture": True,
                        "fixture_bypass_applied": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "SETUP_FAIL_ORDER_SENT" not in codes
    assert "SETUP_BLOCKED_BUT_SENT" not in codes


def test_validator_rejects_fixture_bypass_on_real_adapter(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "QA_LIFECYCLE_SETUP_GATE_BYPASS_APPLIED"}),
                json.dumps(
                    {
                        "event": "ORDER_INTENT_CREATED",
                        "setup_pass": False,
                        "final_action": "OPEN",
                        "nt_adapter": "nt8",
                        "run_mode": "live",
                        "is_deterministic_order_lifecycle_fixture": True,
                        "fixture_bypass_applied": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "SETUP_FAIL_ORDER_SENT" in codes


def test_validator_accepts_aggressive_setup_fail_with_explicit_event(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    _write_json(
        run_dir / "resolved_config.json",
        {
            "phase2_force_open_policy": {
                "enabled": True,
                "mode": "directional_bridge_aggressive",
                "legacy_gate_bypass_allowed": True,
                "allow_setup_fail_entries": True,
            },
            "force_open_policy": {"enabled": True},
            "legacy_gate_bypass_allowed": True,
        },
    )
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "AGGRESSIVE_DIRECTIONAL_BRIDGE_ENTRY", "bar_ts": "2026-05-12T02:45:00-06:00", "side": "LONG"}),
                json.dumps({"event": "ORDER_INTENT_CREATED", "setup_pass": False, "final_action": "OPEN", "bar_ts": "2026-05-12T02:45:00-06:00", "side": "LONG"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "SETUP_FAIL_ORDER_SENT" not in codes
    assert "AGGRESSIVE_MODE_MISSING_ENTRY_EVENT" not in codes


def test_validator_rejects_aggressive_setup_fail_without_event(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    _write_json(
        run_dir / "resolved_config.json",
        {
            "phase2_force_open_policy": {
                "enabled": True,
                "mode": "directional_bridge_aggressive",
                "legacy_gate_bypass_allowed": True,
                "allow_setup_fail_entries": True,
            },
            "force_open_policy": {"enabled": True},
            "legacy_gate_bypass_allowed": True,
        },
    )
    (run_dir / "exec_events.jsonl").write_text(
        json.dumps({"event": "ORDER_INTENT_CREATED", "setup_pass": False, "final_action": "OPEN", "bar_ts": "2026-05-12T02:45:00-06:00", "side": "LONG"})
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "SETUP_FAIL_ORDER_SENT" in codes or "AGGRESSIVE_MODE_MISSING_ENTRY_EVENT" in codes


def test_validator_allows_aggressive_mode_without_setup_fail_send(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    _write_json(
        run_dir / "resolved_config.json",
        {
            "phase2_force_open_policy": {
                "enabled": True,
                "mode": "directional_bridge_aggressive",
                "legacy_gate_bypass_allowed": True,
                "allow_setup_fail_entries": True,
            },
            "force_open_policy": {"enabled": True},
            "legacy_gate_bypass_allowed": True,
        },
    )
    failures = validate_run(run_dir)
    codes = {f["code"] for f in failures}
    assert "AGGRESSIVE_MODE_MISSING_ENTRY_EVENT" not in codes


def test_validator_rejects_pnl_stair_action_before_fill(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "PROTECTION_CONFIRMED_BY_BROKER_SNAPSHOT"}),
                json.dumps({"event": "PNL_STAIR_STOP_UPDATE_REQUESTED", "side": "LONG", "stop_price": 7400.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "PNL_STAIR_ACTION_BEFORE_ENTRY_FILL" in codes


def test_validator_rejects_pnl_stair_action_before_protection(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "FILL"}),
                json.dumps({"event": "PNL_STAIR_STOP_UPDATE_REQUESTED", "side": "LONG", "stop_price": 7400.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "PNL_STAIR_ACTION_BEFORE_PROTECTION_CONFIRM" in codes


def test_validator_rejects_nonmonotonic_long_stair_stop(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "FILL"}),
                json.dumps({"event": "PROTECTION_CONFIRMED_BY_BROKER_SNAPSHOT"}),
                json.dumps({"event": "PNL_STAIR_STOP_UPDATE_REQUESTED", "side": "LONG", "stop_price": 7403.0}),
                json.dumps({"event": "PNL_STAIR_STOP_UPDATE_REQUESTED", "side": "LONG", "stop_price": 7402.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "PNL_STAIR_LONG_STOP_DECREASED" in codes


def test_validator_rejects_nonmonotonic_short_stair_stop(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "FILL"}),
                json.dumps({"event": "PROTECTION_CONFIRMED_BY_BROKER_SNAPSHOT"}),
                json.dumps({"event": "PNL_STAIR_STOP_UPDATE_REQUESTED", "side": "SHORT", "stop_price": 7397.0}),
                json.dumps({"event": "PNL_STAIR_STOP_UPDATE_REQUESTED", "side": "SHORT", "stop_price": 7398.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "PNL_STAIR_SHORT_STOP_INCREASED" in codes


def test_validator_rejects_stair_rejects_without_degrade(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "FILL"}),
                json.dumps({"event": "PROTECTION_CONFIRMED_BY_BROKER_SNAPSHOT"}),
                json.dumps({"event": "PNL_STAIR_STOP_UPDATE_REJECTED"}),
                json.dumps({"event": "PNL_STAIR_STOP_UPDATE_REJECTED"}),
                json.dumps({"event": "PNL_STAIR_STOP_UPDATE_REJECTED"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "PNL_STAIR_REJECTED_WITHOUT_DEGRADE" in codes


def test_validator_rejects_pnl_stair_action_after_flat(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "FILL"}),
                json.dumps({"event": "PROTECTION_CONFIRMED_BY_BROKER_SNAPSHOT"}),
                json.dumps({"event": "POSITION_SNAPSHOT", "snapshot": {"position_state": "FLAT", "position_qty": 0}}),
                json.dumps({"event": "PNL_STAIR_STOP_UPDATE_REQUESTED", "side": "LONG", "stop_price": 7400.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "PNL_STAIR_ACTION_AFTER_FLAT" in codes


def test_validator_rejects_pnl_stair_action_during_desync(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "FILL"}),
                json.dumps({"event": "PROTECTION_CONFIRMED_BY_BROKER_SNAPSHOT"}),
                json.dumps({"event": "BROKER_INTERNAL_POSITION_DESYNC"}),
                json.dumps({"event": "PNL_STAIR_STOP_UPDATE_REQUESTED", "side": "LONG", "stop_price": 7400.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "PNL_STAIR_ACTION_DURING_DESYNC" in codes


def test_validator_rejects_stair_force_exit_event(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "FILL"}),
                json.dumps({"event": "PROTECTION_CONFIRMED_BY_BROKER_SNAPSHOT"}),
                json.dumps({"event": "PNL_STAIR_FORCE_EXIT"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "PNL_STAIR_FORCE_EXIT_FORBIDDEN" in codes


def test_validator_fails_lockout_preserved_spam_over_threshold(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    _write_json(
        run_dir / "status.json",
        {
            "verdict": "ok",
            "position_state": "FLAT",
            "working_order_count": 0,
            "lockout_preserved_spam_threshold": 3,
        },
    )
    (run_dir / "exec_events.jsonl").write_text(
        "\n".join(json.dumps({"event": "lockout_preserved_first_cause"}) for _ in range(4)) + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "LOCKOUT_PRESERVED_SPAM" in codes


def test_validator_accepts_missing_stop_reject_chain_with_terminal_classification(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _base_ok_run(run_dir)
    (run_dir / "signal_to_order.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"client_order_id": "cid-ok", "decision": "SENT", "final_action": "OPEN"}),
                json.dumps({"client_order_id": "cid-ok", "decision": "REJECTED", "reason": "nt_missing_stop_price"}),
                json.dumps({"client_order_id": "cid-ok", "decision": "BLOCKED_SAFETY", "reason": "protection_repair_failed"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "exec_events.jsonl").write_text(
        json.dumps({"event": "entry_fill", "client_order_id": "cid-ok"}) + "\n",
        encoding="utf-8",
    )
    codes = {f["code"] for f in validate_run(run_dir)}
    assert "MISSING_TERMINAL_CLASSIFIED_CAUSE_AFTER_SENT_REJECT_CHAIN" not in codes
