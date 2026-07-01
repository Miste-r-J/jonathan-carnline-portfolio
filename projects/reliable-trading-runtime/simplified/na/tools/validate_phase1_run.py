from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


FORBIDDEN_TEXT_CODES = {
    "SHELF_ARM_ATTEMPT_FORCE_EXIT": "SHELF_ARM_ATTEMPT_FORCE_EXIT",
    "ENTRY_ID_MISSING_AFTER_FILL": "ENTRY_ID_MISSING_AFTER_FILL",
    "repair_flatten_unattributed": "REPAIR_FLATTEN_UNATTRIBUTED",
    "BAD_TIME_IN_TRADE": "BAD_TIME_IN_TRADE",
    "active_fill_truth_missing": "ACTIVE_FILL_TRUTH_MISSING",
    "PNL_STAIR_FORCE_EXIT": "PNL_STAIR_FORCE_EXIT",
    "SHELF_FORCE_EXIT": "SHELF_FORCE_EXIT",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            payload = {"_raw": line}
        payload["_line"] = i
        rows.append(payload)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().replace(".", "", 1).isdigit()):
            x = float(value)
            ax = abs(x)
            if 1_000_000_000 <= ax <= 2_500_000_000:
                return pd.Timestamp(x, unit="s", tz="UTC")
            if 1_000_000_000_000 <= ax <= 2_500_000_000_000:
                return pd.Timestamp(x, unit="ms", tz="UTC")
            if 1_000_000_000_000_000_000 <= ax <= 2_500_000_000_000_000_000:
                return pd.Timestamp(x, unit="ns", tz="UTC")
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    except Exception:
        return None


def _fail(
    failures: list[dict[str, Any]],
    *,
    code: str,
    file: Path,
    line: int | None = None,
    event: str | None = None,
    trade_or_order_id: str | None = None,
    context: Any = None,
    suggested_fix_area: str = "stream_live_csv.py",
) -> None:
    payload: dict[str, Any] = {
        "code": code,
        "file": str(file),
        "suggested_fix_area": suggested_fix_area,
    }
    if line is not None:
        payload["line"] = int(line)
    if event:
        payload["event"] = event
    if trade_or_order_id:
        payload["trade_or_order_id"] = trade_or_order_id
    if context is not None:
        payload["context"] = context
    failures.append(payload)


def validate_run(run_dir: Path, *, require_order_lifecycle: bool = False) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    status_path = run_dir / "status.json"
    health_path = run_dir / "run_health_summary.json"
    state_path = run_dir / "stream_state.json"
    resolved_cfg_path = run_dir / "resolved_config.json"
    run_manifest_path = run_dir / "run_manifest.json"
    diagnostics_path = run_dir / "diagnostics.log"
    exec_events_path = run_dir / "exec_events.jsonl"
    execution_ledger_path = run_dir / "execution_ledger.jsonl"
    trades_path = run_dir / "trades.csv"
    signal_to_order_path = run_dir / "signal_to_order.jsonl"
    lifecycle_path = run_dir / "lifecycle_events.jsonl"

    status = _read_json(status_path)
    health = _read_json(health_path)
    stream_state = _read_json(state_path)
    resolved_cfg = _read_json(resolved_cfg_path)
    run_manifest = _read_json(run_manifest_path)
    exec_events = _read_jsonl(exec_events_path)
    execution_ledger = _read_jsonl(execution_ledger_path)
    trades = _read_csv(trades_path)
    signal_to_order = _read_jsonl(signal_to_order_path)
    lifecycle_rows = _read_jsonl(lifecycle_path)

    cli_argv = [str(x) for x in (run_manifest.get("cli_argv") or [])]
    deterministic_fixture_mode = bool(
        "--qa_emit_signals" in cli_argv
        and "basic_bracket" in cli_argv
        and "--run_mode" in cli_argv
        and "replay" in cli_argv
        and "--nt_adapter" in cli_argv
        and "mock" in cli_argv
        and "--replay_execution_intended" in cli_argv
    )

    # Verdict/state invariants.
    if str(status.get("verdict") or "").lower() == "unsafe" and not deterministic_fixture_mode:
        _fail(failures, code="VERDICT_UNSAFE", file=status_path, context={"verdict": status.get("verdict")})
    if str(health.get("verdict") or "").lower() == "unsafe" and not deterministic_fixture_mode:
        _fail(failures, code="HEALTH_VERDICT_UNSAFE", file=health_path, context={"verdict": health.get("verdict")})
    if str(status.get("position_state") or "").upper() == "IN_POSITION_UNPROTECTED":
        _fail(
            failures,
            code="IN_POSITION_UNPROTECTED",
            file=status_path,
            context={"position_state": status.get("position_state")},
        )
    if str(status.get("position_state") or "").upper() == "FLAT" and int(status.get("working_order_count") or 0) > 0:
        _fail(
            failures,
            code="FLAT_WITH_WORKING_ORDERS",
            file=status_path,
            context={"working_order_count": status.get("working_order_count")},
        )

    # Text-level forbidden markers.
    text_sources = [diagnostics_path, exec_events_path, execution_ledger_path]
    for src in text_sources:
        if not src.exists():
            continue
        lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines, start=1):
            for needle, code in FORBIDDEN_TEXT_CODES.items():
                if needle in line:
                    if deterministic_fixture_mode and needle in {"ENTRY_ID_MISSING_AFTER_FILL"}:
                        continue
                    _fail(
                        failures,
                        code=code,
                        file=src,
                        line=i,
                        event=needle,
                        context=line[:400],
                    )

    # Setup-gate / bypass checks.
    setup_fail_candidates = 0
    setup_block_events = 0
    aggressive_setup_fail_sent_attempts = 0
    fixture_bypass_seen = any(str(ev.get("event") or "") == "QA_LIFECYCLE_SETUP_GATE_BYPASS_APPLIED" for ev in exec_events)
    aggressive_events = [ev for ev in exec_events if str(ev.get("event") or "") == "AGGRESSIVE_DIRECTIONAL_BRIDGE_ENTRY"]
    restored_events = [ev for ev in exec_events if str(ev.get("event") or "") == "RESTORED_DIRECTIONAL_BRIDGE_ENTRY"]
    aggressive_policy = resolved_cfg.get("phase2_force_open_policy") if isinstance(resolved_cfg.get("phase2_force_open_policy"), dict) else {}
    aggressive_mode_enabled = bool(aggressive_policy.get("enabled")) and str(aggressive_policy.get("mode") or "").lower() == "directional_bridge_aggressive" and bool(aggressive_policy.get("legacy_gate_bypass_allowed")) and bool(aggressive_policy.get("allow_setup_fail_entries"))
    restored_mode_enabled = bool(aggressive_policy.get("enabled")) and str(aggressive_policy.get("mode") or "").lower() == "directional_bridge_restored" and bool(aggressive_policy.get("legacy_gate_bypass_allowed")) and bool(aggressive_policy.get("allow_setup_fail_entries"))

    def _trace_counts_as_sent(ev: dict[str, Any]) -> bool:
        if str(ev.get("event") or "") != "ORDER_DECISION_TRACE":
            return True
        stage = str(ev.get("stage") or ev.get("trace_stage") or "").lower()
        if bool(ev.get("order_sent")) or bool(ev.get("order_send_attempted")):
            return True
        if bool(ev.get("final_send_guard_passed")) and stage in {"final_send", "before_execute_intent", "after_execute_intent"}:
            return True
        return False

    for ev in exec_events:
        event = str(ev.get("event") or "")
        if event == "ORDER_BLOCKED_SETUP_FAIL":
            setup_block_events += 1
        setup_pass = ev.get("setup_pass")
        final_action = str(ev.get("final_action") or ev.get("action") or "").upper()
        blocked_by = str(ev.get("blocked_by") or "")
        sentish = final_action in {"OPEN", "FLIP", "SENT"}
        if sentish and not _trace_counts_as_sent(ev):
            sentish = False
        fixture_context_ok = bool(
            fixture_bypass_seen
            and str(ev.get("nt_adapter") or "").lower() == "mock"
            and str(ev.get("mode") or ev.get("run_mode") or "").lower() in {"replay", "offline"}
            and bool(ev.get("is_deterministic_order_lifecycle_fixture") or ev.get("fixture_bypass_applied"))
        )
        if setup_pass is False and sentish:
            if fixture_context_ok:
                continue
            if aggressive_mode_enabled:
                aggressive_setup_fail_sent_attempts += 1
                cid = str(ev.get("client_order_id") or ev.get("intent_id") or "")
                bar_ts = str(ev.get("bar_ts") or "")
                matched = False
                for ae in aggressive_events:
                    if cid and str(ae.get("client_order_id") or "") == cid:
                        matched = True
                        break
                    if bar_ts and str(ae.get("bar_ts") or "") == bar_ts and str(ae.get("side") or "").upper() == str(ev.get("side") or "").upper():
                        matched = True
                        break
                if matched:
                    continue
            if restored_mode_enabled:
                cid = str(ev.get("client_order_id") or ev.get("intent_id") or "")
                bar_ts = str(ev.get("bar_ts") or "")
                matched = False
                for re in restored_events:
                    if cid and str(re.get("client_order_id") or "") == cid:
                        matched = True
                        break
                    if bar_ts and str(re.get("bar_ts") or "") == bar_ts and str(re.get("side") or "").upper() == str(ev.get("side") or "").upper():
                        matched = True
                        break
                if matched:
                    continue
            setup_fail_candidates += 1
            _fail(
                failures,
                code="SETUP_FAIL_ORDER_SENT",
                file=exec_events_path,
                line=ev.get("_line"),
                event=event or "unknown",
                trade_or_order_id=str(ev.get("client_order_id") or ev.get("intent_id") or ""),
                context={"setup_pass": setup_pass, "final_action": final_action, "blocked_by": blocked_by},
            )
        if "setup" in blocked_by.lower() and sentish:
            if fixture_context_ok:
                continue
            _fail(
                failures,
                code="SETUP_BLOCKED_BUT_SENT",
                file=exec_events_path,
                line=ev.get("_line"),
                event=event or "unknown",
                context={"blocked_by": blocked_by, "final_action": final_action},
            )
    if setup_fail_candidates > 0 and setup_block_events == 0:
        _fail(
            failures,
            code="MISSING_ORDER_BLOCKED_SETUP_FAIL",
            file=exec_events_path if exec_events_path.exists() else status_path,
            context={"setup_fail_candidates": setup_fail_candidates},
        )
    if aggressive_mode_enabled and aggressive_setup_fail_sent_attempts > 0 and not aggressive_events:
        _fail(
            failures,
            code="AGGRESSIVE_MODE_MISSING_ENTRY_EVENT",
            file=exec_events_path if exec_events_path.exists() else status_path,
            context={"phase2_force_open_policy": aggressive_policy},
        )
    if restored_mode_enabled and setup_fail_candidates > 0 and not restored_events:
        _fail(
            failures,
            code="RESTORED_MODE_MISSING_ENTRY_EVENT",
            file=exec_events_path if exec_events_path.exists() else status_path,
            context={"phase2_force_open_policy": aggressive_policy},
        )

    bypass_allowed = bool((resolved_cfg.get("force_open_policy") or {}).get("enabled")) and bool(
        resolved_cfg.get("legacy_gate_bypass_allowed")
    )
    if bool(status.get("legacy_gate_bypassed")) and not bypass_allowed:
        _fail(
            failures,
            code="LEGACY_SETUP_BYPASS_ACTIVE",
            file=status_path,
            context={
                "legacy_gate_bypassed": status.get("legacy_gate_bypassed"),
                "force_open_policy_enabled": (resolved_cfg.get("force_open_policy") or {}).get("enabled"),
                "legacy_gate_bypass_allowed": resolved_cfg.get("legacy_gate_bypass_allowed"),
            },
        )

    # Signal-to-order setup invariants for run-folder replay audits.
    for row in signal_to_order:
        final_action = str(row.get("final_action") or row.get("action") or "").upper()
        setup_pass = row.get("setup_pass")
        blocked_by_txt = str(row.get("blocked_by") or "")
        if setup_pass is False and final_action in {"OPEN", "FLIP", "SENT"}:
            _fail(
                failures,
                code="S2O_SETUP_FAIL_TO_OPEN",
                file=signal_to_order_path,
                line=row.get("_line"),
                event=str(row.get("decision") or ""),
                trade_or_order_id=str(row.get("client_order_id") or row.get("signal_id") or ""),
                context={"setup_pass": setup_pass, "final_action": final_action, "blocked_by": row.get("blocked_by")},
            )
        if "setup" in blocked_by_txt.lower() and final_action in {"OPEN", "FLIP", "SENT"}:
            _fail(
                failures,
                code="S2O_SETUP_BLOCKED_BUT_OPEN",
                file=signal_to_order_path,
                line=row.get("_line"),
                event=str(row.get("decision") or ""),
                trade_or_order_id=str(row.get("client_order_id") or row.get("signal_id") or ""),
                context={"blocked_by": row.get("blocked_by"), "final_action": final_action},
            )

    # Fiber ingest classification invariants.
    ingress_stall_classified = any(str(ev.get("event") or "") == "FIBER_INGRESS_STALLED_CLASSIFIED" for ev in exec_events)
    timeout_warn_or_disarm = any(
        str(ev.get("event") or "") in {"fiber_bar_timeout_warn", "fiber_bar_timeout_disarm", "bar_ingress_silence"}
        for ev in exec_events
    )
    if timeout_warn_or_disarm and not ingress_stall_classified:
        _fail(
            failures,
            code="FIBER_TIMEOUT_UNCLASSIFIED",
            file=exec_events_path if exec_events_path.exists() else status_path,
            context={"requires_event": "FIBER_INGRESS_STALLED_CLASSIFIED"},
        )
    stale_gen_drops = [ev for ev in exec_events if str(ev.get("event") or "") == "FIBER_GEN_STALE_DROP"]
    if stale_gen_drops:
        for ev in stale_gen_drops:
            cg = ev.get("conn_generation")
            bad_accept = False
            for later in exec_events:
                if int(later.get("_line") or 0) <= int(ev.get("_line") or 0):
                    continue
                if str(later.get("event") or "") != "fiber_bar_rx":
                    continue
                if later.get("conn_generation") == cg:
                    bad_accept = True
                    break
            if bad_accept:
                _fail(
                    failures,
                    code="FIBER_STALE_GENERATION_ACCEPTED",
                    file=exec_events_path,
                    line=ev.get("_line"),
                    context={"conn_generation": cg},
                )
                break

    # SENT->REJECTED with missing protection after fill must end with a
    # terminal classified cause.
    sent_cids: set[str] = set()
    filled_cids: set[str] = set()
    missing_protection_reject_cids: set[str] = set()
    terminal_classified_cids: set[str] = set()
    terminal_tokens = {
        "cleanup_while_armed",
        "protection_repair_failed",
        "protection_contract_violation",
        "lockout_preserved_first_cause",
        "flatten",
        "flat_settled",
        "terminal_classified",
    }
    for row in signal_to_order:
        cid = str(row.get("client_order_id") or "").strip()
        if not cid:
            continue
        final_action = str(row.get("final_action") or row.get("action") or "").upper()
        decision = str(row.get("decision") or "").upper()
        reason = str(row.get("reason") or row.get("reason_code") or "").lower()
        blocked_txt = str(row.get("blocked_by") or "").lower()
        if final_action in {"OPEN", "FLIP", "SENT"} or decision in {"SENT", "EXECUTED"}:
            sent_cids.add(cid)
        if "fill" in decision.lower() or "filled" in reason:
            filled_cids.add(cid)
        if (
            "reject" in decision.lower()
            and ("missing_stop_price" in reason or "nt_missing_stop_price" in reason or "missing_stop_price" in blocked_txt)
        ):
            missing_protection_reject_cids.add(cid)
        if any(tok in reason for tok in terminal_tokens):
            terminal_classified_cids.add(cid)
    for ev in exec_events:
        cid = str(ev.get("client_order_id") or "").strip()
        if not cid:
            continue
        event = str(ev.get("event") or ev.get("type") or "").lower()
        status_txt = str(ev.get("status") or "").lower()
        reason = str(ev.get("reason") or ev.get("reason_code") or ev.get("broker_reason") or "").lower()
        if "fill" in event or "entry_filled" in status_txt:
            filled_cids.add(cid)
        if ("reject" in event or "rejected" in status_txt) and (
            "missing_stop_price" in reason or "nt_missing_stop_price" in reason
        ):
            missing_protection_reject_cids.add(cid)
        if any(tok in reason for tok in terminal_tokens):
            terminal_classified_cids.add(cid)
    for row in execution_ledger:
        cid = str(row.get("client_order_id") or "").strip()
        if not cid:
            continue
        status_txt = str(row.get("status") or "").lower()
        reason = str(row.get("reason") or row.get("reason_code") or "").lower()
        if status_txt in {"sent", "accepted", "ack"}:
            sent_cids.add(cid)
        if "fill" in status_txt:
            filled_cids.add(cid)
        if "reject" in status_txt and ("missing_stop_price" in reason or "nt_missing_stop_price" in reason):
            missing_protection_reject_cids.add(cid)
        if any(tok in reason for tok in terminal_tokens):
            terminal_classified_cids.add(cid)
    for cid in sorted(missing_protection_reject_cids):
        if cid in sent_cids and cid in filled_cids and cid not in terminal_classified_cids:
            _fail(
                failures,
                code="MISSING_TERMINAL_CLASSIFIED_CAUSE_AFTER_SENT_REJECT_CHAIN",
                file=signal_to_order_path if signal_to_order_path.exists() else exec_events_path,
                trade_or_order_id=cid,
            )

    # Repeated first-cause lockout messages should be bounded.
    lockout_preserved_events = [ev for ev in exec_events if str(ev.get("event") or "") == "lockout_preserved_first_cause"]
    spam_threshold = int(status.get("lockout_preserved_spam_threshold") or 10)
    if len(lockout_preserved_events) > spam_threshold:
        _fail(
            failures,
            code="LOCKOUT_PRESERVED_SPAM",
            file=exec_events_path,
            context={"count": len(lockout_preserved_events), "threshold": spam_threshold},
        )

    # Protection/bracket checks from ledger rows.
    runner_override = bool((resolved_cfg.get("risk") or {}).get("pnl_runner_allow_target_suppression_live"))
    for row in execution_ledger:
        status_txt = str(row.get("status") or "").lower()
        action_txt = str(row.get("intent_action") or row.get("action") or "").upper()
        if action_txt not in {"OPEN", "FLIP"} and status_txt not in {"open", "entry_filled", "sent"}:
            continue
        stop_price = row.get("stop_price")
        target_price = row.get("target_price")
        cid = str(row.get("client_order_id") or "")
        if stop_price in (None, ""):
            _fail(
                failures,
                code="OPEN_MISSING_STOP_PRICE",
                file=execution_ledger_path,
                line=row.get("_line"),
                trade_or_order_id=cid,
                context={"row_status": status_txt, "action": action_txt},
            )
        if target_price in (None, "") and not runner_override:
            _fail(
                failures,
                code="OPEN_MISSING_TARGET_PRICE",
                file=execution_ledger_path,
                line=row.get("_line"),
                trade_or_order_id=cid,
                context={"row_status": status_txt, "action": action_txt},
            )
        if str(row.get("is_executed") or "").lower() in {"true", "1", "yes"} and str(row.get("protection_status") or "").lower() == "planned_only":
            _fail(
                failures,
                code="PLANNED_ONLY_COUNTED_AS_EXECUTED",
                file=execution_ledger_path,
                line=row.get("_line"),
                trade_or_order_id=cid,
            )

    if require_order_lifecycle:
        has_open = False
        has_ack = False
        has_fill = False
        has_stop_submitted = False
        has_target_submitted = False
        has_protection_confirmed = False
        has_snapshot_position = False
        has_exit_or_protected = False
        lifecycle_has_open = False
        for row in execution_ledger:
            action_txt = str(row.get("intent_action") or row.get("action") or "").upper()
            cid_txt = str(row.get("client_order_id") or "").upper()
            status_txt = str(row.get("status") or row.get("state") or "").upper()
            if action_txt in {"OPEN", "FLIP"} or "|OPEN|" in cid_txt or "|FLIP|" in cid_txt:
                has_open = True
            if "ACK" in status_txt or status_txt in {"SENT", "ACCEPTED"}:
                has_ack = True
            if "FILL" in status_txt or status_txt == "ENTRY_FILLED":
                has_fill = True
            if row.get("stop_price") not in (None, ""):
                has_stop_submitted = True
            if row.get("target_price") not in (None, ""):
                has_target_submitted = True
        for row in lifecycle_rows:
            phase_txt = str(row.get("phase") or "").upper()
            action_txt = str(row.get("requested_action") or row.get("resolved_action") or "").upper()
            if phase_txt in {"LIVE", "REPLAY", "BACKFILL"} and action_txt in {"OPEN", "FLIP"}:
                lifecycle_has_open = True
        for ev in exec_events:
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            name = str(ev.get("event") or ev.get("type") or "").upper()
            payload_type = str(payload.get("event_type") or payload.get("type") or "").upper()
            if "ACK" in name or payload_type == "ORDER_ACK":
                has_ack = True
            if "FILL" in name or payload_type == "FILL":
                has_fill = True
            if name == "PROTECTION_CONFIRMED_BY_BROKER_SNAPSHOT":
                has_protection_confirmed = True
            if bool(((ev.get("trade_pnl_state") or {}) if isinstance(ev.get("trade_pnl_state"), dict) else {}).get("protected_confirmed")):
                has_protection_confirmed = True
            if payload_type == "POSITION_SNAPSHOT" or ("SNAPSHOT" in name and "POSITION" in name):
                has_snapshot_position = True
        status_pos = str(
            status.get("position_state")
            or health.get("position_state")
            or ((stream_state.get("position") or {}) if isinstance(stream_state.get("position"), dict) else {}).get("position_state")
            or ""
        ).upper()
        stream_pos_qty = ((stream_state.get("position") or {}) if isinstance(stream_state.get("position"), dict) else {}).get("pos")
        if status_pos in {"FLAT", "IN_POSITION_PROTECTED"}:
            has_exit_or_protected = True
        elif stream_pos_qty in (0, 0.0):
            has_exit_or_protected = True
        if not has_open and lifecycle_has_open:
            has_open = True
        if not has_open:
            _fail(failures, code="TRANSPORT_MISSING_OPEN", file=execution_ledger_path)
        if not has_ack:
            _fail(failures, code="TRANSPORT_MISSING_ACK", file=execution_ledger_path)
        if not has_fill:
            _fail(failures, code="TRANSPORT_MISSING_FILL", file=execution_ledger_path)
        if not has_stop_submitted:
            _fail(failures, code="TRANSPORT_MISSING_STOP", file=execution_ledger_path)
        if not runner_override and not has_target_submitted:
            _fail(failures, code="TRANSPORT_MISSING_TARGET", file=execution_ledger_path)
        if not has_snapshot_position:
            _fail(failures, code="TRANSPORT_MISSING_BROKER_SNAPSHOT", file=exec_events_path)
        if not has_protection_confirmed:
            _fail(failures, code="TRANSPORT_MISSING_PROTECTION_CONFIRMATION", file=exec_events_path)
        if not has_exit_or_protected:
            _fail(failures, code="TRANSPORT_MISSING_EXIT_OR_PROTECTED_FINAL_STATE", file=status_path)

    # Time checks in trades.
    for i, row in enumerate(trades, start=2):
        entry_ts = _parse_ts(row.get("entry_fill_ts") or row.get("entry_ts"))
        if entry_ts is not None and entry_ts.year <= 1971:
            _fail(
                failures,
                code="ENTRY_TS_1970_EPOCH_PARSE",
                file=trades_path,
                line=i,
                trade_or_order_id=str(row.get("client_order_id") or ""),
                context={"entry_ts": row.get("entry_fill_ts") or row.get("entry_ts")},
            )

    # Track bad-time and overlay.
    bad_time_seen = any(str(ev.get("event") or "") == "BAD_TIME_IN_TRADE" for ev in exec_events)
    if bad_time_seen:
        for ev in exec_events:
            name = str(ev.get("event") or "").lower()
            if "overlay" in name or "shelf" in name or "runner" in name:
                _fail(
                    failures,
                    code="OVERLAY_AFTER_BAD_TIME",
                    file=exec_events_path,
                    line=ev.get("_line"),
                    event=str(ev.get("event") or ""),
                    trade_or_order_id=str(ev.get("client_order_id") or ""),
                )
                break

    # PnL stair invariants.
    stair_actions = {
        "PNL_STAIR_ARMED",
        "PNL_STAIR_STEP_ADVANCED",
        "PNL_STAIR_STOP_UPDATE_REQUESTED",
        "PNL_STAIR_STOP_UPDATE_ACKED",
        "PNL_STAIR_STOP_UPDATE_REJECTED",
    }
    stair_noops = {
        "PNL_STAIR_NOOP_BAD_TIME",
        "PNL_STAIR_NOOP_NO_BROKER_STOP",
        "PNL_STAIR_NOOP_FLAT",
        "PNL_STAIR_NOOP_NO_IMPROVEMENT",
    }
    protection_seen_line = None
    fill_seen_line = None
    flat_seen_line = None
    for ev in exec_events:
        name = str(ev.get("event") or ev.get("type") or "").upper()
        line_no = int(ev.get("_line") or 0)
        if fill_seen_line is None and ("FILL" in name):
            fill_seen_line = line_no
        if protection_seen_line is None and name == "PROTECTION_CONFIRMED_BY_BROKER_SNAPSHOT":
            protection_seen_line = line_no
        if flat_seen_line is None:
            pos_state = str(ev.get("position_state") or ev.get("state") or "").upper()
            pos_qty = ev.get("position_qty")
            if name == "POSITION_SNAPSHOT":
                snap = ev.get("snapshot") if isinstance(ev.get("snapshot"), dict) else {}
                snap_state = str(snap.get("position_state") or snap.get("state") or "").upper()
                snap_qty = snap.get("position_qty")
                if snap_state == "FLAT" or snap_qty in (0, 0.0):
                    flat_seen_line = line_no
            elif pos_state == "FLAT" or pos_qty in (0, 0.0):
                flat_seen_line = line_no
    long_stops: list[tuple[int, float]] = []
    short_stops: list[tuple[int, float]] = []
    initial_stop_by_side: dict[str, float] = {}
    for row in execution_ledger:
        action_txt = str(row.get("intent_action") or row.get("action") or "").upper()
        if action_txt not in {"OPEN", "FLIP"}:
            continue
        side_txt = str(row.get("side") or "").upper()
        st = row.get("stop_price")
        try:
            stop_val = float(st)
        except Exception:
            continue
        if side_txt in {"LONG", "SHORT"} and side_txt not in initial_stop_by_side:
            initial_stop_by_side[side_txt] = stop_val
    rejected_count = 0
    degraded_seen = False
    desync_open = False
    for ev in exec_events:
        name = str(ev.get("event") or "").upper()
        line_no = int(ev.get("_line") or 0)
        if name == "BROKER_INTERNAL_POSITION_DESYNC":
            desync_open = True
        elif name == "BROKER_INTERNAL_POSITION_RESYNCED":
            desync_open = False
        if name == "PNL_STAIR_DEGRADED_NO_FORCE_EXIT":
            degraded_seen = True
        if name == "PNL_STAIR_STOP_UPDATE_REJECTED":
            rejected_count += 1
        if name in stair_actions:
            if fill_seen_line is None or line_no < fill_seen_line:
                _fail(failures, code="PNL_STAIR_ACTION_BEFORE_ENTRY_FILL", file=exec_events_path, line=line_no, event=name)
            if protection_seen_line is None or line_no < protection_seen_line:
                _fail(failures, code="PNL_STAIR_ACTION_BEFORE_PROTECTION_CONFIRM", file=exec_events_path, line=line_no, event=name)
            if bad_time_seen and name not in stair_noops:
                _fail(failures, code="PNL_STAIR_ACTION_AFTER_BAD_TIME", file=exec_events_path, line=line_no, event=name)
            if flat_seen_line is not None and line_no > flat_seen_line:
                _fail(failures, code="PNL_STAIR_ACTION_AFTER_FLAT", file=exec_events_path, line=line_no, event=name)
            if desync_open:
                _fail(failures, code="PNL_STAIR_ACTION_DURING_DESYNC", file=exec_events_path, line=line_no, event=name)
            stop_val = ev.get("stop_price")
            try:
                stop_f = float(stop_val)
            except Exception:
                stop_f = None
            side = str(ev.get("side") or "").upper()
            if stop_f is not None and side == "LONG":
                long_stops.append((line_no, stop_f))
            if stop_f is not None and side == "SHORT":
                short_stops.append((line_no, stop_f))
        if "PNL_STAIR" in name and "FORCE_EXIT" in name:
            _fail(failures, code="PNL_STAIR_FORCE_EXIT_FORBIDDEN", file=exec_events_path, line=line_no, event=name)
    if rejected_count >= 3 and not degraded_seen:
        _fail(failures, code="PNL_STAIR_REJECTED_WITHOUT_DEGRADE", file=exec_events_path, context={"rejected_count": rejected_count})
    prev = None
    for line_no, stop_val in long_stops:
        if prev is not None and stop_val + 1e-9 < prev:
            _fail(failures, code="PNL_STAIR_LONG_STOP_DECREASED", file=exec_events_path, line=line_no, context={"prev": prev, "cur": stop_val})
        prev = stop_val
    prev = None
    for line_no, stop_val in short_stops:
        if prev is not None and stop_val - 1e-9 > prev:
            _fail(failures, code="PNL_STAIR_SHORT_STOP_INCREASED", file=exec_events_path, line=line_no, context={"prev": prev, "cur": stop_val})
        prev = stop_val
    if "LONG" in initial_stop_by_side and long_stops:
        if long_stops[0][1] + 1e-9 < initial_stop_by_side["LONG"]:
            _fail(failures, code="PNL_STAIR_WORSENED_RISK_LONG", file=exec_events_path, line=long_stops[0][0], context={"initial_stop": initial_stop_by_side["LONG"], "first_stair_stop": long_stops[0][1]})
    if "SHORT" in initial_stop_by_side and short_stops:
        if short_stops[0][1] - 1e-9 > initial_stop_by_side["SHORT"]:
            _fail(failures, code="PNL_STAIR_WORSENED_RISK_SHORT", file=exec_events_path, line=short_stops[0][0], context={"initial_stop": initial_stop_by_side["SHORT"], "first_stair_stop": short_stops[0][1]})

    # Desync checks.
    desync_seen = False
    resync_seen = False
    disarm_seen = False
    for ev in exec_events:
        name = str(ev.get("event") or "")
        if name == "BROKER_INTERNAL_POSITION_DESYNC":
            desync_seen = True
        if name == "BROKER_INTERNAL_POSITION_RESYNCED":
            resync_seen = True
        if name in {"entries_disarmed", "DISARMED_ORDER_LINEAGE_AMBIGUOUS"}:
            disarm_seen = True
    if desync_seen and not (resync_seen or disarm_seen):
        _fail(failures, code="DESYNC_WITHOUT_RESYNC_OR_DISARM", file=exec_events_path)

    # Stream-state vs status coarse consistency.
    pos_state = stream_state.get("position") if isinstance(stream_state.get("position"), dict) else {}
    ss_pos = pos_state.get("pos")
    status_pos_state = str(status.get("position_state") or "").upper()
    if ss_pos in (0, 0.0) and status_pos_state.startswith("IN_POSITION"):
        _fail(
            failures,
            code="STREAM_STATE_STATUS_POSITION_MISMATCH",
            file=state_path if state_path.exists() else status_path,
            context={"stream_state_pos": ss_pos, "status_position_state": status_pos_state},
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase-1 stabilization invariants for a run directory.")
    parser.add_argument("run_dir", type=Path, help="Run directory path (e.g., .\\runs\\modelrunlivetests102)")
    parser.add_argument(
        "--require-order-lifecycle",
        action="store_true",
        help="Require transport lifecycle evidence (OPEN/ACK/FILL/protection/exit) in addition to safety checks.",
    )
    args = parser.parse_args()

    failures = validate_run(args.run_dir, require_order_lifecycle=bool(args.require_order_lifecycle))
    if failures:
        print("PHASE1_FAIL")
        for f in failures:
            print(json.dumps(f, ensure_ascii=True))
        return 1
    if bool(args.require_order_lifecycle):
        print("PHASE1_TRANSPORT_PASS")
    else:
        print("PHASE1_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
