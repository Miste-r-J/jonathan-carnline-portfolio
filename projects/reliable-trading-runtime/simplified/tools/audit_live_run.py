from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def _load_json(path: Path) -> Tuple[Dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"missing:{path.name}"
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace")), None
    except Exception as exc:
        return {}, f"invalid:{path.name}:{exc}"


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def _read_lifecycle_rows(run_dir: Path) -> List[Mapping[str, Any]]:
    path = run_dir / "lifecycle_events.jsonl"
    return list(_iter_jsonl(path))


def _age_sec(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - float(path.stat().st_mtime))
    except Exception:
        return None


def _ts_epoch(raw: Any) -> float | None:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _infer_cooldown_is_stale(status: Mapping[str, Any], state: Mapping[str, Any], health: Mapping[str, Any]) -> bool:
    if "cooldown_is_stale" in health:
        return bool(health.get("cooldown_is_stale"))
    position = state.get("position") if isinstance(state.get("position"), dict) else {}
    left = int(position.get("cooldown_left") or status.get("cooldown_left") or 0)
    if left <= 0:
        return False
    health_unresolved = set(str(x) for x in (health.get("unresolved_warnings") or []) if x) if health else set()
    fallback_stale = (not health) or ("flat_cooldown_active" in health_unresolved)
    cooldown_bars = int(status.get("cooldown_bars") or position.get("cooldown_bars") or 0)
    bar_interval = float(status.get("bar_interval_sec") or 0.0)
    started = _ts_epoch(position.get("cooldown_started_bar_ts") or status.get("cooldown_started_bar_ts"))
    current = _ts_epoch(status.get("last_bar_ts") or position.get("last_ts"))
    if cooldown_bars > 0 and bar_interval > 0 and started is not None and current is not None:
        return (current - started) >= float(cooldown_bars + 1) * bar_interval
    return bool(fallback_stale)


def _process_alive_for_run(run_dir: Path, run_id: str | None) -> bool | None:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    needles = {str(run_dir).lower()}
    if run_id:
        needles.add(str(run_id).lower())
    try:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(str(x) for x in (proc.info.get("cmdline") or []))
            except Exception:
                continue
            cmdline_l = cmdline.lower()
            if "audit_live_run.py" in cmdline_l:
                continue
            looks_like_streamer = "stream_live_csv" in cmdline_l or "run_stream_live_csv" in cmdline_l
            if not looks_like_streamer:
                continue
            if any(needle and needle in cmdline_l for needle in needles):
                return True
    except Exception:
        return None
    return False


def audit_run(run_dir: Path) -> Dict[str, Any]:
    status, status_err = _load_json(run_dir / "status.json")
    state, state_err = _load_json(run_dir / "stream_state.json")
    resolved, resolved_err = _load_json(run_dir / "resolved_config.json")
    health, _health_err = _load_json(run_dir / "run_health_summary.json")
    issues: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    position = state.get("position") if isinstance(state.get("position"), dict) else {}
    cooldown_left = int(position.get("cooldown_left") or status.get("cooldown_left") or 0)
    run_id = str(status.get("run_id") or resolved.get("run_id") or run_dir.name)
    process_alive = _process_alive_for_run(run_dir, run_id)
    pos = float(position.get("pos") or 0.0)
    state_payload = state.get("state") if isinstance(state.get("state"), dict) else {}
    position_state = str(state_payload.get("position_state") or status.get("position_state") or "")
    working_orders = int(status.get("snapshot_blocking_orders_count") or status.get("snapshot_orders_count") or 0)
    nt_protected = state_payload.get("nt_protected")
    nt_exits_working = state_payload.get("nt_exits_working")

    if status_err:
        issues.append({"code": "missing_status", "detail": status_err})
    if state_err:
        issues.append({"code": "missing_state", "detail": state_err})
    if resolved_err:
        issues.append({"code": "missing_resolved_config", "detail": resolved_err})
    health_clean_stopped = bool(health.get("clean_stopped")) if health else False
    health_verdict = str(health.get("verdict") or "") if health else ""
    status_age = _age_sec(run_dir / "status.json")
    health_process_alive = bool(health.get("process_alive")) if health else False
    status_recent = status_age is not None and status_age <= 120.0
    if (
        process_alive is False
        and not health_clean_stopped
        and health_verdict != "running_healthy"
        and not (health_process_alive and status_recent)
    ):
        issues.append({"code": "no_process_alive"})

    cooldown_is_stale = _infer_cooldown_is_stale(status, state, health)
    if (
        cooldown_left > 0
        and abs(pos) <= 1e-6
        and position_state.upper() in {"", "FLAT", "UNKNOWN"}
        and working_orders == 0
        and cooldown_is_stale
    ):
        issues.append({"code": "stuck_flat_cooldown", "cooldown_left": cooldown_left})

    feed_health_ok = status.get("feed_health_ok")
    bar_age_sec = status.get("bar_age_sec")
    max_bar_age = status.get("effective_bar_age_max_sec")
    if feed_health_ok is False:
        issues.append({"code": "feed_health_false"})
    try:
        if bar_age_sec is not None and max_bar_age is not None and float(max_bar_age) > 0 and float(bar_age_sec) > float(max_bar_age):
            issues.append({"code": "bar_age_exceeds_max", "bar_age_sec": float(bar_age_sec), "max": float(max_bar_age)})
    except Exception:
        pass

    if status_age is not None and status_age > 120.0 and not health_clean_stopped:
        issues.append({"code": "status_stale", "status_age_sec": round(status_age, 1)})

    order_events = list(_iter_jsonl(run_dir / "order_events.jsonl"))
    exec_events = list(_iter_jsonl(run_dir / "exec_events.jsonl"))
    gating_events = list(_iter_jsonl(run_dir / "gating_events.jsonl"))
    signal_to_order_rows = list(_iter_jsonl(run_dir / "signal_to_order.jsonl"))
    lifecycle_rows = _read_lifecycle_rows(run_dir)
    nt_bridge_rows = list(_iter_jsonl(run_dir / "nt_bridge.jsonl"))
    unresolved_untracked = []
    for e in order_events + exec_events:
        event = str(e.get("event") or "")
        if event in {"orphan_fill", "reassociated_untracked_fill_classified_exit"}:
            unresolved_untracked.append(e)
            continue
        if event == "fallback_fill_reassociated":
            continue
        text = json.dumps(e, ensure_ascii=True)
        if "UNTRACKED fill could not be reconciled" in text:
            unresolved_untracked.append(e)
    if unresolved_untracked:
        issues.append({"code": "unresolved_untracked_fills", "count": len(unresolved_untracked)})

    fallback_close_rows = 0
    false_missing_stop_rows = 0
    trades_path = run_dir / "trades.csv"
    if trades_path.exists():
        try:
            with trades_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                for row in csv.DictReader(fh):
                    exit_reason = str(row.get("exit_reason") or "").strip()
                    if exit_reason == "reconciled_fill_exit":
                        fallback_close_rows += 1
                    if exit_reason == "missing_stop_state":
                        for key in ("planned_stop", "entry_stop", "live_stop", "stop"):
                            try:
                                raw = row.get(key)
                                if raw is not None and str(raw).strip() not in {"", "0", "0.0", "None", "null"}:
                                    if float(raw) > 0:
                                        false_missing_stop_rows += 1
                                        break
                            except Exception:
                                continue
        except Exception:
            fallback_close_rows = 0
            false_missing_stop_rows = 0
    if fallback_close_rows:
        warnings.append({"code": "close_exit_reason_fallback", "count": fallback_close_rows})
    if false_missing_stop_rows:
        warnings.append({"code": "false_missing_stop_state", "count": false_missing_stop_rows})

    addon_flags = status.get("addon_flags") if isinstance(status.get("addon_flags"), dict) else {}
    disabled_safety = [
        key for key in ("protection_repair_enabled", "stop_update_enabled", "auto_flatten_enabled")
        if addon_flags.get(key) is False
    ]
    if disabled_safety:
        issues.append({"code": "disabled_nt_safety_flags", "flags": disabled_safety})

    risk_bypass_count = 0
    for path_name in ("signals.jsonl", "signal_to_order.jsonl", "events.jsonl"):
        for row in _iter_jsonl(run_dir / path_name):
            text = json.dumps(row, ensure_ascii=True)
            if "paper_min_contracts" in text:
                risk_bypass_count += 1
    if risk_bypass_count:
        issues.append({"code": "paper_risk_cap_bypass", "count": risk_bypass_count})

    threshold_p_longs: List[float] = []
    pred_p_longs: List[float] = []
    for row in gating_events:
        try:
            if row.get("phase") and str(row.get("phase")).upper() != "LIVE":
                continue
            thr = row.get("threshold_p_long", row.get("p_buy_effective", row.get("p_buy")))
            pred = row.get("pred_p_long")
            if pred is None:
                phase2 = row.get("phase2") if isinstance(row.get("phase2"), Mapping) else {}
                pred = phase2.get("direction_prob")
            if thr is not None:
                threshold_p_longs.append(float(thr))
            if pred is not None:
                pred_p_longs.append(float(pred))
        except Exception:
            continue
    if threshold_p_longs and pred_p_longs:
        thr_min = min(threshold_p_longs)
        thr_max = max(threshold_p_longs)
        pred_min = min(pred_p_longs)
        pred_max = max(pred_p_longs)
        if abs(thr_max - thr_min) <= 1e-9 and abs(pred_max - pred_min) > 1e-6:
            warnings.append(
                {
                    "code": "threshold_fields_not_model_predictions",
                    "detail": "threshold_p_long is constant while pred_p_long varies; avoid interpreting threshold telemetry as model output.",
                    "threshold_p_long": round(thr_min, 6),
                    "pred_p_long_min": round(pred_min, 6),
                    "pred_p_long_max": round(pred_max, 6),
                    "sample_count": len(pred_p_longs),
                }
            )

    parity_taxonomy_counts: Dict[str, int] = {}
    live_candidate_total = 0
    live_pre_signal_block_total = 0
    live_emit_allowed_total = 0
    live_send_path_total = 0
    live_nt_order_entry_total = 0
    strict_lockout_present = False
    timeout_oscillation_count = 0
    terminal_duplicate_emit = 0
    safety_loop_count = 0
    timeout_seen = False
    for row in exec_events:
        event = str(row.get("event") or "")
        if event == "protection_timeout_terminal_suppressed_repeat":
            terminal_duplicate_emit += 1
        if event == "state_resurrection_suppressed":
            timeout_oscillation_count += 1
        if event == "late_fill_after_lockout":
            safety_loop_count += 1
        if event in {"flatten_due_to_no_protection", "protection_timeout_retry", "protection_lost_anchor_set"}:
            timeout_seen = True
    for row in signal_to_order_rows:
        phase = str(row.get("phase") or "").upper()
        action = str(row.get("signal_action") or "").upper()
        if phase != "LIVE":
            continue
        if action in {"OPEN", "CLOSE", "FLIP", "EXIT"}:
            live_candidate_total += 1
        decision = str(row.get("decision") or "").upper()
        emit_allowed = bool(row.get("emit_allowed"))
        sent_to_nt = bool(row.get("sent_to_nt"))
        blocked_by = [str(x or "").lower() for x in list(row.get("blocked_by") or [])]
        reason = str(row.get("reason") or "").lower()
        if decision in {"BLOCKED", "BLOCKED_SAFETY", "BLOCKED_SYNC", "BLOCKED_EXEC_POLICY"} and (
            "setup" in blocked_by or "gates_block" in reason
        ):
            live_pre_signal_block_total += 1
            parity_taxonomy_counts["SETUP_FAIL_PRE_SIGNAL"] = int(parity_taxonomy_counts.get("SETUP_FAIL_PRE_SIGNAL", 0) or 0) + 1
        if "stop_already_breached" in reason:
            parity_taxonomy_counts["STOP_BREACH_PRE_SIGNAL"] = int(parity_taxonomy_counts.get("STOP_BREACH_PRE_SIGNAL", 0) or 0) + 1
        if emit_allowed:
            live_emit_allowed_total += 1
        if sent_to_nt:
            live_send_path_total += 1
        if emit_allowed and not sent_to_nt and action in {"OPEN", "CLOSE", "FLIP", "EXIT"}:
            parity_taxonomy_counts["EMIT_ALLOWED_NOT_SENT"] = int(parity_taxonomy_counts.get("EMIT_ALLOWED_NOT_SENT", 0) or 0) + 1
    try:
        live_nt_order_entry_total = int((status.get("executor_stats_all_phases") or {}).get("nt_order_entry_total") or 0)
    except Exception:
        live_nt_order_entry_total = 0
    if live_send_path_total > live_nt_order_entry_total:
        parity_taxonomy_counts["SENT_WITHOUT_ORDER_ENTRY"] = int(
            parity_taxonomy_counts.get("SENT_WITHOUT_ORDER_ENTRY", 0) or 0
        ) + (live_send_path_total - live_nt_order_entry_total)
    if bool(status.get("hard_lockout_active")) and str(status.get("hard_lockout_code") or "") == "strict_intent_parity_mismatch":
        strict_lockout_present = True
    # State/execution divergence: state has open-like lifecycle but live has no send path.
    open_like_state_rows = 0
    if lifecycle_rows:
        for row in lifecycle_rows:
            phase = str(row.get("phase") or "").upper()
            requested = str(row.get("requested_action") or "").upper()
            if phase == "LIVE" and requested in {"OPEN", "FLIP"}:
                open_like_state_rows += 1
    else:
        state_csv_path = run_dir / "state.csv"
        if state_csv_path.exists():
            try:
                with state_csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                    for row in csv.DictReader(fh):
                        action = str(row.get("requested_action") or row.get("action") or row.get("type") or "").upper()
                        if action in {"OPEN", "FLIP"}:
                            open_like_state_rows += 1
            except Exception:
                open_like_state_rows = 0
    if open_like_state_rows > 0 and live_send_path_total == 0 and live_candidate_total > 0:
        parity_taxonomy_counts["STATE_EXECUTION_DIVERGENCE"] = int(
            parity_taxonomy_counts.get("STATE_EXECUTION_DIVERGENCE", 0) or 0
        ) + 1

    parity_fail_reasons: List[str] = []
    for code in ("EMIT_ALLOWED_NOT_SENT", "SENT_WITHOUT_ORDER_ENTRY", "STATE_EXECUTION_DIVERGENCE"):
        if int(parity_taxonomy_counts.get(code, 0) or 0) > 0:
            parity_fail_reasons.append(code)
    if strict_lockout_present:
        parity_fail_reasons.append("STRICT_PARITY_LOCKOUT_PRESENT")
    if terminal_duplicate_emit > 0:
        parity_fail_reasons.append("TERMINAL_DUPLICATE_EMIT")
    if timeout_seen and timeout_oscillation_count > 0:
        parity_fail_reasons.append("STATE_OSCILLATION_AFTER_TIMEOUT")
    if safety_loop_count > 2:
        parity_fail_reasons.append("SAFETY_LOOP_DETECTED")
    flip_pending_to_generic_stale = 0
    missing_stop_lockout_spam = 0
    guardrail_nonflat_terminal = False
    direct_flip_sent_total = 0
    flip_transition_order_violations = 0
    flip_pending_by_cid: Dict[str, bool] = {}
    missing_stop_lockout_by_cid: Dict[str, int] = {}
    flip_transition_steps: Dict[str, Dict[str, int]] = {}
    for row in signal_to_order_rows:
        cid = str(row.get("client_order_id") or "")
        action = str(row.get("signal_action") or row.get("action") or "").upper()
        reason = str(row.get("reason") or "").lower()
        phase = str(row.get("phase") or "").upper()
        parity_state = str(row.get("parity_transition_state") or "").lower()
        sent_to_nt = bool(row.get("sent_to_nt"))
        if phase == "LIVE" and sent_to_nt and action == "FLIP":
            direct_flip_sent_total += 1
        requested_action = str(row.get("requested_action") or "").upper()
        transition_id = str(row.get("transition_id") or "")
        if transition_id and requested_action == "FLIP":
            bucket = flip_transition_steps.setdefault(transition_id, {})
            bucket[action] = int(bucket.get(action, 0) or 0) + 1
        if (
            phase == "LIVE"
            and reason == "flip_open_deferred_wait_reconcile"
            and parity_state == "transition_pending"
            and cid
        ):
            flip_pending_by_cid[cid] = True
        if cid in flip_pending_by_cid and "signal_stale" in reason and "flip_open" not in reason:
            flip_pending_to_generic_stale += 1
        if "missing_stop_price" in reason:
            missing_stop_lockout_by_cid[cid] = int(missing_stop_lockout_by_cid.get(cid, 0) or 0) + 1
    for cid, count in missing_stop_lockout_by_cid.items():
        if cid and count > 1:
            missing_stop_lockout_spam += 1
    for steps in flip_transition_steps.values():
        if int(steps.get("OPEN", 0) or 0) > 0 and int(steps.get("CLOSE", 0) or 0) <= 0:
            flip_transition_order_violations += 1
    lockout_state = str(state_payload.get("entries_disarmed_reason") or status.get("entries_disarmed_reason") or "").lower()
    if lockout_state == "guardrail_lockout" and (abs(pos) > 1e-6 or working_orders > 0 or position_state.upper() != "FLAT"):
        guardrail_nonflat_terminal = True
    if flip_pending_to_generic_stale > 0:
        parity_fail_reasons.append("FLIP_PENDING_STALE_COLLAPSE")
    if missing_stop_lockout_spam > 0:
        parity_fail_reasons.append("MISSING_STOP_LOCKOUT_SPAM")
    if guardrail_nonflat_terminal:
        parity_fail_reasons.append("GUARDRAIL_LOCKOUT_NONFLAT_TERMINAL")
    if direct_flip_sent_total > 0:
        parity_fail_reasons.append("DIRECT_FLIP_SENT")
    if flip_transition_order_violations > 0:
        parity_fail_reasons.append("FLIP_TRANSITION_ORDER_VIOLATION")
    parity_decision = "PASS" if not parity_fail_reasons else "FAIL"
    parity_summary = {
        "live_candidate_total": live_candidate_total,
        "live_setup_pass_total": max(0, live_candidate_total - live_pre_signal_block_total),
        "live_pre_signal_block_total": live_pre_signal_block_total,
        "live_emit_allowed_total": live_emit_allowed_total,
        "live_send_path_total": live_send_path_total,
        "live_nt_order_entry_total": live_nt_order_entry_total,
        "taxonomy_counts": parity_taxonomy_counts,
        "safety_loop_count": safety_loop_count,
        "terminal_duplicate_emit": terminal_duplicate_emit,
        "timeout_oscillation_count": timeout_oscillation_count,
        "flip_pending_to_generic_stale": flip_pending_to_generic_stale,
        "missing_stop_lockout_spam": missing_stop_lockout_spam,
        "guardrail_nonflat_terminal": guardrail_nonflat_terminal,
        "direct_flip_sent_total": direct_flip_sent_total,
        "flip_transition_order_violations": flip_transition_order_violations,
    }

    if abs(pos) > 1e-6 or working_orders > 0:
        issues.append({"code": "final_position_or_orders_not_flat", "pos": pos, "working_orders": working_orders})
    if (
        position_state.upper() == "IN_POSITION_UNPROTECTED"
        and (nt_protected is True or nt_exits_working is True)
    ):
        issues.append(
            {
                "code": "contradictory_protection_state",
                "position_state": position_state,
                "nt_protected": nt_protected,
                "nt_exits_working": nt_exits_working,
            }
        )
    if abs(pos) > 1e-6 and working_orders == 0 and position_state.upper() in {"IN_POSITION_UNPROTECTED", "EXITING"}:
        issues.append(
            {
                "code": "nonflat_without_working_orders",
                "pos": pos,
                "position_state": position_state,
            }
        )

    verdict = "PASS" if not issues else "FAIL"
    if health and health.get("verdict") in {"clean_stopped", "running_healthy"} and not issues:
        verdict = "PASS"
    return {
        "run_dir": str(run_dir),
        "process_alive": process_alive,
        "verdict": verdict,
        "issues": issues,
        "warnings": warnings,
        "health_verdict": health.get("verdict") if health else None,
        "health_clean_stopped": health_clean_stopped if health else None,
        "status_present": bool(status),
        "state_present": bool(state),
        "resolved_config_present": bool(resolved),
        "final_position": pos,
        "position_state": position_state or None,
        "nt_protected": nt_protected,
        "nt_exits_working": nt_exits_working,
        "working_order_count": working_orders,
        "cooldown_left": cooldown_left,
        "cooldown_state": health.get("cooldown_state") if health else None,
        "cooldown_is_stale": cooldown_is_stale,
        "feed_health_ok": feed_health_ok,
        "bar_age_sec": bar_age_sec,
        "effective_bar_age_max_sec": max_bar_age,
        "executor_stats": status.get("executor_stats"),
        "parity_summary": parity_summary,
        "parity_decision": parity_decision,
        "parity_fail_reasons": parity_fail_reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a live/paper run folder for unattended-run reliability issues.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = audit_run(args.run_dir)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

