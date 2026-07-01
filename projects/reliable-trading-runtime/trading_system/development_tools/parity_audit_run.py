from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PHASE_NAMES = ("BACKFILL", "CATCHUP", "LIVE")
OPEN_ACTIONS = {"OPEN", "FLIP"}
FILLED_STATUSES = {
    "fill",
    "filled",
    "entry_filled",
    "exit_filled",
    "order_filled",
}
SENT_STATUSES = {
    "sent",
    "ack",
    "acked",
    "acknowledged",
    "entry_acked",
    "exits_submitted",
    "submitted",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing {path.name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"failed to parse {path.name}: {exc}"


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], f"missing {path.name}"
    rows: list[dict[str, Any]] = []
    try:
        for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception as exc:
                return rows, f"failed to parse {path.name}:{line_no}: {exc}"
            if isinstance(row, dict):
                row["_line"] = line_no
                rows.append(row)
        return rows, None
    except Exception as exc:
        return rows, f"failed to read {path.name}: {exc}"


def _load_csv(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], f"missing {path.name}"
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader, start=2):
                if row is None:
                    continue
                clean = {k: v for k, v in row.items()}
                clean["_line"] = idx
                rows.append(clean)
        return rows, None
    except Exception as exc:
        return rows, f"failed to parse {path.name}: {exc}"


def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "nat", "undefined"}:
        return ""
    return text


def _upper(value: Any) -> str:
    return _norm(value).upper()


def _truthy(value: Any) -> bool:
    text = _norm(value).lower()
    return text in {"1", "true", "yes", "y", "t", "on"}


def _has_value(value: Any) -> bool:
    return _norm(value) != ""


def _cid(row: dict[str, Any]) -> str:
    for key in ("client_order_id", "cid", "intent_id", "order_id", "signal_id"):
        value = _norm(row.get(key))
        if value:
            return value
    return ""


def _first_nonempty(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = _norm(row.get(key))
        if value:
            return value
    return ""


def _pick_time(row: dict[str, Any]) -> str:
    return _first_nonempty(row, ("ts", "timestamp", "datetime", "bar_ts", "entry_ts", "exit_ts", "actual_exit_ts"))


def _offender(
    *,
    source: str,
    line: Any,
    ts: str,
    cid: str,
    kind: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "source": source,
        "line": line,
        "ts": ts,
        "cid": cid,
        "kind": kind,
        "details": details,
    }
    return payload


def _top_samples(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return items[: max(0, limit)]


def _dedupe_offenders(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (
            item.get("kind"),
            item.get("cid"),
            item.get("ts"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _build_signal_index(signals: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in signals:
        dt = _norm(row.get("datetime"))
        action = _upper(row.get("type"))
        if dt and action:
            index[(dt, action)].append(row)
    return index


def _match_signal(
    signals_index: dict[tuple[str, str], list[dict[str, Any]]],
    row: dict[str, Any],
    action_keys: Iterable[str] = ("action", "type"),
) -> dict[str, Any] | None:
    timestamp_keys = ("bar_ts", "datetime", "ts", "timestamp", "entry_ts")
    ts = _first_nonempty(row, timestamp_keys)
    if not ts:
        return None
    for action_key in action_keys:
        action = _upper(row.get(action_key))
        if not action:
            continue
        matches = signals_index.get((ts, action))
        if matches:
            return matches[0]
    return None


def _phase_summary(gating_events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    first_ts: dict[str, str] = {}
    last_ts: dict[str, str] = {}
    for row in gating_events:
        phase = _upper(row.get("phase")) or "UNKNOWN"
        counts[phase] += 1
        ts = _pick_time(row)
        if ts and phase not in first_ts:
            first_ts[phase] = ts
        if ts:
            last_ts[phase] = ts
    total = sum(counts.values())
    ordered = {name: counts.get(name, 0) for name in PHASE_NAMES}
    ordered["UNKNOWN"] = counts.get("UNKNOWN", 0)
    shares = {k: (v / total if total else 0.0) for k, v in ordered.items()}
    return {
        "total": total,
        "counts": ordered,
        "shares": shares,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def _check_emitted_open_setup_blocked(
    gating_events: list[dict[str, Any]],
    signals: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    offenders: list[dict[str, Any]] = []
    signals_index = _build_signal_index(signals)

    for row in gating_events:
        action = _upper(row.get("action"))
        if action not in OPEN_ACTIONS:
            continue
        gate_state = row.get("gate_state") if isinstance(row.get("gate_state"), dict) else {}
        blocked_by = row.get("blocked_by") if isinstance(row.get("blocked_by"), list) else []
        setup_pass = gate_state.get("setup")
        strategy_reason = _norm(row.get("strategy_blocked_reason"))
        blocked = (
            setup_pass is False
            or "setup" in {str(x).lower() for x in blocked_by}
            or strategy_reason.lower() == "setup"
        )
        if blocked:
            signal_row = _match_signal(signals_index, row, action_keys=("action",))
            cid = _cid(row) or (_cid(signal_row) if signal_row else "")
            offenders.append(
                _offender(
                    source="gating_events.jsonl",
                    line=row.get("_line"),
                    ts=_pick_time(row),
                    cid=cid,
                    kind="open_while_setup_blocked",
                    details={
                        "action": action,
                        "phase": _upper(row.get("phase")) or "UNKNOWN",
                        "setup_pass": setup_pass,
                        "blocked_by": blocked_by,
                        "override_applied": row.get("override_applied"),
                        "override_confident_long": row.get("override_confident_long"),
                        "strategy_blocked_reason": row.get("strategy_blocked_reason"),
                        "execution_blocked_reason": row.get("execution_blocked_reason"),
                        "matched_signal_side": _norm(signal_row.get("side")) if signal_row else "",
                    },
                )
            )

    for row in signals:
        action = _upper(row.get("type"))
        if action not in OPEN_ACTIONS:
            continue
        blocked = _truthy(row.get("blocked")) or _norm(row.get("blocked_reason")).lower() == "setup"
        if blocked:
            offenders.append(
                _offender(
                    source="signals.csv",
                    line=row.get("_line"),
                    ts=_pick_time(row),
                    cid=_cid(row),
                    kind="open_while_setup_blocked",
                    details={
                        "action": action,
                        "side": _norm(row.get("side")),
                        "blocked": row.get("blocked"),
                        "blocked_reason": row.get("blocked_reason"),
                        "override_applied": row.get("override_applied"),
                    },
                )
            )

    offenders = _dedupe_offenders(offenders)
    return {
        "name": "emitted_open_while_setup_blocked",
        "ok": len(offenders) == 0,
        "count": len(offenders),
    }, offenders


def _check_side_presence(
    gating_events: list[dict[str, Any]],
    signals: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    offenders: list[dict[str, Any]] = []

    def _side_ok(side: str) -> bool:
        return side.upper() in {"LONG", "SHORT"}

    for row in gating_events:
        if "side" not in row:
            continue
        action = _upper(row.get("action"))
        if action not in OPEN_ACTIONS:
            continue
        side = _norm(row.get("side"))
        if not _side_ok(side):
            offenders.append(
                _offender(
                    source="gating_events.jsonl",
                    line=row.get("_line"),
                    ts=_pick_time(row),
                    cid=_cid(row),
                    kind="missing_side_on_open_flip",
                    details={
                        "action": action,
                        "side": side,
                        "phase": _upper(row.get("phase")) or "UNKNOWN",
                    },
                )
            )

    for row in signals:
        action = _upper(row.get("type"))
        if action not in OPEN_ACTIONS:
            continue
        side = _norm(row.get("side"))
        if not _side_ok(side):
            offenders.append(
                _offender(
                    source="signals.csv",
                    line=row.get("_line"),
                    ts=_pick_time(row),
                    cid=_cid(row),
                    kind="missing_side_on_open_flip",
                    details={
                        "action": action,
                        "side": side,
                        "blocked": row.get("blocked"),
                        "blocked_reason": row.get("blocked_reason"),
                    },
                )
            )

    offenders = _dedupe_offenders(offenders)
    return {
        "name": "gating_side_missing_on_open_flip",
        "ok": len(offenders) == 0,
        "count": len(offenders),
    }, offenders


def _check_blocked_vs_sent(
    gating_events: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    order_events: list[dict[str, Any]],
    execution_ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    blocked_by_cid: dict[str, dict[str, Any]] = {}
    sent_by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _capture_block(row: dict[str, Any], source: str) -> None:
        cid = _cid(row)
        if not cid:
            return
        blocked = False
        if source == "signals.csv":
            blocked = _truthy(row.get("blocked")) or _has_value(row.get("blocked_reason"))
        else:
            blocked_by = row.get("blocked_by") if isinstance(row.get("blocked_by"), list) else []
            blocked = bool(blocked_by) or _has_value(row.get("strategy_blocked_reason")) or _has_value(row.get("execution_blocked_reason"))
        if blocked:
            blocked_by_cid.setdefault(
                cid,
                {
                    "cid": cid,
                    "source": source,
                    "ts": _pick_time(row),
                    "line": row.get("_line"),
                    "details": {},
                },
            )

    for row in signals:
        action = _upper(row.get("type"))
        if action in OPEN_ACTIONS:
            _capture_block(row, "signals.csv")

    for row in gating_events:
        action = _upper(row.get("action"))
        if action in OPEN_ACTIONS:
            _capture_block(row, "gating_events.jsonl")

    for row in order_events:
        cid = _cid(row)
        if not cid:
            continue
        status = _norm(row.get("status")).lower()
        if status in SENT_STATUSES:
            sent_by_cid[cid].append(
                {
                    "source": "order_events.jsonl",
                    "ts": _pick_time(row),
                    "line": row.get("_line"),
                    "status": row.get("status"),
                    "details": {
                        "status": row.get("status"),
                        "error": row.get("error"),
                    },
                }
            )

    for row in execution_ledger:
        cid = _cid(row)
        if not cid:
            continue
        status = _norm(row.get("status")).lower()
        if status in SENT_STATUSES:
            sent_by_cid[cid].append(
                {
                    "source": "execution_ledger.jsonl",
                    "ts": _pick_time(row),
                    "line": row.get("_line"),
                    "status": row.get("status"),
                    "details": {
                        "status": row.get("status"),
                        "error": row.get("error"),
                    },
                }
            )

    offenders: list[dict[str, Any]] = []
    for cid, blocked in blocked_by_cid.items():
        if cid in sent_by_cid:
            offenders.append(
                {
                    "cid": cid,
                    "blocked": blocked,
                    "sent": sent_by_cid[cid][:3],
                    "kind": "blocked_but_sent",
                }
            )

    offenders = _dedupe_offenders(offenders)
    return {
        "name": "blocked_vs_sent_contradictions",
        "ok": len(offenders) == 0,
        "count": len(offenders),
    }, offenders


def _check_stale_startup_resync(status: dict[str, Any] | None, gating_events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    offenders: list[dict[str, Any]] = []
    if status:
        phase = _upper(status.get("phase"))
        armed = bool(status.get("armed"))
        fields = {
            "primary_reason_code": _norm(status.get("primary_reason_code")),
            "current_block_code": _norm(status.get("current_block_code")),
            "last_block_code": _norm(status.get("last_block_code")),
            "entries_disarmed_reason": _norm(status.get("entries_disarmed_reason")),
            "hard_lockout_code": _norm(status.get("hard_lockout_code")),
        }
        # first_trade_block_code is historical breadcrumb and should not be treated as active block state
        has_startup_resync = any("startup_resync" in v.lower() for v in fields.values())
        if has_startup_resync and (armed or phase == "LIVE"):
            offenders.append(
                {
                    "source": "status.json",
                    "line": None,
                    "ts": _norm(status.get("current_block_ts")) or _norm(status.get("snapshot_last_ts")) or _utc_now_iso(),
                    "cid": _norm(status.get("run_id")),
                    "kind": "stale_startup_resync_while_armed_live",
                    "details": {
                        "phase": phase,
                        "armed": armed,
                        "primary_reason_code": status.get("primary_reason_code"),
                        "current_block_code": status.get("current_block_code"),
                        "first_trade_block_code": status.get("first_trade_block_code"),
                        "last_block_code": status.get("last_block_code"),
                        "current_block_ts": status.get("current_block_ts"),
                        "snapshot_last_ts": status.get("snapshot_last_ts"),
                    },
                }
            )

    for row in gating_events:
        if _upper(row.get("phase")) != "LIVE":
            continue
        reason = _norm(row.get("reason_code"))
        if "startup_resync" in reason.lower():
            offenders.append(
                _offender(
                    source="gating_events.jsonl",
                    line=row.get("_line"),
                    ts=_pick_time(row),
                    cid=_cid(row),
                    kind="stale_startup_resync_while_armed_live",
                    details={
                        "phase": _upper(row.get("phase")),
                        "reason_code": row.get("reason_code"),
                        "blocked_by": row.get("blocked_by"),
                    },
                )
            )
            break

    offenders = _dedupe_offenders(offenders)
    return {
        "name": "stale_startup_resync_while_armed_live",
        "ok": len(offenders) == 0,
        "count": len(offenders),
    }, offenders


def _check_orphan_fills_and_exit_reason(
    trades: list[dict[str, Any]],
    order_events: list[dict[str, Any]],
    execution_ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    offenders: list[dict[str, Any]] = []
    trade_cids = {_cid(row) for row in trades if _cid(row)}
    trade_by_cid = {cid: row for row in trades if (cid := _cid(row))}

    for row in trades:
        cid = _cid(row)
        if not cid:
            continue
        has_exit = any(
            _has_value(row.get(k))
            for k in ("exit_ts", "actual_exit_ts", "exit_fill_ts", "exit_price", "actual_exit_price", "exit_fill_price")
        )
        exit_reason = _norm(row.get("exit_reason"))
        if has_exit and not exit_reason:
            offenders.append(
                _offender(
                    source="trades.csv",
                    line=row.get("_line"),
                    ts=_first_nonempty(row, ("exit_ts", "actual_exit_ts", "exit_fill_ts", "entry_ts")),
                    cid=cid,
                    kind="missing_exit_reason",
                    details={
                        "side": row.get("side"),
                        "entry_ts": row.get("entry_ts"),
                        "exit_ts": row.get("exit_ts"),
                        "actual_exit_ts": row.get("actual_exit_ts"),
                        "exit_fill_ts": row.get("exit_fill_ts"),
                        "actual_exit_price": row.get("actual_exit_price"),
                        "exit_fill_price": row.get("exit_fill_price"),
                    },
                )
            )

    for row in order_events + execution_ledger:
        cid = _cid(row)
        if not cid:
            continue
        status = _norm(row.get("status")).lower()
        if status not in FILLED_STATUSES and "fill" not in status:
            continue
        if cid not in trade_cids:
            offenders.append(
                _offender(
                    source="execution/order ledgers",
                    line=row.get("_line"),
                    ts=_pick_time(row),
                    cid=cid,
                    kind="orphan_fill",
                    details={
                        "status": row.get("status"),
                        "instrument": row.get("instrument"),
                        "side": row.get("side"),
                        "qty": row.get("qty"),
                    },
                )
            )

    offenders = _dedupe_offenders(offenders)
    return {
        "name": "orphan_fills_and_missing_exit_reason",
        "ok": len(offenders) == 0,
        "count": len(offenders),
    }, offenders


def audit_run(run_dir: Path, max_offenders: int = 25) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    status, status_err = _load_json(run_dir / "status.json")
    gating_events, gating_err = _load_jsonl(run_dir / "gating_events.jsonl")
    signals, signals_err = _load_csv(run_dir / "signals.csv")
    trades, trades_err = _load_csv(run_dir / "trades.csv")
    order_events, order_err = _load_jsonl(run_dir / "order_events.jsonl")
    execution_ledger, exec_err = _load_jsonl(run_dir / "execution_ledger.jsonl")

    input_errors = [e for e in (status_err, gating_err, signals_err, trades_err, order_err, exec_err) if e]

    checks: dict[str, Any] = {}
    offenders_by_check: dict[str, list[dict[str, Any]]] = {}

    check, offenders = _check_emitted_open_setup_blocked(gating_events, signals)
    checks[check["name"]] = check
    offenders_by_check[check["name"]] = _top_samples(offenders, max_offenders)

    check, offenders = _check_side_presence(gating_events, signals)
    checks[check["name"]] = check
    offenders_by_check[check["name"]] = _top_samples(offenders, max_offenders)

    check, offenders = _check_blocked_vs_sent(gating_events, signals, order_events, execution_ledger)
    checks[check["name"]] = check
    offenders_by_check[check["name"]] = _top_samples(offenders, max_offenders)

    check, offenders = _check_stale_startup_resync(status, gating_events)
    checks[check["name"]] = check
    offenders_by_check[check["name"]] = _top_samples(offenders, max_offenders)

    check, offenders = _check_orphan_fills_and_exit_reason(trades, order_events, execution_ledger)
    checks[check["name"]] = check
    offenders_by_check[check["name"]] = _top_samples(offenders, max_offenders)

    phase_summary = _phase_summary(gating_events)

    total_failures = sum(1 for check in checks.values() if not check["ok"])
    warnings_present = bool(input_errors)
    ok = total_failures == 0
    overall_label = "PASS_WITH_WARNINGS" if ok and warnings_present else ("PASS" if ok else "FAIL")

    summary_lines = [
        f"Run dir: {run_dir}",
        f"Overall: {overall_label}",
        f"Phase composition: BACKFILL={phase_summary['counts'].get('BACKFILL', 0)}, "
        f"CATCHUP={phase_summary['counts'].get('CATCHUP', 0)}, "
        f"LIVE={phase_summary['counts'].get('LIVE', 0)}",
    ]
    for check_name, check in checks.items():
        summary_lines.append(f"- {check_name}: {'PASS' if check['ok'] else 'FAIL'} ({check['count']})")
    if input_errors:
        summary_lines.append("- input warnings:")
        for err in input_errors:
            summary_lines.append(f"  * {err}")

    report = {
        "schema_version": "1.0",
        "generated_at": _utc_now_iso(),
        "run_dir": str(run_dir),
        "ok": ok,
        "pass": ok,
        "warnings_present": warnings_present,
        "input_warnings": input_errors,
        "files": {
            "status.json": bool(status),
            "gating_events.jsonl": len(gating_events),
            "signals.csv": len(signals),
            "trades.csv": len(trades),
            "order_events.jsonl": len(order_events),
            "execution_ledger.jsonl": len(execution_ledger),
        },
        "phase_summary": phase_summary,
        "checks": checks,
        "offenders": offenders_by_check,
        "summary_lines": summary_lines,
    }
    return report


def _print_human(report: dict[str, Any]) -> None:
    print(report["summary_lines"][0])
    print(report["summary_lines"][1])
    print(report["summary_lines"][2])
    print("Checks:")
    for name, check in report["checks"].items():
        print(f"  {name}: {'PASS' if check['ok'] else 'FAIL'} ({check['count']})")
        offenders = report["offenders"].get(name, [])
        for item in offenders[:5]:
            details = item.get("details", {})
            cid = item.get("cid") or "(no cid)"
            ts = item.get("ts") or "(no ts)"
            print(f"    - {ts} cid={cid} {details}")
    if report["input_warnings"]:
        print("Input warnings:")
        for err in report["input_warnings"]:
            print(f"  - {err}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a run directory for live/backfill parity invariants."
    )
    parser.add_argument("--run-dir", "--run_dir", required=True, help="Run directory to audit.")
    parser.add_argument("--json-out", help="Optional path to write the JSON report.")
    parser.add_argument(
        "--max-offenders",
        type=int,
        default=25,
        help="Maximum offender samples to retain per check in the JSON output.",
    )
    args = parser.parse_args()

    report = audit_run(Path(args.run_dir), max_offenders=max(1, int(args.max_offenders)))
    _print_human(report)

    if args.json_out:
        json_out = Path(args.json_out).expanduser().resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON report to {json_out}")

    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
