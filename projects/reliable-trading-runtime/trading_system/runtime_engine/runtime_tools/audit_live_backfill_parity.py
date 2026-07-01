from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


DECISION_FILES = ("decision_events.csv", "state.csv", "lifecycle_events.jsonl")
ORDER_INTENT_FILES = ("order_intents.csv", "signal_to_order.jsonl")
TRADES_FILES = ("trades.csv",)
STATUS_FILES = ("status.json",)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _first_existing(base: Path, names: Sequence[str]) -> Path | None:
    if base.is_file():
        return base
    for name in names:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def _upper(value: Any) -> str:
    return str(value or "").strip().upper()


def _num(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, "", 0, "0", "false", "False", "FALSE", "no", "No", "NO"):
        return False
    return bool(value)


def _action_sign(side: str) -> int:
    side_u = _upper(side)
    if side_u == "LONG":
        return 1
    if side_u == "SHORT":
        return -1
    return 0


def _effective_action(row: dict[str, Any]) -> str:
    requested = _upper(row.get("requested_action") or row.get("action"))
    resolved = _upper(row.get("resolved_action") or row.get("final_action") or row.get("execution_intent_action"))
    display = _upper(row.get("display_action"))
    action = resolved or requested or display
    if requested == "HOLD" and resolved == "OPEN":
        return "OPEN"
    if action in {"OPEN", "CLOSE", "FLIP", "HOLD"}:
        return action
    if display in {"OPEN", "CLOSE", "FLIP", "HOLD"}:
        return display
    return requested or "HOLD"


def _dedupe_key(row: dict[str, Any]) -> str:
    action_key = _upper(row.get("requested_action") or row.get("resolved_action") or row.get("order_action") or row.get("action"))
    resolved_key = _upper(row.get("resolved_action") or row.get("order_action") or row.get("action"))
    return "|".join(
        [
            str(row.get("bar_ts") or row.get("Datetime") or row.get("ts") or ""),
            str(row.get("transition_id") or ""),
            str(row.get("signal_id") or ""),
            str(row.get("client_order_id") or ""),
            action_key,
            resolved_key,
        ]
    )


def _normalize_state_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        bar_ts = str(row.get("Datetime") or row.get("bar_ts") or row.get("ts") or "")
        requested = _upper(row.get("requested_action") or row.get("action"))
        resolved = _upper(row.get("resolved_action") or row.get("execution_intent_action") or requested)
        side = _upper(row.get("side"))
        effective = _effective_action(row)
        entry = {
            "row_index": idx,
            "bar_ts": bar_ts,
            "phase": _upper(row.get("phase")),
            "requested_action": requested,
            "resolved_action": resolved,
            "display_action": _upper(row.get("display_action") or row.get("action")),
            "execution_intent_action": _upper(row.get("execution_intent_action") or resolved or requested),
            "side": side,
            "price": _num(row.get("price")),
            "prob": _num(row.get("prob")),
            "position_before": None,
            "position_after": None,
            "transition_id": str(row.get("transition_id") or row.get("client_order_id") or row.get("signal_id") or ""),
            "signal_id": str(row.get("signal_id") or ""),
            "client_order_id": str(row.get("client_order_id") or ""),
            "dedupe_key": str(row.get("dedupe_key") or _dedupe_key(row)),
            "model_version": str(row.get("model_version") or ""),
            "config_hash": str(row.get("config_hash") or ""),
            "feature_hash": str(row.get("feature_hash") or row.get("features_hash") or ""),
            "accepted_for_execution": _bool(row.get("emit_allowed", True)),
            "rejection_reason": str(row.get("blocked_candidate_reason") or row.get("blocked_reason") or ""),
            "effective_action": effective,
        }
        out.append(entry)

    pos = 0
    open_price = None
    open_side = 0
    for row in out:
        row["position_before"] = pos
        action = row["effective_action"]
        sign = _action_sign(row["side"])
        if action == "OPEN":
            if pos == 0:
                pos = sign
                open_side = sign
                open_price = row["price"]
        elif action == "CLOSE":
            pos = 0
            open_side = 0
            open_price = None
        elif action == "FLIP":
            pos = sign if sign != 0 else (-open_side if open_side != 0 else 0)
            open_side = pos
            open_price = row["price"]
        row["position_after"] = pos
        row["entry_price"] = open_price
    return out


def _trade_points(rows: list[dict[str, Any]]) -> float:
    pos = 0
    entry_price = None
    total = 0.0
    for row in rows:
        action = row["effective_action"]
        sign = _action_sign(row["side"])
        price = row["price"]
        if price is None:
            continue
        if action == "OPEN":
            if pos == 0:
                pos = sign
                entry_price = price
        elif action == "CLOSE":
            if pos != 0 and entry_price is not None:
                total += (price - entry_price) * pos
            pos = 0
            entry_price = None
        elif action == "FLIP":
            if pos != 0 and entry_price is not None:
                total += (price - entry_price) * pos
            pos = sign if sign != 0 else (-pos if pos != 0 else 0)
            entry_price = price
    return total


def _load_tape(base: Path) -> dict[str, Any]:
    if base.is_file():
        base = base.parent
    decision_path = _first_existing(base, DECISION_FILES)
    order_path = _first_existing(base, ORDER_INTENT_FILES)
    trades_path = _first_existing(base, TRADES_FILES)
    status_path = _first_existing(base, STATUS_FILES)
    state_path = base / "state.csv"
    state_rows = _normalize_state_rows(_read_csv(state_path)) if state_path.exists() else []
    decision_rows = _read_csv(decision_path) if decision_path and decision_path.suffix == ".csv" else []
    if not decision_rows and state_rows:
        decision_rows = list(state_rows)
    else:
        enriched: list[dict[str, Any]] = []
        by_key = {str(row["dedupe_key"]): row for row in state_rows}
        for row in decision_rows:
            key = str(row.get("dedupe_key") or _dedupe_key(row))
            base_row = by_key.get(key)
            if base_row:
                merged = dict(base_row)
                merged.update(row)
                enriched.append(merged)
            else:
                enriched.append(dict(row))
        decision_rows = _normalize_state_rows(enriched)
    order_rows = _read_csv(order_path) if order_path and order_path.suffix == ".csv" else _read_jsonl(order_path) if order_path else []
    trades = _read_csv(trades_path) if trades_path else []
    status = _read_json(status_path) if status_path else {}
    return {
        "base": base,
        "state_rows": state_rows,
        "decision_rows": decision_rows,
        "order_rows": order_rows,
        "trades": trades,
        "status": status,
    }


def _derive_order_intents(decisions: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in decisions:
        action = row["effective_action"]
        if action not in {"OPEN", "CLOSE", "FLIP"}:
            continue
        transition_id = row["transition_id"] or row["client_order_id"] or row["signal_id"]
        client_order_id = row["client_order_id"] or transition_id
        base = {
            "transition_id": transition_id,
            "bar_ts": row["bar_ts"],
            "side": row["side"],
            "qty": 1,
            "intended_price": row["price"],
            "mode": mode,
            "client_order_id": client_order_id,
            "broker_order_id": "",
        }
        if action == "FLIP":
            rows.append({**base, "requested_action": "FLIP", "resolved_action": "CLOSE", "dedupe_key": f"{transition_id}|CLOSE|1", "order_intent_id": f"{transition_id}|CLOSE|1", "parent_transition_id": transition_id, "sequence_in_transition": 1, "order_action": "CLOSE", "status": "sent" if row["accepted_for_execution"] else "rejected", "reject_reason": row["rejection_reason"]})
            rows.append({**base, "requested_action": "FLIP", "resolved_action": "OPEN", "dedupe_key": f"{transition_id}|OPEN|2", "order_intent_id": f"{transition_id}|OPEN|2", "parent_transition_id": transition_id, "sequence_in_transition": 2, "order_action": "OPEN", "status": "sent" if row["accepted_for_execution"] else "rejected", "reject_reason": row["rejection_reason"]})
        else:
            rows.append({**base, "requested_action": action, "resolved_action": action, "dedupe_key": f"{transition_id}|{action}|1", "order_intent_id": f"{transition_id}|{action}|1", "parent_transition_id": transition_id if action in {"OPEN", "CLOSE"} else "", "sequence_in_transition": 1, "order_action": action, "status": "sent" if row["accepted_for_execution"] else "rejected", "reject_reason": row["rejection_reason"]})
    return rows


def _compare_rows(live: list[dict[str, Any]], backfill: list[dict[str, Any]], fields: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    live_map = {str(row.get("dedupe_key") or _dedupe_key(row)): row for row in live}
    backfill_map = {str(row.get("dedupe_key") or _dedupe_key(row)): row for row in backfill}
    all_keys = sorted(set(live_map) | set(backfill_map))
    mismatches: list[dict[str, Any]] = []
    reconciliation: list[dict[str, Any]] = []
    for key in all_keys:
        lrow = live_map.get(key)
        brow = backfill_map.get(key)
        rec = {
            "dedupe_key": key,
            "live_present": bool(lrow),
            "backfill_present": bool(brow),
        }
        if lrow:
            for field in fields:
                rec[f"live_{field}"] = lrow.get(field)
        if brow:
            for field in fields:
                rec[f"backfill_{field}"] = brow.get(field)
        if lrow and brow:
            row_mismatch = False
            for field in fields:
                if str(lrow.get(field) or "") != str(brow.get(field) or ""):
                    row_mismatch = True
                    mismatches.append(
                        {
                            "mismatch_type": "field_mismatch",
                            "bar_ts": lrow.get("bar_ts") or brow.get("bar_ts"),
                            "transition_id": lrow.get("transition_id") or brow.get("transition_id"),
                            "signal_id": lrow.get("signal_id") or brow.get("signal_id"),
                            "client_order_id": lrow.get("client_order_id") or brow.get("client_order_id"),
                            "field": field,
                            "live_value": lrow.get(field),
                            "backfill_value": brow.get(field),
                        }
                    )
            rec["match"] = not row_mismatch
        else:
            mismatch_type = "missing_backfill" if lrow else "backfill_only"
            mismatches.append(
                {
                    "mismatch_type": mismatch_type,
                    "bar_ts": (lrow or brow or {}).get("bar_ts"),
                    "transition_id": (lrow or brow or {}).get("transition_id"),
                    "signal_id": (lrow or brow or {}).get("signal_id"),
                    "client_order_id": (lrow or brow or {}).get("client_order_id"),
                    "field": "",
                    "live_value": "present" if lrow else "",
                    "backfill_value": "present" if brow else "",
                    "live_requested_action": _upper((lrow or {}).get("requested_action")),
                    "backfill_requested_action": _upper((brow or {}).get("requested_action")),
                    "live_resolved_action": _upper((lrow or {}).get("resolved_action")),
                    "backfill_resolved_action": _upper((brow or {}).get("resolved_action")),
                    "live_order_action": _upper((lrow or {}).get("order_action")),
                    "backfill_order_action": _upper((brow or {}).get("order_action")),
                }
            )
            rec["match"] = False
        reconciliation.append(rec)
    return mismatches, reconciliation


def _pnl_from_trades(rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        for field in ("realized_points", "pnl_points", "points", "pnl"):
            val = _num(row.get(field))
            if val is not None:
                total += val
                break
        else:
            entry = _num(row.get("entry_price"))
            exit_px = _num(row.get("exit_price"))
            side = _action_sign(row.get("side"))
            if entry is not None and exit_px is not None and side != 0:
                total += (exit_px - entry) * side
    return total


def audit_live_backfill_parity(live: Path, backfill: Path, out: Path) -> dict[str, Any]:
    live_tape = _load_tape(live)
    backfill_tape = _load_tape(backfill)
    out.mkdir(parents=True, exist_ok=True)

    live_decisions = live_tape["decision_rows"]
    backfill_decisions = backfill_tape["decision_rows"]
    live_orders = live_tape["order_rows"] or _derive_order_intents(live_decisions, str(live_tape["status"].get("run_mode") or "LIVE").upper())
    backfill_orders = backfill_tape["order_rows"] or _derive_order_intents(backfill_decisions, str(backfill_tape["status"].get("run_mode") or "BACKFILL").upper())

    decision_fields = [
        "bar_ts",
        "phase",
        "requested_action",
        "resolved_action",
        "execution_intent_action",
        "side",
        "price",
        "prob",
        "position_before",
        "position_after",
        "model_version",
        "feature_hash",
        "config_hash",
        "transition_id",
        "signal_id",
        "dedupe_key",
        "accepted_for_execution",
        "rejection_reason",
    ]
    order_fields = [
        "order_action",
        "qty",
        "intended_price",
        "mode",
        "status",
        "client_order_id",
        "broker_order_id",
        "reject_reason",
        "parent_transition_id",
        "sequence_in_transition",
        "transition_id",
        "side",
        "bar_ts",
    ]
    decision_mismatches, decision_recon = _compare_rows(live_decisions, backfill_decisions, decision_fields)
    order_mismatches, order_recon = _compare_rows(live_orders, backfill_orders, order_fields)
    mismatches = decision_mismatches + order_mismatches
    mismatch_types: dict[str, int] = {}
    for row in mismatches:
        mismatch_types[str(row.get("mismatch_type") or "field_mismatch")] = mismatch_types.get(str(row.get("mismatch_type") or "field_mismatch"), 0) + 1
    flip_mismatches = len(
        [
            row
            for row in mismatches
            if "FLIP" in {
                _upper(row.get("live_requested_action")),
                _upper(row.get("backfill_requested_action")),
                _upper(row.get("live_resolved_action")),
                _upper(row.get("backfill_resolved_action")),
                _upper(row.get("live_order_action")),
                _upper(row.get("backfill_order_action")),
            }
        ]
    )
    decision_action_counts = {action: 0 for action in ("OPEN", "CLOSE", "FLIP", "HOLD")}
    for row in live_decisions:
        decision_action_counts[_effective_action(row)] = decision_action_counts.get(_effective_action(row), 0) + 1
    backfill_action_counts = {action: 0 for action in ("OPEN", "CLOSE", "FLIP", "HOLD")}
    for row in backfill_decisions:
        backfill_action_counts[_effective_action(row)] = backfill_action_counts.get(_effective_action(row), 0) + 1
    order_action_counts = {action: 0 for action in ("OPEN", "CLOSE", "FLIP")}
    for row in live_orders + backfill_orders:
        action = _upper(row.get("order_action") or row.get("resolved_action") or row.get("requested_action"))
        if action in order_action_counts:
            order_action_counts[action] += 1
    field_mismatch_counts: dict[str, int] = {}
    for row in mismatches:
        field = str(row.get("field") or "")
        if field:
            field_mismatch_counts[field] = field_mismatch_counts.get(field, 0) + 1
    _write_csv(out / "parity_mismatches.csv", mismatches)
    _write_csv(out / "lifecycle_replay_from_live.csv", live_decisions)
    _write_csv(out / "lifecycle_replay_from_backfill.csv", backfill_decisions)
    _write_csv(out / "order_intent_reconciliation.csv", order_recon)
    report = {
        "live_decisions_total": len(live_decisions),
        "backfill_decisions_total": len(backfill_decisions),
        "exact_matches": len([row for row in decision_recon if row.get("match")]) + len([row for row in order_recon if row.get("match")]),
        "mismatches_total": len(mismatches),
        "skipped_live_trades": len([m for m in mismatches if m.get("mismatch_type") == "missing_backfill"]),
        "backfill_only_trades": len([m for m in mismatches if m.get("mismatch_type") == "backfill_only"]),
        "flip_mismatches": flip_mismatches,
        "open_close_mismatches": sum(field_mismatch_counts.get(field, 0) for field in ("requested_action", "resolved_action", "execution_intent_action")),
        "hold_open_mismatches": sum(field_mismatch_counts.get(field, 0) for field in ("resolved_action", "execution_intent_action")) if any(_upper(row.get("requested_action") or row.get("action")) == "HOLD" for row in live_decisions + backfill_decisions) else 0,
        "duplicate_open_overlap_events": len([row for row in live_decisions if _effective_action(row) == "OPEN" and int(row.get("position_before") or 0) != 0]),
        "rejected_events": len([row for row in live_decisions + backfill_decisions if not _bool(row.get("accepted_for_execution", True))]),
        "lifecycle_pnl_live_points": _trade_points(live_decisions),
        "lifecycle_pnl_backfill_points": _trade_points(backfill_decisions),
        "trades_csv_pnl_live": _pnl_from_trades(live_tape["trades"]),
        "trades_csv_pnl_backfill": _pnl_from_trades(backfill_tape["trades"]),
        "status_live": live_tape["status"],
        "status_backfill": backfill_tape["status"],
        "mismatch_types": mismatch_types,
        "field_mismatch_counts": field_mismatch_counts,
        "live_action_counts": decision_action_counts,
        "backfill_action_counts": backfill_action_counts,
        "order_action_counts": order_action_counts,
        "mismatches": mismatches,
        "decision_recon": decision_recon,
        "order_recon": order_recon,
    }
    (out / "parity_report.md").write_text(
        "\n".join(
            [
                "# LIVE/BACKFILL Parity Report",
                "",
                f"- live decisions: {report['live_decisions_total']}",
                f"- backfill decisions: {report['backfill_decisions_total']}",
                f"- exact matches: {report['exact_matches']}",
                f"- mismatches: {report['mismatches_total']}",
                f"- skipped live trades: {report['skipped_live_trades']}",
                f"- backfill-only trades: {report['backfill_only_trades']}",
                f"- flip mismatches: {report['flip_mismatches']}",
                f"- open/close mismatches: {report['open_close_mismatches']}",
                f"- hold->open mismatches: {report['hold_open_mismatches']}",
                f"- duplicate/open-overlap events: {report['duplicate_open_overlap_events']}",
                f"- rejected events: {report['rejected_events']}",
                f"- lifecycle pnl live points: {report['lifecycle_pnl_live_points']}",
                f"- lifecycle pnl backfill points: {report['lifecycle_pnl_backfill_points']}",
                f"- trades.csv pnl live: {report['trades_csv_pnl_live']}",
                f"- trades.csv pnl backfill: {report['trades_csv_pnl_backfill']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit live/backfill causal parity.")
    parser.add_argument("--live", required=True, type=Path)
    parser.add_argument("--backfill", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = audit_live_backfill_parity(args.live, args.backfill, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
