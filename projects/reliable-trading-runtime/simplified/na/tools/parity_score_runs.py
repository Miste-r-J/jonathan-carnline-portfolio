from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _read_price_by_ts(state_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not state_csv.exists():
        return out
    with state_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = str(row.get("datetime") or "").strip()
            if not ts:
                continue
            try:
                px = float(row.get("price"))
            except Exception:
                continue
            out[ts] = px
    return out


def _backfill_intents(run_dir: Path) -> List[Dict[str, Any]]:
    lifecycle_path = run_dir / "lifecycle_events.jsonl"
    if lifecycle_path.exists():
        out: List[Dict[str, Any]] = []
        for row in _read_jsonl(lifecycle_path):
            if str(row.get("phase") or "").upper() != "BACKFILL":
                continue
            action = str(row.get("requested_action") or row.get("resolved_action") or "").upper()
            if action not in {"OPEN", "CLOSE", "FLIP"}:
                continue
            item = dict(row)
            item["signal_action"] = action
            out.append(item)
        if out:
            return out
    path = run_dir / "signal_to_order.jsonl"
    rows = _read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("type") == "header":
            continue
        if str(row.get("phase") or "").upper() != "BACKFILL":
            continue
        action = str(row.get("signal_action") or row.get("action") or "").upper()
        if action not in {"OPEN", "CLOSE", "FLIP"}:
            continue
        out.append(row)
    return out


def _key_for_overlap(row: Mapping[str, Any]) -> str:
    bar_ts = str(row.get("bar_ts") or "")
    action = str(row.get("signal_action") or row.get("action") or "").upper()
    side = str(row.get("side") or "").upper()
    return f"{bar_ts}|{action}|{side}"


def _counterfactual_pnl_points(intents: List[Mapping[str, Any]], price_by_ts: Mapping[str, float]) -> Dict[str, Any]:
    ordered: List[Tuple[str, str, str, str]] = []
    for row in intents:
        ordered.append(
            (
                str(row.get("ts") or ""),
                str(row.get("bar_ts") or ""),
                str(row.get("signal_action") or row.get("action") or "").upper(),
                str(row.get("side") or "").upper(),
            )
        )
    ordered.sort(key=lambda t: t[0])

    pos = 0
    entry = None
    realized = 0.0
    closed = 0
    ignored = 0
    missing_price = 0

    for _, bar_ts, action, side in ordered:
        px = price_by_ts.get(bar_ts)
        if px is None:
            missing_price += 1
            continue
        desired = 1 if side.startswith("L") else -1 if side.startswith("S") else 0
        if action == "OPEN":
            if pos == 0 and desired != 0:
                pos = desired
                entry = px
            else:
                ignored += 1
        elif action == "CLOSE":
            if pos != 0 and entry is not None:
                realized += (px - entry) if pos == 1 else (entry - px)
                closed += 1
            else:
                ignored += 1
            pos = 0
            entry = None
        elif action == "FLIP":
            if pos != 0 and entry is not None:
                realized += (px - entry) if pos == 1 else (entry - px)
                closed += 1
            if desired != 0:
                pos = desired
                entry = px
            else:
                pos = -pos if pos != 0 else 0
                entry = px if pos != 0 else None

    return {
        "realized_points": realized,
        "realized_usd_1_contract": realized * 50.0,
        "closed_trades": closed,
        "open_position_at_end": pos,
        "ignored_events": ignored,
        "missing_price_events": missing_price,
    }


def score_runs(baseline_dir: Path, candidate_dir: Path) -> Dict[str, Any]:
    base_intents = _backfill_intents(baseline_dir)
    cand_intents = _backfill_intents(candidate_dir)

    base_keys = Counter(_key_for_overlap(r) for r in base_intents)
    cand_keys = Counter(_key_for_overlap(r) for r in cand_intents)
    overlap_keys = set(base_keys.keys()) & set(cand_keys.keys())
    overlap_count = sum(min(base_keys[k], cand_keys[k]) for k in overlap_keys)
    base_total = sum(base_keys.values())
    cand_total = sum(cand_keys.values())

    base_blocks = Counter()
    cand_blocks = Counter()
    for r in base_intents:
        for b in (r.get("blocked_by") or []):
            base_blocks[str(b)] += 1
    for r in cand_intents:
        for b in (r.get("blocked_by") or []):
            cand_blocks[str(b)] += 1

    tracked_codes = ["blocked_not_armed", "setup", "risk", "local_block:gates_block", "local_block:stop_already_breached"]
    block_delta = {
        code: int(cand_blocks.get(code, 0) - base_blocks.get(code, 0))
        for code in tracked_codes
    }

    base_actions = Counter(str(r.get("signal_action") or r.get("action") or "").upper() for r in base_intents)
    cand_actions = Counter(str(r.get("signal_action") or r.get("action") or "").upper() for r in cand_intents)

    base_prices = _read_price_by_ts(baseline_dir / "state.csv")
    cand_prices = _read_price_by_ts(candidate_dir / "state.csv")

    return {
        "baseline_run_dir": str(baseline_dir),
        "candidate_run_dir": str(candidate_dir),
        "lifecycle_source": {
            "baseline": "lifecycle_events.jsonl" if (baseline_dir / "lifecycle_events.jsonl").exists() else "signal_to_order.jsonl",
            "candidate": "lifecycle_events.jsonl" if (candidate_dir / "lifecycle_events.jsonl").exists() else "signal_to_order.jsonl",
        },
        "counts_by_phase_backfill": {
            "baseline": {"OPEN": int(base_actions.get("OPEN", 0)), "CLOSE": int(base_actions.get("CLOSE", 0)), "FLIP": int(base_actions.get("FLIP", 0))},
            "candidate": {"OPEN": int(cand_actions.get("OPEN", 0)), "CLOSE": int(cand_actions.get("CLOSE", 0)), "FLIP": int(cand_actions.get("FLIP", 0))},
        },
        "overlap": {
            "matched_key_count": overlap_count,
            "baseline_key_count": base_total,
            "candidate_key_count": cand_total,
            "baseline_overlap_rate": (overlap_count / base_total) if base_total > 0 else 0.0,
            "candidate_overlap_rate": (overlap_count / cand_total) if cand_total > 0 else 0.0,
        },
        "blocked_histogram_delta": block_delta,
        "counterfactual_pnl_1_contract": {
            "baseline": _counterfactual_pnl_points(base_intents, base_prices),
            "candidate": _counterfactual_pnl_points(cand_intents, cand_prices),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backfill OPEN/CLOSE/FLIP parity between two run folders.")
    parser.add_argument("--baseline", required=True, help="Baseline run directory (e.g. modelrunlivetests135).")
    parser.add_argument("--candidate", required=True, help="Candidate run directory to compare.")
    parser.add_argument("--out", default=None, help="Optional output path. Defaults to <candidate>/parity_report.json.")
    args = parser.parse_args()

    baseline = Path(args.baseline).resolve()
    candidate = Path(args.candidate).resolve()
    report = score_runs(baseline, candidate)

    out_path = Path(args.out).resolve() if args.out else (candidate / "parity_report.json")
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
