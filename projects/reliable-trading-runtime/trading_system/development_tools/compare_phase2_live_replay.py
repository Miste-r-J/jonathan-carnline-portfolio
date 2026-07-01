from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.development_tools.build_mff_phase2_replay_dataset import (
    _iso_utc,
    _parse_datetime,
    _safe_float,
    normalize_contract,
)


METRICS = ("pnl_usd", "mfe_points", "mae_points")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_trade_file(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("trades.csv", "actual_fill_baseline.csv", "replay_trades.csv", "trades.json"):
        candidate = path / name
        if candidate.exists():
            return candidate
    csv_candidates = sorted(path.glob("*trades*.csv")) + sorted(path.glob("*fill*baseline*.csv"))
    if csv_candidates:
        return csv_candidates[0]
    raise FileNotFoundError(f"no supported trade output found under {path}")


def _load_rows(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    source = _resolve_trade_file(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8", errors="replace"))
        if isinstance(payload, dict):
            payload = payload.get("trades") or payload.get("rows") or []
        if not isinstance(payload, list):
            raise ValueError(f"JSON trade output must be a list or contain trades/rows: {source}")
        return source, [dict(row) for row in payload if isinstance(row, Mapping)]
    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return source, [dict(row) for row in csv.DictReader(handle)]


def _first(row: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip() != "":
            return value
    return None


def _canonical_trade(row: Mapping[str, Any], *, point_value: float) -> dict[str, Any]:
    side = str(_first(row, "side", "direction") or "").strip().upper()
    qty = _safe_float(_first(row, "filled_qty", "qty", "size", "contracts")) or 1.0
    entry_price = _safe_float(_first(row, "actual_entry_price", "entry_fill_price", "entry_price"))
    exit_price = _safe_float(_first(row, "actual_exit_price", "exit_fill_price", "exit_price"))
    pnl_usd = _safe_float(_first(row, "pnl_usd", "net_pnl_usd", "gross_pnl_usd", "realized_pnl"))
    if pnl_usd is None and entry_price is not None and exit_price is not None:
        points = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
        pnl_usd = points * qty * point_value if side in {"LONG", "SHORT"} else None
    entry_ts = _iso_utc(_first(row, "actual_entry_ts", "entry_fill_ts", "entry_ts", "datetime"))
    exit_ts = _iso_utc(_first(row, "actual_exit_ts", "exit_fill_ts", "exit_ts"))
    contract = normalize_contract(_first(row, "contract", "instrument", "exec_instrument", "client_order_id"))
    client_order_id = str(_first(row, "client_order_id", "correlation_id", "intent_id") or "")
    prediction_id = str(_first(row, "prediction_id", "signal_id") or "")
    explicit_key = str(_first(row, "trade_key") or "")
    match_seed = "|".join(
        [
            prediction_id or client_order_id,
            contract,
            entry_ts,
            side,
            f"{entry_price:.10g}" if entry_price is not None else "",
        ]
    )
    match_key = explicit_key or hashlib.sha256(match_seed.encode("utf-8")).hexdigest()[:24]
    parsed_entry = _parse_datetime(entry_ts)
    if parsed_entry is not None:
        floored_minute = parsed_entry.minute - (parsed_entry.minute % 5)
        bar_ts = parsed_entry.replace(minute=floored_minute, second=0, microsecond=0).isoformat()
    else:
        bar_ts = entry_ts
    return {
        "match_key": match_key,
        "match_signature": "|".join((contract, bar_ts, side)),
        "contract": contract,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "side": side,
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_usd": pnl_usd,
        "mfe_points": _safe_float(_first(row, "mfe_points", "mfe")),
        "mae_points": _safe_float(_first(row, "mae_points", "mae")),
        "client_order_id": client_order_id,
        "prediction_id": prediction_id,
    }


def load_trade_output(path: Path, *, point_value: float = 50.0) -> dict[str, Any]:
    source, raw_rows = _load_rows(path)
    trades = [_canonical_trade(row, point_value=point_value) for row in raw_rows]
    trades.sort(key=lambda row: (row["entry_ts"], row["match_key"]))
    return {
        "path": str(source.resolve()),
        "sha256": _sha256_file(source),
        "rows": len(raw_rows),
        "trades": trades,
    }


def _summary(trades: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(trades)
    result: dict[str, Any] = {"trade_count": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        result[metric] = {
            "available": len(values),
            "sum": sum(values) if values else None,
            "mean": (sum(values) / len(values)) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    pnl_values = [float(row["pnl_usd"]) for row in rows if row.get("pnl_usd") is not None]
    result["wins"] = sum(value > 0 for value in pnl_values)
    result["losses"] = sum(value < 0 for value in pnl_values)
    result["flats"] = sum(value == 0 for value in pnl_values)
    return result


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def compare_pair(left: Mapping[str, Any], right: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    left_by_key = {str(row["match_key"]): row for row in left["trades"]}
    right_by_key = {str(row["match_key"]): row for row in right["trades"]}
    matched: list[tuple[str, str, str]] = [
        (key, key, "exact_key") for key in sorted(set(left_by_key) & set(right_by_key))
    ]
    matched_left = {item[0] for item in matched}
    matched_right = {item[1] for item in matched}
    left_by_signature: dict[str, list[str]] = {}
    right_by_signature: dict[str, list[str]] = {}
    for key, row in left_by_key.items():
        if key not in matched_left:
            left_by_signature.setdefault(str(row["match_signature"]), []).append(key)
    for key, row in right_by_key.items():
        if key not in matched_right:
            right_by_signature.setdefault(str(row["match_signature"]), []).append(key)
    for signature in sorted(set(left_by_signature) & set(right_by_signature)):
        left_keys = left_by_signature[signature]
        right_keys = right_by_signature[signature]
        if len(left_keys) == 1 and len(right_keys) == 1:
            matched.append((left_keys[0], right_keys[0], "contract_5m_bar_side"))
            matched_left.add(left_keys[0])
            matched_right.add(right_keys[0])
    trade_deltas = []
    for left_key, right_key, match_method in matched:
        old = left_by_key[left_key]
        new = right_by_key[right_key]
        trade_deltas.append(
            {
                "left_match_key": left_key,
                "right_match_key": right_key,
                "match_method": match_method,
                "entry_ts": new.get("entry_ts") or old.get("entry_ts"),
                "side": new.get("side") or old.get("side"),
                "pnl_usd_delta": _delta(old.get("pnl_usd"), new.get("pnl_usd")),
                "mfe_points_delta": _delta(old.get("mfe_points"), new.get("mfe_points")),
                "mae_points_delta": _delta(old.get("mae_points"), new.get("mae_points")),
            }
        )
    left_summary = _summary(left["trades"])
    right_summary = _summary(right["trades"])
    return {
        "label": label,
        "left_path": left["path"],
        "right_path": right["path"],
        "left_summary": left_summary,
        "right_summary": right_summary,
        "aggregate_delta": {
            "trade_count": right_summary["trade_count"] - left_summary["trade_count"],
            **{
                f"{metric}_sum": _delta(
                    left_summary[metric]["sum"], right_summary[metric]["sum"]
                )
                for metric in METRICS
            },
        },
        "matched_trade_count": len(matched),
        "left_only_keys": sorted(set(left_by_key) - matched_left),
        "right_only_keys": sorted(set(right_by_key) - matched_right),
        "trade_deltas": trade_deltas,
    }


def compare_live_old_new(
    *,
    live_baseline: Path | None,
    old_replay: Path,
    new_replay: Path,
    point_value: float = 50.0,
) -> dict[str, Any]:
    old = load_trade_output(old_replay, point_value=point_value)
    new = load_trade_output(new_replay, point_value=point_value)
    inputs: dict[str, Any] = {
        "old_replay": {key: old[key] for key in ("path", "sha256", "rows")},
        "new_replay": {key: new[key] for key in ("path", "sha256", "rows")},
    }
    comparisons = [compare_pair(old, new, label="old_replay_vs_new_replay")]
    if live_baseline is not None:
        live = load_trade_output(live_baseline, point_value=point_value)
        inputs["live_baseline"] = {key: live[key] for key in ("path", "sha256", "rows")}
        comparisons.extend(
            [
                compare_pair(live, old, label="live_vs_old_replay"),
                compare_pair(live, new, label="live_vs_new_replay"),
            ]
        )
    return {
        "schema_version": 1,
        "point_value": point_value,
        "inputs": inputs,
        "comparisons": comparisons,
    }


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != data:
        raise FileExistsError(f"refusing to overwrite different comparison report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Phase-2 live actual fills with old/new replay outputs."
    )
    parser.add_argument("--live-baseline", type=Path)
    parser.add_argument("--old-replay", type=Path, required=True)
    parser.add_argument("--new-replay", type=Path, required=True)
    parser.add_argument("--point-value", type=float, default=50.0)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = compare_live_old_new(
        live_baseline=args.live_baseline,
        old_replay=args.old_replay,
        new_replay=args.new_replay,
        point_value=args.point_value,
    )
    if args.out:
        _write_report(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
