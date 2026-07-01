from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT.parent / "runs" / "live"
DEFAULT_SNAPSHOT_ROOT = (
    Path.home() / "OneDrive" / "Documents" / "bot" / "data" / "intraday" / "es"
)
MFF_ACCOUNT_RE = re.compile(r"\b(MFF[A-Z0-9]+)\b", re.IGNORECASE)
SNAPSHOT_DATE_RE = re.compile(r"(20\d{6})(?=\.csv$)", re.IGNORECASE)
MONTHS = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}
BAR_FIELDS = (
    "Contract",
    "Datetime",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "IntervalMin",
    "SourceTz",
    "SourceFile",
    "SourceSha256",
    "SourceRow",
)
FILL_FIELDS = (
    "trade_key",
    "run_name",
    "run_id",
    "account",
    "contract",
    "entry_ts",
    "exit_ts",
    "side",
    "qty",
    "actual_entry_price",
    "actual_exit_price",
    "pnl_points",
    "pnl_usd",
    "mfe_points",
    "mae_points",
    "excursion_source",
    "bar_count",
    "exit_reason",
    "client_order_id",
    "prediction_id",
    "closed",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _csv_bytes(rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    return buffer.getvalue().encode("utf-8")


def _write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise FileExistsError(f"immutable artifact collision: {path}")
        return
    path.write_bytes(data)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: Any) -> str:
    parsed = _parse_datetime(value)
    if parsed is None:
        return ""
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalize_contract(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().upper())
    if not text:
        return ""
    match = re.search(r"\b([A-Z]{1,4})[\s_]+([A-Z]{3})(\d{2})(?=$|[^A-Z0-9])", text)
    if match and match.group(2) in MONTHS:
        return f"{match.group(1)} {MONTHS[match.group(2)]}-{match.group(3)}"
    match = re.search(r"\b([A-Z]{1,4})[_\s-]+(\d{2})[-_](\d{2})(?=$|[^A-Z0-9])", text)
    if match:
        return f"{match.group(1)} {match.group(2)}-{match.group(3)}"
    match = re.search(r"\b([A-Z]{1,4})[_\s-]+(JUN|SEP)(\d{2})(?=$|[^A-Z0-9])", text)
    if match:
        return f"{match.group(1)} {MONTHS[match.group(2)]}-{match.group(3)}"
    return text.replace("_", " ")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _account_from_payload(payload: Mapping[str, Any]) -> str:
    candidates = [
        payload.get("chosen_account"),
        payload.get("detected_account"),
        payload.get("snapshot_account"),
        payload.get("account"),
    ]
    executor = payload.get("executor_last")
    if isinstance(executor, Mapping):
        candidates.append(executor.get("account"))
    for candidate in candidates:
        match = MFF_ACCOUNT_RE.search(str(candidate or ""))
        if match:
            return match.group(1).upper()
    return ""


def _scan_jsonl_for_mff(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "MFF" not in line.upper():
                    continue
                match = MFF_ACCOUNT_RE.search(line)
                if match:
                    return match.group(1).upper()
    except OSError:
        pass
    return ""


def enumerate_mff_runs(runs_root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    if not runs_root.exists():
        return inventory
    for run_dir in sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        status = _load_json(run_dir / "status.json")
        account = _account_from_payload(status)
        evidence = "status.json" if account else ""
        if not account:
            for name in ("order_events.jsonl", "exec_events.jsonl", "nt_bridge.jsonl"):
                account = _scan_jsonl_for_mff(run_dir / name)
                if account:
                    evidence = name
                    break
        if not account:
            continue
        trades_path = run_dir / "trades.csv"
        inventory.append(
            {
                "run_name": run_dir.name,
                "run_path": str(run_dir.resolve()),
                "run_id": str(status.get("run_id") or ""),
                "account": account,
                "account_evidence": evidence,
                "contract": normalize_contract(
                    status.get("exec_instrument")
                    or status.get("instrument_normalized")
                    or status.get("snapshot_instrument")
                ),
                "preset": str(status.get("preset") or status.get("primary_preset") or ""),
                "config_hash": str(status.get("config_hash") or ""),
                "status_ts": _iso_utc(status.get("ts")),
                "trades_exists": trades_path.exists(),
                "trades_sha256": _sha256_file(trades_path) if trades_path.exists() else None,
                "trades_bytes": trades_path.stat().st_size if trades_path.exists() else 0,
            }
        )
    return inventory


def discover_snapshot_files(snapshot_root: Path) -> list[Path]:
    if not snapshot_root.exists():
        return []
    selected = []
    for path in snapshot_root.glob("*.csv"):
        upper = path.name.upper()
        if re.search(r"ES[_ -](?:JUN26|SEP26|06-26|09-26)", upper):
            selected.append(path)
    return sorted(selected, key=lambda path: (_snapshot_rank(path), path.name.upper()))


def _snapshot_rank(path: Path) -> tuple[int, int]:
    match = SNAPSHOT_DATE_RE.search(path.name)
    name_date = int(match.group(1)) if match else 0
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return name_date, mtime


def _row_value(row: Mapping[str, Any], *names: str) -> str:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def canonicalize_snapshots(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    winners: dict[tuple[str, str], tuple[tuple[int, int, str, int], dict[str, Any]]] = {}
    source_inventory: list[dict[str, Any]] = []
    invalid_rows = 0
    total_rows = 0
    duplicate_rows = 0
    for path in sorted(paths, key=lambda item: (_snapshot_rank(item), item.name.upper())):
        source_hash = _sha256_file(path)
        rank = _snapshot_rank(path)
        source_rows = 0
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                total_rows += 1
                source_rows += 1
                contract = normalize_contract(_row_value(row, "Instrument", "Contract") or path.stem)
                timestamp = _iso_utc(_row_value(row, "Datetime", "Timestamp", "DateTime"))
                values = {
                    name: _safe_float(_row_value(row, name))
                    for name in ("Open", "High", "Low", "Close", "Volume", "IntervalMin")
                }
                if not contract or not timestamp or any(values[name] is None for name in ("Open", "High", "Low", "Close")):
                    invalid_rows += 1
                    continue
                key = (contract, timestamp)
                if key in winners:
                    duplicate_rows += 1
                output_row = {
                    "Contract": contract,
                    "Datetime": timestamp,
                    "Open": values["Open"],
                    "High": values["High"],
                    "Low": values["Low"],
                    "Close": values["Close"],
                    "Volume": values["Volume"] if values["Volume"] is not None else "",
                    "IntervalMin": values["IntervalMin"] if values["IntervalMin"] is not None else "",
                    "SourceTz": _row_value(row, "SourceTz", "Timezone"),
                    "SourceFile": path.name,
                    "SourceSha256": source_hash,
                    "SourceRow": row_number,
                }
                precedence = (rank[0], rank[1], path.name.upper(), row_number)
                if key not in winners or precedence >= winners[key][0]:
                    winners[key] = (precedence, output_row)
        source_inventory.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "sha256": source_hash,
                "bytes": path.stat().st_size,
                "snapshot_date_rank": rank[0],
                "mtime_ns": rank[1],
                "rows": source_rows,
            }
        )
    rows = [payload[1] for _, payload in sorted(winners.items(), key=lambda item: item[0])]
    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_contract[str(row["Contract"])].append(row)
    coverage: dict[str, Any] = {}
    for contract, contract_rows in sorted(by_contract.items()):
        times = [_parse_datetime(row["Datetime"]) for row in contract_rows]
        valid_times = [value for value in times if value is not None]
        gaps = []
        for previous, current in zip(valid_times, valid_times[1:]):
            delta_minutes = (current - previous).total_seconds() / 60.0
            if delta_minutes > 5.0:
                gaps.append(delta_minutes)
        coverage[contract] = {
            "rows": len(contract_rows),
            "start": contract_rows[0]["Datetime"],
            "end": contract_rows[-1]["Datetime"],
            "gaps_over_5m": len(gaps),
            "largest_gap_minutes": max(gaps) if gaps else 0.0,
            "source_files": sorted({str(row["SourceFile"]) for row in contract_rows}),
        }
    metadata = {
        "source_files": source_inventory,
        "input_rows": total_rows,
        "canonical_rows": len(rows),
        "invalid_rows": invalid_rows,
        "duplicate_keys_replaced_or_ignored": duplicate_rows,
        "coverage": coverage,
        "precedence": "snapshot filename date, file mtime, filename, then row number; latest wins",
    }
    return rows, metadata


def _contract_from_trade(row: Mapping[str, Any], fallback: str) -> str:
    for value in (row.get("instrument"), row.get("contract"), row.get("client_order_id")):
        normalized = normalize_contract(value)
        match = re.search(r"\b[A-Z]{1,4} \d{2}-\d{2}\b", normalized)
        if match:
            return match.group(0)
    return fallback


def _trade_excursion(
    row: Mapping[str, Any],
    contract: str,
    bars_by_contract: Mapping[str, list[dict[str, Any]]],
) -> tuple[float | None, float | None, int]:
    entry_ts = _parse_datetime(row.get("actual_entry_ts") or row.get("entry_fill_ts") or row.get("entry_ts"))
    exit_ts = _parse_datetime(row.get("actual_exit_ts") or row.get("exit_fill_ts") or row.get("exit_ts"))
    entry_price = _safe_float(row.get("actual_entry_price") or row.get("entry_fill_price") or row.get("entry_price"))
    side = str(row.get("side") or "").strip().upper()
    if entry_ts is None or exit_ts is None or entry_price is None or side not in {"LONG", "SHORT"}:
        return None, None, 0
    selected = []
    for bar in bars_by_contract.get(contract, []):
        bar_ts = _parse_datetime(bar.get("Datetime"))
        interval = _safe_float(bar.get("IntervalMin")) or 5.0
        if (
            bar_ts is not None
            and bar_ts <= exit_ts
            and bar_ts + timedelta(minutes=interval) > entry_ts
        ):
            selected.append(bar)
    if not selected:
        return None, None, 0
    highs = [float(bar["High"]) for bar in selected]
    lows = [float(bar["Low"]) for bar in selected]
    if side == "LONG":
        mfe = max(0.0, max(highs) - entry_price)
        mae = max(0.0, entry_price - min(lows))
    else:
        mfe = max(0.0, entry_price - min(lows))
        mae = max(0.0, max(highs) - entry_price)
    return mfe, mae, len(selected)


def inventory_actual_fills(
    run_inventory: Iterable[Mapping[str, Any]],
    canonical_bars: Iterable[Mapping[str, Any]],
    *,
    point_value: float = 50.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bars_by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bar in canonical_bars:
        bars_by_contract[str(bar.get("Contract") or "")].append(dict(bar))
    rows: list[dict[str, Any]] = []
    skipped_without_entry = 0
    for run in run_inventory:
        trades_path = Path(str(run["run_path"])) / "trades.csv"
        if not trades_path.exists():
            continue
        with trades_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row_number, trade in enumerate(csv.DictReader(handle), start=2):
                entry_price = _safe_float(
                    trade.get("actual_entry_price") or trade.get("entry_fill_price")
                )
                if entry_price is None:
                    skipped_without_entry += 1
                    continue
                exit_price = _safe_float(
                    trade.get("actual_exit_price") or trade.get("exit_fill_price")
                )
                qty = _safe_float(trade.get("filled_qty") or trade.get("qty")) or 1.0
                side = str(trade.get("side") or "").strip().upper()
                contract = _contract_from_trade(trade, str(run.get("contract") or ""))
                pnl_points = None
                pnl_usd = None
                if exit_price is not None and side in {"LONG", "SHORT"}:
                    pnl_points = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
                    pnl_usd = pnl_points * qty * point_value
                reported_mfe = _safe_float(trade.get("mfe_points"))
                reported_mae = _safe_float(trade.get("mae_points"))
                derived_mfe, derived_mae, bar_count = _trade_excursion(trade, contract, bars_by_contract)
                if reported_mfe is not None and reported_mae is not None:
                    mfe, mae, source = reported_mfe, reported_mae, "trades.csv"
                else:
                    mfe, mae, source = derived_mfe, derived_mae, "canonical_ohlcv_5m_overlap"
                client_order_id = str(trade.get("client_order_id") or "")
                prediction_id = str(trade.get("prediction_id") or "")
                entry_ts = _iso_utc(
                    trade.get("actual_entry_ts") or trade.get("entry_fill_ts") or trade.get("entry_ts")
                )
                exit_ts = _iso_utc(
                    trade.get("actual_exit_ts") or trade.get("exit_fill_ts") or trade.get("exit_ts")
                )
                trade_key_seed = "|".join(
                    [
                        client_order_id or prediction_id or str(run.get("run_id") or ""),
                        contract,
                        entry_ts,
                        side,
                        f"{entry_price:.10g}",
                    ]
                )
                rows.append(
                    {
                        "trade_key": hashlib.sha256(trade_key_seed.encode("utf-8")).hexdigest()[:24],
                        "run_name": run.get("run_name"),
                        "run_id": run.get("run_id"),
                        "account": run.get("account"),
                        "contract": contract,
                        "entry_ts": entry_ts,
                        "exit_ts": exit_ts,
                        "side": side,
                        "qty": qty,
                        "actual_entry_price": entry_price,
                        "actual_exit_price": exit_price if exit_price is not None else "",
                        "pnl_points": pnl_points if pnl_points is not None else "",
                        "pnl_usd": pnl_usd if pnl_usd is not None else "",
                        "mfe_points": mfe if mfe is not None else "",
                        "mae_points": mae if mae is not None else "",
                        "excursion_source": source if mfe is not None and mae is not None else "unavailable",
                        "bar_count": bar_count,
                        "exit_reason": trade.get("exit_reason") or "",
                        "client_order_id": client_order_id,
                        "prediction_id": prediction_id,
                        "closed": exit_price is not None,
                        "_source_row": row_number,
                    }
                )
    rows.sort(key=lambda row: (str(row["entry_ts"]), str(row["run_name"]), str(row["trade_key"])))
    closed = [row for row in rows if row["closed"]]
    metadata = {
        "actual_fill_rows": len(rows),
        "closed_fill_rows": len(closed),
        "open_fill_rows": len(rows) - len(closed),
        "skipped_rows_without_actual_entry": skipped_without_entry,
        "pnl_usd": sum(float(row["pnl_usd"]) for row in closed),
        "mfe_available_rows": sum(row["mfe_points"] != "" for row in rows),
        "mae_available_rows": sum(row["mae_points"] != "" for row in rows),
        "contracts": dict(sorted(Counter(str(row["contract"]) for row in rows).items())),
        "point_value": point_value,
        "pnl_basis": "actual fill prices * quantity * point value; excludes commissions",
        "excursion_basis": "reported trades.csv values, else overlapping canonical 5-minute OHLCV bars",
    }
    return rows, metadata


def build_dataset(
    *,
    runs_root: Path,
    snapshot_root: Path,
    output_dir: Path,
    point_value: float = 50.0,
) -> dict[str, Any]:
    runs = enumerate_mff_runs(runs_root)
    snapshot_files = discover_snapshot_files(snapshot_root)
    bars, bar_metadata = canonicalize_snapshots(snapshot_files)
    fills, fill_metadata = inventory_actual_fills(runs, bars, point_value=point_value)

    bars_data = _csv_bytes(bars, BAR_FIELDS)
    fills_data = _csv_bytes(fills, FILL_FIELDS)
    runs_payload = {
        "schema_version": 1,
        "runs_root": str(runs_root.resolve()),
        "mff_run_count": len(runs),
        "runs": runs,
    }
    runs_data = _json_bytes(runs_payload)
    hashes = {
        "canonical_bars_sha256": _sha256_bytes(bars_data),
        "actual_fill_baseline_sha256": _sha256_bytes(fills_data),
        "mff_run_inventory_sha256": _sha256_bytes(runs_data),
    }
    dataset_id = _sha256_bytes(_json_bytes(hashes))[:20]
    paths = {
        "canonical_bars": output_dir / f"canonical_mff_ohlcv_{hashes['canonical_bars_sha256'][:16]}.csv",
        "actual_fill_baseline": output_dir / f"mff_actual_fill_baseline_{hashes['actual_fill_baseline_sha256'][:16]}.csv",
        "mff_run_inventory": output_dir / f"mff_run_inventory_{hashes['mff_run_inventory_sha256'][:16]}.json",
        "manifest": output_dir / f"mff_phase2_replay_dataset_{dataset_id}.json",
    }
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "read_only_inputs": True,
        "inputs": {
            "runs_root": str(runs_root.resolve()),
            "snapshot_root": str(snapshot_root.resolve()),
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": hashes.get(f"{name}_sha256")}
            for name, path in paths.items()
            if name != "manifest"
        },
        "run_inventory": {"mff_run_count": len(runs)},
        "bar_inventory": bar_metadata,
        "fill_inventory": fill_metadata,
        "hashes": hashes,
    }
    manifest_data = _json_bytes(manifest)
    _write_immutable(paths["canonical_bars"], bars_data)
    _write_immutable(paths["actual_fill_baseline"], fills_data)
    _write_immutable(paths["mff_run_inventory"], runs_data)
    _write_immutable(paths["manifest"], manifest_data)
    return {**manifest, "manifest_path": str(paths["manifest"].resolve())}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a content-addressed, read-only MFF-era Phase-2 replay dataset."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-value", type=float, default=50.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_dataset(
        runs_root=args.runs_root,
        snapshot_root=args.snapshot_root,
        output_dir=args.output_dir,
        point_value=args.point_value,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
