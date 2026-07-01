from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ScanResult:
    path: Path
    schema_version: str
    row_count: int
    ok: bool
    reason: Optional[str]
    instrument: Optional[str]


def detect_schema(df: pd.DataFrame) -> str:
    cols = {str(c).strip().lower() for c in df.columns}
    if {"instrument", "intervalmin", "sourcetz", "sessionid"}.issubset(cols):
        return "v2"
    return "v1"


def validate_df(df: pd.DataFrame, expected_instrument: str, allow_v1: bool) -> tuple[bool, Optional[str], Optional[str]]:
    if df.empty:
        return False, "empty_csv", None
    schema = detect_schema(df)
    if schema == "v1" and not allow_v1:
        return False, "v1_fallback_disabled", None
    if "Datetime" not in df.columns:
        return False, "missing_datetime", None
    dt = pd.to_datetime(df["Datetime"], errors="coerce")
    if dt.isna().any():
        return False, "parse_fail_datetime", None
    if bool((dt.diff().dt.total_seconds() <= 0).fillna(False).any()):
        return False, "non_monotonic_datetime", None
    instrument = None
    if "Instrument" in df.columns:
        observed = df["Instrument"].astype(str).str.upper().str.strip()
        instrument = str(observed.mode().iloc[0]) if not observed.empty else None
        mismatch = (observed != expected_instrument.upper()).sum()
        if int(mismatch) > 0:
            return False, "symbol_mismatch", instrument
    return True, None, instrument


def scan_csv(path: Path, expected_instrument: str, allow_v1: bool) -> ScanResult:
    try:
        df = pd.read_csv(path)
    except Exception:
        return ScanResult(path=path, schema_version="unknown", row_count=0, ok=False, reason="read_error", instrument=None)
    schema = detect_schema(df)
    ok, reason, instrument = validate_df(df, expected_instrument, allow_v1)
    return ScanResult(path=path, schema_version=schema, row_count=int(len(df)), ok=ok, reason=reason, instrument=instrument)


def main() -> int:
    ap = argparse.ArgumentParser(description="Quarantine mixed/bad CSV files and rebuild clean dataset manifest.")
    ap.add_argument("--input_dir", required=True, help="Directory containing source CSV files.")
    ap.add_argument("--output_dir", required=True, help="Directory for clean outputs and manifests.")
    ap.add_argument("--instrument", default="ES", help="Expected instrument symbol (default: ES).")
    ap.add_argument("--allow_v1_fallback", action="store_true", default=True)
    ap.add_argument("--no_allow_v1_fallback", dest="allow_v1_fallback", action="store_false")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_dir = output_dir / "quarantine" / ts
    clean_dir = output_dir / "clean" / ts
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.csv"))
    records: list[dict] = []
    rebuild_id = f"rebuild_{ts}"
    for f in files:
        result = scan_csv(f, str(args.instrument), bool(args.allow_v1_fallback))
        rec = {
            "file": str(f),
            "schema_version": result.schema_version,
            "row_count": result.row_count,
            "ok": result.ok,
            "reason": result.reason,
            "instrument": result.instrument,
        }
        if result.ok:
            dest = clean_dir / f.name
            shutil.copy2(f, dest)
            rec["action"] = "kept"
            rec["clean_path"] = str(dest)
            rec["provenance"] = {
                "schema_version": result.schema_version,
                "instrument": result.instrument or str(args.instrument).upper(),
                "rebuild_id": rebuild_id,
            }
        else:
            dest = quarantine_dir / f.name
            shutil.move(str(f), str(dest))
            rec["action"] = "quarantined"
            rec["quarantine_path"] = str(dest)
        records.append(rec)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "instrument": str(args.instrument).upper(),
        "allow_v1_fallback": bool(args.allow_v1_fallback),
        "rebuild_id": rebuild_id,
        "files_total": len(records),
        "files_kept": sum(1 for r in records if r.get("action") == "kept"),
        "files_quarantined": sum(1 for r in records if r.get("action") == "quarantined"),
        "records": records,
    }
    manifest_path = output_dir / f"quarantine_manifest_{ts}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
