from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


OHLCV = ["Open", "High", "Low", "Close", "Volume"]
ES_PRICE_MIN = 0.0
ES_PRICE_MAX = 25_000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ohlcv(path: Path, *, source_rank: int, contract: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Datetime", *OHLCV}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = frame[["Datetime", *OHLCV]].copy()
    frame["_ts"] = pd.to_datetime(frame["Datetime"], errors="coerce", utc=True)
    for column in OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["_ts", *OHLCV])
    valid_ohlc = (
        frame[["Open", "High", "Low", "Close"]]
        .ge(ES_PRICE_MIN)
        .all(axis=1)
        & frame[["Open", "High", "Low", "Close"]].le(ES_PRICE_MAX).all(axis=1)
        & (frame["High"] >= frame[["Open", "Close"]].max(axis=1))
        & (frame["Low"] <= frame[["Open", "Close"]].min(axis=1))
        & (frame["High"] >= frame["Low"])
        & (frame["Volume"] >= 0)
    )
    invalid_es_rows = int((~valid_ohlc).sum())
    frame = frame.loc[valid_ohlc].copy()
    frame.attrs["invalid_es_rows_dropped"] = invalid_es_rows
    frame["_source_rank"] = int(source_rank)
    frame["_source_path"] = str(path.resolve())
    frame["_contract"] = contract
    return frame


def _contract_from_name(path: Path) -> str | None:
    name = path.name.upper()
    if re.search(r"(?:JUN26|06-26)", name):
        return "ES 06-26"
    if re.search(r"(?:SEP26|09-26)", name):
        return "ES 09-26"
    return None


def build_dataset(
    *,
    historical_csv: Path,
    extension_csvs: Iterable[Path],
    exporter_dir: Path,
    rollover_utc: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = _read_ohlcv(historical_csv, source_rank=0, contract="HISTORICAL")
    base_max = base["_ts"].max()

    extension_frames: list[pd.DataFrame] = []
    extension_sources: list[dict[str, Any]] = []
    extension_max = base_max
    for rank, extension_csv in enumerate(extension_csvs, start=1):
        extension = _read_ohlcv(extension_csv, source_rank=rank, contract="CONTINUOUS")
        if extension.empty:
            continue
        extension_frames.append(extension)
        extension_max = max(extension_max, extension["_ts"].max())
        extension_sources.append(
            {
                "path": str(extension_csv.resolve()),
                "sha256": _sha256(extension_csv),
                "rows_used_before_dedup": int(len(extension)),
                "invalid_es_rows_dropped": int(extension.attrs.get("invalid_es_rows_dropped", 0)),
                "last_timestamp_utc": extension["_ts"].max().isoformat(),
            }
        )

    exporter_frames: list[pd.DataFrame] = []
    exporter_sources: list[dict[str, Any]] = []
    exporter_skipped: list[dict[str, str]] = []
    exporter_rank_start = 1 + len(extension_frames)
    for rank, path in enumerate(
        sorted(exporter_dir.glob("ES_*5m_*.csv")), start=exporter_rank_start
    ):
        contract = _contract_from_name(path)
        if contract is None:
            continue
        try:
            frame = _read_ohlcv(path, source_rank=rank, contract=contract)
        except Exception as exc:
            exporter_skipped.append({"path": str(path.resolve()), "reason": str(exc)})
            continue
        frame = frame.loc[frame["_ts"] > extension_max].copy()
        if contract == "ES 06-26":
            frame = frame.loc[frame["_ts"] < rollover_utc]
        else:
            frame = frame.loc[frame["_ts"] >= rollover_utc]
        if frame.empty:
            continue
        exporter_frames.append(frame)
        exporter_sources.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "rows_used_before_dedup": int(len(frame)),
                "invalid_es_rows_dropped": int(frame.attrs.get("invalid_es_rows_dropped", 0)),
                "contract": contract,
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )

    pieces = [base, *extension_frames, *exporter_frames]
    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.sort_values(["_ts", "_source_rank"], kind="stable")
    combined = combined.drop_duplicates(subset=["_ts"], keep="last")
    combined = combined.sort_values("_ts", kind="stable").reset_index(drop=True)
    if combined["_ts"].duplicated().any():
        raise ValueError("Canonical dataset still contains duplicate timestamps")
    if not combined["_ts"].is_monotonic_increasing:
        raise ValueError("Canonical dataset timestamps are not strictly increasing")

    output = combined[["_ts", *OHLCV]].rename(columns={"_ts": "Datetime"})
    output["Datetime"] = output["Datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    provenance = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "construction": {
            "historical_base_then_newer_extension_then_exporters": True,
            "duplicate_policy": "latest_source_rank_wins",
            "rollover_utc": rollover_utc.isoformat(),
            "june_contract_before_rollover": True,
            "september_contract_at_or_after_rollover": True,
        },
        "sources": {
            "historical": {
                "path": str(historical_csv.resolve()),
                "sha256": _sha256(historical_csv),
                "rows_used": int(len(base)),
                "last_timestamp_utc": base_max.isoformat(),
            },
            "extensions": extension_sources,
            "exporters": exporter_sources,
            "exporters_skipped": exporter_skipped,
        },
        "output": {
            "rows": int(len(output)),
            "first_timestamp_utc": str(output["Datetime"].iloc[0]),
            "last_timestamp_utc": str(output["Datetime"].iloc[-1]),
            "duplicate_timestamps": 0,
        },
    }
    return output, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an immutable Phase2 ES training dataset.")
    parser.add_argument("--historical-csv", type=Path, required=True)
    parser.add_argument("--extension-csv", type=Path, action="append", required=True)
    parser.add_argument("--exporter-dir", type=Path, required=True)
    parser.add_argument("--rollover-utc", default="2026-06-12T00:00:00Z")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-provenance", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rollover = pd.Timestamp(args.rollover_utc)
    if rollover.tzinfo is None:
        rollover = rollover.tz_localize("UTC")
    else:
        rollover = rollover.tz_convert("UTC")
    frame, provenance = build_dataset(
        historical_csv=args.historical_csv,
        extension_csvs=args.extension_csv,
        exporter_dir=args.exporter_dir,
        rollover_utc=rollover,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    provenance["output"]["path"] = str(args.out_csv.resolve())
    provenance["output"]["sha256"] = _sha256(args.out_csv)
    provenance_path = args.out_provenance or args.out_csv.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance["output"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
