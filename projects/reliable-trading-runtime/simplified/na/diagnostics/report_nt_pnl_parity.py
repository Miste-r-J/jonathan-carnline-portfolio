from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _to_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:
        return None
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare NT authoritative unrealized vs computed fallback parity")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--warn-delta-usd", type=float, default=25.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    nt_bridge = run_dir / "nt_bridge.jsonl"
    rows = _read_jsonl(nt_bridge)

    total = 0
    authoritative = 0
    fallback = 0
    deltas = []

    for row in rows:
        t = str(row.get("type") or row.get("event_type") or "").upper()
        if t not in {"PNL_SNAPSHOT", "POSITION_SNAPSHOT", "POSITION_UPDATE"}:
            continue
        total += 1
        src = str(row.get("source") or row.get("pnl_source") or "").lower()
        unreal_auth = _to_float(row.get("unrealized_pnl_currency"))
        if unreal_auth is None:
            unreal_auth = _to_float(row.get("unrealized_pnl"))
        qty = _to_float(row.get("position_qty") if row.get("position_qty") is not None else row.get("qty"))
        avg = _to_float(row.get("avg_price") if row.get("avg_price") is not None else row.get("average_price"))
        last_px = _to_float(row.get("last_price") if row.get("last_price") is not None else row.get("mark_price"))
        point_value = _to_float(row.get("point_value")) or 50.0

        if src == "nt_account_api" and unreal_auth is not None:
            authoritative += 1
        else:
            fallback += 1

        if unreal_auth is not None and qty is not None and avg is not None and last_px is not None:
            computed = qty * (last_px - avg) * point_value
            deltas.append(abs(unreal_auth - computed))

    max_delta = max(deltas) if deltas else 0.0
    avg_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
    warn_hits = sum(1 for d in deltas if d >= float(args.warn_delta_usd))

    report = {
        "run_dir": str(run_dir),
        "total_samples": total,
        "authoritative_samples": authoritative,
        "fallback_samples": fallback,
        "delta_samples": len(deltas),
        "avg_abs_delta_usd": round(avg_delta, 4),
        "max_abs_delta_usd": round(max_delta, 4),
        "warn_delta_usd": float(args.warn_delta_usd),
        "warn_delta_hits": int(warn_hits),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
