from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _normalize_side(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper()
    if text in {"LONG", "BUY", "1"}:
        return "LONG"
    if text in {"SHORT", "SELL", "-1"}:
        return "SHORT"
    return None


def _extract_side_counts(df: pd.DataFrame) -> Dict[str, int]:
    for col in ("side", "Side", "action", "Action", "direction", "Direction"):
        if col in df.columns:
            counts = {"LONG": 0, "SHORT": 0}
            for value in df[col]:
                side = _normalize_side(value)
                if side:
                    counts[side] += 1
            if counts["LONG"] or counts["SHORT"]:
                return counts
    raise ValueError("Could not infer LONG/SHORT side column from trades file.")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _count_shorts_in_uptrend(path: Path, threshold: float) -> int:
    count = 0
    for payload in _iter_jsonl(path):
        side = _normalize_side(payload.get("side"))
        trend_score = payload.get("trend_score")
        try:
            trend_score = float(trend_score)
        except Exception:
            continue
        if side == "SHORT" and trend_score > threshold:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Report LONG/SHORT distribution and counter-trend SHORT counts.")
    parser.add_argument("--path", required=True, help="Path to trades.csv")
    parser.add_argument("--gating-events", default=None, help="Optional path to gating_events.jsonl")
    parser.add_argument("--trend-threshold", type=float, default=0.55, help="Trend score threshold for blocked SHORTs.")
    args = parser.parse_args()

    trades_path = Path(args.path).expanduser().resolve()
    df = pd.read_csv(trades_path)
    counts = _extract_side_counts(df)
    total = counts["LONG"] + counts["SHORT"]
    long_pct = (counts["LONG"] / total * 100.0) if total else 0.0
    short_pct = (counts["SHORT"] / total * 100.0) if total else 0.0

    print(f"trades_path={trades_path}")
    print(f"total_trades={total}")
    print(f"long_count={counts['LONG']} long_pct={long_pct:.2f}")
    print(f"short_count={counts['SHORT']} short_pct={short_pct:.2f}")

    if args.gating_events:
        gating_path = Path(args.gating_events).expanduser().resolve()
        shorts_in_uptrend = _count_shorts_in_uptrend(gating_path, args.trend_threshold)
        print(f"gating_events_path={gating_path}")
        print(f"shorts_with_trend_score_gt_{args.trend_threshold:.2f}={shorts_in_uptrend}")


if __name__ == "__main__":
    main()
