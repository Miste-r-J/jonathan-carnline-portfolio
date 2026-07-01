"""
Simple data QA helper for OHLCV CSV inputs.

Checks:
    * Timestamp monotonicity and duplicates
    * Gap stats between consecutive bars
    * Missing-value counts per column
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _status(flag: bool, warn: bool = False) -> str:
    if flag:
        return "[PASS]"
    if warn:
        return "[WARN]"
    return "[FAIL]"


def run_report(csv: Path, expected_bar_minutes: int = 5) -> List[str]:
    df = pd.read_csv(csv)
    if "Datetime" not in df.columns:
        raise ValueError("CSV must include a 'Datetime' column.")

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=False)
    df = df.sort_values("Datetime").reset_index(drop=True)
    ts = df["Datetime"]

    lines: List[str] = []
    lines.append(f"Data QA Report :: {csv}")
    lines.append(f"Rows={len(df):,}, Columns={len(df.columns)}")
    lines.append("------------------------------------------------------------")

    monotonic = ts.is_monotonic_increasing
    lines.append(f"{_status(monotonic)} timestamps strictly increasing")

    dup_count = ts.duplicated().sum()
    lines.append(f"{_status(dup_count == 0)} duplicate timestamps: {dup_count}")

    diffs = ts.diff().dropna().dt.total_seconds() / 60.0
    large_gap_mask = diffs > (expected_bar_minutes + 1e-9)
    large_gap_count = int(large_gap_mask.sum())
    large_gap_max = float(diffs.max()) if not diffs.empty else 0.0
    lines.append(
        f"{_status(large_gap_count == 0, warn=True)} gaps > {expected_bar_minutes}m: "
        f"{large_gap_count} (max gap {large_gap_max:.1f}m)"
    )
    if large_gap_count:
        largest = diffs[large_gap_mask].sort_values(ascending=False).head(5)
        lines.append("Top gap samples (minutes): " + ", ".join(f"{v:.1f}" for v in largest))

    miss = df.isna().mean()
    max_missing = miss.max()
    miss_lines = [
        f"{col}: {frac * 100:.2f}% ({int(df[col].isna().sum())})"
        for col, frac in miss.items()
    ]
    miss_ok = bool(max_missing == 0.0)
    lines.append(f"{_status(miss_ok, warn=True)} missing values per column")
    lines.extend("    " + m for m in miss_lines)

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA stats for ES CSV inputs.")
    parser.add_argument("csv", type=Path, help="Path to OHLCV CSV with Datetime column")
    parser.add_argument(
        "--bar-minutes",
        type=int,
        default=5,
        help="Expected minutes per bar when flagging large gaps.",
    )
    args = parser.parse_args()

    for line in run_report(args.csv, args.bar_minutes):
        print(line)


if __name__ == "__main__":
    main()
