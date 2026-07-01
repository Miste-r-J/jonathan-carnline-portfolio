from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from na.market.sessions import SessionDefinition, ensure_dataframe_index
from na.bot.config import HIGH_COL, LOW_COL, OPEN_COL, CLOSE_COL

TOLERANCE = 1e-9


def _load_frame(path: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Datetime" not in df.columns:
        raise KeyError("Input CSV must contain a Datetime column.")
    df = df.dropna(subset=["Datetime"])
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df = df.set_index("Datetime")
    df = ensure_dataframe_index(df, tz, naive_is_utc=False)
    return df


def _fmt_ts(ts: pd.Timestamp) -> str:
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    return str(ts)


@dataclass
class CheckResult:
    name: str
    status: str
    violations: List[Dict[str, object]]
    total_samples: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "total_samples": int(self.total_samples),
            "violations": self.violations,
        }


def _collect_violation(buffer: List[Dict[str, object]], limit: int, entry: Dict[str, object]) -> None:
    if len(buffer) < limit:
        buffer.append(entry)


def _check_orb_monotonicity(
    frame: pd.DataFrame,
    session_def: SessionDefinition,
    session_idx: pd.Series,
    orb_minutes: int,
) -> CheckResult:
    orb_high_col = f"orb{orb_minutes}_high"
    orb_low_col = f"orb{orb_minutes}_low"
    if orb_high_col not in frame.columns or orb_low_col not in frame.columns:
        return CheckResult(
            name="orb_window",
            status="skip",
            violations=[{"reason": f"Missing {orb_high_col} / {orb_low_col}"}],
        )
    idx_local = session_def.align_index(frame.index)
    delta_from_open = session_def.minutes_since_open(idx_local, session_index=session_idx)
    orb_mask = (delta_from_open >= 0) & (delta_from_open < orb_minutes)
    violations: List[Dict[str, object]] = []
    samples = 0
    for session_value, group_idx in session_idx.groupby(session_idx):
        group_rows = frame.loc[group_idx.index]
        group_mask = orb_mask.loc[group_idx.index]
        if not group_mask.any():
            continue
        samples += int(group_mask.sum())
        highs = group_rows.loc[group_mask, HIGH_COL]
        lows = group_rows.loc[group_mask, LOW_COL]
        expected_high = highs.cummax()
        expected_low = lows.cummin()
        recorded_high = group_rows.loc[group_mask, orb_high_col]
        recorded_low = group_rows.loc[group_mask, orb_low_col]
        diff_high = (recorded_high - expected_high).abs()
        diff_low = (recorded_low - expected_low).abs()
        bad_high = diff_high > TOLERANCE
        bad_low = diff_low > TOLERANCE
        if bad_high.any():
            bad_idx = bad_high.idxmax()
            _collect_violation(
                violations,
                20,
                {
                    "session": _fmt_ts(session_value),
                    "feature": orb_high_col,
                    "timestamp": _fmt_ts(bad_idx),
                    "expected": float(expected_high.loc[bad_idx]),
                    "observed": float(recorded_high.loc[bad_idx]),
                },
            )
        if bad_low.any():
            bad_idx = bad_low.idxmax()
            _collect_violation(
                violations,
                20,
                {
                    "session": _fmt_ts(session_value),
                    "feature": orb_low_col,
                    "timestamp": _fmt_ts(bad_idx),
                    "expected": float(expected_low.loc[bad_idx]),
                    "observed": float(recorded_low.loc[bad_idx]),
                },
            )
    status = "pass" if not violations else "fail"
    return CheckResult(
        name="orb_window",
        status=status,
        violations=violations,
        total_samples=samples,
    )


def _check_orb_freeze(
    frame: pd.DataFrame,
    session_def: SessionDefinition,
    session_idx: pd.Series,
    orb_minutes: int,
) -> CheckResult:
    orb_high_col = f"orb{orb_minutes}_high"
    orb_low_col = f"orb{orb_minutes}_low"
    if orb_high_col not in frame.columns or orb_low_col not in frame.columns:
        return CheckResult(
            name="orb_freeze",
            status="skip",
            violations=[{"reason": f"Missing {orb_high_col} / {orb_low_col}"}],
        )
    idx_local = session_def.align_index(frame.index)
    delta_from_open = session_def.minutes_since_open(idx_local, session_index=session_idx)
    freeze_mask = delta_from_open >= orb_minutes
    violations: List[Dict[str, object]] = []
    samples = int(freeze_mask.sum())
    grouped = frame[[orb_high_col, orb_low_col]].copy()
    grouped["session"] = session_idx
    grouped["freeze_mask"] = freeze_mask
    for session_value, sub in grouped.groupby("session"):
        post_rows = sub.loc[sub["freeze_mask"]]
        if len(post_rows) <= 1:
            continue
        base_high = post_rows[orb_high_col].iloc[0]
        base_low = post_rows[orb_low_col].iloc[0]
        high_diff = (post_rows[orb_high_col] - base_high).abs()
        low_diff = (post_rows[orb_low_col] - base_low).abs()
        high_violation = post_rows.loc[high_diff > TOLERANCE]
        low_violation = post_rows.loc[low_diff > TOLERANCE]
        if not high_violation.empty:
            idx = high_violation.index[0]
            _collect_violation(
                violations,
                20,
                {
                    "session": _fmt_ts(session_value),
                    "feature": orb_high_col,
                    "timestamp": _fmt_ts(idx),
                    "base_value": float(base_high),
                    "observed": float(high_violation[orb_high_col].iloc[0]),
                },
            )
        if not low_violation.empty:
            idx = low_violation.index[0]
            _collect_violation(
                violations,
                20,
                {
                    "session": _fmt_ts(session_value),
                    "feature": orb_low_col,
                    "timestamp": _fmt_ts(idx),
                    "base_value": float(base_low),
                    "observed": float(low_violation[orb_low_col].iloc[0]),
                },
            )
    status = "pass" if not violations else "fail"
    return CheckResult(
        name="orb_freeze",
        status=status,
        violations=violations,
        total_samples=samples,
    )


def _check_overnight_levels(
    frame: pd.DataFrame,
    session_def: SessionDefinition,
    session_idx: pd.Series,
) -> CheckResult:
    missing_cols = [col for col in ("overnight_high", "overnight_low") if col not in frame.columns]
    if missing_cols:
        return CheckResult(
            name="overnight_levels",
            status="skip",
            violations=[{"reason": f"Missing columns: {', '.join(missing_cols)}"}],
        )
    idx_local = session_def.align_index(frame.index)
    overnight_mask = session_def.in_overnight_mask(idx_local)
    violations: List[Dict[str, object]] = []
    samples = 0
    for session_value, group in frame.groupby(session_idx):
        mask = overnight_mask.loc[group.index]
        overnight_rows = group.loc[mask]
        if overnight_rows.empty:
            continue
        samples += int(len(overnight_rows))
        high_expected = float(overnight_rows[HIGH_COL].max())
        low_expected = float(overnight_rows[LOW_COL].min())
        recorded = group.loc[~mask, ["overnight_high", "overnight_low"]]
        if recorded.empty:
            continue
        high_diff = (recorded["overnight_high"] - high_expected).abs()
        low_diff = (recorded["overnight_low"] - low_expected).abs()
        high_bad = recorded.loc[high_diff > TOLERANCE]
        low_bad = recorded.loc[low_diff > TOLERANCE]
        if not high_bad.empty:
            idx = high_bad.index[0]
            _collect_violation(
                violations,
                20,
                {
                    "session": _fmt_ts(session_value),
                    "feature": "overnight_high",
                    "timestamp": _fmt_ts(idx),
                    "expected": high_expected,
                    "observed": float(high_bad["overnight_high"].iloc[0]),
                },
            )
        if not low_bad.empty:
            idx = low_bad.index[0]
            _collect_violation(
                violations,
                20,
                {
                    "session": _fmt_ts(session_value),
                    "feature": "overnight_low",
                    "timestamp": _fmt_ts(idx),
                    "expected": low_expected,
                    "observed": float(low_bad["overnight_low"].iloc[0]),
                },
            )
    status = "pass" if not violations else "fail"
    return CheckResult(
        name="overnight_levels",
        status=status,
        violations=violations,
        total_samples=samples,
    )


def _check_overnight_gap(frame: pd.DataFrame, session_idx: pd.Series) -> CheckResult:
    if "overnight_gap" not in frame.columns:
        return CheckResult(
            name="overnight_gap",
            status="skip",
            violations=[{"reason": "Missing overnight_gap column"}],
        )
    session_last = frame.groupby(session_idx)[CLOSE_COL].last()
    prev_close_map = session_last.shift(1)
    prev_close = session_idx.map(prev_close_map)
    today_open = frame.groupby(session_idx)[OPEN_COL].transform("first")
    expected_gap = today_open / (prev_close + 1e-12) - 1.0
    recorded = frame["overnight_gap"]
    diff = (recorded - expected_gap).abs()
    mask = diff > 1e-6
    violations: List[Dict[str, object]] = []
    if mask.any():
        idx = mask[mask].index[0]
        _collect_violation(
            violations,
            20,
            {
                "timestamp": _fmt_ts(idx),
                "expected": float(expected_gap.loc[idx]),
                "observed": float(recorded.loc[idx]),
            },
        )
    status = "pass" if not violations else "fail"
    return CheckResult(
        name="overnight_gap",
        status=status,
        violations=violations,
        total_samples=int(len(frame)),
    )


def _check_pivots(frame: pd.DataFrame) -> CheckResult:
    pivot_cols = [col for col in frame.columns if col.startswith("pivot_high_") or col.startswith("pivot_low_")]
    if not pivot_cols:
        return CheckResult(
            name="pivot_confirmation",
            status="skip",
            violations=[{"reason": "No pivot columns found"}],
        )
    violations: List[Dict[str, object]] = []
    samples = 0
    for col in pivot_cols:
        try:
            n = int(col.split("_")[-1])
        except ValueError:
            continue
        span = 2 * n + 1
        is_high = "high" in col
        series = frame[col].dropna()
        for idx, value in series.items():
            pos = frame.index.get_indexer([idx])[0]
            start = pos - n
            end = pos + n
            if start < 0 or end >= len(frame):
                continue
            window = frame.iloc[start : end + 1]
            samples += 1
            ref = window[HIGH_COL] if is_high else window[LOW_COL]
            comparator = ref.max() if is_high else ref.min()
            if abs(float(value) - float(comparator)) > 1e-6:
                _collect_violation(
                    violations,
                    20,
                    {
                        "feature": col,
                        "timestamp": _fmt_ts(idx),
                        "expected": float(comparator),
                        "observed": float(value),
                    },
                )
    status = "pass" if not violations else "fail"
    return CheckResult(
        name="pivot_confirmation",
        status=status,
        violations=violations,
        total_samples=samples,
    )


def run_audit(
    frame: pd.DataFrame,
    session_def: SessionDefinition,
    orb_minutes: int,
) -> Dict[str, object]:
    idx_local = session_def.align_index(frame.index)
    session_idx = session_def.session_index(idx_local)
    checks = [
        _check_orb_monotonicity(frame, session_def, session_idx, orb_minutes),
        _check_orb_freeze(frame, session_def, session_idx, orb_minutes),
        _check_overnight_levels(frame, session_def, session_idx),
        _check_overnight_gap(frame, session_idx),
        _check_pivots(frame),
    ]
    overall = "pass"
    for check in checks:
        if check.status == "fail":
            overall = "fail"
            break
    return {
        "status": overall,
        "rows": int(len(frame)),
        "session": session_def.metadata(),
        "checks": [check.to_dict() for check in checks],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate feature time-causality / session invariants.")
    parser.add_argument("--data", required=True, help="CSV file containing computed features.")
    parser.add_argument("--out", required=True, help="Directory for audit report.")
    parser.add_argument("--session-tz", default="America/Denver")
    parser.add_argument("--rth-start", default="07:30")
    parser.add_argument("--rth-end", default="14:00")
    parser.add_argument("--orb-minutes", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    src = Path(args.data).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(src, args.session_tz)
    session_def = SessionDefinition.from_strings(
        tz=args.session_tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_minutes=int(args.orb_minutes),
    )
    report = run_audit(frame, session_def, int(args.orb_minutes))
    path = out_dir / "feature_causality_audit.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"Wrote audit report to {path}")


if __name__ == "__main__":
    main()
