"""
Leak-safe triple-barrier labeling tailored for ES 5m ORB strategies.

Labels answer: if I enter on the next bar's open with a fixed stop/target and
max hold, does the LONG or SHORT trade pay off first, and what is the realized
R-multiple?
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dt_time
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CLOSE_COL, HIGH_COL, LOW_COL, OPEN_COL

Direction = Literal["LONG", "SHORT", "BOTH"]
TieBreak = Literal["stop_first", "target_first"]
SessionExit = Literal["timeout_close", "forbid_overnight"]
TimeoutExit = Literal["close"]


@dataclass(frozen=True)
class TripleBarrierParams:
    tick_size: float = 0.25
    stop_ticks: int = 8
    target_ticks: int = 12
    max_hold_bars: int = 12
    tie_break: TieBreak = "stop_first"
    session_exit: SessionExit = "timeout_close"
    timeout_exit: TimeoutExit = "close"
    clip_r: Optional[float] = 5.0


def _resolve_datetime_index(df: pd.DataFrame, tz: str) -> pd.DatetimeIndex:
    if "Datetime" in df.columns:
        series = pd.to_datetime(df["Datetime"])
    elif isinstance(df.index, pd.DatetimeIndex):
        series = df.index
    else:
        raise ValueError("DataFrame must include a Datetime column or index for triple-barrier labels.")
    series = pd.DatetimeIndex(series)
    if series.tz is None:
        series = series.tz_localize("UTC").tz_convert(tz)
    else:
        series = series.tz_convert(tz)
    return pd.DatetimeIndex(series)


def _parse_time(value: str) -> dt_time:
    parts = value.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    return dt_time(hour, minute)


def _session_last_index(idx: pd.DatetimeIndex, cutoff: dt_time) -> np.ndarray:
    session_dates = idx.date
    session_times = idx.time
    last_idx: Dict[object, int] = {}
    fallback: Dict[object, int] = {}
    for i, (session_date, t) in enumerate(zip(session_dates, session_times)):
        fallback[session_date] = i
        if t <= cutoff:
            last_idx[session_date] = i
    return np.array(
        [last_idx.get(session_date, fallback[session_date]) for session_date in session_dates],
        dtype=np.int32,
    )


def _simulate_direction(
    *,
    sign: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    session_last_idx: np.ndarray,
    params: TripleBarrierParams,
) -> Dict[str, np.ndarray]:
    n = len(open_arr)
    realized = np.full(n, np.nan, dtype=np.float32)
    mfe = np.full(n, np.nan, dtype=np.float32)
    mae = np.full(n, np.nan, dtype=np.float32)
    tte = np.full(n, -1, dtype=np.int32)
    exit_kind = np.full(n, "", dtype=object)
    stop_dist = params.stop_ticks * params.tick_size
    target_dist = params.target_ticks * params.tick_size
    if stop_dist <= 0 or target_dist <= 0:
        raise ValueError("stop_ticks and target_ticks must be positive.")
    reward_r = target_dist / stop_dist
    max_hold = int(params.max_hold_bars)
    if max_hold <= 0:
        raise ValueError("max_hold_bars must be positive.")
    for event_idx in range(n - 1):
        entry_idx = event_idx + 1
        entry_px = open_arr[entry_idx]
        if not np.isfinite(entry_px):
            continue
        theoretical_end = min(n - 1, entry_idx + max_hold - 1)
        session_end_idx = session_last_idx[entry_idx]
        final_idx = min(theoretical_end, session_end_idx)
        if final_idx < entry_idx:
            # Not enough bars left in session to evaluate this trade.
            continue
        stop_px = entry_px - sign * stop_dist
        target_px = entry_px + sign * target_dist
        best_favor = 0.0
        worst_adverse = 0.0
        outcome = None
        exit_idx = final_idx
        for bar_idx in range(entry_idx, final_idx + 1):
            high = high_arr[bar_idx]
            low = low_arr[bar_idx]
            close = close_arr[bar_idx]
            if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
                continue
            if sign > 0:
                favor = (high - entry_px) / stop_dist
                adverse = (entry_px - low) / stop_dist
                stop_hit = low <= stop_px
                target_hit = high >= target_px
            else:
                favor = (entry_px - low) / stop_dist
                adverse = (high - entry_px) / stop_dist
                stop_hit = high >= stop_px
                target_hit = low <= target_px
            best_favor = max(best_favor, favor)
            worst_adverse = max(worst_adverse, adverse)
            hits = []
            if params.tie_break == "target_first":
                hits.extend(["target", "stop"])
            else:
                hits.extend(["stop", "target"])
            hit_type = None
            for contender in hits:
                if contender == "stop" and stop_hit:
                    hit_type = "stop"
                    break
                if contender == "target" and target_hit:
                    hit_type = "target"
                    break
            if hit_type:
                outcome = hit_type
                exit_idx = bar_idx
                break
        if outcome is None:
            if params.session_exit == "forbid_overnight" and session_end_idx < theoretical_end:
                # Drop ambiguous samples if session cutoff prevented resolution.
                continue
            outcome = "timeout"
        if outcome == "target":
            realized_r = reward_r
        elif outcome == "stop":
            realized_r = -1.0
        else:
            exit_px = close_arr[exit_idx] if params.timeout_exit == "close" else close_arr[exit_idx]
            realized_r = ((exit_px - entry_px) * sign) / stop_dist
        if params.clip_r is not None:
            clip = float(params.clip_r)
            realized_r = float(max(-clip, min(clip, realized_r)))
        realized[event_idx] = realized_r
        mfe[event_idx] = best_favor
        mae[event_idx] = worst_adverse
        tte[event_idx] = exit_idx - event_idx
        exit_kind[event_idx] = outcome
    return {"r": realized, "mfe": mfe, "mae": mae, "tte": tte, "exit": exit_kind}


def make_triple_barrier_labels(
    df: pd.DataFrame,
    *,
    tz: str,
    rth_start: str,
    rth_end: str,
    params: TripleBarrierParams,
    side: Direction = "BOTH",
) -> Dict[str, pd.Series]:
    """
    Build trade-realistic triple-barrier labels.

    Returns a mapping with Series aligned to `df` index:
      - y_dir: {-1, 0, +1}
      - y_r: realized R for the chosen direction (0 when FLAT)
      - y_mfe_r / y_mae_r / y_tte: diagnostics for chosen direction
      - long/short: per-direction realized R, diagnostics, exit counts
      - schema / diagnostics metadata
    """

    df = df.copy()
    needed_cols = [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]
    missing = [col for col in needed_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Triple-barrier labels require columns {missing}.")
    idx_local = _resolve_datetime_index(df, tz)
    start_time = _parse_time(rth_start)
    end_time = _parse_time(rth_end)
    session_last_idx = _session_last_index(idx_local, end_time)
    local_times = pd.Series(idx_local.time, index=df.index)
    event_in_rth = (local_times >= start_time) & (local_times < end_time)
    open_arr = pd.to_numeric(df[OPEN_COL], errors="coerce").to_numpy(dtype=float)
    high_arr = pd.to_numeric(df[HIGH_COL], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(df[LOW_COL], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(df[CLOSE_COL], errors="coerce").to_numpy(dtype=float)

    long_stats = _simulate_direction(
        sign=1,
        open_arr=open_arr,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
        session_last_idx=session_last_idx,
        params=params,
    )
    short_stats = _simulate_direction(
        sign=-1,
        open_arr=open_arr,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
        session_last_idx=session_last_idx,
        params=params,
    )

    index = df.index
    long_r = pd.Series(long_stats["r"], index=index, dtype="float32")
    short_r = pd.Series(short_stats["r"], index=index, dtype="float32")

    def _best_direction() -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        dir_vals = np.full(len(df), np.nan, dtype=float)
        r_vals = np.full(len(df), np.nan, dtype=float)
        mfe_vals = np.full(len(df), np.nan, dtype=float)
        mae_vals = np.full(len(df), np.nan, dtype=float)
        tte_vals = np.full(len(df), np.nan, dtype=float)
        for i in range(len(df)):
            if not bool(event_in_rth.iloc[i]):
                continue
            best_dir = 0
            best_r = 0.0
            best_mfe = 0.0
            best_mae = 0.0
            best_tte = np.nan
            def _consider(direction: int, stats: Dict[str, np.ndarray]):
                nonlocal best_dir, best_r, best_mfe, best_mae, best_tte
                val = stats["r"][i]
                if not np.isfinite(val):
                    return
                label_allowed = (
                    side in {"BOTH", "LONG"} and direction == 1
                    or side in {"BOTH", "SHORT"} and direction == -1
                )
                if not label_allowed:
                    return
                if val > 0 and val > best_r:
                    best_dir = direction
                    best_r = float(val)
                    best_mfe = float(stats["mfe"][i])
                    best_mae = float(stats["mae"][i])
                    best_tte = float(stats["tte"][i])
            _consider(1, long_stats)
            _consider(-1, short_stats)
            if best_dir != 0:
                dir_vals[i] = best_dir
                r_vals[i] = best_r
                mfe_vals[i] = best_mfe
                mae_vals[i] = best_mae
                tte_vals[i] = best_tte
        return (
            pd.Series(dir_vals, index=index, dtype="float64"),
            pd.Series(r_vals, index=index, dtype="float32"),
            pd.Series(mfe_vals, index=index, dtype="float32"),
            pd.Series(mae_vals, index=index, dtype="float32"),
            pd.Series(tte_vals, index=index, dtype="float32"),
        )

    y_dir, y_r, y_mfe, y_mae, y_tte = _best_direction()
    dir_series = y_dir.astype("Float64")

    schema = {
        "name": "triple_barrier_v1",
        "tick_size": float(params.tick_size),
        "stop_ticks": int(params.stop_ticks),
        "target_ticks": int(params.target_ticks),
        "max_hold_bars": int(params.max_hold_bars),
        "tie_break": params.tie_break,
        "session_exit": params.session_exit,
        "timeout_exit": params.timeout_exit,
        "entry_execution": "next_bar_open",
        "direction_mapping": {"-1": "SHORT", "0": "FLAT", "1": "LONG"},
        "target_r_multiple": float(params.target_ticks) / float(params.stop_ticks),
    }

    def _diagnostics(stats: Dict[str, np.ndarray]) -> Dict[str, float]:
        exit_series = pd.Series(stats["exit"], index=index)
        total = float((~pd.Series(stats["r"], index=index).isna()).sum() or 1)
        timeout = float((exit_series == "timeout").sum())
        target = float((exit_series == "target").sum())
        stop = float((exit_series == "stop").sum())
        return {
            "total_samples": int(total),
            "timeout_fraction": timeout / total,
            "target_fraction": target / total,
            "stop_fraction": stop / total,
        }

    diagnostics = {
        "long": _diagnostics(long_stats),
        "short": _diagnostics(short_stats),
    }

    return {
        "y_dir": dir_series,
        "y_r": y_r,
        "y_mfe_r": y_mfe,
        "y_mae_r": y_mae,
        "y_tte": y_tte,
        "long": {
            "r": long_r,
            "mfe_r": pd.Series(long_stats["mfe"], index=index, dtype="float32"),
            "mae_r": pd.Series(long_stats["mae"], index=index, dtype="float32"),
            "tte": pd.Series(long_stats["tte"], index=index, dtype="float32"),
            "exit": pd.Series(long_stats["exit"], index=index),
        },
        "short": {
            "r": short_r,
            "mfe_r": pd.Series(short_stats["mfe"], index=index, dtype="float32"),
            "mae_r": pd.Series(short_stats["mae"], index=index, dtype="float32"),
            "tte": pd.Series(short_stats["tte"], index=index, dtype="float32"),
            "exit": pd.Series(short_stats["exit"], index=index),
        },
        "schema": schema,
        "diagnostics": diagnostics,
    }


__all__ = [
    "TripleBarrierParams",
    "make_triple_barrier_labels",
]
