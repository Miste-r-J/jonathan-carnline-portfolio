"""
Leak-safe ES setup detection (ORB, VWAP, structure) for sniper training.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np
import pandas as pd

from .config import CLOSE_COL, HIGH_COL, LOW_COL, OPEN_COL


@dataclass
class SetupParams:
    orb_minutes: int = 15
    confirm_ticks: int = 2
    vwap_slope_min: float = 0.0
    retest_lookahead_bars: int = 6
    retest_max_dist_ticks: int = 2
    hod_lookback_bars: int = 30
    vwap_reclaim_body_min_atr: float = 0.25
    near_hod_ticks: int = 4
    near_lod_ticks: int = 4
    compression_window: int = 12
    compression_norm_window: int = 48


SETUP_COLUMNS = [
    "setup_orb_breakout_long",
    "setup_orb_breakout_short",
    "setup_vwap_reclaim_long",
    "setup_vwap_reject_short",
    "setup_hod_retest_long",
    "setup_lod_retest_short",
]


def _rolling_pivot(series: pd.Series, window: int, *, is_high: bool) -> pd.Series:
    span = window * 2 + 1
    seg = series.rolling(span, center=True)
    extreme = seg.max() if is_high else seg.min()
    cond = series.eq(extreme)
    return series.where(cond)


def _retest_flag(
    series_break: pd.Series,
    baseline: pd.Series,
    close: pd.Series,
    *,
    side: str,
    params: SetupParams,
    tick_size: float,
) -> pd.Series:
    pending = -1
    origin_level = np.nan
    max_dist = params.retest_max_dist_ticks * tick_size
    flags = np.zeros(len(series_break), dtype=int)
    for idx in range(len(series_break)):
        if bool(series_break.iloc[idx]):
            pending = params.retest_lookahead_bars
            origin_level = baseline.iloc[idx]
            continue
        if pending >= 0:
            if side == "LONG":
                if close.iloc[idx] <= origin_level:
                    dist = origin_level - close.iloc[idx]
                else:
                    dist = np.inf
            else:
                if close.iloc[idx] >= origin_level:
                    dist = close.iloc[idx] - origin_level
                else:
                    dist = np.inf
            if dist <= max_dist:
                flags[idx] = 1
                pending = -1
            else:
                pending -= 1
        else:
            pending = -1
    return pd.Series(flags, index=series_break.index)


def compute_es_setups(
    frame: pd.DataFrame,
    *,
    tick_size: float,
    params: SetupParams,
) -> pd.DataFrame:
    """Return DataFrame with setup flags + structure metrics aligned to frame."""

    out = pd.DataFrame(index=frame.index)
    close = frame[CLOSE_COL]
    high = frame[HIGH_COL]
    low = frame[LOW_COL]
    orb_high = frame[f"orb{params.orb_minutes}_high"]
    orb_low = frame[f"orb{params.orb_minutes}_low"]
    vwap = frame.get("vwap_sess")
    atr = frame.get("atr_14", pd.Series(0.0, index=frame.index)).copy()
    atr = atr.ffill().bfill()
    atr = atr.replace(0.0, np.nan).fillna(0.25)

    # Distance features relative to ORB boundaries
    # REDUNDANT FEATURE — mathematically derived from `dist_to_orb_high` / `dist_to_orb_low`.
    # Kept to satisfy frozen model schema (retrain_v2_full).
    # Listed in NEXT_RETRAIN_REMOVALS for exclusion in next training run.
    out["dist_orb_high_ticks"] = (close - orb_high) / tick_size
    out["dist_orb_low_ticks"] = (orb_low - close) / tick_size
    # NOTE (retrain_v2_full schema): the pipeline also emits dist_to_orb_high_atr/dist_to_orb_low_atr
    # via na.bot.orb_aoi_features.compute_aoi_features(). Keep both columns for the frozen model,
    # but avoid recomputing the same value twice.
    #
    # dist_orb_high_atr is an exact duplicate of dist_to_orb_high_atr.
    dist_to_orb_high_atr = frame.get("dist_to_orb_high_atr")
    if dist_to_orb_high_atr is not None:
        out["dist_orb_high_atr"] = pd.to_numeric(dist_to_orb_high_atr, errors="coerce")
    else:
        out["dist_orb_high_atr"] = (close - orb_high) / (atr + 1e-12)
    #
    # dist_orb_low_atr is the NEGATIVE of dist_to_orb_low_atr (sign-flip is intentional).
    dist_to_orb_low_atr = frame.get("dist_to_orb_low_atr")
    if dist_to_orb_low_atr is not None:
        out["dist_orb_low_atr"] = -pd.to_numeric(dist_to_orb_low_atr, errors="coerce")
    else:
        out["dist_orb_low_atr"] = (orb_low - close) / (atr + 1e-12)

    # Session extremes distances
    session_groups = frame.index.date
    hod = high.groupby(session_groups).cummax()
    lod = low.groupby(session_groups).cummin()
    out["dist_to_hod_ticks"] = (hod - close) / tick_size
    out["dist_to_lod_ticks"] = (close - lod) / tick_size
    out["is_near_hod"] = (out["dist_to_hod_ticks"].abs() <= params.near_hod_ticks).astype(int)
    out["is_near_lod"] = (out["dist_to_lod_ticks"].abs() <= params.near_lod_ticks).astype(int)

    # Pivot structure
    for n in (2, 3, 5):
        out[f"pivot_high_{n}"] = _rolling_pivot(high, n, is_high=True)
        out[f"pivot_low_{n}"] = _rolling_pivot(low, n, is_high=False)
    last_pivot_high = out["pivot_high_5"].ffill()
    last_pivot_low = out["pivot_low_5"].ffill()
    out["dist_to_last_pivot_high_ticks"] = (close - last_pivot_high) / tick_size
    out["dist_to_last_pivot_low_ticks"] = (last_pivot_low - close) / tick_size

    # Break state flags
    broke_hod = close > hod.shift(1)
    broke_lod = close < lod.shift(1)
    out["broke_hod_last_n"] = broke_hod.rolling(params.hod_lookback_bars, min_periods=1).max().fillna(0).astype(int)
    out["broke_lod_last_n"] = broke_lod.rolling(params.hod_lookback_bars, min_periods=1).max().fillna(0).astype(int)

    # Compression / expansion
    rng = (high - low)
    roll_rng = rng.rolling(params.compression_window, min_periods=params.compression_window).mean()
    norm_mean = roll_rng.rolling(params.compression_norm_window, min_periods=params.compression_window).mean()
    norm_std = roll_rng.rolling(params.compression_norm_window, min_periods=params.compression_window).std()
    out["compression_z_12"] = (roll_rng - norm_mean) / (norm_std + 1e-12)

    # ORB breakout setups
    confirm = params.confirm_ticks * tick_size
    close_above_orb = close > (orb_high + confirm)
    close_below_orb = close < (orb_low - confirm)
    vwap_slope = vwap.diff(3) if vwap is not None else pd.Series(0.0, index=frame.index)
    if vwap is not None:
        above_vwap = (close > vwap).astype(bool)
        below_vwap = (close < vwap).astype(bool)
    else:
        above_vwap = pd.Series(False, index=frame.index, dtype=bool)
        below_vwap = pd.Series(False, index=frame.index, dtype=bool)

    out["setup_orb_breakout_long"] = (
        close_above_orb &
        (vwap_slope.fillna(0.0) >= params.vwap_slope_min) &
        above_vwap.fillna(False)
    ).astype(int)
    out["setup_orb_breakout_short"] = (
        close_below_orb &
        (vwap_slope.fillna(0.0) <= -params.vwap_slope_min) &
        below_vwap.fillna(False)
    ).astype(int)

    # VWAP reclaim/reject
    body = (close - frame[OPEN_COL]).abs()
    body_over_atr = body / (atr + 1e-12)
    prev_above = above_vwap.shift(1, fill_value=False)
    prev_below = below_vwap.shift(1, fill_value=False)
    out["setup_vwap_reclaim_long"] = (
        (~prev_above) &
        above_vwap &
        (body_over_atr >= params.vwap_reclaim_body_min_atr)
    ).astype(int)
    out["setup_vwap_reject_short"] = (
        (~prev_below) &
        below_vwap &
        (body_over_atr >= params.vwap_reclaim_body_min_atr)
    ).astype(int)

    out["setup_hod_retest_long"] = _retest_flag(
        broke_hod,
        hod,
        close,
        side="LONG",
        params=params,
        tick_size=tick_size,
    )
    out["setup_lod_retest_short"] = _retest_flag(
        broke_lod,
        lod,
        close,
        side="SHORT",
        params=params,
        tick_size=tick_size,
    )

    for col in SETUP_COLUMNS:
        out[col] = out[col].fillna(0).astype(int)
    # REDUNDANT FEATURE — OR of individual setup_* flags.
    # Kept to satisfy frozen model schema (retrain_v2_full).
    # Listed in NEXT_RETRAIN_REMOVALS for exclusion in next training run.
    out["setup_present"] = out[SETUP_COLUMNS].any(axis=1).astype(int)
    if os.getenv("DEBUG_ASSERTIONS"):
        setup_flag_cols = [c for c in out.columns if c.startswith("setup_") and c != "setup_present"]
        if setup_flag_cols:
            assert (out["setup_present"] == out[setup_flag_cols].any(axis=1).astype(int)).all(), (
                "setup_present is inconsistent with individual setup flags"
            )
    setup_id = pd.Series("none", index=out.index, dtype=object)
    for col in SETUP_COLUMNS:
        flag = (out[col] > 0) & (setup_id == "none")
        pretty = col.replace("setup_", "")
        setup_id = setup_id.where(~flag, other=pretty)
    out["setup_id"] = setup_id.fillna("none")

    pivot_cols = [c for c in out.columns if c.startswith("pivot_")]
    for col in pivot_cols:
        out[col] = out[col].ffill().fillna(close)
    numeric_fill_zero = [
        "dist_orb_high_ticks",
        "dist_orb_low_ticks",
        "dist_orb_high_atr",
        "dist_orb_low_atr",
        "dist_to_hod_ticks",
        "dist_to_lod_ticks",
        "dist_to_last_pivot_high_ticks",
        "dist_to_last_pivot_low_ticks",
        "compression_z_12",
        "is_near_hod",
        "is_near_lod",
        "broke_hod_last_n",
        "broke_lod_last_n",
    ]
    for col in numeric_fill_zero:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    return out
