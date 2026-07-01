from __future__ import annotations

import logging
from datetime import time, timedelta
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .config import CLOSE_COL, HIGH_COL, LOW_COL

LOGGER = logging.getLogger(__name__)

AOI_TYPE_CODES = {
    "prev_rth_high": 0,
    "prev_rth_low": 1,
    "overnight_high": 2,
    "overnight_low": 3,
}
AOI_HIGH_TYPES = {"prev_rth_high", "overnight_high"}
AOI_LOW_TYPES = {"prev_rth_low", "overnight_low"}


def _ensure_local_index(df: pd.DataFrame, tz: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["Datetime"]))
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(tz)
    else:
        idx = idx.tz_convert(tz)
    return idx


def _parse_time(value: str, fallback: str) -> time:
    raw = str(value or fallback)
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time string: {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    return time(hour=hour, minute=minute)


def _merge_strategy_config(
    strategy_config: Mapping[str, object] | None,
    *,
    rth_start: str,
    rth_end: str,
    orb_minutes: int,
) -> dict:
    base = {
        "mode": "orb_aoi",
        "session": {"rth_start": rth_start, "rth_end": rth_end},
        "orb": {
            "enabled": True,
            "open_time": rth_start,
            "duration_minutes": orb_minutes,
            "use_for_features": True,
        },
        "aoi": {
            "enabled": True,
            "use_previous_session_high_low": True,
            "use_overnight_high_low": True,
            "manual_aoi_csv": None,
            "proximity_atr": 0.25,
            "band_ticks": 6,
        },
    }
    if not strategy_config:
        return base
    payload = dict(base)
    mode = str(strategy_config.get("mode") or payload["mode"]).lower()
    payload["mode"] = mode
    for section in ("session", "orb", "aoi"):
        target = dict(payload.get(section) or {})
        source = dict(strategy_config.get(section) or {})
        for key, value in source.items():
            target[key] = value
        payload[section] = target
    return payload


def compute_orb_levels(
    df: pd.DataFrame,
    *,
    strategy_config: Mapping[str, object],
    tz: str,
) -> pd.DataFrame:
    if not strategy_config or str(strategy_config.get("mode") or "").lower() != "orb_aoi":
        return df
    session_cfg = strategy_config.get("session") or {}
    orb_cfg = strategy_config.get("orb") or {}
    orb_duration = int(orb_cfg.get("duration_minutes") or 15)
    orb_open = _parse_time(orb_cfg.get("open_time") or session_cfg.get("rth_start") or "07:30", "07:30")
    idx_local = _ensure_local_index(df, tz)
    minutes = idx_local.hour * 60 + idx_local.minute
    orb_start = orb_open.hour * 60 + orb_open.minute
    orb_end = orb_start + orb_duration
    session_dates = pd.Series(idx_local.normalize(), index=df.index)
    within_orb = (minutes >= orb_start) & (minutes < orb_end)
    within_orb = pd.Series(within_orb, index=df.index)
    orb_high = df[HIGH_COL].where(within_orb).groupby(session_dates).cummax()
    orb_low = df[LOW_COL].where(within_orb).groupby(session_dates).cummin()
    orb_high = orb_high.groupby(session_dates).ffill()
    orb_low = orb_low.groupby(session_dates).ffill()
    fallback_high = df.get(f"orb{orb_duration}_high")
    fallback_low = df.get(f"orb{orb_duration}_low")
    if fallback_high is not None:
        orb_high = orb_high.fillna(fallback_high)
    if fallback_low is not None:
        orb_low = orb_low.fillna(fallback_low)
    df["orb_high"] = orb_high
    df["orb_low"] = orb_low
    df["orb_mid"] = (df["orb_high"] + df["orb_low"]) / 2.0
    df["orb_range"] = df["orb_high"] - df["orb_low"]
    close = pd.to_numeric(df[CLOSE_COL], errors="coerce")
    df["above_orb_flag"] = (close > df["orb_high"]).astype(int)
    df["below_orb_flag"] = (close < df["orb_low"]).astype(int)
    mins_from_open = df.get("mins_from_open")
    if mins_from_open is None:
        mins_from_open = pd.Series(minutes - orb_start, index=df.index)
    minutes_since_end = mins_from_open - orb_duration
    df["minutes_since_orb_end"] = minutes_since_end.where(minutes_since_end >= 0, -1)
    df["after_orb_flag"] = (df["minutes_since_orb_end"] >= 0).astype(int)
    df["is_orb_window"] = ((minutes >= orb_start) & (minutes < orb_end)).astype(int)
    df["price_minus_orb_high"] = close - df["orb_high"]
    df["price_minus_orb_low"] = close - df["orb_low"]
    df["is_inside_orb"] = ((close >= df["orb_low"]) & (close <= df["orb_high"])).astype(int)
    # DEDUP (retrain_v2_full): inside_orb_flag is an exact duplicate of is_inside_orb.
    # Keep both columns to satisfy the frozen model schema, but compute only once.
    df["inside_orb_flag"] = df["is_inside_orb"]
    return df


def compute_session_levels(df: pd.DataFrame, *, strategy_config: Mapping[str, object], tz: str) -> pd.DataFrame:
    mode = str(strategy_config.get("mode") or "").lower() if strategy_config else ""
    aoi_cfg = (strategy_config.get("aoi") or {}) if strategy_config else {}
    if mode != "orb_aoi" or not aoi_cfg or not bool(aoi_cfg.get("enabled", False)):
        return df
    session_cfg = strategy_config.get("session") or {}
    start_time = _parse_time(session_cfg.get("rth_start") or "07:30", "07:30")
    end_time = _parse_time(session_cfg.get("rth_end") or "12:00", "12:00")
    idx_local = _ensure_local_index(df, tz)
    minutes = idx_local.hour * 60 + idx_local.minute
    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute
    base_dates = idx_local.normalize()
    session_dates = pd.Series(base_dates, index=df.index)
    # Bars happening after the RTH end belong to the next session date.
    rollover_mask = minutes >= end_minutes
    if rollover_mask.any():
        session_dates.loc[rollover_mask] = session_dates.loc[rollover_mask] + timedelta(days=1)
    in_rth = (minutes >= start_minutes) & (minutes <= end_minutes)
    in_rth_series = pd.Series(in_rth, index=df.index)
    rth_high_daily = df[HIGH_COL].where(in_rth_series).groupby(session_dates).max()
    rth_low_daily = df[LOW_COL].where(in_rth_series).groupby(session_dates).min()
    prev_high_map = rth_high_daily.shift(1)
    prev_low_map = rth_low_daily.shift(1)
    df["prev_session_high"] = session_dates.map(prev_high_map)
    df["prev_session_low"] = session_dates.map(prev_low_map)
    overnight_mask = ~in_rth_series
    overnight_session_dates = session_dates.copy()
    overnight_high_daily = df[HIGH_COL].where(overnight_mask).groupby(overnight_session_dates).max()
    overnight_low_daily = df[LOW_COL].where(overnight_mask).groupby(overnight_session_dates).min()
    df["overnight_high"] = overnight_session_dates.map(overnight_high_daily)
    df["overnight_low"] = overnight_session_dates.map(overnight_low_daily)

    # Fallbacks when historical context is missing (e.g., RTH-only CSVs)
    close_series = pd.to_numeric(df[CLOSE_COL], errors="coerce")
    high_series = pd.to_numeric(df[HIGH_COL], errors="coerce")
    low_series = pd.to_numeric(df[LOW_COL], errors="coerce")
    default_high = high_series.ffill().fillna(close_series)
    default_low = low_series.ffill().fillna(close_series)

    def _sanitize_levels(name: str, fallback: pd.Series) -> pd.Series:
        series = pd.to_numeric(df.get(name), errors="coerce")
        if series is None:
            return fallback.copy()
        series = series.ffill()
        series = series.fillna(fallback)
        return series

    df["prev_session_high"] = _sanitize_levels("prev_session_high", default_high)
    df["prev_session_low"] = _sanitize_levels("prev_session_low", default_low)

    overnight_high_raw = df.get("overnight_high")
    overnight_low_raw = df.get("overnight_low")
    overnight_high = pd.to_numeric(overnight_high_raw, errors="coerce") if overnight_high_raw is not None else None
    overnight_low = pd.to_numeric(overnight_low_raw, errors="coerce") if overnight_low_raw is not None else None
    df["overnight_high"] = (
        overnight_high.ffill().fillna(df["prev_session_high"]) if overnight_high is not None else df["prev_session_high"]
    )
    df["overnight_low"] = (
        overnight_low.ffill().fillna(df["prev_session_low"]) if overnight_low is not None else df["prev_session_low"]
    )

    df["aoi_prev_rth_high"] = df["prev_session_high"]
    df["aoi_prev_rth_low"] = df["prev_session_low"]
    df["aoi_prev_overnight_high"] = df["overnight_high"]
    df["aoi_prev_overnight_low"] = df["overnight_low"]
    return df


def compute_aoi_features(
    df: pd.DataFrame,
    *,
    strategy_config: Mapping[str, object],
    tz: str,
    atr_column: str = "atr_14",
) -> pd.DataFrame:
    mode = str(strategy_config.get("mode") or "").lower() if strategy_config else ""
    aoi_cfg = (strategy_config.get("aoi") or {}) if strategy_config else {}
    if mode != "orb_aoi" or not aoi_cfg or not bool(aoi_cfg.get("enabled", False)):
        df["near_any_aoi_flag"] = df.get("near_any_aoi_flag", 0)
        return df
    proximity = float(aoi_cfg.get("proximity_atr") or 0.25)
    atr = pd.to_numeric(df.get(atr_column), errors="coerce")
    if atr.isna().all():
        close = pd.to_numeric(df[CLOSE_COL], errors="coerce")
        prev_close = close.shift(1)
        tr = pd.Series(np.maximum.reduce([
            (df[HIGH_COL] - df[LOW_COL]).abs(),
            (df[HIGH_COL] - prev_close).abs(),
            (df[LOW_COL] - prev_close).abs(),
        ]), index=df.index)
        atr = tr.rolling(14, min_periods=2).mean()
    level_cols: list[tuple[str, str]] = []
    if bool(aoi_cfg.get("use_previous_session_high_low", False)):
        level_cols.append(("prev_session_high", "prev_session_high"))
        level_cols.append(("prev_session_low", "prev_session_low"))
    if bool(aoi_cfg.get("use_overnight_high_low", False)):
        level_cols.append(("overnight_high", "overnight_high"))
        level_cols.append(("overnight_low", "overnight_low"))
    orb_cfg = (strategy_config.get("orb") or {}) if strategy_config else {}
    if bool(orb_cfg.get("enabled", False)):
        level_cols.append(("orb_high", "orb_high"))
        level_cols.append(("orb_low", "orb_low"))
    close = pd.to_numeric(df[CLOSE_COL], errors="coerce")
    new_cols: dict[str, pd.Series] = {}
    near_flag_names: list[str] = []
    for col_name, alias in level_cols:
        if col_name not in df:
            continue
        level = pd.to_numeric(df[col_name], errors="coerce")
        dist_pts = close - level
        dist_atr = dist_pts / (atr + 1e-12)
        col_suffix = alias.replace("prev_session_", "prev_").replace("overnight_", "overnight_")
        flag = dist_atr.abs() <= proximity
        flag_name = f"near_{col_suffix}_flag"
        new_cols[f"dist_to_{col_suffix}"] = dist_pts
        new_cols[f"dist_to_{col_suffix}_atr"] = dist_atr
        new_cols[flag_name] = flag.astype(int)
        near_flag_names.append(flag_name)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    if near_flag_names:
        df["near_any_aoi_flag"] = (df[near_flag_names].sum(axis=1) > 0).astype(int)
    else:
        df["near_any_aoi_flag"] = 0
    band_ticks = float(aoi_cfg.get("band_ticks") or 6)
    tick_size = float(aoi_cfg.get("tick_size") or 0.25)
    band_abs = abs(band_ticks) * tick_size
    aoi_levels: dict[str, pd.Series] = {
        "prev_rth_high": df.get("aoi_prev_rth_high"),
        "prev_rth_low": df.get("aoi_prev_rth_low"),
        "overnight_high": df.get("aoi_prev_overnight_high"),
        "overnight_low": df.get("aoi_prev_overnight_low"),
    }
    dist_cols = {}
    for key, level in aoi_levels.items():
        if level is None:
            continue
        dist = close - level
        dist_cols[key] = dist
    level_frame_dict = {key: series for key, series in aoi_levels.items() if series is not None}
    level_frame = pd.DataFrame(level_frame_dict) if level_frame_dict else pd.DataFrame(index=df.index)
    if dist_cols:
        dist_frame = pd.DataFrame(dist_cols)
        abs_frame = dist_frame.abs()
        valid_rows = abs_frame.notna().any(axis=1)
        nearest_labels = pd.Series("", index=dist_frame.index, dtype=object)
        if valid_rows.any():
            nearest_valid = abs_frame.loc[valid_rows].idxmin(axis=1)
            nearest_labels.loc[valid_rows] = nearest_valid
        nearest_dist = pd.Series(0.0, index=dist_frame.index, dtype=float)
        for col in dist_frame.columns:
            mask = (nearest_labels == col) & valid_rows
            if mask.any():
                nearest_dist.loc[mask] = dist_frame.loc[mask, col]
        close_fallback = close.ffill().bfill()
        nearest_price = pd.Series(close_fallback, index=dist_frame.index, dtype=float)
        for col in dist_frame.columns:
            if col not in level_frame:
                continue
            mask = (nearest_labels == col) & valid_rows
            if mask.any():
                nearest_price.loc[mask] = level_frame[col].loc[mask]
        df["nearest_aoi_type"] = nearest_labels
        df["nearest_aoi_distance"] = nearest_dist
        df["nearest_aoi_price"] = nearest_price
        df["nearest_aoi_type_code"] = [
            AOI_TYPE_CODES.get(str(val)) if val else -1 for val in df["nearest_aoi_type"]
        ]
        nearest_abs = nearest_dist.abs()
        df["at_any_aoi_band"] = ((nearest_abs <= band_abs) & valid_rows).astype(int)
        df["at_upper_aoi_band"] = (
            df["at_any_aoi_band"].astype(bool)
            & df["nearest_aoi_type"].isin(AOI_HIGH_TYPES)
            & (nearest_dist >= 0)
        ).astype(int)
        df["at_lower_aoi_band"] = (
            df["at_any_aoi_band"].astype(bool)
            & df["nearest_aoi_type"].isin(AOI_LOW_TYPES)
            & (nearest_dist <= 0)
        ).astype(int)
        abs_cols = {f"dist_abs_to_{level_name}": dist.abs().fillna(0.0) for level_name, dist in dist_cols.items()}
        if abs_cols:
            df = pd.concat([df, pd.DataFrame(abs_cols, index=df.index)], axis=1)
    else:
        close_fallback = close.ffill().bfill().fillna(0.0)
        df["nearest_aoi_type"] = ""
        df["nearest_aoi_distance"] = 0.0
        df["nearest_aoi_price"] = close_fallback
        df["nearest_aoi_type_code"] = -1
        df["at_any_aoi_band"] = 0
        df["at_upper_aoi_band"] = 0
        df["at_lower_aoi_band"] = 0
    return df


__all__ = ["compute_orb_levels", "compute_session_levels", "compute_aoi_features"]
