from __future__ import annotations

import logging
import math
import re
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

from .analysis import (
    PremarketSnapshot,
    _load_prices,
    _select_previous_rth,
    summarize_premarket,
)
from .config import PlannerConfig

LOGGER = logging.getLogger("premarket_planner.core")
TZ_OFFSET_RE = re.compile(r"([+-])(\d{2}):?(\d{2})$")


@dataclass
class WindowStatus:
    """
    Metadata for the overnight window used to compute ETH statistics.
    """

    start: datetime
    end: datetime
    initial_bars: int
    bars: int
    used_fallback: bool
    fallback_reason: Optional[str] = None
    frame: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


@dataclass
class PlanComputation:
    """
    Result bundle produced by generate_plan.
    """

    snapshot: PremarketSnapshot
    window: WindowStatus
    dataframe: pd.DataFrame = field(repr=False)
    source_timezone: str = "UTC"
    vwap_source: str = "snapshot"
    warnings: Tuple[str, ...] = ()


def generate_plan(
    config: PlannerConfig,
    *,
    logger: Optional[logging.Logger] = None,
    now: Optional[datetime] = None,
) -> PlanComputation:
    """
    Compute the latest premarket snapshot and enrich it with hardened window statistics.
    """
    log = logger or LOGGER

    df_raw = _load_prices(str(config.data.csv_path))
    source_tz = _determine_source_timezone(df_raw, config.data.csv_path, config)
    now_dt = now.astimezone(config.session_zone) if now else datetime.now(config.session_zone)
    df_prepared = _apply_timezone(df_raw, source_tz, config.session_zone)
    df_prepared = _maybe_backfill(df_prepared, config, source_tz, now_dt, log)
    log.debug("Loaded %d bars from %s (source tz=%s)", len(df_prepared), config.data.csv_path, source_tz)
    df_local = df_prepared.copy()
    df_local = df_local.set_index(pd.DatetimeIndex(df_local["Datetime"]))

    window = _compute_overnight_window(df_local, config, now_dt, log)
    snapshot = summarize_premarket(df_prepared.copy(), config.session_tz, config.rth_start_str, config.rth_end_str)
    snapshot = _apply_window_to_snapshot(snapshot, window, log)

    vwap_source = "snapshot"
    if config.metrics.compute_rth_vwap:
        snapshot, vwap_source = _ensure_rth_vwap(snapshot, df_local, config, log)

    warnings: list[str] = []
    if window.used_fallback:
        msg = "ETH fallback window applied"
        if window.fallback_reason:
            msg = f"{msg}: {window.fallback_reason}"
        warnings.append(msg)
    if math.isnan(snapshot.prev_session_vwap):
        warnings.append("Unable to compute RTH VWAP")

    return PlanComputation(
        snapshot=snapshot,
        window=window,
        dataframe=df_local,
        source_timezone=source_tz,
        vwap_source=vwap_source,
        warnings=tuple(warnings),
    )


def _determine_source_timezone(
    df: pd.DataFrame,
    csv_path: Path,
    config: PlannerConfig,
) -> str:
    tz = df["Datetime"].dt.tz
    if tz is not None:
        tz_name = getattr(tz, "key", None) or getattr(tz, "zone", None)
        if isinstance(tz_name, str):
            return tz_name
    if config.data.csv_timezone:
        return config.data.csv_timezone
    inferred = _infer_timezone_from_csv(csv_path)
    if inferred:
        return inferred
    return config.session_tz


def _infer_timezone_from_csv(csv_path: Path, sample_rows: int = 25) -> Optional[str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.ParserWarning)
            sample = pd.read_csv(csv_path, nrows=sample_rows, dtype=str, on_bad_lines="skip", engine="python")
    except Exception:
        return None

    datetime_col = None
    for col in sample.columns:
        if str(col).strip().lower() in {"datetime", "timestamp", "time"}:
            datetime_col = col
            break
    if datetime_col is None:
        return None

    series = sample[datetime_col].dropna()
    for value in series:
        token = str(value).strip()
        if not token:
            continue
        if token.endswith("Z"):
            return "UTC"
        match = TZ_OFFSET_RE.search(token)
        if match:
            hours = int(match.group(1) + match.group(2))
            minutes = int(match.group(3))
            if hours == 0 and minutes == 0:
                return "UTC"
            # Offset present but not necessarily mapped to a TZ database name.
            # Prefer UTC fallback to avoid silent misalignment.
            return "UTC"
    return None


def _apply_timezone(
    df: pd.DataFrame,
    source_tz: str,
    target_zone: ZoneInfo,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df["Datetime"])
    if idx.tz is None:
        idx = idx.tz_localize(ZoneInfo(source_tz))
    else:
        idx = idx.tz_convert(ZoneInfo(source_tz))
    idx = idx.tz_convert(target_zone)
    df = df.copy()
    df["Datetime"] = idx
    return df


def _maybe_backfill(
    df_prepared: pd.DataFrame,
    config: PlannerConfig,
    source_tz: str,
    now_dt: datetime,
    log: logging.Logger,
) -> pd.DataFrame:
    if not config.backfill.enabled:
        return df_prepared

    idx = pd.DatetimeIndex(df_prepared["Datetime"])
    if idx.empty:
        age_hours = float("inf")
        should_backfill = True
    else:
        latest = idx.max()
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=config.session_zone)
        age_hours = (now_dt - latest).total_seconds() / 3600.0
        should_backfill = age_hours > float(config.backfill.hours)

    if not should_backfill:
        return df_prepared

    if not _attempt_backfill(config, log):
        if math.isfinite(age_hours):
            log.warning("Price data is %.1f hours old; proceeding without backfill", age_hours)
        else:
            log.warning("Price data set is empty; proceeding without backfill")
        return df_prepared

    refreshed = _load_prices(str(config.data.csv_path))
    log.info("Reloaded price data after backfill request")
    return _apply_timezone(refreshed, source_tz, config.session_zone)


def _compute_overnight_window(
    df_local: pd.DataFrame,
    config: PlannerConfig,
    now_dt: datetime,
    log: logging.Logger,
) -> WindowStatus:
    tz = config.session_zone
    end_time = time(config.windows.eth_end_hour, config.windows.eth_end_minute)
    window_end = min(
        now_dt,
        datetime.combine(now_dt.date(), end_time, tzinfo=tz),
    )

    start_time = time(config.windows.eth_start_hour, 0)
    window_start = datetime.combine(window_end.date(), start_time, tzinfo=tz)
    if window_start > window_end:
        window_start -= timedelta(days=1)

    slice_df = df_local[(df_local.index >= window_start) & (df_local.index <= window_end)]
    used_fallback = False
    fallback_reason: Optional[str] = None
    original_bars = len(slice_df)

    if original_bars < config.windows.min_eth_bars:
        fallback_start = window_end - timedelta(hours=config.windows.last_hours_fallback)
        fallback_df = df_local[(df_local.index > fallback_start) & (df_local.index <= window_end)]
        if not fallback_df.empty:
            slice_df = fallback_df
            used_fallback = True
            fallback_reason = (
                f"Insufficient ETH bars ({original_bars}/{config.windows.min_eth_bars}); "
                f"using last {config.windows.last_hours_fallback}h"
            )
        else:
            used_fallback = True
            fallback_reason = "No data available for fallback window"
            log.warning(
                "No data available for fallback ETH window (%s to %s)",
                fallback_start.isoformat(),
                window_end.isoformat(),
            )

    return WindowStatus(
        start=window_start,
        end=window_end,
        initial_bars=original_bars,
        bars=len(slice_df),
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        frame=slice_df,
    )


def _apply_window_to_snapshot(
    snapshot: PremarketSnapshot,
    window: WindowStatus,
    log: logging.Logger,
) -> PremarketSnapshot:
    if window.frame.empty:
        log.warning("ETH window is empty; overnight statistics unavailable")
        return replace(
            snapshot,
            missing_overnight=True,
            premarket_volume=0.0,
            overnight_high=float("nan"),
            overnight_low=float("nan"),
            overnight_range=float("nan"),
        )

    high = float(window.frame["High"].max())
    low = float(window.frame["Low"].min())
    volume = float(window.frame["Volume"].sum())

    log.debug(
        "Computed ETH window %s — %s (%d bars, fallback=%s)",
        window.start.isoformat(),
        window.end.isoformat(),
        window.bars,
        window.used_fallback,
    )

    return replace(
        snapshot,
        missing_overnight=False,
        overnight_high=high,
        overnight_low=low,
        overnight_range=high - low,
        premarket_volume=volume,
    )


def _ensure_rth_vwap(
    snapshot: PremarketSnapshot,
    df_local: pd.DataFrame,
    config: PlannerConfig,
    log: logging.Logger,
) -> Tuple[PremarketSnapshot, str]:
    if not (math.isnan(snapshot.prev_session_vwap) or snapshot.prev_session_vwap == 0.0):
        return snapshot, "snapshot"

    rth_df = df_local.between_time(config.rth_start_str, config.rth_end_str, inclusive="both")
    if rth_df.empty:
        log.warning("RTH slice is empty; cannot compute VWAP")
        return snapshot, "missing"

    try:
        _, prev_rth = _select_previous_rth(rth_df, snapshot.analysis_date, config.rth_start_str)
    except Exception as exc:
        log.warning("Failed to select previous RTH session: %s", exc)
        return snapshot, "missing"

    if prev_rth.empty:
        log.warning("Previous RTH session data empty after selection")
        return snapshot, "missing"

    volume = float(prev_rth["Volume"].sum())
    if volume <= 0:
        fallback = float(prev_rth["Close"].iloc[-1])
        log.warning("Previous RTH volume is zero; using last close %.2f as VWAP fallback", fallback)
        return replace(snapshot, prev_session_vwap=fallback), "close_fallback"

    vwap = float((prev_rth["Close"] * prev_rth["Volume"]).sum() / volume)
    log.debug("Recomputed RTH VWAP: %.4f (volume %.0f)", vwap, volume)
    return replace(snapshot, prev_session_vwap=vwap, prev_volume=volume), "recalculated"


def _attempt_backfill(config: PlannerConfig, log: logging.Logger) -> bool:
    try:
        from trading_system.runtime_engine.runtime_tools.data import backfill_csv  # type: ignore
    except Exception:
        log.debug("Backfill helper trading_system.runtime_engine.runtime_tools.data.backfill_csv not available; skipping")
        return False

    try:
        backfill_csv(
            path=config.data.csv_path,
            hours=config.backfill.hours,
            instrument=config.instrument,
        )
        log.info(
            "Triggered backfill helper for %s (%sh window)",
            config.data.csv_path,
            config.backfill.hours,
        )
        return True
    except Exception as exc:
        log.warning("Backfill helper failed: %s", exc)
        return False
