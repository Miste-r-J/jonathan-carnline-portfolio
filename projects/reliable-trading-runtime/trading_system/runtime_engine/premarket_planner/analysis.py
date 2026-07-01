from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

from trading_system.runtime_engine.modeling.config import PROB_BANDS

__all__ = [
    "PremarketSnapshot",
    "SignalSnapshot",
    "_determine_bias",
    "_format_confidence",
    "_format_pct",
    "_format_price",
    "_format_volume",
    "_load_latest_signal",
    "_load_prices",
    "_select_previous_rth",
    "_signal_age_hours",
    "summarize_premarket",
]


@dataclass
class PremarketSnapshot:
    analysis_date: date
    current_price: float
    prev_close: float
    prev_high: float
    prev_low: float
    prev_range: float
    prev_volume: float
    prev_session_vwap: float
    gap_points: float
    gap_pct: float
    overnight_high: float
    overnight_low: float
    overnight_range: float
    premarket_volume: float
    atr14: Optional[float]
    last_update: datetime
    session_tz: str
    missing_overnight: bool = False


@dataclass
class SignalSnapshot:
    side: str
    probability: float
    confidence: float
    grade: str
    timestamp: datetime


def _load_prices(csv_path: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.ParserWarning)
        df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")

    canonical = {
        "datetime": "Datetime",
        "timestamp": "Datetime",
        "time": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in canonical:
            rename_map[col] = canonical[key]
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"Datetime", "Open", "High", "Low", "Close", "Volume"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")
    return df.reset_index(drop=True)


def _localize_index(df: pd.DataFrame, session_tz: str) -> pd.DataFrame:
    tz = ZoneInfo(session_tz)
    idx = pd.DatetimeIndex(df["Datetime"])
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    df = df.set_index(idx)
    df.index.name = "Datetime"
    return df


def _between(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    try:
        return df.between_time(start, end, inclusive="both")
    except TypeError:
        return df.between_time(start, end)


def _select_previous_rth(
    rth_df: pd.DataFrame,
    analysis_date: date,
    rth_start: str,
) -> Tuple[date, pd.DataFrame]:
    if rth_df.empty:
        raise ValueError("No RTH bars available in the dataset")

    session_dates = sorted(set(rth_df.index.date))
    start_t = time.fromisoformat(rth_start)
    latest_ts = rth_df.index.max()

    if latest_ts.date() == analysis_date and latest_ts.time() >= start_t:
        target_date = analysis_date
    else:
        candidates = [d for d in session_dates if d < analysis_date]
        target_date = candidates[-1] if candidates else session_dates[-1]

    prev_rth = rth_df[rth_df.index.date == target_date]
    if prev_rth.empty:
        raise ValueError(f"No RTH data for target session {target_date}")
    return target_date, prev_rth


def _compute_daily_atr(rth_df: pd.DataFrame, window: int = 14) -> Optional[float]:
    if rth_df.empty:
        return None
    session_dates = sorted(set(rth_df.index.date))
    if len(session_dates) < 2:
        return None

    daily_tr: List[float] = []
    prev_close: Optional[float] = None
    for day in session_dates:
        session = rth_df[rth_df.index.date == day]
        if session.empty:
            continue
        high = float(session["High"].max())
        low = float(session["Low"].min())
        close = float(session["Close"].iloc[-1])
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        daily_tr.append(tr)
        prev_close = close

    if len(daily_tr) < window:
        return None
    atr = pd.Series(daily_tr, dtype=float).ewm(alpha=1 / window, adjust=False).mean().iloc[-1]
    return float(atr)


def summarize_premarket(
    df: pd.DataFrame,
    session_tz: str,
    rth_start: str,
    rth_end: str,
) -> PremarketSnapshot:
    df_local = _localize_index(df.copy(), session_tz)
    latest_ts = df_local.index[-1]
    analysis_date = latest_ts.date()

    rth_df = _between(df_local, rth_start, rth_end)
    session_date, prev_rth = _select_previous_rth(rth_df, analysis_date, rth_start)

    prev_close = float(prev_rth["Close"].iloc[-1])
    prev_high = float(prev_rth["High"].max())
    prev_low = float(prev_rth["Low"].min())
    prev_range = prev_high - prev_low
    prev_volume = float(prev_rth["Volume"].sum())
    prev_vwap = float((prev_rth["Close"] * prev_rth["Volume"]).sum() / prev_volume) if prev_volume else float("nan")

    tz = ZoneInfo(session_tz)
    prev_session_end = datetime.combine(session_date, time.fromisoformat(rth_end), tzinfo=tz)
    open_today = datetime.combine(analysis_date, time.fromisoformat(rth_start), tzinfo=tz)
    if open_today <= prev_session_end:
        open_today += timedelta(days=1)

    overnight_slice = df_local[(df_local.index > prev_session_end) & (df_local.index <= open_today)]
    if overnight_slice.empty:
        overnight_slice = df_local[df_local.index > prev_session_end]
    missing_overnight = overnight_slice.empty

    overnight_high = float(overnight_slice["High"].max()) if not overnight_slice.empty else float("nan")
    overnight_low = float(overnight_slice["Low"].min()) if not overnight_slice.empty else float("nan")
    overnight_range = overnight_high - overnight_low if not overnight_slice.empty else float("nan")
    premarket_volume = float(overnight_slice["Volume"].sum()) if not overnight_slice.empty else 0.0

    current_price = float(df_local["Close"].iloc[-1])
    gap_points = current_price - prev_close
    gap_pct = gap_points / prev_close if prev_close else float("nan")
    atr14 = _compute_daily_atr(rth_df)

    return PremarketSnapshot(
        analysis_date=analysis_date,
        current_price=current_price,
        prev_close=prev_close,
        prev_high=prev_high,
        prev_low=prev_low,
        prev_range=prev_range,
        prev_volume=prev_volume,
        prev_session_vwap=prev_vwap,
        gap_points=gap_points,
        gap_pct=gap_pct,
        overnight_high=overnight_high,
        overnight_low=overnight_low,
        overnight_range=overnight_range,
        premarket_volume=premarket_volume,
        atr14=atr14,
        last_update=latest_ts,
        session_tz=session_tz,
        missing_overnight=missing_overnight,
    )


def _load_latest_signal(signals_path: Optional[str], session_tz: str) -> Optional[SignalSnapshot]:
    if not signals_path:
        return None
    try:
        df = pd.read_csv(signals_path)
        if "prob" not in df.columns:
            df = pd.read_csv(
                signals_path,
                header=None,
                names=["datetime", "type", "side", "price", "prob", "grade", "stop", "target"],
            )
    except Exception:
        return None
    if df.empty or "prob" not in df.columns or "side" not in df.columns:
        return None
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    if df.empty:
        return None
    df["type"] = df["type"].astype(str).str.upper()
    df = df[df["type"].isin(["OPEN", "FLIP"])]
    if df.empty:
        return None
    row = df.iloc[-1]
    side = str(row.get("side", "")).upper()
    prob = float(row.get("prob", float("nan")))
    if pd.isna(prob):
        return None

    confidence = 1.0 - prob if side == "SHORT" else prob
    confidence = max(0.0, min(1.0, confidence))
    grade = _prob_to_grade(confidence)

    ts = row["datetime"]
    if ts.tzinfo is None:
        ts = ts.tz_localize(ZoneInfo(session_tz))
    else:
        ts = ts.tz_convert(ZoneInfo(session_tz))

    return SignalSnapshot(side=side or "FLAT", probability=prob, confidence=confidence, grade=grade, timestamp=ts)


def _prob_to_grade(confidence: float) -> str:
    if pd.isna(confidence):
        return "N/A"
    bands = PROB_BANDS["long"]
    if confidence >= bands["A+"]:
        return "A+"
    if confidence >= bands["B+"]:
        return "B+"
    return "C"


def _determine_bias(signal: Optional[SignalSnapshot], blockers: List[str]) -> str:
    if signal is None or blockers:
        return "FLAT"
    if signal.confidence < 0.55:
        return "FLAT"
    if signal.side in ("LONG", "SHORT"):
        return signal.side
    return "FLAT"


def _format_confidence(signal: Optional[SignalSnapshot]) -> str:
    if signal is None:
        return "N/A"
    return f"{signal.confidence:.2f} (grade {signal.grade})"


def _format_price(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


def _format_volume(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def _format_pct(value: float) -> str:
    if pd.isna(value) or not math.isfinite(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _signal_age_hours(
    signal: Optional[SignalSnapshot],
    now_dt: datetime,
    tz: ZoneInfo,
) -> Optional[float]:
    if signal is None:
        return None
    signal_ts = signal.timestamp
    if signal_ts.tzinfo is None:
        signal_ts = signal_ts.replace(tzinfo=tz)
    else:
        signal_ts = signal_ts.astimezone(tz)
    delta = abs((now_dt - signal_ts).total_seconds()) / 3600.0
    return float(delta)
