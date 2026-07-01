"""
ES-specific regime heuristics and features.

The goal is to expose leak-safe regime features that can be consumed
by downstream setup filters, threshold optimizers, and trainers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict

import numpy as np
import pandas as pd

from .config import CLOSE_COL, HIGH_COL, LOW_COL, OPEN_COL

EPS = 1e-12


@dataclass
class RegimeParams:
    vol_lookback_bars: int = 24
    trend_ema_fast: int = 20
    trend_ema_slow: int = 50
    chop_range_lookback: int = 36
    open_drive_minutes: int = 30
    # thresholds
    trend_slope_min: float = 0.0
    chop_compression_pct: float = 0.25


def _col_numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    series = frame.get(name)
    if series is None:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def _true_range(frame: pd.DataFrame) -> pd.Series:
    high = _col_numeric(frame, HIGH_COL)
    low = _col_numeric(frame, LOW_COL)
    close = _col_numeric(frame, CLOSE_COL)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum.reduce([tr1.fillna(0.0), tr2.fillna(0.0), tr3.fillna(0.0)])
    tr = pd.Series(tr, index=frame.index)
    return tr.fillna(0.0)


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    span = max(4, int(window))
    mean = series.rolling(span, min_periods=max(2, span // 2)).mean()
    std = series.rolling(span, min_periods=max(2, span // 2)).std()
    return (series - mean) / (std + EPS)


def _resolve_params(params: RegimeParams | Dict[str, float] | None) -> RegimeParams:
    if params is None:
        return RegimeParams()
    if isinstance(params, RegimeParams):
        return params
    allowed = {f.name for f in fields(RegimeParams)}
    kwargs = {k: v for k, v in dict(params).items() if k in allowed}
    return RegimeParams(**kwargs)


def compute_es_regime_features(df: pd.DataFrame, *, params: RegimeParams | Dict[str, float]) -> pd.DataFrame:
    """
    Compute leak-safe regime features including volatility z-scores, EMA trend slopes,
    VWAP slope proxies, compression, range expansion, and time bucket indicators.
    """

    params = _resolve_params(params)
    if df.empty:
        return pd.DataFrame(index=df.index)

    close = _col_numeric(df, CLOSE_COL)
    high = _col_numeric(df, HIGH_COL)
    low = _col_numeric(df, LOW_COL)
    atr = _col_numeric(df, "atr_14")
    true_range = _true_range(df)
    if atr.isna().all():
        atr = true_range.rolling(14, min_periods=5).mean()
    atr = atr.replace(0.0, np.nan).fillna(true_range.rolling(20, min_periods=5).mean()).fillna(1.0)

    returns = close.pct_change()
    vol_series = returns.rolling(params.vol_lookback_bars, min_periods=max(4, params.vol_lookback_bars // 2)).std()
    vol_z = _rolling_z(vol_series, max(params.vol_lookback_bars * 2, params.vol_lookback_bars + 4)).fillna(0.0)

    ema_fast = close.ewm(span=params.trend_ema_fast, adjust=False).mean()
    ema_slow = close.ewm(span=params.trend_ema_slow, adjust=False).mean()
    ema_spread = ema_fast - ema_slow
    ema_norm = ema_spread / (atr + EPS)
    ema_slope = ema_norm.rolling(3, min_periods=1).mean().fillna(0.0)

    vwap = _col_numeric(df, "vwap_sess")
    if vwap.isna().all():
        vwap_slope = pd.Series(0.0, index=df.index)
    else:
        vwap_slope = (vwap.diff().rolling(3, min_periods=1).mean() / (atr + EPS)).fillna(0.0)

    short_window = max(4, params.chop_range_lookback // 2)
    long_window = max(params.chop_range_lookback, short_window * 3)
    range_short = true_range.rolling(short_window, min_periods=max(3, short_window // 2)).mean()
    range_long = true_range.rolling(long_window, min_periods=max(5, long_window // 3)).mean()
    compression = (range_short / (range_long + EPS)).clip(0.0, 5.0).fillna(0.0)
    range_expansion = (true_range / (range_long + EPS)).clip(0.0, 10.0).fillna(0.0)

    mins_from_open = _col_numeric(df, "mins_from_open")
    bucket_open = ((mins_from_open >= 0) & (mins_from_open <= params.open_drive_minutes)).astype(float).replace(np.nan, 0.0)
    mid_end = params.open_drive_minutes + 180
    bucket_mid = (
        (mins_from_open > params.open_drive_minutes) & (mins_from_open <= mid_end)
    ).astype(float).replace(np.nan, 0.0)
    bucket_close = ((mins_from_open > mid_end) | mins_from_open.isna()).astype(float)
    bucket_close = bucket_close.replace(np.nan, 0.0)

    return pd.DataFrame(
        {
            "regime_vol_z": vol_z,
            "regime_ema_spread": ema_norm.fillna(0.0),
            "regime_ema_slope": ema_slope,
            "regime_vwap_slope": vwap_slope,
            "regime_compression_score": compression,
            "regime_range_expansion": range_expansion,
            "regime_time_open": bucket_open,
            "regime_time_mid": bucket_mid,
            "regime_time_close": bucket_close,
        },
        index=df.index,
    )


def assign_es_regime_label(df: pd.DataFrame, *, params: RegimeParams | Dict[str, float]) -> pd.Series:
    """
    Assign heuristic regime labels:
      0 = chop/balance
      1 = trend
      2 = expansion/open-drive
    """

    params = _resolve_params(params)
    if "regime_vol_z" not in df.columns:
        features = compute_es_regime_features(df, params=params)
        df = df.join(features, how="left")

    vol_z = _col_numeric(df, "regime_vol_z").fillna(0.0)
    ema_slope = _col_numeric(df, "regime_ema_slope").fillna(0.0)
    compression = _col_numeric(df, "regime_compression_score").fillna(0.0)
    range_expansion = _col_numeric(df, "regime_range_expansion").fillna(0.0)
    mins_from_open = _col_numeric(df, "mins_from_open")

    labels = pd.Series(0, index=df.index, dtype="int8")
    expansion_mask = (range_expansion >= 1.35) & (vol_z >= 0.5)
    if not mins_from_open.isna().all():
        expansion_mask &= (mins_from_open <= params.open_drive_minutes) | (range_expansion >= 1.6)
    labels.loc[expansion_mask] = 2

    threshold = params.trend_slope_min if params.trend_slope_min > 0 else 0.25
    trend_mask = (labels == 0) & (ema_slope.abs() >= threshold) & (compression > params.chop_compression_pct)
    labels.loc[trend_mask] = 1
    labels = labels.fillna(0).astype("int8")
    return labels


__all__ = [
    "RegimeParams",
    "compute_es_regime_features",
    "assign_es_regime_label",
]
