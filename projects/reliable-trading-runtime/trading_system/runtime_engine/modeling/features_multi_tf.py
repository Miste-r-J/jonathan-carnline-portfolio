from __future__ import annotations

import pandas as pd
from typing import Dict, List

# Make pandas future downcasting behavior explicit to avoid noisy warnings.
pd.set_option("future.no_silent_downcasting", True)

DEFAULT_TIMEFRAMES = ["5min", "15min", "60min", "240min"]  # 5m, 15m, 1h, 4h


def resample_ohlc(df_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample base 5m OHLCV into a higher timeframe (e.g. 15T, 60T, 240T).
    Assumes df_5m has a DateTimeIndex and columns: open, high, low, close, volume.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df_5m.resample(rule).agg(agg).dropna()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()


def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(length, min_periods=1).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["close"] * df["volume"]).cumsum()
    vol = df["volume"].cumsum().replace(0, pd.NA)
    return pv / vol


def _trend_slope(series: pd.Series, window: int) -> pd.Series:
    """Simple slope estimate using rolling linear regression approximation."""
    import numpy as np

    def slope(vals: pd.Series) -> float:
        y = vals.values
        x = np.arange(len(y))
        if len(y) < 2:
            return 0.0
        x_mean = x.mean()
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        return float(num / den) if den != 0 else 0.0

    return series.rolling(window, min_periods=2).apply(slope, raw=False)


def build_multi_tf_features(
    df_5m: pd.DataFrame,
    timeframes: List[str] | None = None,
) -> pd.DataFrame:
    """
    Given a base 5m OHLCV DataFrame (indexed by timestamp),
    compute multi-timeframe features and align them back to the 5m index.

    Returns a feature DataFrame indexed like df_5m, with columns such as:
      ema_5T_20, atr_5T_14, vwap_5T, ...
    """
    if timeframes is None:
        timeframes = DEFAULT_TIMEFRAMES
    if df_5m.empty:
        return df_5m.copy()

    feats: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        resampled = resample_ohlc(df_5m, tf)
        f = pd.DataFrame(index=resampled.index)
        f[f"ema_{tf}_20"] = _ema(resampled["close"], 20)
        f[f"ema_{tf}_50"] = _ema(resampled["close"], 50)
        f[f"atr_{tf}_14"] = _atr(resampled, 14)
        f[f"rsi_{tf}_14"] = _rsi(resampled["close"], 14)
        f[f"vwap_{tf}"] = _vwap(resampled)
        f[f"trend_slope_{tf}_20"] = _trend_slope(resampled["close"], 20)
        resampled_aligned = f.reindex(df_5m.index)
        resampled_aligned = resampled_aligned.ffill()
        resampled_aligned = resampled_aligned.infer_objects(copy=False)
        feats[tf] = resampled_aligned

    return pd.concat(feats.values(), axis=1)
