from __future__ import annotations

"""
Microstructure feature set for scalper models.

This module stays completely opt-in. Nothing imports it by default unless a
caller explicitly asks for the ``scalp_micro_v1`` feature set.
"""

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    pv = (df["close"] * df["volume"]).rolling(window).sum()
    vv = df["volume"].rolling(window).sum()
    return pv / vv


def build_features_scalp_micro_v1(df: pd.DataFrame, tick_size: float = 0.25) -> pd.DataFrame:
    """Leak-safe, bar-close aligned micro features for 1m (or 2m) OHLCV."""
    df_norm = df.copy()
    df_norm.columns = [str(c).strip().lower() for c in df_norm.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df_norm.columns))
    if missing:
        raise KeyError(f"Missing required columns for scalp features: {', '.join(missing)}")

    f = pd.DataFrame(index=df_norm.index)
    # Microstructure
    f["ret_1"] = df_norm["close"].pct_change().fillna(0.0)
    f["range"] = (df_norm["high"] - df_norm["low"]) / tick_size
    f["upper_wick"] = (df_norm["high"] - df_norm[["close", "open"]].max(axis=1)) / tick_size
    f["lower_wick"] = (df_norm[["close", "open"]].min(axis=1) - df_norm["low"]) / tick_size
    # EMAs & slopes
    ema3, ema8, ema21 = _ema(df_norm["close"], 3), _ema(df_norm["close"], 8), _ema(df_norm["close"], 21)
    f["ema3_delta"] = (df_norm["close"] - ema3) / (tick_size * 4)
    f["ema8_delta"] = (df_norm["close"] - ema8) / (tick_size * 8)
    f["ema21_delta"] = (df_norm["close"] - ema21) / (tick_size * 21)
    f["ema3_slope"] = ema3.diff() / tick_size
    f["ema8_slope"] = ema8.diff() / tick_size
    # VWAP deviation
    vwap20 = _rolling_vwap(df_norm, 20)
    f["vwap20_dev"] = (df_norm["close"] - vwap20) / tick_size
    # ATR (fast) in ticks
    tr = pd.concat(
        [
            (df_norm["high"] - df_norm["low"]).abs(),
            (df_norm["high"] - df_norm["close"].shift()).abs(),
            (df_norm["low"] - df_norm["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    f["atr_ticks"] = (tr.rolling(14).mean().bfill() / tick_size).clip(0, 100)
    # Volume context
    vol1 = df_norm["volume"]
    f["vol_rel_20"] = (vol1 / (vol1.rolling(20).mean().replace(0, np.nan))).fillna(0.0)
    f["vol_change"] = vol1.pct_change().replace([np.inf, -np.inf], 0).fillna(0.0)
    # Micro-ORB
    hh5, ll5 = df_norm["high"].rolling(5).max(), df_norm["low"].rolling(5).min()
    f["above_hh5"] = (df_norm["close"] > hh5.shift()).astype("int8")
    f["below_ll5"] = (df_norm["close"] < ll5.shift()).astype("int8")
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)
