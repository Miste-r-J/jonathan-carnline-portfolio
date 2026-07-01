from __future__ import annotations

"""
Microstructure features derived from tick data (v1 approximations).
"""

from typing import Optional

import numpy as np
import pandas as pd

from .tick_data import normalize_tick_df


def compute_microstructure_features(tick_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_tick_df(tick_df, symbol=str(tick_df.get("symbol", "SYM")[0] if len(tick_df) else "SYM"))
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = (df["ask"] - df["bid"]).clip(lower=0.0)
    df["ob_imbalance"] = df["bid_size"] / (df["bid_size"] + df["ask_size"]).replace(0.0, np.nan)
    df["ob_imbalance"] = df["ob_imbalance"].fillna(0.5)
    # trade direction proxy: last vs mid
    df["trade_sign"] = np.where(df["last_price"] >= df["mid"], 1.0, -1.0)
    df["aggressive_buy_vol"] = np.where(df["trade_sign"] > 0, df["last_size"], 0.0)
    df["aggressive_sell_vol"] = np.where(df["trade_sign"] < 0, df["last_size"], 0.0)
    # short-horizon realized vol (rolling std of mid returns)
    df["mid_ret"] = df["mid"].pct_change().fillna(0.0)
    df["rv_10s"] = df["mid_ret"].rolling(10, min_periods=3).std().fillna(0.0)
    df["rv_60s"] = df["mid_ret"].rolling(60, min_periods=10).std().fillna(0.0)
    df["spread_trend"] = df["spread"].diff().fillna(0.0)
    return df


def add_microstructure_features_to_bars(bar_df: pd.DataFrame, tick_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Aggregate tick microstructure signals to bar cadence and merge into bar_df (on index).
    """
    ticks = normalize_tick_df(tick_df, symbol)
    feats = compute_microstructure_features(ticks)
    if "timestamp" in feats.columns:
        feats = feats.set_index("timestamp")
    agg = feats.resample("1T").agg(
        {
            "ob_imbalance": "mean",
            "rv_10s": "mean",
            "rv_60s": "mean",
            "spread": "mean",
            "spread_trend": "mean",
            "aggressive_buy_vol": "sum",
            "aggressive_sell_vol": "sum",
        }
    ).rename(columns=lambda c: f"micro_{c}")
    bars = bar_df.copy()
    if "Datetime" in bars.columns:
        bars = bars.set_index("Datetime")
    merged = bars.join(agg, how="left")
    return merged


__all__ = ["compute_microstructure_features", "add_microstructure_features_to_bars"]
