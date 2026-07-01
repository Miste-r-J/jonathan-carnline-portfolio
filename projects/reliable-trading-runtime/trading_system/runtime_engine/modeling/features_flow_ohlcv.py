from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import FeaturePipelineError

_EPS = 1e-12


def add_flow_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add OHLCV-based 'market flow' proxies.
    Assumes df has columns: Open, High, Low, Close, Volume.
    Uses bar-close information (OK if your decision is made after bar close).
    """
    out = df.copy()

    o = out["Open"].astype(float)
    h = out["High"].astype(float)
    l = out["Low"].astype(float)
    c = out["Close"].astype(float)
    v = out["Volume"].astype(float)

    rng = (h - l)
    body = (c - o)
    body_abs = body.abs()

    # Who controlled the bar? (-1..+1)
    out["clv"] = (((c - l) - (h - c)) / (rng + _EPS)).clip(-3, 3)

    # Candle structure
    out["range"] = rng
    out["body"] = body
    out["body_to_range"] = np.where(rng == 0, 0, body_abs / rng).clip(0, 5)

    out["upper_wick"] = (h - np.maximum(o, c)).clip(lower=0)
    out["lower_wick"] = (np.minimum(o, c) - l).clip(lower=0)
    out["upper_wick_ratio"] = np.where(rng == 0, 0, out["upper_wick"] / rng).clip(0, 5)
    out["lower_wick_ratio"] = np.where(rng == 0, 0, out["lower_wick"] / rng).clip(0, 5)

    # Returns / impulse proxies
    # ret_1 is computed in the primary feature pipeline (features.py).
    # Do NOT recompute/overwrite it here.
    if "ret_1" not in out.columns:
        raise FeaturePipelineError("ret_1 missing in flow_ohlcv stage; primary pipeline did not run first.")
    out["logret_1"] = np.log(c).diff().replace([np.inf, -np.inf], np.nan)
    atr = pd.to_numeric(out.get("atr_14"), errors="coerce")
    out["impulse_atr"] = (body_abs / (atr + _EPS)).replace([np.inf, -np.inf], np.nan).clip(0, 100)

    # Signed volume & "CVD-like" proxy (very common OHLCV approximation)
    out["signed_vol"] = np.sign(body).replace(0, np.nan).fillna(0.0) * v

    for w in (5, 10, 20, 50):
        out[f"signed_vol_sum_{w}"] = out["signed_vol"].rolling(w, min_periods=max(3, w // 2)).sum()
        out[f"vol_sum_{w}"] = v.rolling(w, min_periods=max(3, w // 2)).sum()
        out[f"pressure_{w}"] = (out[f"signed_vol_sum_{w}"] / (out[f"vol_sum_{w}"] + _EPS)).clip(-5, 5)

    # Effort vs result (high volume but low range can imply absorption)
    out["vol_per_range"] = np.where(rng == 0, 0, v / rng).clip(0, 1e6)
    out["pressure_5_vs_20"] = out["pressure_5"] - out["pressure_20"]
    out["pressure_10_vs_50"] = out["pressure_10"] - out["pressure_50"]

    # Bar “dominance” streaks (simple human-like tape feel)
    up = (c > o).astype(int)
    dn = (c < o).astype(int)
    for w in (5, 10, 20):
        out[f"up_count_{w}"] = up.rolling(w, min_periods=max(3, w // 2)).sum()
        out[f"dn_count_{w}"] = dn.rolling(w, min_periods=max(3, w // 2)).sum()
        out[f"trend_bias_{w}"] = (out[f"up_count_{w}"] - out[f"dn_count_{w}"]) / float(w)

    wick_thresh = 0.6
    out["wick_reject_upper"] = ((out["upper_wick_ratio"] > wick_thresh) & (body < 0)).astype(float)
    out["wick_reject_lower"] = ((out["lower_wick_ratio"] > wick_thresh) & (body > 0)).astype(float)
    out["clv_smooth_5"] = out["clv"].rolling(5, min_periods=3).mean()

    return out
