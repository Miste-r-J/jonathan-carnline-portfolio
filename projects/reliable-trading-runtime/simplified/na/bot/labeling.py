#labeling.py

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .config import CLOSE_COL, HORIZON, RET_THRESHOLD
from ..config.loader import load_app_config
from ..config.app_config import LabelSection


logger = logging.getLogger(__name__)

_LABEL_DEFAULTS: Optional[LabelSection] = None


def get_default_label_config() -> LabelSection:
    global _LABEL_DEFAULTS
    if _LABEL_DEFAULTS is None:
        _LABEL_DEFAULTS = load_app_config().labels
    return _LABEL_DEFAULTS


def make_labels(
    df: pd.DataFrame,
    close_col: str = CLOSE_COL,
    horizon: Optional[int] = None,
    threshold: float = RET_THRESHOLD,
    scheme: Optional[str] = None,   # "binary" | "ternary" | "direction" | "trend_aware_ternary"
    use_log: bool = False,
    *,
    quantile: Optional[float] = None,
    drop_flat: Optional[bool] = None,
    use_htf_trend_aware: Optional[bool] = None,  # Experimental: incorporate HTF trend filter to labels
) -> pd.DataFrame:
    """
    Create forward-return labels.

    scheme:
      - "binary": 1 if fwd_ret > threshold else 0
      - "ternary": 1 if > +threshold, -1 if < -threshold, 0 otherwise
      - "direction": sign(fwd_ret) in {-1, 0, 1}
    """

    cfg = get_default_label_config()
    if horizon is None:
        horizon = cfg.horizon_bars or HORIZON
    if scheme is None:
        scheme = cfg.domain or "binary"
    if drop_flat is None:
        drop_flat = bool(cfg.drop_flats)
    if use_htf_trend_aware is None:
        use_htf_trend_aware = scheme == "trend_aware_ternary"

    if close_col not in df.columns:
        raise KeyError(f"Column '{close_col}' not present in frame; cannot build labels")
    if horizon <= 0:
        raise ValueError("Label horizon must be positive")
    if len(df) <= horizon:
        raise ValueError("Not enough rows to compute forward returns for requested horizon")

    out = df.copy()

    if isinstance(out.index, pd.DatetimeIndex) and not out.index.is_monotonic_increasing:
        logger.warning("Label frame index was not sorted; sorting to enforce time monotonicity.")
        out = out.sort_index()

    # Forward return over `horizon` bars
    if use_log:
        fwd_ret = np.log(out[close_col].shift(-horizon)) - np.log(out[close_col])
    else:
        fwd_ret = out[close_col].shift(-horizon) / out[close_col] - 1.0

    out["fwd_ret"] = fwd_ret

    if out["fwd_ret"].isna().all():
        raise ValueError("Forward returns are all NaN; check horizon and price column")

    threshold_used = float(threshold)
    if quantile is not None:
        q = float(quantile)
        if not 0.0 < q < 0.5:
            raise ValueError("quantile must lie in (0, 0.5)")
        ret_no_nan = fwd_ret.dropna()
        if ret_no_nan.empty:
            raise ValueError("Cannot compute threshold from empty returns")
        pos_thr = float(ret_no_nan.quantile(1.0 - q))
        neg_thr = float(ret_no_nan.quantile(q))
        threshold_used = max(abs(pos_thr), abs(neg_thr))
        if threshold_used <= 0.0:
            threshold_used = float(ret_no_nan.abs().quantile(1.0 - q))
        if threshold_used <= 0.0:
            raise ValueError("Auto-computed threshold is non-positive")

    # Targets
    if scheme == "binary":
        out["target"] = (fwd_ret > threshold_used).astype(np.int8)
    elif scheme == "ternary" or scheme == "trend_aware_ternary":
        if use_htf_trend_aware:
            # Implementation fix: incorporate HTF trend to only label in trend direction
            # This prevents up labels in downtrends and up signals on obvious selloffs
            ma_window = cfg.trend_ma_window or 200
            slope_window = cfg.trend_slope_window or 20
            ma = out[close_col].rolling(ma_window, min_periods=ma_window).mean()
            # Handle potential NaN in slope calculation
            def safe_slope(x):
                if len(x) < slope_window or x.isna().any():
                    return np.nan
                try:
                    coeff = np.polyfit(np.arange(len(x)), x, 1)
                    return coeff[0]
                except:
                    return np.nan
            out["htf_ma_slope"] = ma.rolling(slope_window, min_periods=slope_window).apply(safe_slope)
            htf_trend = np.sign(out["htf_ma_slope"].fillna(0)).astype(int)  # -1: down, 0: chop, 1: up
            out["htf_trend"] = htf_trend
            # Only label if fwd_ret direction aligns with HTF trend and exceeds threshold
            # Otherwise, label as no-trade (0)
            fwd_dir = np.sign(fwd_ret)
            trend_aligned = (fwd_dir == htf_trend) & (htf_trend != 0)  # Only aligned in clear up/down trends
            large_enough = fwd_ret.abs() > threshold_used
            out["target"] = np.where(
                trend_aligned & large_enough & (fwd_dir == 1), 1,  # up in uptrend
                np.where(trend_aligned & large_enough & (fwd_dir == -1), -1,  # down in downtrend
                         0  # no trade otherwise
                )
            ).astype(np.int8)
        else:
            out["target"] = np.where(
                fwd_ret > threshold_used, 1,
                np.where(fwd_ret < -threshold_used, -1, 0)
            ).astype(np.int8)
    elif scheme == "direction":
        out["target"] = np.sign(fwd_ret).astype(np.int8)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # Drop the trailing rows with undefined forward returns (no index reset)
    if horizon > 0:
        out = out.iloc[:-horizon]

    if drop_flat and scheme in {"ternary", "direction", "trend_aware_ternary"}:
        out = out[out["target"] != 0]

    out.attrs["threshold_used"] = threshold_used

    return out
