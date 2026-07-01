# bot/features.py
import logging
import os
from functools import lru_cache
from typing import Any, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .exceptions import ConfigurationError, FeaturePipelineError
from .feature_constants import MANDATORY_MODEL_FEATURES, SAFE_ZERO_FILL_FEATURES
from .features_flow_ohlcv import add_flow_ohlcv_features
from .orb_aoi_features import (
    compute_aoi_features,
    compute_orb_levels,
    compute_session_levels,
    _merge_strategy_config,
)
from .setups_es import SetupParams, compute_es_setups
from .regimes_es import RegimeParams, compute_es_regime_features, assign_es_regime_label
from .config import (
    CLOSE_COL,
    EMA_WINDOWS,
    HIGH_COL,
    LOW_COL,
    OPEN_COL,
    RSI_WINDOW,
    SMA_WINDOWS,
    VOL_WINDOW,
    VOLUME_COL,
)
from ..runtime_config.loader import load_app_config
from trading_system.runtime_engine.market_data.sessions import SessionDefinition, ensure_dataframe_index


logger = logging.getLogger(__name__)

# Core feature set required by current downstream models.
MANDATORY_STATIC_FEATURES = [
    "ret_1", "ret_5", "ret_10",
    "hl_range", "oc_range",
    "vol_rel", "vol_z", "vol_spike",
    "gap_sma_5", "gap_sma_10", "gap_sma_20", "gap_sma_50",
    "slope_ema_9", "slope_ema_20", "slope_ema_50",
    "rsi_14", "rsx_14",
    "vol_realized_5", "vol_realized_10",
    "vol_regime", "vol_regime_z", "vol_regime_bin",
    "true_range", "atr_14", "rexp_over_atr",
    "kc_up_20", "kc_dn_20",
    "bb_up_20", "bb_dn_20", "bb_bw_20",
    "in_rth",
    "vwap_sess", "vwap_cross", "vwap_cross_streak",
    "above_orb_high", "below_orb_low",
    "dist_orb_high", "dist_orb_low",
    "mins_from_open",
    "mins_since_orb_hi_break", "mins_since_orb_lo_break",
    "hi_break_follow", "lo_break_follow",
    "streak_up", "streak_dn",
    "mom_10", "mom10_over_vol",
    "overnight_gap",
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4",
    "cd_proxy", "cd_proxy_norm",
    "regime_id",
    "regime_vol_z",
    "regime_compression_score",
    "regime_ema_slope",
    "regime_vwap_slope",
    "regime_range_expansion",
]

_MANDATORY_ORB_TEMPLATES = [
    "orb{orb}_high",
    "orb{orb}_low",
    "orb{orb}_mid",
    "orb{orb}_rng",
]


def mandatory_features(
    orb_minutes: int = 15,
    extra_orb_minutes: Optional[Iterable[int]] = None,
) -> List[str]:
    orb_windows: List[int] = []
    for value in (orb_minutes, *(extra_orb_minutes or [])):
        try:
            window = int(value)
        except (TypeError, ValueError):
            continue
        if window not in orb_windows:
            orb_windows.append(window)
    orb_cols = [tpl.format(orb=window) for window in orb_windows for tpl in _MANDATORY_ORB_TEMPLATES]
    return MANDATORY_STATIC_FEATURES + orb_cols


# Backwards-compatible constant (defaults to ORB 15).
MANDATORY_FEATURES = mandatory_features()


@lru_cache(maxsize=1)
def _runtime_feature_flags() -> Mapping[str, Any]:
    try:
        cfg = load_app_config()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to load runtime config for feature flags: %s", exc)
        return {}
    payload = getattr(cfg, "features", None)
    if isinstance(payload, Mapping):
        return payload
    return {}


def _flow_features_enabled(strategy_config: Optional[Mapping[str, Any]]) -> bool:
    candidates: List[Mapping[str, Any]] = []
    if strategy_config:
        features_cfg = strategy_config.get("features")
        if isinstance(features_cfg, Mapping):
            candidates.append(features_cfg)
    runtime_flags = _runtime_feature_flags()
    if runtime_flags:
        candidates.append(runtime_flags)
    for cfg in candidates:
        flow_cfg = cfg.get("flow_ohlcv")
        if isinstance(flow_cfg, Mapping):
            enabled = bool(flow_cfg.get("enabled", True))
            if not enabled:
                needs_flow = any(
                    name.startswith(
                        ("pressure_", "signed_vol", "up_count_", "dn_count_", "trend_bias_", "wick_", "logret_")
                    )
                    for name in MANDATORY_MODEL_FEATURES
                )
                if needs_flow:
                    raise ConfigurationError(
                        "flow_ohlcv.enabled=false in config, but the production model requires flow features."
                    )
            return enabled
        if isinstance(flow_cfg, bool):
            enabled = bool(flow_cfg)
            if not enabled:
                needs_flow = any(
                    name.startswith(
                        ("pressure_", "signed_vol", "up_count_", "dn_count_", "trend_bias_", "wick_", "logret_")
                    )
                    for name in MANDATORY_MODEL_FEATURES
                )
                if needs_flow:
                    raise ConfigurationError(
                        "flow_ohlcv.enabled=false in config, but the production model requires flow features."
                    )
            return enabled
    return True

# ---------- helpers ----------
def _assign_orb_columns(
    out: pd.DataFrame,
    *,
    window: int,
    in_rth: pd.Series,
    delta_from_open_min: pd.Series,
    session_key: pd.Series,
    group_idx: pd.Series,
    bar_min: int,
) -> None:
    """
    Populate ORB high/low/mid/range columns for the requested window without look-ahead.

    Within the ORB window, highs/lows are expanding values up to the current bar.
    After the ORB window completes, the final ORB high/low are forward-filled for the rest of the session.
    """
    mask_orb = in_rth & (delta_from_open_min >= 0) & (delta_from_open_min < window)
    fallback_bars = max(1, window // max(1, bar_min))
    first_n_mask = group_idx < fallback_bars

    high_series = out[HIGH_COL].where(mask_orb)
    low_series = out[LOW_COL].where(mask_orb)

    # Expanding up to current bar (no future info) during ORB
    high_expanding = high_series.groupby(session_key).cummax()
    low_expanding = low_series.groupby(session_key).cummin()

    # Fallback in case of sparse data at session start
    high_fb = out[HIGH_COL].where(first_n_mask).groupby(session_key).cummax()
    low_fb = out[LOW_COL].where(first_n_mask).groupby(session_key).cummin()

    col_high = f"orb{window}_high"
    col_low = f"orb{window}_low"
    col_mid = f"orb{window}_mid"
    col_rng = f"orb{window}_rng"

    # Combine expanding + fallback, then hold constant after ORB ends via ffill
    out[col_high] = high_expanding.fillna(high_fb).groupby(session_key).ffill()
    out[col_low] = low_expanding.fillna(low_fb).groupby(session_key).ffill()
    out[col_mid] = (out[col_high] + out[col_low]) / 2.0
    out[col_rng] = (out[col_high] - out[col_low]) / (out[col_mid] + 1e-12)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _rsx_proxy(series: pd.Series, window: int = 14) -> pd.Series:
    rsi = _rsi(series, window)
    return rsi.ewm(span=max(2, window//2), adjust=False).mean().ewm(span=max(2, window//2), adjust=False).mean()

def _rolling_z(x: pd.Series, w: int) -> pd.Series:
    m = x.rolling(w, min_periods=w).mean()
    s = x.rolling(w, min_periods=w).std()
    return (x - m) / (s + 1e-12)

def _linreg_slope(x: pd.Series, w: int) -> pd.Series:
    """Slope of y ~ a + b*t over last w points (unscaled b), computed efficiently."""
    t = np.arange(len(x), dtype=float)
    ts = pd.Series(t, index=x.index)
    y  = x
    S_t  = ts.rolling(w, min_periods=w).sum()
    S_y  = y.rolling(w, min_periods=w).sum()
    S_tt = (ts*ts).rolling(w, min_periods=w).sum()
    S_ty = (ts*y).rolling(w, min_periods=w).sum()
    n = y.rolling(w, min_periods=w).count()
    num = n*S_ty - S_t*S_y
    den = n*S_tt - S_t*S_t
    return num / (den + 1e-12)

def _onehot_dayofweek(idx: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DataFrame(index=idx)
    for k in range(5):
        d[f"dow_{k}"] = (idx.dayofweek == k).astype(float)
    return d

def _streaks(binary_series: pd.Series, key: pd.Series) -> pd.Series:
    """Consecutive-count streak helper for per-session binary signals (0/1)."""
    g = (binary_series == 0).groupby(key).cumsum()
    return binary_series.groupby([key, g]).cumcount() + binary_series  # adds 0/1 start


def _ffill_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Forward-fill values within each group without relying on pandas groupby."""
    arr = np.asarray(values, dtype=float).copy()
    groups = np.asarray(groups)
    if len(arr) == 0:
        return arr
    last = np.nan
    prev_group = groups[0]
    if not np.isnan(arr[0]):
        last = arr[0]
    for i in range(1, len(arr)):
        g = groups[i]
        if g != prev_group:
            last = np.nan
            prev_group = g
        if np.isnan(arr[i]):
            if not np.isnan(last):
                arr[i] = last
        else:
            last = arr[i]
    return arr


def _build_primary_features(
    df: pd.DataFrame,
    *,
    tz: str,
    rth_start: str,
    rth_end: str,
    orb_minutes: int,
    extra_orb_minutes: Optional[Iterable[int]],
    csv_naive_is_utc: bool,
    keltner_mult: float,
    boll_mult: float,
    tick_size: float,
    strategy_config: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    session_def = SessionDefinition.from_strings(
        tz=tz,
        rth_start=rth_start,
        rth_end=rth_end,
        orb_minutes=int(orb_minutes),
    )
    out = ensure_dataframe_index(df, tz, naive_is_utc=csv_naive_is_utc)
    initial_cols = set(out.columns)
    out = out.sort_index()
    # Parse core price/volume fields to numeric early to avoid object dtypes breaking cumsums.
    for col in (OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL])
    idx_local = session_def.align_index(out.index)
    session_key = session_def.session_index(idx_local)
    group_idx = out.groupby(session_key).cumcount()

    # --- Core returns/ranges ---
    # ret_1 is canonical here — do not recompute downstream.
    out["ret_1"]    = out[CLOSE_COL].pct_change()
    out["ret_5"]    = out[CLOSE_COL].pct_change(5)
    out["ret_10"]   = out[CLOSE_COL].pct_change(10)
    out["hl_range"] = (out[HIGH_COL] - out[LOW_COL]) / out[CLOSE_COL]
    out["oc_range"] = (out[CLOSE_COL] - out[OPEN_COL]).abs() / out[CLOSE_COL]

    # --- Volume features ---
    vol_mean = out[VOLUME_COL].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean()
    vol_std  = out[VOLUME_COL].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    out["vol_rel"]   = out[VOLUME_COL] / (vol_mean + 1e-12) - 1.0
    out["vol_z"]     = (out[VOLUME_COL] - vol_mean) / (vol_std + 1e-12)
    out["vol_spike"] = (out["vol_z"] > 3.0).astype(float)

    # --- SMA/EMA gaps & slopes ---
    for w in SMA_WINDOWS:
        sma = out[CLOSE_COL].rolling(w, min_periods=w).mean()
        out[f"sma_{w}"] = sma
        out[f"gap_sma_{w}"] = out[CLOSE_COL] / (sma + 1e-12) - 1.0

    for w in EMA_WINDOWS:
        ema = out[CLOSE_COL].ewm(span=w, adjust=False).mean()
        out[f"ema_{w}"] = ema
        out[f"slope_ema_{w}"] = _linreg_slope(ema, max(6, w//2))
    if "ema_20" in out.columns and "ema_50" in out.columns:
        out["above_ema_fast"] = (out[CLOSE_COL] > out["ema_20"]).astype(int)
        out["above_ema_slow"] = (out[CLOSE_COL] > out["ema_50"]).astype(int)
    else:
        out["above_ema_fast"] = 0
        out["above_ema_slow"] = 0

    # --- RSI + RSX (proxy) ---
    out[f"rsi_{RSI_WINDOW}"] = _rsi(out[CLOSE_COL], RSI_WINDOW).fillna(50.0)
    out[f"rsx_{RSI_WINDOW}"] = _rsx_proxy(out[CLOSE_COL], RSI_WINDOW).fillna(50.0)

    # --- Realized vol + regime ---
    out["vol_realized_5"]  = out["ret_1"].rolling(5,  min_periods=5).std()
    out["vol_realized_10"] = out["ret_1"].rolling(10, min_periods=10).std()
    regime = out["ret_1"].rolling(60, min_periods=60).std()
    out["vol_regime"]    = regime.bfill().fillna(0.0)
    out["vol_regime_z"]  = _rolling_z(regime, 120).fillna(0.0)
    # Vectorized rank calculation
    rolling_regime = regime.rolling(240, min_periods=240)
    ranks = rolling_regime.rank(pct=True)
    out["vol_regime_bin"] = (
        pd.cut(ranks, bins=[-np.inf, 1 / 3, 2 / 3, np.inf], labels=[0, 1, 2]).astype("float").fillna(1.0)
    )

    # --- ATR(14) + true range ---
    prev_close = out[CLOSE_COL].shift(1)
    tr1 = (out[HIGH_COL] - out[LOW_COL]).abs()
    tr2 = (out[HIGH_COL] - prev_close).abs()
    tr3 = (out[LOW_COL]  - prev_close).abs()
    out["true_range"] = np.maximum.reduce([tr1, tr2, tr3])
    out["atr_14"]     = out["true_range"].rolling(14, min_periods=14).mean()
    out["rexp_over_atr"] = out["true_range"] / (out["atr_14"] + 1e-12)

    # --- Keltner(EMA20 ± k*ATR14) ---
    ema20 = out[CLOSE_COL].ewm(span=20, adjust=False).mean()
    out["kc_mid_20"]   = ema20
    out["kc_up_20"]    = ema20 + keltner_mult * out["atr_14"]
    out["kc_dn_20"]    = ema20 - keltner_mult * out["atr_14"]

    # --- Bollinger(20, 2σ) ---
    bb_mid = out[CLOSE_COL].rolling(20, min_periods=20).mean()
    bb_std = out[CLOSE_COL].rolling(20, min_periods=20).std()
    out["bb_mid_20"]   = bb_mid
    out["bb_up_20"]    = bb_mid + boll_mult * bb_std
    out["bb_dn_20"]    = bb_mid - boll_mult * bb_std
    out["bb_bw_20"]    = (out["bb_up_20"] - out["bb_dn_20"]) / (bb_mid + 1e-12)

    # --- Session masks ---
    in_rth = session_def.in_rth_mask(idx_local)
    out["in_rth"] = in_rth.astype(float)
    out["is_rth"] = in_rth.astype(int)

    # --- VWAP (session with fallback) ---
    tp = (out[HIGH_COL] + out[LOW_COL] + out[CLOSE_COL]) / 3.0
    pv = tp * out[VOLUME_COL]
    cum_pv_all = pv.groupby(session_key).cumsum()
    cum_vol_all = out[VOLUME_COL].groupby(session_key).cumsum()
    vwap_all = cum_pv_all / (cum_vol_all + 1e-12)
    cum_pv_rth = pv.where(in_rth).groupby(session_key).cumsum()
    cum_vol_rth = out[VOLUME_COL].where(in_rth).groupby(session_key).cumsum()
    vwap_rth = cum_pv_rth / (cum_vol_rth + 1e-12)
    out["vwap_sess"] = vwap_all
    mask_vwap = vwap_rth.notna()
    if mask_vwap.any():
        out.loc[mask_vwap, "vwap_sess"] = vwap_rth.loc[mask_vwap]
    out["slope_vwap_20"] = _linreg_slope(out["vwap_sess"], 20)
    out["above_vwap"] = (out[CLOSE_COL] > out["vwap_sess"]).astype(int)

    # --- VWAP cross & streak (NaN-safe, session-safe) ---
    diff_vwap = (out[CLOSE_COL] - out["vwap_sess"]).astype(float)
    vwap_cross = np.where(diff_vwap > 0.0, 1, np.where(diff_vwap < 0.0, -1, 0)).astype("int8")
    out["vwap_cross"] = vwap_cross

    # run-length streak per session preserving sign; zeros reset
    s = pd.Series(vwap_cross, index=out.index)
    def _signed_streak_array(z: np.ndarray) -> np.ndarray:
        out_arr = np.zeros_like(z, dtype=int)
        cur = 0
        prev = 0
        for i, v in enumerate(z):
            if v == 0:
                cur = 0
                prev = 0
                out_arr[i] = 0
            else:
                sgn = 1 if v > 0 else -1
                if sgn == prev:
                    cur += 1
                else:
                    cur = 1
                    prev = sgn
                out_arr[i] = sgn * cur
        return out_arr

    streak_arr = np.zeros(len(s), dtype=int)
    grouped_indices = s.groupby(session_key).indices
    for positions in grouped_indices.values():
        positions = np.asarray(positions, dtype=int)
        vals = s.to_numpy()[positions]
        streak_arr[positions] = _signed_streak_array(vals)
    out["vwap_cross_streak"] = streak_arr
    out["trend_alignment_long"] = (
        (out["above_vwap"] > 0) & (out["above_ema_fast"] > 0) & (out["above_ema_slow"] > 0)
    ).astype(int)
    out["trend_alignment_short"] = (
        (out["above_vwap"] == 0) & (out["above_ema_fast"] == 0) & (out["above_ema_slow"] == 0)
    ).astype(int)

    # --- Running session high/low & distances (RTH only) ---
    sess_high = out[HIGH_COL].where(in_rth).groupby(session_key).cummax()
    sess_low  = out[LOW_COL].where(in_rth).groupby(session_key).cummin()
    # Allow outside-RTH bars to inherit latest RTH extremes to avoid NaNs in live/pipeline
    sess_high = sess_high.groupby(session_key).ffill()
    sess_low = sess_low.groupby(session_key).ffill()
    out["dist_to_sess_high"] = out[CLOSE_COL] / (sess_high + 1e-12) - 1.0
    out["dist_to_sess_low"]  = out[CLOSE_COL] / (sess_low  + 1e-12) - 1.0

    # --- ORB (RTH or fallback to first N bars per day) ---
    rth_open_aligned = session_def.session_open_datetimes(session_key)
    delta_from_open_min = session_def.minutes_since_open(idx_local, session_index=session_key)
    delta_from_open_min = delta_from_open_min.reindex(out.index)

    if len(idx_local) > 1:
        diffs_sec = np.diff(idx_local.asi8) / 1e9  # seconds
        bar_min = max(1, int(round(np.median(diffs_sec) / 60.0)))
    else:
        bar_min = 5

    _assign_orb_columns(
        out,
        window=int(orb_minutes),
        in_rth=in_rth,
        delta_from_open_min=delta_from_open_min,
        session_key=session_key,
        group_idx=group_idx,
        bar_min=bar_min,
    )

    extra_windows: List[int] = []
    if extra_orb_minutes:
        for value in extra_orb_minutes:
            try:
                win = int(value)
            except (TypeError, ValueError):
                continue
            if win != int(orb_minutes) and win not in extra_windows:
                extra_windows.append(win)
        for win in extra_windows:
            _assign_orb_columns(
                out,
                window=win,
                in_rth=in_rth,
                delta_from_open_min=delta_from_open_min,
                session_key=session_key,
                group_idx=group_idx,
                bar_min=bar_min,
            )
    out["above_orb_high"] = (out[CLOSE_COL] > out[f"orb{orb_minutes}_high"]).astype(float)
    out["below_orb_low"]  = (out[CLOSE_COL] < out[f"orb{orb_minutes}_low"]).astype(float)
    # REDUNDANT FEATURE — mathematically derived from `dist_to_orb_high` / `dist_to_orb_low`.
    # Kept to satisfy frozen model schema (retrain_v2_full).
    # Listed in NEXT_RETRAIN_REMOVALS for exclusion in next training run.
    out["dist_orb_high"]  = out[CLOSE_COL] / (out[f"orb{orb_minutes}_high"] + 1e-12) - 1.0
    out["dist_orb_low"]   = out[CLOSE_COL] / (out[f"orb{orb_minutes}_low"]  + 1e-12) - 1.0

    # --- Time since RTH open (use Series → clip) ---
    mins_from_rth_open = (idx_local - rth_open_aligned) / pd.Timedelta(minutes=1)
    mins_from_rth_open = pd.Series(mins_from_rth_open, index=out.index)
    out["minutes_since_rth_open"] = mins_from_rth_open
    out["mins_from_open"] = mins_from_rth_open.clip(lower=0)

    # --- Breakout follow-through (NO-LOOKAHEAD, tz-safe) ---
    grp = session_key
    above = (out[CLOSE_COL] > out[f"orb{orb_minutes}_high"])
    below = (out[CLOSE_COL] < out[f"orb{orb_minutes}_low"])

    # first break flags within the session (no lookahead)
    first_hi_flag = above & (above.groupby(grp).cumsum() == 1)
    first_lo_flag = below & (below.groupby(grp).cumsum() == 1)

    # tz-aware timestamp series
    ts = pd.Series(out.index, index=out.index, dtype=out.index.dtype)
    hi_start = ts.where(first_hi_flag, pd.NaT)
    lo_start = ts.where(first_lo_flag, pd.NaT)

    first_hi_time = hi_start.groupby(grp).ffill()
    first_lo_time = lo_start.groupby(grp).ffill()

    idx_s = pd.Series(out.index, index=out.index, dtype=out.index.dtype)
    t_hi = (idx_s - first_hi_time).dt.total_seconds() / 60.0
    t_lo = (idx_s - first_lo_time).dt.total_seconds() / 60.0

    out["mins_since_orb_hi_break"] = t_hi.where(t_hi >= 0)
    out["mins_since_orb_lo_break"] = t_lo.where(t_lo >= 0)

    # max excursion from ORB boundary after first break (normalized by ATR)
    max_close_after = out[CLOSE_COL].groupby(grp).cummax()
    min_close_after = out[CLOSE_COL].groupby(grp).cummin()
    out["hi_break_follow"] = (max_close_after - out[f"orb{orb_minutes}_high"]) / (out["atr_14"] + 1e-12)
    out["lo_break_follow"] = (out[f"orb{orb_minutes}_low"] - min_close_after) / (out["atr_14"] + 1e-12)

    # pullback after break (% of excursion)
    peak_after_break   = out[CLOSE_COL].where(above).groupby(grp).cummax()
    trough_after_break = out[CLOSE_COL].where(below).groupby(grp).cummin()
    out["hi_break_pullback"] = (peak_after_break - out[CLOSE_COL]) / (peak_after_break - out[f"orb{orb_minutes}_high"] + 1e-12)
    out["lo_break_pullback"] = (out[CLOSE_COL] - trough_after_break) / (out[f"orb{orb_minutes}_low"] - trough_after_break + 1e-12)

    # --- Streak & momentum over regime vol ---
    up = (out[CLOSE_COL] > out[CLOSE_COL].shift(1)).astype(int)
    dn = (out[CLOSE_COL] < out[CLOSE_COL].shift(1)).astype(int)
    out["streak_up"] = _streaks(up, grp)
    out["streak_dn"] = _streaks(dn, grp)

    out["mom_10"] = out[CLOSE_COL] / out[CLOSE_COL].shift(10) - 1.0
    out["mom10_over_vol"] = out["mom_10"] / (out["vol_regime"] + 1e-12)

    # --- Overnight gap (helps trend days) ---
    session_close_map = out.groupby(session_key)[CLOSE_COL].last()
    prev_close_map = session_close_map.shift(1)
    prev_day_close = session_key.map(prev_close_map)
    today_open = out.groupby(session_key)[OPEN_COL].transform("first")
    out["overnight_gap"] = today_open / (prev_day_close + 1e-12) - 1.0
    
    out["overnight_gap"] = out["overnight_gap"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # --- Time features (one-hot weekday) ---
    out = pd.concat([out, _onehot_dayofweek(out.index)], axis=1)

    # --- Cumulative delta / market-flow ---
    # Prefer real delta/CVD columns when provided by the data source; otherwise fall back
    # to a candle-body proxy so training/live remain functional on OHLCV-only feeds.
    cols_lc = {str(c).strip().lower(): c for c in out.columns}
    cvd_col = next(
        (cols_lc.get(k) for k in ("cvd", "cumulative_delta", "cum_delta", "cumdelta", "cumulative delta")),
        None,
    )
    delta_col = next(
        (cols_lc.get(k) for k in ("delta", "bar_delta", "net_delta", "order_delta", "flow_delta")),
        None,
    )
    cum_vol_sess = out[VOLUME_COL].groupby(grp).cumsum()
    cd_source = None
    if cvd_col is not None:
        cvd = pd.to_numeric(out[cvd_col], errors="coerce")
        out["cd_proxy"] = cvd.groupby(grp).ffill()
        cd_source = f"cvd:{cvd_col}"
    elif delta_col is not None:
        bar_delta = pd.to_numeric(out[delta_col], errors="coerce").fillna(0.0)
        out["cd_proxy"] = bar_delta.groupby(grp).cumsum()
        cd_source = f"delta:{delta_col}"
    else:
        body = (out[CLOSE_COL] - out[OPEN_COL])
        true_rng = (out[HIGH_COL] - out[LOW_COL]).replace(0, np.nan)
        imbalance = (body / true_rng).clip(-1, 1).fillna(0.0) * out[VOLUME_COL]
        out["cd_proxy"] = imbalance.groupby(grp).cumsum()
        cd_source = "proxy"
    out["cd_proxy_norm"] = out["cd_proxy"] / (cum_vol_sess + 1e-12)
    if cd_source and cd_source != "proxy":
        logger.info("Using market-flow source for cd_proxy: %s", cd_source)

    # Avoid pandas block fragmentation before adding large feature groups downstream.
    out = out.copy()

    out = _apply_orb_aoi_features(
        out,
        strategy_config=strategy_config,
        tz=tz,
        tick_size=tick_size,
        orb_minutes=orb_minutes,
        rth_start=rth_start,
        rth_end=rth_end,
    )

    setup_params = SetupParams(orb_minutes=int(orb_minutes))
    setups = compute_es_setups(out, tick_size=tick_size, params=setup_params)
    if not setups.empty:
        out = pd.concat([out, setups], axis=1)

    regime_params = RegimeParams()
    regime_features = compute_es_regime_features(out, params=regime_params)
    regime_frame = pd.concat([out, regime_features], axis=1) if not regime_features.empty else out
    regime_id = assign_es_regime_label(regime_frame, params=regime_params)
    regime_block = regime_features.copy() if not regime_features.empty else pd.DataFrame(index=out.index)
    regime_block["regime_id"] = regime_id
    out = pd.concat([out, regime_block], axis=1)

    if _flow_features_enabled(strategy_config):
        out = add_flow_ohlcv_features(out)


    # --- Final cleanup & lookback cut ---
    out = out.replace([np.inf, -np.inf], np.nan)
    min_lookback = min(max(int(VOL_WINDOW), 60, 20, 14), len(out) // 2)
    out = out.iloc[min_lookback:]
    grp = session_key.loc[out.index]

    # Forward-fill session-level features **within each session** (no cross-day leakage)
    critical = [
        "vwap_sess", f"orb{orb_minutes}_high", f"orb{orb_minutes}_low", "atr_14",
        "kc_mid_20", "bb_mid_20", f"rsi_{RSI_WINDOW}", f"rsx_{RSI_WINDOW}", "vol_regime",
        "dist_to_sess_high", "dist_to_sess_low",
        "mins_since_orb_hi_break", "mins_since_orb_lo_break",
        "hi_break_pullback", "lo_break_pullback",
        "hi_break_follow", "lo_break_follow",
    ]
    for win in extra_windows:
        critical.extend([f"orb{win}_high", f"orb{win}_low"])
    critical_cols = [c for c in critical if c in out.columns]
    if critical_cols:
        out[critical_cols] = out[critical_cols].groupby(grp).ffill()

    # Drop only if all essentials missing
    essentials = [c for c in ["vwap_sess", f"orb{orb_minutes}_high", f"orb{orb_minutes}_low"] if c in out.columns]
    if essentials:
        out = out.dropna(subset=essentials, how="all")

    # Fill remaining NaNs in non-critical derived distance/time features to keep stream live-safe
    fill_zero_cols = [
        "dist_to_sess_high", "dist_to_sess_low",
        "mins_since_orb_hi_break", "mins_since_orb_lo_break",
        "hi_break_pullback", "lo_break_pullback",
        "hi_break_follow", "lo_break_follow",
    ]
    fill_zero_present = [c for c in fill_zero_cols if c in out.columns]
    if fill_zero_present:
        out[fill_zero_present] = out[fill_zero_present].fillna(0.0)

    dupes = out.columns[out.columns.duplicated()].tolist()
    if dupes:
        raise FeaturePipelineError(f"Duplicate columns in feature output: {dupes}")

    missing_model_features = [f for f in MANDATORY_MODEL_FEATURES if f not in out.columns]
    if missing_model_features:
        raise FeaturePipelineError(
            f"Missing {len(missing_model_features)} model-required features: {missing_model_features}"
        )

    missing_safe = [c for c in SAFE_ZERO_FILL_FEATURES if c not in out.columns]
    if missing_safe:
        out = out.assign(**{c: 0.0 for c in missing_safe})

    # Index back to a 'Datetime' column for downstream
    out.index.name = "Datetime"
    out = out.reset_index()

    # Drop non-core raw columns carried from input to avoid NaN drops on unused fields.
    keep_raw = {OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL, "Datetime"}
    drop_raw = [c for c in initial_cols if c in out.columns and c not in keep_raw]
    if drop_raw:
        out = out.drop(columns=drop_raw)

    return _ensure_feature_integrity(out, context="primary_features")


# ---------- main ----------
def build_features(
    df: pd.DataFrame,
    *,
    tz: str = "America/Denver",
    rth_start: str = "07:30",
    rth_end: str = "14:00",
    orb_minutes: int = 15,
    extra_orb_minutes: Optional[Iterable[int]] = None,
    csv_naive_is_utc: bool = True,
    keltner_mult: float = 1.5,
    boll_mult: float = 2.0,
    feature_set: Optional[str] = None,
    tick_size: float = 0.25,
    strategy_config: Optional[Mapping[str, Any]] = None,
    stack_setup_prob: bool = False,
) -> pd.DataFrame:
    if feature_set == "scalp_micro_v1":
        from .scalp_features import build_features_scalp_micro_v1

        base = _build_primary_features(
            df,
            tz=tz,
            rth_start=rth_start,
            rth_end=rth_end,
            orb_minutes=orb_minutes,
            extra_orb_minutes=extra_orb_minutes,
            csv_naive_is_utc=csv_naive_is_utc,
            keltner_mult=keltner_mult,
            boll_mult=boll_mult,
            tick_size=tick_size,
            strategy_config=strategy_config,
        )
        df_aligned = _ensure_dt_index(df, tz, naive_is_utc=csv_naive_is_utc).sort_index()
        micro_source = df_aligned[[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]].rename(columns=str.lower)
        micro = build_features_scalp_micro_v1(micro_source, tick_size=tick_size)
        datetimes = pd.DatetimeIndex(base["Datetime"])
        micro_idx = micro.index
        if getattr(micro_idx, "tz", None) is not None and datetimes.tz is None:
            datetimes = datetimes.tz_localize(micro_idx.tz)
        elif getattr(micro_idx, "tz", None) is None and datetimes.tz is not None:
            datetimes = datetimes.tz_convert(None)
        elif getattr(micro_idx, "tz", None) is not None and datetimes.tz is not None and micro_idx.tz != datetimes.tz:
            datetimes = datetimes.tz_convert(micro_idx.tz)
        micro = micro.reindex(datetimes).add_prefix("micro_")
        micro = micro.reset_index().rename(columns={"index": "Datetime"})
        combined = base.merge(micro, on="Datetime", how="left")
        micro_cols = [c for c in combined.columns if c.startswith("micro_")]
        if micro_cols:
            combined[micro_cols] = combined[micro_cols].fillna(0.0)
        return combined

    return _build_primary_features(
        df,
        tz=tz,
        rth_start=rth_start,
        rth_end=rth_end,
        orb_minutes=orb_minutes,
        extra_orb_minutes=extra_orb_minutes,
        csv_naive_is_utc=csv_naive_is_utc,
        keltner_mult=keltner_mult,
        boll_mult=boll_mult,
        tick_size=tick_size,
        strategy_config=strategy_config,
    )


def _apply_orb_aoi_features(
    frame: pd.DataFrame,
    *,
    strategy_config: Optional[Mapping[str, Any]],
    tz: str,
    tick_size: float,
    orb_minutes: int,
    rth_start: str,
    rth_end: str,
) -> pd.DataFrame:
    merged = _merge_strategy_config(
        strategy_config,
        rth_start=rth_start,
        rth_end=rth_end,
        orb_minutes=orb_minutes,
    )
    if str(merged.get("mode") or "").lower() != "orb_aoi":
        return frame
    merged.setdefault("aoi", {})
    merged["aoi"]["tick_size"] = tick_size
    frame = compute_orb_levels(frame, strategy_config=merged, tz=tz)
    if os.getenv("DEBUG_ASSERTIONS"):
        rth = frame.get("is_rth")
        if rth is not None:
            rth_mask = pd.to_numeric(rth, errors="coerce").fillna(0).astype(bool)
            if bool(rth_mask.any()):
                for col_a, col_b in (
                    ("orb_high", "orb15_high"),
                    ("orb_low", "orb15_low"),
                    ("orb_mid", "orb15_mid"),
                ):
                    if col_a not in frame.columns or col_b not in frame.columns:
                        continue
                    a = pd.to_numeric(frame.loc[rth_mask, col_a], errors="coerce")
                    b = pd.to_numeric(frame.loc[rth_mask, col_b], errors="coerce")
                    max_diff = float((a - b).abs().max()) if len(a) else 0.0
                    if max_diff > 1e-6:
                        raise AssertionError(f"{col_a} vs {col_b} diverge on RTH rows: max_diff={max_diff}")
                if all(c in frame.columns for c in ("orb_high", "orb_low", "orb_mid", "orb15_rng")):
                    hi = pd.to_numeric(frame.loc[rth_mask, "orb_high"], errors="coerce")
                    lo = pd.to_numeric(frame.loc[rth_mask, "orb_low"], errors="coerce")
                    mid = pd.to_numeric(frame.loc[rth_mask, "orb_mid"], errors="coerce")
                    orb_range_raw = hi - lo
                    if "orb_range" in frame.columns:
                        declared_raw = pd.to_numeric(frame.loc[rth_mask, "orb_range"], errors="coerce")
                        max_diff = float((orb_range_raw - declared_raw).abs().max()) if len(orb_range_raw) else 0.0
                        if max_diff > 1e-6:
                            raise AssertionError(f"orb_range (raw) vs (orb_high-orb_low) diverge on RTH rows: max_diff={max_diff}")
                    derived = orb_range_raw / (mid + 1e-12)
                    rng = pd.to_numeric(frame.loc[rth_mask, "orb15_rng"], errors="coerce")
                    max_diff = float((derived - rng).abs().max()) if len(derived) else 0.0
                    if max_diff > 1e-6:
                        raise AssertionError(f"orb_range/mid vs orb15_rng diverge on RTH rows: max_diff={max_diff}")
    frame = compute_session_levels(frame, strategy_config=merged, tz=tz)
    frame = compute_aoi_features(frame, strategy_config=merged, tz=tz, atr_column="atr_14")
    frame = _compute_orb_aoi_direction(frame)
    return frame


def _compute_orb_aoi_direction(frame: pd.DataFrame) -> pd.DataFrame:
    trend_long = pd.to_numeric(frame.get("trend_alignment_long"), errors="coerce").fillna(0)
    trend_short = pd.to_numeric(frame.get("trend_alignment_short"), errors="coerce").fillna(0)
    near_band = pd.to_numeric(frame.get("at_any_aoi_band"), errors="coerce").fillna(0)
    near_upper = pd.to_numeric(frame.get("at_upper_aoi_band"), errors="coerce").fillna(0)
    near_lower = pd.to_numeric(frame.get("at_lower_aoi_band"), errors="coerce").fillna(0)
    nearest_dist = pd.to_numeric(frame.get("nearest_aoi_distance"), errors="coerce").fillna(0.0)
    nearest_type = frame.get("nearest_aoi_type")
    above_orb = pd.to_numeric(frame.get("above_orb_flag"), errors="coerce").fillna(0)
    below_orb = pd.to_numeric(frame.get("below_orb_flag"), errors="coerce").fillna(0)
    inside_orb = pd.to_numeric(frame.get("is_inside_orb"), errors="coerce").fillna(0)

    long_mask = (
        (near_band > 0)
        & (trend_long > 0)
        & ((nearest_dist >= 0) | (near_upper > 0) | (above_orb > 0) | (inside_orb > 0))
    )
    short_mask = (
        (near_band > 0)
        & (trend_short > 0)
        & ((nearest_dist <= 0) | (near_lower > 0) | (below_orb > 0) | (inside_orb > 0))
    )
    direction = np.where(long_mask, 1, np.where(short_mask, -1, 0))
    direction_series = pd.Series(direction, index=frame.index)

    bias = np.where(nearest_dist >= 0, 1, -1)
    bias_series = pd.Series(bias, index=frame.index)
    if isinstance(nearest_type, pd.Series):
        bias_series = bias_series.where(nearest_type.notna(), 0)

    is_after_orb = pd.to_numeric(frame.get("after_orb_flag", 0), errors="coerce").fillna(0).astype(int)
    new_cols = {
        "orb_aoi_long_candidate": long_mask.astype(int),
        "orb_aoi_short_candidate": short_mask.astype(int),
        "orb_aoi_direction": direction_series,
        "orb_aoi_bias": bias_series,
        "orb_aoi_confirmation": (direction_series != 0).astype(int),
        "is_after_orb": is_after_orb,
    }
    return pd.concat([frame, pd.DataFrame(new_cols, index=frame.index)], axis=1)

def _ensure_feature_integrity(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Drop (and log) rows containing NaN/inf features.

    The live engine and training stack assume fully populated numeric features.
    Rather than letting NaNs leak deep into the model, enforce the invariant
    here and emit a concise summary so issues can be debugged upstream.
    """

    if frame.empty:
        return frame

    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return frame

    bad = ~np.isfinite(numeric)
    if bad.any().any():
        bad_cols = {col: int(bad[col].sum()) for col in numeric.columns if bad[col].any()}
        mask = bad.any(axis=1)
        logger.warning(
            "Dropping %d row(s) with NaN/inf features (%s) in %s",
            int(mask.sum()),
            bad_cols,
            context,
        )
        frame = frame.loc[~mask].copy()

    remaining = frame.select_dtypes(include=[np.number])
    if remaining.isna().any().any():
        missing_cols = {
            col: int(remaining[col].isna().sum())
            for col in remaining.columns
            if remaining[col].isna().any()
        }
        raise ValueError(f"Feature frame still contains NaNs after cleanup: {missing_cols}")

    return frame
