# bot/features.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from datetime import time

from .config import (
    OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL,
    VOL_WINDOW, SMA_WINDOWS, EMA_WINDOWS, RSI_WINDOW
)

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
]

_MANDATORY_ORB_TEMPLATES = [
    "orb{orb}_high",
    "orb{orb}_low",
    "orb{orb}_mid",
    "orb{orb}_rng",
]


def mandatory_features(orb_minutes: int = 15) -> List[str]:
    orb_cols = [tpl.format(orb=orb_minutes) for tpl in _MANDATORY_ORB_TEMPLATES]
    return MANDATORY_STATIC_FEATURES + orb_cols


# Backwards-compatible constant (defaults to ORB 15).
MANDATORY_FEATURES = mandatory_features()

# ---------- helpers ----------
def _ensure_dt_index(df: pd.DataFrame, tz: str, *, naive_is_utc: bool = True) -> pd.DataFrame:
    """
    Ensure a tz-aware DatetimeIndex in the requested session tz.

    If your timestamps are tz-naive but actually represent UTC (e.g., the yfinance path
    in market_replay_models wrote tz-naive UTC), set naive_is_utc=True (default).
    If your CSV is already in *local session time* and tz-naive, set naive_is_utc=False.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "Datetime" in out.columns:
            out = out.set_index("Datetime")
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'Datetime' column.")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")

    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        if naive_is_utc:
            idx = idx.tz_localize("UTC").tz_convert(tz).tz_localize(None)
        else:
            idx = pd.DatetimeIndex(idx.to_numpy())
    else:
        idx = idx.tz_convert(tz).tz_localize(None)
    out.index = idx
    if out.index.isna().any():
        out = out[~out.index.isna()]
    # Drop duplicate bars (keep latest) to avoid groupby/reindex errors downstream.
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out

def _local_index(idx: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx)

def _tz_align(dtlike, tz: str) -> pd.DatetimeIndex:
    di = pd.DatetimeIndex(pd.to_datetime(dtlike, errors="coerce"))
    if di.tz is not None:
        di = di.tz_convert(tz).tz_localize(None)
    return di

def _time_in_range(idx_local: pd.DatetimeIndex, start: str, end: str) -> pd.Series:
    """Index-aligned boolean Series for bars whose *local* time is in [start, end)."""
    t = pd.Series(idx_local.time, index=idx_local)
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    start_time = time(sh, sm)
    end_time   = time(eh, em)
    return (t >= start_time) & (t < end_time)

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
    csv_naive_is_utc: bool,
    keltner_mult: float,
    boll_mult: float,
) -> pd.DataFrame:
    out = _ensure_dt_index(df, tz, naive_is_utc=csv_naive_is_utc)
    out = out.sort_index()
    idx_local = _local_index(out.index, tz)
    session_key = pd.Series(idx_local.date, index=out.index, name="session_date")
    required_cols = mandatory_features(orb_minutes)

    # --- Core returns/ranges ---
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

    # --- RSI + RSX (proxy) ---
    out[f"rsi_{RSI_WINDOW}"] = _rsi(out[CLOSE_COL], RSI_WINDOW)
    out[f"rsx_{RSI_WINDOW}"] = _rsx_proxy(out[CLOSE_COL], RSI_WINDOW)

    # --- Realized vol + regime ---
    out["vol_realized_5"]  = out["ret_1"].rolling(5,  min_periods=5).std()
    out["vol_realized_10"] = out["ret_1"].rolling(10, min_periods=10).std()
    regime = out["ret_1"].rolling(60, min_periods=60).std()
    out["vol_regime"]    = regime
    out["vol_regime_z"]  = _rolling_z(regime, 120)
    # Vectorized rank calculation
    rolling_regime = regime.rolling(240, min_periods=240)
    ranks = rolling_regime.rank(pct=True)
    out["vol_regime_bin"] = pd.cut(ranks, bins=[-np.inf, 1/3, 2/3, np.inf], labels=[0, 1, 2]).astype("float")

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
    in_rth = _time_in_range(idx_local, rth_start, rth_end)
    out["in_rth"] = in_rth.astype(float)

    # --- VWAP (session with fallback) ---
    tp = (out[HIGH_COL] + out[LOW_COL] + out[CLOSE_COL]) / 3.0
    pv = tp * out[VOLUME_COL]
    cum_pv  = pv.where(in_rth).groupby(session_key).cumsum()
    cum_vol = out[VOLUME_COL].where(in_rth).groupby(session_key).cumsum()
    out["vwap_sess"] = cum_pv / (cum_vol + 1e-12)
    if out["vwap_sess"].notna().sum() == 0:
        cum_pv  = pv.groupby(session_key).cumsum()
        cum_vol = out[VOLUME_COL].groupby(session_key).cumsum()
        out["vwap_sess"] = cum_pv / (cum_vol + 1e-12)
    out["slope_vwap_20"] = _linreg_slope(out["vwap_sess"], 20)

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

    # --- Running session high/low & distances (RTH only) ---
    sess_high = out[HIGH_COL].where(in_rth).groupby(session_key).cummax()
    sess_low  = out[LOW_COL].where(in_rth).groupby(session_key).cummin()
    out["dist_to_sess_high"] = out[CLOSE_COL] / (sess_high + 1e-12) - 1.0
    out["dist_to_sess_low"]  = out[CLOSE_COL] / (sess_low  + 1e-12) - 1.0

    # --- ORB (RTH or fallback to first N bars per day) ---
    unique_days = pd.Index(session_key.unique(), name="session_date")
    rth_open_daily = pd.to_datetime(unique_days.astype(str) + f" {rth_start}", errors="coerce")
    rth_open_daily = _tz_align(rth_open_daily, tz)
    rth_open_map = pd.Series(rth_open_daily.values, index=unique_days)
    rth_open_aligned = pd.DatetimeIndex(rth_open_map.reindex(session_key.values).values)
    rth_open_aligned = _tz_align(rth_open_aligned, tz)

    # Cast to Series for safe boolean ops
    delta_from_open_min = (idx_local - rth_open_aligned) / pd.Timedelta(minutes=1)
    delta_from_open_min = pd.Series(delta_from_open_min, index=out.index)
    in_orb = (in_rth) & (delta_from_open_min >= 0) & (delta_from_open_min < orb_minutes)

    out[f"orb{orb_minutes}_high"] = out[HIGH_COL].where(in_orb).groupby(session_key).transform("max")
    out[f"orb{orb_minutes}_low"]  = out[LOW_COL].where(in_orb).groupby(session_key).transform("min")
    if out[f"orb{orb_minutes}_high"].notna().sum() == 0:
        # infer bar minutes from median index diff; fallback to 5 if unknown
        if len(idx_local) > 1:
            diffs_sec = np.diff(idx_local.asi8) / 1e9  # seconds
            bar_min = max(1, int(round(np.median(diffs_sec) / 60.0)))
        else:
            bar_min = 5
        group = out.groupby(session_key).cumcount()
        first_n_mask = group < max(1, orb_minutes // max(1, bar_min))
        out[f"orb{orb_minutes}_high"] = out[HIGH_COL].where(first_n_mask).groupby(session_key).transform("max")
        out[f"orb{orb_minutes}_low"]  = out[LOW_COL].where(first_n_mask).groupby(session_key).transform("min")

    out[f"orb{orb_minutes}_mid"]  = (out[f"orb{orb_minutes}_high"] + out[f"orb{orb_minutes}_low"]) / 2.0
    out[f"orb{orb_minutes}_rng"]  = (out[f"orb{orb_minutes}_high"] - out[f"orb{orb_minutes}_low"]) / (out[f"orb{orb_minutes}_mid"] + 1e-12)
    out["above_orb_high"] = (out[CLOSE_COL] > out[f"orb{orb_minutes}_high"]).astype(float)
    out["below_orb_low"]  = (out[CLOSE_COL] < out[f"orb{orb_minutes}_low"]).astype(float)
    out["dist_orb_high"]  = out[CLOSE_COL] / (out[f"orb{orb_minutes}_high"] + 1e-12) - 1.0
    out["dist_orb_low"]   = out[CLOSE_COL] / (out[f"orb{orb_minutes}_low"]  + 1e-12) - 1.0

    # --- Time since RTH open (use Series → clip) ---
    mins_from_rth_open = (idx_local - rth_open_aligned) / pd.Timedelta(minutes=1)
    mins_from_rth_open = pd.Series(mins_from_rth_open, index=out.index)
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
    prev_day_close = out[CLOSE_COL].groupby(grp).transform("last").shift(1)
    today_open = out[OPEN_COL].groupby(grp).transform("first")
    out["overnight_gap"] = today_open / (prev_day_close + 1e-12) - 1.0

    # --- Time features (one-hot weekday) ---
    out = pd.concat([out, _onehot_dayofweek(out.index)], axis=1)

    # --- Cumulative delta proxy ---
    body = (out[CLOSE_COL] - out[OPEN_COL])
    true_rng = (out[HIGH_COL] - out[LOW_COL]).replace(0, np.nan)
    imbalance = (body / true_rng).clip(-1, 1).fillna(0.0) * out[VOLUME_COL]
    out["cd_proxy"] = imbalance.groupby(grp).cumsum()
    cum_vol_sess = out[VOLUME_COL].groupby(grp).cumsum()
    out["cd_proxy_norm"] = out["cd_proxy"] / (cum_vol_sess + 1e-12)

    # --- Final cleanup & lookback cut ---
    out = out.replace([np.inf, -np.inf], np.nan)
    min_lookback = min(max(int(VOL_WINDOW), 60, 20, 14), len(out) // 2)
    out = out.iloc[min_lookback:]
    grp = session_key.loc[out.index]

    # Forward-fill session-level features **within each session** (no cross-day leakage)
    critical = [
        "vwap_sess", f"orb{orb_minutes}_high", f"orb{orb_minutes}_low", "atr_14",
        "kc_mid_20", "bb_mid_20", f"rsi_{RSI_WINDOW}", f"rsx_{RSI_WINDOW}", "vol_regime"
    ]
    grp_values = grp.to_numpy()
    for col in [c for c in critical if c in out.columns]:
        # Forward-fill session-level features within each session.  Use the modern
        # groupby().ffill() API instead of SeriesGroupBy.fillna(method='ffill'),
        # which is deprecated in pandas 2.1+ and will be removed in a future
        # release.  groupby(...)[col].ffill() produces the same behaviour and
        # preserves intra-session isolation.
        out[col] = out.groupby(grp, sort=False)[col].ffill()

    # Drop only if all essentials missing
    essentials = [c for c in ["vwap_sess", f"orb{orb_minutes}_high", f"orb{orb_minutes}_low"] if c in out.columns]
    if essentials:
        out = out.dropna(subset=essentials, how="all")

    for col in required_cols:
        if col not in out.columns:
            out[col] = 0.0

    # Index back to a 'Datetime' column for downstream
    name = out.index.name or "Datetime"
    out = out.reset_index().rename(columns={name: "Datetime"})

    return out


# ---------- main ----------
def build_features(
    df: pd.DataFrame,
    *,
    tz: str = "America/Denver",
    rth_start: str = "07:30",
    rth_end: str = "14:00",
    orb_minutes: int = 15,
    csv_naive_is_utc: bool = True,
    keltner_mult: float = 1.5,
    boll_mult: float = 2.0,
    feature_set: Optional[str] = None,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    if feature_set == "scalp_micro_v1":
        from .scalp_features import build_features_scalp_micro_v1

        base = _build_primary_features(
            df,
            tz=tz,
            rth_start=rth_start,
            rth_end=rth_end,
            orb_minutes=orb_minutes,
            csv_naive_is_utc=csv_naive_is_utc,
            keltner_mult=keltner_mult,
            boll_mult=boll_mult,
        )
        df_aligned = _ensure_dt_index(df, tz, naive_is_utc=csv_naive_is_utc).sort_index()
        micro_source = df_aligned[[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]].rename(columns=str.lower)
        micro = build_features_scalp_micro_v1(micro_source, tick_size=tick_size)
        micro.index = pd.DatetimeIndex(pd.to_datetime(micro.index, errors="coerce"))
        datetimes = pd.DatetimeIndex(base["Datetime"])
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
        csv_naive_is_utc=csv_naive_is_utc,
        keltner_mult=keltner_mult,
        boll_mult=boll_mult,
    )
