#!/usr/bin/env python3
"""
MisterJ Trades — Model Training v2
====================================
Multi-timeframe feature learning with calibrated ensemble for ES futures.

Run:
    python tools/train_v2.py --csv ES6.csv --out models/v2_candidate

Features:
  - 5m + 15m + daily feature stack
  - Volatility-adjusted triple-barrier labels
  - Recency-weighted LightGBM ensemble
  - Isotonic calibration for probability spread
  - Feature importance pruning (top-K selection)
  - Session-aware train/val/test splits

Author: Claw <misterj_trades>   Apr 2026
"""

from __future__ import annotations

import argparse, hashlib, json, logging, os, sys, warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# optional deps — graceful fallback if LightGBM not installed
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    balanced_accuracy_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

log = logging.getLogger("train_v2")

# ===========================================================================
# 1  DATA LOADING & ANCHOR
# ===========================================================================

TIMEZONE = "America/Denver"
RTH_START = "07:30"
RTH_END = "14:00"
EPOCH_OFFSET = pd.Timestamp("2019-01-01").tz_localize(TIMEZONE)


def load_csv(path: str, tz: str = TIMEZONE) -> pd.DataFrame:
    """Load OHLCV CSV; expect columns: Datetime,Open,High,Low,Close,Volume."""
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df.rename(columns=lambda c: c.strip().title(), inplace=True)
    # Normalize mixed-offset timestamps (e.g., DST -07:00/-06:00) via UTC first.
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df.dropna(subset=["Datetime"], inplace=True)
    df.set_index("Datetime", inplace=True)
    df.index = df.index.tz_convert(tz)
    df.sort_index(inplace=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    log.info("Loaded %d bars  %s  →  %s", len(df), df.index[0], df.index[-1])
    return df


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample to higher timeframe, dropping incomplete final bar."""
    ohlc = df.resample(freq).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return ohlc


# ===========================================================================
# 2  MULTI-TIMEFRAME FEATURES
# ===========================================================================

def _rth_mask(idx: pd.DatetimeIndex, start: str = RTH_START, end: str = RTH_END) -> np.ndarray:
    t = idx.tz_localize(None).time
    s = pd.Timestamp(start).time()
    e = pd.Timestamp(end).time()
    return np.array([s <= ti <= e for ti in t], dtype=bool)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, w: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(span=w, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=w, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, w: int = 14) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=w, adjust=False).mean()


def build_features_5m(df: pd.DataFrame) -> pd.DataFrame:
    """Core 5m feature set — ~80 features, no forward look."""
    f = pd.DataFrame(index=df.index)
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]

    # --- returns ---
    for p in [1, 2, 5, 10, 20]:
        f[f"ret_{p}"] = c.pct_change(p).clip(-0.05, 0.05)
        f[f"logret_{p}"] = np.log(c / c.shift(p)).clip(-0.05, 0.05)

    # --- volatility ---
    atr14 = _atr(df, 14)
    f["atr_14"] = atr14
    f["atr_pct"] = atr14 / c * 100
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["true_range"] = tr
    for w in [5, 10, 20]:
        f[f"vol_realized_{w}"] = c.pct_change().rolling(w).std().clip(0, 0.05)
        f[f"range_ratio_{w}"] = tr.rolling(w).mean() / atr14.clip(lower=0.01)
    f["vol_spike"] = tr / tr.rolling(50).mean().clip(lower=0.01)

    # --- momentum ---
    for w in [5, 9, 20, 50]:
        f[f"sma_{w}"] = c.rolling(w).mean()
        f[f"dist_sma_{w}"] = (c - f[f"sma_{w}"]) / f[f"sma_{w}"].clip(lower=0.01)
        f[f"ema_{w}"] = _ema(c, w)
        f[f"slope_ema_{w}"] = _ema(c, w).pct_change(5)
    for w, span in [(9, 26), (12, 26)]:
        f[f"macd_{w}_{span}"] = _ema(c, w) - _ema(c, span)

    # --- RSI/RSX ---
    f["rsi_14"] = _rsi(c, 14)
    for w in [7, 21]:
        f[f"rsi_{w}"] = _rsi(c, w)

    # --- Bollinger Bands ---
    for w in [20]:
        sma = c.rolling(w).mean()
        std = c.rolling(w).std()
        f[f"bb_up_{w}"] = sma + 2 * std
        f[f"bb_dn_{w}"] = sma - 2 * std
        f[f"bb_bw_{w}"] = (f[f"bb_up_{w}"] - f[f"bb_dn_{w}"]) / sma.clip(lower=0.01)
        f[f"bb_pct_{w}"] = (c - f[f"bb_dn_{w}"]) / (f[f"bb_up_{w}"] - f[f"bb_dn_{w}"] + 1e-10).clip(0, 1)

    # --- Keltner ---
    for w in [20]:
        ma = _ema(c, w)
        k_atr = _atr(df, w)
        f[f"kc_up_{w}"] = ma + 1.5 * k_atr
        f[f"kc_dn_{w}"] = ma - 1.5 * k_atr

    # --- range ---
    f["hl_range"] = (h - l) / c * 100
    f["oc_range"] = (c - o).abs() / c * 100
    f["gap"] = (o - c.shift()) / c.shift().clip(lower=0.01) * 100
    f["upper_wick"] = (h - c.abs())
    f["lower_wick"] = (o - l)

    # --- volume ---
    v_ma = v.rolling(20).mean().clip(lower=1)
    f["vol_rel"] = v / v_ma
    f["vol_z"] = (v - v_ma) / v.rolling(20).std().clip(lower=1)
    f["vol_pct_chg"] = v.pct_change(5).clip(-0.8, 0.8)

    # --- streaks ---
    up = (c > c.shift()).astype(int)
    dn = (c < c.shift()).astype(int)
    f["streak_up"] = up.groupby((up == 0).cumsum()).cumsum()
    f["streak_dn"] = dn.groupby((dn == 0).cumsum()).cumsum()
    f["consec_bull"] = ((c > _ema(c, 20)) & (c > c.shift())).astype(int).rolling(5).sum()
    f["consec_bear"] = ((c < _ema(c, 20)) & (c < c.shift())).astype(int).rolling(5).sum()

    # --- RTH / session ---
    in_rth = _rth_mask(df.index)
    f["in_rth"] = in_rth.astype(int)
    f["hours_from_open"] = df.index.hour + df.index.minute / 60 - 6.5  # RTH open = 06:30 ET

    # --- intraday position ---
    f["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    for d in range(5):
        f[f"dow_{d}"] = (df.index.dayofweek == d).astype(int)

    # --- overnight gap ---
    f["overnight_gap"] = (o - c.shift()).abs() / c.shift().clip(lower=0.01) * 100
    f["overnight_dir"] = np.sign(o - c.shift())

    return f


def build_features_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """15m trend features sampled at 5m intervals."""
    df15 = resample_ohlcv(df_5m, "15min")
    f15 = pd.DataFrame(index=df15.index)
    c15 = df15["Close"]
    atr15 = _atr(df15, 14)

    # Slower momentum
    f15["ret_15m_3"] = c15.pct_change(3).clip(-0.05, 0.05)
    f15["ret_15m_6"] = c15.pct_change(6).clip(-0.05, 0.05)
    f15["atr_15m_pct"] = atr15 / c15 * 100
    f15["rsi_15m"] = _rsi(c15, 14)
    f15["trend_strength_15m"] = (c15 - c15.shift(12)).abs() / atr15.clip(lower=0.01)
    f15["macd_15m"] = _ema(c15, 12) - _ema(c15, 26)
    f15["vol_ratio_15m"] = df15["Volume"] / df15["Volume"].rolling(20).mean().clip(lower=1)

    # Reindex to 5m
    f15_5m = f15.reindex(df_5m.index, method="ffill")
    return f15_5m.add_suffix("_tf15")


def build_features_daily(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Daily features sampled at 5m."""
    d1 = resample_ohlcv(df_5m, "1D")
    cd = d1["Close"]
    atrd = _atr(d1, 14)
    fd = pd.DataFrame(index=d1.index)
    for w in [5, 10, 20]:
        fd[f"sma_day_{w}"] = cd.rolling(w).mean()
        fd[f"dist_sma_day_{w}"] = (cd - fd[f"sma_day_{w}"]) / fd[f"sma_day_{w}"].clip(lower=0.01)
    fd["rsi_day"] = _rsi(cd, 14)
    fd["atr_day_pct"] = atrd / cd * 100
    fd["ret_day"] = cd.pct_change().clip(-0.05, 0.05)
    fd["ret_day_5"] = cd.pct_change(5).clip(-0.05, 0.05)
    fd["range_day_pct"] = (d1["High"] - d1["Low"]) / cd * 100
    fd_day = fd.reindex(df_5m.index, method="ffill")
    return fd_day.add_suffix("_day")


# ===========================================================================
# 3  VOLATILITY-ADJUSTED LABELS (Triple Barrier inspired, no lookahead)
# ===========================================================================

def make_labels(
    df: pd.DataFrame, horizon: int = 8, atr_mult: float = 1.0, min_r: float = 0.0
) -> pd.Series:
    """
    Produce ternary label at each bar:
        +1 ⇒ LONG  (price went up enough before it went down enough)
        -1 ⇒ SHORT (price went down enough before it went up enough)
         0 ⇒ FLAT  (neither barrier hit within horizon)
    Uses volatility-adjusted barriers so labels are market-regime independent.
    """
    atr = _atr(df, 14)
    barrier = atr * atr_mult
    c = df["Close"].values
    n = len(c)
    label = np.zeros(n, dtype=int)
    up_hit = np.zeros(n, dtype=bool)
    dn_hit = np.zeros(n, dtype=bool)

    for i in range(n - horizon):
        upper = c[i] + barrier.iloc[i]
        lower = c[i] - barrier.iloc[i]
        for j in range(1, horizon + 1):
            if c[i + j] >= upper:
                up_hit[i] = True
                break
            if c[i + j] <= lower:
                dn_hit[i] = True
                break
    label[up_hit] = 1
    label[dn_hit] = -1
    # Apply minimum return filter
    ret = c / np.roll(c, 1) - 1
    label[np.abs(ret) < min_r] = 0
    label[(up_hit | dn_hit) & (np.abs(ret) < min_r)] = 0
    return pd.Series(label, index=df.index).rename("target")


# ===========================================================================
# 4  RECENCY WEIGHTS
# ===========================================================================

def recency_weights(index: pd.DatetimeIndex, half_life_days: float = 180) -> np.ndarray:
    """Exponential decay weight — recent bars weighted more."""
    last = index[-1]
    delta_days = np.asarray((last - index).days, dtype=np.float64)
    weights = np.exp(-delta_days / float(half_life_days) * np.log(2))
    return weights / float(np.mean(weights))


# ===========================================================================
# 5  TIME-SERIES SPLIT (preserves order, no leaks)
# ===========================================================================

def ts_split(
    X: pd.DataFrame, y: pd.Series, val_pct: float = 0.10, test_pct: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    n = len(X)
    tv = int(n * (1 - test_pct))
    tr = int(tv * (1 - val_pct / (1 - test_pct)))
    return (
        X.iloc[:tr], X.iloc[tr:tv], X.iloc[tv:],
        y.iloc[:tr], y.iloc[tr:tv], y.iloc[tv:],
    )


# ===========================================================================
# 6  MODEL — STACKED ENSEMBLE
# ===========================================================================

def _prune_features(X: pd.DataFrame, model, top_k: int = 80) -> List[str]:
    """Keep top-K features by importance; drop the rest."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return list(X.columns)
    ranked = sorted(zip(X.columns, imp), key=lambda x: -x[1])
    keep = [c for c, _ in ranked[:top_k] if _ > 0]
    if len(keep) < 10:
        return list(X.columns)
    log.info("Pruned %d → %d features", len(X.columns), len(keep))
    return keep


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    calibrate: bool = True,
    use_lgb: bool = True,
    feature_top_k: int = 80,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a stacked ensemble:
      1. Multi-algo base models with tuned params for futures data
      2. Logistic regression meta on validation OOF predictions
      3. Optional isotonic calibration for probability spread
    """
    models: Dict[str, Any] = {}
    oof_val: Dict[str, np.ndarray] = {}

    if use_lgb and HAS_LGB:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=7,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.75,
            colsample_bytree=0.7,
            reg_alpha=0.05,
            reg_lambda=0.5,
            min_split_gain=0.001,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
            sample_weight=sample_weight,
        )
        models["lgbm"] = lgb_model
        oof_val["lgbm"] = lgb_model.predict_proba(X_val)[:, 1]
        log.info("LGBM  val AUC: %.4f", roc_auc_score(y_val, oof_val["lgbm"]))

    xgb_model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=7,
        min_child_weight=10,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.05,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42,
        tree_method="hist",
        eval_metric="logloss",
        use_label_encoder=False,
    )
    try:
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False,
            sample_weight=sample_weight,
        )
    except TypeError:
        # XGBoost API compatibility: some builds remove fit(..., early_stopping_rounds=...).
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            sample_weight=sample_weight,
        )
    models["xgb"] = xgb_model
    best_iteration = getattr(xgb_model, "best_iteration", None)
    if best_iteration is not None:
        oof_val["xgb"] = xgb_model.predict_proba(X_val, iteration_range=(0, int(best_iteration) + 1))[:, 1]
    else:
        oof_val["xgb"] = xgb_model.predict_proba(X_val)[:, 1]
    log.info("XGBoost val AUC: %.4f", roc_auc_score(y_val, oof_val["xgb"]))

    # Feature pruning based on XGBoost importance
    keep = _prune_features(X_train, xgb_model, feature_top_k)

    ext = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=25,
        max_features=0.5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    ext.fit(X_train[keep], y_train, sample_weight=sample_weight)
    models["extra_trees"] = ext
    oof_val["extra_trees"] = ext.predict_proba(X_val[keep])[:, 1]
    log.info("ExtraTrees val AUC: %.4f", roc_auc_score(y_val, oof_val["extra_trees"]))

    # Stacker
    stack_train = np.column_stack([oof_val[k] for k in models])
    stacker = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)
    stacker.fit(stack_train, y_val)
    models["stacker"] = stacker

    ensemble = {"models": models, "keep_features": keep if feature_top_k < 1000 else list(X_train.columns)}

    # Pooled probability
    p_val = stacker.predict_proba(stack_train)[:, 1]
    log.info("Stacker  val AUC: %.4f", roc_auc_score(y_val, p_val))

    # Calibration
    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV as CalCV
        if FrozenEstimator is not None:
            cal = CalCV(FrozenEstimator(stacker), method="isotonic")
        else:
            cal = CalCV(stacker, method="isotonic", cv="prefit")
        cal.fit(stack_train, y_val)
        p_val_cal = cal.predict_proba(stack_train)[:, 1]
        log.info("Calibrated val AUC: %.4f   Brier: %.4f",
                 roc_auc_score(y_val, p_val_cal), brier_score_loss(y_val, p_val_cal))
        ensemble["calibrated_stacker"] = cal

    return ensemble, {"keep_features": keep, "n_base_models": len(models)}


def ensemble_predict(ensemble: Dict, X: pd.DataFrame) -> np.ndarray:
    """Blended prediction from the ensemble."""
    models = ensemble["models"]
    keep = ensemble.get("keep_features", list(X.columns))
    preds = []
    if "lgbm" in models:
        preds.append(models["lgbm"].predict_proba(X)[:, 1])
    if "xgb" in models:
        xgb_model = models["xgb"]
        best_iteration = getattr(xgb_model, "best_iteration", None)
        if best_iteration is not None:
            preds.append(xgb_model.predict_proba(X, iteration_range=(0, int(best_iteration) + 1))[:, 1])
        else:
            preds.append(xgb_model.predict_proba(X)[:, 1])
    if "extra_trees" in models:
        preds.append(models["extra_trees"].predict_proba(X[keep])[:, 1])
    if not preds:
        raise RuntimeError("No base models in ensemble")
    stack_in = np.column_stack(preds)
    if "calibrated_stacker" in ensemble:
        return ensemble["calibrated_stacker"].predict_proba(stack_in)[:, 1]
    return models["stacker"].predict_proba(stack_in)[:, 1]


# ===========================================================================
# 7  METRICS
# ===========================================================================

def compute_metrics(y_true: pd.Series, p: np.ndarray, tag: str = "") -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    m["auc"] = roc_auc_score(y_true, p)
    m["avg_precision"] = average_precision_score(y_true, p)
    m["brier"] = brier_score_loss(y_true, p)
    m["log_loss"] = log_loss(y_true, p)
    m["prob_mean"] = float(np.mean(p))
    m["prob_std"] = float(np.std(p))
    # Probability spread
    bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    hist, _ = np.histogram(p, bins=bins)
    m["prob_hist"] = hist.tolist()
    m["extreme_frac"] = float(np.mean((p < 0.2) | (p > 0.8)))
    # Calibration
    try:
        prob_true, prob_pred = calibration_curve(y_true, p, n_bins=10)
        m["calibration_error"] = float(np.abs(prob_true - prob_pred).max())
    except Exception:
        m["calibration_error"] = None
    # Balanced accuracy at various thresholds
    for thr in [0.5, 0.55, 0.6, 0.7, 0.8]:
        pred = (p > thr).astype(int)
        m[f"bal_acc_{thr:.2f}"] = balanced_accuracy_score(y_true, pred)
        m[f"precision_{thr:.2f}"] = float(
            np.sum((pred == 1) & (y_true == 1)) / max(np.sum(pred == 1), 1)
        )
    log.info("[%s] AUC=%.4f  Brier=%.4f  ExtremeFrac=%.4f  ProbStd=%.4f",
             tag, m["auc"], m["brier"], m["extreme_frac"], m["prob_std"])
    return m


# ===========================================================================
# 8  MAIN PIPELINE
# ===========================================================================

def train_pipeline(
    csv_path: str,
    output_dir: str,
    horizon: int = 8,
    atr_mult: float = 1.0,
    min_r: float = 0.0,
    val_pct: float = 0.10,
    test_pct: float = 0.15,
    calibrate: bool = True,
    half_life_days: float = 0,
    feature_top_k: int = 80,
    use_lgb: bool = True,
    skip_close_model: bool = False,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load & build features ----
    log.info("=" * 60)
    log.info("Loading data from %s", csv_path)
    df = load_csv(csv_path)

    if train_start:
        df = df.loc[pd.Timestamp(train_start).tz_localize(TIMEZONE):]
    if train_end:
        df = df.loc[:pd.Timestamp(train_end).tz_localize(TIMEZONE)]
    log.info("Training window: %s → %s (%d bars)", df.index[0], df.index[-1], len(df))

    log.info("Building 5m features ...")
    f5 = build_features_5m(df)
    log.info("Building 15m features ...")
    f15 = build_features_15m(df)
    log.info("Building daily features ...")
    fd = build_features_daily(df)

    X = pd.concat([f5, f15, fd], axis=1)
    log.info("Total features: %d", X.shape[1])

    # ---- Labels ----
    log.info("Making labels (horizon=%d atr_mult=%.1f) ...", horizon, atr_mult)
    y = make_labels(df, horizon=horizon, atr_mult=atr_mult, min_r=min_r)

    # Drop rows with NaN features or labels
    valid = y.notna() & (X.notna().all(axis=1))
    X = X.loc[valid]
    y = y.loc[valid]

    # Directional binary: drop flats, map LONG=1, SHORT=0
    dir_mask = y != 0
    X_dir = X.loc[dir_mask]
    y_dir = (y.loc[dir_mask] == 1).astype(int)

    label_counts = {"long": int((y == 1).sum()), "short": int((y == -1).sum()), "flat": int((y == 0).sum())}
    log.info("Labels: %s", label_counts)

    # ---- Recency weights ----
    if half_life_days > 0:
        w = recency_weights(X_dir.index, half_life_days)
        log.info("Recency weighting: half-life=%dd  weight range=[%.2f, %.2f]",
                 half_life_days, w.min(), w.max())
    else:
        w = None

    # ---- Split ----
    X_tr, X_va, X_te, y_tr, y_va, y_te = ts_split(X_dir, y_dir, val_pct, test_pct)
    log.info("Split:  train=%d  val=%d  test=%d", len(X_tr), len(X_va), len(X_te))

    # ---- Scale ----
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
    X_va_s = pd.DataFrame(scaler.transform(X_va), index=X_va.index, columns=X_va.columns)
    X_te_s = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)

    # ---- Train ----
    log.info("Training ensemble ...")
    ensemble, info = train_ensemble(
        X_tr_s, y_tr, X_va_s, y_va,
        sample_weight=w[:len(X_tr)] if w is not None else None,
        calibrate=calibrate,
        use_lgb=use_lgb and HAS_LGB,
        feature_top_k=feature_top_k,
    )

    # ---- Evaluate ----
    p_tr = ensemble_predict(ensemble, X_tr_s)
    p_va = ensemble_predict(ensemble, X_va_s)
    p_te = ensemble_predict(ensemble, X_te_s)

    metrics = {
        "train": compute_metrics(y_tr, p_tr, "TRAIN"),
        "val": compute_metrics(y_va, p_va, "VAL"),
        "test": compute_metrics(y_te, p_te, "TEST"),
        "label_counts": label_counts,
        "label_config": {"horizon": horizon, "atr_mult": atr_mult, "min_r": min_r},
        "n_features": X.shape[1],
        "n_features_after_prune": len(info["keep_features"]),
    }

    # ---- Save artifacts ----
    run_id = hashlib.sha256(
        f"{csv_path}:{horizon}:{atr_mult}:{datetime.now()}".encode()
    ).hexdigest()[:12]
    generated_at = datetime.now(timezone.utc).isoformat()

    model_path = out / f"direction_v2_{run_id}.joblib"
    joblib.dump({
        "ensemble": ensemble,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "keep_features": info["keep_features"],
        "config": {
            "horizon": horizon,
            "atr_mult": atr_mult,
            "min_r": min_r,
            "half_life_days": half_life_days,
            "feature_top_k": feature_top_k,
            "calibrate": calibrate,
        },
        "metrics": metrics,
        "run_id": run_id,
        "generated_at": generated_at,
    }, model_path)
    log.info("Model saved: %s", model_path)

    meta_path = out / f"direction_v2_{run_id}.meta.json"
    meta = {
        "run_id": run_id,
        "csv": csv_path,
        "model": "direction_v2_ensemble",
        "timeframe": "5m",
        "feature_sources": ["5m", "15m", "daily"],
        "label_type": "vol_adjusted_triple_barrier",
        "metrics": metrics,
        "config": {
            "horizon": horizon,
            "atr_mult": atr_mult,
            "min_r": min_r,
            "val_pct": val_pct,
            "test_pct": test_pct,
            "half_life_days": half_life_days,
            "feature_top_k": feature_top_k,
            "calibration": "isotonic" if calibrate else "none",
        },
        "model_path": str(model_path),
        "generated_at": generated_at,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Meta saved: %s", meta_path)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        m = metrics[split]
        print(f"  {split.upper():6s}  AUC={m['auc']:.4f}  Brier={m['brier']:.4f}  "
              f"ExtremeFrac={m['extreme_frac']:.4f}  ProbStd={m['prob_std']:.4f}  "
              f"BalAcc@0.6={m.get('bal_acc_0.60', 0):.4f}")
    print(f"\n  Labels: {label_counts}")
    print(f"  Features: {X.shape[1]} total, {len(info['keep_features'])} after pruning")
    print(f"  Run ID:  {run_id}")
    print(f"  Model:   {model_path}")
    print("=" * 60)

    return meta


# ===========================================================================
# 9  CLI
# ===========================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MisterJ Trades — v2 Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/train_v2.py --csv data/es_5m_rth_sample.csv -o models/v2 --horizon 10 --atr 1.5
  python tools/train_v2.py --csv ../data/intraday/es/ES6.csv -o models/v2 --full --no-calibrate
  python tools/train_v2.py --csv ES6.csv -o models/v2 --top-k 60 --half-life 120
        """,
    )
    p.add_argument("--csv", required=True, help="Path to ES 5m OHLCV CSV")
    p.add_argument("-o", "--output", default="./v2_models", help="Output directory for models")
    p.add_argument("--horizon", type=int, default=8, help="Label horizon in bars (default: 8)")
    p.add_argument("--atr", "--atr-mult", dest="atr_mult", type=float, default=1.0,
                   help="ATR multiplier for barriers (default: 1.0)")
    p.add_argument("--min-r", type=float, default=0.0,
                   help="Minimum return threshold for labels (default: 0.0)")
    p.add_argument("--val-pct", type=float, default=0.10, help="Validation split ratio")
    p.add_argument("--test-pct", type=float, default=0.15, help="Test split ratio")
    p.add_argument("--half-life", type=float, default=180,
                   help="Recency half-life in days. 0 = uniform (default: 180)")
    p.add_argument("--top-k", type=int, default=80,
                   help="Keep top-K features by importance (default: 80). 0 = keep all.")
    p.add_argument("--no-lgb", action="store_true", help="Skip LightGBM (use XGBoost+ExtraTrees only)")
    p.add_argument("--no-calibrate", action="store_true", help="Skip isotonic calibration")
    p.add_argument("--train-start", help="Earliest date for training (YYYY-MM-DD)")
    p.add_argument("--train-end", help="Latest date for training (YYYY-MM-DD)")
    p.add_argument("--full", action="store_true",
                   help="Convenience: use 2021-01-01 to 2024-12-31 training window")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.full:
        args.train_start = "2021-01-01"
        args.train_end = "2024-12-31"

    if args.top_k == 0:
        args.top_k = 10000  # effectively keep all

    train_pipeline(
        csv_path=args.csv,
        output_dir=args.output,
        horizon=args.horizon,
        atr_mult=args.atr_mult,
        min_r=args.min_r,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        calibrate=not args.no_calibrate,
        half_life_days=args.half_life if args.half_life > 0 else 0,
        feature_top_k=args.top_k,
        use_lgb=not args.no_lgb,
        train_start=args.train_start,
        train_end=args.train_end,
    )


if __name__ == "__main__":
    main()
