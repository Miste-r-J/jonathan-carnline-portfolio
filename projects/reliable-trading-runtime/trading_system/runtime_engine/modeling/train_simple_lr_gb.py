#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Advanced, leak-safe time-series classifier trainer for OHLCV data.

Highlights
----------
- Strict time-ordered splits (train → val → test)
- No information leakage (never calibrate on test)
- Robust feature sanitation (drop toxic columns, impute remaining NaNs)
- XGBoost (GPU/CPU) or sklearn baselines (LogReg / GB / HistGB)
- Validation-only calibration (XGB) or TimeSeriesSplit calibration (sklearn)
- Sidecar *.features.json saved next to model to lock downstream alignment
- Optional OOS prediction CSV with datetime alignment

Usage (examples)
----------------
python -m train_ts_classifier \
  --csv data/intraday/ESF/5m/esf5m.csv --model xgb \
  --tz America/Denver --rth-start 07:30 --rth-end 12:00 --orb-minutes 15 \
  --horizon 5 --test-ratio 0.2 --val-ratio 0.1 \
  --save-model artifacts/es_xgb_orb15.joblib --gpu
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Try to import your local feature builder (recommended) ---
try:
    from .features import build_features, mandatory_features, MANDATORY_FEATURES  # type: ignore
except Exception:  # fallback to same dir
    from features import build_features, mandatory_features, MANDATORY_FEATURES  # type: ignore

# Optional config with column names
try:
    from .config import OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL  # type: ignore
except Exception:
    OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL = "Open", "High", "Low", "Close", "Volume"


# =============================================================================
# IO / Column normalization
# =============================================================================

def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust column normalizer:
    - Parses a datetime-like column into 'Datetime'
    - Renames common OHLCV variants to configured names
    - Coerces to numeric & drops NaNs/duplicates
    """
    rename_map: dict[str, str] = {}
    dt_candidates: List[str] = []
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("datetime", "date", "time", "timestamp"):
            dt_candidates.append(c)
        elif "open" in lc:
            rename_map[c] = OPEN_COL
        elif "high" in lc:
            rename_map[c] = HIGH_COL
        elif "low" in lc:
            rename_map[c] = LOW_COL
        elif "close" in lc:
            rename_map[c] = CLOSE_COL
        elif "volume" in lc:
            rename_map[c] = VOLUME_COL

    if rename_map:
        df = df.rename(columns=rename_map)

    if not dt_candidates:
        raise ValueError("Cannot find a Datetime column (datetime/date/time/timestamp).")

    # Pick the datetime-like column with the most valid parses
    best, best_valid = None, -1
    for col in dt_candidates:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
        valid = int(parsed.notna().sum())
        if valid > best_valid:
            best, best_valid = parsed, valid
    assert best is not None
    df["Datetime"] = best
    df = df.dropna(subset=["Datetime"]).copy()

    # Coerce OHLCV numerics
    for col_name in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    df = df.dropna(subset=[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL])

    # De-duplicate duplicate column names
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df.reset_index(drop=True)


# =============================================================================
# Labels
# =============================================================================

def make_up_label(feats: pd.DataFrame, horizon: int, close_col: str) -> Tuple[pd.Series, pd.Series]:
    """Binary 'up' label at a fixed horizon using a 5 bps threshold (~2 ES ticks)."""
    close = feats[close_col]
    fwd_close = close.shift(-horizon)
    threshold = 0.0005
    y = ((fwd_close - close) / close > threshold).astype(int)
    mask = y.notna()
    return y[mask], close[mask]


# =============================================================================
# Feature filtering / sanitation
# =============================================================================

def _stationarity_filter(
    X: pd.DataFrame,
    *,
    keep_all: bool = False,
    mandatory_set: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Drop level features that can leak raw price levels; keep ORB/VWAP/distances.

    When keep_all=True, return the frame unchanged so downstream models can
    decide how to handle raw price-level features.
    """
    if keep_all:
        return X

    if mandatory_set is None:
        mandatory_set = set(MANDATORY_FEATURES)

    BAD_EXACT = {"Open", "High", "Low", "Close", "Volume", "kc_mid_20", "bb_mid_20"}
    BAD_PREFIX = ("sma_", "ema_", "kc_mid_", "bb_mid_")
    cols = []
    for c in X.columns:
        if c in mandatory_set:
            cols.append(c)
            continue
        if c in BAD_EXACT:
            continue
        if any(c.startswith(p) for p in BAD_PREFIX):
            continue
        cols.append(c)
    return X[cols]


def _sanitize_features(
    X: pd.DataFrame,
    *,
    max_nan_col_frac: float = 0.40,
    keep_all: bool = False,
    mandatory_set: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Column-wise sanitation:
      1) drop constant / all-NaN columns
      2) drop columns with too many NaNs (e.g., session-only features)
      3) ffill / bfill remaining NaNs then fill residual with 0.0
    This preserves rows while removing toxic columns.
    """
    X = X.copy()

    if keep_all:
        X = X.ffill().bfill().fillna(0.0)
        return X.astype(float)

    if mandatory_set is None:
        mandatory_set = set(MANDATORY_FEATURES)

    # Drop constant or all-NaN
    nunique = X.nunique(dropna=True)
    keep_cols = [c for c in X.columns if nunique.get(c, 0) > 1 or c in mandatory_set]
    X = X[keep_cols]

    # Drop columns with excessive NaNs
    nan_frac = X.isna().mean()
    good_cols = [c for c in X.columns if nan_frac.get(c, 0.0) <= max_nan_col_frac or c in mandatory_set]
    X = X[good_cols]

def _strict_stationary_filter(
    X: pd.DataFrame,
    mandatory_set: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Legacy alias; keep stationarity filter behavior explicit."""
    return _stationarity_filter(X, mandatory_set=mandatory_set)

    # Impute remaining NaNs conservatively (within-day features usually okay with ffill/bfill)
    X = X.ffill().bfill().fillna(0.0)

    # Coerce to float
    return X.astype(float)


def _required_orb_columns(orb_minutes: int) -> list[str]:
    pfx = f"orb{orb_minutes}_"
    return ["vwap_sess", pfx + "high", pfx + "low", pfx + "mid", pfx + "rng", "dist_orb_high", "dist_orb_low"]


# =============================================================================
# Splitting / checks
# =============================================================================

def time_splits(n: int, test_ratio: float = 0.2, val_ratio: float = 0.1) -> tuple[range, range, range]:
    """Return (train_idx, val_idx, test_idx) ranges for length n."""
    n_test = max(64, int(round(n * test_ratio)))
    n_val = max(64, int(round(n * val_ratio)))
    n_train = n - n_val - n_test
    if n_train < 64:
        n_val = max(0, min(n_val, max(0, n - n_test - 64)))
        n_train = n - n_val - n_test
        if n_train < 64:
            raise ValueError("Not enough samples for time-aware splits. Provide more data.")
    return range(0, n_train), range(n_train, n_train + n_val), range(n_train + n_val, n)


def _ensure_both_classes(y: pd.Series) -> bool:
    vals = set(map(int, pd.unique(y.dropna())))
    return (0 in vals) and (1 in vals)


# =============================================================================
# Model factories
# =============================================================================

def make_baseline(model_type: str, class_weight: Optional[str], random_state: int):
    mtype = model_type.lower()
    if mtype in {"logit", "logistic", "lr"}:
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight=None if class_weight in (None, "None") else class_weight,
                solver="lbfgs",
                random_state=random_state
            )),
        ])
    if mtype in {"gb", "gboost", "gbc"}:
        return GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=random_state
        )
    if mtype in {"hgb", "histgb", "hist"}:
        return HistGradientBoostingClassifier(
            max_depth=6, max_iter=600, learning_rate=0.05,
            l2_regularization=1e-3, early_stopping=True,
            random_state=random_state
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def make_xgb(params: dict, use_gpu: bool):
    # Local import to keep optional
    from xgboost import XGBClassifier
    p = dict(
        n_estimators=params.get("n_estimators", 800),
        learning_rate=params.get("learning_rate", 0.03),
        max_depth=params.get("max_depth", 6),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        max_bin=params.get("max_bin", 256),
        min_child_weight=params.get("min_child_weight", 1.0),
        gamma=params.get("gamma", 0.0),
        tree_method="gpu_hist" if use_gpu else "hist",
        predictor="gpu_predictor" if use_gpu else "auto",
        eval_metric="logloss",
        enable_categorical=False,
        n_jobs=-1,
        random_state=params.get("random_state", 42),
        verbosity=0,
        base_score=float(params.get("base_score", 0.5)),  # ensure valid in (0,1)
    )
    return XGBClassifier(**p)


# =============================================================================
# Utilities
# =============================================================================

def _compute_scale_pos_weight(y: pd.Series) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return max(1.0, (neg / max(1.0, pos))) if pos > 0 else 1.0


def _get_datetime_series(feats: pd.DataFrame, X_index) -> pd.Series:
    """Return a Datetime series aligned with X_index whether Datetime is index or column."""
    if isinstance(feats.index, pd.DatetimeIndex):
        return pd.Series(feats.index, index=feats.index).loc[X_index]
    if "Datetime" in feats.columns:
        return pd.to_datetime(feats.loc[X_index, "Datetime"])
    # Fallback: echo index (may be integers)
    return pd.to_datetime(pd.Index(X_index), errors="coerce")


# =============================================================================
# Training
# =============================================================================

def train_and_eval(
    df_raw: pd.DataFrame,
    tz: str = "America/Denver",
    rth_start: str = "07:30",
    rth_end: str = "14:00",
    orb_minutes: int = 15,
    horizon: int = 5,
    model_type: str = "hgb",  # "xgb" | "logit" | "gb" | "hgb"
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    class_weight: Optional[str] = None,
    random_state: int = 42,
    use_gpu: bool = False,
    xgb_params: Optional[dict] = None,
    save_model_path: Optional[str] = None,
) -> tuple[CalibratedClassifierCV | object, dict]:
    """
    Train model with time-aware splits & calibration. Returns (fitted_model, metrics).
    XGB: early stopping on validation, calibration on validation (no test leakage).
    Sklearn: CalibratedClassifierCV(TimeSeriesSplit) on train+val (still time-ordered).
    """
    # 1) Build features
    feats = build_features(
        df_raw,
        tz=tz,
        rth_start=rth_start,
        rth_end=rth_end,
        orb_minutes=orb_minutes,
    )
    mandatory_set = set(mandatory_features(orb_minutes))

    # 2) Label
    y, _ = make_up_label(feats, horizon, CLOSE_COL)

    # 3) Numeric features + filters
    X_all = feats.loc[y.index].select_dtypes(include=[np.number, "bool"]).astype(float)
    X_all = _stationarity_filter(X_all, mandatory_set=mandatory_set)
    X_all = _sanitize_features(X_all, max_nan_col_frac=0.40, mandatory_set=mandatory_set)

    # 4) Presence check for ORB/VWAP (warn only)
    missing = [c for c in _required_orb_columns(orb_minutes) if c not in X_all.columns]
    if missing:
        print("[WARN] Missing ORB/VWAP features:", missing)

    # 5) Time-aware split
    n = len(X_all)
    tr, va, te = time_splits(n, test_ratio=test_ratio, val_ratio=val_ratio)
    X_tr, y_tr = X_all.iloc[list(tr)], y.iloc[list(tr)]
    X_va, y_va = X_all.iloc[list(va)], y.iloc[list(va)]
    X_te, y_te = X_all.iloc[list(te)], y.iloc[list(te)]

    # 6) Ensure both classes in train; borrow earliest val samples if needed
    if not _ensure_both_classes(y_tr) and len(y_va) > 0:
        take = 0
        while take < len(y_va) and not _ensure_both_classes(pd.concat([y_tr, y_va.iloc[:take + 1]])):
            take += 1
        if take > 0:
            X_tr = pd.concat([X_tr, X_va.iloc[:take]])
            y_tr = pd.concat([y_tr, y_va.iloc[:take]])
            X_va = X_va.iloc[take:]
            y_va = y_va.iloc[take:]

    if not _ensure_both_classes(y_tr):
        raise ValueError(
            "Training split has only one class after sanitation. "
            "Try a smaller horizon or a smaller label threshold."
        )

    model_name = model_type.lower()
    calibrated_model: CalibratedClassifierCV | object

    if model_name == "xgb":
        params = dict(xgb_params or {})
        params.setdefault("scale_pos_weight", _compute_scale_pos_weight(y_tr))
        xgb = make_xgb(params, use_gpu)

        # Early stopping on validation if non-empty
        fit_kwargs = {}
        if len(X_va) > 0 and _ensure_both_classes(y_va):
            fit_kwargs.update(dict(eval_set=[(X_va, y_va)], early_stopping_rounds=50, verbose=False))

        xgb.fit(X_tr, y_tr, **fit_kwargs)

        # Calibration
        if len(X_va) > 0 and _ensure_both_classes(y_va):
            calibrated_model = CalibratedClassifierCV(xgb, method="isotonic", cv="prefit")
            calibrated_model.fit(X_va, y_va)
        else:
            # Fallback calibration on train with time CV if enough data, else use raw model
            if len(X_tr) >= 200 and _ensure_both_classes(y_tr):
                cvcv = TimeSeriesSplit(n_splits=3)
                calibrated_model = CalibratedClassifierCV(xgb, method="sigmoid", cv=cvcv)
                calibrated_model.fit(X_tr, y_tr)
            else:
                calibrated_model = xgb

        proba_te = calibrated_model.predict_proba(X_te)[:, 1]

    else:
        base = make_baseline(model_name, class_weight, random_state)

        trva_X = pd.concat([X_tr, X_va]) if len(X_va) > 0 else X_tr
        trva_y = pd.concat([y_tr, y_va]) if len(y_va) > 0 else y_tr

        # If too small or degenerate for calibration, fit base directly
        if len(trva_X) < 50 or trva_y.nunique() < 2:
            base.fit(X_tr, y_tr)
            calibrated_model = base
        else:
            # Choose splits proportional to data size
            max_splits = max(2, min(5, len(trva_X) // 200))
            cvcv = TimeSeriesSplit(n_splits=max_splits)
            calibrated_model = CalibratedClassifierCV(base, method="isotonic", cv=cvcv)
            calibrated_model.fit(trva_X, trva_y)

        proba_te = calibrated_model.predict_proba(X_te)[:, 1]

    # Metrics
    try:
        roc = roc_auc_score(y_te, proba_te)
    except Exception:
        roc = float("nan")
    try:
        ap = average_precision_score(y_te, proba_te)
    except Exception:
        ap = float("nan")
    ll = log_loss(y_te, np.clip(proba_te, 1e-6, 1 - 1e-6))
    bs = brier_score_loss(y_te, proba_te)

    metrics = {
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "roc_auc": float(roc),
        "avg_precision": float(ap),
        "log_loss": float(ll),
        "brier": float(bs),
        "feature_count": int(X_all.shape[1]),
        "model_type": model_name,
        "horizon": int(horizon),
        "orb_minutes": int(orb_minutes),
        "scale_pos_weight": float(xgb_params.get("scale_pos_weight")) if (model_name == "xgb" and xgb_params and "scale_pos_weight" in xgb_params) else (
            float(_compute_scale_pos_weight(y_tr)) if model_name == "xgb" else None
        ),
    }

    print(json.dumps(metrics, indent=2))

    # Persist model + exact features
    if save_model_path:
        outp = Path(save_model_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        try:
            setattr(calibrated_model, "feature_names_in_", np.array(list(X_all.columns)))
        except Exception:
            pass
        joblib.dump(calibrated_model, outp)
        sidecar = outp.with_suffix(".features.json")
        sidecar.write_text(json.dumps({"features": list(X_all.columns)}, indent=2))
        print(f"[OK] Saved model → {outp}")
        print(f"[OK] Saved feature list → {sidecar}")

    return calibrated_model, metrics


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Train time-series classifier with calibration (advanced, leak-safe)")
    ap.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    ap.add_argument("--save-model", default=None, help="Path to save trained model (.joblib)")
    ap.add_argument("--save-preds", default=None, help="Optional CSV for OOS predictions")

    ap.add_argument("--tz", default="America/Denver")
    ap.add_argument("--rth-start", default="07:30")
    ap.add_argument("--rth-end", default="14:00")
    ap.add_argument("--orb-minutes", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=5)

    ap.add_argument("--model", default="xgb", choices=["xgb", "logit", "gb", "hgb"])
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--class-weight", default=None)
    ap.add_argument("--gpu", action="store_true")

    # XGB hyperparams
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-alpha", type=float, default=0.1)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument("--random-state", type=int, default=42)

    args = ap.parse_args()

    df = pd.read_csv(Path(args.csv))
    df = _normalize_ohlcv_columns(df)

    xgb_params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        max_bin=args.max_bin,
        random_state=args.random_state,
    )

    model, metrics = train_and_eval(
        df_raw=df,
        tz=args.tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_minutes=args.orb_minutes,
        horizon=args.horizon,
        model_type=args.model,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        class_weight=args.class_weight,
        random_state=args.random_state,
        use_gpu=args.gpu,
        xgb_params=xgb_params,
        save_model_path=args.save_model,
    )

    if args.save_preds:
        # Recompute OOS proba for saving (same split logic)
        feats = build_features(df, tz=args.tz, rth_start=args.rth_start, rth_end=args.rth_end, orb_minutes=args.orb_minutes)
        y, _ = make_up_label(feats, args.horizon, CLOSE_COL)
        X_all = feats.loc[y.index].select_dtypes(include=[np.number, "bool"]).astype(float)
        mandatory_set = set(mandatory_features(args.orb_minutes))
        X_all = _stationarity_filter(X_all, mandatory_set=mandatory_set)
        X_all = _sanitize_features(X_all, max_nan_col_frac=0.40, mandatory_set=mandatory_set)

        n = len(X_all)
        tr, va, te = time_splits(n, test_ratio=args.test_ratio, val_ratio=args.val_ratio)
        X_te = X_all.iloc[list(te)]
        y_te = y.iloc[list(te)]

        # Datetime alignment regardless of index/column placement
        dt_series = _get_datetime_series(feats.loc[y.index], X_te.index)

        p_te = model.predict_proba(X_te)[:, 1]
        out = pd.DataFrame({
            "Datetime": pd.to_datetime(dt_series.values),
            "y_true": y_te.astype(int).values,
            "p_up": p_te
        }).sort_values("Datetime").reset_index(drop=True)

        outp = Path(args.save_preds)
        outp.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(outp, index=False)
        print(f"[OK] Saved predictions → {outp}")


if __name__ == "__main__":
    main()
