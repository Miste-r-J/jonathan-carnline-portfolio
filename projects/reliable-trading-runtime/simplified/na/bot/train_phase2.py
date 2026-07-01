from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBClassifier
try:
    from scipy.stats import ks_2samp  # type: ignore
except Exception:  # pragma: no cover - scipy may be unavailable
    ks_2samp = None  # type: ignore

try:  # sklearn 1.5+
    from sklearn.frozen import FrozenEstimator  # type: ignore
except Exception:  # pragma: no cover - fallback for older sklearn
    FrozenEstimator = None  # type: ignore

from .features import build_features
from .features_multi_tf import build_multi_tf_features
from .config import instrument_by_alias, ENGINE
from .phase2_sim import Phase2DecisionPolicy, Phase2SimConfig, phase2_decisions, simulate_trades
from .train import (
    _apply_event_filter,
    _index_bounds,
    _label_config_from_app,
    _safe_metric,
    _three_way_time_split,
    _training_params_dict,
    check_no_label_leakage,
)
from .train_simple_lr_gb import _required_orb_columns, _strict_stationary_filter
from .labeling import make_labels
from ..config.loader import load_app_config
from ..common.label_schema import LabelSchema
from ..common.ids import generate_model_id
from ..market.sessions import ensure_dataframe_index
from .feature_hash import compute_feature_hash
from .feature_constants import MANDATORY_MODEL_FEATURES


def _predict_long_probabilities(model_path: Path, feature_frame: pd.DataFrame) -> np.ndarray:
    model = joblib.load(model_path)
    feature_order = _feature_order_from_sidecar(model_path)
    X = feature_frame.select_dtypes(include=[np.number, "bool"]).astype(float).copy()
    if feature_order:
        for col in feature_order:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_order]
    raw = model.predict_proba(X)
    probs = np.asarray(raw, dtype=float)
    if probs.ndim == 2:
        classes = getattr(model, "classes_", None)
        idx = 1
        if classes is not None:
            try:
                idx = list(classes).index(1)
            except Exception:
                try:
                    idx = list(classes).index("LONG")
                except Exception:
                    idx = 1
        probs = probs[:, idx]
    probs = np.clip(probs, 0.0, 1.0)
    return probs


logger = logging.getLogger(__name__)

SETUP_MODEL_DEFAULT = Path("artifacts/es_xgb_setup_5m.joblib")
DIR_MODEL_DEFAULT = Path("artifacts/es_xgb_dir_5m.joblib")
CLOSE_MODEL_DEFAULT = Path("artifacts/es_xgb_close_5m.joblib")


@dataclass
class Phase2LabelBundle:
    setup: pd.Series
    direction: pd.Series
    direction_info: Dict[str, Any]
    trimmed_frame: pd.DataFrame


@dataclass
class Phase2SplitIndices:
    train_index: pd.Index
    val_index: pd.Index
    test_index: pd.Index


def _feature_order_from_sidecar(model_path: Path) -> Optional[List[str]]:
    sidecar = model_path.with_suffix(".features.json")
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text())
    except Exception:
        return None
    if isinstance(payload, dict):
        feats = payload.get("features")
        if isinstance(feats, list):
            return [str(f) for f in feats]
    if isinstance(payload, list):
        return [str(f) for f in payload]
    return None


def _datetime_series(frame: pd.DataFrame, tz: str) -> pd.Series:
    if "Datetime" not in frame.columns:
        raise ValueError("Feature frame missing Datetime column; cannot apply time-based splits.")
    series = pd.to_datetime(frame["Datetime"], errors="coerce")
    if getattr(series.dt, "tz", None) is None:
        series = series.dt.tz_localize(tz)
    else:
        series = series.dt.tz_convert(tz)
    return series


def _parse_time_bound(value: Optional[str], tz: str, *, is_end: bool) -> Tuple[Optional[pd.Timestamp], bool]:
    if not value:
        return None, False
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid datetime bound: {value}")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(tz)
    else:
        parsed = parsed.tz_convert(tz)
    has_clock = bool(re.search(r"\d{2}:\d{2}", str(value)))
    if is_end and not has_clock:
        return parsed + pd.Timedelta(days=1), True
    return parsed, False


def _time_window_mask(
    dt_series: pd.Series,
    *,
    start: Optional[str],
    end: Optional[str],
    tz: str,
) -> pd.Series:
    start_ts, _ = _parse_time_bound(start, tz, is_end=False)
    end_ts, end_exclusive = _parse_time_bound(end, tz, is_end=True)
    mask = pd.Series(True, index=dt_series.index)
    if start_ts is not None:
        mask &= dt_series >= start_ts
    if end_ts is not None:
        if end_exclusive:
            mask &= dt_series < end_ts
        else:
            mask &= dt_series <= end_ts
    return mask


def _validate_time_windows(ranges: Dict[str, Tuple[Optional[str], Optional[str]]], tz: str) -> None:
    parsed: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], bool]] = {}
    for key, (start, end) in ranges.items():
        start_ts, _ = _parse_time_bound(start, tz, is_end=False)
        end_ts, end_exclusive = _parse_time_bound(end, tz, is_end=True)
        if start_ts is None or end_ts is None:
            raise ValueError(f"Time split '{key}' requires both start and end.")
        if end_ts <= start_ts:
            raise ValueError(f"Time split '{key}' has end <= start ({start} -> {end}).")
        parsed[key] = (start_ts, end_ts, end_exclusive)
    order = ["train", "val", "test"]
    for idx in range(len(order) - 1):
        left = order[idx]
        right = order[idx + 1]
        left_end = parsed[left][1]
        right_start = parsed[right][0]
        if left_end > right_start:
            raise ValueError(f"Time splits overlap or are out of order between {left} and {right}.")


def _split_indices_by_time(
    frame: pd.DataFrame,
    *,
    tz: str,
    train_start: Optional[str],
    train_end: Optional[str],
    val_start: Optional[str],
    val_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
) -> Phase2SplitIndices:
    ranges = {
        "train": (train_start, train_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
    }
    _validate_time_windows(ranges, tz)
    dt_series = _datetime_series(frame, tz)
    train_mask = _time_window_mask(dt_series, start=train_start, end=train_end, tz=tz)
    val_mask = _time_window_mask(dt_series, start=val_start, end=val_end, tz=tz)
    test_mask = _time_window_mask(dt_series, start=test_start, end=test_end, tz=tz)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Time split produced empty train/val/test slice; check date bounds.")
    return Phase2SplitIndices(
        train_index=frame.index[train_mask].copy(),
        val_index=frame.index[val_mask].copy(),
        test_index=frame.index[test_mask].copy(),
    )


def _apply_purge_embargo(
    splits: Phase2SplitIndices,
    *,
    purge_bars: int,
    embargo_bars: int,
) -> Phase2SplitIndices:
    """Create leakage-safe gaps around chronological split boundaries."""
    purge = max(int(purge_bars), 0)
    embargo = max(int(embargo_bars), 0)

    train_index = splits.train_index[:-purge] if purge else splits.train_index.copy()
    val_start = min(embargo, len(splits.val_index))
    val_stop = len(splits.val_index) - purge if purge else len(splits.val_index)
    val_index = splits.val_index[val_start:max(val_start, val_stop)].copy()
    test_index = splits.test_index[min(embargo, len(splits.test_index)):].copy()

    if len(train_index) == 0 or len(val_index) == 0 or len(test_index) == 0:
        raise ValueError(
            "Purge/embargo produced an empty train/validation/test split; "
            "reduce --purge-bars/--embargo-bars or provide more data."
        )
    return Phase2SplitIndices(
        train_index=train_index.copy(),
        val_index=val_index,
        test_index=test_index,
    )


def _split_ts_metrics(frame: pd.DataFrame, splits: Phase2SplitIndices, tz: str) -> Dict[str, Any]:
    dt_series = _datetime_series(frame, tz)
    def _bounds(index: pd.Index) -> Tuple[Optional[str], Optional[str]]:
        if len(index) == 0:
            return None, None
        subset = dt_series.loc[index].dropna()
        if subset.empty:
            return None, None
        return subset.iloc[0].isoformat(), subset.iloc[-1].isoformat()

    train_start, train_end = _bounds(splits.train_index)
    val_start, val_end = _bounds(splits.val_index)
    test_start, test_end = _bounds(splits.test_index)
    return {
        "train_start_ts": train_start,
        "train_end_ts": train_end,
        "val_start_ts": val_start,
        "val_end_ts": val_end,
        "test_start_ts": test_start,
        "test_end_ts": test_end,
    }


def _recency_weights(
    dt_series: pd.Series,
    *,
    mode: str,
    max_weight: float,
    half_life_days: float,
) -> Optional[pd.Series]:
    if mode == "none":
        return None
    clean = dt_series.dropna()
    if clean.empty:
        return None
    max_ts = clean.max()
    ages = (max_ts - dt_series).dt.total_seconds() / 86400.0
    ages = ages.fillna(ages.max())
    max_weight = max(1.0, float(max_weight))
    if mode == "linear":
        max_age = max(float(ages.max()), 1.0)
        weights = 1.0 + (1.0 - (ages / max_age)) * (max_weight - 1.0)
        return weights.clip(lower=1.0, upper=max_weight)
    if mode == "exp":
        half_life = max(float(half_life_days), 1.0)
        # Interpret `half_life_days` literally: weight halves every `half_life_days`.
        raw = np.exp(-(np.log(2.0) * ages) / half_life)
        denom = float(raw.max() - raw.min())
        if denom <= 1e-9:
            return pd.Series(1.0, index=dt_series.index)
        scaled = 1.0 + (max_weight - 1.0) * (raw - raw.min()) / denom
        return pd.Series(scaled, index=dt_series.index).clip(lower=1.0, upper=max_weight)
    raise ValueError(f"Unknown recency weighting mode: {mode}")


def _feature_bounds(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    bounds: Dict[str, Dict[str, float]] = {}
    for col in frame.columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        finite = series[np.isfinite(series)]
        if finite.empty:
            continue
        bounds[str(col)] = {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "p05": float(finite.quantile(0.05)),
            "p95": float(finite.quantile(0.95)),
        }
    return bounds


def _feature_contract_payload(feature_names: Sequence[str], *, model_role: str) -> Dict[str, Any]:
    ordered = [str(name) for name in feature_names]
    if not ordered:
        raise ValueError(f"Model role {model_role} has no fitted features after filtering/selection")
    return {
        "features": ordered,
        "feature_count": len(ordered),
        "feature_hash": compute_feature_hash(ordered),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase2_role": model_role,
    }


def _predict_setup_stack_probs(model_path: Path, feature_frame: pd.DataFrame) -> pd.Series:
    probs = _predict_long_probabilities(model_path, feature_frame)
    return pd.Series(probs, index=feature_frame.index, name="stack_setup_prob")


# -------------------------------------------------------------------------
# TRAINING SPLIT DESIGN - retrain_v2_full (established 2026-02-24)
# -------------------------------------------------------------------------
# Train end:        2024-12-31
# Early stop start: 2025-01-01
# Early stop end:   2025-06-03  (~103 days / ~3.4 months of 2025 data)
# Val start:        2025-06-04
# Test end:         2026-01-14
#
# The early stop window intentionally excludes 2025 data from training
# to prevent gradient leakage into the out-of-sample validation period.
#
# KNOWN LIMITATION: The model has not been explicitly trained or validated
# on 2025 trend/regime behavior. The early stop gradients provide only
# indirect exposure to Jan-May 2025 market conditions.
#
# TODO (next retrain): Consider shrinking early stop window to 30-45 days
# to give validation more 2025 coverage while preserving gradient hygiene.
# -------------------------------------------------------------------------
def _split_calibration_subset(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split validation window chronologically into early-stop vs calibration slices."""
    n = len(X_val)
    if n < 20:
        raise ValueError("Validation slice is too small to split for calibration; provide more data or adjust ratios.")
    split_idx = max(1, int(n * 0.5))
    if split_idx >= n:
        split_idx = n - 1
    purge = max(int(purge_bars), 0)
    embargo = max(int(embargo_bars), 0)
    early_stop = split_idx - purge
    cal_start = split_idx + embargo
    if early_stop <= 0 or cal_start >= n:
        raise ValueError(
            "Purge/embargo leaves too little validation data for early stopping and calibration."
        )
    X_early = X_val.iloc[:early_stop].copy()
    y_early = y_val.iloc[:early_stop].copy()
    X_cal = X_val.iloc[cal_start:].copy()
    y_cal = y_val.iloc[cal_start:].copy()
    if len(X_cal) < 10:
        raise ValueError("Calibration slice is too small; provide more data or adjust ratios.")
    return X_early, X_cal, y_early, y_cal


def _scale_pos_weight(y: pd.Series) -> float:
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    return float(negatives / positives)


def _calibrate_classifier(
    estimator: XGBClassifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    method: str = "sigmoid",
) -> Tuple[CalibratedClassifierCV, Dict[str, Any]]:
    """Calibrate estimator on a held-out slice without leaking early-stop data."""
    chosen_method = str(method or "sigmoid").strip().lower()
    if chosen_method not in {"sigmoid", "isotonic"}:
        chosen_method = "sigmoid"
    cv_meta: Dict[str, Any]
    if (
        FrozenEstimator is not None
        and len(X_cal) >= 50
        and y_cal.nunique() > 1
        and (y_cal.value_counts() >= 5).all()
    ):
        frozen = FrozenEstimator(estimator)
        calibrator = CalibratedClassifierCV(frozen, method=chosen_method, cv=5)
        cv_meta = {"method": chosen_method, "cv": 5}
    else:
        calibrator = CalibratedClassifierCV(estimator, method=chosen_method, cv="prefit")
        cv_meta = {"method": chosen_method, "cv": "prefit"}
    calibrator.fit(X_cal, y_cal)
    return calibrator, cv_meta


def _parse_feature_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _ks_statistic(train_col: pd.Series, test_col: pd.Series) -> Tuple[float, Optional[float]]:
    train_vals = pd.to_numeric(train_col, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    test_vals = pd.to_numeric(test_col, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(train_vals) < 20 or len(test_vals) < 20:
        return 0.0, None
    if ks_2samp is not None:
        stat, pval = ks_2samp(train_vals, test_vals, alternative="two-sided", mode="auto")
        return float(stat), float(pval)
    xs = np.unique(np.concatenate([train_vals, test_vals]))
    if xs.size == 0:
        return 0.0, None
    a_sorted = np.sort(train_vals)
    b_sorted = np.sort(test_vals)
    cdf_a = np.searchsorted(a_sorted, xs, side="right") / float(len(a_sorted))
    cdf_b = np.searchsorted(b_sorted, xs, side="right") / float(len(b_sorted))
    return float(np.max(np.abs(cdf_a - cdf_b))), None


def _drift_features_to_drop(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    *,
    ks_limit: float,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    to_drop: List[str] = []
    diagnostics: List[Dict[str, Any]] = []
    for col in list(X_train.columns):
        stat, pval = _ks_statistic(X_train[col], X_validation[col])
        if stat > ks_limit:
            to_drop.append(col)
            diagnostics.append({"feature": str(col), "ks_stat": float(stat), "p_value": pval})
    return to_drop, diagnostics


def _trade_rate_threshold(
    probs: np.ndarray,
    *,
    target_low: float = 0.05,
    target_high: float = 0.20,
    default: float = 0.45,
) -> float:
    if probs.size == 0:
        return default
    candidates = [
        0.35,
        0.38,
        0.40,
        0.42,
        0.45,
        0.48,
        0.50,
        0.52,
        0.54,
        0.55,
        0.57,
        0.60,
        0.62,
        0.65,
        0.68,
        0.70,
        0.72,
        0.75,
    ]
    for threshold in candidates:
        rate = float((probs >= threshold).mean())
        if target_low <= rate <= target_high:
            return round(threshold, 3)
    percentile = float(np.quantile(probs, 1.0 - target_low))
    percentile = max(0.35, min(0.8, percentile))
    return round(percentile, 3)


def _precision_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    min_precision: float = 0.55,
    default: float = 0.58,
) -> float:
    candidates = [0.5, 0.52, 0.55, 0.58, 0.6, 0.62, 0.65, 0.68, 0.7, 0.75]
    best = default
    for threshold in candidates:
        preds = probs >= threshold
        if preds.mean() < 0.05:
            continue
        try:
            score = precision_score(labels, preds, zero_division=0)
        except Exception:
            continue
        if score >= min_precision:
            best = threshold
    return round(max(0.5, min(0.8, best)), 3)


def _direction_thresholds(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.size == 0 or labels.size == 0:
        return 0.58, 0.58
    long_threshold = _precision_threshold(probs, labels)
    short_probs = 1.0 - probs
    short_labels = 1 - labels
    sell_threshold = _precision_threshold(short_probs, short_labels)
    # p_short is short-class confidence; runtime converts it to a
    # long-probability cutoff with (1 - p_short).
    return long_threshold, sell_threshold


def _threshold_values(start: float, end: float, step: float) -> List[float]:
    count = int(round((end - start) / step)) + 1
    return [round(start + i * step, 3) for i in range(count)]


def _normalize_max_dd(value: float, account_size: float = 50000.0) -> float:
    if account_size <= 0:
        return 0.0
    return abs(float(value)) / account_size


def _score_threshold_combo(sim_result: Dict[str, Any], objective: str) -> float:
    sharpe = float(sim_result.get("sharpe") or 0.0)
    max_dd = float(sim_result.get("max_drawdown") or 0.0)
    flip_rate = float(sim_result.get("flip_rate_per_day") or 0.0)
    total_pnl = float(sim_result.get("total_pnl_usd") or 0.0)
    avg_trade = float(sim_result.get("avg_trade_usd") or 0.0)
    account_size = 50000.0
    max_dd_norm = _normalize_max_dd(max_dd, account_size)
    obj = objective.lower()
    if obj == "calmar":
        denom = abs(max_dd)
        if denom <= 1e-6:
            return total_pnl
        return total_pnl / denom
    if obj == "ev":
        return avg_trade
    # default sharpe-based score
    base = sharpe if np.isfinite(sharpe) else -1e9
    return base - 0.25 * max_dd_norm - 0.05 * flip_rate


def _trading_summary(sim_result: Dict[str, Any]) -> Dict[str, Any]:
    account_size = 50000.0
    trades = int(sim_result.get("trade_count") or len(sim_result.get("trades", [])))
    hold_bars = sim_result.get("hold_bars") or 0.0
    if isinstance(hold_bars, dict):
        hold_bars = hold_bars.get("mean") or 0.0
    return {
        "total_pnl_usd": float(sim_result.get("total_pnl_usd") or 0.0),
        "sharpe": float(sim_result.get("sharpe") or 0.0),
        "profit_factor": float(sim_result.get("profit_factor") or 0.0),
        "max_dd": -_normalize_max_dd(sim_result.get("max_drawdown") or 0.0, account_size),
        "trades": trades,
        "win_rate": float(sim_result.get("win_rate") or 0.0),
        "avg_trade_usd": float(sim_result.get("avg_trade_usd") or 0.0),
        "avg_hold_bars": float(hold_bars or 0.0),
        "trades_per_day": float(sim_result.get("trades_per_day") or 0.0),
        "flip_rate_per_day": float(sim_result.get("flip_rate_per_day") or 0.0),
    }


def _parse_mtf_timeframes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value]
    else:
        items = [item.strip() for item in str(value).split(",")]
    normalized: List[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        lower = item.lower()
        if lower.endswith("m") and lower[:-1].isdigit():
            lower = f"{lower[:-1]}min"
        if lower.isdigit():
            lower = f"{lower}min"
        if lower not in seen:
            normalized.append(lower)
            seen.add(lower)
    return normalized


def _prepare_multi_tf_source(
    df_raw: pd.DataFrame,
    *,
    tz: str,
    csv_naive_is_utc: bool,
) -> pd.DataFrame:
    frame = df_raw.copy()
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    frame = frame.rename(columns={key: value for key, value in rename_map.items() if key in frame.columns})
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"MTF feature generation missing OHLCV columns: {missing}")
    aligned = ensure_dataframe_index(frame, tz, naive_is_utc=csv_naive_is_utc)
    return aligned[required].sort_index()


def _attach_multi_tf_features(
    feats: pd.DataFrame,
    *,
    df_raw: pd.DataFrame,
    timeframes: List[str],
    tz: str,
    csv_naive_is_utc: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    if not timeframes:
        return feats, []
    mtf_source = _prepare_multi_tf_source(df_raw, tz=tz, csv_naive_is_utc=csv_naive_is_utc)
    mtf_features = build_multi_tf_features(mtf_source, timeframes=timeframes)
    if mtf_features.empty:
        return feats, []
    feat_ts = pd.to_datetime(feats["Datetime"], errors="coerce")
    if getattr(feat_ts.dt, "tz", None) is None:
        feat_ts = feat_ts.dt.tz_localize(tz)
    else:
        feat_ts = feat_ts.dt.tz_convert(tz)
    aligned = mtf_features.reindex(pd.DatetimeIndex(feat_ts)).ffill()
    aligned.index = feats.index
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.ffill().bfill()
    merged = pd.concat([feats, aligned], axis=1)
    return merged, list(aligned.columns)


def _walkforward_replay_summary(
    frame: pd.DataFrame,
    *,
    setup_probs: pd.Series,
    dir_probs: pd.Series,
    thresholds: Dict[str, float],
    sim_cfg: Phase2SimConfig,
    windows: int,
    decision_policy: Phase2DecisionPolicy,
) -> Dict[str, Any]:
    if frame.empty:
        return {"windows": [], "summary": {"count": 0}}
    window_count = max(1, int(windows or 1))
    index_chunks = np.array_split(frame.index.to_numpy(), window_count)
    rows: List[Dict[str, Any]] = []
    total_pnl = 0.0
    total_trades = 0
    max_dd = 0.0
    for idx, chunk in enumerate(index_chunks, start=1):
        chunk_index = pd.Index(chunk)
        if chunk_index.empty:
            continue
        chunk_frame = frame.loc[chunk_index].copy()
        phase = phase2_decisions(
            chunk_frame,
            setup_probs.reindex(chunk_index).to_numpy(dtype=float),
            dir_probs.reindex(chunk_index).to_numpy(dtype=float),
            thresholds,
            policy=decision_policy,
        )
        sim = simulate_trades(phase, thresholds, cfg=sim_cfg)
        total_pnl += float(sim.get("total_pnl_usd") or 0.0)
        total_trades += int(sim.get("trade_count") or len(sim.get("trades") or []))
        max_dd = min(max_dd, float(sim.get("max_drawdown") or 0.0))
        rows.append(
            {
                "window": idx,
                "start": str(chunk_frame["Datetime"].iloc[0]),
                "end": str(chunk_frame["Datetime"].iloc[-1]),
                **_trading_summary(sim),
            }
        )
    summary = {
        "count": len(rows),
        "total_pnl_usd": total_pnl,
        "trade_count": total_trades,
        "avg_profit_factor": float(np.mean([row["profit_factor"] for row in rows])) if rows else 0.0,
        "avg_sharpe": float(np.mean([row["sharpe"] for row in rows])) if rows else 0.0,
        "max_drawdown_usd": max_dd,
    }
    return {"windows": rows, "summary": summary}


def _load_live_shadow_gate(summary_path: Optional[str]) -> Dict[str, Any]:
    path_text = str(summary_path or "").strip()
    if not path_text:
        return {
            "path": None,
            "exists": False,
            "passed": False,
            "status": "missing",
            "reasons": ["live shadow summary path not configured"],
            "summary": None,
        }
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "passed": False,
            "status": "missing",
            "reasons": ["live shadow summary file not found"],
            "summary": None,
        }
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "passed": False,
            "status": "invalid",
            "reasons": [f"unable to parse live shadow summary: {exc}"],
            "summary": None,
        }
    verdict = str(payload.get("verdict") or "").strip().lower()
    feed_health_ok = bool(payload.get("feed_health_ok"))
    unresolved = [str(item) for item in (payload.get("unresolved_warnings") or [])]
    position_state = str(payload.get("position_state") or "")
    reasons: List[str] = []
    if not feed_health_ok:
        reasons.append("feed_health_not_ok")
    if verdict in {"unsafe", "blocked", "fail", "failed"}:
        reasons.append(f"verdict_{verdict}")
    if unresolved:
        reasons.extend(f"warning_{item}" for item in unresolved)
    passed = feed_health_ok and verdict not in {"unsafe", "blocked", "fail", "failed"} and not unresolved
    return {
        "path": str(path),
        "exists": True,
        "passed": passed,
        "status": "ok" if passed else "blocked",
        "reasons": reasons,
        "summary": {
            "ts": payload.get("ts"),
            "verdict": payload.get("verdict"),
            "feed_health_ok": feed_health_ok,
            "status_stale_sec": payload.get("status_stale_sec"),
            "bar_age_sec": payload.get("bar_age_sec"),
            "position_state": position_state,
            "unresolved_warnings": unresolved,
        },
    }


def _direction_probability_diagnostics(probs: pd.Series) -> Dict[str, float]:
    arr = probs.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "extreme_frac": 1.0}
    extreme_mask = (finite <= 0.05) | (finite >= 0.95)
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "extreme_frac": float(extreme_mask.mean()),
    }


def optimize_phase2_thresholds(
    frame: pd.DataFrame,
    setup_probs: pd.Series,
    dir_probs: pd.Series,
    *,
    sim_cfg: Phase2SimConfig,
    decision_policy: Phase2DecisionPolicy,
    objective: str,
    min_trades_val: int,
    max_flip_rate: float,
    max_bad_fade_rate: float,
    setup_threshold_start: float = 0.20,
    setup_threshold_end: float = 0.85,
    setup_threshold_step: float = 0.03,
    direction_threshold_start: float = 0.55,
    direction_threshold_end: float = 0.82,
    direction_threshold_step: float = 0.02,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    setup_values = _threshold_values(setup_threshold_start, setup_threshold_end, setup_threshold_step)
    long_values = _threshold_values(direction_threshold_start, direction_threshold_end, direction_threshold_step)
    short_values = _threshold_values(direction_threshold_start, direction_threshold_end, direction_threshold_step)
    best_score = float("-inf")
    best_thresholds: Optional[Dict[str, float]] = None
    best_sim: Optional[Dict[str, Any]] = None
    best_audit: Optional[Dict[str, Any]] = None
    valid_combos = 0
    total_combos = 0
    rejected_low_trades = 0
    rejected_high_flip = 0
    rejected_bad_fade = 0
    best_any_score = float("-inf")
    best_any: Optional[Dict[str, Any]] = None

    setup_arr = setup_probs.reindex(frame.index).to_numpy(dtype=float)
    dir_arr = dir_probs.reindex(frame.index).to_numpy(dtype=float)
    finite_setup = setup_arr[np.isfinite(setup_arr)]
    setup_quantiles: Dict[str, float] = {}
    if finite_setup.size:
        for q in (0.5, 0.75, 0.9, 0.95, 0.99):
            setup_quantiles[f"q{int(q * 100)}"] = float(np.quantile(finite_setup, q))

    for p_setup in setup_values:
        for p_long in long_values:
            for p_short in short_values:
                total_combos += 1
                thresholds = {"p_setup": p_setup, "p_long": p_long, "p_short": p_short}
                df_phase = phase2_decisions(frame.copy(), setup_arr, dir_arr, thresholds, policy=decision_policy)
                sim = simulate_trades(df_phase, thresholds, cfg=sim_cfg)
                audit = _phase2_behavior_audit(df_phase)
                trades = int(sim.get("trade_count") or len(sim.get("trades", [])))
                flip = float(sim.get("flip_rate_per_day") or 0.0)
                bad_fade = float(audit.get("countertrend_rate") or 0.0)
                score = _score_threshold_combo(sim, objective) - bad_fade * 2.0
                if np.isfinite(score) and score > best_any_score:
                    best_any_score = score
                    best_any = {
                        "thresholds": thresholds,
                        "score": float(score),
                        "trades": trades,
                        "flip_rate_per_day": flip,
                        "bad_fade_rate": bad_fade,
                        "profit_factor": float(sim.get("profit_factor") or 0.0),
                        "sharpe": float(sim.get("sharpe") or 0.0),
                        "total_pnl_usd": float(sim.get("total_pnl_usd") or 0.0),
                        "max_drawdown": float(sim.get("max_drawdown") or 0.0),
                    }
                if trades < min_trades_val:
                    rejected_low_trades += 1
                    continue
                if flip > max_flip_rate:
                    rejected_high_flip += 1
                    continue
                if bad_fade > float(max_bad_fade_rate):
                    rejected_bad_fade += 1
                    continue
                valid_combos += 1
                if not np.isfinite(score):
                    continue
                if score > best_score:
                    best_score = score
                    best_thresholds = thresholds
                    best_sim = sim
                    best_audit = audit

    diagnostics = {
        "grid": {
            "p_setup": {"start": float(setup_threshold_start), "end": float(setup_threshold_end), "step": float(setup_threshold_step), "count": len(setup_values)},
            "p_long": {"start": float(direction_threshold_start), "end": float(direction_threshold_end), "step": float(direction_threshold_step), "count": len(long_values)},
            "p_short": {"start": float(direction_threshold_start), "end": float(direction_threshold_end), "step": float(direction_threshold_step), "count": len(short_values)},
        },
        "total_combos": int(total_combos),
        "valid_combos": int(valid_combos),
        "rejected_low_trades": int(rejected_low_trades),
        "rejected_high_flip": int(rejected_high_flip),
        "rejected_bad_fade": int(rejected_bad_fade),
        "constraints": {
            "min_trades_val": int(min_trades_val),
            "max_flip_rate": float(max_flip_rate),
            "max_bad_fade_rate": float(max_bad_fade_rate),
        },
        "setup_prob_quantiles": setup_quantiles,
        "setup_prob_max": float(np.max(finite_setup)) if finite_setup.size else None,
        "best_any_combo": best_any,
    }
    if not best_thresholds or not best_sim:
        reason = (
            f"No threshold combo met constraints (min_trades_val={min_trades_val}, "
            f"max_flip_rate={max_flip_rate})"
        )
        if valid_combos == 0:
            return None, None, None, reason, diagnostics
        return None, None, None, "Threshold optimizer could not improve objective", diagnostics
    if best_score <= 0.0 or float(best_sim.get("total_pnl_usd") or 0.0) <= 0.0:
        diagnostics["selected_combo"] = {
            "thresholds": best_thresholds,
            "score": float(best_score),
            "trades": int(best_sim.get("trade_count") or len(best_sim.get("trades", []))),
            "flip_rate_per_day": float(best_sim.get("flip_rate_per_day") or 0.0),
            "bad_fade_rate": float((best_audit or {}).get("countertrend_rate") or 0.0),
        }
        return (
            None,
            None,
            None,
            "Best threshold combo has non-positive validation expectancy/PnL",
            diagnostics,
        )

    diagnostics["selected_combo"] = {
        "thresholds": best_thresholds,
        "score": float(best_score),
        "trades": int(best_sim.get("trade_count") or len(best_sim.get("trades", []))),
        "flip_rate_per_day": float(best_sim.get("flip_rate_per_day") or 0.0),
        "bad_fade_rate": float((best_audit or {}).get("countertrend_rate") or 0.0),
    }
    return best_thresholds, best_sim, best_audit, None, diagnostics


def _build_phase2_labels(
    feats: pd.DataFrame,
    *,
    horizon: int,
    threshold: float,
    use_log: bool,
    scheme: str,
    mode: str,
    instrument: str,
    max_hold_bars: int,
    commission_per_contract: float,
    slippage_ticks: float,
    setup_threshold_multiplier: float,
) -> Phase2LabelBundle:
    if mode == "exec":
        base = _build_exec_label_frame(
            feats,
            threshold=threshold,
            instrument=instrument,
            max_hold_bars=max_hold_bars,
            commission_per_contract=commission_per_contract,
            slippage_ticks=slippage_ticks,
        )
    else:
        base = make_labels(
            feats,
            horizon=horizon,
            threshold=threshold,
            scheme=scheme,
            use_log=use_log,
            drop_flat=False,
            use_htf_trend_aware=(scheme == "trend_aware_ternary"),
        )
    target = base["target"].astype("Int64")
    trimmed = feats.loc[target.index]
    move_threshold = max(float(threshold) * max(float(setup_threshold_multiplier), 1.0), float(threshold))
    close_series = pd.to_numeric(trimmed.get("Close"), errors="coerce")
    if mode == "exec" and "setup_move_abs_return" in base.columns:
        gross_move = pd.to_numeric(base.get("setup_move_abs_return"), errors="coerce")
    else:
        future_close = pd.to_numeric(trimmed.get("Close"), errors="coerce").shift(-int(horizon))
        gross_move = (future_close - close_series).abs() / close_series.replace(0.0, np.nan)
    selective_setup = gross_move.fillna(0.0) >= move_threshold
    y_setup = ((target != 0) & selective_setup).astype(int)
    signed_counts = {
        "long": int((target == 1).sum()),
        "short": int((target == -1).sum()),
        "flat": int((target == 0).sum()),
    }
    mask_direction = target.isin([1, -1])
    direction = (target[mask_direction] == 1).astype(int)
    info = {"signed_counts": signed_counts, "dropped_flats": True, "scheme": scheme}
    return Phase2LabelBundle(setup=y_setup, direction=direction, direction_info=info, trimmed_frame=trimmed)


def _phase2_behavior_audit(df_phase: pd.DataFrame) -> Dict[str, Any]:
    required = {"phase2_direction_signal", "Close", "vwap_sess", "ema_20", "ema_50"}
    if not required.issubset(set(df_phase.columns)):
        return {
            "entries": 0,
            "countertrend_entries": 0,
            "countertrend_rate": 0.0,
            "bad_short_in_up": 0,
            "bad_long_in_down": 0,
        }
    entries = pd.to_numeric(df_phase["phase2_direction_signal"], errors="coerce").fillna(0.0)
    close = pd.to_numeric(df_phase["Close"], errors="coerce")
    vwap = pd.to_numeric(df_phase["vwap_sess"], errors="coerce")
    ema20 = pd.to_numeric(df_phase["ema_20"], errors="coerce")
    ema50 = pd.to_numeric(df_phase["ema_50"], errors="coerce")
    strong_up = (close > vwap) & (ema20 > ema50)
    strong_down = (close < vwap) & (ema20 < ema50)
    bad_short = (entries < 0) & strong_up
    bad_long = (entries > 0) & strong_down
    entry_count = int((entries != 0).sum())
    countertrend_count = int(bad_short.sum() + bad_long.sum())
    return {
        "entries": entry_count,
        "countertrend_entries": countertrend_count,
        "countertrend_rate": float(countertrend_count / max(entry_count, 1)),
        "bad_short_in_up": int(bad_short.sum()),
        "bad_long_in_down": int(bad_long.sum()),
    }


def _build_exec_label_frame(
    feats: pd.DataFrame,
    *,
    threshold: float,
    instrument: str,
    max_hold_bars: int,
    commission_per_contract: float,
    slippage_ticks: float,
) -> pd.DataFrame:
    inst = instrument_by_alias(instrument)
    if inst.point_value <= 0:
        raise ValueError(f"Instrument {instrument} has invalid point value.")
    required_cols = ["Open", "High", "Low", "Close", "Datetime"]
    missing = [col for col in required_cols if col not in feats.columns]
    if missing:
        raise ValueError(f"Execution label mode requires columns {missing}")

    opens = pd.to_numeric(feats["Open"], errors="coerce")
    highs = pd.to_numeric(feats["High"], errors="coerce")
    lows = pd.to_numeric(feats["Low"], errors="coerce")
    closes = pd.to_numeric(feats["Close"], errors="coerce")
    if opens.isna().any() or highs.isna().any() or lows.isna().any() or closes.isna().any():
        raise ValueError("Price columns contain NaNs; cannot build execution labels.")

    per_side_cost = commission_per_contract + slippage_ticks * inst.tick_value
    cost_points_total = 2.0 * per_side_cost / inst.point_value
    max_hold = max(int(max_hold_bars), 1)
    valid_length = len(feats) - max_hold - 1
    if valid_length <= 0:
        raise ValueError("Not enough rows to compute execution labels; extend dataset or reduce max_hold_bars.")

    targets = []
    setup_moves = []
    index_values: list = []
    for base_offset in range(valid_length):
        entry_idx = base_offset + 1
        entry_price = float(opens.iloc[entry_idx])
        if entry_price <= 0:
            targets.append(0)
            setup_moves.append(0.0)
            index_values.append(feats.index[base_offset])
            continue

        cost_return = cost_points_total / max(entry_price, 1e-6)
        target_return = threshold + cost_return
        up_price = entry_price * (1.0 + target_return)
        down_price = entry_price * (1.0 - target_return)
        exit_price = float(closes.iloc[min(entry_idx + max_hold, len(feats) - 1)])
        label = 0
        ambiguous_same_bar = False
        last_offset = min(entry_idx + max_hold, len(feats) - 1)
        best_move_return = 0.0
        for offset in range(entry_idx, last_offset + 1):
            hi = float(highs.iloc[offset])
            lo = float(lows.iloc[offset])
            best_move_return = max(
                best_move_return,
                abs(hi - entry_price) / max(entry_price, 1e-6),
                abs(entry_price - lo) / max(entry_price, 1e-6),
            )
            hit_up = hi >= up_price
            hit_down = lo <= down_price
            if hit_up and hit_down:
                ambiguous_same_bar = True
                label = 0
                break
            if hit_up:
                exit_price = up_price
                label = 1
                break
            if hit_down:
                exit_price = down_price
                label = -1
                break
        if label == 0 and not ambiguous_same_bar:
            raw_change = exit_price - entry_price
            net_long = raw_change - cost_points_total
            net_short = -raw_change - cost_points_total
            if net_long > 0:
                label = 1
            elif net_short > 0:
                label = -1
        targets.append(label)
        setup_moves.append(best_move_return)
        index_values.append(feats.index[base_offset])

    result = feats.loc[index_values].copy()
    result["target"] = pd.Series(targets, index=result.index).astype("Int64")
    result["setup_move_abs_return"] = pd.Series(setup_moves, index=result.index).astype(float)
    return result


def _close_schema(label_cfg, runtime_labels) -> LabelSchema:
    return LabelSchema(
        domain="phase2_close_binary",
        horizon_bars=label_cfg.horizon,
        trend_ma_window=runtime_labels.trend_ma_window,
        trend_slope_window=runtime_labels.trend_slope_window,
        drop_flats=True,
        positive_label=1,
        negative_label=0,
        params={
            "source": "phase2_close_overlay_replay",
            "mapping": {"1": "CLOSE", "0": "HOLD"},
            "threshold": float(label_cfg.threshold),
        },
    )


def _trade_close_feature_names() -> List[str]:
    return [
        "bars_in_trade",
        "unrealized_r",
        "mfe_r",
        "mae_r",
        "giveback_r",
        "distance_to_stop_r",
        "distance_to_target_r",
        "target_attached",
        "protected_confirmed",
    ]


def _build_close_training_frame(
    feats: pd.DataFrame,
    *,
    trades: List[Dict[str, object]],
    instrument: str,
    label_threshold: float,
    pnl_giveback_activate_r: float,
    pnl_giveback_close_r: float,
    pnl_stall_bars: int,
    pnl_stall_min_mfe_r: float,
    pnl_stall_close_below_r: float,
    pnl_severe_adverse_r: float,
    pnl_target_arm_min_hold_bars: int,
    pnl_target_arm_min_unrealized_r: float,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    inst = instrument_by_alias(instrument)
    tick_size = float(getattr(inst, "tick_size", 0.25) or 0.25)
    dt_series = pd.to_datetime(feats["Datetime"], errors="coerce")
    dt_lookup: Dict[str, int] = {}
    for idx, ts_val in zip(feats.index, dt_series):
        if pd.isna(ts_val):
            continue
        dt_lookup[pd.Timestamp(ts_val).isoformat()] = idx

    close_rows: List[pd.Series] = []
    close_labels: List[int] = []
    label_reasons: Dict[str, int] = {
        "overlay_severe_adverse": 0,
        "overlay_giveback": 0,
        "overlay_stall": 0,
        "trade_exit": 0,
        "hold": 0,
    }

    if "Datetime" not in feats.columns or "Close" not in feats.columns or "High" not in feats.columns or "Low" not in feats.columns:
        raise ValueError("Close-model training requires Datetime/Open/High/Low/Close columns.")

    for trade in trades:
        side_num = int(trade.get("side") or 0)
        if side_num not in (-1, 1):
            continue
        entry_ts_raw = trade.get("entry_ts")
        exit_ts_raw = trade.get("exit_ts")
        if not entry_ts_raw or not exit_ts_raw:
            continue
        entry_idx = dt_lookup.get(pd.Timestamp(entry_ts_raw).isoformat())
        exit_idx = dt_lookup.get(pd.Timestamp(exit_ts_raw).isoformat())
        if entry_idx is None or exit_idx is None or exit_idx < entry_idx:
            continue
        entry_pos = feats.index.get_loc(entry_idx)
        exit_pos = feats.index.get_loc(exit_idx)
        if not isinstance(entry_pos, int) or not isinstance(exit_pos, int) or exit_pos < entry_pos:
            continue

        entry_price = float(trade.get("entry_price") or feats.iloc[entry_pos].get("Close") or 0.0)
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        risk_points = max(abs(entry_price) * float(label_threshold), tick_size)
        stop_price = entry_price - float(side_num) * risk_points
        target_price = entry_price + float(side_num) * risk_points
        max_favorable_r = 0.0
        max_adverse_r = 0.0

        for pos in range(entry_pos, exit_pos + 1):
            row = feats.iloc[pos].copy()
            close_val = float(pd.to_numeric(pd.Series([row.get("Close")]), errors="coerce").iloc[0])
            high_val = float(pd.to_numeric(pd.Series([row.get("High")]), errors="coerce").iloc[0])
            low_val = float(pd.to_numeric(pd.Series([row.get("Low")]), errors="coerce").iloc[0])
            bars_in_trade = pos - entry_pos

            if side_num > 0:
                unrealized_points = close_val - entry_price
                favorable_r = max(0.0, (high_val - entry_price) / risk_points)
                adverse_r = max(0.0, (entry_price - low_val) / risk_points)
                distance_to_stop_r = (close_val - stop_price) / risk_points
                distance_to_target_r = (target_price - close_val) / risk_points
            else:
                unrealized_points = entry_price - close_val
                favorable_r = max(0.0, (entry_price - low_val) / risk_points)
                adverse_r = max(0.0, (high_val - entry_price) / risk_points)
                distance_to_stop_r = (stop_price - close_val) / risk_points
                distance_to_target_r = (close_val - target_price) / risk_points

            unrealized_r = unrealized_points / risk_points
            max_favorable_r = max(max_favorable_r, favorable_r)
            max_adverse_r = max(max_adverse_r, adverse_r)
            giveback_r = max(0.0, max_favorable_r - unrealized_r)
            target_attached = bool(
                bars_in_trade >= int(pnl_target_arm_min_hold_bars)
                or unrealized_r >= float(pnl_target_arm_min_unrealized_r)
            )
            protected_confirmed = True

            row["bars_in_trade"] = int(bars_in_trade)
            row["unrealized_r"] = float(unrealized_r)
            row["mfe_r"] = float(max_favorable_r)
            row["mae_r"] = float(max_adverse_r)
            row["giveback_r"] = float(giveback_r)
            row["distance_to_stop_r"] = float(distance_to_stop_r)
            row["distance_to_target_r"] = float(distance_to_target_r)
            row["target_attached"] = int(target_attached)
            row["protected_confirmed"] = int(protected_confirmed)

            label = 0
            label_reason = "hold"
            if bars_in_trade == (exit_pos - entry_pos):
                label = 1
                label_reason = "trade_exit"
            elif unrealized_r <= -float(pnl_severe_adverse_r):
                label = 1
                label_reason = "overlay_severe_adverse"
            elif (
                max_favorable_r >= float(pnl_giveback_activate_r)
                and giveback_r >= float(pnl_giveback_close_r)
            ):
                label = 1
                label_reason = "overlay_giveback"
            elif (
                bars_in_trade >= int(pnl_stall_bars)
                and max_favorable_r < float(pnl_stall_min_mfe_r)
                and unrealized_r <= float(pnl_stall_close_below_r)
            ):
                label = 1
                label_reason = "overlay_stall"

            close_rows.append(row)
            close_labels.append(label)
            label_reasons[label_reason] = label_reasons.get(label_reason, 0) + 1

    if not close_rows:
        raise ValueError("Close-model replay frame is empty; no simulated trades were available.")

    frame = pd.DataFrame(close_rows)
    labels = pd.Series(close_labels, index=frame.index, dtype="int64")
    label_info = {
        "class_counts": {
            "close": int((labels == 1).sum()),
            "hold": int((labels == 0).sum()),
        },
        "label_reasons": label_reasons,
        "trade_count": int(len(trades)),
        "trade_state_features": _trade_close_feature_names(),
    }
    return frame, labels, label_info


def _write_phase2_candidate_manifest(
    *,
    args: argparse.Namespace,
    artifact_dir: Path,
    setup_metrics: Dict[str, Any],
    direction_metrics: Dict[str, Any],
    close_metrics: Optional[Dict[str, Any]],
    extras: Dict[str, Any],
) -> Path:
    setup_path = Path(args.setup_model_path).resolve()
    dir_path = Path(args.direction_model_path).resolve()
    close_path = Path(args.close_model_path).resolve()
    close_features_path = close_path.with_suffix(".features.json")
    close_schema_path = close_path.with_suffix(".label_schema.json")
    setup_features = json.loads(setup_path.with_suffix(".features.json").read_text())
    dir_features = json.loads(dir_path.with_suffix(".features.json").read_text())
    close_artifact_present = close_path.exists() and close_features_path.exists() and close_schema_path.exists() and close_metrics is not None
    close_features = json.loads(close_features_path.read_text()) if close_artifact_present else {}
    setup_feature_list = list((setup_features or {}).get("features") or [])
    dir_feature_list = list((dir_features or {}).get("features") or [])
    close_feature_list = list((close_features or {}).get("features") or [])
    feature_hash = compute_feature_hash(dir_feature_list or setup_feature_list)
    runtime_feature_hash = compute_feature_hash(MANDATORY_MODEL_FEATURES)
    close_feature_hash = compute_feature_hash(close_feature_list) if close_feature_list else None
    manifest = {
        "tag": artifact_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv": str(getattr(args, "csv", "")),
        "instrument": str(getattr(args, "instrument", "ES")),
        "timeframe": "5m",
        "artifact_dir": ".",
        "setup_model_path": setup_path.name,
        "dir_model_path": dir_path.name,
        "close_model_path": close_path.name if close_artifact_present else None,
        "thresholds": dict(extras.get("thresholds") or {}),
        "close": {
            "enabled": close_artifact_present,
            "threshold": float(getattr(args, "close_threshold", 0.60)),
            "model_path": close_path.name if close_artifact_present else None,
            "feature_hash": close_feature_hash,
            "feature_count": len(close_feature_list),
            "label_schema_file": close_schema_path.name if close_artifact_present else None,
        },
        "trading_val": extras.get("trading_val"),
        "trading_test": extras.get("trading_test"),
        "feature_hash": runtime_feature_hash,
        "runtime_feature_hash": runtime_feature_hash,
        "feature_hashes": {
            "setup": compute_feature_hash(setup_feature_list) if setup_feature_list else None,
            "direction": feature_hash,
            "close": close_feature_hash,
        },
        "metrics": {
            "setup": setup_metrics,
            "direction": direction_metrics,
            "close": close_metrics,
        },
        "direction_diagnostics": extras.get("direction_diagnostics"),
        "threshold_diagnostics": extras.get("threshold_diagnostics"),
        "behavior_audit_val": extras.get("behavior_audit_val"),
        "behavior_audit_test": extras.get("behavior_audit_test"),
        "walkforward": extras.get("walkforward"),
        "live_shadow_gate": extras.get("live_shadow_gate"),
        "promotion_blocked": bool(extras.get("promotion_blocked")),
        "promotion_blocked_reason": extras.get("promotion_blocked_reason"),
        "rejected": bool(extras.get("rejected")),
        "rejected_reason": extras.get("rejected_reason"),
        "config": {
            "tz": args.tz,
            "rth_start": args.rth_start,
            "rth_end": args.rth_end,
            "orb_minutes": args.orb_minutes,
            "horizon": args.horizon,
            "label_mode": str(getattr(args, "label_mode", "horizon")),
            "label_threshold": args.label_threshold,
            "label_log": bool(args.label_log),
            "label_commission_per_contract": float(
                getattr(args, "label_commission_per_contract", 2.0)
            ),
            "label_slippage_ticks": float(getattr(args, "label_slippage_ticks", 1.0)),
            "label_max_hold_bars": int(
                getattr(args, "label_max_hold_bars", args.horizon)
            ),
            "htf_trend_aware": bool(args.htf_trend_aware),
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "train_start": args.train_start,
            "train_end": args.train_end,
            "val_start": args.val_start,
            "val_end": args.val_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "csv_naive_is_utc": bool(args.csv_naive_is_utc),
            "stack_setup_prob": bool(args.stack_setup_prob),
            "mtf_timeframes": list(_parse_mtf_timeframes(getattr(args, "mtf_timeframes", []))),
            "threshold_objective": args.threshold_objective,
            "min_trades_val": args.min_trades_val,
            "max_flip_rate": args.max_flip_rate,
            "walkforward_windows": int(getattr(args, "walkforward_windows", 0) or 0),
            "live_shadow_summary_path": getattr(args, "live_shadow_summary_path", None),
            "require_safe_shadow_pass": bool(getattr(args, "require_safe_shadow_pass", False)),
            "commission_per_contract": args.commission_per_contract,
            "slippage_ticks": args.slippage_ticks,
            "recency_weighting": args.recency_weighting,
            "recency_max_weight": args.recency_max_weight,
            "recency_half_life_days": args.recency_half_life_days,
            "setup_threshold_multiplier": float(getattr(args, "setup_threshold_multiplier", 1.75)),
            "max_bad_fade_rate": float(getattr(args, "max_bad_fade_rate", 0.18)),
            "decision_policy": Phase2DecisionPolicy(
                entry_trend_filter=str(getattr(args, "entry_trend_filter", "none") or "none"),
                min_signal_persistence_bars=int(getattr(args, "min_signal_persistence_bars", 1) or 1),
                cooldown_bars_after_flip=int(getattr(args, "cooldown_bars_after_flip", 0) or 0),
            ).to_manifest(),
            "close_threshold": float(getattr(args, "close_threshold", 0.60)),
            "close_replay": {
                "pnl_giveback_activate_r": float(getattr(args, "close_giveback_activate_r", 1.0)),
                "pnl_giveback_close_r": float(getattr(args, "close_giveback_close_r", 0.5)),
                "pnl_stall_bars": int(getattr(args, "close_stall_bars", 4)),
                "pnl_stall_min_mfe_r": float(getattr(args, "close_stall_min_mfe_r", 0.25)),
                "pnl_stall_close_below_r": float(getattr(args, "close_stall_close_below_r", -0.10)),
                "pnl_severe_adverse_r": float(getattr(args, "close_severe_adverse_r", 0.90)),
                "pnl_target_arm_min_hold_bars": int(getattr(args, "close_target_arm_min_hold_bars", 3)),
                "pnl_target_arm_min_unrealized_r": float(getattr(args, "close_target_arm_min_unrealized_r", 0.75)),
            },
        },
    }
    path = artifact_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _train_binary_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    save_path: Path,
    runtime_labels,
    label_cfg,
    label_info: Dict[str, Any],
    label_schema: LabelSchema,
    instrument: str,
    tz: str,
    rth_start: str,
    rth_end: str,
    orb_minutes: int,
    val_ratio: float,
    test_ratio: float,
    event_filter: str,
    model_role: str,
    extra_metrics: Optional[Dict[str, Any]] = None,
    horizon_minutes: Optional[int],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
    early_stopping_rounds: int,
    min_child_weight: Optional[float] = None,
    scale_pos_weight: Optional[float] = None,
    random_state: int,
    use_gpu: bool,
    model_id_suffix: str,
    split_indices: Optional[Phase2SplitIndices] = None,
    sample_weight: Optional[pd.Series] = None,
    split_meta: Optional[Dict[str, Any]] = None,
    sample_weight_meta: Optional[Dict[str, Any]] = None,
    force_drop_features: Optional[List[str]] = None,
    top_n_features: Optional[int] = None,
    drift_ks_stat_max: Optional[float] = None,
    calibration_method: str = "sigmoid",
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> Tuple[Dict[str, Any], Phase2SplitIndices]:
    numeric = X.select_dtypes(include=[np.number])
    if numeric.isna().any().any():
        bad = {col: int(numeric[col].isna().sum()) for col in numeric.columns if numeric[col].isna().any()}
        raise ValueError(f"Feature matrix contains NaNs: {bad}")
    numeric = _strict_stationary_filter(numeric)
    dropped_forced_features: List[str] = []
    if force_drop_features:
        drop_set = {str(x) for x in force_drop_features}
        dropped_forced_features = [c for c in numeric.columns if c in drop_set]
        if dropped_forced_features:
            numeric = numeric.drop(columns=dropped_forced_features, errors="ignore")
    check_no_label_leakage(numeric)
    missing = [c for c in _required_orb_columns(orb_minutes) if c not in numeric.columns]
    if missing:
        logger.warning("Model role %s missing mandatory ORB columns: %s", model_role, missing)

    if split_indices is None:
        X_train, X_val, X_test, y_train, y_val, y_test = _three_way_time_split(
            numeric,
            y,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        raw_splits = Phase2SplitIndices(
            train_index=X_train.index.copy(),
            val_index=X_val.index.copy(),
            test_index=X_test.index.copy(),
        )
    else:
        raw_splits = split_indices
    safe_splits = _apply_purge_embargo(
        raw_splits,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    X_train = numeric.loc[safe_splits.train_index].copy()
    X_val = numeric.loc[safe_splits.val_index].copy()
    X_test = numeric.loc[safe_splits.test_index].copy()
    y_train = y.loc[safe_splits.train_index].copy()
    y_val = y.loc[safe_splits.val_index].copy()
    y_test = y.loc[safe_splits.test_index].copy()
    min_samples = 25
    if len(X_val) < min_samples or len(X_test) < min_samples:
        raise ValueError(
            f"Validation/test splits too small after purge/embargo "
            f"(need >= {min_samples} samples each); add data or adjust splits"
        )

    fit_weight = None
    if sample_weight is not None:
        fit_weight = pd.to_numeric(sample_weight.reindex(X_train.index), errors="coerce").fillna(1.0).to_numpy()

    dropped_ks_features: List[Dict[str, Any]] = []
    ks_limit = float(drift_ks_stat_max or 0.0)
    if ks_limit > 0.0 and X_train.shape[1] > 0:
        to_drop, dropped_ks_features = _drift_features_to_drop(
            X_train,
            X_val,
            ks_limit=ks_limit,
        )
        if to_drop and len(to_drop) < X_train.shape[1]:
            X_train = X_train.drop(columns=to_drop, errors="ignore")
            X_val = X_val.drop(columns=to_drop, errors="ignore")
            X_test = X_test.drop(columns=to_drop, errors="ignore")

    selected_top_features: List[str] = []
    if top_n_features is not None and int(top_n_features) > 0 and X_train.shape[1] > int(top_n_features):
        try:
            top_n = int(top_n_features)
            scores = mutual_info_classif(
                X_train.to_numpy(dtype=float),
                y_train.to_numpy(dtype=int),
                discrete_features=False,
                random_state=random_state,
            )
            ranked = sorted(zip(X_train.columns.tolist(), scores.tolist()), key=lambda x: float(x[1]), reverse=True)
            selected_top_features = [name for name, _ in ranked[:top_n]]
            if selected_top_features:
                X_train = X_train[selected_top_features].copy()
                X_val = X_val[selected_top_features].copy()
                X_test = X_test[selected_top_features].copy()
        except Exception:
            selected_top_features = []

    X_early, X_cal, y_early, y_cal = _split_calibration_subset(
        X_val,
        y_val,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )

    computed_scale_pos = _scale_pos_weight(y_train)
    scale_pos = computed_scale_pos
    if scale_pos_weight is not None:
        try:
            scale_pos = float(scale_pos_weight)
        except (TypeError, ValueError):
            scale_pos = computed_scale_pos
    logger.info(
        "[%s] scale_pos_weight=%.3f (pos=%d neg=%d; computed=%.3f)",
        model_role,
        scale_pos,
        int((y_train == 1).sum()),
        int((y_train == 0).sum()),
        computed_scale_pos,
    )
    xgb_params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        scale_pos_weight=scale_pos,
        n_jobs=-1,
    )
    if use_gpu:
        # XGBoost >= 2.x uses `device='cuda'` with `tree_method='hist'`.
        xgb_params["device"] = "cuda"
    if min_child_weight is not None:
        try:
            xgb_params["min_child_weight"] = float(min_child_weight)
        except (TypeError, ValueError):
            pass

    estimator = XGBClassifier(**xgb_params)
    eval_set = [(X_early, y_early)]
    try:
        estimator.fit(
            X_train,
            y_train,
            sample_weight=fit_weight,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
    except TypeError:
        estimator.fit(X_train, y_train, sample_weight=fit_weight, eval_set=eval_set, verbose=False)

    # Train on GPU when requested, then switch predictor device to CPU for
    # calibration/inference with pandas inputs to avoid device-mismatch warnings.
    if use_gpu:
        try:
            estimator.set_params(device="cpu")
        except Exception:
            pass

    calibrated, calibration_meta = _calibrate_classifier(estimator, X_cal, y_cal, method=calibration_method)
    p_cal = calibrated.predict_proba(X_cal)[:, 1]
    p_test = calibrated.predict_proba(X_test)[:, 1]

    train_start, train_end = _index_bounds(X_train.index)
    early_start, early_end = _index_bounds(X_early.index)
    cal_start, cal_end = _index_bounds(X_cal.index)
    test_start, test_end = _index_bounds(X_test.index)

    metrics: Dict[str, Any] = {
        "roc_auc_val": _safe_metric(roc_auc_score, y_cal, p_cal),
        "avg_precision_val": _safe_metric(average_precision_score, y_cal, p_cal),
        "log_loss_val": _safe_metric(log_loss, y_cal, p_cal),
        "brier_score_val": _safe_metric(brier_score_loss, y_cal, p_cal),
        "roc_auc_test": _safe_metric(roc_auc_score, y_test, p_test),
        "avg_precision_test": _safe_metric(average_precision_score, y_test, p_test),
        "log_loss_test": _safe_metric(log_loss, y_test, p_test),
        "brier_score_test": _safe_metric(brier_score_loss, y_test, p_test),
        "n_train": len(X_train),
        "n_val": len(X_cal),
        "n_test": len(X_test),
        "feature_count": numeric.shape[1],
        "feature_count_after_filtering": int(X_train.shape[1]),
        "train_start": train_start,
        "train_end": train_end,
        "val_start": cal_start,
        "val_end": cal_end,
        "early_stop_start": early_start,
        "early_stop_end": early_end,
        "test_start": test_start,
        "test_end": test_end,
        "event_filter": event_filter,
        "label_info": label_info,
        "model_role": model_role,
        "calibration": calibration_meta,
        "dropped_forced_features": dropped_forced_features,
        "dropped_ks_features": dropped_ks_features,
        "selected_top_features": selected_top_features,
        "purge_bars": int(max(purge_bars, 0)),
        "embargo_bars": int(max(embargo_bars, 0)),
    }

    recommended_thresholds: Dict[str, float] = {"p_setup": 0.55, "p_long": 0.58, "p_short": 0.58}

    if model_role == "setup":
        recommended_thresholds["p_setup"] = _trade_rate_threshold(p_cal)
        try:
            thresholds = [0.50, 0.55, 0.60, 0.65]
            for th in thresholds:
                preds_val = (p_cal >= th).astype(int)
                metrics[f"precision_{th:.2f}_val"] = float(precision_score(y_cal, preds_val, zero_division=0))
                metrics[f"recall_{th:.2f}_val"] = float(recall_score(y_cal, preds_val, zero_division=0))
            eval_threshold = recommended_thresholds["p_setup"]
            metrics["trade_threshold"] = float(eval_threshold)
            preds_test = (p_test >= eval_threshold).astype(int)
            metrics["precision_trade_test"] = float(precision_score(y_test, preds_test, zero_division=0))
            metrics["recall_trade_test"] = float(recall_score(y_test, preds_test, zero_division=0))
            metrics["pr_auc_val"] = _safe_metric(average_precision_score, y_cal, p_cal)
        except Exception:
            metrics["precision_trade_test"] = None
            metrics["recall_trade_test"] = None
            metrics["pr_auc_val"] = None
    if model_role == "direction":
        try:
            preds = (p_test >= 0.5).astype(int)
            metrics["balanced_accuracy_test"] = float(balanced_accuracy_score(y_test, preds))
            preds_val = (p_cal >= 0.5).astype(int)
            metrics["balanced_accuracy_val"] = float(balanced_accuracy_score(y_cal, preds_val))
        except Exception:
            metrics["balanced_accuracy_test"] = None
            metrics["balanced_accuracy_val"] = None
        long_thr, short_thr = _direction_thresholds(p_cal, y_cal.to_numpy())
        recommended_thresholds["p_long"] = long_thr
        recommended_thresholds["p_short"] = short_thr

    if extra_metrics:
        metrics.update(extra_metrics)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    raw_artifact = save_path.with_name(f"{save_path.stem}_raw.joblib")
    joblib.dump(estimator, raw_artifact)
    joblib.dump(calibrated, save_path)

    feature_payload = _feature_contract_payload(X_train.columns, model_role=model_role)
    save_path.with_suffix(".features.json").write_text(json.dumps(feature_payload, indent=2))

    schema_path = save_path.with_suffix(".label_schema.json")
    schema_path.write_text(label_schema.json(indent=2))

    resolved_model_id = generate_model_id(
        instrument,
        label_schema.domain or runtime_labels.domain or "binary",
        label_schema.horizon_bars or horizon_minutes or label_cfg.horizon,
    )
    if model_id_suffix:
        resolved_model_id = f"{resolved_model_id}_{model_id_suffix}"

    bounds_path = save_path.with_suffix(".feature_bounds.json")
    meta = {
        "tz": tz,
        "rth_start": rth_start,
        "rth_end": rth_end,
        "orb_minutes": orb_minutes,
        "horizon": label_cfg.horizon,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "label_schema": label_schema.model_dump(),
        "model_id": resolved_model_id,
        "instrument": instrument.upper(),
        "model_role": model_role,
        "artifacts": {
            "calibrated_model": save_path.name,
            "raw_model": raw_artifact.name,
            "features_file": save_path.with_suffix(".features.json").name,
            "label_schema_file": schema_path.name,
            "feature_bounds_file": bounds_path.name,
        },
        "calibration": calibration_meta,
        "recommended_thresholds": recommended_thresholds,
        "metrics": metrics,
    }
    if split_meta:
        meta["split_meta"] = split_meta
    if sample_weight_meta:
        meta["sample_weighting"] = sample_weight_meta
    save_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    save_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))

    training_params = _training_params_dict(
        tz=tz,
        rth_start=rth_start,
        rth_end=rth_end,
        orb_minutes=orb_minutes,
        horizon=label_cfg.horizon,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        label_threshold=label_cfg.threshold,
        use_log_label=label_cfg.use_log,
        use_htf_trend_aware=False,
        early_stopping_rounds=early_stopping_rounds,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        use_gpu=use_gpu,
    )
    if split_meta:
        training_params["date_splits"] = split_meta
    if sample_weight_meta:
        training_params["sample_weighting"] = sample_weight_meta
    record = {
        "model_path": str(save_path),
        "model_role": model_role,
        "training_config": {
            "params": training_params,
            "metrics": metrics,
        },
    }
    save_path.with_suffix(".registry.json").write_text(json.dumps(record, indent=2))
    split_indices = Phase2SplitIndices(
        train_index=X_train.index.copy(),
        val_index=X_cal.index.copy(),
        test_index=X_test.index.copy(),
    )
    logger.info("[%s] metrics: %s", model_role, json.dumps(metrics))
    return metrics, split_indices


def _setup_schema(label_cfg, runtime_labels) -> LabelSchema:
    return LabelSchema(
        domain="setup_binary",
        horizon_bars=label_cfg.horizon,
        trend_ma_window=runtime_labels.trend_ma_window,
        trend_slope_window=runtime_labels.trend_slope_window,
        drop_flats=False,
        positive_label=1,
        negative_label=0,
        params={
            "source": "phase2_setup",
            "mapping": {"1": "TRADE", "0": "FLAT"},
            "threshold": float(label_cfg.threshold),
        },
    )


def _direction_schema(label_cfg, runtime_labels) -> LabelSchema:
    return LabelSchema(
        domain="directional_binary",
        horizon_bars=label_cfg.horizon,
        trend_ma_window=runtime_labels.trend_ma_window,
        trend_slope_window=runtime_labels.trend_slope_window,
        drop_flats=True,
        positive_label=1,
        negative_label=0,
        params={
            "source": "phase2_direction",
            "mapping": {"1": "LONG", "0": "SHORT"},
            "threshold": float(label_cfg.threshold),
        },
    )


def _phase2_label_domain(runtime_labels, use_htf: bool) -> str:
    base = str(runtime_labels.domain or "ternary").lower()
    if use_htf:
        return "trend_aware_ternary"
    return base


def train_phase2_models(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    runtime_labels = load_app_config().labels
    label_cfg = _label_config_from_app(
        runtime_labels,
        horizon=args.horizon,
        threshold=args.label_threshold,
        use_log_label=args.label_log,
        use_htf_trend_aware=args.htf_trend_aware,
    )
    df_raw = pd.read_csv(args.csv)
    feats = build_features(
        df_raw,
        tz=args.tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_minutes=args.orb_minutes,
        csv_naive_is_utc=args.csv_naive_is_utc,
    )
    mtf_timeframes = _parse_mtf_timeframes(getattr(args, "mtf_timeframes", []))
    mtf_feature_columns: List[str] = []
    if mtf_timeframes:
        feats, mtf_feature_columns = _attach_multi_tf_features(
            feats,
            df_raw=df_raw,
            timeframes=mtf_timeframes,
            tz=args.tz,
            csv_naive_is_utc=args.csv_naive_is_utc,
        )
    feats = _apply_event_filter(feats, args.event_filter)
    if feats.empty:
        raise ValueError("Feature frame is empty after preprocessing.")

    scheme = _phase2_label_domain(runtime_labels, args.htf_trend_aware)
    labels = _build_phase2_labels(
        feats,
        horizon=label_cfg.horizon,
        threshold=label_cfg.threshold,
        use_log=label_cfg.use_log,
        scheme=scheme,
        mode=getattr(args, "label_mode", "horizon"),
        instrument=args.instrument,
        max_hold_bars=getattr(args, "label_max_hold_bars", label_cfg.horizon),
        commission_per_contract=getattr(args, "label_commission_per_contract", 2.0),
        slippage_ticks=getattr(args, "label_slippage_ticks", 1.0),
        setup_threshold_multiplier=float(getattr(args, "setup_threshold_multiplier", 1.75)),
    )
    trimmed = labels.trimmed_frame
    y_setup = labels.setup
    y_direction = labels.direction
    direction_info = labels.direction_info

    if y_setup.nunique() < 2:
        raise ValueError("Setup labels contain a single class; ensure flats are present (event_filter=all).")

    setup_counts = {
        "trade": int((y_setup == 1).sum()),
        "flat": int((y_setup == 0).sum()),
    }
    logger.info("[phase2] Setup label counts: %s", setup_counts)
    logger.info("[phase2] Base label counts (long/short/flat): %s", direction_info.get("signed_counts"))

    setup_schema = _setup_schema(label_cfg, runtime_labels)
    direction_schema = _direction_schema(label_cfg, runtime_labels)

    setup_features = trimmed.loc[y_setup.index]
    direction_features = trimmed.loc[y_direction.index]
    label_lookahead_bars = (
        int(getattr(args, "label_max_hold_bars", label_cfg.horizon))
        if getattr(args, "label_mode", "horizon") == "exec"
        else int(label_cfg.horizon)
    )
    purge_bars = getattr(args, "purge_bars", None)
    purge_bars = label_lookahead_bars if purge_bars is None else max(int(purge_bars), 0)
    embargo_bars = max(int(getattr(args, "embargo_bars", 0)), 0)

    use_time_splits = any(
        getattr(args, key, None)
        for key in (
            "train_start",
            "train_end",
            "val_start",
            "val_end",
            "test_start",
            "test_end",
        )
    )
    setup_split_indices = None
    direction_split_indices = None
    split_meta = None
    if use_time_splits:
        setup_split_indices = _split_indices_by_time(
            setup_features,
            tz=args.tz,
            train_start=args.train_start,
            train_end=args.train_end,
            val_start=args.val_start,
            val_end=args.val_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        direction_split_indices = _split_indices_by_time(
            direction_features,
            tz=args.tz,
            train_start=args.train_start,
            train_end=args.train_end,
            val_start=args.val_start,
            val_end=args.val_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        split_meta = {
            "train_start": args.train_start,
            "train_end": args.train_end,
            "val_start": args.val_start,
            "val_end": args.val_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "purge_bars": purge_bars,
            "embargo_bars": embargo_bars,
        }
        # Align setup vs direction validation start timestamps.
        # Direction labels drop flats, which can shift the first available bar in the val slice.
        try:
            setup_ts = _split_ts_metrics(setup_features, setup_split_indices, args.tz)
            direction_ts = _split_ts_metrics(direction_features, direction_split_indices, args.tz)
            setup_val_start = setup_ts.get("val_start_ts")
            direction_val_start = direction_ts.get("val_start_ts")
            if setup_val_start and direction_val_start:
                setup_dt = pd.Timestamp(setup_val_start)
                direction_dt = pd.Timestamp(direction_val_start)
                delta_days = abs((setup_dt - direction_dt).days)
                if delta_days > 7:
                    logger.warning(
                        "[phase2] Val start mismatch (>%dd): setup=%s direction=%s. "
                        "Shrinking direction validation window to match setup val start.",
                        7,
                        setup_val_start,
                        direction_val_start,
                    )
                if direction_dt < setup_dt:
                    dt_series = _datetime_series(direction_features, args.tz)
                    keep_mask = dt_series.loc[direction_split_indices.val_index] >= setup_dt
                    new_val_index = direction_split_indices.val_index[keep_mask.to_numpy()]
                    if len(new_val_index) == 0:
                        raise ValueError("Direction validation slice became empty after val_start alignment.")
                    direction_split_indices = Phase2SplitIndices(
                        train_index=direction_split_indices.train_index,
                        val_index=new_val_index,
                        test_index=direction_split_indices.test_index,
                    )
        except Exception as exc:
            logger.warning("[phase2] Failed to align direction val_start to setup: %s", exc)

    setup_weights = None
    direction_weights = None
    sample_weight_meta = None
    recency_mode = getattr(args, "recency_weighting", "none")
    if recency_mode == "exponential":
        recency_mode = "exp"
    if recency_mode and recency_mode != "none":
        if setup_split_indices is None or direction_split_indices is None:
            raise ValueError("Recency weighting requires explicit time splits.")
        setup_dt = _datetime_series(setup_features.loc[setup_split_indices.train_index], args.tz)
        direction_dt = _datetime_series(direction_features.loc[direction_split_indices.train_index], args.tz)
        setup_weights = _recency_weights(
            setup_dt,
            mode=recency_mode,
            max_weight=float(getattr(args, "recency_max_weight", 2.0)),
            half_life_days=float(getattr(args, "recency_half_life_days", 180.0)),
        )
        direction_weights = _recency_weights(
            direction_dt,
            mode=recency_mode,
            max_weight=float(getattr(args, "recency_max_weight", 2.0)),
            half_life_days=float(getattr(args, "recency_half_life_days", 180.0)),
        )
        sample_weight_meta = {
            "mode": recency_mode,
            "max_weight": float(getattr(args, "recency_max_weight", 2.0)),
            "half_life_days": float(getattr(args, "recency_half_life_days", 180.0)),
        }

    setup_metrics, setup_splits = _train_binary_xgb(
        setup_features,
        y_setup,
        save_path=Path(args.setup_model_path),
        runtime_labels=runtime_labels,
        label_cfg=label_cfg,
        label_info={"class_counts": setup_counts},
        label_schema=setup_schema,
        instrument=args.instrument,
        tz=args.tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_minutes=args.orb_minutes,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        event_filter=args.event_filter,
        model_role="setup",
        horizon_minutes=args.horizon,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=int(getattr(args, "setup_early_stopping_rounds", args.early_stopping_rounds)),
        min_child_weight=float(getattr(args, "setup_min_child_weight", getattr(args, "min_child_weight", 1.0))),
        scale_pos_weight=float(getattr(args, "setup_scale_pos_weight", 0.0) or 0.0) or None,
        random_state=args.random_state,
        use_gpu=args.gpu,
        model_id_suffix="setup",
        split_indices=setup_split_indices,
        sample_weight=setup_weights,
        split_meta=split_meta,
        sample_weight_meta=sample_weight_meta,
        force_drop_features=_parse_feature_list(getattr(args, "force_drop_features", "")),
        top_n_features=getattr(args, "setup_feature_top_n", None),
        drift_ks_stat_max=getattr(args, "drift_ks_stat_max", None),
        calibration_method=str(getattr(args, "calibration_method", "sigmoid") or "sigmoid"),
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )

    stacked_probs: Optional[pd.Series] = None
    if getattr(args, "stack_setup_prob", False):
        stacked_probs = _predict_setup_stack_probs(Path(args.setup_model_path), trimmed)

    if stacked_probs is not None:
        direction_features = direction_features.copy()
        stacked_subset = stacked_probs.reindex(direction_features.index)
        if stacked_subset.isna().any():
            fill_value = float(stacked_subset.mean(skipna=True))
            if not np.isfinite(fill_value):
                fill_value = 0.5
            stacked_subset = stacked_subset.fillna(fill_value)
        direction_features["stack_setup_prob"] = stacked_subset.to_numpy()

    direction_metrics, direction_splits = _train_binary_xgb(
        direction_features,
        y_direction,
        save_path=Path(args.direction_model_path),
        runtime_labels=runtime_labels,
        label_cfg=label_cfg,
        label_info=direction_info,
        label_schema=direction_schema,
        instrument=args.instrument,
        tz=args.tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_minutes=args.orb_minutes,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        event_filter=args.event_filter,
        model_role="direction",
        horizon_minutes=args.horizon,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=int(getattr(args, "direction_max_depth", args.max_depth)),
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=int(getattr(args, "direction_early_stopping_rounds", args.early_stopping_rounds)),
        min_child_weight=float(getattr(args, "direction_min_child_weight", getattr(args, "min_child_weight", 1.0))),
        random_state=args.random_state,
        use_gpu=args.gpu,
        model_id_suffix="dir",
        split_indices=direction_split_indices,
        sample_weight=direction_weights,
        split_meta=split_meta,
        sample_weight_meta=sample_weight_meta,
        force_drop_features=_parse_feature_list(getattr(args, "force_drop_features", "")),
        top_n_features=getattr(args, "direction_feature_top_n", None),
        drift_ks_stat_max=getattr(args, "drift_ks_stat_max", None),
        calibration_method=str(getattr(args, "calibration_method", "sigmoid") or "sigmoid"),
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )

    setup_metrics.update(_split_ts_metrics(setup_features, setup_splits, args.tz))
    direction_metrics.update(_split_ts_metrics(direction_features, direction_splits, args.tz))

    setup_train_frame = setup_features.loc[setup_splits.train_index]
    dir_train_frame = direction_features.loc[direction_splits.train_index]
    setup_feature_payload = json.loads(
        Path(args.setup_model_path).with_suffix(".features.json").read_text(encoding="utf-8")
    )
    direction_feature_payload = json.loads(
        Path(args.direction_model_path).with_suffix(".features.json").read_text(encoding="utf-8")
    )
    setup_feature_names = [str(name) for name in setup_feature_payload.get("features", [])]
    direction_feature_names = [str(name) for name in direction_feature_payload.get("features", [])]
    setup_bounds = _feature_bounds(
        setup_train_frame.reindex(columns=setup_feature_names).select_dtypes(include=[np.number])
    )
    dir_bounds = _feature_bounds(
        dir_train_frame.reindex(columns=direction_feature_names).select_dtypes(include=[np.number])
    )
    Path(args.setup_model_path).with_suffix(".feature_bounds.json").write_text(json.dumps(setup_bounds, indent=2))
    Path(args.direction_model_path).with_suffix(".feature_bounds.json").write_text(json.dumps(dir_bounds, indent=2))

    if getattr(args, "stack_setup_prob", False):
        meta_path = Path(args.direction_model_path).with_suffix(".meta.json")
        try:
            meta_payload = json.loads(meta_path.read_text())
        except Exception:
            meta_payload = {}
        meta_payload["stack_setup_prob"] = True
        meta_path.write_text(json.dumps(meta_payload, indent=2))
        direction_metrics["stack_setup_prob"] = True

    extras: Dict[str, Any] = {
        "thresholds": {},
        "trading_val": None,
        "trading_test": None,
        "walkforward": None,
        "live_shadow_gate": None,
        "promotion_blocked": False,
        "promotion_blocked_reason": None,
        "rejected": False,
        "rejected_reason": None,
        "direction_diagnostics": None,
        "threshold_diagnostics": None,
        "mtf_feature_columns": mtf_feature_columns,
    }
    extras["live_shadow_gate"] = _load_live_shadow_gate(getattr(args, "live_shadow_summary_path", None))
    if bool(getattr(args, "require_safe_shadow_pass", False)) and not bool((extras["live_shadow_gate"] or {}).get("passed")):
        extras["promotion_blocked"] = True
        extras["promotion_blocked_reason"] = "live_shadow_gate_failed"

    artifact_dir = Path(args.setup_model_path).resolve().parent
    commission = getattr(
        args,
        "commission_per_contract",
        getattr(args, "label_commission_per_contract", 2.0),
    )
    slippage = getattr(args, "slippage_ticks", getattr(args, "label_slippage_ticks", 1.0))
    min_trades_val = getattr(args, "min_trades_val", 30)
    max_flip_rate = getattr(args, "max_flip_rate", 0.2)
    objective = getattr(args, "threshold_objective", "sharpe")
    trade_window_start = getattr(args, "trade_window_start", ENGINE.trade_window_start)
    trade_window_end = getattr(args, "trade_window_end", ENGINE.trade_window_end)
    max_hold_bars = getattr(args, "label_max_hold_bars", label_cfg.horizon)

    setup_prob_series = pd.Series(
        _predict_long_probabilities(Path(args.setup_model_path), trimmed),
        index=trimmed.index,
    )
    dir_feature_frame = trimmed.copy()
    direction_meta_path = Path(args.direction_model_path).with_suffix(".meta.json")
    try:
        direction_meta_payload = json.loads(direction_meta_path.read_text())
    except Exception:
        direction_meta_payload = {}
    requires_stack = bool(direction_meta_payload.get("stack_setup_prob") or getattr(args, "stack_setup_prob", False))
    if requires_stack:
        dir_feature_frame["stack_setup_prob"] = setup_prob_series.reindex(dir_feature_frame.index).to_numpy()
    dir_prob_series = pd.Series(
        _predict_long_probabilities(Path(args.direction_model_path), dir_feature_frame),
        index=dir_feature_frame.index,
    )

    val_idx = setup_splits.val_index
    test_idx = setup_splits.test_index
    val_frame = trimmed.loc[val_idx]
    test_frame = trimmed.loc[test_idx]

    dir_val_probs = dir_prob_series.reindex(val_frame.index)
    diagnostics = _direction_probability_diagnostics(dir_val_probs)
    extras["direction_diagnostics"] = diagnostics
    if diagnostics["extreme_frac"] > 0.40:
        extras["rejected"] = True
        extras["rejected_reason"] = "direction_probability_saturation"

    instrument = instrument_by_alias(args.instrument)
    sim_cfg = Phase2SimConfig(
        tz=args.tz,
        trade_window_start=trade_window_start,
        trade_window_end=trade_window_end,
        point_value=instrument.point_value,
        tick_value=instrument.tick_value,
        contracts=1,
        max_hold_bars=max_hold_bars,
        commission_per_contract=commission,
        slippage_ticks=slippage,
    )
    decision_policy = Phase2DecisionPolicy(
        entry_trend_filter=str(getattr(args, "entry_trend_filter", "none") or "none"),
        min_signal_persistence_bars=int(getattr(args, "min_signal_persistence_bars", 1) or 1),
        cooldown_bars_after_flip=int(getattr(args, "cooldown_bars_after_flip", 0) or 0),
    )

    thresholds: Dict[str, float] = {}
    trading_val: Optional[Dict[str, Any]] = None
    trading_test: Optional[Dict[str, Any]] = None
    close_metrics: Optional[Dict[str, Any]] = None

    if not extras["rejected"]:
        best_thresholds, best_sim_val, best_audit_val, rejection_reason, threshold_diagnostics = optimize_phase2_thresholds(
            val_frame,
            setup_prob_series,
            dir_prob_series,
            sim_cfg=sim_cfg,
            decision_policy=decision_policy,
            objective=objective,
            min_trades_val=min_trades_val,
            max_flip_rate=max_flip_rate,
            max_bad_fade_rate=float(getattr(args, "max_bad_fade_rate", 0.18) or 0.18),
            setup_threshold_start=float(getattr(args, "setup_threshold_start", 0.20) or 0.20),
            setup_threshold_end=float(getattr(args, "setup_threshold_end", 0.85) or 0.85),
            setup_threshold_step=float(getattr(args, "setup_threshold_step", 0.03) or 0.03),
            direction_threshold_start=float(getattr(args, "direction_threshold_start", 0.55) or 0.55),
            direction_threshold_end=float(getattr(args, "direction_threshold_end", 0.82) or 0.82),
            direction_threshold_step=float(getattr(args, "direction_threshold_step", 0.02) or 0.02),
        )
        extras["threshold_diagnostics"] = threshold_diagnostics
        if not best_thresholds or not best_sim_val:
            extras["rejected"] = True
            extras["rejected_reason"] = rejection_reason or "threshold_optimization_failed"
        else:
            thresholds = best_thresholds
            trading_val = _trading_summary(best_sim_val)
            test_setup_probs = setup_prob_series.reindex(test_frame.index).to_numpy(dtype=float)
            test_dir_probs = dir_prob_series.reindex(test_frame.index).to_numpy(dtype=float)
            df_val_phase = phase2_decisions(
                val_frame.copy(),
                setup_prob_series.reindex(val_frame.index).to_numpy(dtype=float),
                dir_prob_series.reindex(val_frame.index).to_numpy(dtype=float),
                thresholds,
                policy=decision_policy,
            )
            df_test_phase = phase2_decisions(
                test_frame.copy(),
                test_setup_probs,
                test_dir_probs,
                thresholds,
                policy=decision_policy,
            )
            val_sim_final = simulate_trades(df_val_phase, thresholds, cfg=sim_cfg)
            test_sim_final = simulate_trades(df_test_phase, thresholds, cfg=sim_cfg)
            trading_val = _trading_summary(val_sim_final)
            trading_test = _trading_summary(test_sim_final)
            extras["trading_val"] = trading_val
            extras["trading_test"] = trading_test
            extras["thresholds"] = thresholds
            extras["behavior_audit_val"] = best_audit_val or _phase2_behavior_audit(df_val_phase)
            extras["behavior_audit_test"] = _phase2_behavior_audit(df_test_phase)
            holdout_index = val_frame.index.append(test_frame.index).unique()
            holdout_frame = trimmed.loc[holdout_index].sort_index()
            extras["walkforward"] = _walkforward_replay_summary(
                holdout_frame,
                setup_probs=setup_prob_series,
                dir_probs=dir_prob_series,
                thresholds=thresholds,
                sim_cfg=sim_cfg,
                windows=int(getattr(args, "walkforward_windows", 3) or 3),
                decision_policy=decision_policy,
            )
            if val_sim_final.get("trades") is not None:
                pd.DataFrame(val_sim_final["trades"]).to_csv(artifact_dir / "val_trades.csv", index=False)
            if test_sim_final.get("trades") is not None:
                pd.DataFrame(test_sim_final["trades"]).to_csv(artifact_dir / "test_trades.csv", index=False)

            full_phase = phase2_decisions(
                trimmed.copy(),
                setup_prob_series.reindex(trimmed.index).to_numpy(dtype=float),
                dir_prob_series.reindex(trimmed.index).to_numpy(dtype=float),
                thresholds,
                policy=decision_policy,
            )
            replay = simulate_trades(full_phase, thresholds, cfg=sim_cfg)
            close_frame, close_labels, close_label_info = _build_close_training_frame(
                full_phase,
                trades=list(replay.get("trades") or []),
                instrument=args.instrument,
                label_threshold=float(args.label_threshold),
                pnl_giveback_activate_r=float(getattr(args, "close_giveback_activate_r", 1.0)),
                pnl_giveback_close_r=float(getattr(args, "close_giveback_close_r", 0.5)),
                pnl_stall_bars=int(getattr(args, "close_stall_bars", 4)),
                pnl_stall_min_mfe_r=float(getattr(args, "close_stall_min_mfe_r", 0.25)),
                pnl_stall_close_below_r=float(getattr(args, "close_stall_close_below_r", -0.10)),
                pnl_severe_adverse_r=float(getattr(args, "close_severe_adverse_r", 0.90)),
                pnl_target_arm_min_hold_bars=int(getattr(args, "close_target_arm_min_hold_bars", 3)),
                pnl_target_arm_min_unrealized_r=float(getattr(args, "close_target_arm_min_unrealized_r", 0.75)),
            )
            if bool(getattr(args, "enable_close_training", True)):
                close_schema = _close_schema(label_cfg, runtime_labels)
                close_metrics, _ = _train_binary_xgb(
                    close_frame,
                    close_labels,
                    save_path=Path(args.close_model_path),
                    runtime_labels=runtime_labels,
                    label_cfg=label_cfg,
                    label_info=close_label_info,
                    label_schema=close_schema,
                    instrument=args.instrument,
                    tz=args.tz,
                    rth_start=args.rth_start,
                    rth_end=args.rth_end,
                    orb_minutes=args.orb_minutes,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    event_filter=args.event_filter,
                    model_role="close",
                    horizon_minutes=args.horizon,
                    n_estimators=args.n_estimators,
                    learning_rate=args.learning_rate,
                    max_depth=int(getattr(args, "close_max_depth", args.direction_max_depth)),
                    subsample=args.subsample,
                    colsample_bytree=args.colsample_bytree,
                    reg_alpha=args.reg_alpha,
                    reg_lambda=args.reg_lambda,
                    early_stopping_rounds=int(getattr(args, "close_early_stopping_rounds", args.direction_early_stopping_rounds)),
                    min_child_weight=float(getattr(args, "close_min_child_weight", getattr(args, "direction_min_child_weight", getattr(args, "min_child_weight", 1.0)))),
                    random_state=args.random_state,
                    use_gpu=args.gpu,
                    model_id_suffix="close",
                    force_drop_features=_parse_feature_list(getattr(args, "force_drop_features", "")),
                    top_n_features=getattr(args, "close_feature_top_n", None),
                    drift_ks_stat_max=getattr(args, "drift_ks_stat_max", None),
                    calibration_method=str(getattr(args, "calibration_method", "sigmoid") or "sigmoid"),
                    purge_bars=purge_bars,
                    embargo_bars=embargo_bars,
                )
                extras["close_metrics"] = close_metrics
                extras["close_replay_trade_count"] = int(len(replay.get("trades") or []))
                extras["manifest_path"] = str(
                    _write_phase2_candidate_manifest(
                        args=args,
                        artifact_dir=artifact_dir,
                        setup_metrics=setup_metrics,
                        direction_metrics=direction_metrics,
                        close_metrics=close_metrics,
                        extras=extras,
                    )
                )

    extras["rejected"] = bool(extras["rejected"])
    if extras.get("thresholds"):
        extras["manifest_path"] = str(
            _write_phase2_candidate_manifest(
                args=args,
                artifact_dir=artifact_dir,
                setup_metrics=setup_metrics,
                direction_metrics=direction_metrics,
                close_metrics=close_metrics,
                extras=extras,
            )
        )
    return setup_metrics, direction_metrics, extras


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train Phase-2 setup+direction XGB models.")
    ap.add_argument("--csv", required=True, help="Training OHLCV CSV path.")
    ap.add_argument("--setup-model-path", default=str(SETUP_MODEL_DEFAULT))
    ap.add_argument("--direction-model-path", default=str(DIR_MODEL_DEFAULT))
    ap.add_argument("--close-model-path", default=str(CLOSE_MODEL_DEFAULT))
    ap.add_argument("--instrument", default="ES")
    ap.add_argument("--tz", default="America/Denver")
    ap.add_argument("--rth-start", default="07:30")
    ap.add_argument("--rth-end", default="14:00")
    ap.add_argument("--orb-minutes", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--label-threshold", type=float, default=0.0015)
    ap.add_argument("--label-log", action="store_true")
    ap.add_argument("--label-mode", choices=["horizon", "exec"], default="horizon")
    ap.add_argument("--label-commission-per-contract", type=float, default=2.0)
    ap.add_argument("--label-slippage-ticks", type=float, default=1.0)
    ap.add_argument("--label-max-hold-bars", type=int, default=24)
    ap.add_argument("--htf-trend-aware", action="store_true")
    ap.add_argument("--event-filter", choices=["all", "setup_only"], default="all")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--train-start", default=None, help="Optional start date for training window (YYYY-MM-DD or timestamp).")
    ap.add_argument("--train-end", default=None, help="Optional end date for training window (YYYY-MM-DD or timestamp).")
    ap.add_argument("--val-start", default=None, help="Optional start date for validation window.")
    ap.add_argument("--val-end", default=None, help="Optional end date for validation window.")
    ap.add_argument("--test-start", default=None, help="Optional start date for test window.")
    ap.add_argument("--test-end", default=None, help="Optional end date for test window.")
    ap.add_argument(
        "--purge-bars",
        type=int,
        default=None,
        help="Bars removed before each later split (default: label lookahead/max hold).",
    )
    ap.add_argument(
        "--embargo-bars",
        type=int,
        default=0,
        help="Bars removed from the start of validation/test after each split boundary.",
    )
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--direction-max-depth", type=int, default=4)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-alpha", type=float, default=0.1)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--early-stopping-rounds", type=int, default=75, help="Setup early stopping rounds (direction has its own default).")
    ap.add_argument("--direction-early-stopping-rounds", type=int, default=50)
    ap.add_argument("--close-early-stopping-rounds", type=int, default=50)
    ap.add_argument("--min-child-weight", type=float, default=15.0, help="Setup min_child_weight (direction has its own default).")
    ap.add_argument("--direction-min-child-weight", type=float, default=10.0)
    ap.add_argument("--close-min-child-weight", type=float, default=10.0)
    ap.add_argument("--close-max-depth", type=int, default=4)
    ap.add_argument("--setup-scale-pos-weight", type=float, default=4.53)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--gpu", action="store_true")
    csv_tz = ap.add_mutually_exclusive_group()
    csv_tz.add_argument(
        "--csv-naive-is-utc",
        dest="csv_naive_is_utc",
        action="store_true",
        help="Treat naive CSV timestamps as UTC (default: local session tz).",
    )
    csv_tz.add_argument(
        "--csv-naive-is-local",
        dest="csv_naive_is_utc",
        action="store_false",
        help="Treat naive CSV timestamps as local session tz.",
    )
    ap.set_defaults(csv_naive_is_utc=False)
    ap.add_argument("--stack-setup-prob", action="store_true", help="Append setup model probabilities as a feature for direction training.")
    ap.add_argument("--mtf-timeframes", default="", help="Comma-separated higher-timeframe context to append from the base 5m feed, e.g. 15m,60m.")
    ap.add_argument("--threshold-objective", choices=["sharpe", "calmar", "ev"], default="sharpe")
    ap.add_argument("--min-trades-val", type=int, default=30)
    ap.add_argument("--max-flip-rate", type=float, default=0.2)
    ap.add_argument("--walkforward-windows", type=int, default=3)
    ap.add_argument("--live-shadow-summary-path", default="run_health_summary.json")
    ap.add_argument("--require-safe-shadow-pass", action="store_true")
    ap.add_argument("--setup-threshold-start", type=float, default=0.20)
    ap.add_argument("--setup-threshold-end", type=float, default=0.85)
    ap.add_argument("--setup-threshold-step", type=float, default=0.03)
    ap.add_argument("--direction-threshold-start", type=float, default=0.55)
    ap.add_argument("--direction-threshold-end", type=float, default=0.82)
    ap.add_argument("--direction-threshold-step", type=float, default=0.02)
    ap.add_argument("--setup-threshold-multiplier", type=float, default=1.75)
    ap.add_argument("--max-bad-fade-rate", type=float, default=0.18)
    ap.add_argument("--entry-trend-filter", choices=["none", "vwap_ema"], default="none")
    ap.add_argument("--min-signal-persistence-bars", type=int, default=1)
    ap.add_argument("--cooldown-bars-after-flip", type=int, default=0)
    ap.add_argument("--commission-per-contract", type=float, default=2.0)
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--close-threshold", type=float, default=0.60)
    ap.add_argument("--close-giveback-activate-r", type=float, default=1.0)
    ap.add_argument("--close-giveback-close-r", type=float, default=0.5)
    ap.add_argument("--close-stall-bars", type=int, default=4)
    ap.add_argument("--close-stall-min-mfe-r", type=float, default=0.25)
    ap.add_argument("--close-stall-close-below-r", type=float, default=-0.10)
    ap.add_argument("--close-severe-adverse-r", type=float, default=0.90)
    ap.add_argument("--close-target-arm-min-hold-bars", type=int, default=3)
    ap.add_argument("--close-target-arm-min-unrealized-r", type=float, default=0.75)
    ap.add_argument("--recency-weighting", choices=["none", "linear", "exp", "exponential"], default="exponential")
    ap.add_argument("--recency-max-weight", type=float, default=2.0)
    ap.add_argument("--recency-half-life-days", type=float, default=365.0)
    ap.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="isotonic")
    ap.add_argument("--force-drop-features", default="", help="Comma-separated feature names to drop before model fitting.")
    ap.add_argument("--setup-feature-top-n", type=int, default=25)
    ap.add_argument("--direction-feature-top-n", type=int, default=25)
    ap.add_argument("--close-feature-top-n", type=int, default=25)
    ap.add_argument("--drift-ks-stat-max", type=float, default=0.20)
    ap.add_argument("--enable-close-training", dest="enable_close_training", action="store_true")
    ap.add_argument("--disable-close-training", dest="enable_close_training", action="store_false")
    ap.set_defaults(enable_close_training=True)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if args.event_filter != "all":
        raise ValueError("Phase2 setup model requires event_filter=all to retain flats.")
    setup_metrics, direction_metrics, extras = train_phase2_models(args)
    logger.info("[phase2] Setup metrics summary: %s", json.dumps(setup_metrics))
    logger.info("[phase2] Direction metrics summary: %s", json.dumps(direction_metrics))
    if extras.get("close_metrics") is not None:
        logger.info("[phase2] Close metrics summary: %s", json.dumps(extras.get("close_metrics")))


if __name__ == "__main__":
    main()


