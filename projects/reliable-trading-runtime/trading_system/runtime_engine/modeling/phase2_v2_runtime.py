from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

_TRAIN_V2_MODULE: Any = None
TRAIN_V2_BARS_PER_DAY = 288
_TRAIN_V2_INTRADAY_REQUIREMENTS = {
    "5m": 50,
    "15m": 60,
}
_TRAIN_V2_DAILY_ROLLING_RE = re.compile(r"^(?:sma_day|dist_sma_day)_(\d+)_day$")
_TRAIN_V2_DAILY_RET_RE = re.compile(r"^ret_day_(\d+)_day$")


def _load_train_v2_module() -> Any:
    global _TRAIN_V2_MODULE
    if _TRAIN_V2_MODULE is None:
        _TRAIN_V2_MODULE = importlib.import_module("tools.train_v2")
    return _TRAIN_V2_MODULE


def is_train_v2_bundle(model: Any, meta: Optional[dict[str, Any]] = None) -> bool:
    if not isinstance(model, dict):
        return False
    if not {"ensemble", "scaler", "feature_names"} <= set(model.keys()):
        return False
    meta_model = str((meta or {}).get("model") or "").strip().lower()
    if meta_model and meta_model != "direction_v2_ensemble":
        return False
    return True


def infer_train_v2_history_requirements(feature_names: Optional[Sequence[str]]) -> dict[str, int]:
    required_bars = max(_TRAIN_V2_INTRADAY_REQUIREMENTS.values())
    required_days = 1
    for raw_name in feature_names or ():
        name = str(raw_name or "").strip()
        if not name:
            continue
        daily_roll = _TRAIN_V2_DAILY_ROLLING_RE.match(name)
        if daily_roll:
            required_days = max(required_days, int(daily_roll.group(1)))
            continue
        daily_ret = _TRAIN_V2_DAILY_RET_RE.match(name)
        if daily_ret:
            required_days = max(required_days, int(daily_ret.group(1)) + 1)
            continue
        if name == "ret_day_day":
            required_days = max(required_days, 2)
    if required_days > 1:
        required_bars = max(required_bars, required_days * TRAIN_V2_BARS_PER_DAY)
    return {
        "required_bars": int(required_bars),
        "required_unique_days": int(required_days),
    }


def classify_train_v2_warmup_issue(
    bad_features: Sequence[str],
    *,
    available_bars: int,
    unique_days: int,
    feature_names: Optional[Sequence[str]],
) -> Optional[dict[str, Any]]:
    normalized = [str(name or "").strip() for name in bad_features if str(name or "").strip()]
    if not normalized:
        return None
    requirements = infer_train_v2_history_requirements(feature_names)
    warmup_features: list[str] = []
    for name in normalized:
        daily_roll = _TRAIN_V2_DAILY_ROLLING_RE.match(name)
        if daily_roll and unique_days < int(daily_roll.group(1)):
            warmup_features.append(name)
            continue
        daily_ret = _TRAIN_V2_DAILY_RET_RE.match(name)
        if daily_ret and unique_days < (int(daily_ret.group(1)) + 1):
            warmup_features.append(name)
            continue
        if name == "ret_day_day" and unique_days < 2:
            warmup_features.append(name)
            continue
        if name.endswith("_tf15") and available_bars < _TRAIN_V2_INTRADAY_REQUIREMENTS["15m"]:
            warmup_features.append(name)
            continue
        if available_bars < _TRAIN_V2_INTRADAY_REQUIREMENTS["5m"]:
            warmup_features.append(name)
    if len(warmup_features) != len(normalized):
        return None
    return {
        "bad_features": normalized,
        "warmup_features": warmup_features,
        "available_bars": int(available_bars),
        "unique_days": int(unique_days),
        "required_bars": int(requirements["required_bars"]),
        "required_unique_days": int(requirements["required_unique_days"]),
    }


def _coerce_live_ohlcv(raw_bars: pd.DataFrame, *, tz: str) -> pd.DataFrame:
    frame = pd.DataFrame(raw_bars).copy()
    frame.rename(columns=lambda c: str(c).strip().title(), inplace=True)
    if "Datetime" not in frame.columns:
        raise KeyError("Train-v2 runtime requires a Datetime column in raw bars.")
    try:
        dt = pd.to_datetime(frame["Datetime"], errors="coerce")
    except ValueError:
        dt = pd.to_datetime(frame["Datetime"], utc=True, errors="coerce")
    frame = frame.loc[dt.notna()].copy()
    dt = dt.loc[dt.notna()]
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(tz)
    else:
        dt = dt.dt.tz_convert(tz)
    frame["Datetime"] = dt
    frame.set_index("Datetime", inplace=True)
    frame.sort_index(inplace=True)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    return frame


def build_train_v2_feature_frame(raw_bars: pd.DataFrame, *, tz: str) -> pd.DataFrame:
    train_v2 = _load_train_v2_module()
    df = _coerce_live_ohlcv(raw_bars, tz=tz)
    f5 = train_v2.build_features_5m(df)
    f15 = train_v2.build_features_15m(df)
    fd = train_v2.build_features_daily(df)
    out = pd.concat([f5, f15, fd], axis=1)
    out = out.reset_index()
    if out.columns[0] != "Datetime":
        out = out.rename(columns={out.columns[0]: "Datetime"})
    return out


def augment_feature_frame_with_train_v2(
    feats: pd.DataFrame,
    raw_bars: pd.DataFrame,
    *,
    tz: str,
) -> pd.DataFrame:
    v2 = build_train_v2_feature_frame(raw_bars, tz=tz)
    left = feats.copy().set_index("Datetime")
    right = v2.set_index("Datetime")
    for col in right.columns:
        left[col] = right[col].reindex(left.index)
    return left.reset_index()


def predict_train_v2_bundle_proba(model: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    train_v2 = _load_train_v2_module()
    feature_names = [str(col) for col in list(model.get("feature_names") or list(X.columns))]
    X_work = X.copy()
    for col in feature_names:
        if col not in X_work.columns:
            X_work[col] = 0.0
    X_work = X_work[feature_names].astype(float)
    scaler = model.get("scaler")
    if scaler is not None:
        scaled = pd.DataFrame(
            scaler.transform(X_work),
            index=X_work.index,
            columns=X_work.columns,
        )
    else:
        scaled = X_work
    return np.asarray(train_v2.ensemble_predict(model["ensemble"], scaled), dtype=float).reshape(-1)
