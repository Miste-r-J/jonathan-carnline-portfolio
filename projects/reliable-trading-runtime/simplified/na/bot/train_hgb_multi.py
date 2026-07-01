#!/usr/bin/env python3
from __future__ import annotations
"""
HistGradientBoosting trainer tailored for ES 5m models with strict
time-aware splits, ternary labels (SHORT/FLAT/LONG), class balancing,
and post-fit calibration.  Uses the project feature builder + labeling
utilities so downstream scoring stays aligned with live pipelines.
"""

import argparse
import json
import logging
import os
import typing as ty
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder




def _json_default(obj: object) -> object:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def _json_dumps(payload: object) -> str:
    return json.dumps(payload, indent=2, default=_json_default)




_GPU_SUPPORT: Optional[bool] = None
_GPU_WARNED = False


def _gpu_accel_available() -> bool:
    """Return True if xgboost GPU training is usable, caching the probe result."""
    global _GPU_SUPPORT
    if _GPU_SUPPORT is not None:
        return _GPU_SUPPORT

    try:  # pragma: no cover - hardware/env dependent
        import xgboost as xgb

        dmat = xgb.DMatrix(np.zeros((4, 1), dtype=np.float32), label=np.zeros(4, dtype=np.float32))
        xgb.train(
            {
                "tree_method": "gpu_hist",
                "max_depth": 1,
                "learning_rate": 0.1,
                "objective": "binary:logistic",
                "verbosity": 0,
            },
            dmat,
            num_boost_round=1,
        )
        _GPU_SUPPORT = True
    except Exception:
        _GPU_SUPPORT = False

    return _GPU_SUPPORT

# --- project imports (package form only) ---

try:
    from .config import CLOSE_COL, OPEN_COL, HIGH_COL, LOW_COL  # type: ignore
    from .features import build_features, mandatory_features, MANDATORY_FEATURES  # type: ignore
    from .labels_triple_barrier import TripleBarrierParams, make_triple_barrier_labels
    from .scalp_labels import scalp_label_series  # type: ignore
    from .threshold_opt import ThresholdOptParams, optimize_thresholds
    from .sim_eval import SimEvalParams, evaluate_policy_grouped
    from .train_simple_lr_gb import (  # type: ignore
        _get_datetime_series,
        _normalize_ohlcv_columns,
        _sanitize_features,
        _stationarity_filter,
        time_splits,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "The 'na' package is not importable. Install the repo (`python -m pip install -e .`) "
        "and run this trainer via `python -m na.bot.train_hgb_multi`."
    ) from exc

from ..common.label_schema import LabelSchema
from ..config.loader import load_app_config

try:
    MANDATORY_FEATURES  # type: ignore[misc]
except NameError:  # pragma: no cover
    MANDATORY_FEATURES = []  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

CLASS_NAME = {
    -1: "SHORT",
    0: "FLAT",
    1: "LONG",
}


def renorm_proba(P: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Clip and row-normalize probability arrays."""
    P = np.clip(P, eps, 1.0)
    s = P.sum(axis=1, keepdims=True)
    s = np.where(s <= eps, 1.0, s)
    return P / s


class OvRCalibratedFuser:
    """Fuse two calibrated binary models (LONG vs rest, SHORT vs rest) into SHORT/FLAT/LONG."""
    def __init__(self, model_long, model_short, order=("SHORT","FLAT","LONG")):
        self.model_long = model_long
        self.model_short = model_short
        self.order = order
        self.feature_names_in_ = getattr(model_long, "feature_names_in_", None)

    def predict_proba(self, X):
        import numpy as np
        p_long  = np.asarray(self.model_long.predict_proba(X))[:, 1]
        p_short = np.asarray(self.model_short.predict_proba(X))[:, 1]
        p_flat  = np.clip(1.0 - p_long - p_short, 1e-9, 1.0)
        P = np.stack([p_short, p_flat, p_long], axis=1)
        return renorm_proba(P)

    # for sklearn compatibility
    def fit(self, *a, **k): return self
    def get_params(self, deep=False): return {}
    def set_params(self, **k): return self


class TempScaler:
    """Multiclass temperature scaling on logits."""
    def __init__(self, T_init=1.0):
        self.T_ = float(T_init)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

    def fit(self, logits: np.ndarray, y: np.ndarray, T_min=0.5, T_max=5.0, steps=200):
        Ts = np.linspace(T_min, T_max, steps)
        best_T, best = None, np.inf
        y = y.astype(int)
        for T in Ts:
            P = self._softmax(logits / T)
            nll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean()
            if nll < best:
                best, best_T = nll, T
        self.T_ = float(best_T)
        return self

    def transform_proba(self, logits: np.ndarray) -> np.ndarray:
        return self._softmax(logits / self.T_)

def _predict_logits_generic(model, X, n_classes: int) -> np.ndarray:
    """Best-effort raw score extraction for temp scaling."""
    # XGBoost margins (preferred)
    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None:
            margins = booster.inplace_predict(X, predict_type="margin")
            Z = np.asarray(margins).reshape(-1, n_classes)
            return Z
    except Exception:
        pass
    # scikit-learn decision_function (HGB etc.)
    try:
        z = model.decision_function(X)  # may be (n,) for binary or (n,C) for multiclass
        Z = np.asarray(z)
        if Z.ndim == 1 and n_classes == 2:
            Z = np.c_[-Z, Z]
        return Z
    except Exception:
        pass
    # Fallback: log-probabilities
    P = np.asarray(model.predict_proba(X))
    if P.ndim == 1 or P.shape[1] == 1:  # binary proba to 2-col
        P = np.c_[1 - P, P]
    return np.log(np.clip(P, 1e-12, 1.0))


def pretty_label(val: int) -> str:
    return CLASS_NAME.get(int(val), str(int(val)))


def compute_class_weights(y: pd.Series) -> Dict[int, float]:
    counts = y.value_counts()
    if counts.empty:
        return {}
    total = counts.sum()
    n_classes = len(counts)
    return {int(cls): float(total / (n_classes * cnt)) for cls, cnt in counts.items()}


def _class_accuracy(y_true: np.ndarray, preds: np.ndarray, cls: Optional[int]) -> Optional[float]:
    if cls is None:
        return None
    mask = y_true == cls
    if not np.any(mask):
        return None
    return float(np.mean(preds[mask] == cls))


def _selection_score(
    metric: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    labels: Sequence[int],
    long_code: Optional[int],
    short_code: Optional[int],
) -> Tuple[float, bool]:
    """Return (score, greater_is_better) for model selection."""
    name = metric.lower()
    if name == "logloss":
        loss = log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6), labels=labels)
        return -float(loss), True  # negate so higher is better
    preds = np.argmax(proba, axis=1)
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, preds)), True
    if name == "long_short_accuracy":
        accs = []
        for cls in (short_code, long_code):
            cls_acc = _class_accuracy(y_true, preds, cls)
            if cls_acc is not None:
                accs.append(cls_acc)
        if not accs:
            return 0.0, True
        return float(np.mean(accs)), True
    raise ValueError(f"Unknown selection metric '{metric}'")


def make_label_frame(
    close: pd.Series,
    *,
    horizon: int,
    threshold: float,
    scheme: str,
    use_log: bool,
    quantile: Optional[float] = None,
) -> Tuple[pd.DataFrame, float]:
    """Replicates bot.labeling.make_labels without requiring config constants."""
    if use_log:
        fwd_ret = np.log(close.shift(-horizon)) - np.log(close)
    else:
        fwd_ret = close.shift(-horizon) / close - 1.0

    threshold_used = float(threshold)
    if quantile is not None:
        q = float(quantile)
        if not 0.0 < q < 0.5:
            raise ValueError("--label-quantile must lie in (0, 0.5)")
        ret_no_nan = fwd_ret.dropna()
        if ret_no_nan.empty:
            raise ValueError("Cannot auto-compute threshold: forward returns are empty.")
        pos_thr = float(ret_no_nan.quantile(1.0 - q))
        neg_thr = float(ret_no_nan.quantile(q))
        threshold_used = float(max(abs(pos_thr), abs(neg_thr)))
        # Guard against degenerate thresholds (e.g., flat returns)
        if threshold_used <= 0.0:
            threshold_used = float(ret_no_nan.abs().quantile(1.0 - q))
        if threshold_used <= 0.0:
            raise ValueError("Auto-computed label threshold is non-positive; adjust --label-quantile.")
        print(f"[INFO] Auto-selected return threshold {threshold_used:.6f} from quantile {q:.3f}")

    if scheme == "binary":
        target = (fwd_ret > threshold_used).astype(np.int8)
    elif scheme == "ternary":
        target = np.where(
            fwd_ret > threshold_used,
            1,
            np.where(fwd_ret < -threshold_used, -1, 0),
        ).astype(np.int8)
    elif scheme == "direction":
        signed = np.sign(fwd_ret).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target = signed.astype(np.int8)
    else:
        raise ValueError(f"Unknown label scheme '{scheme}'")

    df = pd.DataFrame({"fwd_ret": fwd_ret, "target": target})
    if horizon > 0:
        df = df.iloc[:-horizon]
    return df, threshold_used


def encode_counts(counter: Iterable[int], name_map: Dict[int, str]) -> Dict[str, int]:
    print(f"[DEBUG] encode_counts called with counter length: {len(list(counter))}")
    try:
        print(f"[DEBUG] Counter is defined: {Counter}")
    except NameError as e:
        print(f"[DEBUG] Counter not defined: {e}")
    data = Counter(int(v) for v in counter)
    out: Dict[str, int] = {}
    for code, count in data.items():
        out[name_map.get(int(code), str(code))] = int(count)
    return out


def build_code_to_name(
    encoded_classes: Sequence[int],
    *,
    drop_flat: bool,
    scheme: str,
) -> Dict[int, str]:
    """Map encoded label codes to human-readable names."""
    base: Dict[int, str]
    if drop_flat:
        base = {0: "SHORT", 1: "LONG"}
    elif scheme in {"ternary", "direction"}:
        base = {0: "SHORT", 1: "FLAT", 2: "LONG"}
    elif scheme == "binary":
        base = {0: "NEG", 1: "POS"}
    else:
        base = {}

    mapped: Dict[int, str] = {}
    for raw in encoded_classes:
        code = int(raw)
        mapped[code] = base.get(code, str(code))
    return mapped


def decode_labels(values: Iterable[int], name_map: Dict[int, str]) -> List[str]:
    return [name_map.get(int(v), str(int(v))) for v in values]


def ensure_all_classes(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    *,
    n_classes: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Ensure the training split contains every class label."""
    missing = set(range(n_classes)) - set(int(v) for v in pd.unique(y_tr))
    if not missing or len(y_va) == 0:
        return X_tr, y_tr, X_va, y_va

    take = 0
    while missing and take < len(y_va):
        take += 1
        y_try = pd.concat([y_tr, y_va.iloc[:take]])
        missing = set(range(n_classes)) - set(int(v) for v in pd.unique(y_try))
    if missing:
        # Still missing class even after spilling validation samples – warn and continue.
        print(f"[WARN] Training split missing classes after borrow: {sorted(missing)}. "
              f"Model may not learn those classes.")
        return X_tr, y_tr, X_va, y_va

    X_tr = pd.concat([X_tr, X_va.iloc[:take]])
    y_tr = pd.concat([y_tr, y_va.iloc[:take]])
    X_va = X_va.iloc[take:]
    y_va = y_va.iloc[take:]
    return X_tr, y_tr, X_va, y_va


def to_numpy_labels(y: pd.Series, encoder: LabelEncoder) -> np.ndarray:
    return encoder.transform(y.values)


def default_param_grid() -> List[Dict[str, float]]:
    """Reasonable hand-tuned candidates for ES 5m data."""
    return [
        dict(learning_rate=0.04, max_depth=6, min_samples_leaf=60, max_iter=600, l2_regularization=1e-2),
        dict(learning_rate=0.03, max_depth=6, min_samples_leaf=80, max_iter=800, l2_regularization=5e-3),
        dict(learning_rate=0.025, max_depth=7, min_samples_leaf=100, max_iter=900, l2_regularization=1e-3),
        dict(learning_rate=0.05, max_depth=5, min_samples_leaf=80, max_iter=500, l2_regularization=1e-2),
    ]


def default_xgb_param_grid() -> List[Dict[str, float]]:
    return [
        dict(learning_rate=0.035, max_depth=5, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.05, reg_lambda=1.5,
             min_child_weight=4.0, gamma=0.0, n_estimators=800),
        dict(learning_rate=0.03, max_depth=6, subsample=0.75, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.0,
             min_child_weight=6.0, gamma=0.1, n_estimators=900),
        dict(learning_rate=0.02, max_depth=7, subsample=0.7, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.0,
             min_child_weight=8.0, gamma=0.2, n_estimators=1100),
    ]


def fit_histgb(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    encoder: LabelEncoder,
    *,
    max_bins: int,
    class_weight: Optional[Dict[int, float]],
    param_candidates: Sequence[Dict[str, float]],
    random_state: int,
    sample_weight_tr: Optional[pd.Series] = None,
    selection_metric: str = "logloss",
    long_code: Optional[int] = None,
    short_code: Optional[int] = None,
) -> Tuple[HistGradientBoostingClassifier, Dict[str, float]]:
    """Select best params via the requested validation metric."""
    labels = list(range(len(encoder.classes_)))
    best_model: Optional[HistGradientBoostingClassifier] = None
    best_score = -np.inf
    best_params: Dict[str, float] = {}

    for params in param_candidates:
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=params["learning_rate"],
            max_iter=int(params["max_iter"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            max_bins=int(max_bins),
            early_stopping=False,
            class_weight=class_weight,
            random_state=random_state,
        )
        try:
            model.fit(X_tr, to_numpy_labels(y_tr, encoder), sample_weight=None if sample_weight_tr is None else sample_weight_tr.values)
        except TypeError:
            # Handle sklearn version differences for sample_weight parameter
            try:
                model.fit(X_tr, to_numpy_labels(y_tr, encoder))
            except Exception as e:
                print(f"[WARN] Model fit failed: {e}")
                raise

        if len(X_va) > 0 and len(pd.unique(y_va)) > 1:
            proba_va = model.predict_proba(X_va)
            score, _ = _selection_score(
                selection_metric,
                to_numpy_labels(y_va, encoder),
                proba_va,
                labels=labels,
                long_code=long_code,
                short_code=short_code,
            )
        else:
            score = 0.0  # no validation set; accept first candidate

        if score > best_score or best_model is None:
            best_score = score
            best_model = model
            best_params = dict(params)

    assert best_model is not None
    best_params["selection_metric"] = selection_metric
    best_params["selection_score"] = float(best_score)
    return best_model, best_params


def calibrate_model(
    model,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    encoder: LabelEncoder,
    *,
    method: str,
    n_classes: int,
) -> Tuple[CalibratedClassifierCV | HistGradientBoostingClassifier, str]:
    """
    Optionally calibrate the pre-fit estimator using a small hold-out set.
    """
    requested = method.lower()
    if requested not in {"none", "", "sigmoid", "isotonic"}:
        raise ValueError(f"Unsupported calibration method '{method}'")

    if requested in {"", "none"}:
        return model, "none"
    if len(X_cal) == 0 or len(pd.unique(y_cal)) < 2:
        return model, "none"

    # Empirically, sigmoids on multiclass tree ensembles frequently collapse
    # marginal probabilities.
    if requested == "sigmoid" and n_classes > 2:
        print("[WARN] Skipping sigmoid calibration for multiclass model to avoid probability collapse. "
              "Use --calibration isotonic if calibration is required.")
        return model, "none"

    try:
        calibrated = CalibratedClassifierCV(model, cv="prefit", method=requested)
    except TypeError:
        # sklearn ≥1.6 requires keyword-only estimator parameter.
        calibrated = CalibratedClassifierCV(estimator=model, cv="prefit", method=requested)
    calibrated.fit(X_cal, to_numpy_labels(y_cal, encoder))
    return calibrated, requested


# -----------------------------------------------------------------------------
# Training pipeline
# -----------------------------------------------------------------------------

def _filter_setups(feats: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return feats
    if mode == "setup_only":
        if "setup_present" not in feats.columns:
            raise ValueError("setup_present column missing; rebuild features to include setups.")
        mask = feats["setup_present"] > 0
        if not mask.any():
            raise ValueError("setup_only filter removed all rows; expand your dataset or relax the filter.")
        filtered = feats.loc[mask].copy()
        print(f"[INFO] setup_only filter retained {len(filtered)}/{len(feats)} rows.")
        return filtered
    raise ValueError(f"Unknown event filter '{mode}'")


def _build_preds_frame(
    *,
    X_split: pd.DataFrame,
    y_split: pd.Series,
    proba_split: np.ndarray,
    feats_split: pd.DataFrame,
    tb_result: Dict[str, pd.Series],
    encoder: LabelEncoder,
    code_to_name: Dict[int, str],
) -> pd.DataFrame:
    """Assemble a DataFrame with probabilities, realized R, setups, and regimes for a split."""
    if proba_split is None:
        raise ValueError("Probability array is required to build preds frame.")
    preds_df = pd.DataFrame(index=X_split.index)
    preds_df["y_true"] = y_split
    preds_df["y_r"] = tb_result["y_r"].reindex(preds_df.index)

    proba_split = np.asarray(proba_split)
    if proba_split.ndim == 1:
        proba_split = proba_split.reshape(-1, 1)
    proba_map: Dict[str, np.ndarray] = {}
    for idx, cls_code in enumerate(encoder.classes_):
        name = str(code_to_name.get(int(cls_code), str(int(cls_code)))).upper()
        proba_map[name] = proba_split[:, idx]
    preds_df["p_short"] = proba_map.get("SHORT", proba_split[:, 0] if proba_split.shape[1] else 0.0)
    preds_df["p_long"] = proba_map.get(
        "LONG",
        proba_split[:, min(proba_split.shape[1] - 1, max(0, proba_split.shape[1] - 1))] if proba_split.shape[1] else 0.0,
    )
    preds_df["p_flat"] = proba_map.get("FLAT", np.zeros(len(preds_df)))

    if "setup_id" in feats_split.columns:
        preds_df["setup_id"] = feats_split["setup_id"].fillna("none").astype(str)
    elif "setup_present" in feats_split.columns:
        preds_df["setup_id"] = feats_split["setup_present"].fillna(0).astype(int).astype(str)
    else:
        preds_df["setup_id"] = "unknown"

    preds_df["setup_present"] = (
        feats_split["setup_present"].reindex(preds_df.index).fillna(0).astype(int)
        if "setup_present" in feats_split.columns
        else 0
    )

    if "regime_id" in feats_split.columns:
        preds_df["regime_id"] = feats_split["regime_id"].reindex(preds_df.index).fillna(-1).astype(int)
    else:
        preds_df["regime_id"] = -1

    preds_df["r_long"] = tb_result["long"]["r"].reindex(preds_df.index)
    preds_df["r_short"] = tb_result["short"]["r"].reindex(preds_df.index)
    preds_df["is_long"] = (preds_df["r_long"].fillna(0.0) > 0.0).astype(int)
    preds_df["is_short"] = (preds_df["r_short"].fillna(0.0) > 0.0).astype(int)
    return preds_df


def _per_group_precision_ev(
    preds_df: Optional[pd.DataFrame],
    group_cols: Sequence[str],
) -> Dict[str, object]:
    """Compute per-group precision/EV diagnostics for long/short trades."""
    payload: Dict[str, object] = {"groupby": list(group_cols), "global": None, "groups": []}
    if preds_df is None or preds_df.empty:
        return payload

    def _direction_stats(series: pd.Series) -> Dict[str, float]:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        n = len(vals)
        if n == 0:
            return {"n": 0, "precision": 0.0, "ev": 0.0}
        precision = float((vals > 0).sum()) / float(n)
        ev = float(vals.mean())
        return {"n": int(n), "precision": precision, "ev": ev}

    def _metrics(frame: pd.DataFrame) -> Dict[str, object]:
        return {
            "n_events": int(len(frame)),
            "long": _direction_stats(frame["r_long"]),
            "short": _direction_stats(frame["r_short"]),
        }

    payload["global"] = _metrics(preds_df)

    if group_cols:
        grouped = preds_df.groupby(group_cols, dropna=False)
        for key, frame in grouped:
            if isinstance(key, tuple):
                values = list(key)
            else:
                values = [key]
            key_map = {group_cols[idx]: values[idx] for idx in range(len(group_cols))}
            payload["groups"].append({"key": key_map, "metrics": _metrics(frame)})
    return payload


def prepare_matrix(
    df_raw: pd.DataFrame,
    *,
    tz: str,
    rth_start: str,
    rth_end: str,
    orb_minutes: int,
    horizon: int,
    threshold: float,
    scheme: str,
    drop_flat: bool,
    max_nan_frac: float,
    use_log_returns: bool,
    keep_all_features: bool = False,
    label_quantile: Optional[float] = None,
    keep_raw_ohlc: bool = False,
    tick_size: float = 0.25,
    triple_barrier_params: TripleBarrierParams | None = None,
    event_filter: str = "all",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, float, LabelEncoder, Dict[int, str], Optional[int], Optional[int]]:
    """Construct feature matrix and aligned targets."""
    feats = build_features(
        df_raw,
        tz=tz,
        rth_start=rth_start,
        rth_end=rth_end,
        orb_minutes=orb_minutes,
    )
    feats = _filter_setups(feats, event_filter)

    tb_result = None
    if scheme == "triple_barrier":
        params = triple_barrier_params or TripleBarrierParams(
            tick_size=tick_size,
            stop_ticks=8,
            target_ticks=12,
            max_hold_bars=max(1, horizon),
        )
        tb_result = make_triple_barrier_labels(
            feats,
            tz=tz,
            rth_start=rth_start,
            rth_end=rth_end,
            params=params,
            side="BOTH",
        )
        dir_series = tb_result["y_dir"].dropna().astype(int)
        feats = feats.loc[dir_series.index]
        labels_df = pd.DataFrame({"target": dir_series})
        threshold_used = float(params.target_ticks * params.tick_size)
        feats.attrs["triple_barrier"] = tb_result
    else:
        labels_df, threshold_used = make_label_frame(
            feats[CLOSE_COL].copy(),
            horizon=horizon,
            threshold=threshold,
            scheme=scheme,
            use_log=use_log_returns,
            quantile=label_quantile,
        )

    y_raw = labels_df["target"].astype(int)

    # ----------------------------
    # FIX 1: label mapping (include FLAT for ternary-like schemes)
    # ----------------------------
    if drop_flat:
        mask = y_raw != 0
        keep_idx = mask.index[mask]
        y_raw = y_raw.loc[keep_idx]
        labels_df = labels_df.loc[keep_idx]
        feats = feats.loc[keep_idx]
        mapping = {-1: 0, 1: 1}
    else:
        feats = feats.loc[y_raw.index]
        if scheme in {"ternary", "direction", "triple_barrier"}:
            mapping = {-1: 0, 0: 1, 1: 2}  # SHORT, FLAT, LONG
        elif scheme == "binary":
            mapping = {0: 0, 1: 1}
        else:
            unique_vals = sorted(pd.unique(y_raw))
            mapping = {int(v): idx for idx, v in enumerate(unique_vals)}

    y = y_raw.map(mapping).astype(int)

    # Dense-code (0..C-1) remap to handle missing classes (e.g., no FLAT)
    unique_labels = sorted(y.unique())
    label_to_code = {label: i for i, label in enumerate(unique_labels)}
    y_encoded = y.map(label_to_code).astype(int)

    encoder = LabelEncoder()
    encoder.fit(range(len(unique_labels)))
    if len(encoder.classes_) < 2:
        raise ValueError("Need at least two classes to train a classifier.")

    # ----------------------------
    # FIX 2: code_to_name and LONG/SHORT codes in dense-code space
    # ----------------------------
    if drop_flat:
        raw_name = {0: "SHORT", 1: "LONG"}
    elif scheme in {"ternary", "direction", "triple_barrier"}:
        raw_name = {0: "SHORT", 1: "FLAT", 2: "LONG"}
    elif scheme == "binary":
        raw_name = {0: "NEG", 1: "POS"}
    else:
        raw_name = {int(v): str(int(v)) for v in unique_labels}

    code_to_name = {
        dense_code: raw_name.get(int(raw_label), str(int(raw_label)))
        for dense_code, raw_label in enumerate(unique_labels)
    }

    name_to_code = {str(v).upper(): int(k) for k, v in code_to_name.items()}
    short_code = name_to_code.get("SHORT")
    long_code = name_to_code.get("LONG")

    # ----------------------------
    # Build X and enforce mandatory order
    # ----------------------------
    X = feats.select_dtypes(include=[np.number, "bool"]).astype(float)

    DROP_REDUNDANT_FEATURES = os.getenv("DROP_REDUNDANT_FEATURES", "0") == "1"
    if DROP_REDUNDANT_FEATURES:
        try:
            from .feature_constants import NEXT_RETRAIN_REMOVALS

            drop_set = set(NEXT_RETRAIN_REMOVALS)
            cols_to_drop = [c for c in X.columns if c in drop_set]
            if cols_to_drop:
                X = X.drop(columns=cols_to_drop)
                logging.getLogger(__name__).info(
                    "Pruned %d redundant feature(s): %s", len(cols_to_drop), sorted(cols_to_drop)
                )
        except Exception as exc:
            print(f"[train_hgb_multi] WARN: redundant feature pruning unavailable: {exc}")
    else:
        try:
            from .feature_constants import NEXT_RETRAIN_REMOVALS

            logging.getLogger(__name__).warning(
                "DROP_REDUNDANT_FEATURES=0 — %d known-redundant features will be included in this training run. "
                "Set env DROP_REDUNDANT_FEATURES=1 to prune them. Redundant features: %s",
                len(NEXT_RETRAIN_REMOVALS),
                NEXT_RETRAIN_REMOVALS,
            )
        except Exception as exc:
            print(f"[train_hgb_multi] WARN: NEXT_RETRAIN_REMOVALS unavailable: {exc}")

    try:
        mandatory_cols = mandatory_features(orb_minutes)
    except TypeError:
        mandatory_cols = list(MANDATORY_FEATURES)

    base_order = list(mandatory_cols)
    if keep_raw_ohlc:
        for raw in (OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL):
            if raw not in base_order:
                base_order.append(raw)

    ordered_cols = list(dict.fromkeys(base_order + list(X.columns)))
    if DROP_REDUNDANT_FEATURES:
        try:
            from .feature_constants import NEXT_RETRAIN_REMOVALS

            drop_set = set(NEXT_RETRAIN_REMOVALS)
            ordered_cols = [c for c in ordered_cols if c not in drop_set]
        except Exception:
            pass

    # ----------------------------
    # FIX 3: never overwrite existing columns with zeros
    # ----------------------------
    for col in ordered_cols:
        if col in X.columns:
            continue
        if col in feats.columns:
            X[col] = pd.to_numeric(feats[col], errors="coerce").reindex(X.index).astype(float)
        else:
            X[col] = 0.0

    X = X.loc[:, ordered_cols]
    y_encoded = y_encoded.loc[X.index]
    feats = feats.loc[X.index]
    feats.attrs["threshold_used"] = float(threshold_used)

    return X, y_encoded, feats, float(threshold_used), encoder, code_to_name, long_code, short_code


def _make_xgb(n_classes: int, *, random_state: int, use_gpu: bool = False, params: Optional[dict] = None):
    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("xgboost is not installed; install it to use --model xgb") from e

    tree_method = "hist"
    predictor = "auto"
    if use_gpu:
        if _gpu_accel_available():
            tree_method = "gpu_hist"
            predictor = "gpu_predictor"
        else:  # pragma: no cover - hardware/env dependent
            global _GPU_WARNED
            if not _GPU_WARNED:
                print("[WARN] XGBoost GPU acceleration requested but not available; falling back to CPU hist.")
                _GPU_WARNED = True

    p = dict(
        n_estimators= params.get("n_estimators", 700) if params else 700,
        learning_rate= params.get("learning_rate", 0.03) if params else 0.03,
        max_depth= params.get("max_depth", 6) if params else 6,
        subsample= params.get("subsample", 0.8) if params else 0.8,
        colsample_bytree= params.get("colsample_bytree", 0.8) if params else 0.8,
        reg_alpha= params.get("reg_alpha", 0.1) if params else 0.1,
        reg_lambda= params.get("reg_lambda", 1.0) if params else 1.0,
        max_bin= params.get("max_bin", 256) if params else 256,
        tree_method= tree_method,
        predictor= predictor,
        n_jobs = -1,
        random_state = random_state,
        eval_metric = "mlogloss" if n_classes > 2 else "logloss",
        objective = ("multi:softprob" if n_classes > 2 else "binary:logistic"),
    )
    if n_classes > 2:
        p["num_class"] = n_classes
    if params:
        p.update(params)
    return XGBClassifier(**p)


def make_xgb_or_hgb_binary(
    *,
    model: str,
    learning_rate: float = 0.03,
    max_depth: int = 6,
    min_samples_leaf: int = 80,
    max_iter: int = 700,
    max_bins: int = 255,
    l2: float = 5e-3,
    random_state: int = 42,
    use_gpu: bool = False,
    class_weight: str | dict[int, float] | None = None,
    scale_pos_weight: float | None = None,
    reg_alpha: float | None = None,
    subsample: float | None = None,
    colsample_bytree: float | None = None,
    min_child_weight: float | None = None,
    gamma: float | None = None,
    max_leaf_nodes: int | None = None,
) -> object:
    model_lower = model.lower()
    if model_lower == "hgb":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_iter=max_iter,
            max_bins=max_bins,
            l2_regularization=l2,
            random_state=random_state,
            class_weight=class_weight,
            max_leaf_nodes=max_leaf_nodes,
        )
    if model_lower == "xgb":
        params = {
            "n_estimators": max_iter,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "max_bin": max_bins,
            "reg_lambda": l2,
        }
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = float(scale_pos_weight)
        if reg_alpha is not None:
            params["reg_alpha"] = float(reg_alpha)
        if subsample is not None:
            params["subsample"] = float(subsample)
        if colsample_bytree is not None:
            params["colsample_bytree"] = float(colsample_bytree)
        if min_child_weight is not None:
            params["min_child_weight"] = float(min_child_weight)
        if gamma is not None:
            params["gamma"] = float(gamma)
        try:
            return _make_xgb(
                n_classes=2,
                random_state=random_state,
                use_gpu=use_gpu,
                params=params,
            )
        except Exception as e:
            raise ValueError(f"Failed to create XGB model: {e}") from e
    raise ValueError(f"Unknown model type '{model}'. Expected 'hgb' or 'xgb'.")
def choose_label_with_margins(p_short, p_flat, p_long, t_short=0.52, t_long=0.55, margin=0.02):
    if (p_short >= t_short) and (p_short >= p_long + margin):
        return "SHORT"
    if (p_long  >= t_long)  and (p_long  >= p_short + margin):
        return "LONG"
    return "FLAT"


def choose_threshold_by_precision(proba: np.ndarray, y: np.ndarray, min_precision: float = 0.8) -> float:
    if proba.ndim != 1:
        raise ValueError("Probability array must be 1-D for threshold selection.")
    if len(proba) != len(y):
        raise ValueError("Probability and label arrays must align in length.")
    if len(proba) == 0:
        return 0.5
    precision, _recall, thresholds = precision_recall_curve(y, proba)
    mask = np.where(precision[:-1] >= float(min_precision))[0]
    if len(mask):
        return float(thresholds[mask[0]])
    return 0.5


def _auto_tune_side_thresholds(
    p_pos: np.ndarray,
    p_neg: np.ndarray,
    y_is_pos: np.ndarray,
    tau_grid: Tuple[float, float, int] = (0.30, 0.80, 21),
    ratio_grid: Tuple[float, float, int] = (1.05, 2.25, 25),
    precision_bias: float = 0.10,
) -> Dict[str, float]:
    """
    Grid-search thresholds that maximize a precision-biased F1 score for a given side.
    Returns {'tau', 'ratio', 'f1', 'precision', 'recall'}.
    """
    taus = np.linspace(*tau_grid)
    ratios = np.linspace(*ratio_grid)
    best: Optional[Tuple[float, float, float, float, float, float]] = None
    for t in taus:
        for r in ratios:
            dominance = p_pos / (p_neg + 1e-9)
            pred = (p_pos >= t) & (dominance >= r)
            tp = int(((pred == 1) & (y_is_pos == 1)).sum())
            fp = int(((pred == 1) & (y_is_pos == 0)).sum())
            fn = int(((pred == 0) & (y_is_pos == 1)).sum())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 0.0 if (prec + rec) == 0 else (2.0 * prec * rec / (prec + rec))
            score = f1 + precision_bias * prec
            cand = (score, f1, prec, rec, float(t), float(r))
            if best is None or cand > best:
                best = cand
    assert best is not None
    _score, f1, prec, rec, t, r = best
    return {
        "tau": float(t),
        "ratio": float(r),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }


def _infer_calibration_splits(n_samples: int) -> int:
    if n_samples < 40:
        raise ValueError("Need at least 40 samples to calibrate binary scalper models.")
    splits = min(5, max(2, n_samples // 150))
    if splits >= n_samples:
        splits = max(2, n_samples - 1)
    return splits


def _compute_scalp_split_index(n_samples: int, test_ratio: float) -> int:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must lie in (0, 1) for scalper mode.")
    split = int(n_samples * (1.0 - test_ratio))
    split = max(40, split)
    split = min(n_samples - 10, split)
    if split <= 0 or split >= n_samples:
        raise ValueError("Requested --test-ratio leaves no samples for train/test split.")
    return split


def _train_calibrated_binary_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    max_iter: int,
    max_bins: int,
    l2: float,
    random_state: int,
    use_gpu: bool,
    reg_alpha: float | None = None,
    subsample: float | None = None,
    colsample_bytree: float | None = None,
    min_child_weight: float | None = None,
    gamma: float | None = None,
) -> CalibratedClassifierCV:
    # Compute per-side imbalance
    pos = float(np.sum(y == 1))
    neg = float(len(y) - pos)
    # Safe default if pos==0 handled earlier
    spw = max(1.0, neg / max(1.0, pos))

    base = make_xgb_or_hgb_binary(
        model=model,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_iter=max_iter,
        max_bins=max_bins,
        l2=l2,
        random_state=random_state,
        use_gpu=use_gpu,
        # Bias the SHORT/LONG binary fit toward catching positives:
        scale_pos_weight=spw if model.lower() == "xgb" else None,
        class_weight={0: 1.0, 1: spw} if model.lower() == "hgb" else None,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
    )
    cv = TimeSeriesSplit(n_splits=_infer_calibration_splits(len(X)))
    calibrated = CalibratedClassifierCV(
        estimator=base,
        method="isotonic",
        cv=cv,
        n_jobs=1,
    )
    calibrated.fit(X, y)
    return calibrated


def train_pipeline_scalp(args: argparse.Namespace) -> Dict[str, object]:
    if args.save_model:
        raise ValueError("--save-model is not supported when --label-scheme=scalp. Use --save-scalp-dir.")
    if getattr(args, "event_filter", "all") != "all":
        raise ValueError("--event-filter is not supported for --label-scheme=scalp.")

    df = pd.read_csv(Path(args.csv))
    df = _normalize_ohlcv_columns(df)
    if "Datetime" not in df.columns:
        raise ValueError("Normalized CSV must contain a 'Datetime' column.")
    df = df.sort_values("Datetime").set_index("Datetime")

    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in ohlcv_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required OHLCV columns: {', '.join(missing)}")

    df_ohlcv = df[ohlcv_cols]
    feature_frame = build_features(
        df,
        tz=args.tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_minutes=args.orb_minutes,
        csv_naive_is_utc=False,
        feature_set="scalp_micro_v1",
        tick_size=float(args.tick_size),
    ).sort_values("Datetime").set_index("Datetime")
    if getattr(feature_frame.index, "tz", None) is not None:
        feature_frame.index = feature_frame.index.tz_convert("UTC").tz_localize(None)
    if getattr(df_ohlcv.index, "tz", None) is not None:
        df_ohlcv.index = df_ohlcv.index.tz_convert("UTC").tz_localize(None)
    feature_frame = feature_frame.select_dtypes(include=[np.number, "bool"]).astype(float)
    labels_long, labels_short = scalp_label_series(
        df_ohlcv,
        target_ticks=int(args.target_ticks),
        stop_ticks=int(args.stop_ticks),
        horizon_bars=int(args.horizon_bars),
        tick_size=float(args.tick_size),
    )

    y_long_series = pd.Series(labels_long, index=df_ohlcv.index).reindex(feature_frame.index)
    y_short_series = pd.Series(labels_short, index=df_ohlcv.index).reindex(feature_frame.index)
    valid_mask = y_long_series.notna() & y_short_series.notna()
    feature_frame = feature_frame.loc[valid_mask]
    y_long_series = y_long_series.loc[valid_mask].astype(np.int8)
    y_short_series = y_short_series.loc[valid_mask].astype(np.int8)

    feature_columns = list(feature_frame.columns)
    X_all = feature_frame.to_numpy(dtype=float)
    idx_all = feature_frame.index
    n_samples = len(X_all)
    if n_samples < 80:
        raise ValueError("Need at least 80 samples to train scalper models.")

    subsample_opt = float(args.subsample) if getattr(args, "subsample", None) is not None else None
    colsample_opt = float(args.colsample_bytree) if getattr(args, "colsample_bytree", None) is not None else None
    min_child_weight_opt = float(args.min_child_weight) if getattr(args, "min_child_weight", None) is not None else None
    gamma_opt = float(args.gamma) if getattr(args, "gamma", None) is not None else None
    reg_alpha_opt = float(args.reg_alpha) if getattr(args, "reg_alpha", None) is not None else None

    split_idx = _compute_scalp_split_index(n_samples, float(args.test_ratio))
    X_tr, X_te = X_all[:split_idx], X_all[split_idx:]
    idx_te = idx_all[split_idx:]

    side_map: Dict[str, np.ndarray] = {
        "LONG": y_long_series.to_numpy(dtype=np.int8),
        "SHORT": y_short_series.to_numpy(dtype=np.int8),
    }
    if args.side == "BOTH":
        sides = ("LONG", "SHORT")
    else:
        sides = (args.side,)

    results: Dict[str, Dict[str, object]] = {}
    thresholds_payload: Dict[str, Dict[str, object]] = {}
    pred_frames = []
    save_dir: Optional[Path] = Path(args.save_scalp_dir) if args.save_scalp_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.scalp_prefix or "scalper_true"

    for side in sides:
        y_all = side_map[side]
        y_tr = y_all[:split_idx]
        y_te = y_all[split_idx:]
        positives_tr = int(y_tr.sum())
        positives_te = int(y_te.sum())
        if positives_tr == 0:
            raise ValueError(f"No positive samples for side {side} in training split.")

        model_cv = _train_calibrated_binary_model(
            X_tr,
            y_tr,
            model=args.model,
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            max_iter=int(args.max_iter),
            max_bins=int(args.max_bins),
            l2=float(args.l2),
            random_state=int(args.random_state),
            use_gpu=bool(args.gpu),
            reg_alpha=reg_alpha_opt,
            subsample=subsample_opt,
            colsample_bytree=colsample_opt,
            min_child_weight=min_child_weight_opt,
            gamma=gamma_opt,
        )

        if len(X_te) > 0:
            proba_te = model_cv.predict_proba(X_te)[:, 1]
            threshold = choose_threshold_by_precision(
                proba_te,
                y_te,
                min_precision=float(args.scalp_min_precision),
            )
            preds_te = (proba_te >= threshold).astype(np.int8)
            precision_val = (
                float(precision_score(y_te, preds_te, zero_division=0))
                if y_te.sum() > 0
                else None
            )
            recall_val = (
                float(recall_score(y_te, preds_te, zero_division=0))
                if y_te.sum() > 0
                else None
            )
        else:
            proba_te = np.array([], dtype=float)
            preds_te = np.array([], dtype=np.int8)
            threshold = 0.5
            precision_val = None
            recall_val = None

        final_model = _train_calibrated_binary_model(
            X_all,
            y_all,
            model=args.model,
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            max_iter=int(args.max_iter),
            max_bins=int(args.max_bins),
            l2=float(args.l2),
            random_state=int(args.random_state),
            use_gpu=bool(args.gpu),
            reg_alpha=reg_alpha_opt,
            subsample=subsample_opt,
            colsample_bytree=colsample_opt,
            min_child_weight=min_child_weight_opt,
            gamma=gamma_opt,
        )
        try:
            setattr(final_model, "feature_names_in_", np.array(feature_columns))
        except Exception:
            pass

        model_path: Optional[Path] = None
        if save_dir:
            model_path = save_dir / f"{prefix}_{side.lower()}.joblib"
            joblib.dump(final_model, model_path)
            meta = {
                "proba_class": side,
                "model_type": args.model,
                "tick_size": float(args.tick_size),
                "target_ticks": int(args.target_ticks),
                "stop_ticks": int(args.stop_ticks),
                "horizon_bars": int(args.horizon_bars),
                "min_precision": float(args.scalp_min_precision),
                "feature_set": "scalp_micro_v1",
            }
            meta_path = model_path.with_suffix(".meta.json")
            meta_path.write_text(_json_dumps(meta))
            print(f"[OK] Saved {side} model -> {model_path}")

        if args.save_preds and len(X_te):
            preds_df = pd.DataFrame(
                {
                    "Datetime": pd.to_datetime(idx_te),
                    "side": side,
                    "proba": proba_te,
                    "prediction": preds_te,
                    "y_true": y_te,
                    "threshold": threshold,
                }
            )
            pred_frames.append(preds_df)

        results[side] = {
            "train_size": int(len(y_tr)),
            "test_size": int(len(y_te)),
            "positives_train": positives_tr,
            "positives_test": positives_te,
            "threshold": float(threshold),
            "test_precision": precision_val,
            "test_recall": recall_val,
            "model_path": str(model_path) if model_path else None,
        }
        thresholds_payload[side] = {
            "min_confidence": float(threshold),
            "min_precision_target": float(args.scalp_min_precision),
            "positives_train": positives_tr,
            "positives_test": positives_te,
            "test_precision": precision_val,
            "test_recall": recall_val,
        }

    if args.save_preds and pred_frames:
        preds_out = pd.concat(pred_frames).sort_values(["Datetime", "side"]).reset_index(drop=True)
        outp = Path(args.save_preds)
        outp.parent.mkdir(parents=True, exist_ok=True)
        preds_out.to_csv(outp, index=False)
        print(f"[OK] Saved test predictions -> {outp}")

    if save_dir and thresholds_payload:
        thresholds_path = save_dir / f"{prefix}_thresholds.json"
        thresholds_path.write_text(_json_dumps(thresholds_payload))
        print(f"[OK] Saved thresholds -> {thresholds_path}")

    metrics = {
        "mode": "scalp",
        "n_samples": int(n_samples),
        "train_size": int(split_idx),
        "test_size": int(len(X_te)),
        "test_ratio": float(args.test_ratio),
        "min_precision_target": float(args.scalp_min_precision),
        "results": results,
    }

    print(_json_dumps(metrics))

    if args.save_metrics:
        Path(args.save_metrics).write_text(_json_dumps(metrics))
        print(f"[OK] Saved metrics -> {args.save_metrics}")

    return metrics


def train_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    if args.label_scheme == "scalp":
        return train_pipeline_scalp(args)
    df = pd.read_csv(Path(args.csv))
    df = _normalize_ohlcv_columns(df)

    keep_all_features = args.keep_all_features
    if keep_all_features is None:
        keep_all_features = args.model.lower() == "xgb"

    tb_params = None
    if args.label_scheme == "triple_barrier":
        tb_params = TripleBarrierParams(
            tick_size=float(args.tick_size),
            stop_ticks=int(args.tb_stop_ticks),
            target_ticks=int(args.tb_target_ticks),
            max_hold_bars=int(args.tb_max_hold_bars),
            tie_break=args.tb_tie_break,
            session_exit=args.tb_session_exit,
            timeout_exit=args.tb_timeout_exit,
            clip_r=float(args.tb_clip_r),
        )

    X_all, y_all, feats_all, threshold_used, encoder, code_to_name, long_code, short_code = prepare_matrix(
        df_raw=df,
        tz=args.tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_minutes=args.orb_minutes,
        horizon=args.horizon,
        threshold=args.ret_threshold,
        scheme=args.label_scheme,
        drop_flat=args.drop_flat,
        max_nan_frac=args.max_nan_frac,
        use_log_returns=args.use_log_returns,
        keep_all_features=keep_all_features,
        label_quantile=args.label_quantile,
        keep_raw_ohlc=args.keep_raw_ohlc,
        tick_size=float(args.tick_size),
        triple_barrier_params=tb_params,
        event_filter=args.event_filter,
    )
    feats_all.attrs["threshold_used"] = float(threshold_used)
    tb_result = feats_all.attrs.get("triple_barrier") if args.label_scheme == "triple_barrier" else None

    if len(X_all) < 200:
        raise ValueError("Not enough rows after feature/label sanitation; supply more data.")

    class_counts_all = y_all.value_counts().sort_index()
    class_frac_all = (class_counts_all / max(1, len(y_all))).sort_index()
    frac_msg = ", ".join(f"{int(lbl)}={int(class_counts_all[lbl])} ({class_frac_all[lbl]*100:.2f})%" for lbl in class_counts_all.index)
    print(f"[INFO] Class distribution (all samples): {frac_msg}")
    min_class_frac = float(class_frac_all.min()) if not class_frac_all.empty else 0.0
    if min_class_frac < float(getattr(args, "min_class_fraction", 0.0)):
        print(f"[WARN] Minimum class fraction {min_class_frac:.4f} is below --min-class-fraction "
              f"{getattr(args, 'min_class_fraction'):.4f}. Consider lowering --ret-threshold or using --label-quantile "
              "to balance classes.")

    # Compute sample weights to encourage VWAP/EMA confluence
    def _sample_weights(feats: pd.DataFrame, y_codes: pd.Series) -> pd.Series:
        mode = getattr(args, "sample_weight_mode", "confluence")
        if str(mode).lower() == "none":
            return pd.Series(1.0, index=y_codes.index)

        w = pd.Series(1.0, index=y_codes.index)
        has_vwap = "vwap_sess" in feats.columns
        has_ema = ("ema_20" in feats.columns) and ("ema_50" in feats.columns)
        if not (has_vwap and has_ema):
            return w
        above_vwap = feats.loc[y_codes.index, "vwap_sess"]
        close = feats.loc[y_codes.index, CLOSE_COL]
        price_above = (close > above_vwap).astype(bool)
        ema_bull = (feats.loc[y_codes.index, "ema_20"] > feats.loc[y_codes.index, "ema_50"]).astype(bool)
        confluence_long = price_above & ema_bull
        confluence_short = (~price_above) & (~ema_bull)
        # Code mapping (drop_flat: {0:SHORT,1:LONG} else {0:SHORT,1:FLAT,2:LONG})
        if args.drop_flat:
            is_long = y_codes == 1
            is_short = y_codes == 0
        else:
            is_long = y_codes == 2
            is_short = y_codes == 0
        w[is_long & confluence_long] *= float(getattr(args, "weight_confluence_pos", 2.0))
        w[is_long & (~confluence_long)] *= float(getattr(args, "weight_confluence_neg", 0.7))
        w[is_short & confluence_short] *= float(getattr(args, "weight_confluence_pos", 2.0))
        w[is_short & (~confluence_short)] *= float(getattr(args, "weight_confluence_neg", 0.7))
        if not args.drop_flat and (y_codes == 1).any():
            w[y_codes == 1] *= float(getattr(args, "weight_flat", 0.8))

        long_mult = float(getattr(args, "long_weight_multiplier", 1.0))
        short_mult = float(getattr(args, "short_weight_multiplier", 1.0))
        flat_mult = float(getattr(args, "flat_weight_multiplier", 1.0))
        if args.drop_flat:
            w[y_codes == 1] *= long_mult
            w[y_codes == 0] *= short_mult
        else:
            w[y_codes == 2] *= long_mult
            w[y_codes == 0] *= short_mult
            w[y_codes == 1] *= flat_mult
        return w

    sw_all = None if str(getattr(args, "sample_weight_mode", "confluence")).lower() == "none" else _sample_weights(feats_all, y_all)

    # Time-aware split
    tr_idx, va_idx, te_idx = time_splits(
        len(X_all),
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
    )
    X_tr, y_tr = X_all.iloc[list(tr_idx)], y_all.iloc[list(tr_idx)]
    X_va, y_va = X_all.iloc[list(va_idx)], y_all.iloc[list(va_idx)]
    X_te, y_te = X_all.iloc[list(te_idx)], y_all.iloc[list(te_idx)]
    sw_tr = sw_all.iloc[list(tr_idx)] if sw_all is not None else None
    sw_va = sw_all.iloc[list(va_idx)] if sw_all is not None else None

    X_tr, y_tr, X_va, y_va = ensure_all_classes(
        X_tr,
        y_tr,
        X_va,
        y_va,
        n_classes=len(encoder.classes_),
    )

    if len(set(map(int, pd.unique(y_tr)))) < len(encoder.classes_):
        missing = set(range(len(encoder.classes_))) - set(map(int, pd.unique(y_tr)))
        print(f"[WARN] Training split missing classes {sorted(missing)}; model may not learn those classes.")

    class_weights = compute_class_weights(pd.Series(y_tr))

    if args.class_weight == "balanced":
        cw = "balanced"
    else:
        cw = class_weights

    if args.class_weight in {"auto", "balanced"}:
        class_weight_map = class_weights if class_weights else None
    else:
        class_weight_map = None

    def _compose_weights(y_codes: pd.Series, base: Optional[pd.Series], class_map: Optional[Dict[int, float]]) -> pd.Series:
        weights = pd.Series(1.0, index=y_codes.index, dtype=float)
        if base is not None:
            base_aligned = base.reindex(y_codes.index).astype(float).fillna(1.0)
            weights = weights.mul(base_aligned, fill_value=1.0)
        if class_map:
            class_aligned = y_codes.map(class_map).fillna(1.0).astype(float)
            weights = weights.mul(class_aligned, fill_value=1.0)
        return weights

    selection_metric = getattr(args, "selection_metric", "logloss")
    if args.model.lower() == "hgb":
        if args.tune:
            param_grid = default_param_grid()
        else:
            param_grid = [
                dict(
                    learning_rate=args.learning_rate,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    max_iter=args.max_iter,
                    l2_regularization=args.l2,
                )
            ]
        base_model, best_params = fit_histgb(
            X_tr,
            y_tr,
            X_va,
            y_va,
            encoder,
            max_bins=args.max_bins,
            class_weight=cw,
            param_candidates=param_grid,
            random_state=args.random_state,
            sample_weight_tr=sw_tr,
            selection_metric=selection_metric,
            long_code=long_code,
            short_code=short_code,
        )
    else:
        if args.tune:
            param_grid = default_xgb_param_grid()
        else:
            param_grid = [
                dict(
                    learning_rate=args.learning_rate,
                    max_depth=args.max_depth,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=args.l2,
                    n_estimators=args.max_iter,
                    max_bin=args.max_bins,
                )
            ]

        best_score = -np.inf
        best_params: Dict[str, float] = {}
        base_model = None
        for candidate in param_grid:
            params = dict(candidate)
            params.setdefault("n_estimators", args.max_iter)
            params.setdefault("max_bin", args.max_bins)
            params.setdefault("reg_lambda", args.l2)
            if getattr(args, "reg_alpha", None) is not None:
                params["reg_alpha"] = float(args.reg_alpha)
            if getattr(args, "subsample", None) is not None:
                params["subsample"] = float(args.subsample)
            if getattr(args, "colsample_bytree", None) is not None:
                params["colsample_bytree"] = float(args.colsample_bytree)
            if getattr(args, "min_child_weight", None) is not None:
                params["min_child_weight"] = float(args.min_child_weight)
            if getattr(args, "gamma", None) is not None:
                params["gamma"] = float(args.gamma)
            xgb_model = _make_xgb(
                n_classes=len(encoder.classes_),
                random_state=args.random_state,
                use_gpu=getattr(args, "gpu", False),
                params=params,
            )
            train_weights = _compose_weights(y_tr, sw_tr, class_weight_map)
            fit_args: Dict[str, object] = {
                "sample_weight": train_weights.values,
                "verbose": False,
            }
            if len(X_va):
                val_weights = _compose_weights(y_va, sw_va, class_weight_map)
                fit_args["eval_set"] = [(X_va, y_va.values)]
                fit_args["eval_sample_weight"] = [val_weights.values]
                es_rounds = int(getattr(args, "early_stopping_rounds", 0) or 0)
                if es_rounds > 0:
                    fit_args["early_stopping_rounds"] = es_rounds
            def _fit_with_backoff(model, args_dict):
                """Handle older xgboost versions that lack some kwargs."""
                args_local = dict(args_dict)
                for key in ("eval_sample_weight", "early_stopping_rounds"):
                    try:
                        return model.fit(X_tr, y_tr.values, **args_local)
                    except TypeError:
                        if key in args_local:
                            args_local.pop(key, None)
                        else:
                            raise
                # Final attempt without optional args
                return model.fit(X_tr, y_tr.values, **args_local)

            _fit_with_backoff(xgb_model, fit_args)

            if len(X_va) > 0 and len(pd.unique(y_va)) > 1:
                proba_va = xgb_model.predict_proba(X_va)
                score, _ = _selection_score(
                    selection_metric,
                    y_va.values,
                    proba_va,
                    labels=list(range(len(encoder.classes_))),
                    long_code=long_code,
                    short_code=short_code,
                )
            else:
                score = 0.0

            if score > best_score or base_model is None:
                best_score = score
                base_model = xgb_model
                best_params = dict(params)

        assert base_model is not None
        best_params["selection_metric"] = selection_metric
        best_params["selection_score"] = float(best_score)

    calibration_arg = str(getattr(args, "calibration", "auto")).lower()
    if calibration_arg == "auto":
        if args.model.lower() == "hgb":
            calibration_method = "sigmoid" if len(encoder.classes_) <= 2 else "isotonic"
        else:
            calibration_method = "sigmoid" if len(encoder.classes_) == 2 else "none"
    else:
        calibration_method = calibration_arg

    use_xgb_ovr = (
        args.model.lower() == "xgb"
        and len(encoder.classes_) > 2
        and calibration_arg in {"auto", "sigmoid", "isotonic"}
    )

    calibrated_flag = False
    if use_xgb_ovr:
        if long_code is None or short_code is None:
            raise ValueError("Could not locate LONG/SHORT codes for OvR calibration.")
        X_trva = pd.concat([X_tr, X_va])
        y_trva = pd.concat([y_tr, y_va])
        X_trva_np = X_trva.to_numpy(dtype=float)
        y_trva_np = y_trva.to_numpy(dtype=int)
        y_long_bin = (y_trva_np == long_code).astype(int)
        y_short_bin = (y_trva_np == short_code).astype(int)

        subsample_opt = float(args.subsample) if getattr(args, "subsample", None) is not None else None
        colsample_opt = float(args.colsample_bytree) if getattr(args, "colsample_bytree", None) is not None else None
        min_child_weight_opt = float(args.min_child_weight) if getattr(args, "min_child_weight", None) is not None else None
        gamma_opt = float(args.gamma) if getattr(args, "gamma", None) is not None else None
        reg_alpha_opt = float(args.reg_alpha) if getattr(args, "reg_alpha", None) is not None else None

        model_long = _train_calibrated_binary_model(
            X_trva_np,
            y_long_bin,
            model="xgb",
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            max_iter=int(args.max_iter),
            max_bins=int(args.max_bins),
            l2=float(args.l2),
            random_state=int(args.random_state),
            use_gpu=bool(getattr(args, "gpu", False)),
            reg_alpha=reg_alpha_opt,
            subsample=subsample_opt,
            colsample_bytree=colsample_opt,
            min_child_weight=min_child_weight_opt,
            gamma=gamma_opt,
        )
        model_short = _train_calibrated_binary_model(
            X_trva_np,
            y_short_bin,
            model="xgb",
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            max_iter=int(args.max_iter),
            max_bins=int(args.max_bins),
            l2=float(args.l2),
            random_state=int(args.random_state),
            use_gpu=bool(getattr(args, "gpu", False)),
            reg_alpha=reg_alpha_opt,
            subsample=subsample_opt,
            colsample_bytree=colsample_opt,
            min_child_weight=min_child_weight_opt,
            gamma=gamma_opt,
        )
        feature_names_arr = np.array(list(X_trva.columns))
        for estimator in (model_long, model_short):
            try:
                setattr(estimator, "feature_names_in_", feature_names_arr)
            except Exception:
                pass
        final_model = OvRCalibratedFuser(model_long, model_short)
        try:
            setattr(final_model, "feature_names_in_", feature_names_arr)
        except Exception:
            pass
        calibration_applied = "ovr_isotonic"
        calibrated_flag = True
    else:
        final_model, calibration_applied = calibrate_model(
            base_model,
            X_va,
            y_va,
            encoder,
            method=calibration_method,
            n_classes=len(encoder.classes_),
        )
        calibrated_flag = isinstance(final_model, CalibratedClassifierCV)

    try:
        setattr(final_model, "feature_names_in_", np.array(list(X_all.columns)))
    except Exception:
        pass

    try:
        proba_va = final_model.predict_proba(X_va)
        proba_va = renorm_proba(np.asarray(proba_va))
    except Exception:
        proba_va = None

    proba_te = final_model.predict_proba(X_te)
    proba_te = renorm_proba(np.asarray(proba_te))
    labels = list(range(len(encoder.classes_)))
    logloss = log_loss(
        y_te.values,
        np.clip(proba_te, 1e-6, 1 - 1e-6),
        labels=labels,
    )
    preds = np.argmax(proba_te, axis=1)
    bal_acc = balanced_accuracy_score(y_te.values, preds)

    # Add train accuracy logging
    if len(X_tr) > 0:
        proba_tr = final_model.predict_proba(X_tr)
        proba_tr = renorm_proba(np.asarray(proba_tr))
        preds_tr = np.argmax(proba_tr, axis=1)
        bal_acc_tr = balanced_accuracy_score(y_tr.values, preds_tr)
        logloss_tr = log_loss(
            y_tr.values,
            np.clip(proba_tr, 1e-6, 1 - 1e-6),
            labels=labels,
        )
        print(f"[INFO] Train balanced accuracy: {bal_acc_tr:.4f}, logloss: {logloss_tr:.4f}")

    # Add feature importance if available
    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
        feature_names = list(X_all.columns)
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        print(f"[INFO] Top 10 feature importances: {top_features}")
    elif hasattr(final_model, "feature_names_in_"):
        print(f"[INFO] Model trained on {len(final_model.feature_names_in_)} features")

    if args.save_model:
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            setattr(final_model, "feature_names_in_", np.array(list(X_all.columns)))
        except Exception:
            pass
        joblib.dump(final_model, model_path)
        feature_sidecar = model_path.with_suffix(".features.json")
        feature_sidecar.write_text(
            _json_dumps({"features": list(map(str, X_all.columns))})
        )
        meta_payload = {
            "model_type": args.model,
            "label_scheme": args.label_scheme,
            "calibration_applied": calibration_applied,
            "n_features": int(X_all.shape[1]),
            "class_code_to_name": {int(k): v for k, v in code_to_name.items()},
            "drop_flat": bool(args.drop_flat),
            "keep_raw_ohlc": bool(args.keep_raw_ohlc),
        }
        meta_path = model_path.with_suffix(".meta.json")
        meta_path.write_text(_json_dumps(meta_payload))
        print(f"[OK] Saved model -> {model_path}")
        print(f"[OK] Saved feature list -> {feature_sidecar}")
        print(f"[OK] Saved metadata -> {meta_path}")

    # Threshold optimization and SIM-lite (only for triple_barrier)
    thresholds_payload = None
    sim_metrics = {}
    if args.enable_threshold_opt and args.label_scheme == "triple_barrier" and tb_result is not None:
        def _build_split_preds(X_split: pd.DataFrame, y_split: pd.Series, proba_split: np.ndarray) -> pd.DataFrame:
            feats_split = feats_all.loc[X_split.index]
            return _build_preds_frame(
                X_split=X_split,
                y_split=y_split,
                proba_split=proba_split,
                feats_split=feats_split,
                tb_result=tb_result,
                encoder=encoder,
                code_to_name=code_to_name,
            )

        # Validation preds
        if proba_va is not None:
            preds_va = _build_split_preds(X_va, y_va, proba_va)
            # Optimize thresholds on validation
            threshold_params = ThresholdOptParams(
                taus=[round(x, 2) for x in np.arange(0.55, 0.86, 0.02)],
                min_trades=50,
                min_precision=0.55,
                max_dd_r=8.0,
                objective="ev_per_trade",
            )
            thresholds_payload = optimize_thresholds(
                preds_va,
                params=threshold_params,
                groupby=["setup_id", "regime_id"],
            )

        # Test preds
        preds_te_full = _build_split_preds(X_te, y_te, proba_te)

        # Run SIM-lite on test using optimized thresholds
        if thresholds_payload:
            sim_params = SimEvalParams(
                max_trades_per_day=10,
                cooldown_bars=5,
                daily_loss_limit_r=5.0,
                max_hold_bars=int(args.tb_max_hold_bars),
            )
            timestamps = _get_datetime_series(feats_all, preds_te_full.index)
            sim_metrics = evaluate_policy_grouped(
                preds_te_full,
                params=sim_params,
                thresholds=thresholds_payload,
                groupby=["setup_id", "regime_id"],
                timestamps=timestamps,
            )

    orig_te = y_te.values  # encoded codes
    orig_pred = preds       # encoded codes
    target_names = [code_to_name.get(int(v), str(int(v))) for v in encoder.classes_]
    report = classification_report(
        orig_te,
        orig_pred,
        labels=[int(c) for c in encoder.classes_],
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        orig_te,
        orig_pred,
        labels=encoder.classes_,
    )

    name_to_code = {str(v).upper(): int(k) for k, v in code_to_name.items()}
    idx_short = name_to_code.get("SHORT")
    idx_long = name_to_code.get("LONG")
    gate_suggest: Dict[str, Dict[str, float]] = {}
    if (
        proba_va is not None
        and len(X_va) > 0
        and idx_short is not None
        and idx_long is not None
    ):
        y_va_np = y_va.values
        y_is_short = (y_va_np == idx_short).astype(int)
        y_is_long = (y_va_np == idx_long).astype(int)
        p_short_va = proba_va[:, idx_short]
        p_long_va = proba_va[:, idx_long]
        if args.gate_short_tau is None or args.gate_short_ratio is None:
            gate_suggest["SHORT"] = _auto_tune_side_thresholds(
                p_pos=p_short_va,
                p_neg=p_long_va,
                y_is_pos=y_is_short,
                tau_grid=(0.30, 0.70, 17),
                ratio_grid=(1.05, 2.00, 20),
                precision_bias=0.12,
            )
        if args.gate_long_tau is None or args.gate_long_ratio is None:
            gate_suggest["LONG"] = _auto_tune_side_thresholds(
                p_pos=p_long_va,
                p_neg=p_short_va,
                y_is_pos=y_is_long,
                tau_grid=(0.40, 0.80, 17),
                ratio_grid=(1.10, 2.20, 20),
                precision_bias=0.08,
            )

    def _resolve_gate(side: str, tau_arg: Optional[float], ratio_arg: Optional[float]) -> Dict[str, object]:
        suggestion = gate_suggest.get(side)
        tau_val = float(tau_arg) if tau_arg is not None else (
            float(suggestion["tau"]) if suggestion and "tau" in suggestion else None
        )
        ratio_val = float(ratio_arg) if ratio_arg is not None else (
            float(suggestion["ratio"]) if suggestion and "ratio" in suggestion else None
        )
        return {
            "tau": tau_val,
            "ratio": ratio_val,
            "suggest_metrics": suggestion,
        }

    gate_policy = {
        "margin": float(args.gate_margin),
        "short": _resolve_gate("SHORT", args.gate_short_tau, args.gate_short_ratio),
        "long": _resolve_gate("LONG", args.gate_long_tau, args.gate_long_ratio),
    }

    class_counts_all_named = encode_counts(y_all.values, code_to_name)
    class_fraction_all_named = {k: float(v) / max(1, len(y_all)) for k, v in class_counts_all_named.items()}
    encoded_to_name = {int(cls): code_to_name.get(int(cls), str(int(cls))) for cls in encoder.classes_}

    metrics = {
        "n_samples": len(X_all),
        "n_train": len(X_tr),
        "n_val": len(X_va),
        "n_test": len(X_te),
        "classes": [int(c) for c in encoder.classes_],
        "class_names": target_names,
        "ret_threshold": float(threshold_used),
        "label_quantile": None if args.label_quantile is None else float(args.label_quantile),
        "class_counts_all": class_counts_all_named,
        "class_fraction_all": class_fraction_all_named,
        "class_counts_train": encode_counts(y_tr.values, code_to_name),
        "class_counts_val": encode_counts(y_va.values, code_to_name),
        "class_counts_test": encode_counts(y_te.values, code_to_name),
        "log_loss_test": float(logloss),
        "balanced_accuracy_test": float(bal_acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_code_to_name": code_to_name,
        "calibration_requested": calibration_arg,
        "calibration_applied": calibration_applied,
        "best_params": best_params,
        "calibrated": bool(calibrated_flag),
        "keep_all_features": bool(keep_all_features),
        "keep_raw_ohlc": bool(args.keep_raw_ohlc),
        "gate_policy": gate_policy,
        "gate_suggestions": gate_suggest,
    }
    metrics["label_scheme"] = args.label_scheme
    metrics["event_filter"] = args.event_filter
    tb_attr = feats_all.attrs.get("triple_barrier")
    if tb_attr:
        runtime_labels = load_app_config().labels
        schema = LabelSchema(
            domain=tb_attr["schema"].get("name", "triple_barrier_v1"),
            horizon_bars=int(args.tb_max_hold_bars),
            trend_ma_window=runtime_labels.trend_ma_window,
            trend_slope_window=runtime_labels.trend_slope_window,
            drop_flats=bool(args.drop_flat),
            positive_label=1,
            negative_label=-1,
            params=tb_attr["schema"],
        )
        schema_path = args.save_model.with_suffix(".label_schema.json") if args.save_model else None
        if schema_path:
            schema_path.write_text(_json_dumps(schema.model_dump()))

        # Save thresholds.json if enabled
        if thresholds_payload:
            thresholds_path = args.save_model.with_name(args.save_model.stem + "_thresholds.json") if args.save_model else None
            if thresholds_path:
                thresholds_path.write_text(_json_dumps(thresholds_payload))

    # Add SIM-lite metrics
    if sim_metrics:
        metrics["sim_metrics"] = sim_metrics

    setup_series_all = feats_all.get("setup_present")
    if setup_series_all is not None:
        setup_series_all = setup_series_all.fillna(0).astype(int)

        def _setup_counts(index: pd.Index) -> Dict[str, int]:
            subset = setup_series_all.loc[index]
            present = int((subset > 0).sum())
            absent = int((subset <= 0).sum())
            return {"present": present, "absent": absent}

        metrics["setup_counts_all"] = _setup_counts(feats_all.index)
        metrics["setup_counts_train"] = _setup_counts(X_tr.index)
        metrics["setup_counts_val"] = _setup_counts(X_va.index)
        metrics["setup_counts_test"] = _setup_counts(X_te.index)

        setup_for_all = setup_series_all.loc[y_all.index]
        class_breakdown: Dict[str, Dict[str, int]] = {}
        for cls in sorted(pd.unique(y_all)):
            cls_mask = y_all == cls
            present = int(((setup_for_all > 0) & cls_mask).sum())
            absent = int((((setup_for_all <= 0)) & cls_mask).sum())
            class_breakdown[str(int(cls))] = {"present": present, "absent": absent}
        metrics["setup_class_counts_all"] = class_breakdown

    if args.save_preds:
        dt_series = _get_datetime_series(feats_all, X_te.index)
        proba_cols = {
            encoded_to_name[int(cls)]: proba_te[:, idx]
            for idx, cls in enumerate(encoder.classes_)
        }
        preds_df = pd.DataFrame(proba_cols, index=X_te.index)
        preds_df.insert(0, "pred_label", decode_labels(preds, encoded_to_name))
        preds_df.insert(0, "y_true", decode_labels(y_te.values, encoded_to_name))
        preds_df.insert(0, "Datetime", pd.to_datetime(dt_series.values))
        preds_df = preds_df.sort_values("Datetime").reset_index(drop=True)
        outp = Path(args.save_preds)
        outp.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(outp, index=False)
        print(f"[OK] Saved test predictions -> {outp}")
        if proba_va is not None and len(X_va):
            dt_va = _get_datetime_series(feats_all, X_va.index)
            va_cols = {
                encoded_to_name[int(cls)]: proba_va[:, idx]
                for idx, cls in enumerate(encoder.classes_)
            }
            va_df = pd.DataFrame(va_cols, index=X_va.index)
            va_df.insert(0, "y_true", decode_labels(y_va.values, encoded_to_name))
            va_df.insert(0, "Datetime", pd.to_datetime(dt_va.values))
            va_df = va_df.sort_values("Datetime").reset_index(drop=True)
            outp_va = outp.with_name(outp.stem + "_val.csv")
            va_df.to_csv(outp_va, index=False)
            print(f"[OK] Saved validation predictions -> {outp_va}")

    if args.save_metrics:
        Path(args.save_metrics).write_text(_json_dumps(metrics))
        print(f"[OK] Saved metrics -> {args.save_metrics}")

    return metrics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train calibrated HistGradientBoosting multi-class model "
        "(SHORT/FLAT/LONG) on OHLCV data with leak-safe time splits.",
    )
    
    ap.add_argument("--model", default="hgb", choices=["hgb", "xgb"], help="Model type: histogram GB (hgb) or XGBoost (xgb)")
    ap.add_argument("--csv", required=True, help="Path to source OHLCV CSV")
    ap.add_argument("--save-model", default=None, help="Path to save the trained model (.joblib)")
    ap.add_argument("--save-preds", default=None, help="Optional CSV for out-of-sample predictions")
    ap.add_argument("--save-metrics", default=None, help="Optional JSON metrics output path")

    ap.add_argument("--tz", default="America/Denver")
    ap.add_argument("--rth-start", default="07:30")
    ap.add_argument("--rth-end", default="14:00")
    ap.add_argument("--orb-minutes", type=int, default=15)

    ap.add_argument("--horizon", type=int, default=5, help="Forward horizon in bars for labeling")
    ap.add_argument("--ret-threshold", type=float, default=0.0005, help="Return threshold for ternary labels")
    ap.add_argument("--label-quantile", type=float, default=None, help="Auto-select absolute return threshold from tail quantile (0 < q < 0.5)")
    ap.add_argument(
        "--label-scheme",
        default="ternary",
        choices=["binary", "ternary", "direction", "scalp", "triple_barrier"],
    )
    ap.add_argument("--drop-flat", action="store_true", help="Drop flat (0) labels before training")
    ap.add_argument("--use-log-returns", action="store_true", help="Use log returns when building labels")
    ap.add_argument("--target-ticks", type=int, default=6, help="Target ticks for --label-scheme=scalp")
    ap.add_argument("--stop-ticks", type=int, default=4, help="Stop ticks for --label-scheme=scalp")
    ap.add_argument("--horizon-bars", type=int, default=6, help="Forward horizon (bars) for --label-scheme=scalp")
    ap.add_argument("--side", choices=["LONG", "SHORT", "BOTH"], default="BOTH", help="Which sides to train when using --label-scheme=scalp")
    ap.add_argument("--tick-size", type=float, default=0.25, help="Tick size for scalper/triple-barrier features/labels")
    ap.add_argument("--scalp-min-precision", type=float, default=0.8, help="Minimum precision when selecting thresholds (scalp mode)")
    ap.add_argument("--save-scalp-dir", default=None, help="Directory to save scalper models (scalp mode only)")
    ap.add_argument("--scalp-prefix", default="scalper_true", help="Filename prefix for scalper artifacts")
    ap.add_argument("--tb-target-ticks", type=int, default=12, help="Target ticks for --label-scheme=triple_barrier")
    ap.add_argument("--tb-stop-ticks", type=int, default=8, help="Stop ticks for --label-scheme=triple_barrier")
    ap.add_argument("--tb-max-hold-bars", type=int, default=24, help="Max hold in bars for --label-scheme=triple_barrier")
    ap.add_argument("--tb-tie-break", choices=["stop_first", "target_first"], default="stop_first", help="Stop or target hit first in triple-barrier mode")
    ap.add_argument("--tb-session-exit", choices=["timeout_close", "forbid_overnight"], default="timeout_close", help="Default fence for --session-exit (in triple-barrier mode)")
    ap.add_argument("--tb-timeout-exit", choices=["close"], default="close", help="Default fence for --timeout-exit (in triple-barrier mode)")
    ap.add_argument("--tb-clip-r", type=float, default=5.0, help="Default fence for --clip-r (in triple-barrier mode)")
    ap.add_argument("--event-filter", choices=["all", "setup_only"], default="all", help="Restrict training rows to detected setups.")
    ap.add_argument("--enable-threshold-opt", action="store_true", help="Enable threshold optimization per setup/regime using validation data")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--max-nan-frac", type=float, default=0.35, help="Max NaN fraction allowed per feature column")

    feature_controls = ap.add_mutually_exclusive_group(required=False)
    feature_controls.add_argument(
        "--keep-all-features",
        dest="keep_all_features",
        action="store_true",
        help="Preserve every engineered feature (default when --model xgb).",
    )
    feature_controls.add_argument(
        "--filter-features",
        dest="keep_all_features",
        action="store_false",
        help="Enforce leak-safe feature dropping even for XGB.",
    )
    ap.set_defaults(keep_all_features=None)

    ap.add_argument("--min-class-fraction", type=float, default=0.02, help="Warn when any label occupies less than this fraction of samples")
    ap.add_argument("--class-weight", default="auto", choices=["auto", "balanced"], help="Override class weights (default=auto)")
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-samples-leaf", type=int, default=80)
    ap.add_argument("--max-iter", type=int, default=700)
    ap.add_argument("--max-bins", type=int, default=255)
    ap.add_argument("--l2", type=float, default=5e-3)
    ap.add_argument("--reg-alpha", type=float, default=None, help="XGBoost only: L1 regularization term (reg_alpha); defaults to trainer preset when omitted.")
    ap.add_argument("--subsample", type=float, default=None, help="XGBoost only: row subsampling ratio per boosting round.")
    ap.add_argument("--colsample-bytree", type=float, default=None, help="XGBoost only: feature subsampling ratio per tree.")
    ap.add_argument("--min-child-weight", type=float, default=None, help="XGBoost only: minimum sum of instance weight needed in a child node.")
    ap.add_argument("--gamma", type=float, default=None, help="XGBoost only: minimum loss reduction required to make a further partition (tree gamma).")
    ap.add_argument("--tune", action="store_true", help="Evaluate a small hand-tuned parameter grid")
    ap.add_argument("--sample-weight-mode", default="confluence", choices=["confluence", "none"], help="How to build sample weights before class weighting")
    ap.add_argument(
        "--selection-metric",
        default="long_short_accuracy",
        choices=["logloss", "balanced_accuracy", "long_short_accuracy"],
        help="Metric used to pick the best hyper-parameters. `long_short_accuracy` focuses on LONG/SHORT recall.",
    )
    ap.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=80,
        help="XGBoost only: rounds of no validation improvement before stopping (0 disables).",
    )
    ap.add_argument(
        "--calibration",
        default="auto",
        choices=["auto", "none", "sigmoid", "isotonic"],
        help="Calibration strategy applied to the validation split. Auto enables sigmoid for binary models "
             "and disables calibration for multi-class XGB to avoid probability collapse.",
    )
    ap.add_argument(
        "--keep-raw-ohlc",
        action="store_true",
        help="Force raw Open/High/Low/Close columns to survive stationarity filtering.",
    )
    ap.add_argument("--weight-confluence-pos", type=float, default=2.0, help="Sample weight multiplier when VWAP+EMA align with label")
    ap.add_argument("--weight-confluence-neg", type=float, default=0.7, help="Sample weight multiplier when VWAP+EMA oppose the label")
    ap.add_argument("--weight-flat", type=float, default=0.8, help="Relative weight for FLAT class when using ternary labels")
    ap.add_argument("--long-weight-multiplier", type=float, default=1.0, help="Additional multiplier applied to LONG samples before class weighting")
    ap.add_argument("--short-weight-multiplier", type=float, default=1.0, help="Additional multiplier applied to SHORT samples before class weighting")
    ap.add_argument("--flat-weight-multiplier", type=float, default=1.0, help="Additional multiplier applied to FLAT samples before class weighting")
    ap.add_argument("--gpu", action="store_true", help="Use GPU for XGBoost if available")
    # Asymmetric gating parameters
    ap.add_argument(
        "--gate-short-tau",
        type=float,
        default=None,
        help="Optional fixed SHORT probability threshold τ_s for live gating. If omitted, the trainer auto-tunes.",
    )
    ap.add_argument(
        "--gate-long-tau",
        type=float,
        default=None,
        help="Optional fixed LONG probability threshold τ_l for live gating. If omitted, the trainer auto-tunes.",
    )
    ap.add_argument(
        "--gate-short-ratio",
        type=float,
        default=None,
        help="Optional SHORT dominance ratio p_short/(p_long+eps) ≥ r_s. If omitted, the trainer auto-tunes.",
    )
    ap.add_argument(
        "--gate-long-ratio",
        type=float,
        default=None,
        help="Optional LONG dominance ratio p_long/(p_short+eps) ≥ r_l. If omitted, the trainer auto-tunes.",
    )
    ap.add_argument(
        "--gate-margin",
        type=float,
        default=0.02,
        help="Tie-break margin applied when both LONG and SHORT gates pass.",
    )

    ap.add_argument("--random-state", type=int, default=42)
    return ap


def _maybe_path(value: Optional[str]) -> Optional[Path]:
    if value in (None, ""):
        return None
    return Path(value)


def main() -> None:
    args = build_arg_parser().parse_args()
    for attr in ("csv", "save_model", "save_metrics", "save_preds"):
        val = getattr(args, attr, None)
        if isinstance(val, str) or isinstance(val, Path):
            coerced = _maybe_path(val) if isinstance(val, str) else val
            setattr(args, attr, coerced)
    train_pipeline(args)


if __name__ == "__main__":
    main()
