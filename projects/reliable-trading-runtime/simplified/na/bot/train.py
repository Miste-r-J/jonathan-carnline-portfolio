import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Sequence
import hashlib

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from xgboost import XGBClassifier

from .features import build_features
from .config import ModelConfig, load_config_from_yaml
from .train_simple_lr_gb import _required_orb_columns, _strict_stationary_filter
from .label_exit import LabelConfig, make_horizon_labels
from .labels_triple_barrier import TripleBarrierParams, make_triple_barrier_labels
from ..config.loader import load_app_config
from ..common.label_schema import LabelSchema
from ..common.ids import generate_model_id


logger = logging.getLogger(__name__)

_DIRECTIONAL_DOMAINS = {"ternary", "direction", "trend_aware_ternary"}


def _label_config_from_app(
    labels,
    *,
    horizon: Optional[int],
    threshold: float,
    use_log_label: bool,
    use_htf_trend_aware: bool,
) -> LabelConfig:
    resolved_horizon = horizon or labels.horizon_bars
    return LabelConfig(
        horizon=resolved_horizon,
        threshold=threshold,
        use_log=use_log_label,
        use_htf_trend_aware=use_htf_trend_aware,
        drop_flat=bool(labels.drop_flats or use_htf_trend_aware),
    )


def _index_bounds(index: pd.Index) -> tuple[Optional[str], Optional[str]]:
    if not len(index):
        return None, None
    first = index[0]
    last = index[-1]

    def _to_str(value):
        if isinstance(value, (pd.Timestamp, datetime)):
            return pd.Timestamp(value).isoformat()
        return str(value)

    return _to_str(first), _to_str(last)


def check_no_label_leakage(features: pd.DataFrame) -> None:
    """Basic safeguard against obvious future-looking feature columns."""
    forbidden_tokens = ("future", "lead", "ahead")
    suspicious = [col for col in features.columns if any(token in col.lower() for token in forbidden_tokens)]
    if suspicious:
        raise ValueError(f"Potential label leakage detected in features: {suspicious}")


def _safe_metric(fn, y_true: pd.Series, p: np.ndarray) -> float | None:
    try:
        yv = np.asarray(y_true)
        yv = yv[~pd.isna(yv)]
        if len(np.unique(yv)) < 2:
            return None
        return float(fn(y_true, p))
    except Exception:
        return None


def _make_directional_binary_labels(
    feats: pd.DataFrame,
    *,
    horizon: int,
    threshold: float,
    use_log: bool,
    scheme: str,
) -> tuple[pd.Series, dict]:
    """Binary labels for P(LONG), with shorts mapped to 0 and flats dropped."""
    from .labeling import make_labels

    labeled = make_labels(
        feats,
        horizon=horizon,
        threshold=threshold,
        scheme=scheme,
        use_log=use_log,
        drop_flat=False,
        use_htf_trend_aware=(scheme == "trend_aware_ternary"),
    )
    y_signed = labeled["target"].astype("Int64")
    counts = {
        "long": int((y_signed == 1).sum()),
        "short": int((y_signed == -1).sum()),
        "flat": int((y_signed == 0).sum()),
    }
    mask = y_signed.isin([1, -1])
    y = (y_signed[mask] == 1).astype(int)
    return y, {"signed_counts": counts, "dropped_flats": True, "scheme": scheme}


def _apply_event_filter(feats: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return feats
    if mode == "setup_only":
        if "setup_present" not in feats.columns:
            raise ValueError("setup_present column missing; ensure feature builder emitted setup flags.")
        mask = feats["setup_present"] > 0
        if not mask.any():
            raise ValueError("setup_only filter removed all rows; loosen filter or provide more data.")
        filtered = feats.loc[mask].copy()
        logger.info("Event filter 'setup_only' retained %d/%d rows", len(filtered), len(feats))
        return filtered
    raise ValueError(f"Unknown event filter '{mode}'")


def _time_series_split(X: pd.DataFrame, y: pd.Series, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0.01 < test_ratio < 0.5:
        raise ValueError("test_ratio must be in (0.01, 0.5) for time-series split")
    split_idx = int(len(X) * (1.0 - test_ratio))
    split_idx = max(split_idx, max(1, len(X) - int(len(X) * test_ratio * 1.1)))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Unable to compute a non-empty train/test split; adjust test_ratio or provide more data")
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    if len(X_test) < 50:
        raise ValueError("Test split is too small; supply more data or increase test_ratio")
    return X_train, X_test, y_train, y_test


def _three_way_time_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Chronological train/val/test split."""
    if not 0.0 < val_ratio < 0.5 or not 0.0 < test_ratio < 0.5:
        raise ValueError("val_ratio and test_ratio must be in (0, 0.5)")
    if val_ratio + test_ratio >= 0.8:
        raise ValueError("val_ratio + test_ratio too large; need adequate train data")
    n = len(X)
    test_start = int(n * (1.0 - test_ratio))
    val_start = int(n * (1.0 - test_ratio - val_ratio))
    if min(test_start, val_start) <= 0 or test_start <= val_start:
        raise ValueError("Invalid split; adjust ratios or add more data")
    X_train, y_train = X.iloc[:val_start].copy(), y.iloc[:val_start].copy()
    X_val, y_val = X.iloc[val_start:test_start].copy(), y.iloc[val_start:test_start].copy()
    X_test, y_test = X.iloc[test_start:].copy(), y.iloc[test_start:].copy()
    min_samples = 25
    if len(X_val) < min_samples or len(X_test) < min_samples:
        raise ValueError(
            f"Validation/test splits too small (need >= {min_samples} samples each); add data or adjust ratios"
        )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _training_params_dict(
    *,
    tz: str,
    rth_start: str,
    rth_end: str,
    orb_minutes: int,
    horizon: int,
    val_ratio: float,
    test_ratio: float,
    label_threshold: float,
    use_log_label: bool,
    use_htf_trend_aware: bool,
    early_stopping_rounds: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
    random_state: int,
    use_gpu: bool,
) -> Dict[str, Any]:
    return {
        "tz": tz,
        "rth_start": rth_start,
        "rth_end": rth_end,
        "orb_minutes": orb_minutes,
        "horizon": horizon,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "label_threshold": label_threshold,
        "use_log_label": use_log_label,
        "use_htf_trend_aware": use_htf_trend_aware,
        "early_stopping_rounds": early_stopping_rounds,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "random_state": random_state,
        "use_gpu": use_gpu,
    }


def _write_model_registry_record(
    artifact_path: Path,
    *,
    raw_artifact_path: Path,
    model_id: str,
    instrument: str,
    schema: LabelSchema,
    training_params: Dict[str, Any],
    metrics: Dict[str, Any],
    feature_list: Sequence[str],
    session_info: Dict[str, Any],
    session_hash: str,
) -> None:
    payload = {
        "model_id": model_id,
        "instrument": instrument.upper(),
        "artifact_path": str(artifact_path),
        "raw_artifact_path": str(raw_artifact_path),
        "label_schema": schema.model_dump(),
        "training_config": {"params": training_params, "metrics": metrics},
        "feature_list": list(feature_list),
        "session_config": session_info,
        "session_hash": session_hash,
        "artifacts": {
            "calibrated": str(artifact_path),
            "raw_tree": str(raw_artifact_path),
        },
        "status": "experimental",
    }
    record_path = artifact_path.with_suffix(".registry.json")
    record_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote model registry payload to %s", record_path)
    # TODO: POST payload to backend model registry API once runner guarantees connectivity.


def train_and_eval_xgb(
    df_raw: pd.DataFrame,
    tz: str = "America/Denver",
    rth_start: str = "07:30",
    rth_end: str = "14:00",
    orb_minutes: int = 15,
    horizon: Optional[int] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    early_stopping_rounds: int = 20,
    n_estimators: int = 800,
    learning_rate: float = 0.03,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    use_gpu: bool = False,
    save_model_path: str | None = None,
    label_threshold: float = 0.0005,
    use_log_label: bool = False,
    use_htf_trend_aware: bool = False,
    label_scheme: str = "horizon",
    event_filter: str = "all",
    triple_barrier_params: TripleBarrierParams | None = None,
    side: str = "LONG",
    instrument: str = "ES",
    model_id: Optional[str] = None,
):
    feats = build_features(df_raw, tz=tz, rth_start=rth_start, rth_end=rth_end, orb_minutes=orb_minutes)
    feats = _apply_event_filter(feats, event_filter)
    runtime_labels = load_app_config().labels
    tb_result = None
    if label_scheme == "triple_barrier":
        params = triple_barrier_params or TripleBarrierParams()
        tb_result = make_triple_barrier_labels(
            feats,
            tz=tz,
            rth_start=rth_start,
            rth_end=rth_end,
            params=params,
            side="BOTH",
        )
        direction = side.upper()
        if direction not in {"LONG", "SHORT"}:
            raise ValueError("--side must be LONG or SHORT when --label-scheme=triple_barrier.")
        series = tb_result["long"]["r"] if direction == "LONG" else tb_result["short"]["r"]
        mask = series.notna()
        feats = feats.loc[mask]
        y = (series.loc[mask] > 0).astype(int)
        label_cfg = None
    else:
        label_cfg = _label_config_from_app(
            runtime_labels,
            horizon=horizon,
            threshold=label_threshold,
            use_log_label=use_log_label,
            use_htf_trend_aware=use_htf_trend_aware,
        )
        label_domain = str((runtime_labels.domain or "binary")).lower()
        if use_htf_trend_aware or label_domain in _DIRECTIONAL_DOMAINS:
            scheme = "trend_aware_ternary" if use_htf_trend_aware else label_domain
            y, label_info = _make_directional_binary_labels(
                feats,
                horizon=label_cfg.horizon,
                threshold=label_cfg.threshold,
                use_log=label_cfg.use_log,
                scheme=scheme,
            )
            feats = feats.loc[y.index]
        else:
            y, _ = make_horizon_labels(feats, label_cfg)
            feats = feats.loc[y.index]
            label_info = {"dropped_flats": bool(label_cfg.drop_flat), "scheme": label_domain}
    setup_series = feats["setup_present"].astype(int) if "setup_present" in feats.columns else None
    X = feats.select_dtypes(include=[np.number])
    if X.isna().any().any():
        bad = {col: int(X[col].isna().sum()) for col in X.columns if X[col].isna().any()}
        raise ValueError(f"Feature matrix contains NaNs: {bad}")
    X = _strict_stationary_filter(X)
    check_no_label_leakage(X)

    required = _required_orb_columns(orb_minutes)
    missing = [c for c in required if c not in X.columns]
    if missing:
        print("[WARN] Missing features:", missing)

    X_train, X_val, X_test, y_train, y_val, y_test = _three_way_time_split(
        X, y, val_ratio=val_ratio, test_ratio=test_ratio
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
        use_label_encoder=False,
        random_state=random_state,
        tree_method="gpu_hist" if use_gpu else "hist",
    )

    model = XGBClassifier(**xgb_params)
    eval_set = [(X_val, y_val)]
    try:
        model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, verbose=False)
    except TypeError:
        # Fallback for xgboost versions that don't accept early_stopping_rounds in sklearn API
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Calibrate on validation set only to avoid test leakage
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)

    p_val = calibrated.predict_proba(X_val)[:, 1]
    p_test = calibrated.predict_proba(X_test)[:, 1]

    train_start, train_end = _index_bounds(X_train.index)
    val_start, val_end = _index_bounds(X_val.index)
    test_start, test_end = _index_bounds(X_test.index)

    if setup_series is not None:
        setup_aligned = setup_series.loc[y.index]
    else:
        setup_aligned = None
    metrics = {
        "roc_auc_val": _safe_metric(roc_auc_score, y_val, p_val),
        "avg_precision_val": _safe_metric(average_precision_score, y_val, p_val),
        "log_loss_val": _safe_metric(log_loss, y_val, p_val),
        "brier_score_val": _safe_metric(brier_score_loss, y_val, p_val),
        "roc_auc_test": _safe_metric(roc_auc_score, y_test, p_test),
        "avg_precision_test": _safe_metric(average_precision_score, y_test, p_test),
        "log_loss_test": _safe_metric(log_loss, y_test, p_test),
        "brier_score_test": _safe_metric(brier_score_loss, y_test, p_test),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "feature_count": X.shape[1],
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "event_filter": event_filter,
    }
    if setup_aligned is not None:
        present_mask = setup_aligned > 0
        metrics["setup_counts"] = {
            "present": int(present_mask.sum()),
            "absent": int((~present_mask).sum()),
        }
        class_breakdown: Dict[str, Dict[str, int]] = {}
        for cls in sorted(pd.unique(y)):
            cls_mask = y == cls
            class_breakdown[str(int(cls))] = {
                "present": int((present_mask & cls_mask).sum()),
                "absent": int(((~present_mask) & cls_mask).sum()),
            }
        metrics["setup_class_counts"] = class_breakdown
    metrics["label_scheme"] = label_scheme
    if label_scheme != "triple_barrier":
        metrics["label_info"] = label_info
    if tb_result is not None:
        diag = tb_result["diagnostics"]
        metrics["label_scheme_info"] = {
            "scheme": tb_result["schema"].get("name", "triple_barrier_v1"),
            "side": side.upper(),
            "long_stats": diag.get("long"),
            "short_stats": diag.get("short"),
        }

    logger.info("Training metrics: %s", json.dumps(metrics))

    if save_model_path:
        outp = Path(save_model_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        raw_artifact = outp.with_name(f"{outp.stem}_raw.joblib")
        joblib.dump(model, raw_artifact)
        joblib.dump(calibrated, outp)
        sidecar = outp.with_suffix(".features.json")
        feature_payload = {
            "features": list(X.columns),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        sidecar.write_text(json.dumps(feature_payload, indent=2))
        if tb_result is not None:
            params = triple_barrier_params or TripleBarrierParams()
            schema = LabelSchema(
                domain=tb_result["schema"].get("name", "triple_barrier_v1"),
                horizon_bars=int(params.max_hold_bars),
                trend_ma_window=runtime_labels.trend_ma_window,
                trend_slope_window=runtime_labels.trend_slope_window,
                drop_flats=False,
                positive_label=1,
                negative_label=0,
                params=tb_result["schema"],
            )
        else:
            label_domain = str((runtime_labels.domain or "binary")).lower()
            directional = bool(use_htf_trend_aware or label_domain in _DIRECTIONAL_DOMAINS)
            schema_domain = "directional_binary" if directional else "binary"
            schema = LabelSchema(
                domain=schema_domain,
                horizon_bars=label_cfg.horizon,
                trend_ma_window=runtime_labels.trend_ma_window,
                trend_slope_window=runtime_labels.trend_slope_window,
                drop_flats=bool(label_info.get("dropped_flats", label_cfg.drop_flat)),
                positive_label=1,
                negative_label=0,
                params={
                    "source": "horizon",
                    "input_domain": runtime_labels.domain or "binary",
                    "directional_binary": directional,
                    "use_log": bool(label_cfg.use_log),
                    "threshold": float(label_cfg.threshold),
                    "use_htf_trend_aware": bool(use_htf_trend_aware),
                    "mapping": {"1": "LONG", "0": "SHORT"},
                },
            )
        schema_path = outp.with_suffix(".label_schema.json")
        schema_path.write_text(schema.json(indent=2))
        label_domain = schema.domain or runtime_labels.domain or "unknown"
        resolved_model_id = model_id or generate_model_id(
            instrument,
            label_domain,
            schema.horizon_bars or (label_cfg.horizon if label_cfg else 0),
        )
        meta = {
            "tz": tz,
            "rth_start": rth_start,
            "rth_end": rth_end,
            "orb_minutes": orb_minutes,
            "horizon": label_cfg.horizon if label_cfg else int((triple_barrier_params or TripleBarrierParams()).max_hold_bars),
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "metrics": metrics,
            "label_schema": schema.model_dump(),
            "model_id": resolved_model_id,
            "instrument": instrument.upper(),
            "artifacts": {
                "calibrated_model": outp.name,
                "raw_model": raw_artifact.name,
                "features_file": sidecar.name,
                "label_schema_file": schema_path.name,
            },
            "calibration": {"method": "sigmoid", "cv": "prefit"},
        }
        session_info = {"tz": tz, "rth_start": rth_start, "rth_end": rth_end, "orb_minutes": orb_minutes}
        session_hash = hashlib.sha256(json.dumps(session_info, sort_keys=True).encode("utf-8")).hexdigest()
        meta["session_config"] = session_info
        meta["session_hash"] = session_hash
        meta_path = outp.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        training_params = _training_params_dict(
            tz=tz,
            rth_start=rth_start,
            rth_end=rth_end,
            orb_minutes=orb_minutes,
            horizon=label_cfg.horizon if label_cfg else int((triple_barrier_params or TripleBarrierParams()).max_hold_bars),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            label_threshold=label_threshold,
            use_log_label=use_log_label,
            use_htf_trend_aware=use_htf_trend_aware,
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
        training_params["label_scheme"] = label_scheme
        training_params["event_filter"] = event_filter
        if tb_result is not None:
            training_params["triple_barrier"] = tb_result["schema"]
            training_params["side"] = side.upper()
        _write_model_registry_record(
            outp,
            raw_artifact_path=raw_artifact,
            model_id=resolved_model_id,
            instrument=instrument,
            schema=schema,
            training_params=training_params,
            metrics=metrics,
            feature_list=X.columns,
            session_info=session_info,
            session_hash=session_hash,
        )
        logger.info("Saved model %s with %d features", outp, len(X.columns))

    return calibrated, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    ap.add_argument("--save-model", required=False, help="Path to save trained model")
    ap.add_argument("--tz", default="America/Denver")
    ap.add_argument("--rth-start", default="07:30")
    ap.add_argument("--rth-end", default="14:00")
    ap.add_argument("--orb-minutes", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--label-threshold", type=float, default=0.0005)
    ap.add_argument("--label-log", action="store_true")
    ap.add_argument("--htf-trend-aware", action="store_true", help="Use HTF trend-aware labeling to prevent up signals in downtrends")
    ap.add_argument("--label-scheme", choices=["horizon", "triple_barrier"], default="horizon")
    ap.add_argument("--side", choices=["LONG", "SHORT"], default="LONG", help="Relevant for --label-scheme=triple_barrier.")
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--stop-ticks", type=int, default=8)
    ap.add_argument("--target-ticks", type=int, default=12)
    ap.add_argument("--max-hold-bars", type=int, default=12)
    ap.add_argument("--tb-tie-break", choices=["stop_first", "target_first"], default="stop_first")
    ap.add_argument("--tb-session-exit", choices=["timeout_close", "forbid_overnight"], default="timeout_close")
    ap.add_argument("--tb-timeout-exit", choices=["close"], default="close")
    ap.add_argument("--tb-clip-r", type=float, default=5.0)
    ap.add_argument("--event-filter", choices=["all", "setup_only"], default="all", help="Filter training events before labeling.")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--instrument", default="ES")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--config", default=None, help="YAML config path; CLI overrides config values.")
    args = ap.parse_args()

    default_labels = load_app_config().labels
    default_horizon = args.horizon if args.horizon is not None else default_labels.horizon_bars
    model_cfg = ModelConfig(
        model_path=args.save_model,
        horizon=default_horizon,
        label_threshold=args.label_threshold,
        p_buy=0.0,
        p_sell=0.0,
    )
    if args.config:
        try:
            fc = load_config_from_yaml(args.config)
            model_cfg.horizon = fc.model.horizon
            model_cfg.label_threshold = fc.model.label_threshold
            if fc.model.model_path and not args.save_model:
                model_cfg.model_path = fc.model.model_path
        except Exception as exc:
            logging.getLogger(__name__).debug("Config load failed: %s", exc)

    df = pd.read_csv(args.csv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    tb_params = None
    if args.label_scheme == "triple_barrier":
        tb_params = TripleBarrierParams(
            tick_size=float(args.tick_size),
            stop_ticks=int(args.stop_ticks),
            target_ticks=int(args.target_ticks),
            max_hold_bars=int(args.max_hold_bars),
            tie_break=args.tb_tie_break,
            session_exit=args.tb_session_exit,
            timeout_exit=args.tb_timeout_exit,
            clip_r=float(args.tb_clip_r),
        )

    model, metrics = train_and_eval_xgb(
        df_raw=df,
        tz=args.tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_minutes=args.orb_minutes,
        horizon=model_cfg.horizon,
        label_threshold=model_cfg.label_threshold,
        use_log_label=args.label_log,
        use_htf_trend_aware=args.htf_trend_aware,
        label_scheme=args.label_scheme,
        event_filter=args.event_filter,
        triple_barrier_params=tb_params,
        side=args.side,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        use_gpu=args.gpu,
        save_model_path=model_cfg.model_path or args.save_model or "model.joblib",
        instrument=args.instrument,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
