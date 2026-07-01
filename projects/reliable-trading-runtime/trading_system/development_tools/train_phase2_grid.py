from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.runtime_engine.modeling.features import build_features
from trading_system.runtime_engine.modeling.feature_constants import MANDATORY_MODEL_FEATURES
from trading_system.runtime_engine.modeling.feature_hash import compute_feature_hash
from trading_system.runtime_engine.modeling.train_phase2 import train_phase2_models

STACK_SETUP_PROB_FEATURE = 'stack_setup_prob'


@dataclass
class CandidateConfig:
    tag: str
    csv: str
    instrument: str
    timeframe: str
    tz: str
    rth_start: str
    rth_end: str
    orb_minutes: int
    horizon: int
    label_threshold: float
    label_log: bool
    htf_trend_aware: bool
    val_ratio: float
    test_ratio: float
    csv_naive_is_utc: bool
    notes: Optional[str] = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Phase-2 setup+direction pair and emit manifest metadata.")
    p.add_argument("--csv", required=True, help="Training OHLCV CSV.")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--tag", required=True, help="Candidate tag (used for artifact directory).")
    p.add_argument("--artifact-root", default="artifacts/phase2/candidates")
    p.add_argument("--tz", default="America/Denver")
    p.add_argument("--rth-start", default="07:30")
    p.add_argument("--rth-end", default="14:00")
    p.add_argument("--orb-minutes", type=int, default=15)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--label-threshold", type=float, default=0.0015)
    p.add_argument("--label-log", action="store_true")
    p.add_argument("--label-mode", choices=["horizon", "exec"], default="horizon")
    p.add_argument("--label-commission-per-contract", type=float, default=2.0)
    p.add_argument("--label-slippage-ticks", type=float, default=1.0)
    p.add_argument("--label-max-hold-bars", type=int, default=24)
    p.add_argument("--htf-trend-aware", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--train-start", default=None, help="Optional start date for training window.")
    p.add_argument("--train-end", default=None, help="Optional end date for training window.")
    p.add_argument("--val-start", default=None, help="Optional start date for validation window.")
    p.add_argument("--val-end", default=None, help="Optional end date for validation window.")
    p.add_argument("--test-start", default=None, help="Optional start date for test window.")
    p.add_argument("--test-end", default=None, help="Optional end date for test window.")
    p.add_argument("--n-estimators", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--direction-max-depth", type=int, default=4)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--reg-alpha", type=float, default=0.1)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--early-stopping-rounds", type=int, default=75, help="Setup early stopping rounds (direction has its own default).")
    p.add_argument("--direction-early-stopping-rounds", type=int, default=50)
    p.add_argument("--close-early-stopping-rounds", type=int, default=50)
    p.add_argument("--min-child-weight", type=float, default=15.0, help="Setup min_child_weight (direction has its own default).")
    p.add_argument("--direction-min-child-weight", type=float, default=10.0)
    p.add_argument("--close-min-child-weight", type=float, default=10.0)
    p.add_argument("--close-max-depth", type=int, default=4)
    p.add_argument("--setup-scale-pos-weight", type=float, default=4.53)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--gpu", action="store_true")
    csv_tz = p.add_mutually_exclusive_group()
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
    p.set_defaults(csv_naive_is_utc=False)
    p.add_argument("--notes", default=None)
    p.add_argument("--stack_setup_prob", action="store_true", help="Include setup probabilities as stacked feature for direction model.")
    p.add_argument("--mtf-timeframes", default="", help="Comma-separated higher-timeframe context built from the same 5m feed.")
    p.add_argument("--threshold-objective", choices=["sharpe", "calmar", "ev"], default="sharpe")
    p.add_argument("--min-trades-val", type=int, default=30)
    p.add_argument("--max-flip-rate", type=float, default=0.2)
    p.add_argument("--walkforward-windows", type=int, default=3)
    p.add_argument("--live-shadow-summary-path", default="run_health_summary.json")
    p.add_argument("--require-safe-shadow-pass", action="store_true")
    p.add_argument("--setup-threshold-start", type=float, default=0.20)
    p.add_argument("--setup-threshold-end", type=float, default=0.85)
    p.add_argument("--setup-threshold-step", type=float, default=0.03)
    p.add_argument("--direction-threshold-start", type=float, default=0.55)
    p.add_argument("--direction-threshold-end", type=float, default=0.82)
    p.add_argument("--direction-threshold-step", type=float, default=0.02)
    p.add_argument("--setup-threshold-multiplier", type=float, default=1.75)
    p.add_argument("--max-bad-fade-rate", type=float, default=0.18)
    p.add_argument("--entry-trend-filter", choices=["none", "vwap_ema"], default="none")
    p.add_argument("--min-signal-persistence-bars", type=int, default=1)
    p.add_argument("--cooldown-bars-after-flip", type=int, default=0)
    p.add_argument("--commission-per-contract", type=float, default=2.0)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--close-threshold", type=float, default=0.60)
    p.add_argument("--close-giveback-activate-r", type=float, default=1.0)
    p.add_argument("--close-giveback-close-r", type=float, default=0.5)
    p.add_argument("--close-stall-bars", type=int, default=4)
    p.add_argument("--close-stall-min-mfe-r", type=float, default=0.25)
    p.add_argument("--close-stall-close-below-r", type=float, default=-0.10)
    p.add_argument("--close-severe-adverse-r", type=float, default=0.90)
    p.add_argument("--close-target-arm-min-hold-bars", type=int, default=3)
    p.add_argument("--close-target-arm-min-unrealized-r", type=float, default=0.75)
    p.add_argument("--recency-weighting", choices=["none", "linear", "exp", "exponential"], default="exponential")
    p.add_argument("--recency-max-weight", type=float, default=2.0)
    p.add_argument("--recency-half-life-days", type=float, default=365.0)
    p.add_argument("--generation", type=int, default=None)
    p.add_argument("--parent-tag", default=None)
    p.add_argument("--promotion-target-tag", default=None)
    p.add_argument("--promotion-result", default=None)
    return p.parse_args()


def _build_idx_lookup(args: argparse.Namespace) -> Dict[int, str]:
    df_raw = pd.read_csv(args.csv)
    feats = build_features(
        df_raw,
        tz=args.tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_minutes=args.orb_minutes,
        csv_naive_is_utc=args.csv_naive_is_utc,
        stack_setup_prob=args.stack_setup_prob,
    )
    idx_map: Dict[int, str] = {}
    datetimes = pd.to_datetime(feats["Datetime"], errors="coerce")
    if getattr(datetimes.dt, "tz", None) is not None:
        datetimes = datetimes.dt.tz_convert("UTC")
    else:
        datetimes = datetimes.dt.tz_localize("UTC")
    datetimes = datetimes.dt.tz_localize(None)
    for idx, ts in zip(feats.index.astype(int), datetimes):
        if pd.isna(ts):
            continue
        idx_map[int(idx)] = pd.Timestamp(ts).isoformat()
    return idx_map


def _idx_to_ts(idx_map: Dict[int, str], value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        idx = int(float(value))
    except Exception:
        return None
    return idx_map.get(idx)


def _load_features_list(path: Path) -> list[str]:
    features: list[str] = []
    try:
        blob = json.loads(path.read_text())
        if isinstance(blob, dict):
            features = list(blob.get("features") or [])
        elif isinstance(blob, list):
            features = [str(x) for x in blob]
    except Exception:
        features = []
    return [str(x) for x in features]


def _feature_hash(path: Path) -> str:
    return compute_feature_hash(_load_features_list(path))


def _validate_setup_direction_schema(
    setup_features: list[str],
    dir_features: list[str],
    *,
    stack_setup_prob: bool,
) -> None:
    if not stack_setup_prob:
        if setup_features != dir_features:
            raise RuntimeError(
                "Setup and direction feature schemas differ.\n"
                f"  setup_count: {len(setup_features)}\n"
                f"  dir_count:   {len(dir_features)}\n"
                "This indicates the two models were trained on different schemas."
            )
        return

    expected_dir_features = list(setup_features) + [STACK_SETUP_PROB_FEATURE]
    if dir_features != expected_dir_features:
        setup_set = set(setup_features)
        dir_set = set(dir_features)
        missing = sorted(setup_set - dir_set)
        extra = sorted(dir_set - setup_set - {STACK_SETUP_PROB_FEATURE})
        stack_count = sum(1 for item in dir_features if item == STACK_SETUP_PROB_FEATURE)
        raise RuntimeError(
            "Stacked setup probability schema is invalid.\n"
            f"  expected direction schema: setup features plus trailing {STACK_SETUP_PROB_FEATURE!r}\n"
            f"  setup_count: {len(setup_features)}\n"
            f"  dir_count:   {len(dir_features)}\n"
            f"  missing_setup_features: {missing}\n"
            f"  unexpected_direction_features: {extra}\n"
            f"  stack_feature_count: {stack_count}"
        )


def _load_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _write_manifest(out_dir: Path, manifest: Dict[str, Any]) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _update_global_manifest(root: Path, entry: Dict[str, Any]) -> None:
    agg_path = root / "manifest.json"
    payload: Dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "candidates": {}}
    if agg_path.exists():
        try:
            payload = json.loads(agg_path.read_text())
        except Exception:
            payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "candidates": {}}
    payload.setdefault("candidates", {})
    payload["candidates"][entry["tag"]] = entry
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    agg_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = _parse_args()
    out_root = Path(args.artifact_root).expanduser().resolve()
    out_dir = out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_path = out_dir / "setup.joblib"
    dir_path = out_dir / "dir.joblib"
    close_path = out_dir / "close.joblib"

    phase2_args = argparse.Namespace(
        csv=args.csv,
        setup_model_path=str(setup_path),
        direction_model_path=str(dir_path),
        close_model_path=str(close_path),
        instrument=args.instrument,
        tz=args.tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_minutes=args.orb_minutes,
        horizon=args.horizon,
        label_threshold=args.label_threshold,
        label_log=args.label_log,
        label_mode=args.label_mode,
        label_commission_per_contract=args.label_commission_per_contract,
        label_slippage_ticks=args.label_slippage_ticks,
        label_max_hold_bars=args.label_max_hold_bars,
        htf_trend_aware=args.htf_trend_aware,
        event_filter="all",
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=args.early_stopping_rounds,
        direction_early_stopping_rounds=args.direction_early_stopping_rounds,
        close_early_stopping_rounds=args.close_early_stopping_rounds,
        direction_max_depth=args.direction_max_depth,
        close_max_depth=args.close_max_depth,
        min_child_weight=args.min_child_weight,
        direction_min_child_weight=args.direction_min_child_weight,
        close_min_child_weight=args.close_min_child_weight,
        setup_scale_pos_weight=args.setup_scale_pos_weight,
        random_state=args.random_state,
        gpu=args.gpu,
        csv_naive_is_utc=args.csv_naive_is_utc,
        stack_setup_prob=args.stack_setup_prob,
        mtf_timeframes=args.mtf_timeframes,
        threshold_objective=args.threshold_objective,
        min_trades_val=args.min_trades_val,
        max_flip_rate=args.max_flip_rate,
        walkforward_windows=args.walkforward_windows,
        live_shadow_summary_path=args.live_shadow_summary_path,
        require_safe_shadow_pass=args.require_safe_shadow_pass,
        setup_threshold_start=args.setup_threshold_start,
        setup_threshold_end=args.setup_threshold_end,
        setup_threshold_step=args.setup_threshold_step,
        direction_threshold_start=args.direction_threshold_start,
        direction_threshold_end=args.direction_threshold_end,
        direction_threshold_step=args.direction_threshold_step,
        setup_threshold_multiplier=args.setup_threshold_multiplier,
        max_bad_fade_rate=args.max_bad_fade_rate,
        entry_trend_filter=args.entry_trend_filter,
        min_signal_persistence_bars=args.min_signal_persistence_bars,
        cooldown_bars_after_flip=args.cooldown_bars_after_flip,
        commission_per_contract=args.commission_per_contract,
        slippage_ticks=args.slippage_ticks,
        close_threshold=args.close_threshold,
        close_giveback_activate_r=args.close_giveback_activate_r,
        close_giveback_close_r=args.close_giveback_close_r,
        close_stall_bars=args.close_stall_bars,
        close_stall_min_mfe_r=args.close_stall_min_mfe_r,
        close_stall_close_below_r=args.close_stall_close_below_r,
        close_severe_adverse_r=args.close_severe_adverse_r,
        close_target_arm_min_hold_bars=args.close_target_arm_min_hold_bars,
        close_target_arm_min_unrealized_r=args.close_target_arm_min_unrealized_r,
        recency_weighting=args.recency_weighting,
        recency_max_weight=args.recency_max_weight,
        recency_half_life_days=args.recency_half_life_days,
)

    setup_metrics, dir_metrics, extras = train_phase2_models(phase2_args)
    thresholds = dict(extras.get("thresholds") or {})
    trading_val = extras.get("trading_val")
    trading_test = extras.get("trading_test")
    rejected = bool(extras.get("rejected"))
    rejected_reason = extras.get("rejected_reason")

    idx_map = _build_idx_lookup(args)
    setup_features_path = setup_path.with_suffix(".features.json")
    dir_features_path = dir_path.with_suffix(".features.json")

    setup_features = _load_features_list(setup_features_path)
    dir_features = _load_features_list(dir_features_path)
    setup_hash = compute_feature_hash(setup_features)
    dir_hash = compute_feature_hash(dir_features)
    _validate_setup_direction_schema(
        setup_features,
        dir_features,
        stack_setup_prob=bool(args.stack_setup_prob),
    )
    feature_hash = dir_hash
    close_features_path = close_path.with_suffix(".features.json")
    close_schema_path = close_path.with_suffix(".label_schema.json")
    close_artifact_present = (
        close_path.exists()
        and close_features_path.exists()
        and close_schema_path.exists()
        and extras.get("close_metrics") is not None
    )
    close_feature_list = _load_features_list(close_features_path) if close_artifact_present else []
    close_feature_hash = compute_feature_hash(close_feature_list) if close_feature_list else None

    # Warn (and force visibility) if the trained schema differs from the current production constants.
    current = set(MANDATORY_MODEL_FEATURES)
    trained = set(setup_features)
    added = trained - current
    removed = current - trained
    if added or removed:
        warnings.warn(
            "Feature schema changed vs feature_constants.py!\n"
            f"  Added:   {sorted(added)}\n"
            f"  Removed: {sorted(removed)}\n"
            "Run tools/gen_feature_constants.py and re-run the full test suite before deploying.",
            UserWarning,
            stacklevel=2,
        )

    ranges = {
        "train": {
            "start_idx": setup_metrics.get("train_start"),
            "end_idx": setup_metrics.get("train_end"),
            "start_ts": _idx_to_ts(idx_map, setup_metrics.get("train_start")),
            "end_ts": _idx_to_ts(idx_map, setup_metrics.get("train_end")),
        },
        "val": {
            "start_idx": setup_metrics.get("val_start"),
            "end_idx": setup_metrics.get("val_end"),
            "start_ts": _idx_to_ts(idx_map, setup_metrics.get("val_start")),
            "end_ts": _idx_to_ts(idx_map, setup_metrics.get("val_end")),
        },
        "test": {
            "start_idx": setup_metrics.get("test_start"),
            "end_idx": setup_metrics.get("test_end"),
            "start_ts": _idx_to_ts(idx_map, setup_metrics.get("test_start")),
            "end_ts": _idx_to_ts(idx_map, setup_metrics.get("test_end")),
        },
    }

    recency_label = args.recency_weighting
    if recency_label == "exp":
        recency_label = "exponential"

    setup_class_counts = (((setup_metrics.get("label_info") or {}).get("class_counts") or {}) if isinstance(setup_metrics, dict) else {})
    setup_trade_count = int(setup_class_counts.get("trade") or 0)
    setup_flat_count = int(setup_class_counts.get("flat") or 0)
    setup_total = max(setup_trade_count + setup_flat_count, 1)
    setup_selectivity_stats = {
        "trade_labels": setup_trade_count,
        "flat_labels": setup_flat_count,
        "trade_rate": float(setup_trade_count) / float(setup_total),
        "trade_threshold": float(setup_metrics.get("trade_threshold") or 0.0),
        "precision_trade_test": float(setup_metrics.get("precision_trade_test") or 0.0),
        "recall_trade_test": float(setup_metrics.get("recall_trade_test") or 0.0),
    }

    manifest = {
        "tag": args.tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv": args.csv,
        "instrument": args.instrument,
        "timeframe": args.timeframe,
        "artifact_dir": ".",
        "setup_model_path": "setup.joblib",
        "dir_model_path": "dir.joblib",
        "close_model_path": "close.joblib" if close_artifact_present else None,
        "thresholds": thresholds,
        "close": {
            "enabled": close_artifact_present,
            "threshold": float(args.close_threshold),
            "model_path": "close.joblib" if close_artifact_present else None,
            "feature_hash": close_feature_hash,
            "feature_count": len(close_feature_list),
            "label_schema_file": "close.label_schema.json" if close_artifact_present else None,
        },
        "feature_hash": feature_hash,
        "feature_hashes": {
            "setup": setup_hash,
            "direction": dir_hash,
            "close": close_feature_hash,
        },
        "config": {
            "tz": args.tz,
            "rth_start": args.rth_start,
            "rth_end": args.rth_end,
            "orb_minutes": args.orb_minutes,
            "horizon": args.horizon,
            "label_mode": args.label_mode,
            "label_threshold": args.label_threshold,
            "label_log": bool(args.label_log),
            "label_commission_per_contract": args.label_commission_per_contract,
            "label_slippage_ticks": args.label_slippage_ticks,
            "label_max_hold_bars": args.label_max_hold_bars,
            "htf_trend_aware": bool(args.htf_trend_aware),
            "train_start": args.train_start,
            "train_end": args.train_end,
            "val_start": args.val_start,
            "val_end": args.val_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "csv_naive_is_utc": bool(args.csv_naive_is_utc),
            "stack_setup_prob": bool(args.stack_setup_prob),
            "mtf_timeframes": [item.strip() for item in str(args.mtf_timeframes).split(",") if item.strip()],
            "threshold_objective": args.threshold_objective,
            "min_trades_val": args.min_trades_val,
            "max_flip_rate": args.max_flip_rate,
            "walkforward_windows": args.walkforward_windows,
            "live_shadow_summary_path": args.live_shadow_summary_path,
            "require_safe_shadow_pass": bool(args.require_safe_shadow_pass),
            "threshold_grid": {
                "setup_start": args.setup_threshold_start,
                "setup_end": args.setup_threshold_end,
                "setup_step": args.setup_threshold_step,
                "direction_start": args.direction_threshold_start,
                "direction_end": args.direction_threshold_end,
                "direction_step": args.direction_threshold_step,
            },
            "setup_threshold_multiplier": args.setup_threshold_multiplier,
            "max_bad_fade_rate": args.max_bad_fade_rate,
            "decision_policy": {
                "entry_trend_filter": args.entry_trend_filter,
                "min_signal_persistence_bars": int(args.min_signal_persistence_bars),
                "cooldown_bars_after_flip": int(args.cooldown_bars_after_flip),
            },
            "commission_per_contract": args.commission_per_contract,
            "slippage_ticks": args.slippage_ticks,
            "recency_weighting": recency_label,
            "recency_max_weight": args.recency_max_weight,
            "recency_half_life_days": args.recency_half_life_days,
            "random_state": args.random_state,
            "close_threshold": args.close_threshold,
            "close_replay": {
                "pnl_giveback_activate_r": args.close_giveback_activate_r,
                "pnl_giveback_close_r": args.close_giveback_close_r,
                "pnl_stall_bars": args.close_stall_bars,
                "pnl_stall_min_mfe_r": args.close_stall_min_mfe_r,
                "pnl_stall_close_below_r": args.close_stall_close_below_r,
                "pnl_severe_adverse_r": args.close_severe_adverse_r,
                "pnl_target_arm_min_hold_bars": args.close_target_arm_min_hold_bars,
                "pnl_target_arm_min_unrealized_r": args.close_target_arm_min_unrealized_r,
            },
        },
        "metrics": {
            "setup": setup_metrics,
            "direction": dir_metrics,
            "close": extras.get("close_metrics"),
        },
        "trading_val": trading_val,
        "trading_test": trading_test,
        "behavior_audit_val": extras.get("behavior_audit_val"),
        "behavior_audit_test": extras.get("behavior_audit_test"),
        "threshold_diagnostics": extras.get("threshold_diagnostics"),
        "walkforward": extras.get("walkforward"),
        "live_shadow_gate": extras.get("live_shadow_gate"),
        "promotion_blocked": bool(extras.get("promotion_blocked")),
        "promotion_blocked_reason": extras.get("promotion_blocked_reason"),
        "setup_selectivity_stats": setup_selectivity_stats,
    }
    if extras.get("direction_diagnostics"):
        manifest["direction_diagnostics"] = extras["direction_diagnostics"]
    if args.generation is not None:
        manifest["generation"] = int(args.generation)
    if args.parent_tag:
        manifest["parent_tag"] = str(args.parent_tag)
    if args.promotion_target_tag:
        manifest["promotion_target_tag"] = str(args.promotion_target_tag)
    if args.promotion_result:
        manifest["promotion_result"] = str(args.promotion_result)
    manifest["rejected"] = rejected
    manifest["rejected_reason"] = rejected_reason

    _write_manifest(out_dir, manifest)
    _update_global_manifest(
        out_root,
        {
            "tag": args.tag,
            "setup_model_path": str(setup_path),
            "dir_model_path": str(dir_path),
            "close_model_path": str(close_path),
        },
    )

    print(json.dumps({
        "tag": args.tag,
        "setup_model_path": str(setup_path),
        "dir_model_path": str(dir_path),
        "close_model_path": str(close_path),
        "thresholds": thresholds,
        "feature_hash": feature_hash,
        "rejected": rejected,
        "rejected_reason": rejected_reason,
    }, indent=2))


if __name__ == "__main__":
    main()



