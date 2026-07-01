from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.runtime_engine.modeling.train_phase2 import train_phase2_models


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward Phase-2 trainer with optional stacking.")
    parser.add_argument("--csv", required=True, help="Full OHLCV CSV path.")
    parser.add_argument("--tag", required=True, help="Experiment tag (used for output dirs).")
    parser.add_argument("--instrument", default="ES")
    parser.add_argument("--tz", default="America/Denver")
    parser.add_argument("--rth-start", default="07:30")
    parser.add_argument("--rth-end", default="14:00")
    parser.add_argument("--orb-minutes", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--label-threshold", type=float, default=0.0015)
    parser.add_argument("--label-log", action="store_true")
    parser.add_argument("--htf-trend-aware", action="store_true")
    parser.add_argument("--event-filter", choices=["all", "setup_only"], default="all")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--direction-early-stopping-rounds", type=int, default=20)
    parser.add_argument("--close-early-stopping-rounds", type=int, default=20)
    parser.add_argument("--direction-max-depth", type=int, default=4)
    parser.add_argument("--close-max-depth", type=int, default=4)
    parser.add_argument("--min-child-weight", type=float, default=15.0)
    parser.add_argument("--direction-min-child-weight", type=float, default=10.0)
    parser.add_argument("--close-min-child-weight", type=float, default=10.0)
    parser.add_argument("--setup-scale-pos-weight", type=float, default=4.53)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--csv-naive-is-utc", action="store_true")
    parser.add_argument("--stack-setup-prob", action="store_true")
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument("--artifact-root", default="artifacts/phase2/wf")
    parser.add_argument("--wf-dir", default="runs/wf")
    parser.add_argument("--threshold-objective", choices=["sharpe", "calmar", "ev"], default="sharpe")
    parser.add_argument("--min-trades-val", type=int, default=30)
    parser.add_argument("--max-flip-rate", type=float, default=0.2)
    parser.add_argument("--setup-threshold-multiplier", type=float, default=1.75)
    parser.add_argument("--max-bad-fade-rate", type=float, default=0.18)
    parser.add_argument("--commission-per-contract", type=float, default=2.0)
    parser.add_argument("--slippage-ticks", type=float, default=1.0)
    parser.add_argument("--recency-weighting", choices=["none", "linear", "exp", "exponential"], default="none")
    parser.add_argument("--recency-max-weight", type=float, default=2.0)
    parser.add_argument("--recency-half-life-days", type=float, default=365.0)
    parser.add_argument("--close-threshold", type=float, default=0.60)
    parser.add_argument("--close-giveback-activate-r", type=float, default=1.0)
    parser.add_argument("--close-giveback-close-r", type=float, default=0.5)
    parser.add_argument("--close-stall-bars", type=int, default=4)
    parser.add_argument("--close-stall-min-mfe-r", type=float, default=0.25)
    parser.add_argument("--close-stall-close-below-r", type=float, default=-0.10)
    parser.add_argument("--close-severe-adverse-r", type=float, default=0.90)
    parser.add_argument("--close-target-arm-min-hold-bars", type=int, default=3)
    parser.add_argument("--close-target-arm-min-unrealized-r", type=float, default=0.75)
    return parser.parse_args()


def _chunk_days(days: List[pd.Timestamp], n_blocks: int) -> List[np.ndarray]:
    blocks = np.array_split(np.array(days), n_blocks)
    return [blk for blk in blocks if len(blk)]


def _day_series(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(df["Datetime"], errors="coerce")
    return dt.dt.tz_localize(None).dt.normalize()


def main() -> None:
    args = _parse_args()
    raw_df = pd.read_csv(args.csv)
    if "Datetime" not in raw_df.columns:
        raise SystemExit("CSV must contain a Datetime column.")
    raw_df = raw_df.sort_values("Datetime").reset_index(drop=True)
    raw_df["trade_day"] = _day_series(raw_df)
    unique_days = sorted(raw_df["trade_day"].dropna().unique())
    if len(unique_days) < args.n_blocks:
        raise SystemExit(f"Not enough trading days ({len(unique_days)}) for {args.n_blocks} blocks.")

    day_blocks = _chunk_days(list(pd.to_datetime(unique_days)), args.n_blocks)
    wf_root = Path(args.wf_dir).expanduser() / args.tag
    artifact_root = Path(args.artifact_root).expanduser() / args.tag
    wf_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for fold_idx in range(len(day_blocks) - 1):
        train_blocks = day_blocks[: fold_idx + 1]
        eval_block = day_blocks[fold_idx + 1]
        train_days = np.concatenate(train_blocks)
        if len(train_days) < args.min_train_days:
            continue
        eval_days = eval_block
        train_mask = raw_df["trade_day"].isin(train_days)
        eval_mask = raw_df["trade_day"].isin(eval_days)
        train_df = raw_df.loc[train_mask].copy()
        eval_df = raw_df.loc[eval_mask].copy()
        if eval_df.empty or train_df.empty:
            continue

        fold_name = f"fold_{fold_idx + 1:02d}"
        fold_dir = wf_root / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_csv = fold_dir / "train.csv"
        eval_csv = fold_dir / "eval.csv"
        train_df.drop(columns=["trade_day"]).to_csv(train_csv, index=False)
        eval_df.drop(columns=["trade_day"]).to_csv(eval_csv, index=False)

        artifact_dir = artifact_root / fold_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        setup_path = artifact_dir / "setup.joblib"
        dir_path = artifact_dir / "dir.joblib"
        close_path = artifact_dir / "close.joblib"

        phase2_args = argparse.Namespace(
            csv=str(train_csv),
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
            htf_trend_aware=args.htf_trend_aware,
            event_filter=args.event_filter,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
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
            threshold_objective=args.threshold_objective,
            min_trades_val=args.min_trades_val,
            max_flip_rate=args.max_flip_rate,
            setup_threshold_multiplier=args.setup_threshold_multiplier,
            max_bad_fade_rate=args.max_bad_fade_rate,
            commission_per_contract=args.commission_per_contract,
            slippage_ticks=args.slippage_ticks,
            recency_weighting=args.recency_weighting,
            recency_max_weight=args.recency_max_weight,
            recency_half_life_days=args.recency_half_life_days,
            close_threshold=args.close_threshold,
            close_giveback_activate_r=args.close_giveback_activate_r,
            close_giveback_close_r=args.close_giveback_close_r,
            close_stall_bars=args.close_stall_bars,
            close_stall_min_mfe_r=args.close_stall_min_mfe_r,
            close_stall_close_below_r=args.close_stall_close_below_r,
            close_severe_adverse_r=args.close_severe_adverse_r,
            close_target_arm_min_hold_bars=args.close_target_arm_min_hold_bars,
            close_target_arm_min_unrealized_r=args.close_target_arm_min_unrealized_r,
        )

        setup_metrics, dir_metrics, extras = train_phase2_models(phase2_args)

        fold_manifest = {
            "tag": args.tag,
            "fold": fold_idx + 1,
            "train_days": [str(d.date()) for d in train_days],
            "eval_days": [str(d.date()) for d in eval_days],
            "train_start": str(train_df["Datetime"].iloc[0]),
            "train_end": str(train_df["Datetime"].iloc[-1]),
            "eval_start": str(eval_df["Datetime"].iloc[0]),
            "eval_end": str(eval_df["Datetime"].iloc[-1]),
            "artifact_dir": str(artifact_dir),
            "setup_model_path": str(setup_path),
            "dir_model_path": str(dir_path),
            "close_model_path": str(close_path),
            "stack_setup_prob": bool(args.stack_setup_prob),
            "eval_csv": str(eval_csv),
            "config": {
                "tz": args.tz,
                "rth_start": args.rth_start,
                "rth_end": args.rth_end,
                "orb_minutes": args.orb_minutes,
                "csv_naive_is_utc": bool(args.csv_naive_is_utc),
                "notes": f"walk-forward {fold_name}",
            },
            "metrics": {
                "setup": setup_metrics,
                "direction": dir_metrics,
                "close": extras.get("close_metrics"),
            },
            "thresholds": extras.get("thresholds") or {},
        }
        (fold_dir / "manifest.json").write_text(json.dumps(fold_manifest, indent=2))

        summary_rows.append(
            {
                "fold": fold_idx + 1,
                "train_start": fold_manifest["train_start"],
                "train_end": fold_manifest["train_end"],
                "eval_start": fold_manifest["eval_start"],
                "eval_end": fold_manifest["eval_end"],
                "train_days": len(train_days),
                "eval_days": len(eval_days),
            }
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(wf_root / "wf_summary.csv", index=False)
        config_payload = {
            "csv": str(Path(args.csv).expanduser()),
            "instrument": args.instrument,
            "tz": args.tz,
            "rth_start": args.rth_start,
            "rth_end": args.rth_end,
            "orb_minutes": args.orb_minutes,
            "horizon": args.horizon,
            "label_threshold": args.label_threshold,
            "label_log": bool(args.label_log),
            "htf_trend_aware": bool(args.htf_trend_aware),
            "event_filter": args.event_filter,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "early_stopping_rounds": args.early_stopping_rounds,
            "direction_early_stopping_rounds": args.direction_early_stopping_rounds,
            "close_early_stopping_rounds": args.close_early_stopping_rounds,
            "direction_max_depth": args.direction_max_depth,
            "close_max_depth": args.close_max_depth,
            "min_child_weight": args.min_child_weight,
            "direction_min_child_weight": args.direction_min_child_weight,
            "close_min_child_weight": args.close_min_child_weight,
            "setup_scale_pos_weight": args.setup_scale_pos_weight,
            "random_state": args.random_state,
            "gpu": bool(args.gpu),
            "csv_naive_is_utc": bool(args.csv_naive_is_utc),
            "stack_setup_prob": bool(args.stack_setup_prob),
            "threshold_objective": args.threshold_objective,
            "min_trades_val": args.min_trades_val,
            "max_flip_rate": args.max_flip_rate,
            "setup_threshold_multiplier": args.setup_threshold_multiplier,
            "max_bad_fade_rate": args.max_bad_fade_rate,
            "commission_per_contract": args.commission_per_contract,
            "slippage_ticks": args.slippage_ticks,
            "recency_weighting": args.recency_weighting,
            "recency_max_weight": args.recency_max_weight,
            "recency_half_life_days": args.recency_half_life_days,
            "close_threshold": args.close_threshold,
            "tag": args.tag,
        }
        (wf_root / "config.json").write_text(json.dumps(config_payload, indent=2))
    else:
        raise SystemExit("No valid folds were produced; verify --min-train-days and --n-blocks.")


if __name__ == "__main__":
    main()
