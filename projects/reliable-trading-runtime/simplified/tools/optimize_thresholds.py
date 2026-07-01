from __future__ import annotations

import argparse
import json
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import sys
import numpy as np
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_sharp import run_candidate  # type: ignore
from na.bot.train_phase2 import train_phase2_models


def _feature_hash(features_path: Path) -> str:
    try:
        payload = json.loads(features_path.read_text())
    except Exception:
        payload = {}
    feats: List[str] = []
    if isinstance(payload, dict):
        feats = payload.get("features") or []
    elif isinstance(payload, list):
        feats = payload
    joined = "|".join(sorted(str(f) for f in feats))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold optimizer across walk-forward folds.")
    parser.add_argument("--wf-dir", required=True, help="runs/wf/<tag> directory produced by train_phase2_wf.")
    parser.add_argument("--instrument", default="MES")
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--goal-profit", type=float, default=3000.0)
    parser.add_argument("--max-dd", type=float, default=2000.0)
    parser.add_argument("--max-trades-per-day", type=float, default=4.0)
    parser.add_argument("--max-flip-rate", type=float, default=1.0)
    parser.add_argument("--slippage-levels", default="0,1,2")
    parser.add_argument("--commission", type=float, default=2.0)
    parser.add_argument("--max-hold-bars", type=int, default=24)
    parser.add_argument("--trade-window-start", default=None)
    parser.add_argument("--trade-window-end", default=None)
    parser.add_argument("--out-dir", default=None, help="Override output dir (defaults to wf dir).")
    return parser.parse_args()


def _load_folds(wf_dir: Path) -> List[Dict[str, str]]:
    folds: List[Dict[str, str]] = []
    for fold_dir in sorted(wf_dir.glob("fold_*")):
        manifest = fold_dir / "manifest.json"
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text())
        data["manifest_path"] = str(manifest)
        data["eval_csv"] = data.get("eval_csv", str(fold_dir / "eval.csv"))
        folds.append(data)
    if not folds:
        raise SystemExit(f"No fold manifests found under {wf_dir}")
    return folds


def _threshold_grid() -> List[Tuple[float, float, float]]:
    setup_opts = [0.58, 0.62, 0.66, 0.68, 0.70, 0.72]
    long_opts = [0.56, 0.58, 0.60, 0.62, 0.64]
    short_opts = [0.56, 0.58, 0.60, 0.62, 0.64]
    return [(ps, pl, psht) for ps in setup_opts for pl in long_opts for psht in short_opts]


def _run_sharp_fold(manifest_path: str, eval_csv: str, args: argparse.Namespace, thresholds: Tuple[float, float, float], slippage: float):

    ns = argparse.Namespace(
        tag=None,
        manifest=manifest_path,
        csv=eval_csv,
        instrument=args.instrument,
        contracts=args.contracts,
        trade_window_start=args.trade_window_start,
        trade_window_end=args.trade_window_end,
        max_hold_bars=args.max_hold_bars,
        out_dir=str(Path(args.out_dir or args.wf_dir) / "_sharp_tmp"),
        p_setup=thresholds[0],
        p_long=thresholds[1],
        p_short=thresholds[2],
        commission_per_contract=args.commission,
        slippage_ticks=slippage,
        skip_store=True,
    )
    return run_candidate(ns)


def _scale_constraints(sim: Dict[str, Any], goal_profit: float, max_dd: float, trades_cap: float, flip_cap: float) -> Tuple[bool, float, float]:
    pnl = sim["sim"]["total_pnl_usd"]
    dd = sim["sim"]["max_drawdown"]
    trades_day = sim["sim"]["trades_per_day"]
    flip_rate = sim["sim"]["flip_rate_per_day"]
    if pnl <= 0:
        return False, float("inf"), float("inf")
    scale = math.ceil(goal_profit / pnl)
    scaled_dd = dd * scale
    if scaled_dd > max_dd:
        return False, scale, scaled_dd
    if trades_day > trades_cap or flip_rate > flip_cap:
        return False, scale, scaled_dd
    return True, scale, scaled_dd


def main() -> None:
    args = _parse_args()
    wf_dir = Path(args.wf_dir).expanduser()
    folds = _load_folds(wf_dir)
    slippage_levels = [float(x) for x in str(args.slippage_levels).split(",") if x.strip() != ""]
    combos = _threshold_grid()
    results: Dict[Tuple[float, float, float], Dict[int, Dict[float, Dict[str, Any]]]] = {}

    for combo in combos:
        combo_results: Dict[int, Dict[float, Dict[str, Any]]] = {}
        for fold in folds:
            fold_idx = int(fold.get("fold"))
            fold_results: Dict[float, Dict[str, Any]] = {}
            for slippage in slippage_levels:
                sim = _run_sharp_fold(
                    manifest_path=fold["manifest_path"],
                    eval_csv=fold["eval_csv"],
                    args=args,
                    thresholds=combo,
                    slippage=slippage,
                )
                fold_results[slippage] = sim
            combo_results[fold_idx] = fold_results
        results[combo] = combo_results

    fold_top_rows: Dict[int, List[Dict[str, Any]]] = {int(fold["fold"]): [] for fold in folds}
    robust_rows: List[Dict[str, Any]] = []

    for combo, fold_data in results.items():
        passes_all = True
        fold_pnls = []
        fold_dds = []
        for fold_idx, slip_results in fold_data.items():
            slip_one = slip_results.get(1.0) or slip_results.get(1) or next(iter(slip_results.values()))
            ok, scale, scaled_dd = _scale_constraints(
                slip_one,
                args.goal_profit,
                args.max_dd,
                args.max_trades_per_day,
                args.max_flip_rate,
            )
            if ok:
                fold_top_rows[fold_idx].append(
                    {
                        "fold": fold_idx,
                        "p_setup": combo[0],
                        "p_long": combo[1],
                        "p_short": combo[2],
                        "slippage": 1.0,
                        "pnl_usd": slip_one["sim"]["total_pnl_usd"],
                        "max_dd": slip_one["sim"]["max_drawdown"],
                        "scaled_dd": scaled_dd,
                        "scale_to_goal": scale,
                        "trades_per_day": slip_one["sim"]["trades_per_day"],
                        "flip_rate_per_day": slip_one["sim"]["flip_rate_per_day"],
                    }
                )
                fold_pnls.append(slip_one["sim"]["total_pnl_usd"])
                fold_dds.append(scaled_dd)
            else:
                passes_all = False
        if passes_all and len(fold_pnls) == len(folds):
            robust_rows.append(
                {
                    "p_setup": combo[0],
                    "p_long": combo[1],
                    "p_short": combo[2],
                    "worst_pnl": min(fold_pnls),
                    "mean_pnl": float(np.mean(fold_pnls)),
                    "pnl_std": float(np.std(fold_pnls)),
                    "worst_scaled_dd": max(fold_dds),
                }
            )

    for fold_idx, rows in fold_top_rows.items():
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["pnl_usd"], ascending=False).head(10)
            fold_dir = wf_dir / f"fold_{fold_idx:02d}"
            df.to_csv(fold_dir / "thresholds_top10.csv", index=False)

    overall_df = pd.DataFrame(robust_rows)
    overall_path = Path(args.out_dir or wf_dir) / "thresholds_top10.csv"
    if not overall_df.empty:
        overall_df = overall_df.sort_values(
            ["worst_pnl", "worst_scaled_dd", "pnl_std", "mean_pnl"],
            ascending=[False, True, True, False],
        ).head(10)
        overall_df.to_csv(overall_path, index=False)
    else:
        overall_path.write_text("p_setup,p_long,p_short,worst_pnl,mean_pnl,pnl_std,worst_scaled_dd\n")
        print("No robust threshold set satisfied constraints across folds.")
        return

    champion_row = overall_df.iloc[0]
    combo_key = (float(champion_row["p_setup"]), float(champion_row["p_long"]), float(champion_row["p_short"]))
    config_path = wf_dir / "config.json"
    if not config_path.exists():
        print("walk-forward config.json missing; skipping champion packaging.")
        return
    cfg = json.loads(config_path.read_text())
    champion_dir = Path("artifacts/phase2/champion").expanduser() / cfg["tag"]
    champion_dir.mkdir(parents=True, exist_ok=True)
    setup_path = champion_dir / "setup.joblib"
    dir_path = champion_dir / "dir.joblib"
    close_path = champion_dir / "close.joblib"

    champion_args = argparse.Namespace(
        csv=cfg["csv"],
        setup_model_path=str(setup_path),
        direction_model_path=str(dir_path),
        close_model_path=str(close_path),
        instrument=cfg["instrument"],
        tz=cfg["tz"],
        rth_start=cfg["rth_start"],
        rth_end=cfg["rth_end"],
        orb_minutes=cfg["orb_minutes"],
        horizon=cfg["horizon"],
        label_threshold=cfg["label_threshold"],
        label_log=cfg["label_log"],
        htf_trend_aware=cfg["htf_trend_aware"],
        event_filter=cfg["event_filter"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        direction_early_stopping_rounds=cfg.get("direction_early_stopping_rounds", cfg["early_stopping_rounds"]),
        close_early_stopping_rounds=cfg.get("close_early_stopping_rounds", cfg["early_stopping_rounds"]),
        direction_max_depth=cfg.get("direction_max_depth", 4),
        close_max_depth=cfg.get("close_max_depth", 4),
        min_child_weight=cfg.get("min_child_weight", 15.0),
        direction_min_child_weight=cfg.get("direction_min_child_weight", 10.0),
        close_min_child_weight=cfg.get("close_min_child_weight", 10.0),
        setup_scale_pos_weight=cfg.get("setup_scale_pos_weight", 4.53),
        random_state=cfg["random_state"],
        gpu=cfg["gpu"],
        csv_naive_is_utc=cfg["csv_naive_is_utc"],
        stack_setup_prob=cfg["stack_setup_prob"],
        threshold_objective=cfg.get("threshold_objective", "sharpe"),
        min_trades_val=cfg.get("min_trades_val", 30),
        max_flip_rate=cfg.get("max_flip_rate", 0.2),
        commission_per_contract=cfg.get("commission_per_contract", args.commission),
        slippage_ticks=cfg.get("slippage_ticks", 1.0),
        recency_weighting=cfg.get("recency_weighting", "none"),
        recency_max_weight=cfg.get("recency_max_weight", 2.0),
        recency_half_life_days=cfg.get("recency_half_life_days", 365.0),
        close_threshold=cfg.get("close_threshold", 0.60),
        close_giveback_activate_r=cfg.get("close_giveback_activate_r", 1.0),
        close_giveback_close_r=cfg.get("close_giveback_close_r", 0.5),
        close_stall_bars=cfg.get("close_stall_bars", 4),
        close_stall_min_mfe_r=cfg.get("close_stall_min_mfe_r", 0.25),
        close_stall_close_below_r=cfg.get("close_stall_close_below_r", -0.10),
        close_severe_adverse_r=cfg.get("close_severe_adverse_r", 0.90),
        close_target_arm_min_hold_bars=cfg.get("close_target_arm_min_hold_bars", 3),
        close_target_arm_min_unrealized_r=cfg.get("close_target_arm_min_unrealized_r", 0.75),
    )
    train_phase2_models(champion_args)

    fold_metrics = {}
    for fold_idx, slip_map in results[combo_key].items():
        slip_one = slip_map.get(1.0) or slip_map.get(1)
        stats = slip_one["sim"]
        fold_metrics[str(fold_idx)] = {
            "pnl_usd": stats["total_pnl_usd"],
            "max_dd": stats["max_drawdown"],
            "trades_per_day": stats["trades_per_day"],
            "flip_rate_per_day": stats["flip_rate_per_day"],
        }

    manifest_payload = {
        "tag": cfg["tag"],
        "setup_model_path": str(setup_path),
        "dir_model_path": str(dir_path),
        "close_model_path": str(close_path),
        "close": {
            "enabled": True,
            "threshold": float(cfg.get("close_threshold", 0.60)),
            "model_path": str(close_path),
            "feature_hash": _feature_hash(close_path.with_suffix(".features.json")) if close_path.with_suffix(".features.json").exists() else None,
        },
        "thresholds": {
            "p_setup": combo_key[0],
            "p_long": combo_key[1],
            "p_short": combo_key[2],
        },
        "stack_setup_prob": bool(cfg["stack_setup_prob"]),
        "config": {
            "tz": cfg["tz"],
            "rth_start": cfg["rth_start"],
            "rth_end": cfg["rth_end"],
            "orb_minutes": cfg["orb_minutes"],
            "csv_naive_is_utc": cfg["csv_naive_is_utc"],
        },
        "feature_hash": _feature_hash(dir_path.with_suffix(".features.json")),
        "fold_metrics": fold_metrics,
    }
    champion_manifest_path = champion_dir / "manifest.json"
    champion_manifest_path.write_text(json.dumps(manifest_payload, indent=2))

    slippage_metrics = []
    for slip in slippage_levels:
        sim = _run_sharp_fold(
            manifest_path=str(champion_manifest_path),
            eval_csv=str(cfg["csv"]),
            args=args,
            thresholds=combo_key,
            slippage=slip,
        )
        slippage_metrics.append(
            {
                "slippage": slip,
                "pnl_usd": sim["sim"]["total_pnl_usd"],
                "max_dd": sim["sim"]["max_drawdown"],
                "trades_per_day": sim["sim"]["trades_per_day"],
                "flip_rate_per_day": sim["sim"]["flip_rate_per_day"],
            }
        )
    manifest_payload["slippage_metrics"] = slippage_metrics
    champion_manifest_path.write_text(json.dumps(manifest_payload, indent=2))

    rehearsal_cmd = (
        f'python -m na.discord_addons.cli.stream_live_csv '
        f'--csv "{cfg["csv"]}" --phase2 '
        f'--setup_model_path "{setup_path}" '
        f'--dir_model_path "{dir_path}" '
        f'--model "{dir_path}" '
        f'--p_setup {combo_key[0]:.3f} --p_long {combo_key[1]:.3f} --p_short {combo_key[2]:.3f} '
        f'--instrument {args.instrument} --contracts {args.contracts} '
        f'--risk_config configs/risk/prop_2k_3k.json --risk_profile prop_2k '
        f'--sim_mode --offline_run --reset_state --print_signals '
        f'--out_dir "runs/live/{cfg["tag"]}_champion_rehearsal"'
    )
    live_cmd = rehearsal_cmd.replace("--sim_mode --offline_run --reset_state --print_signals ", "")
    print("\nChampion thresholds selected:", combo_key)
    print("\nRehearsal command:\n", rehearsal_cmd)
    print("\nLive command:\n", live_cmd)


if __name__ == "__main__":
    main()
