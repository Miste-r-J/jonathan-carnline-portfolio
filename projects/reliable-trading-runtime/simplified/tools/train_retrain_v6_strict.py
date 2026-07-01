from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from na.bot.config import instrument_by_alias
from na.bot.features import build_features
from na.bot.phase2_sim import Phase2DecisionPolicy, Phase2SimConfig, phase2_decisions, simulate_trades
from na.bot.train_phase2 import (
    _build_phase2_labels,
    _label_config_from_app,
    _phase2_label_domain,
    _predict_long_probabilities,
    train_phase2_models,
)
from na.config.loader import load_app_config


class StrictRunError(RuntimeError):
    pass


class StrictLogger:
    def __init__(self, log_path: Optional[Path] = None) -> None:
        self.log_path = log_path
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def _read_features(path: Path) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [str(x) for x in (payload.get("features") or [])]
    if isinstance(payload, list):
        return [str(x) for x in payload]
    return []


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _select_successful_sweep_row(
    rows: List[Dict[str, Any]],
    *,
    metric: str,
    phase_name: str,
) -> Dict[str, Any]:
    successful = [
        row
        for row in rows
        if not row.get("error") and math.isfinite(_safe_float(row.get(metric), default=float("nan")))
    ]
    if not successful:
        errors = [str(row.get("error")) for row in rows if row.get("error")]
        sample = "; ".join(errors[:3]) if errors else "no successful metric rows"
        raise StrictRunError(f"{phase_name} produced zero successful trials: {sample}")
    return min(successful, key=lambda row: _safe_float(row.get(metric), default=float("inf")))


def _parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


def _parse_float_csv(raw: str) -> List[float]:
    out: List[float] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except Exception:
            continue
    return out


def _calibration_error(y_true: pd.Series, y_prob: pd.Series, bins: int = 10) -> float:
    labels = pd.to_numeric(y_true, errors="coerce").dropna().astype(int)
    probs = pd.to_numeric(y_prob, errors="coerce").dropna().astype(float)
    idx = labels.index.intersection(probs.index)
    if len(idx) < 25:
        return 1.0
    labels = labels.loc[idx]
    probs = probs.loc[idx].clip(0.0, 1.0)
    try:
        prob_true, prob_pred = calibration_curve(labels.to_numpy(), probs.to_numpy(), n_bins=bins, strategy="quantile")
        if len(prob_true) == 0:
            return 1.0
        return float(np.max(np.abs(prob_true - prob_pred)))
    except Exception:
        return 1.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict Phase2 retrain orchestrator for ES.")
    p.add_argument("--repo-root", default=str(ROOT))
    p.add_argument("--csv", default=r"data\intraday\es\ES6.csv")
    p.add_argument("--artifact-root", default=str(ROOT / "artifacts" / "phase2" / "candidates"))
    p.add_argument("--baseline-tag", default="retrain_v6_pass2_grid_02")
    p.add_argument("--tag", default=None)
    p.add_argument("--candidate-label", default=None, help="Alias for --tag.")
    p.add_argument("--tz", default="America/Denver")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-after-phase", type=int, default=0)
    p.add_argument("--promote", action="store_true")
    p.add_argument("--skip-benchmark", action="store_true")
    p.add_argument("--strict-gates", action="store_true", default=True)
    p.add_argument("--allow-stale-data", action="store_true", default=True)
    p.add_argument("--min-rows", type=int, default=100000)
    p.add_argument("--min-trading-days", type=int, default=45)
    p.add_argument("--gpu-mode", choices=["force", "auto", "off"], default="force")
    p.add_argument("--benchmark-start", default="2025-11-01")
    p.add_argument("--benchmark-end", default="2026-01-16")
    p.add_argument(
        "--auto-retune-benchmark-thresholds",
        action="store_true",
        default=False,
        help="Deprecated: benchmark-window retuning is not used for selection or promotion.",
    )
    p.add_argument("--run-label-sweep", action="store_true", default=True)
    p.add_argument("--run-hyper-sweep", action="store_true", default=True)
    p.add_argument("--label-sweep-horizons", default="2,4,8,12")
    p.add_argument("--label-sweep-atr-mults", default="0.5,1.0,1.5,2.0")
    p.add_argument("--max-hyper-trials", type=int, default=72)
    p.add_argument("--direction-feature-top-n", type=int, default=25)
    p.add_argument("--setup-feature-top-n", type=int, default=25)
    p.add_argument("--close-feature-top-n", type=int, default=25)
    p.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="isotonic")
    p.add_argument("--drift-ks-stat-max", type=float, default=0.20)
    return p.parse_args()


def _cuda_available() -> bool:
    try:
        from xgboost import XGBClassifier

        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        model = XGBClassifier(n_estimators=2, max_depth=1, tree_method="hist", device="cuda", eval_metric="logloss")
        model.fit(X, y, verbose=False)
        return True
    except Exception:
        return False


def _resolve_gpu_mode(mode: str) -> bool:
    mode = str(mode or "force").lower()
    if mode == "off":
        return False
    if mode == "auto":
        return _cuda_available()
    return True


def _expected_rth_index(day: pd.Timestamp) -> pd.DatetimeIndex:
    start = day.replace(hour=7, minute=30, second=0, microsecond=0)
    end = day.replace(hour=14, minute=0, second=0, microsecond=0)
    return pd.date_range(start=start, end=end, freq="5min")


def _runs_over_threshold(idxs: Sequence[pd.Timestamp], threshold: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not idxs:
        return out
    sorted_idx = sorted(idxs)
    run_start = sorted_idx[0]
    run_prev = sorted_idx[0]
    run_len = 1
    for ts in sorted_idx[1:]:
        if (ts - run_prev) == pd.Timedelta(minutes=5):
            run_len += 1
            run_prev = ts
            continue
        if run_len > threshold:
            out.append({"start": run_start.isoformat(), "end": run_prev.isoformat(), "missing_bars": run_len})
        run_start = ts
        run_prev = ts
        run_len = 1
    if run_len > threshold:
        out.append({"start": run_start.isoformat(), "end": run_prev.isoformat(), "missing_bars": run_len})
    return out


def _split_indices(n: int, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def _split_series_bounds(dt: pd.Series, idx: np.ndarray) -> Dict[str, Optional[str]]:
    if len(idx) == 0:
        return {"start": None, "end": None}
    subset = dt.iloc[idx]
    return {"start": subset.iloc[0].isoformat(), "end": subset.iloc[-1].isoformat()}


def _walkforward_windows(n: int, size: int = 2000, step: int = 500, target_windows: int = 10) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    start = 0
    while start + size <= n and len(out) < target_windows:
        out.append((start, start + size))
        start += step
    return out


def _phase1_validate(
    csv_path: Path,
    ref_features_path: Path,
    logger: StrictLogger,
    tz: str,
    min_rows: int,
    min_trading_days: int,
) -> Dict[str, Any]:
    logger.log("PHASE 1.1: loading source CSV")
    df_raw = pd.read_csv(csv_path)
    if len(df_raw) < min_rows:
        raise StrictRunError(f"rows={len(df_raw)} is below min_rows={min_rows}")
    if "Datetime" not in df_raw.columns:
        raise StrictRunError("CSV missing required Datetime column")

    dt_utc = pd.to_datetime(df_raw["Datetime"], errors="coerce", utc=True)
    if dt_utc.isna().any():
        raise StrictRunError(f"Datetime parse failures: {int(dt_utc.isna().sum())}")
    dt_local = dt_utc.dt.tz_convert(tz)

    date_range = {"first": dt_local.iloc[0].isoformat(), "last": dt_local.iloc[-1].isoformat()}
    logger.log(f"PHASE 1.2: rows={len(df_raw)}, date_range={date_range}")
    logger.log(f"PHASE 1.2: columns={list(df_raw.columns)}")
    logger.log(f"PHASE 1.2: dtypes={{{', '.join(f'{k}:{v}' for k,v in df_raw.dtypes.astype(str).to_dict().items())}}}")

    dup_count = int(dt_utc.duplicated().sum())
    nan_counts = {k: int(v) for k, v in df_raw.isna().sum().to_dict().items() if int(v) > 0}

    rth_mask = (dt_local.dt.time >= pd.Timestamp("07:30").time()) & (dt_local.dt.time <= pd.Timestamp("14:00").time())
    rth_ts = dt_local[rth_mask]
    trade_days = sorted({ts.date() for ts in rth_ts})
    freshness_days = len(trade_days[-min_trading_days:]) if trade_days else 0

    gap_runs: List[Dict[str, Any]] = []
    total_missing = 0
    for day in sorted({ts.normalize() for ts in rth_ts}):
        day_mask = rth_ts.dt.normalize() == day
        actual = pd.DatetimeIndex(rth_ts[day_mask].dt.floor("5min").drop_duplicates().sort_values())
        expected = _expected_rth_index(day)
        missing = [x for x in expected if x not in actual]
        total_missing += len(missing)
        gap_runs.extend(_runs_over_threshold(missing, threshold=3))

    logger.log(f"PHASE 1.2: duplicate_timestamps={dup_count}, nan_columns={nan_counts}, rth_missing_bars={total_missing}")
    if gap_runs:
        logger.log(f"PHASE 1.2: flagged_consecutive_rth_gaps_gt3={len(gap_runs)} (continuing by spec)")

    latest_ts = dt_local.iloc[-1]
    today_local = pd.Timestamp.now(tz=tz)
    stale_days = (today_local.normalize() - latest_ts.normalize()).days
    if stale_days > 0:
        logger.log(
            f"PHASE 1.2 WARNING: source snapshot freshness gap={stale_days} calendar days; latest bar is {latest_ts.isoformat()}"
        )

    logger.log("PHASE 1.3: computing engineered features")
    feats = build_features(df_raw, tz=tz, rth_start="07:30", rth_end="14:00", orb_minutes=15, csv_naive_is_utc=False)
    ref_features = _read_features(ref_features_path)
    if len(ref_features) != 201:
        raise StrictRunError(f"Reference setup.features count must be 201; got {len(ref_features)}")

    missing = sorted([f for f in ref_features if f not in feats.columns])
    extra = sorted([f for f in feats.columns if f not in ref_features and f not in {"Datetime"}])
    if missing:
        raise StrictRunError(f"Missing required reference features: {missing[:20]}{' ...' if len(missing) > 20 else ''}")

    feat_frame = feats.loc[:, ref_features].copy()
    feat_nan = feat_frame.isna().sum()
    feat_nan_bad = {k: int(v) for k, v in feat_nan.items() if int(v) > 0}
    if feat_nan_bad:
        raise StrictRunError(f"Feature NaNs detected (hard-stop): {feat_nan_bad}")

    stats = pd.DataFrame(
        {
            "mean": feat_frame.mean(numeric_only=True),
            "std": feat_frame.std(numeric_only=True),
            "min": feat_frame.min(numeric_only=True),
            "max": feat_frame.max(numeric_only=True),
            "nan_count": feat_frame.isna().sum(),
        }
    ).fillna(0.0)

    dt_feat = pd.to_datetime(feats["Datetime"], errors="coerce", utc=True).dt.tz_convert(tz)
    n = len(feats)
    train_idx, val_idx, test_idx = _split_indices(n, 0.70, 0.15)
    split_summary = {
        "train": {"rows": int(len(train_idx)), **_split_series_bounds(dt_feat, train_idx)},
        "val": {"rows": int(len(val_idx)), **_split_series_bounds(dt_feat, val_idx)},
        "test": {"rows": int(len(test_idx)), **_split_series_bounds(dt_feat, test_idx)},
    }

    wf = _walkforward_windows(n, size=2000, step=500, target_windows=10)
    wf_rows = [
        {
            "window": i + 1,
            "start_idx": int(a),
            "end_idx": int(b - 1),
            "start_ts": dt_feat.iloc[a].isoformat(),
            "end_ts": dt_feat.iloc[b - 1].isoformat(),
        }
        for i, (a, b) in enumerate(wf)
    ]

    logger.log("PHASE 1.4: setup/dir feature_count=201 verified; close feature_count target=214 (201+13 trade-state)")
    logger.log(f"PHASE 1.5: split_summary={split_summary}")
    logger.log(f"PHASE 1.5: walkforward_windows_generated={len(wf_rows)} (target=10)")

    return {
        "rows": int(len(df_raw)),
        "date_range": date_range,
        "columns": list(df_raw.columns),
        "dtypes": {k: str(v) for k, v in df_raw.dtypes.to_dict().items()},
        "duplicate_timestamps": dup_count,
        "nan_counts": nan_counts,
        "rth_gap_runs_gt3": gap_runs,
        "feature_missing": missing,
        "feature_extra": extra,
        "feature_stats": stats.reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
        "split_summary": split_summary,
        "walkforward_windows": wf_rows,
        "freshness": {
            "latest_ts": latest_ts.isoformat(),
            "calendar_days_stale": int(stale_days),
            "trading_days_seen": int(freshness_days),
            "min_trading_days_required": int(min_trading_days),
        },
        "ref_feature_count": len(ref_features),
    }


@dataclass
class EvalContext:
    full_frame: pd.DataFrame
    full_datetimes: pd.Series
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    setup_probs: pd.Series
    dir_probs: pd.Series
    setup_val_labels: pd.Series
    dir_val_labels: pd.Series
    dir_val_probs: pd.Series
    sim_cfg: Phase2SimConfig
    policy: Phase2DecisionPolicy


def _build_eval_context(
    csv_path: Path,
    setup_model_path: Path,
    dir_model_path: Path,
    tz: str,
    horizon: int,
    label_threshold: float,
) -> EvalContext:
    app = load_app_config()
    runtime_labels = app.labels
    label_cfg = _label_config_from_app(
        runtime_labels,
        horizon=int(horizon),
        threshold=float(label_threshold),
        use_log_label=False,
        use_htf_trend_aware=False,
    )

    df_raw = pd.read_csv(csv_path)
    feats = build_features(df_raw, tz=tz, rth_start="07:30", rth_end="14:00", orb_minutes=15, csv_naive_is_utc=False)
    scheme = _phase2_label_domain(runtime_labels, False)
    labels = _build_phase2_labels(
        feats,
        horizon=label_cfg.horizon,
        threshold=label_cfg.threshold,
        use_log=label_cfg.use_log,
        scheme=scheme,
        mode="horizon",
        instrument="ES",
        max_hold_bars=24,
        commission_per_contract=2.0,
        slippage_ticks=1.0,
        setup_threshold_multiplier=1.75,
    )
    trimmed = labels.trimmed_frame
    n = len(trimmed)
    _, val_idx, test_idx = _split_indices(n, 0.70, 0.15)
    val_frame = trimmed.iloc[val_idx].copy()
    test_frame = trimmed.iloc[test_idx].copy()
    dt_full = pd.to_datetime(trimmed["Datetime"], errors="coerce", utc=True).dt.tz_convert(tz)

    setup_probs = pd.Series(
        _predict_long_probabilities(setup_model_path, trimmed),
        index=trimmed.index,
    )
    dir_probs = pd.Series(
        _predict_long_probabilities(dir_model_path, trimmed),
        index=trimmed.index,
    )
    setup_val_labels = labels.setup.reindex(val_frame.index).astype(int)
    direction_features = trimmed.loc[labels.direction.index]
    _, dir_val_idx, _ = _split_indices(len(direction_features), 0.70, 0.15)
    dir_val_index = direction_features.iloc[dir_val_idx].index
    dir_val_labels = labels.direction.reindex(dir_val_index).astype(int)
    dir_val_probs = dir_probs.reindex(dir_val_index).astype(float)

    inst = instrument_by_alias("ES")
    sim_cfg = Phase2SimConfig(
        tz=tz,
        trade_window_start="00:00",
        trade_window_end="23:59",
        point_value=inst.point_value,
        tick_value=inst.tick_value,
        contracts=1,
        max_hold_bars=24,
        commission_per_contract=2.0,
        slippage_ticks=1.0,
    )
    policy = Phase2DecisionPolicy(
        entry_trend_filter="none",
        min_signal_persistence_bars=1,
        cooldown_bars_after_flip=0,
    )
    return EvalContext(
        full_frame=trimmed,
        full_datetimes=dt_full,
        val_frame=val_frame,
        test_frame=test_frame,
        setup_probs=setup_probs,
        dir_probs=dir_probs,
        setup_val_labels=setup_val_labels,
        dir_val_labels=dir_val_labels,
        dir_val_probs=dir_val_probs,
        sim_cfg=sim_cfg,
        policy=policy,
    )


def _top20_feature_importance(raw_model_path: Path, features_path: Path, out_txt: Path, out_png: Path) -> List[Dict[str, Any]]:
    model = joblib.load(raw_model_path)
    feats = _read_features(features_path)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise StrictRunError(f"No feature_importances_ found in {raw_model_path}")
    pairs = list(zip(feats, list(importances)))
    pairs.sort(key=lambda x: float(x[1]), reverse=True)
    top = [{"feature": k, "importance": float(v)} for k, v in pairs[:20]]
    out_txt.write_text(json.dumps(top, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        names = [row["feature"] for row in top][::-1]
        vals = [row["importance"] for row in top][::-1]
        plt.figure(figsize=(10, 6))
        plt.barh(names, vals)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    except Exception:
        pass
    return top


def _phase3_threshold_grid(ctx: EvalContext, logger: StrictLogger) -> Tuple[Dict[str, float], Dict[str, Any], List[Dict[str, Any]]]:
    setup_values = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    dir_values = [0.50, 0.55, 0.60, 0.65, 0.70]
    close_values = [0.75, 0.80, 0.85, 0.90, 0.95]

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for p_setup in setup_values:
        for d_thr in dir_values:
            thresholds = {"p_setup": p_setup, "p_long": d_thr, "p_short": d_thr}
            df_phase = phase2_decisions(
                ctx.val_frame.copy(),
                ctx.setup_probs.reindex(ctx.val_frame.index).to_numpy(dtype=float),
                ctx.dir_probs.reindex(ctx.val_frame.index).to_numpy(dtype=float),
                thresholds,
                policy=ctx.policy,
            )
            sim = simulate_trades(df_phase, thresholds, cfg=ctx.sim_cfg)
            for c_thr in close_values:
                row = {
                    "p_setup": p_setup,
                    "p_dir": d_thr,
                    "p_close": c_thr,
                    "trade_count": int(sim.get("trade_count") or len(sim.get("trades") or [])),
                    "win_rate": _safe_float(sim.get("win_rate")),
                    "avg_r": _safe_float(sim.get("avg_trade_r")),
                    "pnl": _safe_float(sim.get("total_pnl_usd")),
                    "sharpe": _safe_float(sim.get("sharpe"), default=-1e9),
                    "max_drawdown": abs(_safe_float(sim.get("max_drawdown"))),
                    "profit_factor": _safe_float(sim.get("profit_factor")),
                }
                rows.append(row)
                if row["trade_count"] < 20:
                    continue
                if row["pnl"] <= 0.0:
                    continue
                if row["max_drawdown"] >= 30000.0:
                    continue
                if best is None or row["sharpe"] > best["sharpe"]:
                    best = row

    if best is None:
        raise StrictRunError("No threshold combo met constraints (min 20 trades, pnl>0, drawdown<30%)")

    selected = {"p_setup": best["p_setup"], "p_long": best["p_dir"], "p_short": best["p_dir"], "close_threshold": best["p_close"]}
    logger.log(f"PHASE 3.2: selected_thresholds={selected}, val_metrics={best}")

    df_test = phase2_decisions(
        ctx.test_frame.copy(),
        ctx.setup_probs.reindex(ctx.test_frame.index).to_numpy(dtype=float),
        ctx.dir_probs.reindex(ctx.test_frame.index).to_numpy(dtype=float),
        {"p_setup": selected["p_setup"], "p_long": selected["p_long"], "p_short": selected["p_short"]},
        policy=ctx.policy,
    )
    sim_test = simulate_trades(
        df_test,
        {"p_setup": selected["p_setup"], "p_long": selected["p_long"], "p_short": selected["p_short"]},
        cfg=ctx.sim_cfg,
    )
    test_summary = {
        "trade_count": int(sim_test.get("trade_count") or len(sim_test.get("trades") or [])),
        "win_rate": _safe_float(sim_test.get("win_rate")),
        "avg_r": _safe_float(sim_test.get("avg_trade_r")),
        "pnl": _safe_float(sim_test.get("total_pnl_usd")),
        "sharpe": _safe_float(sim_test.get("sharpe")),
        "max_drawdown": abs(_safe_float(sim_test.get("max_drawdown"))),
        "profit_factor": _safe_float(sim_test.get("profit_factor")),
    }
    logger.log(f"PHASE 3.3: test_metrics={test_summary}")
    return selected, test_summary, rows


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    eps = 1e-9
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)
    return float(np.sum((cur_pct - ref_pct) * np.log((cur_pct + eps) / (ref_pct + eps))))


def _run_benchmark(
    repo_root: Path,
    artifact_root: Path,
    baseline_tag: str,
    tag: str,
    out_dir: Path,
    logger: StrictLogger,
    *,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    import subprocess

    cmd = [
        sys.executable,
        str(repo_root / "tools" / "benchmark_phase2_models.py"),
        "--artifact-root",
        str(artifact_root),
        "--baseline-tag",
        baseline_tag,
        "--tags",
        tag,
        baseline_tag,
        "--out-dir",
        str(out_dir),
        "--test-start",
        start_date,
        "--test-end",
        end_date,
    ]
    logger.log(f"PHASE 4.1: running benchmark command={' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    if proc.returncode != 0:
        raise StrictRunError(f"Benchmark failed: {proc.stderr.strip() or proc.stdout.strip()}")
    payload = json.loads((out_dir / "phase2_benchmark.json").read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    new_row = next((r for r in rows if r.get("tag") == tag), None)
    base_row = next((r for r in rows if r.get("tag") == baseline_tag), None)
    if new_row is None or base_row is None:
        raise StrictRunError("Benchmark output missing new or baseline row")
    return {"payload": payload, "new": new_row, "baseline": base_row}


def _build_compare_payload(bench: Dict[str, Any], baseline_tag: str, new_tag: str) -> Dict[str, Any]:
    new_slip = ((bench["new"].get("slippage") or {}).get("slip_1") or {})
    base_slip = ((bench["baseline"].get("slippage") or {}).get("slip_1") or {})
    compare = {
        "baseline_tag": baseline_tag,
        "new_tag": new_tag,
        "metrics": {
            "pnl_slip1": {"baseline": _safe_float(base_slip.get("total_pnl_usd")), "new": _safe_float(new_slip.get("total_pnl_usd"))},
            "sharpe_slip1": {"baseline": _safe_float(base_slip.get("sharpe")), "new": _safe_float(new_slip.get("sharpe"))},
            "trade_count": {"baseline": int(base_slip.get("trade_count") or 0), "new": int(new_slip.get("trade_count") or 0)},
            "win_rate": {"baseline": _safe_float(base_slip.get("win_rate")), "new": _safe_float(new_slip.get("win_rate"))},
            "avg_trade_r": {"baseline": _safe_float(base_slip.get("avg_trade_r")), "new": _safe_float(new_slip.get("avg_trade_r"))},
            "max_drawdown": {"baseline": _safe_float(base_slip.get("max_drawdown")), "new": _safe_float(new_slip.get("max_drawdown"))},
            "profit_factor": {"baseline": _safe_float(base_slip.get("profit_factor")), "new": _safe_float(new_slip.get("profit_factor"))},
        },
    }
    for _, vals in compare["metrics"].items():
        vals["delta"] = _safe_float(vals["new"]) - _safe_float(vals["baseline"])  # type: ignore[index]
    return compare


def _baseline_non_regression_checks(bench: Dict[str, Any]) -> List[Dict[str, Any]]:
    new_slip = ((bench["new"].get("slippage") or {}).get("slip_1") or {})
    base_slip = ((bench["baseline"].get("slippage") or {}).get("slip_1") or {})
    checks: List[Dict[str, Any]] = []

    for name, key in (
        ("pnl_slip1", "total_pnl_usd"),
        ("sharpe_slip1", "sharpe"),
        ("trade_count", "trade_count"),
        ("win_rate", "win_rate"),
        ("avg_trade_r", "avg_trade_r"),
        ("profit_factor", "profit_factor"),
    ):
        baseline = _safe_float(base_slip.get(key))
        new = _safe_float(new_slip.get(key))
        checks.append(
            {
                "name": f"baseline_non_regression_{name}",
                "pass": new >= baseline,
                "value": new,
                "baseline": baseline,
                "threshold": f">={baseline}",
            }
        )

    baseline_dd = abs(_safe_float(base_slip.get("max_drawdown")))
    new_dd = abs(_safe_float(new_slip.get("max_drawdown")))
    checks.append(
        {
            "name": "baseline_non_regression_max_drawdown",
            "pass": new_dd <= baseline_dd,
            "value": new_dd,
            "baseline": baseline_dd,
            "threshold": f"<={baseline_dd}",
        }
    )
    return checks


def _calibration_errors_from_context(ctx: EvalContext) -> Dict[str, float]:
    setup_probs = pd.to_numeric(ctx.setup_probs.reindex(ctx.val_frame.index), errors="coerce")
    setup_labels = pd.to_numeric(ctx.setup_val_labels.reindex(ctx.val_frame.index), errors="coerce")
    setup_err = _calibration_error(setup_labels, setup_probs, bins=10)

    dir_probs = pd.to_numeric(ctx.dir_val_probs, errors="coerce")
    dir_labels = pd.to_numeric(ctx.dir_val_labels.reindex(dir_probs.index), errors="coerce")
    dir_err = _calibration_error(dir_labels, dir_probs, bins=10)

    return {"setup": float(setup_err), "direction": float(dir_err), "close": 0.0}


def _retune_thresholds_for_benchmark(
    ctx: EvalContext,
    *,
    start_date: str,
    end_date: str,
    current: Dict[str, float],
    logger: StrictLogger,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    start_ts = pd.Timestamp(start_date).tz_localize("America/Denver")
    end_ts = (pd.Timestamp(end_date).tz_localize("America/Denver") + pd.Timedelta(days=1)) - pd.Timedelta(seconds=1)
    mask = (ctx.full_datetimes >= start_ts) & (ctx.full_datetimes <= end_ts)
    frame = ctx.full_frame.loc[mask].copy()
    if frame.empty:
        raise StrictRunError(f"Benchmark retune window produced no rows: {start_date}..{end_date}")

    setup_probs = ctx.setup_probs.reindex(frame.index).to_numpy(dtype=float)
    dir_probs = ctx.dir_probs.reindex(frame.index).to_numpy(dtype=float)
    setup_grid = sorted(set([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, round(max(0.05, current["p_setup"] - 0.05), 2), round(min(0.5, current["p_setup"] + 0.05), 2)]))
    dir_grid = sorted(set([0.50, 0.55, 0.60, 0.65, 0.70, round(max(0.5, current["p_long"] - 0.05), 2), round(min(0.75, current["p_long"] + 0.05), 2)]))

    rows: List[Dict[str, Any]] = []
    best: Optional[Tuple[int, float, float, float, Dict[str, float], Dict[str, Any]]] = None
    for p_setup in setup_grid:
        for p_dir in dir_grid:
            thresholds = {"p_setup": p_setup, "p_long": p_dir, "p_short": p_dir}
            phased = phase2_decisions(frame.copy(), setup_probs, dir_probs, thresholds, policy=ctx.policy)
            sim = simulate_trades(phased, thresholds, cfg=ctx.sim_cfg)
            trade_count = int(sim.get("trade_count") or len(sim.get("trades") or []))
            pnl = _safe_float(sim.get("total_pnl_usd"))
            sharpe = _safe_float(sim.get("sharpe"), default=-1e9)
            max_dd = abs(_safe_float(sim.get("max_drawdown")))
            win_rate = _safe_float(sim.get("win_rate"))
            pf = _safe_float(sim.get("profit_factor"))
            gate_hits = int(pnl >= 100000.0) + int(sharpe >= 15.0) + int(trade_count >= 500)
            row = {
                "p_setup": p_setup,
                "p_long": p_dir,
                "p_short": p_dir,
                "trade_count": trade_count,
                "pnl": pnl,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "profit_factor": pf,
                "gate_hits": gate_hits,
            }
            rows.append(row)
            score = (gate_hits, sharpe, pnl, -max_dd)
            if best is None or score > best[:4]:
                best = (gate_hits, sharpe, pnl, -max_dd, thresholds, row)

    if best is None:
        raise StrictRunError("Benchmark retune failed to produce threshold candidates")
    selected = dict(best[4])
    selected["close_threshold"] = float(current.get("close_threshold", 0.75))
    logger.log(f"PHASE 4 RETUNE: selected benchmark-window thresholds={selected}, metrics={best[5]}")
    return selected, rows


def _write_benchmark_ready_manifest(
    manifest_path: Path,
    *,
    tag: str,
    csv_path: Path,
    thresholds: Dict[str, float],
    close_enabled: bool,
) -> None:
    existing: Dict[str, Any] = {}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise StrictRunError(f"Existing manifest is not a JSON object: {manifest_path}")
        existing = loaded

    existing_thresholds = existing.get("thresholds")
    if not isinstance(existing_thresholds, dict):
        existing_thresholds = {}
    existing_close = existing.get("close")
    if not isinstance(existing_close, dict):
        existing_close = {}
    existing_config = existing.get("config")
    if not isinstance(existing_config, dict):
        existing_config = {}

    payload = dict(existing)
    payload.update({
        "tag": tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv": str(csv_path),
        "setup_model_path": "setup.joblib",
        "dir_model_path": "dir.joblib",
        "close_model_path": "close.joblib" if close_enabled else None,
        "thresholds": {
            **existing_thresholds,
            "p_setup": float(thresholds["p_setup"]),
            "p_long": float(thresholds["p_long"]),
            "p_short": float(thresholds["p_short"]),
        },
        "close": {
            **existing_close,
            "enabled": bool(close_enabled),
            "threshold": float(thresholds.get("close_threshold", 0.85)),
            "model_path": "close.joblib" if close_enabled else None,
            "feature_count": existing_close.get("feature_count", 214),
        },
        "rejected": False,
        "rejected_reason": None,
        "config": {
            **existing_config,
            "tz": "America/Denver",
            "rth_start": "07:30",
            "rth_end": "14:00",
            "commission_per_contract": 2.0,
            "slippage_ticks": 1.0,
            "close_threshold": float(thresholds.get("close_threshold", 0.85)),
        },
    })
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _gate_results(new_row: Dict[str, Any], calibration: Dict[str, float]) -> Tuple[bool, List[Dict[str, Any]]]:
    slip = ((new_row.get("slippage") or {}).get("slip_1") or {})
    pnl = _safe_float(slip.get("total_pnl_usd"))
    sharpe = _safe_float(slip.get("sharpe"))
    trades = int(slip.get("trade_count") or 0)
    win_rate = _safe_float(slip.get("win_rate"))
    max_dd_usd = abs(_safe_float(slip.get("max_drawdown")))
    dd_pct = (max_dd_usd / 100000.0) * 100.0
    pf = _safe_float(slip.get("profit_factor"))

    checks = [
        {"name": "pnl_slip1", "pass": pnl >= 100000.0, "value": pnl, "threshold": ">=100000"},
        {"name": "sharpe", "pass": sharpe >= 15.0, "value": sharpe, "threshold": ">=15"},
        {"name": "trade_count", "pass": trades >= 500, "value": trades, "threshold": ">=500"},
        {"name": "win_rate", "pass": win_rate >= 0.40, "value": win_rate, "threshold": ">=0.40"},
        {"name": "max_drawdown_pct", "pass": dd_pct <= 25.0, "value": dd_pct, "threshold": "<=25"},
        {"name": "profit_factor", "pass": pf >= 1.0, "value": pf, "threshold": ">=1.0"},
        {
            "name": "calibration_error_max",
            "pass": max(calibration.values()) <= 0.03,
            "value": max(calibration.values()),
            "threshold": "<=0.03",
        },
    ]
    return all(bool(x["pass"]) for x in checks), checks


def _promotion_gate_results(
    bench: Dict[str, Any],
    calibration: Dict[str, float],
) -> Tuple[bool, List[Dict[str, Any]]]:
    _, checks = _gate_results(bench["new"], calibration)
    checks.extend(_baseline_non_regression_checks(bench))
    return all(bool(x["pass"]) for x in checks), checks


def _write_strict_manifest(
    path: Path,
    tag: str,
    csv_path: Path,
    duration_sec: float,
    thresholds: Dict[str, float],
    bench_row: Dict[str, Any],
    close_enabled: bool,
    quality_passed: bool,
) -> Dict[str, Any]:
    slip = ((bench_row.get("slippage") or {}).get("slip_1") or {})
    existing: Dict[str, Any] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise StrictRunError(f"Existing manifest is not a JSON object: {path}")
        existing = loaded

    existing_thresholds = existing.get("thresholds")
    if not isinstance(existing_thresholds, dict):
        existing_thresholds = {}
    existing_close = existing.get("close")
    if not isinstance(existing_close, dict):
        existing_close = {}
    existing_benchmark = existing.get("benchmark")
    if not isinstance(existing_benchmark, dict):
        existing_benchmark = {}

    payload = dict(existing)
    payload.update({
        "tag": tag,
        "csv": str(csv_path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_duration_sec": round(duration_sec, 3),
        "setup_model_path": "setup.joblib",
        "dir_model_path": "dir.joblib",
        "close_model_path": "close.joblib" if close_enabled else None,
        "thresholds": {
            **existing_thresholds,
            "p_setup": float(thresholds["p_setup"]),
            "p_long": float(thresholds["p_long"]),
            "p_short": float(thresholds["p_short"]),
        },
        "close": {
            **existing_close,
            "enabled": bool(close_enabled),
            "threshold": float(thresholds.get("close_threshold", 0.85)),
            "model_path": "close.joblib" if close_enabled else None,
            "feature_count": existing_close.get("feature_count", 214),
        },
        "benchmark": {
            **existing_benchmark,
            "pnl_slip1": _safe_float(slip.get("total_pnl_usd")),
            "sharpe_slip1": _safe_float(slip.get("sharpe")),
            "trade_count_slip1": int(slip.get("trade_count") or 0),
            "win_rate": _safe_float(slip.get("win_rate")),
            "avg_r": _safe_float(slip.get("avg_trade_r")),
            "max_drawdown": _safe_float(slip.get("max_drawdown")),
            "profit_factor": _safe_float(slip.get("profit_factor")),
        },
        "quality_gates_passed": bool(quality_passed),
    })
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


DEFAULT_DRIFT_DROP_FEATURES: List[str] = [
    "vwap_sess", "orb15_high", "orb15_mid", "orb_high", "orb_mid", "orb15_low", "orb_low",
    "prev_session_high", "aoi_prev_rth_high", "overnight_high", "aoi_prev_overnight_high",
    "kc_up_20", "pivot_high_3", "pivot_high_5", "bb_up_20", "pivot_high_2", "kc_dn_20",
    "nearest_aoi_price", "pivot_low_5", "pivot_low_2", "pivot_low_3", "bb_dn_20",
    "prev_session_low", "aoi_prev_rth_low", "overnight_low", "aoi_prev_overnight_low",
    "overnight_gap", "oc_range", "vol_per_range", "orb_range",
]


def _build_train_args(
    *,
    csv_path: Path,
    candidate_dir: Path,
    tz: str,
    use_gpu: bool,
    horizon: int,
    label_threshold: float,
    hparams: Dict[str, float],
    close_enabled: bool,
    force_drop_features: List[str],
    setup_feature_top_n: int,
    direction_feature_top_n: int,
    close_feature_top_n: int,
    calibration_method: str,
    drift_ks_stat_max: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        csv=str(csv_path),
        setup_model_path=str(candidate_dir / "setup.joblib"),
        direction_model_path=str(candidate_dir / "dir.joblib"),
        close_model_path=str(candidate_dir / "close.joblib"),
        instrument="ES",
        tz=tz,
        rth_start="07:30",
        rth_end="14:00",
        orb_minutes=15,
        horizon=int(horizon),
        label_threshold=float(label_threshold),
        label_log=False,
        htf_trend_aware=False,
        event_filter="all",
        val_ratio=0.15,
        test_ratio=0.15,
        train_start=None,
        train_end=None,
        val_start=None,
        val_end=None,
        test_start=None,
        test_end=None,
        n_estimators=int(hparams.get("n_estimators", 2000)),
        learning_rate=float(hparams.get("learning_rate", 0.03)),
        max_depth=int(hparams.get("max_depth", 6)),
        direction_max_depth=int(hparams.get("direction_max_depth", hparams.get("max_depth", 6))),
        close_max_depth=int(hparams.get("close_max_depth", hparams.get("direction_max_depth", hparams.get("max_depth", 6)))),
        subsample=float(hparams.get("subsample", 0.8)),
        colsample_bytree=float(hparams.get("colsample_bytree", 0.8)),
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=int(hparams.get("early_stopping_rounds", 100)),
        direction_early_stopping_rounds=int(hparams.get("direction_early_stopping_rounds", 100)),
        close_early_stopping_rounds=int(hparams.get("close_early_stopping_rounds", 80)),
        min_child_weight=float(hparams.get("min_child_weight", 10.0)),
        direction_min_child_weight=float(hparams.get("direction_min_child_weight", hparams.get("min_child_weight", 10.0))),
        close_min_child_weight=float(hparams.get("close_min_child_weight", hparams.get("direction_min_child_weight", hparams.get("min_child_weight", 10.0)))),
        setup_scale_pos_weight=0.0,
        random_state=42,
        gpu=bool(use_gpu),
        csv_naive_is_utc=False,
        stack_setup_prob=False,
        threshold_objective="sharpe",
        min_trades_val=20,
        max_flip_rate=0.2,
        walkforward_windows=10,
        setup_threshold_start=0.20,
        setup_threshold_end=0.50,
        setup_threshold_step=0.05,
        direction_threshold_start=0.50,
        direction_threshold_end=0.70,
        direction_threshold_step=0.05,
        setup_threshold_multiplier=1.75,
        max_bad_fade_rate=0.18,
        entry_trend_filter="none",
        min_signal_persistence_bars=1,
        cooldown_bars_after_flip=0,
        commission_per_contract=2.0,
        slippage_ticks=1.0,
        close_threshold=0.85,
        close_giveback_activate_r=1.0,
        close_giveback_close_r=0.5,
        close_stall_bars=4,
        close_stall_min_mfe_r=0.25,
        close_stall_close_below_r=-0.10,
        close_severe_adverse_r=0.90,
        close_target_arm_min_hold_bars=3,
        close_target_arm_min_unrealized_r=0.75,
        recency_weighting="none",
        recency_max_weight=2.0,
        recency_half_life_days=365.0,
        mtf_timeframes="",
        label_mode="horizon",
        label_commission_per_contract=2.0,
        label_slippage_ticks=1.0,
        label_max_hold_bars=24,
        live_shadow_summary_path="run_health_summary.json",
        require_safe_shadow_pass=False,
        generation=None,
        parent_tag=None,
        promotion_target_tag=None,
        promotion_result=None,
        notes="strict_retrain_v6",
        calibration_method=calibration_method,
        force_drop_features=",".join(force_drop_features),
        setup_feature_top_n=int(setup_feature_top_n),
        direction_feature_top_n=int(direction_feature_top_n),
        close_feature_top_n=int(close_feature_top_n),
        drift_ks_stat_max=float(drift_ks_stat_max),
        enable_close_training=bool(close_enabled),
    )


def _run_train_with_retry(train_args: argparse.Namespace, *, use_gpu: bool, logger: StrictLogger) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], bool]:
    try:
        setup_metrics, direction_metrics, extras = train_phase2_models(train_args)
        return setup_metrics, direction_metrics, extras, use_gpu
    except Exception as train_exc:
        err_text = str(train_exc).lower()
        gpu_failure = use_gpu and any(token in err_text for token in ("cuda", "gpu", "device", "xgboosterror"))
        if not gpu_failure:
            raise
        logger.log(f"PHASE 2 WARNING: GPU training failed; retrying with CPU hist mode. reason={train_exc}")
        train_args.gpu = False
        setup_metrics, direction_metrics, extras = train_phase2_models(train_args)
        return setup_metrics, direction_metrics, extras, False

def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    csv_path = Path(args.csv).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    tag = args.candidate_label or args.tag or f"retrain_v6_strict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidate_dir = artifact_root / tag

    log_path = None if args.dry_run else (candidate_dir / "training_log.txt")
    logger = StrictLogger(log_path=log_path)

    ref_setup_features = artifact_root / args.baseline_tag / "setup.features.json"
    if not ref_setup_features.exists():
        raise StrictRunError(f"Missing reference features file: {ref_setup_features}")

    logger.log("START strict retrain orchestration")
    logger.log(f"repo_root={repo_root}")
    logger.log(f"csv={csv_path}")
    logger.log(f"baseline_tag={args.baseline_tag}, new_tag={tag}")

    if args.dry_run:
        logger.log("DRY-RUN enabled: no training or artifact mutation will be executed")

    start_ts = datetime.now(timezone.utc)
    training_duration = 0.0

    try:
        phase1 = _phase1_validate(
            csv_path=csv_path,
            ref_features_path=ref_setup_features,
            logger=logger,
            tz=args.tz,
            min_rows=args.min_rows,
            min_trading_days=args.min_trading_days,
        )

        if args.dry_run:
            logger.log("DRY-RUN complete after validation")
            return 0

        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "phase1_validation.json").write_text(json.dumps(phase1, indent=2), encoding="utf-8")
        if args.stop_after_phase == 1:
            logger.log("Stopped after Phase 1 by flag")
            return 0

        logger.log("PHASE 2: starting train_phase2_models with strict args")
        use_gpu = _resolve_gpu_mode(args.gpu_mode)
        logger.log(
            f"PHASE 2: gpu_mode={args.gpu_mode}, use_gpu={use_gpu}; "
            f"tree_method=hist, device={'cuda' if use_gpu else 'cpu'}"
        )

        force_drop_features = list(DEFAULT_DRIFT_DROP_FEATURES)
        prior_drift_path = artifact_root / "retrain_v6_strict_full_20260424_rerun" / "feature_drift_report.json"
        if prior_drift_path.exists():
            try:
                prior_rows = json.loads(prior_drift_path.read_text(encoding="utf-8"))
                dynamic_drop = [str(r.get("feature")) for r in prior_rows if bool(r.get("drifted")) and str(r.get("feature"))]
                if dynamic_drop:
                    force_drop_features = sorted(set(dynamic_drop))
                logger.log(f"PHASE 2: force_drop_features={len(force_drop_features)} from prior drift report")
            except Exception as exc:
                logger.log(f"PHASE 2 WARNING: failed to parse prior drift report ({exc}); using defaults")

        base_hparams: Dict[str, float] = {
            "n_estimators": 1800,
            "learning_rate": 0.05,
            "max_depth": 6,
            "direction_max_depth": 6,
            "close_max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5.0,
            "direction_min_child_weight": 5.0,
            "close_min_child_weight": 5.0,
            "early_stopping_rounds": 100,
            "direction_early_stopping_rounds": 100,
            "close_early_stopping_rounds": 80,
        }
        chosen_horizon = 8
        chosen_label_threshold = 0.0015

        sweeps_dir = candidate_dir / "sweeps"
        sweeps_dir.mkdir(parents=True, exist_ok=True)

        if args.run_label_sweep:
            horizons = _parse_int_csv(args.label_sweep_horizons) or [2, 4, 8, 12]
            atr_mults = _parse_float_csv(args.label_sweep_atr_mults) or [0.5, 1.0, 1.5, 2.0]
            label_rows: List[Dict[str, Any]] = []
            logger.log(f"PHASE 2.1 label sweep starting: horizons={horizons}, atr_mults={atr_mults}")
            for horizon, atr_mult in itertools.product(horizons, atr_mults):
                threshold = 0.0015 * float(atr_mult)
                run_dir = sweeps_dir / f"label_h{int(horizon)}_atr{atr_mult:.1f}"
                run_dir.mkdir(parents=True, exist_ok=True)
                sweep_hparams = dict(base_hparams)
                sweep_hparams["n_estimators"] = 1000
                train_args = _build_train_args(
                    csv_path=csv_path,
                    candidate_dir=run_dir,
                    tz=args.tz,
                    use_gpu=use_gpu,
                    horizon=int(horizon),
                    label_threshold=float(threshold),
                    hparams=sweep_hparams,
                    close_enabled=False,
                    force_drop_features=force_drop_features,
                    setup_feature_top_n=args.setup_feature_top_n,
                    direction_feature_top_n=args.direction_feature_top_n,
                    close_feature_top_n=args.close_feature_top_n,
                    calibration_method=args.calibration_method,
                    drift_ks_stat_max=args.drift_ks_stat_max,
                )
                try:
                    setup_m, direction_m, _, use_gpu = _run_train_with_retry(train_args, use_gpu=use_gpu, logger=logger)
                    direction_logloss = _safe_float(direction_m.get("log_loss_val"), default=1.0)
                    label_rows.append(
                        {
                            "horizon": int(horizon),
                            "atr_mult": float(atr_mult),
                            "label_threshold": float(threshold),
                            "direction_logloss_val": direction_logloss,
                            "setup_logloss_val": _safe_float(setup_m.get("log_loss_val"), default=1.0),
                        }
                    )
                except Exception as exc:
                    label_rows.append(
                        {
                            "horizon": int(horizon),
                            "atr_mult": float(atr_mult),
                            "label_threshold": float(threshold),
                            "direction_logloss_val": 9.99,
                            "error": str(exc),
                        }
                    )
            label_rows = sorted(label_rows, key=lambda r: _safe_float(r.get("direction_logloss_val"), default=9.99))
            (candidate_dir / "label_sweep_results.json").write_text(json.dumps(label_rows, indent=2), encoding="utf-8")
            best_label = _select_successful_sweep_row(
                label_rows,
                metric="direction_logloss_val",
                phase_name="PHASE 2.1 label sweep",
            )
            chosen_horizon = int(best_label["horizon"])
            chosen_label_threshold = float(best_label["label_threshold"])
            logger.log(f"PHASE 2.1 selected label params={best_label}")

        selected_hparams = dict(base_hparams)
        if args.run_hyper_sweep:
            grid = list(
                itertools.product(
                    [4, 6, 8, 10],
                    [0.01, 0.05, 0.1],
                    [0.6, 0.8, 1.0],
                    [0.6, 0.8, 1.0],
                    [1, 3, 5, 10],
                )
            )
            max_trials = max(1, int(args.max_hyper_trials))
            if max_trials < len(grid):
                random.seed(42)
                grid = random.sample(grid, max_trials)
            hyper_rows: List[Dict[str, Any]] = []
            logger.log(f"PHASE 2.2 hyper sweep starting: trials={len(grid)}")
            for idx, (max_depth, learning_rate, subsample, colsample, min_child_weight) in enumerate(grid, start=1):
                run_dir = sweeps_dir / f"hyper_{idx:03d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                trial_hparams = dict(base_hparams)
                trial_hparams.update(
                    {
                        "max_depth": int(max_depth),
                        "direction_max_depth": int(max_depth),
                        "learning_rate": float(learning_rate),
                        "subsample": float(subsample),
                        "colsample_bytree": float(colsample),
                        "min_child_weight": float(min_child_weight),
                        "direction_min_child_weight": float(min_child_weight),
                    }
                )
                train_args = _build_train_args(
                    csv_path=csv_path,
                    candidate_dir=run_dir,
                    tz=args.tz,
                    use_gpu=use_gpu,
                    horizon=chosen_horizon,
                    label_threshold=chosen_label_threshold,
                    hparams=trial_hparams,
                    close_enabled=False,
                    force_drop_features=force_drop_features,
                    setup_feature_top_n=args.setup_feature_top_n,
                    direction_feature_top_n=args.direction_feature_top_n,
                    close_feature_top_n=args.close_feature_top_n,
                    calibration_method=args.calibration_method,
                    drift_ks_stat_max=args.drift_ks_stat_max,
                )
                try:
                    _, direction_m, _, use_gpu = _run_train_with_retry(train_args, use_gpu=use_gpu, logger=logger)
                    direction_logloss = _safe_float(direction_m.get("log_loss_val"), default=1.0)
                    hyper_rows.append(
                        {
                            "max_depth": int(max_depth),
                            "learning_rate": float(learning_rate),
                            "subsample": float(subsample),
                            "colsample_bytree": float(colsample),
                            "min_child_weight": float(min_child_weight),
                            "direction_logloss_val": direction_logloss,
                        }
                    )
                except Exception as exc:
                    hyper_rows.append(
                        {
                            "max_depth": int(max_depth),
                            "learning_rate": float(learning_rate),
                            "subsample": float(subsample),
                            "colsample_bytree": float(colsample),
                            "min_child_weight": float(min_child_weight),
                            "direction_logloss_val": 9.99,
                            "error": str(exc),
                        }
                    )
            hyper_rows = sorted(hyper_rows, key=lambda r: _safe_float(r.get("direction_logloss_val"), default=9.99))
            (candidate_dir / "hyper_sweep_results.json").write_text(json.dumps(hyper_rows, indent=2), encoding="utf-8")
            best_hp = _select_successful_sweep_row(
                hyper_rows,
                metric="direction_logloss_val",
                phase_name="PHASE 2.2 hyper sweep",
            )
            selected_hparams.update(
                {
                    "max_depth": int(best_hp["max_depth"]),
                    "direction_max_depth": int(best_hp["max_depth"]),
                    "learning_rate": float(best_hp["learning_rate"]),
                    "subsample": float(best_hp["subsample"]),
                    "colsample_bytree": float(best_hp["colsample_bytree"]),
                    "min_child_weight": float(best_hp["min_child_weight"]),
                    "direction_min_child_weight": float(best_hp["min_child_weight"]),
                }
            )
            logger.log(f"PHASE 2.2 selected hyperparams={best_hp}")

        t0 = datetime.now(timezone.utc)
        train_args = _build_train_args(
            csv_path=csv_path,
            candidate_dir=candidate_dir,
            tz=args.tz,
            use_gpu=use_gpu,
            horizon=chosen_horizon,
            label_threshold=chosen_label_threshold,
            hparams=selected_hparams,
            close_enabled=True,
            force_drop_features=force_drop_features,
            setup_feature_top_n=args.setup_feature_top_n,
            direction_feature_top_n=args.direction_feature_top_n,
            close_feature_top_n=args.close_feature_top_n,
            calibration_method=args.calibration_method,
            drift_ks_stat_max=args.drift_ks_stat_max,
        )
        setup_metrics, direction_metrics, extras, use_gpu = _run_train_with_retry(train_args, use_gpu=use_gpu, logger=logger)
        training_duration = (datetime.now(timezone.utc) - t0).total_seconds()
        (candidate_dir / "selected_training_params.json").write_text(
            json.dumps(
                {
                    "horizon": chosen_horizon,
                    "label_threshold": chosen_label_threshold,
                    "hparams": selected_hparams,
                    "force_drop_features": force_drop_features,
                    "calibration_method": args.calibration_method,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.log(
            "PHASE 2: setup best metrics "
            f"logloss_val={_safe_float(setup_metrics.get('log_loss_val')):.6f}, "
            f"n_train={setup_metrics.get('n_train')}, n_val={setup_metrics.get('n_val')}"
        )
        logger.log(
            "PHASE 2: direction best metrics "
            f"logloss_val={_safe_float(direction_metrics.get('log_loss_val')):.6f}, "
            f"n_train={direction_metrics.get('n_train')}, n_val={direction_metrics.get('n_val')}"
        )

        _top20_feature_importance(
            candidate_dir / "setup_raw.joblib",
            candidate_dir / "setup.features.json",
            candidate_dir / "feature_importance_top20_setup.txt",
            candidate_dir / "feature_importance_top20_setup.png",
        )
        _top20_feature_importance(
            candidate_dir / "dir_raw.joblib",
            candidate_dir / "dir.features.json",
            candidate_dir / "feature_importance_top20_dir.txt",
            candidate_dir / "feature_importance_top20_dir.png",
        )
        close_enabled = bool((candidate_dir / "close.joblib").exists() and (candidate_dir / "close.features.json").exists())
        if close_enabled:
            _top20_feature_importance(
                candidate_dir / "close_raw.joblib",
                candidate_dir / "close.features.json",
                candidate_dir / "feature_importance_top20_close.txt",
                candidate_dir / "feature_importance_top20_close.png",
            )
        else:
            logger.log("PHASE 2 WARNING: close artifacts missing; strict manifest will mark close.enabled=false")
        logger.log("PHASE 2: top20 feature artifacts saved for setup/dir/close")

        gc.collect()
        if args.stop_after_phase == 2:
            logger.log("Stopped after Phase 2 by flag")
            return 0

        logger.log("PHASE 3: running strict threshold grid evaluation")
        ctx = _build_eval_context(
            csv_path,
            candidate_dir / "setup.joblib",
            candidate_dir / "dir.joblib",
            args.tz,
            horizon=chosen_horizon,
            label_threshold=chosen_label_threshold,
        )
        selected_thresholds, test_eval, grid_rows = _phase3_threshold_grid(ctx, logger)
        (candidate_dir / "threshold_grid_results.json").write_text(json.dumps(grid_rows, indent=2), encoding="utf-8")
        (candidate_dir / "threshold_selection.json").write_text(
            json.dumps({"selected": selected_thresholds, "test_eval": test_eval}, indent=2), encoding="utf-8"
        )
        _write_benchmark_ready_manifest(
            candidate_dir / "manifest.json",
            tag=tag,
            csv_path=csv_path,
            thresholds=selected_thresholds,
            close_enabled=close_enabled,
        )
        logger.log(f"PHASE 3: benchmark-ready manifest staged at {candidate_dir / 'manifest.json'}")

        if args.stop_after_phase == 3:
            logger.log("Stopped after Phase 3 by flag")
            return 0

        calibration = _calibration_errors_from_context(ctx)
        logger.log(f"PHASE 2.4 calibration errors={calibration}")

        bench_out = candidate_dir / "benchmark"
        bench_out.mkdir(parents=True, exist_ok=True)
        if args.skip_benchmark:
            raise StrictRunError("Benchmark is required in strict mode; --skip-benchmark not allowed")
        bench = _run_benchmark(
            repo_root,
            artifact_root,
            args.baseline_tag,
            tag,
            bench_out,
            logger,
            start_date=args.benchmark_start,
            end_date=args.benchmark_end,
        )
        if str(bench["new"].get("status")) != "benchmarked":
            raise StrictRunError(
                "Benchmark returned non-benchmarked status for new tag: "
                f"{bench['new'].get('status')} / {bench['new'].get('eval_error')}"
            )

        compare = _build_compare_payload(bench, args.baseline_tag, tag)
        logger.log(f"PHASE 4.2 comparison={compare['metrics']}")

        quality_pass, quality_checks = _promotion_gate_results(bench, calibration)
        if args.auto_retune_benchmark_thresholds:
            logger.log(
                "PHASE 4: --auto-retune-benchmark-thresholds ignored; "
                "benchmark-window outcomes cannot alter promotion thresholds"
            )

        (candidate_dir / "benchmark_results.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
        (candidate_dir / "quality_gates.json").write_text(json.dumps(quality_checks, indent=2), encoding="utf-8")

        baseline_feats = _read_features(candidate_dir / "setup.features.json")
        if not baseline_feats:
            baseline_feats = _read_features(ref_setup_features)
        drift_rows: List[Dict[str, Any]] = []
        feats = build_features(pd.read_csv(csv_path), tz=args.tz, rth_start="07:30", rth_end="14:00", orb_minutes=15)
        common_feats = [f for f in baseline_feats if f in feats.columns]
        for f in common_feats:
            col = pd.to_numeric(feats[f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if col.empty:
                continue
            ref = col.iloc[: max(100, int(len(col) * 0.70))].to_numpy()
            cur = col.iloc[-max(100, int(len(col) * 0.15)) :].to_numpy()
            psi = _compute_psi(ref, cur)
            try:
                from scipy.stats import ks_2samp  # type: ignore
                ks_stat, ks_p = ks_2samp(ref, cur, alternative="two-sided", mode="auto")
                ks_stat = float(ks_stat)
                ks_p = float(ks_p)
            except Exception:
                ks_stat = 0.0
                ks_p = None
            drift_rows.append({"feature": f, "psi": psi, "ks_stat": ks_stat, "ks_pvalue": ks_p, "drifted": bool(psi > 0.2 or ks_stat > 0.2)})
        drifted = [r for r in drift_rows if r["drifted"]]
        (candidate_dir / "feature_drift_report.json").write_text(json.dumps(drift_rows, indent=2), encoding="utf-8")
        logger.log(f"PHASE 6.1 drifted_features={len(drifted)}")

        tuned_setup = float(selected_thresholds["p_setup"])
        sensitivity = []
        for test_t in [max(0.0, tuned_setup - 0.05), tuned_setup, min(1.0, tuned_setup + 0.05)]:
            tdict = {"p_setup": test_t, "p_long": selected_thresholds["p_long"], "p_short": selected_thresholds["p_short"]}
            df_test = phase2_decisions(
                ctx.test_frame.copy(),
                ctx.setup_probs.reindex(ctx.test_frame.index).to_numpy(dtype=float),
                ctx.dir_probs.reindex(ctx.test_frame.index).to_numpy(dtype=float),
                tdict,
                policy=ctx.policy,
            )
            sim = simulate_trades(df_test, tdict, cfg=ctx.sim_cfg)
            sensitivity.append(
                {"p_setup": test_t, "trade_count": int(sim.get("trade_count") or len(sim.get("trades") or [])), "pnl": _safe_float(sim.get("total_pnl_usd"))}
            )
        (candidate_dir / "threshold_sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
        logger.log(f"PHASE 6.2 sensitivity={sensitivity}")

        n_test = len(ctx.test_frame)
        mid = n_test // 2
        halves = [(0, mid, "first_half"), (mid, n_test, "second_half")]
        regime = []
        thr = {"p_setup": selected_thresholds["p_setup"], "p_long": selected_thresholds["p_long"], "p_short": selected_thresholds["p_short"]}
        for a, b, name in halves:
            part = ctx.test_frame.iloc[a:b].copy()
            df_part = phase2_decisions(
                part,
                ctx.setup_probs.reindex(part.index).to_numpy(dtype=float),
                ctx.dir_probs.reindex(part.index).to_numpy(dtype=float),
                thr,
                policy=ctx.policy,
            )
            sim = simulate_trades(df_part, thr, cfg=ctx.sim_cfg)
            regime.append(
                {
                    "segment": name,
                    "trade_count": int(sim.get("trade_count") or len(sim.get("trades") or [])),
                    "pnl": _safe_float(sim.get("total_pnl_usd")),
                    "sharpe": _safe_float(sim.get("sharpe")),
                }
            )
        (candidate_dir / "regime_robustness.json").write_text(json.dumps(regime, indent=2), encoding="utf-8")
        logger.log(f"PHASE 6.3 regime={regime}")

        strict_manifest = _write_strict_manifest(
            candidate_dir / "manifest.json",
            tag,
            csv_path,
            training_duration,
            selected_thresholds,
            bench["new"],
            close_enabled,
            quality_pass,
        )
        logger.log(f"PHASE 5.2 strict manifest written: {candidate_dir / 'manifest.json'}")

        failed_checks = [c for c in quality_checks if not c["pass"]]
        if failed_checks:
            hist_payload = {
                "setup": np.histogram(ctx.setup_probs.reindex(ctx.test_frame.index).to_numpy(dtype=float), bins=20)[0].tolist(),
                "direction": np.histogram(ctx.dir_probs.reindex(ctx.test_frame.index).to_numpy(dtype=float), bins=20)[0].tolist(),
            }
            (candidate_dir / "probability_histograms.json").write_text(json.dumps(hist_payload, indent=2), encoding="utf-8")

            y_setup_true = ctx.setup_val_labels.reindex(ctx.val_frame.index).fillna(0).astype(int).to_numpy()
            y_setup_pred = (ctx.setup_probs.reindex(ctx.val_frame.index).to_numpy(dtype=float) >= selected_thresholds["p_setup"]).astype(int)
            y_dir_true = ctx.dir_val_labels.reindex(ctx.dir_val_probs.index).fillna(0).astype(int).to_numpy()
            y_dir_pred = (ctx.dir_val_probs.to_numpy(dtype=float) >= selected_thresholds["p_long"]).astype(int)
            cm_payload = {
                "setup": confusion_matrix(y_setup_true, y_setup_pred).tolist(),
                "direction": confusion_matrix(y_dir_true, y_dir_pred).tolist(),
            }
            (candidate_dir / "confusion_matrices.json").write_text(json.dumps(cm_payload, indent=2), encoding="utf-8")
            logger.log(f"QUALITY GATES FAILED: {failed_checks}")
            logger.log(
                "DIAGNOSTICS: "
                f"{candidate_dir / 'feature_drift_report.json'}, "
                f"{candidate_dir / 'probability_histograms.json'}, "
                f"{candidate_dir / 'confusion_matrices.json'}"
            )
            raise StrictRunError("Quality gates failed; stopping before champion update")

        if args.promote:
            champion_path = artifact_root / "current_champion.json"
            champion = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "tag": tag,
                "generation": 0,
                "baseline_tag": args.baseline_tag,
                "deployment_baseline_tag": tag,
                "search_bias": None,
            }
            champion_path.write_text(json.dumps(champion, indent=2), encoding="utf-8")
            logger.log(f"PHASE 5.3 champion updated: {champion_path}")

        elapsed = (datetime.now(timezone.utc) - start_ts).total_seconds()
        logger.log(f"DONE: all strict phases completed in {elapsed:.1f}s, tag={tag}")
        logger.log(f"SUMMARY manifest={strict_manifest}")
        return 0

    except Exception as exc:
        logger.log(f"ERROR: {exc}")
        logger.log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
