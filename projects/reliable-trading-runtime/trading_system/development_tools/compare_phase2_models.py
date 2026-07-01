from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.runtime_engine.modeling.config import ENGINE, instrument_by_alias
from trading_system.runtime_engine.modeling.features import build_features
from trading_system.runtime_engine.modeling.phase2_sim import Phase2SimConfig, phase2_decisions, simulate_trades
from trading_system.runtime_engine.modeling.train import _apply_event_filter
from trading_system.runtime_engine.modeling.train_phase2 import _build_phase2_labels, _phase2_label_domain, _predict_long_probabilities
from trading_system.runtime_engine.runtime_config.loader import load_app_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Phase-2 models on a fixed test window.")
    p.add_argument("--csv", required=True, help="OHLCV CSV path.")
    p.add_argument("--baseline-tag", required=True, help="Baseline Phase-2 manifest tag.")
    p.add_argument("--v2-tag", required=True, help="Retrain v2 manifest tag.")
    p.add_argument("--v3-tag", required=True, help="Retrain v3 manifest tag.")
    p.add_argument("--test-start", required=True, help="Test window start (YYYY-MM-DD or timestamp).")
    p.add_argument("--test-end", required=True, help="Test window end (YYYY-MM-DD or timestamp).")
    p.add_argument("--out", default="TRAINING_COMPARISON_REPORT.md", help="Output report path.")
    return p.parse_args()


def _load_manifest(tag: str) -> Dict[str, Any]:
    path = ROOT / "artifacts" / "phase2" / "candidates" / tag / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found for tag '{tag}': {path}")
    payload = json.loads(path.read_text())
    payload["_manifest_dir"] = str(path.parent.resolve())
    return payload


def _manifest_path(manifest: Dict[str, Any], key: str) -> Path:
    raw = manifest.get(key)
    if not raw:
        raise FileNotFoundError(f"Manifest missing {key}")
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    manifest_dir = Path(str(manifest.get("_manifest_dir") or "."))
    candidate = (manifest_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (ROOT / path).resolve()


def _parse_time_bound(value: str, tz: str, *, is_end: bool) -> Tuple[pd.Timestamp, bool]:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid datetime bound: {value}")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(tz)
    else:
        parsed = parsed.tz_convert(tz)
    has_clock = ":" in str(value)
    if is_end and not has_clock:
        return parsed + pd.Timedelta(days=1), True
    return parsed, False


def _time_window_mask(dt_series: pd.Series, start: str, end: str, tz: str) -> pd.Series:
    start_ts, _ = _parse_time_bound(start, tz, is_end=False)
    end_ts, end_exclusive = _parse_time_bound(end, tz, is_end=True)
    mask = (dt_series >= start_ts)
    if end_exclusive:
        mask &= dt_series < end_ts
    else:
        mask &= dt_series <= end_ts
    return mask


def _metric_summary(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def _max_consecutive_losses(trades: list[Dict[str, Any]]) -> int:
    max_streak = 0
    streak = 0
    for trade in trades:
        pnl = float(trade.get("pnl_usd") or 0.0)
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _monthly_stability(trades: list[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for trade in trades:
        entry_ts = trade.get("entry_ts")
        ts = pd.to_datetime(entry_ts, errors="coerce")
        if pd.isna(ts):
            continue
        key = ts.strftime("%Y-%m")
        bucket = buckets.setdefault(key, {"trades": 0, "wins": 0, "pnl": 0.0})
        bucket["trades"] += 1
        pnl = float(trade.get("pnl_usd") or 0.0)
        bucket["pnl"] += pnl
        if pnl > 0:
            bucket["wins"] += 1
    summary: Dict[str, Dict[str, float]] = {}
    for key, bucket in sorted(buckets.items()):
        trades = max(bucket["trades"], 1)
        summary[key] = {
            "trades": float(bucket["trades"]),
            "win_rate": float(bucket["wins"]) / trades,
            "pnl_usd": float(bucket["pnl"]),
        }
    return summary


def _evaluate_model(tag: str, csv_path: str, test_start: str, test_end: str) -> Dict[str, Any]:
    manifest = _load_manifest(tag)
    cfg = manifest.get("config", {})
    tz = str(cfg.get("tz") or "America/Denver")
    runtime_labels = load_app_config().labels
    df_raw = pd.read_csv(csv_path)
    feats = build_features(
        df_raw,
        tz=tz,
        rth_start=str(cfg.get("rth_start") or ENGINE.rth_start),
        rth_end=str(cfg.get("rth_end") or ENGINE.rth_end),
        orb_minutes=int(cfg.get("orb_minutes") or 15),
        csv_naive_is_utc=bool(cfg.get("csv_naive_is_utc", False)),
    )
    feats = _apply_event_filter(feats, "all")
    scheme = _phase2_label_domain(runtime_labels, bool(cfg.get("htf_trend_aware", False)))
    labels = _build_phase2_labels(
        feats,
        horizon=int(cfg.get("horizon") or runtime_labels.horizon_bars),
        threshold=float(cfg.get("label_threshold") or 0.0015),
        use_log=bool(cfg.get("label_log", False)),
        scheme=scheme,
        mode=str(cfg.get("label_mode") or "horizon"),
        instrument=str(manifest.get("instrument") or "ES"),
        max_hold_bars=int(cfg.get("label_max_hold_bars") or cfg.get("horizon") or runtime_labels.horizon_bars),
        commission_per_contract=float(cfg.get("label_commission_per_contract") or 2.0),
        slippage_ticks=float(cfg.get("label_slippage_ticks") or 1.0),
        setup_threshold_multiplier=float(cfg.get("setup_threshold_multiplier") or 1.75),
    )
    trimmed = labels.trimmed_frame
    y_setup = labels.setup
    y_direction = labels.direction
    dt_series = pd.to_datetime(trimmed["Datetime"], errors="coerce")
    if getattr(dt_series.dt, "tz", None) is None:
        dt_series = dt_series.dt.tz_localize(tz)
    else:
        dt_series = dt_series.dt.tz_convert(tz)
    test_mask = _time_window_mask(dt_series, test_start, test_end, tz)
    if not test_mask.any():
        raise ValueError(f"No rows in test window for tag {tag}.")

    setup_path = _manifest_path(manifest, "setup_model_path")
    dir_path = _manifest_path(manifest, "dir_model_path")
    setup_probs = _predict_long_probabilities(setup_path, trimmed)
    setup_prob_series = pd.Series(setup_probs, index=trimmed.index)

    dir_frame_full = trimmed.copy()
    if bool(cfg.get("stack_setup_prob", False)):
        dir_frame_full["stack_setup_prob"] = setup_prob_series.reindex(dir_frame_full.index).to_numpy()
    dir_probs_full = _predict_long_probabilities(dir_path, dir_frame_full)
    dir_prob_series_full = pd.Series(dir_probs_full, index=dir_frame_full.index)

    direction_features = trimmed.loc[y_direction.index]
    if bool(cfg.get("stack_setup_prob", False)):
        direction_features = direction_features.copy()
        direction_features["stack_setup_prob"] = setup_prob_series.reindex(direction_features.index).to_numpy()
    dir_probs_label = _predict_long_probabilities(dir_path, direction_features)

    thresholds = manifest.get("thresholds") or {}
    p_setup = float(thresholds.get("p_setup") or 0.6)
    p_long = float(thresholds.get("p_long") or 0.6)
    p_short = float(thresholds.get("p_short") or 0.6)

    test_idx = trimmed.index[test_mask]
    setup_test_probs = setup_prob_series.loc[test_idx].to_numpy(dtype=float)
    dir_test_probs = dir_prob_series_full.loc[test_idx].to_numpy(dtype=float)
    test_frame = trimmed.loc[test_idx].copy()

    setup_labels_test = y_setup.reindex(test_idx)
    setup_preds = setup_test_probs >= p_setup
    precision = float((setup_preds & (setup_labels_test == 1)).sum() / max(setup_preds.sum(), 1))
    recall = float((setup_preds & (setup_labels_test == 1)).sum() / max((setup_labels_test == 1).sum(), 1))

    sim_cfg = Phase2SimConfig(
        tz=tz,
        trade_window_start=str(cfg.get("trade_window_start") or ENGINE.trade_window_start),
        trade_window_end=str(cfg.get("trade_window_end") or ENGINE.trade_window_end),
        point_value=instrument_by_alias(manifest.get("instrument") or "ES").point_value,
        tick_value=instrument_by_alias(manifest.get("instrument") or "ES").tick_value,
        contracts=1,
        max_hold_bars=int(cfg.get("label_max_hold_bars") or cfg.get("horizon") or runtime_labels.horizon_bars),
        commission_per_contract=float(cfg.get("label_commission_per_contract") or 2.0),
        slippage_ticks=float(cfg.get("label_slippage_ticks") or 1.0),
    )
    decisions = phase2_decisions(test_frame, setup_test_probs, dir_test_probs, thresholds)
    sim = simulate_trades(decisions, thresholds, cfg=sim_cfg)

    trades = sim.get("trades") or []
    max_loss_streak = _max_consecutive_losses(trades)
    monthly = _monthly_stability(trades)

    direction_probs = np.asarray(dir_probs_label, dtype=float)
    p_sell = 1.0 - direction_probs
    return {
        "tag": tag,
        "thresholds": {"p_setup": p_setup, "p_long": p_long, "p_short": p_short},
        "trade_frequency": float((decisions["phase2_direction_signal"] != 0).mean()),
        "setup_precision": precision,
        "setup_recall": recall,
        "setup_prob_stats": _metric_summary(setup_test_probs),
        "dir_prob_stats": _metric_summary(direction_probs),
        "p_sell_stats": _metric_summary(p_sell),
        "max_consecutive_losses": max_loss_streak,
        "sim": {
            "profit_factor": float(sim.get("profit_factor") or 0.0),
            "max_drawdown": float(sim.get("max_drawdown") or 0.0),
            "avg_trade_usd": float(sim.get("avg_trade_usd") or 0.0),
            "trades": int(sim.get("trade_count") or len(trades)),
            "trades_per_day": float(sim.get("trades_per_day") or 0.0),
        },
        "monthly_stability": monthly,
    }


def _write_report(out_path: Path, baseline: Dict[str, Any], v2: Dict[str, Any], v3: Dict[str, Any], test_start: str, test_end: str) -> None:
    rows = [baseline, v2, v3]
    lines: list[str] = []
    lines.append("# TRAINING_COMPARISON_REPORT")
    lines.append("")
    lines.append(f"- Test window: {test_start} -> {test_end}")
    lines.append("")
    lines.append("## Summary")
    for row in rows:
        sim = row["sim"]
        lines.append(
            f"- {row['tag']}: trades={sim['trades']} trades/day={sim['trades_per_day']:.2f} "
            f"pf={sim['profit_factor']:.2f} max_dd={sim['max_drawdown']:.2f} "
            f"avg_trade_usd={sim['avg_trade_usd']:.2f} max_loss_streak={row['max_consecutive_losses']}"
        )
    lines.append("")
    lines.append("## Details")
    for row in rows:
        lines.append(f"### {row['tag']}")
        lines.append(f"- thresholds: {row['thresholds']}")
        lines.append(f"- trade_frequency: {row['trade_frequency']:.4f}")
        lines.append(f"- setup_precision: {row['setup_precision']:.4f}")
        lines.append(f"- setup_recall: {row['setup_recall']:.4f}")
        lines.append(f"- setup_prob_stats: {row['setup_prob_stats']}")
        lines.append(f"- dir_prob_stats: {row['dir_prob_stats']}")
        lines.append(f"- p_sell_stats: {row['p_sell_stats']}")
        lines.append(f"- monthly_stability: {row['monthly_stability']}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    baseline = _evaluate_model(args.baseline_tag, args.csv, args.test_start, args.test_end)
    v2 = _evaluate_model(args.v2_tag, args.csv, args.test_start, args.test_end)
    v3 = _evaluate_model(args.v3_tag, args.csv, args.test_start, args.test_end)
    out_path = Path(args.out).expanduser()
    _write_report(out_path, baseline, v2, v3, args.test_start, args.test_end)
    print(json.dumps({"out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
