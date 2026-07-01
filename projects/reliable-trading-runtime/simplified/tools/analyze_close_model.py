from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _print_counter(title: str, values: Counter[str]) -> None:
    print(title)
    total = sum(values.values())
    if total <= 0:
        print("  none")
        return
    for key, count in values.most_common():
        pct = 100.0 * float(count) / float(total)
        print(f"  {key}: {count} ({pct:.1f}%)")


def _profit_bucket(value: Any) -> str:
    try:
        num = float(value)
    except Exception:
        return "unknown"
    if num > 0:
        return "profit"
    if num < 0:
        return "loss"
    return "flat"


def _trend_bucket(score: Any) -> str:
    try:
        val = float(score)
    except Exception:
        return "unknown"
    if val > 0.55:
        return "up_gt_0.55"
    if val < -0.55:
        return "down_lt_-0.55"
    return "neutral"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize close-model behavior, thresholds, and exit mixes.")
    parser.add_argument("--manifest", required=True, help="Path to phase2 manifest.json")
    parser.add_argument("--events", default=None, help="Optional path to events.jsonl with phase2_close_* events")
    parser.add_argument("--trades", default=None, help="Optional path to trades.csv")
    parser.add_argument("--close-preds", default=None, help="Optional path to CSV with close_prob and label columns")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = _load_json(manifest_path)
    print(f"manifest={manifest_path}")
    print(f"close_threshold={float(((manifest.get('close') or {}).get('threshold')) or ((manifest.get('config') or {}).get('close_threshold')) or 0.0):.2f}")

    metrics_path = manifest_path.with_name("close.metrics.json")
    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        print(
            "metrics:"
            f" roc_auc_test={float(metrics.get('roc_auc_test', 0.0)):.6f}"
            f" avg_precision_test={float(metrics.get('avg_precision_test', 0.0)):.6f}"
            f" n_test={int(metrics.get('n_test', 0) or 0)}"
        )

    if args.trades:
        trades_path = Path(args.trades).expanduser().resolve()
        trades = pd.read_csv(trades_path)
        if "exit_reason" in trades.columns:
            _print_counter("exit_reason_mix:", Counter(trades["exit_reason"].fillna("missing").astype(str)))
        elif "reason" in trades.columns:
            _print_counter("exit_reason_mix:", Counter(trades["reason"].fillna("missing").astype(str)))
        else:
            print("exit_reason_mix:\n  unavailable")

    if args.events:
        events_path = Path(args.events).expanduser().resolve()
        events = _load_jsonl(events_path)
        close_probs: list[float] = []
        close_reasons: Counter[str] = Counter()
        close_profit_state: Counter[str] = Counter()
        close_trend_state: Counter[str] = Counter()
        suppressed_reasons: Counter[str] = Counter()
        for row in events:
            event = str(row.get("event") or "")
            if event == "phase2_close_model_action":
                prob = (row.get("phase2_close_prob"))
                try:
                    close_probs.append(float(prob))
                except Exception:
                    pass
                state = row.get("trade_pnl_state") or {}
                close_reasons["model_close"] += 1
                close_profit_state[_profit_bucket(state.get("unrealized_r"))] += 1
                close_trend_state[_trend_bucket(state.get("trend_score", row.get("trend_score")))] += 1
            elif event == "pnl_overlay_action":
                close_reasons[str(row.get("reason") or "overlay_unknown")] += 1
            elif event == "phase2_close_suppressed":
                suppressed_reasons[str(row.get("reason") or "suppressed_unknown")] += 1
        if close_probs:
            series = pd.Series(close_probs, dtype=float)
            print(
                "close_prob_distribution:"
                f" count={len(series)} min={series.min():.4f} p50={series.quantile(0.5):.4f}"
                f" p90={series.quantile(0.9):.4f} max={series.max():.4f}"
            )
        else:
            print("close_prob_distribution: unavailable")
        _print_counter("close_reason_mix:", close_reasons)
        _print_counter("model_close_profit_state:", close_profit_state)
        _print_counter("model_close_trend_regime:", close_trend_state)
        _print_counter("close_suppressed_reasons:", suppressed_reasons)

    if args.close_preds:
        preds_path = Path(args.close_preds).expanduser().resolve()
        preds = pd.read_csv(preds_path)
        if {"close_prob", "label"}.issubset(preds.columns):
            bins = pd.cut(preds["close_prob"], bins=[0.0, 0.5, 0.7, 0.85, 0.95, 1.0], include_lowest=True)
            grouped = preds.groupby(bins, observed=False)["label"].agg(["count", "mean"])
            print("calibration_bins:")
            for idx, row in grouped.iterrows():
                print(f"  {idx}: count={int(row['count'])} empirical_rate={float(row['mean']):.4f}")
        else:
            print("calibration_bins:\n  unavailable (requires close_prob and label columns)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
