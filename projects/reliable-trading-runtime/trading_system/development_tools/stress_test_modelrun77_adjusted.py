from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_modelrun77_live_month import (
    _apply_risk_rails,
    _load_frame,
    _max_drawdown,
    _metrics,
    _simulate,
)
from trading_system.runtime_engine.modeling.config import instrument_by_alias
from trading_system.runtime_engine.modeling.phase2_sim import Phase2DecisionPolicy


def _bootstrap_drawdown(pnls: list[float], *, samples: int = 10_000) -> dict[str, float]:
    if not pnls:
        return {"samples": samples, "p95_drawdown_usd": 0.0, "prob_dd_ge_2000": 0.0}
    rng = np.random.default_rng(77)
    drawdowns = np.empty(samples, dtype=float)
    values = np.asarray(pnls, dtype=float)
    for idx in range(samples):
        drawdowns[idx] = _max_drawdown(rng.choice(values, size=len(values), replace=True))
    return {
        "samples": samples,
        "p95_drawdown_usd": float(np.quantile(drawdowns, 0.95)),
        "prob_dd_ge_2000": float(np.mean(drawdowns >= 2000.0)),
    }


def _evaluate(
    frames: list[tuple[str, pd.DataFrame]],
    *,
    thresholds: dict[str, float],
    max_hold_bars: int,
    slippage_ticks: float,
    max_risk: float,
    daily_stop: float,
) -> tuple[dict, list[dict]]:
    instrument = instrument_by_alias("ES")
    trades = _simulate(
        frames,
        thresholds=thresholds,
        policy=Phase2DecisionPolicy(),
        trade_window=("00:00", "23:59"),
        max_hold_bars=max_hold_bars,
        point_value=instrument.point_value,
        tick_value=instrument.tick_value,
        commission=2.0,
        slippage_ticks=slippage_ticks,
    )
    accepted = _apply_risk_rails(
        trades,
        max_risk_per_trade=max_risk,
        daily_loss_stop=daily_stop,
        max_trades_per_day=6,
        max_losses_per_day=2,
    )
    return _metrics(accepted), accepted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--p-setup", type=float, required=True)
    parser.add_argument("--p-long", type=float, required=True)
    parser.add_argument("--p-short", type=float, required=True)
    parser.add_argument("--max-hold-bars", type=int, required=True)
    args = parser.parse_args()

    frames = [
        (f"{path.parent.parent.name}:{path.stem}", _load_frame(path))
        for path in args.eval_csv
    ]
    thresholds = {
        "p_setup": args.p_setup,
        "p_long": args.p_long,
        "p_short": args.p_short,
    }
    scenarios = []
    for slippage in (1.0, 2.0, 4.0):
        for max_risk in (300.0, 400.0, 500.0):
            for daily_stop in (400.0, 500.0, 600.0):
                metrics, _ = _evaluate(
                    frames,
                    thresholds=thresholds,
                    max_hold_bars=args.max_hold_bars,
                    slippage_ticks=slippage,
                    max_risk=max_risk,
                    daily_stop=daily_stop,
                )
                scenarios.append(
                    {
                        "slippage_ticks_per_side": slippage,
                        "max_risk_per_trade_usd": max_risk,
                        "daily_loss_stop_usd": daily_stop,
                        **metrics,
                    }
                )

    split_results = []
    for segment, frame in frames:
        midpoint = len(frame) // 2
        for label, split in (("early", frame.iloc[:midpoint].copy()), ("late", frame.iloc[midpoint:].copy())):
            metrics, _ = _evaluate(
                [(f"{segment}:{label}", split)],
                thresholds=thresholds,
                max_hold_bars=args.max_hold_bars,
                slippage_ticks=1.0,
                max_risk=400.0,
                daily_stop=500.0,
            )
            split_results.append({"segment": segment, "split": label, **metrics})

    base_metrics, base_trades = _evaluate(
        frames,
        thresholds=thresholds,
        max_hold_bars=args.max_hold_bars,
        slippage_ticks=1.0,
        max_risk=400.0,
        daily_stop=500.0,
    )
    bootstrap = _bootstrap_drawdown([float(row["pnl_usd"]) for row in base_trades])
    payload = {
        "schema_version": 1,
        "candidate": {
            "parent_tag": "retrain_v6_fixed_v2_hyper_016",
            "thresholds": thresholds,
            "max_hold_bars": args.max_hold_bars,
            "force_open": False,
            "setup_fail_entries": False,
        },
        "base": base_metrics,
        "friction_and_risk_scenarios": scenarios,
        "chronological_splits": split_results,
        "bootstrap": bootstrap,
        "stress_summary": {
            "scenario_count": len(scenarios),
            "positive_scenarios": sum(row["net_pnl_usd"] > 0 for row in scenarios),
            "scenarios_dd_le_2000": sum(row["max_drawdown_usd"] <= 2000 for row in scenarios),
            "worst_scenario_net_usd": min(row["net_pnl_usd"] for row in scenarios),
            "worst_scenario_drawdown_usd": max(row["max_drawdown_usd"] for row in scenarios),
            "positive_chronological_splits": sum(row["net_pnl_usd"] > 0 for row in split_results),
            "chronological_split_count": len(split_results),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"base": base_metrics, "bootstrap": bootstrap, **payload["stress_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
