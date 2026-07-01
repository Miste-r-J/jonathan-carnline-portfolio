from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from na.bot.config import instrument_by_alias
from na.bot.phase2_sim import Phase2DecisionPolicy, Phase2SimConfig, phase2_decisions, simulate_trades


def values(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def daily_metrics(trades: list[dict[str, Any]], limit: float) -> dict[str, Any]:
    daily: dict[str, float] = {}
    for trade in trades:
        day = str(trade.get("entry_ts") or "")[:10]
        daily[day] = daily.get(day, 0.0) + float(trade.get("pnl_usd") or 0.0)
    return {
        "worst_day_usd": min(daily.values()) if daily else 0.0,
        "daily_loss_breaches": sum(pnl <= -abs(limit) for pnl in daily.values()),
        "positive_day_rate": sum(pnl > 0 for pnl in daily.values()) / max(1, len(daily)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Phase-2 thresholds from one cached probability frame.")
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--p-setup", default="0.10,0.15,0.20,0.25,0.30,0.35,0.45,0.60")
    parser.add_argument("--p-long", default="0.58,0.62,0.66")
    parser.add_argument("--p-short", default="0.65,0.70,0.75")
    parser.add_argument("--instrument", default="ES")
    parser.add_argument("--daily-loss-limit", type=float, default=500.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.eval_csv)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="coerce", utc=True)
    setup = frame["phase2_setup_prob"].astype(float).to_numpy()
    direction = frame["dir_prob_raw"].astype(float).to_numpy()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = manifest.get("config") or {}
    policy = Phase2DecisionPolicy.from_mapping(config.get("decision_policy") or {})
    instrument = instrument_by_alias(args.instrument)
    sim_config = Phase2SimConfig(
        tz=config.get("tz", "America/Denver"),
        trade_window_start="00:00",
        trade_window_end="23:59",
        point_value=instrument.point_value,
        tick_value=instrument.tick_value,
        contracts=1,
        max_hold_bars=24,
        commission_per_contract=2.0,
        slippage_ticks=1.0,
    )

    results = []
    for p_setup, p_long, p_short in itertools.product(values(args.p_setup), values(args.p_long), values(args.p_short)):
        thresholds = {"p_setup": p_setup, "p_long": p_long, "p_short": p_short}
        decided = phase2_decisions(frame.copy(), setup, direction, thresholds, policy=policy)
        sim = simulate_trades(decided, thresholds, cfg=sim_config)
        trades = list(sim.get("trades") or [])
        daily = daily_metrics(trades, args.daily_loss_limit)
        net = float(sim.get("total_pnl_usd") or 0.0)
        drawdown = float(sim.get("max_drawdown") or 0.0)
        profit_factor = float(sim.get("profit_factor") or 0.0)
        trade_count = int(sim.get("trade_count") or len(trades))
        deployable = (
            net > 0 and drawdown <= 2000 and profit_factor >= 1.5 and trade_count >= 75
            and daily["daily_loss_breaches"] == 0
        )
        score = (net / max(drawdown, 1.0)) * min(profit_factor, 4.0) * min(trade_count / 150.0, 1.0)
        results.append({
            **thresholds,
            "trades": trade_count,
            "net_pnl_usd": net,
            "profit_factor": profit_factor,
            "max_drawdown_usd": drawdown,
            "sharpe": float(sim.get("sharpe") or 0.0),
            **daily,
            "deployable_validation": deployable,
            "selection_score": score,
        })
    results.sort(key=lambda row: (row["deployable_validation"], row["selection_score"]), reverse=True)
    payload = {
        "schema_version": 1,
        "selection_period": {"source": str(args.eval_csv.resolve()), "purpose": "validation_only"},
        "constraints": {"max_drawdown_usd": 2000, "daily_loss_limit_usd": args.daily_loss_limit, "min_profit_factor": 1.5, "min_trades": 75},
        "selected": results[0] if results else None,
        "deployable_count": sum(bool(row["deployable_validation"]) for row in results),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": payload["selected"], "deployable_count": payload["deployable_count"], "tested": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
