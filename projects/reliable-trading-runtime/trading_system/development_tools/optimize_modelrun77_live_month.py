from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.runtime_engine.modeling.config import instrument_by_alias
from trading_system.runtime_engine.modeling.phase2_sim import (
    Phase2DecisionPolicy,
    Phase2SimConfig,
    phase2_decisions,
    simulate_trades,
)


def _values(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="coerce", utc=True)
    return frame


def _max_drawdown(pnls: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for pnl in pnls:
        equity += float(pnl)
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def _apply_risk_rails(
    trades: list[dict[str, Any]],
    *,
    max_risk_per_trade: float,
    daily_loss_stop: float,
    max_trades_per_day: int,
    max_losses_per_day: int,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    state: dict[str, dict[str, float | int]] = {}
    for source in sorted(trades, key=lambda row: str(row.get("entry_ts") or "")):
        day = str(source.get("entry_ts") or "")[:10]
        daily = state.setdefault(day, {"pnl": 0.0, "trades": 0, "losses": 0})
        if (
            float(daily["pnl"]) <= -abs(daily_loss_stop)
            or int(daily["trades"]) >= max_trades_per_day
            or int(daily["losses"]) >= max_losses_per_day
        ):
            continue
        row = dict(source)
        costs = abs(float(row.get("costs_usd") or 0.0))
        raw_pnl = float(row.get("pnl_usd") or 0.0)
        # Runtime max-risk is measured before round-trip friction.
        pnl = max(raw_pnl, -(abs(max_risk_per_trade) + costs))
        row["raw_pnl_usd"] = raw_pnl
        row["pnl_usd"] = pnl
        row["net_pnl_usd"] = pnl
        accepted.append(row)
        daily["pnl"] = float(daily["pnl"]) + pnl
        daily["trades"] = int(daily["trades"]) + 1
        if pnl < 0:
            daily["losses"] = int(daily["losses"]) + 1
    return accepted


def _metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [float(row.get("pnl_usd") or 0.0) for row in trades]
    wins = sum(pnl > 0 for pnl in pnls)
    gross_win = sum(max(pnl, 0.0) for pnl in pnls)
    gross_loss = abs(sum(min(pnl, 0.0) for pnl in pnls))
    daily: dict[str, float] = {}
    contract: dict[str, float] = {}
    loss_streak = 0
    worst_loss_streak = 0
    for row, pnl in zip(trades, pnls):
        day = str(row.get("entry_ts") or "")[:10]
        daily[day] = daily.get(day, 0.0) + pnl
        segment = str(row.get("_segment") or "unknown")
        contract[segment] = contract.get(segment, 0.0) + pnl
        if pnl < 0:
            loss_streak += 1
            worst_loss_streak = max(worst_loss_streak, loss_streak)
        else:
            loss_streak = 0
    return {
        "trades": len(trades),
        "net_pnl_usd": sum(pnls),
        "profit_factor": gross_win / gross_loss if gross_loss else (999.0 if gross_win else 0.0),
        "win_rate": wins / len(pnls) if pnls else 0.0,
        "max_drawdown_usd": _max_drawdown(pnls),
        "worst_day_usd": min(daily.values()) if daily else 0.0,
        "positive_day_rate": sum(value > 0 for value in daily.values()) / max(1, len(daily)),
        "worst_loss_streak": worst_loss_streak,
        "segment_pnl": contract,
    }


def _simulate(
    frames: list[tuple[str, pd.DataFrame]],
    *,
    thresholds: dict[str, float],
    policy: Phase2DecisionPolicy,
    trade_window: tuple[str, str],
    max_hold_bars: int,
    point_value: float,
    tick_value: float,
    commission: float,
    slippage_ticks: float,
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for segment, frame in frames:
        setup = frame["phase2_setup_prob"].astype(float).to_numpy()
        direction = frame["dir_prob_raw"].astype(float).to_numpy()
        decided = phase2_decisions(frame.copy(), setup, direction, thresholds, policy=policy)
        sim = simulate_trades(
            decided,
            thresholds,
            cfg=Phase2SimConfig(
                tz="America/Denver",
                trade_window_start=trade_window[0],
                trade_window_end=trade_window[1],
                point_value=point_value,
                tick_value=tick_value,
                contracts=1,
                max_hold_bars=max_hold_bars,
                commission_per_contract=commission,
                slippage_ticks=slippage_ticks,
            ),
        )
        for trade in sim.get("trades") or []:
            row = dict(trade)
            row["_segment"] = segment
            combined.append(row)
    return combined


def _score(metrics: dict[str, Any]) -> float:
    drawdown = max(float(metrics["max_drawdown_usd"]), 1.0)
    pf = min(float(metrics["profit_factor"]), 4.0)
    coverage = min(float(metrics["trades"]) / 40.0, 1.0)
    segment_values = list((metrics.get("segment_pnl") or {}).values())
    robustness = 1.0 if segment_values and min(segment_values) > 0 else 0.35
    return float(metrics["net_pnl_usd"]) / drawdown * pf * coverage * robustness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--p-setup", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50")
    parser.add_argument("--p-long", default="0.58,0.62,0.66,0.68,0.70")
    parser.add_argument("--p-short", default="0.58,0.62,0.66,0.70,0.75")
    parser.add_argument("--max-risk-per-trade", type=float, default=400.0)
    parser.add_argument("--daily-loss-stop", type=float, default=500.0)
    parser.add_argument("--max-trades-per-day", type=int, default=6)
    parser.add_argument("--max-losses-per-day", type=int, default=2)
    parser.add_argument("--commission", type=float, default=2.0)
    parser.add_argument("--slippage-ticks", type=float, default=1.0)
    args = parser.parse_args()

    frames = [
        (f"{path.parent.parent.name}:{path.stem}", _load_frame(path))
        for path in args.eval_csv
    ]
    instrument = instrument_by_alias("ES")
    threshold_rows: list[dict[str, Any]] = []
    for p_setup, p_long, p_short in itertools.product(
        _values(args.p_setup), _values(args.p_long), _values(args.p_short)
    ):
        thresholds = {"p_setup": p_setup, "p_long": p_long, "p_short": p_short}
        trades = _simulate(
            frames,
            thresholds=thresholds,
            policy=Phase2DecisionPolicy(),
            trade_window=("00:00", "23:59"),
            max_hold_bars=20,
            point_value=instrument.point_value,
            tick_value=instrument.tick_value,
            commission=args.commission,
            slippage_ticks=args.slippage_ticks,
        )
        accepted = _apply_risk_rails(
            trades,
            max_risk_per_trade=args.max_risk_per_trade,
            daily_loss_stop=args.daily_loss_stop,
            max_trades_per_day=args.max_trades_per_day,
            max_losses_per_day=args.max_losses_per_day,
        )
        metrics = _metrics(accepted)
        row = {**thresholds, **metrics}
        row["selection_score"] = _score(metrics)
        threshold_rows.append(row)
    threshold_rows.sort(key=lambda row: row["selection_score"], reverse=True)

    refinement_rows: list[dict[str, Any]] = []
    windows = [
        ("00:00", "23:59"),
        ("00:00", "07:00"),
        ("07:30", "12:00"),
        ("17:00", "19:00"),
        ("17:00", "20:00"),
    ]
    policies = [
        ("none", Phase2DecisionPolicy()),
        ("trend", Phase2DecisionPolicy(entry_trend_filter="vwap_ema")),
        ("cooldown2", Phase2DecisionPolicy(cooldown_bars_after_flip=2)),
        (
            "trend_cooldown2",
            Phase2DecisionPolicy(entry_trend_filter="vwap_ema", cooldown_bars_after_flip=2),
        ),
    ]
    checkpoint = {
        "schema_version": 1,
        "inputs": [str(path.resolve()) for path in args.eval_csv],
        "risk_rails": {
            "max_risk_per_trade_usd": args.max_risk_per_trade,
            "daily_loss_stop_usd": args.daily_loss_stop,
            "max_trades_per_day": args.max_trades_per_day,
            "max_losses_per_day": args.max_losses_per_day,
        },
        "threshold_stage": {"tested": len(threshold_rows), "top": threshold_rows[:30]},
        "refinement_stage": {"status": "running"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")

    top_thresholds = threshold_rows[:6]
    for base, window, (policy_name, policy), hold in itertools.product(
        top_thresholds, windows, policies, (8, 12, 20)
    ):
        thresholds = {
            "p_setup": float(base["p_setup"]),
            "p_long": float(base["p_long"]),
            "p_short": float(base["p_short"]),
        }
        trades = _simulate(
            frames,
            thresholds=thresholds,
            policy=policy,
            trade_window=window,
            max_hold_bars=hold,
            point_value=instrument.point_value,
            tick_value=instrument.tick_value,
            commission=args.commission,
            slippage_ticks=args.slippage_ticks,
        )
        accepted = _apply_risk_rails(
            trades,
            max_risk_per_trade=args.max_risk_per_trade,
            daily_loss_stop=args.daily_loss_stop,
            max_trades_per_day=args.max_trades_per_day,
            max_losses_per_day=args.max_losses_per_day,
        )
        metrics = _metrics(accepted)
        safe = (
            metrics["net_pnl_usd"] > 0
            and metrics["max_drawdown_usd"] <= 1800
            and metrics["worst_day_usd"] >= -650
            and metrics["profit_factor"] >= 1.5
            and metrics["trades"] >= 20
            and min((metrics.get("segment_pnl") or {"none": -1}).values()) > 0
        )
        row = {
            **thresholds,
            "window_start": window[0],
            "window_end": window[1],
            "policy": policy_name,
            "policy_config": policy.to_manifest(),
            "max_hold_bars": hold,
            **metrics,
            "prop_safe": safe,
        }
        row["selection_score"] = _score(metrics)
        refinement_rows.append(row)
    refinement_rows.sort(
        key=lambda row: (bool(row["prop_safe"]), float(row["selection_score"])),
        reverse=True,
    )

    payload = {
        "schema_version": 1,
        "inputs": [str(path.resolve()) for path in args.eval_csv],
        "risk_rails": {
            "max_risk_per_trade_usd": args.max_risk_per_trade,
            "daily_loss_stop_usd": args.daily_loss_stop,
            "max_trades_per_day": args.max_trades_per_day,
            "max_losses_per_day": args.max_losses_per_day,
        },
        "threshold_stage": {
            "tested": len(threshold_rows),
            "top": threshold_rows[:30],
        },
        "refinement_stage": {
            "tested": len(refinement_rows),
            "prop_safe_count": sum(bool(row["prop_safe"]) for row in refinement_rows),
            "selected": refinement_rows[0] if refinement_rows else None,
            "top": refinement_rows[:50],
        },
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "thresholds_tested": len(threshold_rows),
                "refinements_tested": len(refinement_rows),
                "prop_safe_count": payload["refinement_stage"]["prop_safe_count"],
                "selected": payload["refinement_stage"]["selected"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
