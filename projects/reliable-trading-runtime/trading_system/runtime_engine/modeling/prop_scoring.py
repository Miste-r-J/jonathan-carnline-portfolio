from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class PropScoringConfig:
    """Risk and behavior targets for prop-style Phase-2 candidate ranking."""

    profile: str = "prop_aggressive"
    max_daily_loss: float = 900.0
    max_drawdown: float = 5000.0
    max_trades_per_day: int = 10
    min_trades_per_day: float = 4.0
    max_consecutive_losses: int = 3
    max_flip_rate_per_day: float = 0.25
    preferred_flip_rate_per_day: float = 0.20
    max_bad_fade_rate: float = 0.18
    min_profit_factor: float = 2.0
    target_profit_factor: float = 3.0
    min_trade_count: int = 50
    min_total_pnl_usd: float = 0.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    if result != result:
        return default
    return result


def _parse_trade_day(trade: Mapping[str, Any]) -> str | None:
    raw = trade.get("entry_ts") or trade.get("exit_ts")
    if raw is None:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None
    return dt.date().isoformat()


def daily_trade_stats(trades: Iterable[Mapping[str, Any]], *, max_daily_loss: float) -> Dict[str, Any]:
    daily_pnl: Dict[str, float] = {}
    daily_trades: Dict[str, int] = {}
    for trade in trades:
        day = _parse_trade_day(trade)
        if day is None:
            continue
        daily_pnl[day] = daily_pnl.get(day, 0.0) + _to_float(trade.get("pnl_usd"))
        daily_trades[day] = daily_trades.get(day, 0) + 1

    losing_days = [pnl for pnl in daily_pnl.values() if pnl < 0.0]
    largest_losing_day = min(losing_days) if losing_days else 0.0
    return {
        "days": len(daily_pnl),
        "daily_pnl": dict(sorted(daily_pnl.items())),
        "daily_trades": dict(sorted(daily_trades.items())),
        "max_trades_day": max(daily_trades.values()) if daily_trades else 0,
        "avg_trades_day": (sum(daily_trades.values()) / max(1, len(daily_trades))) if daily_trades else 0.0,
        "largest_winning_day": max(daily_pnl.values()) if daily_pnl else 0.0,
        "largest_losing_day": largest_losing_day,
        "daily_loss_breach_count": sum(1 for pnl in daily_pnl.values() if pnl <= -abs(max_daily_loss)),
        "positive_day_rate": (
            sum(1 for pnl in daily_pnl.values() if pnl > 0.0) / max(1, len(daily_pnl))
            if daily_pnl
            else 0.0
        ),
    }


def max_consecutive_losses(trades: Iterable[Mapping[str, Any]]) -> int:
    streak = 0
    max_streak = 0
    for trade in trades:
        if _to_float(trade.get("pnl_usd")) < 0.0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def evaluate_prop_candidate(
    *,
    slip1: Mapping[str, Any],
    slippage: Mapping[str, Mapping[str, Any]],
    trades: Iterable[Mapping[str, Any]],
    bad_fade_rate: float,
    config: PropScoringConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or PropScoringConfig()
    trade_list = list(trades)
    daily = daily_trade_stats(trade_list, max_daily_loss=cfg.max_daily_loss)
    loss_streak = max_consecutive_losses(trade_list)

    total_pnl = _to_float(slip1.get("total_pnl_usd"))
    profit_factor = _to_float(slip1.get("profit_factor"))
    sharpe = _to_float(slip1.get("sharpe"))
    max_drawdown = abs(_to_float(slip1.get("max_drawdown")))
    trade_count = int(_to_float(slip1.get("trade_count")))
    trades_per_day = _to_float(slip1.get("trades_per_day"))
    flip_rate = _to_float(slip1.get("flip_rate_per_day"))
    avg_trade = _to_float(slip1.get("avg_trade_usd"))
    win_rate = _to_float(slip1.get("win_rate"))

    slippage_profitable = all(
        _to_float(payload.get("total_pnl_usd")) > 0.0 and _to_float(payload.get("profit_factor")) >= 1.0
        for payload in slippage.values()
    )
    overtrade_penalty = max(0.0, trades_per_day - float(cfg.max_trades_per_day)) * 350.0
    undertrade_penalty = max(0.0, float(cfg.min_trades_per_day) - trades_per_day) * 100.0
    hard_flip_penalty = max(0.0, flip_rate - cfg.preferred_flip_rate_per_day) * 750.0
    drawdown_penalty = max_drawdown / 35.0
    daily_loss_penalty = int(daily["daily_loss_breach_count"]) * 1250.0
    loss_streak_penalty = max(0, loss_streak - cfg.max_consecutive_losses) * 400.0
    max_trades_day_penalty = max(0, int(daily["max_trades_day"]) - cfg.max_trades_per_day) * 125.0
    slippage_penalty = 0.0 if slippage_profitable else 1500.0

    score = (
        total_pnl / 50.0
        + sharpe * 120.0
        + min(profit_factor, 8.0) * 125.0
        + avg_trade * 1.5
        + win_rate * 300.0
        + min(trade_count, 250) * 1.0
        - drawdown_penalty
        - loss_streak_penalty
        - daily_loss_penalty
        - max_trades_day_penalty
        - overtrade_penalty
        - undertrade_penalty
        - hard_flip_penalty
        - bad_fade_rate * 1500.0
        - slippage_penalty
    )

    errors = []
    if total_pnl <= cfg.min_total_pnl_usd:
        errors.append("non_positive_test_pnl")
    if profit_factor < cfg.min_profit_factor:
        errors.append("low_profit_factor")
    if trade_count < cfg.min_trade_count:
        errors.append("low_trade_count")
    if max_drawdown > cfg.max_drawdown:
        errors.append("max_drawdown_breach")
    if loss_streak > cfg.max_consecutive_losses:
        errors.append("loss_streak_breach")
    if flip_rate > cfg.max_flip_rate_per_day:
        errors.append("flip_rate_breach")
    if bad_fade_rate > cfg.max_bad_fade_rate:
        errors.append("bad_fade_breach")
    if int(daily["daily_loss_breach_count"]) > 0:
        errors.append("daily_loss_breach")
    if not slippage_profitable:
        errors.append("slippage_not_profitable")

    return {
        "profile": cfg.profile,
        "score": float(score),
        "deployable": not errors,
        "errors": errors,
        "daily": daily,
        "max_consecutive_losses": loss_streak,
        "slippage_profitable": slippage_profitable,
        "targets": {
            "max_daily_loss": cfg.max_daily_loss,
            "max_drawdown": cfg.max_drawdown,
            "max_trades_per_day": cfg.max_trades_per_day,
            "min_trades_per_day": cfg.min_trades_per_day,
            "max_consecutive_losses": cfg.max_consecutive_losses,
            "max_flip_rate_per_day": cfg.max_flip_rate_per_day,
            "preferred_flip_rate_per_day": cfg.preferred_flip_rate_per_day,
            "max_bad_fade_rate": cfg.max_bad_fade_rate,
            "min_profit_factor": cfg.min_profit_factor,
            "target_profit_factor": cfg.target_profit_factor,
            "min_trade_count": cfg.min_trade_count,
        },
    }
