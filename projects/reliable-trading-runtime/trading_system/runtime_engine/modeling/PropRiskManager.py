"""Prop account risk manager with hard limits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Tuple


class RiskLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACH = "breach"


@dataclass
class PropRiskConfig:
    """Prop account risk limits."""

    max_daily_loss: float = 800.0
    daily_loss_warning: float = 300.0
    max_drawdown: float = 2000.0
    drawdown_warning: float = 1500.0
    max_trades_per_day: int = 5
    max_consecutive_losses: int = 2
    max_contracts: int = 2
    max_risk_per_trade: float = 600.0
    allowed_start_time: str = "00:30"
    allowed_end_time: str = "00:29"
    no_trade_before: str = "00:30"
    no_trade_after: str = "00:29"

    def to_dict(self) -> dict:
        return {
            "max_daily_loss": self.max_daily_loss,
            "daily_loss_warning": self.daily_loss_warning,
            "max_drawdown": self.max_drawdown,
            "drawdown_warning": self.drawdown_warning,
            "max_trades_per_day": self.max_trades_per_day,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_contracts": self.max_contracts,
            "max_risk_per_trade": self.max_risk_per_trade,
            "allowed_start_time": self.allowed_start_time,
            "allowed_end_time": self.allowed_end_time,
            "no_trade_before": self.no_trade_before,
            "no_trade_after": self.no_trade_after,
        }


@dataclass
class TradeRecord:
    """Record of a trade for risk tracking."""

    timestamp: datetime
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    contracts: int
    signal_id: str
    stop_distance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "contracts": self.contracts,
            "signal_id": self.signal_id,
            "stop_distance": self.stop_distance,
        }


@dataclass
class DailyRiskState:
    """Current day's risk state."""

    date: date
    trades: List[TradeRecord] = field(default_factory=list)
    realized_pnl: float = 0.0
    open_pnl: float = 0.0
    peak_pnl: float = 0.0
    trough_pnl: float = 0.0
    consecutive_losses: int = 0
    is_locked: bool = False
    lock_reason: str = ""
    starting_balance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "date": str(self.date),
            "trades": [t.to_dict() for t in self.trades],
            "realized_pnl": self.realized_pnl,
            "open_pnl": self.open_pnl,
            "peak_pnl": self.peak_pnl,
            "trough_pnl": self.trough_pnl,
            "consecutive_losses": self.consecutive_losses,
            "is_locked": self.is_locked,
            "lock_reason": self.lock_reason,
            "starting_balance": self.starting_balance,
        }


class PropRiskManager:
    """Hard risk limits for prop account trading."""

    def __init__(self, config: PropRiskConfig, point_value: float = 50.0):
        self.config = config
        self.point_value = point_value
        self._state: Optional[DailyRiskState] = None
        self._account_starting_balance: float = 0.0

    def start_day(self, starting_balance: float) -> None:
        self._state = DailyRiskState(
            date=date.today(),
            starting_balance=starting_balance,
        )
        self._account_starting_balance = starting_balance

    def can_open_trade(
        self,
        current_time: datetime,
        contracts: int,
        stop_distance_points: float,
        check_time: bool = True,
    ) -> Tuple[bool, str, RiskLevel]:
        if self._state is None:
            return False, "day_not_initialized", RiskLevel.CRITICAL
        if self._state.is_locked:
            return False, f"account_locked:{self._state.lock_reason}", RiskLevel.BREACH

        if check_time:
            time_str = current_time.strftime("%H:%M")
            if time_str < self.config.no_trade_before:
                return False, f"before_no_trade_time:{time_str}<{self.config.no_trade_before}", RiskLevel.NORMAL
            if time_str > self.config.no_trade_after:
                return False, f"after_no_trade_time:{time_str}>{self.config.no_trade_after}", RiskLevel.NORMAL
            if time_str < self.config.allowed_start_time or time_str > self.config.allowed_end_time:
                return False, f"outside_trading_hours:{time_str}", RiskLevel.NORMAL

        if len(self._state.trades) >= self.config.max_trades_per_day:
            return (
                False,
                f"max_trades_reached:{len(self._state.trades)}>={self.config.max_trades_per_day}",
                RiskLevel.WARNING,
            )
        if self._state.consecutive_losses >= self.config.max_consecutive_losses:
            return (
                False,
                f"consecutive_loss_limit:{self._state.consecutive_losses}>={self.config.max_consecutive_losses}",
                RiskLevel.WARNING,
            )
        if contracts > self.config.max_contracts:
            return False, f"exceeds_max_contracts:{contracts}>{self.config.max_contracts}", RiskLevel.CRITICAL

        risk_per_trade = contracts * stop_distance_points * self.point_value
        if risk_per_trade > self.config.max_risk_per_trade:
            return (
                False,
                f"risk_too_high:${risk_per_trade:.0f}>${self.config.max_risk_per_trade:.0f}",
                RiskLevel.WARNING,
            )
        if self._state.realized_pnl < -self.config.daily_loss_warning:
            return (
                False,
                f"approaching_daily_limit:${abs(self._state.realized_pnl):.0f}>warning_${self.config.daily_loss_warning:.0f}",
                RiskLevel.WARNING,
            )
        if self._state.realized_pnl < -self.config.max_daily_loss:
            self._state.is_locked = True
            self._state.lock_reason = "daily_loss_limit_breach"
            return (
                False,
                f"daily_loss_breach:${abs(self._state.realized_pnl):.0f}>${self.config.max_daily_loss:.0f}",
                RiskLevel.BREACH,
            )

        current_dd = self._calculate_drawdown()
        if current_dd > self.config.drawdown_warning:
            return (
                False,
                f"approaching_drawdown_limit:${current_dd:.0f}>warning_${self.config.drawdown_warning:.0f}",
                RiskLevel.WARNING,
            )
        if current_dd > self.config.max_drawdown:
            self._state.is_locked = True
            self._state.lock_reason = "max_drawdown_breach"
            return (
                False,
                f"drawdown_breach:${current_dd:.0f}>${self.config.max_drawdown:.0f}",
                RiskLevel.BREACH,
            )
        return True, "ok", RiskLevel.NORMAL

    def record_trade(self, trade: TradeRecord) -> None:
        if self._state is None:
            return
        self._state.trades.append(trade)
        self._state.realized_pnl += trade.pnl
        self._state.peak_pnl = max(self._state.peak_pnl, self._state.realized_pnl)
        self._state.trough_pnl = min(self._state.trough_pnl, self._state.realized_pnl)
        self._state.consecutive_losses = self._state.consecutive_losses + 1 if trade.pnl < 0 else 0
        self._check_limits()

    def _calculate_drawdown(self) -> float:
        if self._state is None:
            return 0.0
        return max(0.0, self._state.peak_pnl - self._state.realized_pnl)

    def _check_limits(self) -> None:
        if self._state is None:
            return
        if self._state.realized_pnl < -self.config.max_daily_loss:
            self._state.is_locked = True
            self._state.lock_reason = "daily_loss_limit_breach"
            return
        if self._calculate_drawdown() > self.config.max_drawdown:
            self._state.is_locked = True
            self._state.lock_reason = "max_drawdown_breach"

    def update_open_pnl(self, open_pnl: float) -> None:
        if self._state is None:
            return
        self._state.open_pnl = open_pnl
        total_pnl = self._state.realized_pnl + open_pnl
        if total_pnl < -self.config.max_daily_loss:
            self._state.is_locked = True
            self._state.lock_reason = "unrealized_daily_loss_breach"

    def get_state_summary(self) -> dict:
        if self._state is None:
            return {"status": "not_initialized"}
        total_pnl = self._state.realized_pnl + self._state.open_pnl
        return {
            "date": str(self._state.date),
            "trades_count": len(self._state.trades),
            "realized_pnl": self._state.realized_pnl,
            "open_pnl": self._state.open_pnl,
            "total_pnl": total_pnl,
            "daily_pnl_remaining": self.config.max_daily_loss + total_pnl,
            "drawdown": self._calculate_drawdown(),
            "consecutive_losses": self._state.consecutive_losses,
            "is_locked": self._state.is_locked,
            "lock_reason": self._state.lock_reason,
            "peak_pnl": self._state.peak_pnl,
            "trough_pnl": self._state.trough_pnl,
        }

    def to_json(self) -> str:
        return json.dumps(self.get_state_summary(), indent=2)

    def reset(self) -> None:
        self._state = None
