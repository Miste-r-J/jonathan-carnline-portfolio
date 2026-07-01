from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


class Reason(str, Enum):
    OK = "OK"
    SHORT_EDGE_WEAK = "SHORT_EDGE_WEAK"
    LONG_EDGE_WEAK = "LONG_EDGE_WEAK"
    MISALIGNED_TREND_VWAP = "MISALIGNED_TREND_VWAP"
    LUNCH_CHOP_WINDOW = "LUNCH_CHOP_WINDOW"
    R_TOO_LOW = "R_TOO_LOW"
    STOP_OUT_OF_BOUNDS = "STOP_OUT_OF_BOUNDS"
    LOW_VOL_WINDOW = "LOW_VOL_WINDOW"
    DAILY_RISK_LIMIT = "DAILY_RISK_LIMIT"
    MAX_TRADES_REACHED = "MAX_TRADES_REACHED"
    DAILY_LOCK_ACTIVE = "DAILY_LOCK_ACTIVE"
    TRAILING_DRAWDOWN_LIMIT = "TRAILING_DRAWDOWN_LIMIT"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    STRUCTURE_RULE_VIOLATION = "STRUCTURE_RULE_VIOLATION"
    NOT_RTH = "NOT_RTH"


class ExitReason(str, Enum):
    TARGET = "TARGET"
    STOP = "STOP"
    PROB_REVERSAL = "PROB_REVERSAL"
    TIME_CAP = "TIME_CAP"
    TRAIL = "TRAIL"
    MANUAL = "MANUAL"


@dataclass(frozen=True)
class L2Signal:
    side: Side
    entry: float
    stop: float
    target: float
    p_long: float
    grade: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class EntryAssessment:
    decision: Decision
    reason: Reason
    planned_R: float
    size_contracts: int


@dataclass
class Position:
    side: Side
    entry: float
    stop: float
    target: float
    size: int
    open_time: float
    unrealized_ticks: int = 0


@dataclass(frozen=True)
class MarketSnapshot:
    price: float
    vwap: float
    ema_fast: float
    ema_slow: float
    atr_ticks: int
    now_epoch: float
    tz: str
    is_rth: bool = True
    market_structure: str = ""


@dataclass
class RiskContext:
    trades_today: int = 0
    pnl_today: float = 0.0
    drawdown_usd: float = 0.0
    daily_loss_limit_usd: Optional[float] = None
    max_trades: Optional[int] = None
    trailing_drawdown_limit_usd: Optional[float] = None
    cooldown_active: bool = False
    cooldown_minutes: int = 0
    last_stop_time: Optional[float] = None
