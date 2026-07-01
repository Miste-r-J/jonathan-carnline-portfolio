from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TradeActionSnapshot(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL", "buy", "sell", "long", "short"]
    qty: float = Field(..., gt=0)
    price: float = Field(..., gt=0)


class RiskSnapshot(BaseModel):
    daily_loss: float = 0.0
    max_intraday_dd: float = 0.0


class JournalEnvelope(BaseModel):
    ts_utc: datetime
    session_date: date
    instrument: str
    mode: str = "SIM"
    allowed: bool = True
    preset_id: Optional[str] = None
    model_id: Optional[str] = None
    action: TradeActionSnapshot
    risk: RiskSnapshot
    pnl_before: float = 0.0
    pnl_after: float = 0.0
    pnl_delta: float = 0.0
    guard_reasons: list[str] = Field(default_factory=list)

