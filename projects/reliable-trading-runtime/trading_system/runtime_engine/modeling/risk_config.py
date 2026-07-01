from __future__ import annotations

"""
Centralized risk and cost configuration for futures backtests/live alignment.

Defaults target ES/MES-style instruments; override via CLI or env in runners.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from .config import INSTRUMENTS, InstrumentSpec


@dataclass(frozen=True)
class CostAssumptions:
    commission_per_contract: float = 2.0  # USD per side
    slippage_ticks_per_side: float = 1.0  # ticks per side (conservative baseline)


@dataclass(frozen=True)
class LiquidityAssumptions:
    participation_limit: float = 0.05  # fraction of recent bar volume
    lookback_bars: int = 5
    extra_slippage_ticks_illiquid: float = 0.5


@dataclass(frozen=True)
class RiskLimits:
    account_size: float = 100_000.0
    risk_per_trade_usd: float = 300.0
    max_daily_loss_usd: float = 2_000.0
    max_drawdown_usd: float = 10_000.0
    target_vol_annualized: float = 0.60
    max_leverage: float = 3.0
    max_hold_bars: int = 5  # align with label horizon by default


@dataclass(frozen=True)
class RiskConfig:
    instrument: InstrumentSpec
    costs: CostAssumptions = CostAssumptions()
    liquidity: LiquidityAssumptions = LiquidityAssumptions()
    limits: RiskLimits = RiskLimits()
    bars_per_day: Optional[int] = None
    trading_days_per_year: int = 252
    annualize_from_time: bool = True

    @property
    def tick_value(self) -> float:
        return self.instrument.tick_value


def default_risk_config(alias: str = "ES") -> RiskConfig:
    spec = INSTRUMENTS.get(alias)
    if spec is None:
        raise KeyError(f"Unknown instrument alias: {alias}")
    return RiskConfig(instrument=spec)
