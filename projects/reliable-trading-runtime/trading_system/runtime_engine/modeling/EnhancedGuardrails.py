"""Enhanced guardrails integrating volatility and risk management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .PropRiskManager import PropRiskConfig, PropRiskManager, RiskLevel, TradeRecord
from .VolatilityFilter import VolatilityConfig, VolatilityFilter, VolatilityState


@dataclass(frozen=True)
class EnhancedGuardRailConfig:
    """Extended guardrail configuration for prop-ready trading."""

    price_age_max_sec: float = 10.0
    bar_age_max_sec: float = 90.0
    snapshot_age_max_sec: float = 60.0
    atr_spike_multiplier: float = 2.0
    max_slippage_points: float = 2.0
    signal_max_age_sec: float = 3.0
    max_daily_loss: float = 500.0
    max_trades_per_day: int = 5
    max_consecutive_losses: int = 2
    fill_slippage_max: float = 4.0
    signal_validation_enabled: bool = True
    volatility_check_enabled: bool = True


@dataclass
class EnhancedGuardRailState:
    """Complete guardrail state including volatility and risk."""

    allowed_to_arm: bool
    allowed_to_emit_entries: bool
    required_action: str
    reasons: List[str] = field(default_factory=list)
    volatility_state: Optional[VolatilityState] = None
    risk_level: RiskLevel = RiskLevel.NORMAL
    signal_validated: bool = True
    execution_validated: bool = True
    lockout_code: Optional[str] = None
    preflight_ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_to_arm": self.allowed_to_arm,
            "allowed_to_emit_entries": self.allowed_to_emit_entries,
            "required_action": self.required_action,
            "reasons": list(self.reasons),
            "risk_level": self.risk_level.value,
            "signal_validated": self.signal_validated,
            "execution_validated": self.execution_validated,
            "lockout_code": self.lockout_code,
            "preflight_ok": self.preflight_ok,
            "volatility_regime": self.volatility_state.regime if self.volatility_state else None,
            "atr_ratio": self.volatility_state.atr_ratio if self.volatility_state else None,
        }


class EnhancedGuardrails:
    """Combined guardrails with volatility and prop-risk management."""

    def __init__(
        self,
        guard_config: EnhancedGuardRailConfig,
        vol_config: VolatilityConfig,
        risk_config: PropRiskConfig,
        tick_size: float = 0.25,
        point_value: float = 50.0,
    ):
        self.config = guard_config
        self.volatility_filter = VolatilityFilter(vol_config, tick_size)
        self.risk_manager = PropRiskManager(risk_config, point_value)
        self._current_vol_state: Optional[VolatilityState] = None

    def start_day(self, starting_balance: float) -> None:
        self.risk_manager.start_day(starting_balance)

    def update_volatility(self, atr: float, price: float, timestamp: float) -> VolatilityState:
        self._current_vol_state = self.volatility_filter.update(atr, price, timestamp)
        return self._current_vol_state

    def check_pre_signal(
        self,
        contracts: int = 1,
        stop_distance: float = 8.0,
        check_time: bool = True,
        current_time: Optional[datetime] = None,
    ) -> EnhancedGuardRailState:
        reasons: List[str] = []
        allowed = True
        risk_level = RiskLevel.NORMAL

        if self._current_vol_state and self._current_vol_state.should_block:
            allowed = False
            reasons.append(self._current_vol_state.block_reason)
            risk_level = RiskLevel.CRITICAL

        check_now = current_time if isinstance(current_time, datetime) else datetime.now()
        can_trade, risk_reason, risk_lvl = self.risk_manager.can_open_trade(
            current_time=check_now,
            contracts=contracts,
            stop_distance_points=stop_distance,
            check_time=check_time,
        )
        if not can_trade:
            allowed = False
            reasons.append(risk_reason)
            if risk_lvl in {RiskLevel.CRITICAL, RiskLevel.BREACH}:
                risk_level = risk_lvl
            elif risk_level == RiskLevel.NORMAL:
                risk_level = risk_lvl

        return EnhancedGuardRailState(
            allowed_to_arm=allowed,
            allowed_to_emit_entries=allowed,
            required_action="block" if not allowed else "allow",
            reasons=reasons,
            volatility_state=self._current_vol_state,
            risk_level=risk_level,
            lockout_code=(
                "volatility_block"
                if self._current_vol_state and self._current_vol_state.should_block
                else None
            ),
        )

    def validate_signal_for_execution(
        self,
        signal_price: float,
        current_price: float,
        signal_timestamp: float,
        current_timestamp: float,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        if not self.config.signal_validation_enabled:
            return True, "", {}

        # Startup bootstrap: volatility state may not be initialized on the
        # first eligible signal after process start. In that case, do not
        # hard-block execution solely for missing volatility context.
        vol_state = self._current_vol_state

        details = {
            "signal_price": signal_price,
            "current_price": current_price,
            "slippage_points": abs(current_price - signal_price),
            "signal_age_sec": current_timestamp - signal_timestamp,
            "volatility_regime": vol_state.regime if vol_state else None,
            "atr_ratio": vol_state.atr_ratio if vol_state else None,
            "price_velocity": vol_state.price_velocity if vol_state else None,
            "volatility_state_initialized": bool(vol_state is not None),
        }

        signal_age = current_timestamp - signal_timestamp
        if signal_age > self.config.signal_max_age_sec:
            return False, f"signal_stale:age={signal_age:.1f}s>max={self.config.signal_max_age_sec}s", details

        slippage = abs(current_price - signal_price)
        if slippage > self.config.max_slippage_points:
            return False, f"slippage_too_high:{slippage:.1f}pts>max={self.config.max_slippage_points}pts", details

        if self.config.volatility_check_enabled and vol_state and vol_state.should_block:
            return False, vol_state.block_reason, details
        return True, "", details

    def validate_fill(self, expected_price: float, fill_price: float, side: str) -> Tuple[bool, str]:
        del side
        slippage = abs(fill_price - expected_price)
        if slippage > self.config.fill_slippage_max:
            return False, f"fill_slippage_excessive:{slippage:.1f}pts>max_{self.config.fill_slippage_max}pts"
        return True, ""

    def record_trade(self, trade: TradeRecord) -> None:
        self.risk_manager.record_trade(trade)

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "guardrails": {
                "signal_validation_enabled": self.config.signal_validation_enabled,
                "volatility_check_enabled": self.config.volatility_check_enabled,
            },
            "volatility": self._current_vol_state.__dict__ if self._current_vol_state else None,
            "risk": self.risk_manager.get_state_summary(),
        }
