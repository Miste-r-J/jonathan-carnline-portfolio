"""Volatility and crash detection filter.

Blocks signals during dangerous market conditions.
This module addresses the critical gap that caused the 5.23.26 session loss:
- ATR spike detection
- Price velocity (crash) detection
- Market regime classification
- Signal validation before execution
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


@dataclass
class VolatilityConfig:
    """Configuration for volatility-based blocking."""
    # ATR spike detection
    atr_spike_multiplier: float = 2.0  # Block if ATR > 2x recent average
    atr_lookback_bars: int = 20  # Lookback for ATR average
    
    # Price velocity detection (crash detection)
    price_velocity_threshold: float = 5.0  # Points per minute
    velocity_lookback_bars: int = 5
    
    # Volatility regime
    high_vol_regime_atr_mult: float = 1.5  # Reduce trading in high vol
    extreme_vol_regime_atr_mult: float = 2.5  # Block all trading
    
    # Signal blocking
    max_signal_age_sec: float = 3.0  # Reject signals older than 3s
    max_slippage_points: float = 2.0  # Reject if slippage > 2 pts


@dataclass
class VolatilityState:
    """Current volatility state."""
    current_atr: float
    avg_atr: float
    atr_ratio: float
    price_velocity: float  # Points per minute
    regime: str  # "normal", "elevated", "high", "extreme"
    should_block: bool
    block_reason: str
    timestamp: float = 0.0


class VolatilityFilter:
    """
    Detects and blocks signals during dangerous volatility conditions.
    
    This filter addresses the root cause of the 5.23.26 crash trade:
    - The model generated a LONG signal at 6670.75
    - Market crashed 76+ points in seconds
    - No volatility filter was in place to block the signal
    
    Usage:
        filter = VolatilityFilter(config)
        state = filter.update(current_atr, current_price, timestamp)
        if state.should_block:
            # Don't generate signals
            pass
    """
    
    def __init__(self, config: VolatilityConfig, tick_size: float = 0.25):
        self.config = config
        self.tick_size = tick_size
        self._atr_history: List[float] = []
        self._price_history: List[Tuple[float, float]] = []  # (timestamp, price)
        
    def update(self, current_atr: float, current_price: float, timestamp: float) -> VolatilityState:
        """
        Update volatility state with new data. Call this on every bar.
        
        Args:
            current_atr: Current ATR value in points
            current_price: Current price
            timestamp: Current timestamp (seconds)
            
        Returns:
            VolatilityState with current regime and blocking status
        """
        # Update ATR history
        self._atr_history.append(current_atr)
        if len(self._atr_history) > self.config.atr_lookback_bars:
            self._atr_history.pop(0)
            
        # Calculate average ATR
        avg_atr = float(np.mean(self._atr_history)) if self._atr_history else current_atr
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        # Update price history for velocity
        self._price_history.append((timestamp, current_price))
        
        # Keep only last N bars (assume 1-min bars, so 5 bars = 5 minutes)
        cutoff = timestamp - (self.config.velocity_lookback_bars * 60)
        self._price_history = [(t, p) for t, p in self._price_history if t >= cutoff]
        
        # Calculate price velocity (points per minute)
        price_velocity = 0.0
        if len(self._price_history) >= 2:
            oldest_t, oldest_p = self._price_history[0]
            newest_t, newest_p = self._price_history[-1]
            time_diff = newest_t - oldest_t
            if time_diff > 0:
                price_velocity = abs(newest_p - oldest_p) / (time_diff / 60)
        
        # Determine regime
        regime = self._classify_regime(atr_ratio, price_velocity)
        
        # Determine if should block
        should_block, block_reason = self._should_block_signal(atr_ratio, price_velocity, regime)
        
        return VolatilityState(
            current_atr=current_atr,
            avg_atr=avg_atr,
            atr_ratio=atr_ratio,
            price_velocity=price_velocity,
            regime=regime,
            should_block=should_block,
            block_reason=block_reason,
            timestamp=timestamp
        )
    
    def _classify_regime(self, atr_ratio: float, price_velocity: float) -> str:
        """Classify current market regime."""
        if atr_ratio > self.config.extreme_vol_regime_atr_mult:
            return "extreme"
        elif atr_ratio > self.config.high_vol_regime_atr_mult:
            return "high"
        elif atr_ratio > 1.2 or price_velocity > self.config.price_velocity_threshold:
            return "elevated"
        else:
            return "normal"
    
    def _should_block_signal(self, atr_ratio: float, price_velocity: float, regime: str) -> Tuple[bool, str]:
        """Determine if signals should be blocked."""
        # Block in extreme volatility
        if regime == "extreme":
            return True, f"extreme_volatility_regime:atr_ratio={atr_ratio:.2f}"
        
        # Block on ATR spike
        if atr_ratio > self.config.atr_spike_multiplier:
            return True, f"atr_spike:ratio={atr_ratio:.2f}>threshold={self.config.atr_spike_multiplier}"
        
        # Block on price crash/velocity (critical for flash crash detection)
        if price_velocity > self.config.price_velocity_threshold * 2:
            return True, f"crash_detected:velocity={price_velocity:.1f}pts/min"
        
        return False, ""
    
    def validate_signal(
        self,
        signal_price: float,
        current_price: float,
        signal_timestamp: float,
        current_timestamp: float,
        state: Optional[VolatilityState] = None
    ) -> Tuple[bool, str, dict]:
        """
        Validate a signal before execution.
        
        This catches:
        - Stale signals (generated too long ago)
        - Excessive slippage (price moved too much)
        - Volatility-based blocking
        
        Args:
            signal_price: Price when signal was generated
            current_price: Current market price
            signal_timestamp: When signal was generated (seconds)
            current_timestamp: Current time (seconds)
            state: Current volatility state (optional, will use last state if not provided)
            
        Returns:
            Tuple of (is_valid, rejection_reason, details_dict)
        """
        if state is None:
            state = self._last_state
            if state is None:
                return False, "volatility_state_not_initialized", {}
        
        details = {
            "signal_price": signal_price,
            "current_price": current_price,
            "slippage_points": abs(current_price - signal_price),
            "signal_age_sec": current_timestamp - signal_timestamp,
            "volatility_regime": state.regime,
            "atr_ratio": state.atr_ratio,
            "price_velocity": state.price_velocity,
        }
        
        # Check signal age
        signal_age = current_timestamp - signal_timestamp
        if signal_age > self.config.max_signal_age_sec:
            return False, f"signal_stale:age={signal_age:.1f}s>max={self.config.max_signal_age_sec}s", details
        
        # Check slippage
        slippage = abs(current_price - signal_price)
        if slippage > self.config.max_slippage_points:
            return False, f"slippage_too_high:{slippage:.1f}pts>max={self.config.max_slippage_points}pts", details
        
        # Check volatility state
        if state.should_block:
            return False, state.block_reason, details
        
        return True, "", details
    
    # Store last state for validation
    _last_state: Optional[VolatilityState] = None
    
    def update(self, *args, **kwargs) -> VolatilityState:
        state = self._update_impl(*args, **kwargs)
        self._last_state = state
        return state
    
    def _update_impl(self, current_atr: float, current_price: float, timestamp: float) -> VolatilityState:
        """Implementation of update method."""
        # Update ATR history
        self._atr_history.append(current_atr)
        if len(self._atr_history) > self.config.atr_lookback_bars:
            self._atr_history.pop(0)
            
        # Calculate average ATR
        avg_atr = float(np.mean(self._atr_history)) if self._atr_history else current_atr
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        # Update price history for velocity
        self._price_history.append((timestamp, current_price))
        
        # Keep only last N bars
        cutoff = timestamp - (self.config.velocity_lookback_bars * 60)
        self._price_history = [(t, p) for t, p in self._price_history if t >= cutoff]
        
        # Calculate price velocity
        price_velocity = 0.0
        if len(self._price_history) >= 2:
            oldest_t, oldest_p = self._price_history[0]
            newest_t, newest_p = self._price_history[-1]
            time_diff = newest_t - oldest_t
            if time_diff > 0:
                price_velocity = abs(newest_p - oldest_p) / (time_diff / 60)
        
        # Determine regime
        regime = self._classify_regime(atr_ratio, price_velocity)
        
        # Determine if should block
        should_block, block_reason = self._should_block_signal(atr_ratio, price_velocity, regime)
        
        return VolatilityState(
            current_atr=current_atr,
            avg_atr=avg_atr,
            atr_ratio=atr_ratio,
            price_velocity=price_velocity,
            regime=regime,
            should_block=should_block,
            block_reason=block_reason,
            timestamp=timestamp
        )
