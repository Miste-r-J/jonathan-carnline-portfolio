# signal_engine.py
"""
Signal processing engine for threshold-based entry signals.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from na.common.signal import SignalPayload
from .feature_flags import FeatureContext
from .gates import evaluate_structure_gates, evaluate_microstructure_gates


def process_signal_entry(
    idx: int,
    desired: int,
    selected_prob: float,
    structure_cfg: Dict[str, Any],
    ema_fast_values: Optional[np.ndarray],
    ema_slow_values: Optional[np.ndarray],
    vwap_values: Optional[np.ndarray],
    vwap_slope_values: Optional[np.ndarray],
    price_vs_vwap: np.ndarray,
    above_vwap_bool: np.ndarray,
    feature_context: FeatureContext,
    tick_size: float,
    side_proba: Optional[np.ndarray] = None,
    atr_ticks_values: Optional[np.ndarray] = None,
    trades_1m_values: Optional[np.ndarray] = None,
) -> Tuple[bool, str, float]:
    """
    Process a potential entry signal through structure-aware gates.

    Args:
        idx: Current bar index
        desired: Desired position (-1, 0, 1)
        selected_prob: Selected probability
        structure_cfg: Structure rules config
        ema_fast_values: EMA fast array
        ema_slow_values: EMA slow array
        vwap_values: VWAP array
        vwap_slope_values: VWAP slope array
        price_vs_vwap: Price vs VWAP array
        above_vwap_bool: Above VWAP boolean array
        feature_context: Feature context

    Returns:
        Tuple of (allow_entry, reason_if_blocked, size_multiplier)
    """
    if desired == 0:
    return True, "", 1.0


def build_signal_payload(
    *,
    symbol: str,
    side: str,
    qty: float,
    prob: float,
    regime: Optional[str] = None,
    aoi_id: Optional[str] = None,
    preset_id: Optional[str] = None,
    micro_ok: Optional[bool] = None,
    structure_ok: Optional[bool] = None,
) -> SignalPayload:
    """
    Build a SignalPayload instance that matches router expectations.
    """
    return SignalPayload(
        symbol=symbol,
        side=side,
        qty=qty,
        prob=prob,
        regime=regime,
        aoi_id=aoi_id,
        preset_id=preset_id,
        micro_ok=micro_ok,
        structure_ok=structure_ok,
    )

    # Check kill-switch
    if feature_context.kill_switch:
        return False, "kill_switch_active", 1.0

    # Evaluate microstructure gates
    should_block, reason = evaluate_microstructure_gates(
        idx=idx,
        feature_context=feature_context,
        atr_ticks_values=atr_ticks_values,
        trades_1m_values=trades_1m_values,
    )

    if should_block:
        return False, reason, 1.0

    # Evaluate structure gates
    should_block, reason = evaluate_structure_gates(
        idx=idx,
        side=desired,
        selected_prob=selected_prob,
        structure_cfg=structure_cfg,
        ema_fast_values=ema_fast_values,
        ema_slow_values=ema_slow_values,
        vwap_values=vwap_values,
        vwap_slope_values=vwap_slope_values,
        price_vs_vwap=price_vs_vwap,
        above_vwap_bool=above_vwap_bool,
        feature_context=feature_context,
        tick_size=tick_size,
    )

    if should_block:
        return False, reason, 1.0

    # Prob-slope nudge for SHORT signals
    size_mult = 1.0
    if feature_context.is_enabled("prob_slope_nudge") and desired < 0:
        cfg = feature_context.config_for("prob_slope_nudge").get("short", {})
        p_min = cfg.get("p_min")
        p_max = cfg.get("p_max")
        dp_last2_min = cfg.get("dp_last2_min")
        if p_min is not None and p_max is not None and dp_last2_min is not None:
            if selected_prob >= p_min and selected_prob < p_max:
                if idx >= 2 and side_proba is not None and len(side_proba) > idx - 2 and idx - 2 >= 0:
                    prev_prob = side_proba[idx - 2]
                    if np.isfinite(prev_prob):
                        delta = selected_prob - prev_prob
                        if delta >= dp_last2_min:
                            size_mult = cfg.get("size_mult", 1.0)

    return True, "", size_mult
