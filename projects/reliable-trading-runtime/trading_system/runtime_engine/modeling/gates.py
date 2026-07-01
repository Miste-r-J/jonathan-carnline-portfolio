# gates.py
"""
Gate logic for entry signals, including structure-aware checks.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from .feature_flags import FeatureContext


def evaluate_structure_gates(
    idx: int,
    side: int,
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
) -> Tuple[bool, str]:
    """
    Evaluate structure-aware entry gates.

    Args:
        idx: Current bar index
        side: Signal side (1=long, -1=short)
        selected_prob: Selected probability for the signal
        structure_cfg: Structure rules configuration
        ema_fast_values: EMA fast values array
        ema_slow_values: EMA slow values array
        vwap_values: VWAP values array
        vwap_slope_values: VWAP slope values array
        price_vs_vwap: Price vs VWAP array
        above_vwap_bool: Boolean array for above VWAP
        feature_context: Feature context for flags

    Returns:
        Tuple of (should_block, reason)
    """
    if not feature_context.should_evaluate("structure_entries"):
        return False, ""

    price = price_vs_vwap[idx] + (vwap_values[idx] if vwap_values is not None and idx < len(vwap_values) else np.nan)

    should_block = False
    reason = ""

    # VWAP no-trade band check
    if feature_context.should_evaluate("vwap_no_trade_band"):
        band_ticks = feature_context.get_config("vwap_no_trade_band", {}).get("band_ticks", 0)
        if band_ticks > 0 and vwap_values is not None and idx < len(vwap_values):
            vwap = vwap_values[idx]
            if np.isfinite(price) and np.isfinite(vwap):
                diff = abs(price - vwap)
                band_size = band_ticks * tick_size
                if diff > band_size:
                    if feature_context.is_enabled("vwap_no_trade_band"):
                        should_block = True
                        reason = "vwap_no_trade_band"
                    elif feature_context.shadow_enabled():
                        reason = "vwap_no_trade_band:block"

    # VWAP requirements
    if side > 0 and structure_cfg.get("long_above_vwap_required"):
        if vwap_values is None or idx >= len(vwap_values):
            if feature_context.is_enabled("structure_entries"):
                should_block = True
                reason = "structure:vwap_missing"
            elif feature_context.shadow_enabled():
                reason = "structure:vwap_missing"
        elif price <= vwap_values[idx]:
            # Check vwap_reclaim if enabled
            if feature_context.should_evaluate("vwap_reclaim"):
                reclaim_ok, reclaim_reason = _vwap_reclaim_allows(
                    idx, selected_prob, structure_cfg, above_vwap_bool, vwap_slope_values, ema_fast_values, ema_slow_values
                )
                if not reclaim_ok:
                    if feature_context.is_enabled("vwap_reclaim"):
                        should_block = True
                        reason = reclaim_reason
                    elif feature_context.shadow_enabled():
                        reason = reclaim_reason or "vwap_reclaim:block"
            else:
                if feature_context.is_enabled("structure_entries"):
                    should_block = True
                    reason = "structure:below_vwap"
                elif feature_context.shadow_enabled():
                    reason = "structure:below_vwap"

    elif side < 0 and structure_cfg.get("short_below_vwap_required"):
        if vwap_values is None or idx >= len(vwap_values):
            if feature_context.is_enabled("structure_entries"):
                should_block = True
                reason = "structure:vwap_missing"
            elif feature_context.shadow_enabled():
                reason = "structure:vwap_missing"
        elif price >= vwap_values[idx]:
            if feature_context.is_enabled("structure_entries"):
                should_block = True
                reason = "structure:above_vwap_short"
            elif feature_context.shadow_enabled():
                reason = "structure:above_vwap_short"

    # Trend expressions
    if side > 0:
        expr = structure_cfg.get("long_trend_expr")
        if expr:
            result = _eval_expression(expr, idx, ema_fast_values, ema_slow_values, vwap_values, vwap_slope_values, price_vs_vwap, selected_prob)
            if result is False:
                if feature_context.is_enabled("structure_entries"):
                    should_block = True
                    reason = "structure:long_trend"
                elif feature_context.shadow_enabled():
                    reason = "structure:long_trend"
    elif side < 0:
        expr = structure_cfg.get("short_trend_expr")
        if expr:
            result = _eval_expression(expr, idx, ema_fast_values, ema_slow_values, vwap_values, vwap_slope_values, price_vs_vwap, selected_prob)
            if result is False:
                if feature_context.is_enabled("structure_entries"):
                    should_block = True
                    reason = "structure:short_trend"
                elif feature_context.shadow_enabled():
                    reason = "structure:short_trend"

    return should_block, reason


def _vwap_reclaim_allows(
    idx: int,
    selected_prob: float,
    structure_cfg: Dict[str, Any],
    above_vwap_bool: np.ndarray,
    vwap_slope_values: Optional[np.ndarray],
    ema_fast_values: Optional[np.ndarray],
    ema_slow_values: Optional[np.ndarray],
) -> Tuple[bool, str]:
    """
    Check if VWAP reclaim criteria are met.
    """
    cfg = structure_cfg.get("vwap_reclaim_long", {})
    if not cfg:
        return False, "structure:vwap_reclaim_disabled"

    required_closes = int(cfg.get("min_closes_above_vwap", 0) or 0)
    if required_closes > 0:
        start = max(0, idx - required_closes)
        closes = above_vwap_bool[start:idx]
        if closes.size < required_closes or closes.sum() < required_closes:
            return False, "structure:reclaim_closes"

    min_slope = cfg.get("min_vwap_slope")
    if min_slope is not None and vwap_slope_values is not None:
        slope_val = vwap_slope_values[idx]
        if not (np.isfinite(slope_val) and slope_val >= min_slope):
            return False, "structure:reclaim_slope"

    require_fast = bool(cfg.get("require_fast_over_slow", False))
    if require_fast and ema_fast_values is not None and ema_slow_values is not None:
        fast_val = ema_fast_values[idx] if idx < len(ema_fast_values) else np.nan
        slow_val = ema_slow_values[idx] if idx < len(ema_slow_values) else np.nan
        if not (np.isfinite(fast_val) and np.isfinite(slow_val) and fast_val > slow_val):
            return False, "structure:reclaim_ema"

    prob_min = cfg.get("p_long_min")
    if prob_min is not None and not (selected_prob >= prob_min):
        return False, "structure:reclaim_prob"

    return True, "vwap_reclaim"


def evaluate_microstructure_gates(
    idx: int,
    feature_context: FeatureContext,
    atr_ticks_values: Optional[np.ndarray],
    trades_1m_values: Optional[np.ndarray],
) -> Tuple[bool, str]:
    """
    Evaluate microstructure gates.

    Args:
        idx: Current bar index
        feature_context: Feature context for flags
        atr_ticks_values: ATR ticks 5m array
        trades_1m_values: Trades 1m array

    Returns:
        Tuple of (should_block, reason)
    """
    if not feature_context.should_evaluate("microstructure"):
        return False, ""

    if feature_context.kill_switch:
        return False, "kill_switch_active"

    cfg = feature_context.config_for("microstructure")
    min_atr = cfg.get("min_atr_ticks_5m")
    min_trades = cfg.get("min_tick_trades_1m")

    should_block = False
    reason = ""

    if min_atr is not None and atr_ticks_values is not None:
        atr_val = atr_ticks_values[idx] if idx < len(atr_ticks_values) else np.nan
        if not (np.isfinite(atr_val) and atr_val >= min_atr):
            if feature_context.is_enabled("microstructure"):
                should_block = True
                reason = "micro:atr"
            elif feature_context.shadow_enabled():
                reason = "micro:atr"

    if min_trades is not None and trades_1m_values is not None:
        trades_val = trades_1m_values[idx] if idx < len(trades_1m_values) else np.nan
        if not (np.isfinite(trades_val) and trades_val >= min_trades):
            if feature_context.is_enabled("microstructure"):
                should_block = True
                reason = "micro:trades"
            elif feature_context.shadow_enabled():
                reason = "micro:trades"

    return should_block, reason


def _eval_expression(
    expr: str,
    idx: int,
    ema_fast_values: Optional[np.ndarray],
    ema_slow_values: Optional[np.ndarray],
    vwap_values: Optional[np.ndarray],
    vwap_slope_values: Optional[np.ndarray],
    price_vs_vwap: np.ndarray,
    prob: float,
) -> Optional[bool]:
    """
    Safely evaluate a trend expression.
    """
    try:
        ema_fast = ema_fast_values[idx] if ema_fast_values is not None and idx < len(ema_fast_values) else np.nan
        ema_slow = ema_slow_values[idx] if ema_slow_values is not None and idx < len(ema_slow_values) else np.nan
        vwap = vwap_values[idx] if vwap_values is not None and idx < len(vwap_values) else np.nan
        vwap_slope = vwap_slope_values[idx] if vwap_slope_values is not None and idx < len(vwap_slope_values) else np.nan
        pv = price_vs_vwap[idx] if idx < len(price_vs_vwap) else np.nan

        context = {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "vwap": vwap,
            "vwap_slope": vwap_slope,
            "price": pv + vwap if np.isfinite(vwap) else np.nan,
            "close": pv + vwap if np.isfinite(vwap) else np.nan,
            "price_vs_vwap": pv,
            "prob": prob,
            "p": prob,
            "abs": abs,
            "max": max,
            "min": min,
            "math": __import__("math"),
            "np": np,
        }
        return bool(eval(expr, {"__builtins__": {}}, context))
    except Exception:
        return None