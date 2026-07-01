# backtest_threshold.py
from __future__ import annotations

"""
Backtest: probability thresholds + prop-firm risk integration (futures, contract mode).

Patches applied (production-ready):
- Wire CLI/preset trade_window_start/end **and** `exit_at_window_end` into RiskConfig.
- Fix weekend/Globex gating bug: actually blocks NEW entries on Saturdays and on Sundays before 18:00 session open.
- Keep optional prop-friendly auto-flatten when *currently in a trade* and outside window via RiskEngine flag.
- Telemetry consistency: mark `gate_tod=False` when calendar-blocked.

Additional wiring (this patch):
- NEW `signal_ui`: normalized, policy-managed sign for UI/live usage (+1=LONG, 0=FLAT, -1=SHORT).
- NEW per-row `p_buy`/`p_sell` columns (cost/ATR-aware when available; else scalar).
- NEW emit context fields for Discord/UI publishers (`emit_*` + `emit_ctx` JSON).
- FIX: flip classification requires both previous and current non-zero.
- FIX: enforce minimum lot on entries so OPEN never shows Units=0 due to flooring.
"""

import math
import json
import logging
from datetime import time as dtime
from typing import Dict, Tuple, List, Mapping, Optional, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from trading_system.runtime_engine.l3.bot.risk_engine import (
    RiskConfig,
    RiskEngine,
    ewm_realized_vol,
    vol_target_leverage,
    compute_atr,
)
from .feature_flags import FeatureContext
from .gates import evaluate_structure_gates, evaluate_microstructure_gates
from .signal_engine import process_signal_entry

from .config import (
    CONTRACT_COST as DEFAULT_CONTRACT_COST,
    ENGINE as DEFAULTS,
    PROB_BANDS as CONFIG_PROB_BANDS,
    RISK_DEFAULTS,
    CostConfig,               # legacy bps (for backward compatibility)
    ContractCostConfig,       # contract-mode costs
    InstrumentSpec,           # instrument spec (tick/point values)
)


LOGGER = logging.getLogger(__name__)


# -------------------- Utilities --------------------

def _to_datetime(s: pd.Series) -> pd.Series:
    """Robust datetime parser: returns tz-naive UTC timestamps (ns)."""
    if is_datetime64_any_dtype(s):
        # If tz-aware, convert to UTC and drop tz; else pass through
        if getattr(s.dtype, "tz", None) is not None:
            return s.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
        return s.astype("datetime64[ns]")
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_localize(None).astype("datetime64[ns]")


def _parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm))


def _grade_by_prob(p: float, side: int, bands: Dict[str, Dict[str, float]]) -> str:
    """
    Grade by model probability.
      - For LONG (side=+1): higher p is stronger
      - For SHORT (side=-1): lower p is stronger
    Returns one of: "A+", "B+", "C"
    """
    if side > 0:
        if p >= bands["long"]["A+"]:
            return "A+"
        if p >= bands["long"]["B+"]:
            return "B+"
    elif side < 0:
        if p <= bands["short"]["A+"]:
            return "A+"
        if p <= bands["short"]["B+"]:
            return "B+"
    return "C"


def _find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -------------------- Core backtest engine (contract-mode first) --------------------

def backtest_threshold_futures(
    df: pd.DataFrame,
    proba: np.ndarray,
    # Config-owned defaults (None -> hydrate from config)
    p_buy: float | None = None,
    p_sell: float | None = None,
    cost: ContractCostConfig | CostConfig | None = None,  # accept both modes
    risk: RiskConfig | None = None,
    close_col: str = "Close",
    datetime_col: str = "Datetime",
    fwd_ret_col: str | None = None,  # kept for API compat; unused in this EV-first version
    initial_equity: float | None = None,
    # Session calendar & trade window
    session_tz: str | None = None,
    trade_window_start: str | None = None,
    trade_window_end: str | None = None,
    max_trades_per_day: int | None = None,
    # Grading
    allowed_grades: tuple[str, ...] | None = None,
    prob_bands: dict[str, dict[str, float]] | None = None,
    # Drawdown circuit breaker
    enable_dd_circuit: bool | None = None,
    dd_limit: float | None = None,
    dd_resume_hysteresis: float | None = None,
    dd_disable_from_next_bar: bool | None = None,
    # Volatility targeting
    enable_vol_target: bool | None = None,
    vol_annualize_k: float | None = None,
    vol_ema_span: int | None = None,
    target_vol: float | None = None,
    pos_cap: float | None = None,
    # LLM risk officer (optional; pass-through telemetry only)
    use_llm: bool | None = None,
    llm_review_all: bool | None = None,
    llm_max_risk_bps: int | None = None,
    llm_cooldown_min: int | None = None,
    symbol: str | None = None,
    # Contract mode specifics
    instrument: InstrumentSpec | None = None,
    account_scale_usd: float | None = None,  # baseline used to translate pct-based thresholds
    # Prop/EOD trailing helpers (from CLI wrapper)
    profit_lock_usd: float | None = None,          # lock EOD peak once profit >= this USD
    near_breach_buffer_usd: float | None = 100.0,  # UI flagging only; RiskEngine has its own buffer
    # EV-aware thresholding (optional)
    target_r: float | None = None,
    policy_margin: float = 0.0,
    feature_context: FeatureContext | None = None,
    preset_config: Optional[Mapping[str, Any]] = None,
    # Context gates (optional): VWAP/EMA bias (auto-detect columns)
    enable_vwap_gate: bool = True,
    enable_ema_gate: bool = True,
    # News risk gate (optional): True means "blocked by news" at that bar
    news_block_mask: pd.Series | np.ndarray | None = None,
    # Window exit behavior
    exit_outside_window: bool = True,  # prop-friendly: force flat outside time window
    # Dual-model probabilities (optional)
    proba_long: np.ndarray | None = None,
    proba_short: np.ndarray | None = None,
    min_confidence: float | None = None,
    # Scalper gates (optional)
    min_volume_1m: float | None = None,
    max_spread_ticks: float | None = None,
    cooldown_bars: int | None = None,
    reopen_block_bars: int | None = None,
    # ATR-derived stops (optional)
    use_atr_ticks: bool | None = None,
    atr_mult: float | None = None,
    atr_lookback: int | None = None,
    stop_ticks_min: float | None = None,
    stop_ticks_max: float | None = None,
    target_ticks: float | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:

    feature_context = feature_context or FeatureContext.disabled()
    preset_config = dict(preset_config or {})
    kill_switch = feature_context.kill_switch
    shadow_mode = feature_context.shadow_enabled()
    debug_enabled = feature_context.debug_enabled()
    structure_eval = feature_context.should_evaluate("structure_entries")
    structure_enforce = feature_context.is_enabled("structure_entries")
    vwap_reclaim_eval = feature_context.should_evaluate("vwap_reclaim")
    vwap_reclaim_enforce = feature_context.is_enabled("vwap_reclaim")
    vwap_band_eval = feature_context.should_evaluate("vwap_no_trade_band")
    vwap_band_enforce = feature_context.is_enabled("vwap_no_trade_band")
    micro_eval = feature_context.should_evaluate("microstructure")
    micro_enforce = feature_context.is_enabled("microstructure")
    prob_nudge_eval = feature_context.should_evaluate("prob_slope_nudge")
    prob_nudge_enforce = feature_context.is_enabled("prob_slope_nudge")
    structure_cfg: Dict[str, Any] = {}

    # ---- hydrate defaults ----
    cost = cost or DEFAULT_CONTRACT_COST
    p_buy = DEFAULTS.p_buy if p_buy is None else p_buy
    p_sell = DEFAULTS.p_sell if p_sell is None else p_sell

    session_tz = session_tz or DEFAULTS.session_tz
    if not session_tz:
        raise ValueError("session_tz must be set (e.g., 'America/New_York').")

    trade_window_start = trade_window_start or DEFAULTS.trade_window_start
    trade_window_end = trade_window_end or DEFAULTS.trade_window_end
    max_trades_per_day = max_trades_per_day or DEFAULTS.max_trades_per_day

    # In contract mode, equity is a notional ledger seeded at account_scale_usd
    account_scale_usd = float(DEFAULTS.account_scale_usd if account_scale_usd is None else account_scale_usd)
    initial_equity = account_scale_usd if initial_equity is None else float(initial_equity)

    allowed_grades = allowed_grades or DEFAULTS.allowed_grades
    allowed_grades = tuple(
        g for g in (
            str(x).strip() for x in allowed_grades
            if x is not None
        ) if g
    ) or tuple(
        g for g in (
            str(x).strip() for x in DEFAULTS.allowed_grades
            if x is not None
        ) if g
    )
    allowed_grade_set = {g.upper() for g in allowed_grades}
    prob_bands = prob_bands or CONFIG_PROB_BANDS

    enable_dd_circuit = DEFAULTS.enable_dd_circuit if enable_dd_circuit is None else enable_dd_circuit
    dd_limit = DEFAULTS.dd_limit if dd_limit is None else dd_limit
    dd_resume_hysteresis = DEFAULTS.dd_resume_hysteresis if dd_resume_hysteresis is None else dd_resume_hysteresis
    dd_disable_from_next_bar = DEFAULTS.dd_disable_from_next_bar if dd_disable_from_next_bar is None else dd_disable_from_next_bar

    enable_vol_target = DEFAULTS.enable_vol_target if enable_vol_target is None else enable_vol_target
    target_vol = DEFAULTS.target_vol if target_vol is None else target_vol
    vol_ema_span = DEFAULTS.vol_ema_span if vol_ema_span is None else vol_ema_span
    vol_annualize_k = DEFAULTS.vol_annualize_k if vol_annualize_k is None else vol_annualize_k
    pos_cap = DEFAULTS.pos_cap if pos_cap is None else pos_cap

    use_llm = DEFAULTS.use_llm if use_llm is None else use_llm
    llm_review_all = DEFAULTS.llm_review_all if llm_review_all is None else llm_review_all
    llm_max_risk_bps = DEFAULTS.llm_max_risk_bps if llm_max_risk_bps is None else llm_max_risk_bps
    llm_cooldown_min = DEFAULTS.llm_cooldown_min if llm_cooldown_min is None else llm_cooldown_min
    symbol = symbol or DEFAULTS.symbol

    min_confidence = float(min_confidence) if min_confidence is not None else None
    min_volume_1m = float(min_volume_1m) if min_volume_1m is not None else None
    max_spread_ticks = float(max_spread_ticks) if max_spread_ticks is not None else None
    cooldown_bars = int(cooldown_bars) if cooldown_bars not in (None, 0) else None
    reopen_block_bars = int(reopen_block_bars) if reopen_block_bars not in (None, 0) else None
    use_atr_ticks = bool(use_atr_ticks) if use_atr_ticks is not None else False
    atr_mult = float(atr_mult) if atr_mult is not None else None
    atr_lookback = int(atr_lookback) if atr_lookback not in (None, 0) else None
    stop_ticks_min = float(stop_ticks_min) if stop_ticks_min is not None else None
    stop_ticks_max = float(stop_ticks_max) if stop_ticks_max is not None else None
    target_ticks = float(target_ticks) if target_ticks is not None else None

    # NEW: configurable minimum lot on entry (avoid floor->0 on OPEN)
    min_units_on_entry = float(getattr(DEFAULTS, "min_units_on_entry", 1.0))

    # Risk config hydrate + ensure engine sees the requested window/behavior
    risk = risk or RiskConfig(**RISK_DEFAULTS)
    tw_start = _parse_hhmm(trade_window_start)
    tw_end = _parse_hhmm(trade_window_end)
    risk.trade_window_start = tw_start
    risk.trade_window_end = tw_end
    # Keep RiskEngine's internal hard exit aligned with function arg
    if hasattr(risk, "exit_at_window_end"):
        risk.exit_at_window_end = bool(exit_outside_window)

    # Feature-aware risk overrides
    if feature_context.is_enabled("max_hold"):
        max_hold_bars_val = preset_config.get("max_hold_bars")
        if max_hold_bars_val not in (None, "", False):
            try:
                risk.max_hold_bars = int(max_hold_bars_val)
            except Exception:
                LOGGER.debug("Invalid max_hold_bars=%r; keeping RiskConfig default", max_hold_bars_val)
    if hasattr(risk, "daily_lock_enabled"):
        setattr(risk, "daily_lock_enabled", bool(feature_context.is_enabled("daily_lock")))
    else:
        try:
            risk.daily_lock_enabled = bool(feature_context.is_enabled("daily_lock"))  # type: ignore[attr-defined]
        except Exception:
            pass

    # Instrument (default ES)
    if instrument is None:
        from .config import instrument_by_alias
        instrument = instrument_by_alias(DEFAULTS.instrument_alias)

    # ---- input normalization ----
    data = df.copy()
    if datetime_col not in data.columns:
        raise ValueError(f"'{datetime_col}' column not found.")
    data[datetime_col] = _to_datetime(data[datetime_col]).astype("datetime64[ns]")
    data = data.reset_index(drop=True)

    if close_col not in data.columns:
        raise ValueError(f"'{close_col}' column not found.")

    n = len(data)
    if n == 0:
        return data.assign(emit_ctx=None), dict(bars=0, trades=0)

    if len(proba) != n:
        raise ValueError(f"proba length {len(proba)} != df length {n}")
    data["proba"] = np.asarray(proba, dtype=float)

    # Optional news mask normalization (True = blocked)
    if news_block_mask is not None:
        if isinstance(news_block_mask, pd.Series):
            if len(news_block_mask) != n:
                raise ValueError("news_block_mask length must match df length.")
            news_block_mask_arr = news_block_mask.values.astype(bool)
        else:
            news_block_mask_arr = np.asarray(news_block_mask, dtype=bool)
            if news_block_mask_arr.shape[0] != n:
                raise ValueError("news_block_mask length must match df length.")
    else:
        news_block_mask_arr = np.zeros(n, dtype=bool)

    # Feature cols (for telemetry only; not used directly)
    _exclude = {
        datetime_col, close_col, "Open", "High", "Low", "Volume",
        "ret","proba","signal","position","pnl_gross","pnl_net","equity","day_pnl",
        "peak_equity","drawdown",
    }
    feature_cols = [c for c in data.columns if c not in _exclude and pd.api.types.is_numeric_dtype(data[c])]

    volume_col = _find_col(data, ("Volume", "volume", "VOL", "vol", "vol_1m", "volume_1m"))
    spread_col = _find_col(data, ("spread_ticks", "spread", "spread_tick", "spreadTicks"))

    # LLM telemetry (pass-through)
    llm_action = np.full(n, "", dtype=object)
    llm_conf = np.full(n, np.nan)
    llm_rr = np.full(n, np.nan)
    llm_size = np.full(n, np.nan)
    llm_rbps = np.full(n, np.nan)
    llm_reasons = np.full(n, "", dtype=object)
    llm_used = np.zeros(n, dtype=bool)
    pending_stop_type: str | None = None
    pending_stop_dist: float | None = None

    # --- ATR (optional) ---
    atr = None
    have_hilo = {"High", "Low", close_col}.issubset(set(data.columns))
    if (
        risk.atr_len is not None
        and int(risk.atr_len) > 0
        and have_hilo
    ):
        atr = compute_atr(
            data["High"].astype(float).values,
            data["Low"].astype(float).values,
            data[close_col].astype(float).values,
            length=int(risk.atr_len),
        )

    atr_ticks_series: np.ndarray | None = None
    if use_atr_ticks and have_hilo and tick_size > 0:
        atr_len_use = atr_lookback if atr_lookback is not None else (int(risk.atr_len) if getattr(risk, "atr_len", None) else 14)
        if atr_len_use and atr_len_use > 0:
            atr_points = compute_atr(
                data["High"].astype(float).values,
                data["Low"].astype(float).values,
                data[close_col].astype(float).values,
                length=int(atr_len_use),
            )
            atr_ticks_series = np.asarray(atr_points, dtype=float) / float(tick_size)

    # ---- Context gates: VWAP + EMA trend (auto-detect) ----
    vwap_col = _find_col(data, ("vwap_sess","vwap","VWAP")) if enable_vwap_gate else None
    ema20_col = _find_col(data, ("ema20","ema_20","EMA20","ema_fast","ema_21")) if enable_ema_gate else None
    ema50_col = _find_col(data, ("ema50","ema_50","EMA50","ema_slow","ema_55")) if enable_ema_gate else None
    stop_hint_col = _find_col(data, ("stop","stop_price","emit_stop","risk_stop"))
    target_hint_col = _find_col(data, ("target","target_price","emit_target","risk_target"))
    rr_col = _find_col(data, ("rr","planned_r","planned_R"))

    # ---- EV-aware thresholds (optional) ----
    # Base thresholds (scalars)
    if target_r is not None and target_r > 0:
        p_star = 1.0 / (1.0 + float(target_r))
        ev_thresh_long = p_star + float(policy_margin)
        ev_thresh_short = 1.0 - ev_thresh_long
    else:
        ev_thresh_long = float(p_buy)
        ev_thresh_short = float(p_sell)

    # Vectorized thresholds (defaults to scalar values)
    ev_long_arr = np.full(n, float(ev_thresh_long))
    ev_short_arr = np.full(n, float(ev_thresh_short))

    # Optional cost- and ATR-aware per-bar EV thresholds (only if available)
    if (
        target_r is not None and target_r > 0 and
        atr is not None and isinstance(cost, ContractCostConfig) and
        hasattr(risk, "atr_k") and getattr(risk, "atr_k") not in (None, 0)
    ):
        atr_k = float(getattr(risk, "atr_k"))
        stop_pts = np.maximum(atr_k * np.asarray(atr, dtype=float), 1e-12)
        rt_cost_usd = 2.0 * (
            float(cost.commission_per_contract)
            + float(cost.slippage_ticks_per_side) * float(instrument.tick_value)
        )
        cost_points = rt_cost_usd / float(instrument.point_value)
        cost_R = cost_points / stop_pts
        p_req = (1.0 + cost_R + float(policy_margin or 0.0)) / (1.0 + float(target_r))
        ev_long_arr = np.clip(p_req, 0.01, 0.99)
        ev_short_arr = 1.0 - ev_long_arr

    proba_arr = np.asarray(proba, dtype=float)
    if proba_arr.shape[0] != n:
        raise ValueError("proba length must match df length.")
    if "proba" not in data.columns:
        data["proba"] = proba_arr
    else:
        data["proba"] = data["proba"].astype(float)

    # Optional dual probabilities
    dual_mode = False
    if proba_long is not None:
        p_long_arr = np.asarray(proba_long, dtype=float)
        if p_long_arr.shape[0] != n:
            raise ValueError("proba_long length must match df length.")
        dual_mode = True
    elif "proba_long" in data.columns:
        p_long_arr = data["proba_long"].astype(float).values
        dual_mode = True
    else:
        p_long_arr = proba_arr.copy()

    if proba_short is not None:
        p_short_arr = np.asarray(proba_short, dtype=float)
        if p_short_arr.shape[0] != n:
            raise ValueError("proba_short length must match df length.")
        dual_mode = True
    elif "proba_short" in data.columns:
        p_short_arr = data["proba_short"].astype(float).values
        dual_mode = True
    else:
        p_short_arr = 1.0 - proba_arr
        if dual_mode:
            p_short_arr = 1.0 - p_long_arr

    if dual_mode:
        data["proba_long"] = p_long_arr
        data["proba_short"] = p_short_arr

    long_threshold_arr = ev_long_arr.copy()
    short_threshold_arr = 1.0 - ev_short_arr
    if min_confidence is not None:
        long_threshold_arr = np.maximum(long_threshold_arr, float(min_confidence))
        short_threshold_arr = np.maximum(short_threshold_arr, float(min_confidence))

    sig = np.zeros(n, dtype=int)
    p_for_grade = np.full(n, 0.5, dtype=float)
    side_proba = np.zeros(n, dtype=float)
    for i in range(n):
        p_long_val = float(p_long_arr[i]) if np.isfinite(p_long_arr[i]) else np.nan
        p_short_val = float(p_short_arr[i]) if np.isfinite(p_short_arr[i]) else np.nan
        if dual_mode:
            long_pass = np.isfinite(p_long_val) and (p_long_val >= long_threshold_arr[i])
            short_pass = np.isfinite(p_short_val) and (p_short_val >= short_threshold_arr[i])
            if long_pass and (not short_pass or (np.isfinite(p_short_val) and p_long_val >= p_short_val)):
                sig[i] = 1
                p_for_grade[i] = p_long_val if np.isfinite(p_long_val) else 0.5
                side_proba[i] = p_long_val if np.isfinite(p_long_val) else 0.0
            elif short_pass and (not long_pass or (np.isfinite(p_long_val) and p_short_val > p_long_val)):
                sig[i] = -1
                if np.isfinite(p_short_val):
                    p_for_grade[i] = 1.0 - p_short_val
                    side_proba[i] = p_short_val
                else:
                    p_for_grade[i] = p_long_val if np.isfinite(p_long_val) else 0.5
                    side_proba[i] = 0.0
            else:
                sig[i] = 0
                if np.isfinite(p_long_val):
                    p_for_grade[i] = p_long_val
                elif np.isfinite(p_short_val):
                    p_for_grade[i] = 1.0 - p_short_val
                else:
                    p_for_grade[i] = 0.5
                side_proba[i] = max(
                    np.nan_to_num(p_long_val, nan=0.0),
                    np.nan_to_num(p_short_val, nan=0.0),
                )
        else:
            if np.isfinite(p_long_val) and p_long_val >= ev_long_arr[i]:
                if min_confidence is None or p_long_val >= float(min_confidence):
                    sig[i] = 1
            elif np.isfinite(p_long_val) and p_long_val <= ev_short_arr[i]:
                short_proba_val = 1.0 - p_long_val
                if min_confidence is None or short_proba_val >= float(min_confidence):
                    sig[i] = -1
            p_for_grade[i] = p_long_val if np.isfinite(p_long_val) else 0.5
            side_proba[i] = (
                p_long_val if (sig[i] >= 0 and np.isfinite(p_long_val)) else
                (1.0 - p_long_val if np.isfinite(p_long_val) else 0.5)
            )

    # Returns for realized vol sizing
    c = data[close_col].astype(float).values
    close_values = c
    vwap_values = data[vwap_col].astype(float).values if vwap_col else None

    if structure_eval and isinstance(preset_config.get("structure_rules"), Mapping):
        raw_sr = preset_config.get("structure_rules", {})  # type: ignore[arg-type]
        try:
            structure_cfg = {
                "long_above_vwap": bool(raw_sr.get("long_above_vwap_required", False)),
                "short_below_vwap": bool(raw_sr.get("short_below_vwap_required", False)),
                "ema_fast_len": int(raw_sr.get("ema_fast")) if raw_sr.get("ema_fast") not in (None, "") else None,
                "ema_slow_len": int(raw_sr.get("ema_slow")) if raw_sr.get("ema_slow") not in (None, "") else None,
                "long_trend_expr": str(raw_sr.get("long_trend")) if raw_sr.get("long_trend") is not None else None,
                "short_trend_expr": str(raw_sr.get("short_trend")) if raw_sr.get("short_trend") is not None else None,
                "no_trade_band_ticks": int(raw_sr.get("no_trade_vwap_band_ticks", 0) or 0),
                "vwap_reclaim_long": raw_sr.get("vwap_reclaim_long") if isinstance(raw_sr.get("vwap_reclaim_long"), Mapping) else {},
            }
        except Exception as exc:
            LOGGER.debug("Failed to parse structure_rules: %s", exc)
            structure_cfg = {}
    else:
        structure_cfg = {}

    if not structure_cfg:
        structure_eval = False
        structure_enforce = False
    
    def _resolve_ema_series(length: Optional[int], fallback_col: Optional[str]) -> tuple[Optional[str], Optional[np.ndarray]]:
        col_name: Optional[str] = None
        if length and length > 0:
            candidates = (
                f"ema{length}",
                f"ema_{length}",
                f"EMA{length}",
                f"ema{length}_close",
                f"ema_{length}_close",
            )
            col_name = _find_col(data, candidates)
            if col_name is None and fallback_col:
                if length in (20, 21) and fallback_col:
                    col_name = fallback_col
                elif length in (50, 55) and fallback_col:
                    col_name = fallback_col
        elif fallback_col:
            col_name = fallback_col

        values: Optional[np.ndarray]
        if col_name:
            try:
                values = data[col_name].astype(float).values
            except Exception:
                values = None
        else:
            values = None
        return col_name, values

    ema_fast_values: Optional[np.ndarray] = None
    ema_slow_values: Optional[np.ndarray] = None
    ema_fast_col_struct: Optional[str] = None
    ema_slow_col_struct: Optional[str] = None
    if structure_eval:
        ema_fast_col_struct, ema_fast_values = _resolve_ema_series(structure_cfg.get("ema_fast_len"), ema20_col)
        ema_slow_col_struct, ema_slow_values = _resolve_ema_series(structure_cfg.get("ema_slow_len"), ema50_col)

    vwap_slope_values: Optional[np.ndarray] = None
    if vwap_values is not None:
        slope_col = _find_col(data, ("vwap_slope","slope_vwap","vwap_slope_5","vwap_slope_short"))
        if slope_col:
            try:
                vwap_slope_values = data[slope_col].astype(float).values
            except Exception:
                vwap_slope_values = None
        if vwap_slope_values is None:
            vwap_slope_values = np.zeros(n, dtype=float)
            vwap_slope_values[1:] = np.subtract(vwap_values[1:], vwap_values[:-1])

    if vwap_values is not None:
        price_vs_vwap = np.subtract(close_values, vwap_values)
        above_vwap_bool = np.where(np.isfinite(price_vs_vwap), price_vs_vwap > 0, False)
        below_vwap_bool = np.where(np.isfinite(price_vs_vwap), price_vs_vwap < 0, False)
    else:
        price_vs_vwap = np.full(n, np.nan)
        above_vwap_bool = np.zeros(n, dtype=bool)
        below_vwap_bool = np.zeros(n, dtype=bool)

    if structure_eval:
        structure_cfg["ema_fast_col"] = ema_fast_col_struct
        structure_cfg["ema_slow_col"] = ema_slow_col_struct
        structure_cfg["ema_fast_values"] = ema_fast_values
        structure_cfg["ema_slow_values"] = ema_slow_values
        structure_cfg["vwap_values"] = vwap_values
        structure_cfg["vwap_slope_values"] = vwap_slope_values
        structure_cfg["price_vs_vwap"] = price_vs_vwap
        structure_cfg["above_vwap_bool"] = above_vwap_bool
        structure_cfg["below_vwap_bool"] = below_vwap_bool

    vwap_band_cfg = feature_context.config_for("vwap_no_trade_band") if vwap_band_eval else {}
    try:
        vwap_band_ticks = int(vwap_band_cfg.get("band_ticks", 0) or 0)
    except Exception:
        vwap_band_ticks = 0
    if not vwap_band_ticks and structure_cfg.get("no_trade_band_ticks"):
        try:
            vwap_band_ticks = int(structure_cfg.get("no_trade_band_ticks", 0) or 0)
        except Exception:
            vwap_band_ticks = 0
    if vwap_values is None or tick_size <= 0 or vwap_band_ticks <= 0:
        vwap_band_eval = False
        vwap_band_enforce = False
        vwap_band_ticks = 0
    vwap_band_cfg_resolved = {"band_ticks": max(0, vwap_band_ticks)}

    micro_cfg = feature_context.config_for("microstructure") if micro_eval else {}

    def _to_optional_float(value: Any) -> Optional[float]:
        try:
            if value in (None, "", False):
                return None
            return float(value)
        except Exception:
            return None

    micro_min_atr = _to_optional_float((micro_cfg or {}).get("min_atr_ticks_5m"))
    micro_min_trades = _to_optional_float((micro_cfg or {}).get("min_tick_trades_1m"))
    atr_ticks_values: Optional[np.ndarray] = None
    trades_1m_values: Optional[np.ndarray] = None
    if micro_eval and (micro_min_atr is None and micro_min_trades is None):
        micro_eval = False
        micro_enforce = False
    if micro_eval:
        atr_col_candidates = ("atr5m_ticks","atr_5m_ticks","ATR5m_ticks","atr_ticks_5m","atr_ticks")
        atr_ticks_col = _find_col(data, atr_col_candidates)
        if atr_ticks_col:
            try:
                atr_ticks_values = data[atr_ticks_col].astype(float).values
            except Exception:
                atr_ticks_values = None
        trade_col_candidates = ("trades1m","trades_1m","tick_trades_1m","tick_trades","trades_last_1m")
        trades_col = _find_col(data, trade_col_candidates)
        if trades_col:
            try:
                trades_1m_values = data[trades_col].astype(float).values
            except Exception:
                trades_1m_values = None
        if atr_ticks_values is None and micro_min_atr is not None:
            micro_eval = False
            micro_enforce = False
            if debug_enabled:
                LOGGER.debug("Microstructure ATR ticks column missing; disabling microstructure gate")
        if trades_1m_values is None and micro_min_trades is not None:
            micro_min_trades = None


    prob_slope_cfg = feature_context.config_for("prob_slope_nudge") if prob_nudge_eval else {}
    prob_short_params: Dict[str, Any] = {}
    prob_long_params: Dict[str, Any] = {}
    if prob_nudge_eval:
        if isinstance(prob_slope_cfg.get("short"), Mapping):
            prob_short_params = dict(prob_slope_cfg.get("short", {}))  # type: ignore[arg-type]
        if isinstance(prob_slope_cfg.get("long"), Mapping):
            prob_long_params = dict(prob_slope_cfg.get("long", {}))  # type: ignore[arg-type]
        if not prob_short_params and not prob_long_params:
            prob_nudge_eval = False
            prob_nudge_enforce = False

    def _normalize_prob_params(raw: Dict[str, Any]) -> Dict[str, Any]:
        if not raw:
            return {}
        return {
            "p_min": _to_optional_float(raw.get("p_min")),
            "p_max": _to_optional_float(raw.get("p_max")),
            "dp_last2_min": _to_optional_float(raw.get("dp_last2_min")),
            "size_mult": _to_optional_float(raw.get("size_mult")) or 1.0,
            "require_all_gates": bool(raw.get("require_all_gates", False)),
        }

    prob_short_params = _normalize_prob_params(prob_short_params)
    prob_long_params = _normalize_prob_params(prob_long_params)
    if prob_nudge_eval and not prob_short_params and not prob_long_params:
        prob_nudge_eval = False
        prob_nudge_enforce = False

    gate_structure = np.ones(n, dtype=bool)
    gate_micro = np.ones(n, dtype=bool)
    gate_vwap_band = np.ones(n, dtype=bool)
    prob_slope_applied = np.zeros(n, dtype=bool)
    debug_notes: Optional[List[List[str]]] = [[] for _ in range(n)] if debug_enabled else None
    shadow_notes: Optional[List[List[str]]] = [[] for _ in range(n)] if shadow_mode else None

    def _log_debug(idx: int, message: str) -> None:
        if debug_notes is not None and message:
            debug_notes[idx].append(message)

    def _log_shadow(idx: int, message: str) -> None:
        if shadow_notes is not None and message:
            shadow_notes[idx].append(message)

    def _eval_expression(expr: Optional[str], context: Dict[str, Any]) -> Optional[bool]:
        if not expr:
            return None
        try:
            return bool(eval(expr, {"__builtins__": {}}, context))
        except Exception as exc:  # pragma: no cover - debug aid
            if debug_enabled:
                LOGGER.debug("Failed to evaluate expression '%s': %s", expr, exc)
            return None

    # VWAP reclaim logic moved to gates.py

    # Structure gate logic moved to gates.py and signal_engine.py


    def _vwap_band_should_block(idx: int, price: float, vwap_val: Optional[float]) -> tuple[bool, str]:
        if not vwap_band_eval or vwap_val is None or tick_size <= 0:
            return False, ""
        distance = abs(price - vwap_val) / float(tick_size)
        if distance <= vwap_band_ticks:
            return True, "vwap_band"
        return False, ""

    def _prob_slope_adjustment(idx: int, side: int, selected_prob: float) -> tuple[bool, float, str]:
        if not prob_nudge_eval:
            return False, 1.0, ""
        cfg: Dict[str, Any]
        if side < 0 and prob_short_params:
            cfg = prob_short_params
        elif side > 0 and prob_long_params:
            cfg = prob_long_params
        else:
            return False, 1.0, ""

        p_min = cfg.get("p_min")
        p_max = cfg.get("p_max")
        if p_min is not None and selected_prob < p_min:
            return False, 1.0, ""
        if p_max is not None and selected_prob >= p_max:
            return False, 1.0, ""
        if idx < 2:
            return False, 1.0, ""
        prev_prob = side_proba[idx - 2]
        if not np.isfinite(prev_prob):
            return False, 1.0, ""
        delta = selected_prob - prev_prob
        dp_min = cfg.get("dp_last2_min")
        if dp_min is not None and delta < dp_min:
            return False, 1.0, ""
        size_mult = cfg.get("size_mult", 1.0)
        try:
            size_mult = float(size_mult)
        except Exception:
            size_mult = 1.0
        reason = f"prob_slope:{delta:.3f}"
        return True, size_mult, reason




    r_past = np.zeros(n)
    if n > 1:
        r_past[1:] = np.diff(np.log(c))

    # Session tz & dates
    dt_local = pd.to_datetime(data[datetime_col], errors="coerce", utc=True).dt.tz_convert(session_tz)
    local_date = dt_local.dt.date
    local_time = dt_local.dt.time

    # Grading
    grade_on_signal = np.full(n, "", dtype=object)
    rejected_grade = np.zeros(n, dtype=bool)
    entry_reason = np.full(n, "", dtype=object)
    entry_planned_r = np.full(n, np.nan)
    entry_tags = np.full(n, "", dtype=object)

    # Vol-target sizing (units_scale). Base on realized price vol; round to int contracts later.
    units_scale = np.ones(n, dtype=float)
    if enable_vol_target and (target_vol is not None):
        vol = ewm_realized_vol(r_past, span=int(vol_ema_span), annualize_k=float(vol_annualize_k))
        lev = vol_target_leverage(
            vol, target_vol=float(target_vol), max_leverage=float(pos_cap), ema_span=int(vol_ema_span)
        )
        units_scale = np.nan_to_num(lev, nan=0.0, posinf=float(pos_cap), neginf=0.0)

    # --------- State ---------
    equity = float(initial_equity)       # notional ledger (account_scale_usd)
    # EOD trailing basis + optional profit lock baseline
    eod_peak_equity = equity
    if profit_lock_usd is not None:
        eod_peak_equity = max(eod_peak_equity, equity + float(profit_lock_usd))
    intraday_peak_equity = equity  # for reporting only
    prev_day = local_date.iloc[0] if n else None

    tick_val = instrument.tick_value
    tick_size = instrument.tick_size
    point_val = instrument.point_value
    closes = data[close_col].astype(float).values
    highs = data["High"].astype(float).values if have_hilo else None
    lows  = data["Low"].astype(float).values  if have_hilo else None

    engine = RiskEngine(
        risk,
        enable_dd_circuit=bool(enable_dd_circuit),
        dd_limit=float(dd_limit),
        dd_resume_hysteresis=float(dd_resume_hysteresis),
        dd_disable_from_next_bar=bool(dd_disable_from_next_bar),
        max_trades_per_day=int(max_trades_per_day),
    )
    engine.state.start_equity_today = equity

    # --------- Outputs ---------
    position = np.zeros(n, dtype=float)      # integer contracts (float dtype for vector ops)
    pnl_gross = np.zeros(n)
    costs = np.zeros(n)
    pnl_net = np.zeros(n)
    equities = np.zeros(n)
    day_pnl = np.zeros(n)
    within_window = np.zeros(n, dtype=bool)
    trades_used = np.zeros(n, dtype=int)
    hit_trade_limit = np.zeros(n, dtype=bool)
    paused_dd = np.zeros(n, dtype=bool)
    trail_anchor = np.full(n, np.nan)
    units_taken = np.zeros(n, dtype=float)
    stop_price_series = np.full(n, np.nan)
    risk_target_series = np.full(n, np.nan)

    # stop flags
    stopped_day = np.zeros(n, dtype=bool)
    stopped_trail = np.zeros(n, dtype=bool)
    stopped_trade = np.zeros(n, dtype=bool)
    stopped_prob = np.zeros(n, dtype=bool)
    stopped_trail_profit = np.zeros(n, dtype=bool)
    stopped_be = np.zeros(n, dtype=bool)
    stopped_atr = np.zeros(n, dtype=bool)
    stopped_window = np.zeros(n, dtype=bool)  # flattened because outside time window
    cooldown_active = np.zeros(n, dtype=bool)
    prop_breached = np.zeros(n, dtype=bool)
    prop_near_breach = np.zeros(n, dtype=bool)

    # gates telemetry
    gate_prob = np.zeros(n, dtype=bool)
    gate_vwap = np.zeros(n, dtype=bool)
    gate_trend = np.zeros(n, dtype=bool)
    gate_tod = np.zeros(n, dtype=bool)
    gate_news = np.zeros(n, dtype=bool)
    gate_volume = np.ones(n, dtype=bool)
    gate_spread = np.ones(n, dtype=bool)
    gate_cooldown = np.ones(n, dtype=bool)

    # grading counters
    attempts_Ap = attempts_Bp = attempts_C = 0
    taken_Ap = taken_Bp = 0

    cooldown_remaining = 0
    block_long_remaining = 0
    block_short_remaining = 0

    # ---- MAIN LOOP ----
    for i in range(n):
        # EOD trailing baseline maintenance
        day_i = local_date.iloc[i]
        if i > 0 and day_i != prev_day:
            eod_peak_equity = max(eod_peak_equity, equity)
            if profit_lock_usd is not None:
                eod_peak_equity = max(eod_peak_equity, initial_equity + float(profit_lock_usd))
            prev_day = day_i

        peak_for_dd = eod_peak_equity

        # ---- engine pre-bar ----
        flags = engine.begin_bar(
            equity=float(equity),
            peak_equity=float(peak_for_dd),
            local_date=local_date.iloc[i],
            local_time=local_time.iloc[i],
        )
        within_window[i] = flags["within_window"]
        paused_dd[i] = flags["paused_dd"]
        stopped_trail[i] = flags["stopped_trailing"]
        stopped_day[i] = flags["stopped_day"]
        cooldown_active[i] = flags["cooldown_active"]
        prop_breached[i] = flags.get("prop_breached", False)
        prop_near_breach[i] = flags.get("prop_near_breach", False)

        force_flat_now = False
        if exit_outside_window and engine.state.current_pos != 0 and not flags["within_window"]:
            force_flat_now = True
            stopped_window[i] = True

        # Futures calendar: block Saturday entirely; block Sunday until 18:00 local session time
        weekday = local_date.iloc[i].weekday()  # 0=Mon ... 6=Sun
        cal_block_now = (weekday == 5) or (weekday == 6 and local_time.iloc[i] < dtime(18, 0))

        # ---- Context gates ----
        raw_p_now = p_for_grade[i]
        p_now = float(raw_p_now) if np.isfinite(raw_p_now) else 0.5
        selected_prob = float(side_proba[i]) if np.isfinite(side_proba[i]) else 0.0

        # raw desired side from thresholds (before gates)
        desired = int(np.clip(sig[i], -risk.max_position, risk.max_position))
        side = 1 if desired > 0 else (-1 if desired < 0 else 0)
        size_multiplier = 1.0
        prob_reason = ""

        # Debug flags: print gate reasons, active trade window, vwap band status
        if debug_enabled:
            _log_debug(i, f"gates: tod={int(within_window[i])}, vwap={int(gate_vwap[i])}, trend={int(gate_trend[i])}, volume={int(gate_volume[i])}, spread={int(gate_spread[i])}, cooldown={int(gate_cooldown[i])}")
            if not within_window[i]:
                _log_debug(i, "NO-TRADE: outside trade window")

        gate_prob[i] = bool(sig[i])
        cooldown_active_now = cooldown_remaining > 0
        block_long_active = block_long_remaining > 0
        block_short_active = block_short_remaining > 0

        # VWAP gate uses desired side
        if vwap_col is not None and close_col in data.columns:
            if side > 0:
                gate_vwap[i] = bool(data[close_col].iloc[i] > data[vwap_col].iloc[i])
            elif side < 0:
                gate_vwap[i] = bool(data[close_col].iloc[i] < data[vwap_col].iloc[i])
            else:
                gate_vwap[i] = True
        else:
            gate_vwap[i] = True

        # Trend gate uses desired side
        if ema20_col and ema50_col:
            if side > 0:
                gate_trend[i] = bool(data[ema20_col].iloc[i] > data[ema50_col].iloc[i])
            elif side < 0:
                gate_trend[i] = bool(data[ema20_col].iloc[i] < data[ema50_col].iloc[i])
            else:
                gate_trend[i] = True
        else:
            gate_trend[i] = True

        # Time-of-day window (from engine)
        gate_tod[i] = within_window[i]
        if cal_block_now:
            gate_tod[i] = False

        # News gate (True means OK to trade)
        gate_news[i] = not bool(news_block_mask_arr[i])

        vol_pass = True
        if min_volume_1m is not None and volume_col:
            try:
                vol_val = float(data[volume_col].iloc[i])
            except Exception:
                vol_val = math.nan
            vol_pass = bool(np.isfinite(vol_val) and vol_val >= float(min_volume_1m))
            gate_volume[i] = vol_pass
        else:
            gate_volume[i] = True

        spread_pass = True
        if max_spread_ticks is not None:
            if spread_col:
                try:
                    spread_val = float(data[spread_col].iloc[i])
                except Exception:
                    spread_val = math.nan
            elif have_hilo and tick_size > 0:
                spread_val = (float(data["High"].iloc[i]) - float(data["Low"].iloc[i])) / float(tick_size)
            else:
                spread_val = math.nan
            spread_pass = bool(np.isfinite(spread_val) and spread_val <= float(max_spread_ticks))
            gate_spread[i] = spread_pass
        else:
            gate_spread[i] = True

        cooldown_pass = True
        if (cooldown_bars or reopen_block_bars):
            blocked = False
            if cooldown_active_now:
                blocked = True
            elif side > 0 and block_long_active:
                blocked = True
            elif side < 0 and block_short_active:
                blocked = True
            cooldown_pass = not blocked
            gate_cooldown[i] = cooldown_pass
        else:
            gate_cooldown[i] = True

        # Apply context gates on NEW entries only
        entering = (engine.state.current_pos == 0 and desired != 0)
        attempted_entry = entering  # for telemetry
        if cal_block_now and entering:
            desired = 0
            entering = False
        feature_gate_reasons: List[str] = []
        if entering:
            gates_ok = gate_prob[i] and gate_vwap[i] and gate_trend[i] and gate_tod[i] and gate_news[i]
            if min_volume_1m is not None and volume_col:
                gates_ok = gates_ok and gate_volume[i]
            if max_spread_ticks is not None:
                gates_ok = gates_ok and gate_spread[i]
            if (cooldown_bars or reopen_block_bars):
                gates_ok = gates_ok and gate_cooldown[i]

            price_now = float(c[i])
            vwap_val_i = vwap_values[i] if vwap_values is not None and i < len(vwap_values) else None
            if vwap_val_i is not None and not np.isfinite(vwap_val_i):
                vwap_val_i = None

            if structure_eval and gates_ok:
                allow_entry, reason, size_multiplier = process_signal_entry(
                    idx=i,
                    desired=desired,
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
                    side_proba=side_proba,
                    atr_ticks_values=atr_ticks_values,
                    trades_1m_values=trades_1m_values,
                )
                if not allow_entry:
                    if structure_enforce:
                        gates_ok = False
                        gate_structure[i] = False
                        feature_gate_reasons.append(reason or "structure")
                    elif shadow_mode:
                        _log_shadow(i, reason or "structure:block")
                else:
                    gate_structure[i] = True
                    # Apply prob_slope_nudge size multiplier if applicable
                    if size_multiplier != 1.0:
                        size_multiplier *= size_multiplier
            else:
                if not structure_eval:
                    gate_structure[i] = True

            # Microstructure gates are now handled in process_signal_entry
            gate_micro[i] = True

            if vwap_band_eval and gates_ok:
                should_block, reason = _vwap_band_should_block(i, price_now, vwap_val_i)
                if should_block:
                    if vwap_band_enforce:
                        gates_ok = False
                        gate_vwap_band[i] = False
                        feature_gate_reasons.append(reason or "vwap_band")
                    elif shadow_mode:
                        _log_shadow(i, reason or "vwap_band:block")
                else:
                    gate_vwap_band[i] = True
            else:
                if not vwap_band_eval:
                    gate_vwap_band[i] = True

            if not gates_ok:
                if feature_gate_reasons and debug_enabled:
                    for reason in feature_gate_reasons:
                        _log_debug(i, reason)
                desired = 0
                entering = False
                if feature_gate_reasons:
                    entry_reason[i] = feature_gate_reasons[0].upper()
            else:
                if prob_nudge_eval and desired != 0:
                    apply_nudge, mult, prob_reason = _prob_slope_adjustment(i, side, selected_prob)
                    if apply_nudge:
                        if prob_nudge_enforce:
                            size_multiplier *= mult
                            prob_slope_applied[i] = True
                            if prob_reason and debug_enabled:
                                _log_debug(i, prob_reason)
                        elif shadow_mode:
                            _log_shadow(i, prob_reason or "prob_slope_nudge")

        if debug_enabled:
            _log_debug(
                i,
                f"gates:prob={int(gate_prob[i])} vwap={int(gate_vwap[i])} trend={int(gate_trend[i])} tod={int(gate_tod[i])} news={int(gate_news[i])}",
            )

        # Grade gating on new entries
        grade = ""
        if desired != 0:
            grade = _grade_by_prob(p_now, side, prob_bands)
            grade_on_signal[i] = grade

        assessment = None
        if entering:
            def _flt(val):
                try:
                    fv = float(val)
                except (TypeError, ValueError):
                    return None
                return fv if np.isfinite(fv) else None

            stop_hint_val = _flt(data[stop_hint_col].iloc[i]) if stop_hint_col else None
            target_hint_val = _flt(data[target_hint_col].iloc[i]) if target_hint_col else None
            planned_r_hint = _flt(data[rr_col].iloc[i]) if rr_col else None
            vwap_val = _flt(data[vwap_col].iloc[i]) if vwap_col else None
            ema20_val = _flt(data[ema20_col].iloc[i]) if ema20_col else None
            ema50_val = _flt(data[ema50_col].iloc[i]) if ema50_col else None
            atr_val = _flt(atr[i]) if atr is not None else None
            price_now = float(c[i])
            meta = {
                "price": price_now,
                "vwap": vwap_val,
                "price_vs_vwap": (price_now - vwap_val) if (vwap_val is not None) else None,
                "ema20": ema20_val,
                "ema50": ema50_val,
                "session_tz": session_tz,
                "tick_size": tick_size,
                "planned_R": planned_r_hint,
                "atr": atr_val,
            }

            assessment = engine.assess_entry(
                ts=dt_local.iloc[i].to_pydatetime(),
                side=side,
                proba_long=p_now,
                grade=grade,
                entry_price=price_now,
                stop_hint=stop_hint_val,
                target_hint=target_hint_val,
                meta=meta,
            )

            entry_reason[i] = assessment.reason
            entry_planned_r[i] = assessment.planned_R if assessment.planned_R is not None else np.nan
            tags_payload: Dict[str, Any] = dict(assessment.tags or {})
            if prob_slope_applied[i]:
                tags_payload["prob_slope_nudge"] = prob_reason or "applied"
            if tags_payload:
                try:
                    entry_tags[i] = json.dumps(tags_payload)
                except TypeError:
                    entry_tags[i] = str(tags_payload)
            else:
                entry_tags[i] = ""

            effective_grade = tags_payload.get("effective_grade") if tags_payload else None
            if effective_grade and effective_grade != grade:
                grade = effective_grade
                grade_on_signal[i] = grade

            grade_norm = (grade or "").strip().upper()
            if grade_norm == "A+":   attempts_Ap += 1
            elif grade_norm == "B+": attempts_Bp += 1
            else:                    attempts_C  += 1

            if not assessment.allow or grade_norm not in allowed_grade_set:
                desired = 0
                rejected_grade[i] = True
                entering = False
                pending_stop_type = None
                pending_stop_dist = None
            else:
                if assessment.stop is not None and tick_size > 0:
                    stop_price_series[i] = float(assessment.stop)
                    ticks = abs(float(assessment.stop) - price_now) / float(tick_size)
                    pending_stop_type = "TICKS"
                    pending_stop_dist = float(ticks)
                if use_atr_ticks and atr_ticks_series is not None and tick_size > 0:
                    atr_ticks_val = float(atr_ticks_series[i]) if i < len(atr_ticks_series) else math.nan
                    if np.isfinite(atr_ticks_val):
                        if atr_mult is not None:
                            atr_ticks_val *= float(atr_mult)
                        if stop_ticks_min is not None:
                            atr_ticks_val = max(atr_ticks_val, float(stop_ticks_min))
                        if stop_ticks_max is not None:
                            atr_ticks_val = min(atr_ticks_val, float(stop_ticks_max))
                        pending_stop_type = "TICKS"
                        pending_stop_dist = float(max(atr_ticks_val, 0.0))
                        stop_offset = float(pending_stop_dist) * float(tick_size)
                        if side > 0:
                            stop_price_series[i] = price_now - stop_offset
                        elif side < 0:
                            stop_price_series[i] = price_now + stop_offset
                if assessment.target is not None:
                    risk_target_series[i] = float(assessment.target)
                elif target_ticks is not None and tick_size > 0:
                    offset = float(target_ticks) * float(tick_size)
                    if side > 0:
                        risk_target_series[i] = price_now + offset
                    elif side < 0:
                        risk_target_series[i] = price_now - offset
        elif desired != 0:
            entry_reason[i] = "SKIP"

        # Risk gates (pre-trade)
        desired = int(engine.gate_desired(int(desired)))
        if flags.get("prop_breached"):
            desired = 0
            entering = False

        # Prop-friendly hard exit if outside window
        if force_flat_now:
            desired = 0
            entering = False

        # ---- V-shape reversal detection and halt ----
        vshape_halt_now = False
        if feature_context.is_enabled("halt_on_vshape") and engine.state.current_pos != 0:
            cfg = feature_context.config_for("halt_on_vshape")
            bars = int(cfg.get("bars", 2))
            flip_p_min = float(cfg.get("flip_p_min", 0.60))

            if i >= bars:
                # Check for V-shape: recent bars show reversal pattern
                recent_prices = closes[i-bars:i+1]
                if len(recent_prices) >= bars + 1:
                    # Simple V-shape detection: price went down then up (or up then down)
                    mid_idx = bars // 2
                    first_half = recent_prices[:mid_idx+1]
                    second_half = recent_prices[mid_idx:]

                    # Check if first half is decreasing and second half is increasing (V-shape)
                    # or first half increasing and second half decreasing (inverted V)
                    first_trend = first_half[-1] < first_half[0]  # decreasing
                    second_trend = second_half[-1] > second_half[0]  # increasing

                    if first_trend and second_trend:
                        # Calculate flip probability (standalone as price change magnitude)
                        price_range = max(recent_prices) - min(recent_prices)
                        if price_range > 0:
                            flip_p = abs(recent_prices[-1] - recent_prices[0]) / price_range
                            if flip_p >= flip_p_min:
                                vshape_halt_now = True
                                if debug_enabled:
                                    _log_debug(i, f"V-SHAPE HALT: bars={bars}, flip_p={flip_p:.3f} >= {flip_p_min}")

        # ---- Exits while IN a position (engine) ----
        if engine.state.current_pos != 0 and not force_flat_now and not vshape_halt_now:
            # unrealized PnL at CURRENT bar close vs entry (USD)
            unreal = float(engine.state.current_pos) * (closes[i] - engine.state.entry_price) * float(point_val)
            out = engine.in_position_exits(
                proba=p_now,
                equity=float(equity),
                unreal_pnl_usd=unreal,
                close_i=float(closes[i]),
                atr_i=(float(atr[i]) if atr is not None else None),
                high_i=(float(highs[i]) if highs is not None else None),
                low_i=(float(lows[i]) if lows is not None else None),
            )
            if not math.isnan(out.get("trail_anchor", math.nan)):
                trail_anchor[i] = out["trail_anchor"]

            if out["exit"]:
                desired = 0
                stopped_prob[i] |= out["stopped_prob"]
                stopped_trail_profit[i] |= out["stopped_trail_profit"]
                stopped_be[i] |= out["stopped_be"]
                stopped_atr[i] |= out["stopped_atr"]
                stopped_trade[i] |= out["stopped_trade"]

                if out["equity_target"] is not None:
                    # Equity-clip stop: jump equity to target, charge exit cost now
                    delta_pos = -engine.state.current_pos
                    if isinstance(cost, ContractCostConfig):
                        per_side_cost_usd = cost.commission_per_contract + cost.slippage_ticks_per_side * tick_val
                        costs[i] += abs(delta_pos) * per_side_cost_usd
                    else:
                        per_side_cost = (cost.fee_bps + cost.slippage_bps) / 10_000.0
                        notional = abs(delta_pos) * float(closes[i]) * float(point_val)
                        costs[i] += per_side_cost * notional

                    pnl_gross[i] = float(out["equity_target"]) - equity
                    pnl_net[i] = pnl_gross[i] - costs[i]
                    equity += pnl_net[i]

                    engine.on_position_change(
                        prev_pos=engine.state.current_pos,
                        new_pos=0.0,
                        entry_price=float(closes[i]),
                        equity=float(equity),
                        atr_i=(float(atr[i]) if atr is not None else None),
                        realized_trade_pnl=pnl_net[i],
                    )
                    position[i] = 0.0
                    equities[i] = equity
                    day_pnl[i] = equity - engine.state.start_equity_today
                    trades_used[i] = engine.state.trades_today
                    stop_price_series[i] = engine.state.stop_price
                    intraday_peak_equity = max(intraday_peak_equity, equity)
                    engine.after_bar()
                    continue

        if engine.max_bars_check():
            desired = 0
            stopped_trade[i] = True

        entering = engine.state.current_pos == 0 and desired != 0

        # Telemetry: if an attempted entry was blocked solely by daily limit
        if attempted_entry and desired == 0 and flags.get("trades_left", 1) <= 0:
            hit_trade_limit[i] = True

        # ---- Position Sizing (integer contracts) ----
        desired_units = 0.0
        if desired != 0:
            cap_units = min(float(pos_cap), float(risk.max_position))

            if entering:
                # grade-based base size
                if grade == "A+":
                    base_units = cap_units
                elif grade == "B+":
                    base_units = min(2.0, cap_units)
                else:
                    base_units = min(1.0, cap_units)

                # APPLY vol-targeting scale
                base_units *= float(units_scale[i])

                # Prob-slope nudge multiplier (if any)
                base_units *= float(size_multiplier)

                # Optional LLM override
                if use_llm and not np.isnan(llm_size[i]) and llm_size[i] > 0:
                    base_units = float(min(abs(llm_size[i]), cap_units))

                desired_units = np.sign(desired) * float(base_units)

                # --- ENFORCE MINIMUM LOT ON ENTRY (avoid floor -> 0) ---
                abs_units = max(min_units_on_entry, abs(desired_units))
                desired_units = float(np.sign(desired_units) * abs_units)

            else:
                # while in position, keep current units unless reversing
                desired_units = float(
                    engine.state.current_pos if engine.state.current_pos != 0 else np.sign(desired)
                )

            # final clamp and integerize (after min-lot)
            desired_units = float(np.sign(desired_units) * min(np.floor(abs(desired_units)), cap_units))

        units_taken[i] = desired_units

        # ---- Execute position change + costs ----
        if desired_units != engine.state.current_pos:
            prev_pos_state = engine.state.current_pos
            delta_pos = desired_units - engine.state.current_pos
            if isinstance(cost, ContractCostConfig):
                per_side_cost_usd = cost.commission_per_contract + cost.slippage_ticks_per_side * tick_val
                costs[i] += abs(delta_pos) * per_side_cost_usd
            else:
                per_side_cost = (cost.fee_bps + cost.slippage_bps) / 10_000.0
                notional = abs(delta_pos) * float(closes[i]) * float(point_val)
                costs[i] += per_side_cost * notional

            engine.on_position_change(
                prev_pos=engine.state.current_pos,
                new_pos=float(desired_units),
                entry_price=float(closes[i]),
                equity=float(equity),
                atr_i=(float(atr[i]) if atr is not None else None),
                pending_stop_type=pending_stop_type,
                pending_stop_dist=pending_stop_dist,
            )
            pending_stop_type = None
            pending_stop_dist = None

            exited_now = prev_pos_state != 0 and engine.state.current_pos == 0
            if exited_now:
                if cooldown_bars:
                    cooldown_remaining = max(cooldown_remaining, int(cooldown_bars) + 1)
                if reopen_block_bars:
                    if prev_pos_state > 0:
                        block_long_remaining = max(block_long_remaining, int(reopen_block_bars) + 1)
                    elif prev_pos_state < 0:
                        block_short_remaining = max(block_short_remaining, int(reopen_block_bars) + 1)

            if entering:
                if grade == "A+": taken_Ap += 1
                elif grade == "B+": taken_Bp += 1

        # ---- PnL accrual (CONTRACT MODE) ----
        pos_now = engine.state.current_pos
        position[i] = pos_now

        if i < n - 1:
            delta_price_points = (closes[i+1] - closes[i])
        else:
            delta_price_points = 0.0

        gross_usd = float(pos_now) * delta_price_points * point_val
        pnl_gross[i] = gross_usd
        pnl_net[i] = gross_usd - costs[i]
        equity += pnl_net[i]
        equities[i] = equity
        day_pnl[i] = equity - engine.state.start_equity_today
        trades_used[i] = engine.state.trades_today
        stop_price_series[i] = engine.state.stop_price
        intraday_peak_equity = max(intraday_peak_equity, equity)

        engine.after_bar()

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        if block_long_remaining > 0:
            block_long_remaining -= 1
        if block_short_remaining > 0:
            block_short_remaining -= 1

    # -------- Assemble output (after loop) --------
    data["signal"] = sig                                  # raw threshold direction (pre-gates)
    data["signal_ui"] = np.sign(position).astype(int)

    # grade must exist before emit filtering
    data["grade"] = grade_on_signal
    data["rejected_grade"] = rejected_grade

    # --- Transitions (correct flip/close typing) ---
    prev = data["signal_ui"].shift(1).fillna(0).astype(int)
    data["emit_open"]  = (prev == 0) & (data["signal_ui"] != 0)
    data["emit_close"] = (prev != 0) & (data["signal_ui"] == 0)
    data["emit_flip"]  = (
        (prev != 0) &
        (data["signal_ui"] != 0) &
        (np.sign(prev) != np.sign(data["signal_ui"]))
    )
    data["emit_mask"]  = data["emit_open"] | data["emit_flip"] | data["emit_close"]
    # Gate opens/flips by allowed grades, but ALWAYS allow closes
    _grade_ok = (
        pd.Series(data["grade"])
        .fillna("")
        .map(lambda g: str(g).strip().upper())
        .isin(allowed_grade_set)
    )
    data["emit_mask"] &= (~(data["emit_open"] | data["emit_flip"])) | _grade_ok

    # === Emit context fields (Discord/UI) ===
    data["emit_side"]  = data["signal_ui"].map({1: "LONG", -1: "SHORT", 0: "FLAT"})
    data["emit_conf"]  = data.get("proba_adj", data["proba"])
    data["emit_entry"] = data[close_col].astype(float)
    data["emit_stop"]  = pd.to_numeric(pd.Series(stop_price_series), errors="coerce")
    # size on event bars (use units_taken to reflect executed size)
    data["emit_units"] = np.where(data["emit_open"] | data["emit_flip"], np.abs(units_taken), 0.0)

    # Gates snapshot (persist raw + emit for UI/backfills)
    data["proba_side"] = side_proba
    data["gate_prob"] = gate_prob
    data["gate_vwap"] = gate_vwap
    data["gate_trend"] = gate_trend
    data["gate_tod"] = gate_tod
    data["gate_news"] = gate_news
    data["gate_volume"] = gate_volume
    data["gate_spread"] = gate_spread
    data["gate_cooldown"] = gate_cooldown
    data["gate_structure"] = gate_structure
    data["gate_micro"] = gate_micro
    data["gate_vwap_band"] = gate_vwap_band
    data["prob_slope_nudge"] = prob_slope_applied

    if debug_notes is not None:
        data["feature_debug"] = [";".join(notes) if notes else "" for notes in debug_notes]
    if shadow_notes is not None:
        data["shadow_notes"] = [";".join(notes) if notes else "" for notes in shadow_notes]

    data["emit_gate_prob"] = gate_prob
    data["emit_gate_vwap"] = gate_vwap
    data["emit_gate_ema"] = gate_trend
    data["emit_gate_tod"] = gate_tod
    data["emit_gate_news"] = gate_news
    data["emit_gate_volume"] = gate_volume
    data["emit_gate_spread"] = gate_spread
    data["emit_gate_cooldown"] = gate_cooldown
    data["emit_gate_all"] = (
        gate_prob
        & gate_vwap
        & gate_trend
        & gate_tod
        & gate_news
        & gate_volume
        & gate_spread
        & gate_cooldown
    )

    # Thresholds (also expose plain versions)
    data["emit_p_buy"]  = ev_long_arr
    data["emit_p_sell"] = ev_short_arr
    data["p_buy"] = ev_long_arr
    data["p_sell"] = ev_short_arr

    # Exit and risk tags (pipe-joined for readability)
    _exit_cols = pd.DataFrame({
        "prob":   np.where(stopped_prob,         "prob_exit",    ""),
        "trail":  np.where(stopped_trail_profit, "trail_profit", ""),
        "be":     np.where(stopped_be,           "breakeven",    ""),
        "atr":    np.where(stopped_atr,          "atr_stop",     ""),
        "trade":  np.where(stopped_trade,        "trade_guard",  ""),
        "window": np.where(stopped_window,       "window_exit",  ""),
        "day":    np.where(stopped_day,          "day_stop",     ""),
        "dd":     np.where(paused_dd,            "dd_circuit",   ""),
        "prop":   np.where(prop_breached,        "prop_breach",  ""),
    })
    data["emit_exit_tags"] = _exit_cols.apply(lambda r: "|".join([t for t in r if t]), axis=1)

    _risk_cols = pd.DataFrame({
        "paused_dd":   np.where(paused_dd,        "dd_active",     ""),
        "near_breach": np.where(prop_near_breach, "prop_near",     ""),
        "breach":      np.where(prop_breached,    "prop_breached", ""),
    })
    data["emit_risk_tags"] = _risk_cols.apply(lambda r: "|".join([t for t in r if t]), axis=1)

    # Optional confluence tags passthrough (if present from your confluence layer)
    data["emit_conf_tags"] = data["conf_tags"] if "conf_tags" in data.columns else ""

    # Compact JSON blob for publishers that want a single field (NaN-safe)
    def _f(x):
        if x is None:
            return None
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else float(x)
        if isinstance(x, (np.integer, int)):
            return float(x)
        return x

    def _ctx_row(row):
        return {
            "side":        row["emit_side"],
            "entry":       _f(row["emit_entry"]),
            "stop":        _f(row["emit_stop"]),
            "units":       _f(row["emit_units"]),
            "confidence":  _f(row["emit_conf"]),
            "grade":       (row["grade"] or ""),
            "p_buy":       _f(row["emit_p_buy"]),
            "p_sell":      _f(row["emit_p_sell"]),
            "gates": {
                "prob":  bool(row["gate_prob"]),
                "vwap":  bool(row["gate_vwap"]),
                "ema":   bool(row["gate_trend"]),
                "tod":   bool(row["gate_tod"]),
                "news":  bool(row["gate_news"]),
                "volume": bool(row.get("gate_volume", True)),
                "spread": bool(row.get("gate_spread", True)),
                "cooldown": bool(row.get("gate_cooldown", True)),
            },
            "exit_tags":   (row["emit_exit_tags"] or ""),
            "risk_tags":   (row["emit_risk_tags"] or ""),
            "conf_tags":   (row.get("emit_conf_tags","") or ""),
        }

    data["emit_ctx"] = None
    _emit_idx = data.index[data["emit_mask"]]
    if len(_emit_idx) > 0:
        data.loc[_emit_idx, "emit_ctx"] = data.loc[_emit_idx].apply(lambda r: json.dumps(_ctx_row(r)), axis=1)

    # --- Remaining telemetry ---
    data["position"] = position
    data["pnl_gross"] = pnl_gross
    data["cost"] = costs
    data["pnl_net"] = pnl_net
    data["equity"] = equities
    data["day_pnl"] = day_pnl
    data["peak_equity"] = pd.Series(equities).cummax()
    data["drawdown"] = data["equity"] - data["peak_equity"]
    data["stopped_day"] = stopped_day
    data["stopped_trailing"] = stopped_trail
    data["stopped_trade"] = stopped_trade
    data["stopped_prob"] = stopped_prob
    data["stopped_trail_profit"] = stopped_trail_profit
    data["stopped_be"] = stopped_be
    data["stopped_atr"] = stopped_atr
    data["stopped_window"] = stopped_window
    data["within_window"] = within_window
    data["trades_today"] = trades_used
    data["hit_trade_limit"] = hit_trade_limit
    data["paused_dd"] = paused_dd
    data["trail_anchor"] = trail_anchor
    data["cooldown_active"] = cooldown_active
    data["stop_price"] = stop_price_series
    data["risk_target"] = risk_target_series
    data["prop_breached"] = prop_breached
    data["prop_near_breach"] = prop_near_breach
    data["entry_reason"] = entry_reason
    data["planned_r"] = entry_planned_r
    data["entry_tags"] = entry_tags

    # LLM audit
    data["llm_used"] = llm_used
    data["llm_action"] = llm_action
    data["llm_confidence"] = llm_conf
    data["llm_rr"] = llm_rr
    data["llm_size"] = llm_size
    data["llm_risk_bps"] = llm_rbps
    data["llm_reasons"] = llm_reasons

    # -------- Summary --------
    total_return = (equities[-1] / initial_equity) - 1.0 if initial_equity else float("nan")
    dt_index = pd.to_datetime(data[datetime_col], errors="coerce")
    days = max(1, (dt_index.iloc[-1] - dt_index.iloc[0]).days)
    years = max(1e-9, days / 365.25)
    cagr = (equities[-1] / initial_equity) ** (1.0 / years) - 1.0 if initial_equity and years > 1e-6 else float("nan")
    min_dd = float(data["drawdown"].min()) if initial_equity else float("nan")
    mdd = (abs(min_dd) / initial_equity) if initial_equity else float("nan")

    # entries + reversals (no resize inflation)
    pos_arr = np.asarray(position, dtype=float)
    sgn = np.sign(pos_arr)
    flips = (np.diff(sgn) != 0) & (sgn[1:] != 0)
    entries = (sgn[:-1] == 0) & (sgn[1:] != 0)
    trades = int(entries.sum() + flips.sum())

    sharpe = float("nan")
    if n > 10 and initial_equity:
        eq_series = pd.Series(equities)
        rets = pd.Series(pnl_net) / eq_series.shift(1).fillna(initial_equity)
        std = rets.std()
        if std and std > 0:
            sharpe = (rets.mean() / std) * math.sqrt(252)

    summary = dict(
        instrument=instrument.alias,
        tick_size=instrument.tick_size,
        tick_value=instrument.tick_value,
        point_value=point_val,

        initial_equity=initial_equity,
        final_equity=float(equities[-1]),
        total_pnl_usd=float(equities[-1] - initial_equity),
        total_return=float(total_return) if not math.isnan(total_return) else float("nan"),
        cagr=float(cagr) if not math.isnan(cagr) else float("nan"),
        max_drawdown=float(mdd) if not math.isnan(mdd) else float("nan"),
        trades=trades,
        bars=n,

        p_buy=float(p_buy), p_sell=float(p_sell),
        ev_thresh_long=float(np.median(ev_long_arr)),
        ev_thresh_short=float(np.median(ev_short_arr)),
        target_r=(float(target_r) if target_r is not None else 0.0),
        policy_margin=float(policy_margin or 0.0),

        costs_mode=("contract" if isinstance(cost, ContractCostConfig) else "equity_bps"),
        commission_per_contract=(getattr(cost, "commission_per_contract", None)),
        slippage_ticks_per_side=(getattr(cost, "slippage_ticks_per_side", None)),
        fee_bps=(getattr(cost, "fee_bps", None)),
        slippage_bps=(getattr(cost, "slippage_bps", None)),

        per_trade_stop_pct=(risk.per_trade_stop_pct or 0.0),
        per_trade_stop_usd=(risk.per_trade_stop_usd or 0.0),
        daily_loss_stop_pct=(getattr(risk, "daily_loss_stop_pct", 0.0) or 0.0),
        daily_loss_stop_usd=(getattr(risk, "daily_loss_stop_usd", 0.0) or 0.0),
        trailing_drawdown_pct=(risk.trailing_drawdown_pct or 0.0),
        proba_cut_bad=(risk.proba_cut_bad or 0.0),
        trail_profit_pct=(risk.trail_profit_pct or 0.0),
        trail_profit_activate_pct=float(risk.trail_profit_activate_pct or 0.0),

        trade_window=f"{trade_window_start}-{trade_window_end} {session_tz}",
        max_trades_per_day=max_trades_per_day,
        dd_circuit=enable_dd_circuit,
        dd_limit=dd_limit,
        dd_resume_hysteresis=dd_resume_hysteresis,

        vol_target=bool(enable_vol_target and (target_vol is not None)),
        sharpe=float(sharpe) if sharpe == sharpe else float("nan"),
        prob_cut_exits=int(stopped_prob.sum()),
        trail_profit_exits=int(stopped_trail_profit.sum()),
        stopped_be_exits=int(stopped_be.sum()),
        stopped_atr_exits=int(stopped_atr.sum()),
        stopped_window_exits=int(stopped_window.sum()),
        grade_attempts_Ap=int(attempts_Ap),
        grade_attempts_Bp=int(attempts_Bp),
        grade_attempts_C=int(attempts_C),
        grade_taken_Ap=int(taken_Ap),
        grade_taken_Bp=int(taken_Bp),
        avg_abs_units=float(np.mean(np.abs(units_taken))) if n else 0.0,
        pct_units_ge_2=float(np.mean(np.abs(units_taken) >= 2.0)) if n else 0.0,
        account_scale_usd=account_scale_usd,

        # Prop telemetry
        prop_trailing_dd_usd=(getattr(risk, "prop_trailing_dd_usd", None) or 0.0),
        prop_near_breach=float(prop_near_breach.mean()) if n else 0.0,
        prop_breaches=int(prop_breached.sum()),
        profit_lock_usd=(profit_lock_usd or 0.0),
        near_breach_buffer_usd=(near_breach_buffer_usd or 0.0),
    )

    # ---- Summarize gates for the last bar (so callers don't have to reconstruct) ----
    last = data.iloc[-1]
    max_tr_left = None
    try:
        td = int(last["trades_today"])
        max_tr_left = max(0, int(max_trades_per_day) - td)
    except Exception:
        max_tr_left = None

    summary["gates"] = {
        "in_window": bool(last["within_window"]),
        "in_trade_window": bool(last["within_window"]),  # alias used by some UIs
        "gate_vwap": bool(last["gate_vwap"]),
        "gate_ema": bool(last["gate_trend"]),            # normalize name
        "gate_volume": bool(last.get("gate_volume", True)),
        "gate_spread": bool(last.get("gate_spread", True)),
        "gate_cooldown": bool(last.get("gate_cooldown", True)),
        "gate_structure": bool(last.get("gate_structure", True)),
        "gate_micro": bool(last.get("gate_micro", True)),
        "gate_vwap_band": bool(last.get("gate_vwap_band", True)),
        "prob_slope_nudge": bool(last.get("prob_slope_nudge", False)),
        "dd_active": bool(last["paused_dd"]),
        "news_block": not bool(last["gate_news"]),       # invert to "blocked?"
        "max_trades_left": max_tr_left,
        "grade": str(last.get("grade", "") or ""),
    }

    return data, summary



__all__ = [
    "CostConfig",
    "ContractCostConfig",
    "InstrumentSpec",
    "backtest_threshold_futures",
]
