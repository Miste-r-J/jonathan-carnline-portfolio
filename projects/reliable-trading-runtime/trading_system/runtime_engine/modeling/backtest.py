#backtest.py (refactored for realism & leakage fixes)
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLOSE_COL, MAX_LOSSES_PER_DAY, VOLUME_COL
from .risk_config import RiskConfig, default_risk_config
from trading_system.runtime_engine.l3.bot.risk_engine import realized_vol, vol_target_leverage  # drawdown handled inline


def _bars_per_day(df: pd.DataFrame) -> int:
    """Estimate bars per day from datetime context."""
    if "date" in df:
        n_days = df["date"].nunique()
    elif "Datetime" in df.columns:
        n_days = pd.to_datetime(df["Datetime"]).dt.date.nunique()
    else:
        n_days = pd.to_datetime(df.index).date.nunique()
    if n_days <= 0:
        return 252
    return max(int(len(df) / n_days), 1)


def _annualize_k(df: pd.DataFrame, risk_cfg: RiskConfig) -> float:
    if not risk_cfg.annualize_from_time:
        return float(risk_cfg.limits.max_hold_bars * risk_cfg.trading_days_per_year)
    bars_per_day = risk_cfg.bars_per_day or _bars_per_day(df)
    return float(bars_per_day * risk_cfg.trading_days_per_year)


def _trade_cost_usd(
    delta_pos: float,
    tick_value: float,
    commission_per_contract: float,
    slippage_ticks_per_side: float,
) -> float:
    """
    Per-side cost model tied to contracts/ticks (more realistic than bps for futures).
    Applies cost each time position changes (entry/exit/flip).
    """
    per_side = commission_per_contract + (slippage_ticks_per_side * tick_value)
    return abs(delta_pos) * per_side


def _liquidity_cap(position: float, df: pd.DataFrame, idx: int, risk_cfg: RiskConfig) -> float:
    """Limit size to participation of recent volume; deterministic and simple."""
    if VOLUME_COL not in df.columns:
        return position
    lookback = risk_cfg.liquidity.lookback_bars
    recent = df[VOLUME_COL].iloc[max(0, idx - lookback + 1): idx + 1]
    avg_vol = recent.mean()
    if avg_vol <= 0:
        return position
    cap = risk_cfg.liquidity.participation_limit * avg_vol
    return float(np.sign(position) * min(abs(position), cap))


def simple_strategy_equity(
    df: pd.DataFrame,
    proba: np.ndarray,
    p_buy: float = 0.55,
    p_sell: float = 0.45,
    risk_cfg: RiskConfig | None = None,
    annualize_k: float | None = None,
    target_vol: float | None = None,
    dd_limit: float = 0.10,               # 10% account DD pause
    dd_resume_hysteresis: float = 0.03,   # resume when DD ≤ 7%
    max_hold_bars: int | None = None,     # align to label horizon if set
    liquidity_cap_enabled: bool = True,
    fee_bps: float | None = None,         # legacy placeholder to avoid breaking callers
    slippage_bps: float | None = None,    # legacy placeholder to avoid breaking callers
    decision_to_fill_lag: int = 1,        # bars from decision to execution fill
    verbose: bool = False,
    allow_shorts: bool = True,
) -> pd.DataFrame:
    """
    Bar-close decision; execute next bar. Vol targeting uses *past* returns.
    Costs in contract/tick terms; risk uses centralized config.
    Outputs equity factor, pnl_usd, final positions, leverage, costs, turnover, and stop reasons.
    """
    df = df.copy()
    risk_cfg = risk_cfg or default_risk_config()
    initial_capital = risk_cfg.limits.account_size
    risk_per_trade = risk_cfg.limits.risk_per_trade_usd
    account_loss_limit = risk_cfg.limits.max_drawdown_usd
    max_hold_bars = max_hold_bars or risk_cfg.limits.max_hold_bars
    annualize_k = annualize_k or _annualize_k(df, risk_cfg)
    target_vol = target_vol if target_vol is not None else risk_cfg.limits.target_vol_annualized

    if "date" not in df:
        # requires a Datetime-like index/column named 'Datetime' in your pipeline
        if "Datetime" in df.columns:
            df["date"] = pd.to_datetime(df["Datetime"], errors="coerce").dt.date
        else:
            df["date"] = pd.to_datetime(df.index, errors="coerce").date

    # --- Returns to trade on (past, not forward) ---
    if "ret_1" in df.columns:
        ret = df["ret_1"].to_numpy()
    else:
        ret = df[CLOSE_COL].pct_change().fillna(0.0).to_numpy()

    n = len(df)
    if n == 0 or len(proba) == 0:
        empty = df[[CLOSE_COL]].iloc[0:0].copy()
        empty["pos_raw"] = []
        empty["lev"] = []
        empty["pos"] = []
        empty["strat_ret"] = []
        empty["gross_ret"] = []
        empty["cost_ret"] = []
        empty["equity"] = []
        empty["pnl_usd"] = []
        empty["stop_reason"] = []
        empty.attrs["turnover"] = 0.0
        empty.attrs["annualize_k"] = annualize_k
        empty.attrs["target_vol"] = target_vol
        return empty

    # --- Raw side from probabilities ---
    side = np.zeros(n, dtype=float)
    valid_proba = np.isfinite(proba)
    side[(proba > p_buy) & valid_proba] = 1.0
    side[(proba < p_sell) & valid_proba] = -1.0
    if not allow_shorts:
        side[side < 0] = 0.0

    # --- Vol targeting on past realized vol (shifted) ---
    vol_ann = realized_vol(ret, annualize_k=annualize_k)       # uses past bars only
    lev_raw = vol_target_leverage(vol_ann, target_vol=target_vol)  # clip & smooth inside risk.py if you kept that
    lev_exec = np.roll(lev_raw, 1)
    lev_exec[0] = 0.0          # use previous bar’s leverage

    # --- Target pos and execution lag (no look-ahead) ---
    lag = max(1, int(decision_to_fill_lag))
    pos_target = side * lev_exec
    pos_exec = np.roll(pos_target, lag)
    pos_exec[:lag] = 0.0

    # --- Costs ---
    cost_cfg = risk_cfg.costs
    tick_value = risk_cfg.tick_value

    # --- Loop state ---
    equity = [1.0]                 # equity factor
    pnl_usd = [0.0]
    gross_ret = []
    cost_ret = []
    pos_out = []
    strat_ret = []
    stop_reason_out = []
    turnover = 0.0

    daily_losses = 0
    active_today = True
    cur_day = df["date"].iloc[0]

    # Per-trade state
    prev_pos = 0.0
    in_trade = False
    entry_equity = 1.0
    trade_peak_equity = 1.0
    bars_in_trade = 0

    # Account-level high-water mark for trailing stop
    acct_hwm_usd = 0.0

    # Drawdown circuit state
    dd_paused = False

    for i in range(n):
        # --- day roll ---
        if df["date"].iloc[i] != cur_day:
            cur_day = df["date"].iloc[i]
            daily_losses = 0
            active_today = True  # new day, reset daily pause

        # --- drawdown circuit check (based on equity up to previous bar) ---
        cur_eq = equity[-1]
        acct_hwm_eq = max(cur_eq, np.max(equity))  # prior peak
        dd = 1.0 - (cur_eq / max(acct_hwm_eq, 1e-12))
        if not dd_paused and dd >= dd_limit:
            dd_paused = True
        elif dd_paused and dd <= max(dd_limit - dd_resume_hysteresis, 0.0):
            dd_paused = False

        # --- Choose position for this bar ---
        if not active_today or dd_paused:
            pos_use = 0.0
            reason = "inactive" if not active_today else "dd_circuit"
        else:
            pos_use = float(pos_exec[i])
            reason = None

        if liquidity_cap_enabled:
            pos_use = _liquidity_cap(pos_use, df, i, risk_cfg)

        # Detect fresh entry (sign change from flat or flip)
        if np.sign(pos_use) != np.sign(prev_pos) and pos_use != 0.0:
            in_trade = True
            entry_equity = equity[-1]
            trade_peak_equity = equity[-1]
            bars_in_trade = 0

        # --- Return & costs for this bar ---
        r_gross = pos_use * ret[i]
        delta_pos = pos_use - prev_pos
        trade_cost = _trade_cost_usd(
            delta_pos,
            tick_value=tick_value,
            commission_per_contract=cost_cfg.commission_per_contract,
            slippage_ticks_per_side=cost_cfg.slippage_ticks_per_side,
        )
        trade_cost_ret = trade_cost / max(initial_capital, 1e-12)
        r_net = r_gross - trade_cost_ret
        gross_ret.append(r_gross)
        cost_ret.append(trade_cost_ret)

        next_eq = equity[-1] * (1.0 + r_net)
        next_pnl_usd = initial_capital * (next_eq - 1.0)
        turnover += abs(delta_pos)

        # --- Per-trade bookkeeping for stops (since-entry, next-bar effect) ---
        if in_trade:
            trade_peak_equity = max(trade_peak_equity, next_eq)
            trade_pnl_usd = initial_capital * (next_eq - entry_equity)
            runup_usd = initial_capital * (trade_peak_equity - entry_equity)
            bars_in_trade += 1

            # Per-trade hard stop
            if trade_pnl_usd <= -risk_per_trade:
                active_today = True     # not a day pause; just exit next bar
                pos_exec[i + 1 if i + 1 < n else i] = 0.0
                reason = reason or "trade_loss"

            # Breakeven stop (protect when up ≥ $500, don’t allow loss)
            if runup_usd >= 500 and trade_pnl_usd < 0:
                pos_exec[i + 1 if i + 1 < n else i] = 0.0
                reason = reason or "breakeven_stop"

            # Trailing stop (give back > $500 from peak)
            if (runup_usd - trade_pnl_usd) > 500:
                pos_exec[i + 1 if i + 1 < n else i] = 0.0
                reason = reason or "trailing_stop"

            # Horizon-aligned exit to match labels (bars open >= max_hold_bars)
            if max_hold_bars and bars_in_trade >= max_hold_bars:
                pos_exec[i + 1 if i + 1 < n else i] = 0.0
                reason = reason or "max_hold_horizon"

        # --- Daily loss streak stop (counts losing bars) ---
        if next_eq < equity[-1]:
            daily_losses += 1
            if daily_losses >= MAX_LOSSES_PER_DAY:
                active_today = False
                reason = reason or "daily_losses"

        # --- Absolute account stop (hard floor) ---
        if next_pnl_usd <= -account_loss_limit:
            active_today = False
            reason = reason or "account_stop"

        # --- Dynamic trailing account stop: pause when down $10k from HWM ---
        if next_pnl_usd > acct_hwm_usd:
            acct_hwm_usd = next_pnl_usd
        if next_pnl_usd <= (acct_hwm_usd - 10_000):
            active_today = False
            reason = reason or "account_trailing_stop"

        # Save step
        equity.append(next_eq)
        pnl_usd.append(next_pnl_usd)
        pos_out.append(pos_use)
        strat_ret.append(r_net)
        stop_reason_out.append(reason)

        prev_pos = pos_use
        if verbose:
            print(f"i={i} pos={pos_use:.2f} r={r_net:.5f} eq={next_eq:.5f} pnl={next_pnl_usd:.2f} reason={reason}")

        # If we just flattened (pos->0), end trade state
        if pos_use == 0.0 and in_trade:
            in_trade = False

    # Trim the seed
    equity = np.asarray(equity[1:], dtype=float)
    pnl_usd = np.asarray(pnl_usd[1:], dtype=float)

    # --- Output table ---
    out = df[[CLOSE_COL]].copy()
    out["pos_raw"] = side
    out["lev"] = lev_exec
    out["pos"] = np.asarray(pos_out, dtype=float)
    out["strat_ret"] = np.asarray(strat_ret, dtype=float)
    out["gross_ret"] = np.asarray(gross_ret, dtype=float)
    out["cost_ret"] = np.asarray(cost_ret, dtype=float)
    out["equity"] = equity
    out["pnl_usd"] = pnl_usd
    out["stop_reason"] = stop_reason_out
    out.attrs["turnover"] = turnover
    out.attrs["annualize_k"] = annualize_k
    out.attrs["target_vol"] = target_vol
    return out
