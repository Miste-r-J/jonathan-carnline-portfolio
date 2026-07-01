"""
SIM-lite evaluator for policy evaluation using triple-barrier y_r.

This is not full order execution. It is "policy evaluation" using your triple-barrier y_r:
- Inputs: model predictions + thresholds + event-filtered rows
- Risk limits: max trades/day, cooldown, daily loss limit in R
- Outputs: trade list, equity curve in R, metrics (EV/trade, total R, max DD R, win rate, avg win/loss, trades/day)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SimEvalParams:
    max_trades_per_day: int = 10
    cooldown_bars: int = 5
    daily_loss_limit_r: float = 5.0  # in R units
    max_hold_bars: int = 12  # from triple barrier


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    equity = values.cumsum()
    peaks = equity.cummax()
    dd = peaks - equity
    dd = dd.fillna(0.0)
    return float(dd.max() if len(dd) else 0.0)


def _apply_risk_limits(
    trades: pd.DataFrame,
    *,
    params: SimEvalParams,
    timestamps: pd.Series,
) -> pd.DataFrame:
    """Apply risk limits: max trades/day, cooldown, daily loss limit."""
    if trades.empty:
        return trades

    # Group by session (date)
    session_key = timestamps.dt.date
    trades = trades.copy()
    trades["session"] = session_key.loc[trades.index]

    # Daily loss limit
    daily_pnl = trades.groupby("session")["r"].cumsum()
    cumulative_loss = daily_pnl.groupby(trades["session"]).cummin()
    mask_loss_limit = cumulative_loss >= -params.daily_loss_limit_r
    trades = trades.loc[mask_loss_limit]

    # Max trades per day
    trade_counts = trades.groupby("session").cumcount() + 1
    mask_max_trades = trade_counts <= params.max_trades_per_day
    trades = trades.loc[mask_max_trades]

    # Cooldown (simple bar-based, not time-based for simplicity)
    # Assume index is monotonic, cooldown by skipping bars
    valid_indices = trades.index
    filtered = []
    last_trade_idx = -np.inf
    for idx in valid_indices:
        if (idx - last_trade_idx) >= params.cooldown_bars:
            filtered.append(idx)
            last_trade_idx = idx
    trades = trades.loc[filtered]

    return trades.drop(columns=["session"], errors="ignore")


def evaluate_policy(
    preds_df: pd.DataFrame,
    *,
    params: SimEvalParams,
    tau_long: float,
    tau_short: float,
    timestamps: pd.Series,
) -> Dict[str, object]:
    """
    Run SIM-lite evaluation.

    preds_df must contain:
    - y_true (int): true label
    - y_r (float): triple-barrier R
    - p_long (float): long probability
    - p_short (float): short probability
    - setup_id (optional): setup identifier
    - regime_id (optional): regime identifier
    """
    required_cols = ["y_true", "y_r", "p_long", "p_short"]
    for col in required_cols:
        if col not in preds_df.columns:
            raise ValueError(f"preds_df missing required column: {col}")

    # Filter to event-filtered rows (assume preds_df is already filtered)
    df = preds_df.copy()

    # Apply thresholds
    long_trades = df.loc[(df["p_long"] >= tau_long) & (df["y_true"] == 1)].copy()
    long_trades["side"] = "LONG"
    long_trades["r"] = df.loc[long_trades.index, "y_r"]

    short_trades = df.loc[(df["p_short"] >= tau_short) & (df["y_true"] == -1)].copy()
    short_trades["side"] = "SHORT"
    short_trades["r"] = -df.loc[short_trades.index, "y_r"]  # flip for short

    trades = pd.concat([long_trades, short_trades], ignore_index=False).sort_index()

    # Apply risk limits
    trades = _apply_risk_limits(trades, params=params, timestamps=timestamps.loc[trades.index])

    if trades.empty:
        return {
            "n_trades": 0,
            "ev_per_trade": 0.0,
            "total_r": 0.0,
            "max_drawdown_r": 0.0,
            "win_rate": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "trades_per_day": 0.0,
            "equity_curve": [],
            "trade_list": [],
        }

    pnl = trades["r"]
    n_trades = len(trades)
    ev_per_trade = float(pnl.mean())
    total_r = float(pnl.sum())
    max_dd = _max_drawdown(pnl)
    win_rate = float((pnl > 0).sum()) / n_trades
    avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
    avg_loss = float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0

    # Trades per day
    n_days = len(timestamps.dt.date.unique())
    trades_per_day = n_trades / max(1, n_days)

    # Equity curve
    equity_curve = pnl.cumsum().tolist()

    # Trade list
    trade_list = trades[["side", "r"]].to_dict("records")

    return {
        "n_trades": int(n_trades),
        "ev_per_trade": ev_per_trade,
        "total_r": total_r,
        "max_drawdown_r": max_dd,
        "win_rate": win_rate,
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "trades_per_day": trades_per_day,
        "equity_curve": equity_curve,
        "trade_list": trade_list,
    }


def evaluate_policy_grouped(
    preds_df: pd.DataFrame,
    *,
    params: SimEvalParams,
    thresholds: Dict[str, object],
    groupby: Sequence[str],
    timestamps: pd.Series,
) -> Dict[str, object]:
    """
    Evaluate policy per group (e.g., setup_id, regime_id) using optimized thresholds.
    """
    results = {}

    if groupby:
        grouped = preds_df.groupby(groupby, dropna=False)
        for key, frame in grouped:
            if isinstance(key, tuple):
                key_str = "_".join(str(k) for k in key)
            else:
                key_str = str(key)

            # Get thresholds for this group
            group_thresholds = None
            for group_entry in thresholds.get("groups", []):
                if group_entry["key"] == dict(zip(groupby, key if isinstance(key, tuple) else [key])):
                    group_thresholds = group_entry
                    break

            if group_thresholds:
                tau_long = group_thresholds["long"]["tau"]
                tau_short = group_thresholds["short"]["tau"]
            else:
                # Fallback to global
                tau_long = thresholds["global"]["long"]["tau"]
                tau_short = thresholds["global"]["short"]["tau"]

            if tau_long is None or tau_short is None:
                continue

            result = evaluate_policy(
                frame,
                params=params,
                tau_long=tau_long,
                tau_short=tau_short,
                timestamps=timestamps.loc[frame.index],
            )
            results[key_str] = result
    else:
        # Global evaluation
        tau_long = thresholds["global"]["long"]["tau"]
        tau_short = thresholds["global"]["short"]["tau"]
        if tau_long is not None and tau_short is not None:
            result = evaluate_policy(
                preds_df,
                params=params,
                tau_long=tau_long,
                tau_short=tau_short,
                timestamps=timestamps,
            )
            results["global"] = result

    return results


__all__ = [
    "SimEvalParams",
    "evaluate_policy",
    "evaluate_policy_grouped",
]