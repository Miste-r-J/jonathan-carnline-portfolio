"""
Threshold search utilities for per-setup / per-regime gating.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ThresholdOptParams:
    taus: Sequence[float]
    min_trades: int = 50
    min_precision: float = 0.55
    max_dd_r: float = 8.0
    objective: str = "ev_per_trade"  # or "ev_total"


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    equity = values.cumsum()
    peaks = equity.cummax()
    dd = peaks - equity
    dd = dd.fillna(0.0)
    return float(dd.max() if len(dd) else 0.0)


def _resolve_params(params: ThresholdOptParams | Mapping[str, object]) -> ThresholdOptParams:
    if isinstance(params, ThresholdOptParams):
        return params
    allowed = {f.name for f in fields(ThresholdOptParams)}
    kwargs = {k: v for k, v in dict(params).items() if k in allowed}
    return ThresholdOptParams(**kwargs)


def _evaluate_side(
    frame: pd.DataFrame,
    *,
    side: str,
    tau: float,
    params: ThresholdOptParams,
) -> Optional[Dict[str, float]]:
    col_proba = "p_long" if side == "LONG" else "p_short"
    col_flag = "is_long" if side == "LONG" else "is_short"
    col_r = "r_long" if side == "LONG" else "r_short"
    trades = frame.loc[frame[col_proba] >= tau].copy()
    n_trades = len(trades)
    if n_trades < max(1, int(params.min_trades)):
        return None
    precision = float((trades[col_flag]).sum()) / float(n_trades or 1)
    if precision < params.min_precision:
        return None
    pnl = trades[col_r].fillna(0.0)
    ev_total = float(pnl.sum())
    ev_per_trade = ev_total / float(n_trades or 1)
    max_dd = _max_drawdown(pnl)
    if params.max_dd_r > 0 and max_dd > params.max_dd_r:
        return None
    win_rate = float((pnl > 0).sum()) / float(n_trades or 1)
    avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
    avg_loss = float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0
    return {
        "tau": float(tau),
        "n_trades": int(n_trades),
        "precision": float(precision),
        "ev_total": ev_total,
        "ev_per_trade": ev_per_trade,
        "max_drawdown_r": float(max_dd),
        "win_rate": float(win_rate),
        "avg_win_r": float(avg_win),
        "avg_loss_r": float(avg_loss),
    }


def _select_threshold(frame: pd.DataFrame, *, side: str, params: ThresholdOptParams) -> Dict[str, object]:
    best_metrics: Optional[Dict[str, float]] = None
    best_score = -np.inf
    taus = sorted(float(t) for t in params.taus)
    for tau in taus:
        metrics = _evaluate_side(frame, side=side, tau=tau, params=params)
        if not metrics:
            continue
        score = metrics["ev_per_trade"] if params.objective == "ev_per_trade" else metrics["ev_total"]
        if score > best_score:
            best_score = score
            best_metrics = metrics
    if best_metrics is None:
        return {
            "tau": None,
            "n_trades": 0,
            "precision": 0.0,
            "ev_total": 0.0,
            "ev_per_trade": 0.0,
            "max_drawdown_r": 0.0,
            "win_rate": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "status": "no_solution",
        }
    best_metrics["status"] = "ok"
    return best_metrics


def optimize_thresholds(
    preds_df: pd.DataFrame,
    *,
    params: ThresholdOptParams | Mapping[str, object],
    groupby: Sequence[str],
) -> Dict[str, object]:
    """
    Optimize probability thresholds for LONG/SHORT per specified groups.
    """

    params = _resolve_params(params)
    if not params.taus:
        raise ValueError("Threshold grid (taus) must be non-empty.")
    group_cols = list(groupby or [])
    groups_payload: List[Dict[str, object]] = []

    if group_cols:
        grouped = preds_df.groupby(group_cols, dropna=False)
    else:
        grouped = [((), preds_df)]

    for key, frame in grouped:
        if isinstance(key, tuple):
            values = list(key)
        else:
            values = [key]
        key_map = {
            group_cols[idx]: values[idx]
            for idx in range(len(group_cols))
        } if group_cols else {}
        entry = {
            "key": key_map,
            "long": _select_threshold(frame, side="LONG", params=params),
            "short": _select_threshold(frame, side="SHORT", params=params),
        }
        groups_payload.append(entry)

    global_entry = {
        "long": _select_threshold(preds_df, side="LONG", params=params),
        "short": _select_threshold(preds_df, side="SHORT", params=params),
    }

    return {
        "params": asdict(params),
        "groupby": group_cols,
        "groups": groups_payload,
        "global": global_entry,
        "taus_tested": sorted(float(t) for t in params.taus),
    }


__all__ = [
    "ThresholdOptParams",
    "optimize_thresholds",
]
