from __future__ import annotations

"""
Meta-strategy allocation utilities (strategy-of-strategies).
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class MetaStrategyConfig:
    window_trades: int = 100
    max_weight_per_strategy: float = 0.7
    metric: str = "sharpe"  # sharpe | avg_r_multiple
    regularization_strength: float = 0.2
    min_weight: float = 0.0


def _metric(ts: pd.Series, metric: str) -> float:
    if metric == "avg_r_multiple":
        return float(ts.mean())
    std = float(ts.std()) if ts.size > 1 else 0.0
    return float(ts.mean() / max(std, 1e-9))


def compute_strategy_weights(strategy_pnl: Dict[str, pd.Series], cfg: MetaStrategyConfig) -> Dict[str, pd.Series]:
    weights: Dict[str, pd.Series] = {}
    all_index = None
    for s, pnl in strategy_pnl.items():
        idx = pnl.index
        all_index = idx if all_index is None else all_index.union(idx)
    if all_index is None:
        return {}

    for s, pnl in strategy_pnl.items():
        pnl = pnl.reindex(all_index).fillna(0.0)
        rolling = pnl.rolling(cfg.window_trades, min_periods=max(5, int(0.2 * cfg.window_trades))).apply(
            lambda x: _metric(pd.Series(x), cfg.metric), raw=False
        )
        weights[s] = rolling

    # convert metrics to weights via softmax-style normalization
    weight_df = pd.DataFrame(weights).fillna(0.0)
    exp_scores = np.exp(weight_df - weight_df.max(axis=1).values.reshape(-1, 1))
    softmax = exp_scores.div(exp_scores.sum(axis=1), axis=0).fillna(0.0)
    softmax = softmax.clip(lower=cfg.min_weight, upper=cfg.max_weight_per_strategy)
    row_sums = softmax.sum(axis=1).replace(0, 1.0)
    softmax = softmax.div(row_sums, axis=0)
    out = {col: softmax[col] for col in softmax.columns}
    return out


def combine_strategies_pnl(strategy_pnl: Dict[str, pd.Series], weights: Dict[str, pd.Series]) -> pd.Series:
    aligned = pd.DataFrame(strategy_pnl).fillna(0.0)
    weight_df = pd.DataFrame(weights).reindex_like(aligned).fillna(method="ffill").fillna(0.0)
    weight_df = weight_df.div(weight_df.sum(axis=1).replace(0, 1.0), axis=0)
    combined = (aligned * weight_df).sum(axis=1)
    return combined


__all__ = ["MetaStrategyConfig", "compute_strategy_weights", "combine_strategies_pnl"]
