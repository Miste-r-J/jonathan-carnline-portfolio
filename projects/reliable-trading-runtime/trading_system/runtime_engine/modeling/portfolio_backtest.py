from __future__ import annotations

"""
Multi-asset portfolio backtest wrapper around per-symbol probabilities/thresholds.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from .risk_config import RiskConfig, default_risk_config
from .backtest import simple_strategy_equity


@dataclass
class PortfolioConfig:
    max_gross_exposure: float = 1.0
    max_symbol_exposure: float = 1.0
    max_daily_loss: Optional[float] = None
    max_drawdown: Optional[float] = None
    rebalance_mode: str = "per_signal"  # or bar_close
    allocation_mode: str = "equal_risk"  # or fixed_notional


def run_portfolio_backtest(
    df: pd.DataFrame,
    proba_by_symbol: Dict[str, pd.Series],
    risk_cfg_by_symbol: Optional[Dict[str, RiskConfig]] = None,
    thresholds_by_symbol: Optional[Dict[str, Tuple[float, float]]] = None,
    portfolio_cfg: Optional[PortfolioConfig] = None,
) -> Dict[str, any]:
    pcfg = portfolio_cfg or PortfolioConfig()
    risk_cfg_by_symbol = risk_cfg_by_symbol or {}
    thresholds_by_symbol = thresholds_by_symbol or {}
    per_sym_equity = {}
    per_sym_returns = {}
    for sym, proba in proba_by_symbol.items():
        sub_df = df[df["symbol"] == sym] if "symbol" in df.columns else df.copy()
        p_buy, p_sell = thresholds_by_symbol.get(sym, (0.6, 0.4))
        rcfg = risk_cfg_by_symbol.get(sym, default_risk_config(sym))
        eq = simple_strategy_equity(sub_df, proba, p_buy=p_buy, p_sell=p_sell, risk_cfg=rcfg)
        per_sym_equity[sym] = eq
        per_sym_returns[sym] = eq["equity"].pct_change().fillna(0.0)

    # combine with equal risk weight per symbol
    symbols = list(per_sym_returns.keys())
    if not symbols:
        return {"error": "no symbols provided"}
    weights = np.ones(len(symbols)) / len(symbols)
    aligned = pd.DataFrame(per_sym_returns)
    port_ret = aligned.mul(weights, axis=1).sum(axis=1)
    port_equity = (1.0 + port_ret).cumprod()
    max_dd = (port_equity.cummax() - port_equity).max()
    res = {
        "portfolio_equity": port_equity,
        "portfolio_returns": port_ret,
        "per_symbol_equity": per_sym_equity,
        "per_symbol_returns": per_sym_returns,
        "max_drawdown": float(max_dd),
        "sharpe_like": float(port_ret.mean() / max(port_ret.std(), 1e-9)),
    }
    return res


__all__ = ["PortfolioConfig", "run_portfolio_backtest"]
