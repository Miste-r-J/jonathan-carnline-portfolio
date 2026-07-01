"""
Shared label/exit logic to keep train/backtest/live aligned.

Definitions:
- Label: forward return over `horizon` bars above `threshold` → 1 else 0.
- Exit: horizon-based flatten after `horizon` bars unless stopped earlier.

All helpers are past-only and horizon-aligned to avoid drift between train/backtest/live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..config.loader import load_app_config

import numpy as np
import pandas as pd

from .config import CLOSE_COL


@dataclass(frozen=True)
class LabelConfig:
    horizon: int = 5
    threshold: float = 0.0005  # 5 bps ~= 2 ES ticks
    use_log: bool = False
    use_htf_trend_aware: bool = False  # Experimental: incorporate HTF trend to only label in trend direction
    drop_flat: bool = False

    @classmethod
    def from_app_defaults(
        cls,
        *,
        threshold: float | None = None,
        use_log: bool | None = None,
        use_htf_trend_aware: bool | None = None,
    ) -> "LabelConfig":
        """Derive a LabelConfig from runtime label defaults."""
        cfg = load_app_config().labels
        derived_threshold = threshold if threshold is not None else 0.0005
        derived_use_log = bool(use_log) if use_log is not None else False
        default_domain = (cfg.domain or "binary").lower()
        derived_trend = (
            use_htf_trend_aware
            if use_htf_trend_aware is not None
            else default_domain.startswith("trend_aware")
        )
        return cls(
            horizon=int(cfg.horizon_bars or 5),
            threshold=derived_threshold,
            use_log=derived_use_log,
            use_htf_trend_aware=derived_trend,
            drop_flat=bool(cfg.drop_flats or derived_trend),
        )


def make_horizon_labels(df: pd.DataFrame, cfg: LabelConfig) -> Tuple[pd.Series, pd.Series]:
    if cfg.horizon <= 0:
        raise ValueError("Label horizon must be positive")
    if cfg.use_htf_trend_aware:
        # Use trend-aware labeling to prevent up labels in downtrends
        from .labeling import make_labels
        result = make_labels(
            df,
            close_col=CLOSE_COL,
            horizon=cfg.horizon,
            threshold=cfg.threshold,
            scheme="trend_aware_ternary",
            use_log=cfg.use_log,
            drop_flat=bool(cfg.drop_flat),
        )
        y = result["target"]
        close = result["close"]
        mask = y.notna()
        if cfg.drop_flat:
            mask &= y != 0
        return y[mask].astype(int), close[mask]
    else:
        close = df[CLOSE_COL]
        if cfg.use_log:
            fwd_ret = np.log(close.shift(-cfg.horizon)) - np.log(close)
        else:
            fwd_ret = close.shift(-cfg.horizon) / close - 1.0
        valid = fwd_ret.notna()
        y = (fwd_ret > cfg.threshold).astype(int)
        mask = valid
        if cfg.drop_flat:
            mask &= y != 0
        return y[mask], close[mask]


def horizon_flatten_bar_index(horizon: int) -> int:
    """
    The bar offset (>=1) at which to force-flatten to stay aligned with label horizon.
    For bar-close decision with next-bar execution, use horizon as-is.
    """
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    return horizon
