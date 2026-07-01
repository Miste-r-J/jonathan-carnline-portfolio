from __future__ import annotations

"""
Central configuration for online learning / contextual bandit adapter.
All defaults live here so sims/streaming share a single source of truth.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OnlineLearningConfig:
    # Blending + update cadence
    alpha_online: float = 0.10
    update_every_n_trades: int = 10
    min_buffer_size: int = 100
    max_buffer_size: int = 1000
    max_daily_updates: int = 50

    # Reward shaping
    reward_clip_min: float = -5.0
    reward_clip_max: float = 5.0
    min_sample_weight: float = 0.1
    drawdown_penalty_threshold: float = 5.0
    drawdown_penalty: float = 0.5

    # Safety / rollback
    rollback_window_trades: int = 100
    underperf_threshold: float = 0.5  # base_mean_r - online_mean_r tolerance (R units)
    alpha_decay_step: float = 0.05
    auto_freeze: bool = True

    # Exploration
    exploration_eps: float = 0.0
    exploration_delta: float = 0.02

    # Persistence
    autosave_dir: Optional[str] = None


def default_online_config(symbol: str | None = None) -> OnlineLearningConfig:
    """
    Conservative defaults for ES/NQ-style intraday futures.
    Symbol parameter reserved for future per-instrument tuning.
    """
    _ = symbol  # placeholder for future instrument-specific tuning
    return OnlineLearningConfig(
        alpha_online=0.10,
        update_every_n_trades=10,
        min_buffer_size=100,
        max_buffer_size=1000,
        max_daily_updates=50,
        reward_clip_min=-5.0,
        reward_clip_max=5.0,
        min_sample_weight=0.1,
        rollback_window_trades=100,
        underperf_threshold=0.5,
        alpha_decay_step=0.05,
        auto_freeze=True,
        exploration_eps=0.0,
    )


__all__ = ["OnlineLearningConfig", "default_online_config"]
