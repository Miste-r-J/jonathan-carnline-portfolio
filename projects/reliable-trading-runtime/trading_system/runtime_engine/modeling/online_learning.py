"""
Minimal online-learning shim for the standalone gym.

The production stack replaces the base model with contextual bandits and
sliding-window retraining.  Here we only keep enough structure for the
streamer to log experience and expose heartbeat metrics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, Optional, Sequence

import numpy as np

from trading_system.runtime_engine.shared.experience import Experience


@dataclass
class OnlineLearningConfig:
    """
    Configuration for the lightweight online learner.

    Attributes mirror the production config so CLI arguments remain compatible,
    but the standalone version keeps the semantics intentionally loose.
    """

    enabled: bool = False
    experience_window: int = 200
    min_trades_before_update: int = 50
    cooldown_bars: int = 50
    learning_rate: float = 0.05
    status_emit_interval: int = 100  # bars between status snapshots


def default_online_config(symbol: str = "ES") -> OnlineLearningConfig:
    # Symbol hook left in place for compatibility; not used in the standalone build.
    return OnlineLearningConfig(enabled=False)


class OnlineLearner:
    """
    Append-only buffer that mimics the production learner interface.

    We retain experiences for telemetry and basic health reporting, but the
    base model is never mutated.  This keeps the gym deterministic while
    preserving the hooks researchers expect to call.
    """

    def __init__(
        self,
        *,
        base_model: Any,
        config: OnlineLearningConfig,
        risk_cfg: Any,
        alert_sink: Any = None,
    ) -> None:
        self.base_model = base_model
        self.config = config
        self.risk_cfg = risk_cfg
        self.alert_sink = alert_sink
        self._experience: Deque[Experience] = deque(maxlen=max(int(config.experience_window), 1))
        self._updates = 0
        self._last_status_bar = 0
        self._last_update_ts: Optional[datetime] = None

    # ---------------- public API mirrored for the streamer ----------------
    def add_experience(self, experience: Experience) -> None:
        self._experience.append(experience)

    def maybe_update_model(self) -> bool:
        """
        Trigger placeholder updates once enough samples accumulate.
        Returns ``True`` if downstream consumers should refresh metrics.
        """
        if not self.config.enabled:
            return False
        if len(self._experience) < self.config.min_trades_before_update:
            return False
        # We do not modify the base model; we only mark the update for telemetry.
        self._updates += 1
        self._last_update_ts = datetime.utcnow()
        return True

    def predict_proba(self, features: np.ndarray, base_proba: Sequence[float]) -> np.ndarray:
        """
        Placeholder hook that simply returns the base probability.
        """
        return np.asarray(base_proba, dtype=float)

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "samples": len(self._experience),
            "updates": self._updates,
            "last_update": self._last_update_ts.isoformat() if self._last_update_ts else None,
        }


__all__ = ["OnlineLearner", "OnlineLearningConfig", "default_online_config"]
