"""
Runtime application configuration dataclasses.

This mirrors the public surface of ``trading_system.runtime_engine.runtime_config.app_config`` so callers that
expect the richer repo keep working, but the actual structure is intentionally
lean: only fields referenced by the runtime modeling modules are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class RiskSection:
    daily_loss_limit: float
    max_intraday_dd: Optional[float] = None
    per_instrument_risk: Mapping[str, float] | None = None


@dataclass(frozen=True)
class RiskProfileConfig:
    name: str
    max_drawdown: float
    profit_target: float
    lockout_on_violation: bool = True
    max_trades: Optional[int] = None


@dataclass(frozen=True)
class SessionSection:
    tz: str
    rth_start: str
    rth_end: str


@dataclass(frozen=True)
class RouterSection:
    hmac_secret_env: str
    min_prob: float
    instrument_whitelist: Sequence[str]


@dataclass(frozen=True)
class LabelSection:
    trend_ma_window: int
    trend_slope_window: int
    horizon_bars: int
    domain: str
    drop_flats: bool = False


@dataclass(frozen=True)
class EvolutionThresholds:
    min_trades: int = 50
    min_pnl: float = 0.0
    max_drawdown: float = -1000.0
    min_winrate: float = 0.45


@dataclass(frozen=True)
class EvolutionConfig:
    sim_thresholds: EvolutionThresholds = field(default_factory=EvolutionThresholds)
    live_thresholds: EvolutionThresholds = field(default_factory=EvolutionThresholds)


@dataclass(frozen=True)
class JournalSection:
    backend_url: Optional[str]
    auth_header: str
    auth_token_env: Optional[str]
    enable_http: bool = False


@dataclass(frozen=True)
class AppConfig:
    risk: RiskSection
    session: SessionSection
    router: RouterSection
    labels: LabelSection
    journal: JournalSection
    evolution: EvolutionConfig
    risk_profiles: Mapping[str, RiskProfileConfig] = field(default_factory=dict)
    features: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "RiskSection",
    "RiskProfileConfig",
    "SessionSection",
    "RouterSection",
    "LabelSection",
    "EvolutionThresholds",
    "EvolutionConfig",
    "JournalSection",
    "AppConfig",
]
