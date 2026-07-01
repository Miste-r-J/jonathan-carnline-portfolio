"""
Lightweight configuration shims for the simplified training/gym harness.

The real repository exposes a rich ``na.config`` package with dozens of
dataclasses and YAML loaders.  The simplified build only needs a tiny slice
to keep ``na.bot`` and ``discord_addons`` bootstraps happy, so we expose the
same public helpers with greatly reduced scope.
"""

from .app_config import (
    AppConfig,
    EvolutionConfig,
    EvolutionThresholds,
    JournalSection,
    LabelSection,
    RiskProfileConfig,
    RiskSection,
    RouterSection,
    SessionSection,
)
from .loader import load_app_config, load_master
from .registry import (
    ChampionConfig,
    ExperienceConfig,
    InstrumentConfig,
    ManualPresetConfig,
    RiskPresetConfig,
    ShadowRegistryConfig,
    Registry,
    get_registry,
)

__all__ = [
    "AppConfig",
    "EvolutionConfig",
    "EvolutionThresholds",
    "JournalSection",
    "LabelSection",
    "RiskProfileConfig",
    "RiskSection",
    "RouterSection",
    "SessionSection",
    "load_app_config",
    "load_master",
    "InstrumentConfig",
    "RiskPresetConfig",
    "ManualPresetConfig",
    "ChampionConfig",
    "ExperienceConfig",
    "ShadowRegistryConfig",
    "Registry",
    "get_registry",
]
