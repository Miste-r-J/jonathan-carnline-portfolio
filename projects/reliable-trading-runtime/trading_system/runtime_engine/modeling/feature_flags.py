from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class FeatureToggle:
    """Container for an individual feature flag and its configuration payload."""

    name: str
    raw_enabled: bool
    config: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


@dataclass
class FeatureContext:
    """
    Resolved feature toggle state for a preset or runtime configuration.

    - kill_switch: when True, all new functionality must be disabled.
    - shadow_mode: evaluate feature logic but do not enforce behaviour (for telemetry only).
    - toggles: map of feature name -> FeatureToggle.
    """

    kill_switch: bool
    shadow_mode: bool
    toggles: Dict[str, FeatureToggle] = field(default_factory=dict)
    preset_name: Optional[str] = None
    shadow_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        *,
        kill_switch: bool,
        features: Optional[Mapping[str, Mapping[str, Any]]] = None,
        preset_name: Optional[str] = None,
    ) -> "FeatureContext":
        toggles: Dict[str, FeatureToggle] = {}
        shadow_toggle: Optional[FeatureToggle] = None

        for name, value in (features or {}).items():
            if isinstance(value, Mapping):
                raw_enabled = bool(value.get("enabled"))
                config_payload = {k: v for k, v in value.items() if k != "enabled"}
            else:
                raw_enabled = bool(value)
                config_payload = {}

            toggle = FeatureToggle(name=name, raw_enabled=raw_enabled, config=dict(config_payload))
            if name == "shadow_mode":
                shadow_toggle = toggle
            else:
                toggles[name] = toggle

        shadow_mode = bool(shadow_toggle.raw_enabled) if shadow_toggle else False
        # Kill switch is authoritative: shadow mode cannot be active when legacy path enforced.
        shadow_mode = shadow_mode and (not kill_switch)

        return cls(
            kill_switch=bool(kill_switch),
            shadow_mode=shadow_mode,
            toggles=toggles,
            preset_name=preset_name,
            shadow_config=dict(shadow_toggle.config if shadow_toggle else {}),
        )

    @classmethod
    def disabled(cls, *, kill_switch: bool = False, preset_name: Optional[str] = None) -> "FeatureContext":
        return cls(kill_switch=bool(kill_switch), shadow_mode=False, toggles={}, preset_name=preset_name)

    def raw_enabled(self, name: str) -> bool:
        toggle = self.toggles.get(name)
        return bool(toggle.raw_enabled) if toggle else False

    def is_enabled(self, name: str) -> bool:
        if self.kill_switch:
            return False
        return self.raw_enabled(name)

    def should_evaluate(self, name: str) -> bool:
        """Return True when feature logic should run (enforce OR evaluate in shadow mode)."""
        if self.kill_switch:
            return False
        toggle = self.toggles.get(name)
        if not toggle:
            return False
        return bool(toggle.raw_enabled or self.shadow_mode)

    def config_for(self, name: str) -> Dict[str, Any]:
        toggle = self.toggles.get(name)
        return dict(toggle.config) if toggle else {}

    def debug_enabled(self) -> bool:
        """Debug logging allowed when debug flag active and kill-switch off."""
        return self.is_enabled("debug_flags")

    def shadow_enabled(self) -> bool:
        return bool(self.shadow_mode and not self.kill_switch)


__all__ = ["FeatureToggle", "FeatureContext"]
