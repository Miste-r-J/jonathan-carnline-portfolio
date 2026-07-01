from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from trading_system.runtime_engine.integrations.addons.risk_guard.config import load_effective_profile


@dataclass
class ModelGuardDecision:
    action: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class RiskGuardModel:
    """Market-only RiskGuard view. No execution outcome reads."""

    def __init__(
        self,
        *,
        profile: str,
        yaml_path: str,
        instrument: str,
        risk_strict: bool = True,
        cli_overrides: Optional[Dict[str, Any]] = None,
        env_json_overrides_var: str = "RISK_JSON_OVERRIDES",
        default_tz: str = "America/Chicago",
    ) -> None:
        profile_cfg = load_effective_profile(
            profile=profile,
            instrument=instrument,
            yaml_path=str(yaml_path),
            materialized_path=None,
            risk_strict=risk_strict,
            cli_overrides=cli_overrides,
            env_json_overrides_var=env_json_overrides_var,
            default_tz=default_tz,
        )
        self.profile = profile_cfg.name
        self.cfg = profile_cfg.data
        self.instrument = instrument or self.cfg.get("instrument", "") or ""
        self._tz = ZoneInfo(str(self.cfg.get("tz") or default_tz))
        confirmations = self.cfg.get("confirmations") or {}
        anti = self.cfg.get("anti_overtrade") or {}
        self._min_signal_bars = (
            int(confirmations.get("min_signal_bars")) if isinstance(confirmations, dict) and confirmations.get("min_signal_bars") is not None else None
        )
        self._same_dir_dedupe_min = (
            int(anti.get("same_dir_dedupe_min")) if isinstance(anti, dict) and anti.get("same_dir_dedupe_min") is not None else None
        )

    def evaluate_entry(self, *, signal_bars: Optional[int], last_direction: Optional[str]) -> ModelGuardDecision:
        if self._min_signal_bars is not None and signal_bars is not None and signal_bars < self._min_signal_bars:
            return ModelGuardDecision(
                action="block",
                reason="min_signal_bars",
                details={"signal_bars": int(signal_bars), "min_signal_bars": int(self._min_signal_bars)},
            )
        if self._same_dir_dedupe_min is not None and last_direction:
            return ModelGuardDecision(
                action="allow",
                reason="dedupe_model_only",
                details={"same_dir_dedupe_min": int(self._same_dir_dedupe_min), "last_direction": str(last_direction)},
            )
        return ModelGuardDecision(action="allow", reason="model_guard_allow")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "instrument": self.instrument,
            "min_signal_bars": self._min_signal_bars,
            "same_dir_dedupe_min": self._same_dir_dedupe_min,
            "tz": str(self._tz),
        }
