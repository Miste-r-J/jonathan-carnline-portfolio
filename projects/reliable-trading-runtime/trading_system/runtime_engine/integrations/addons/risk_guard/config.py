from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


@dataclass(frozen=True)
class RiskProfileConfig:
    name: str
    data: Dict[str, Any]


def load_effective_profile(
    *,
    profile: str,
    instrument: str,
    yaml_path: str | None = None,
    materialized_path: str | None = None,
    risk_strict: bool = True,
    cli_overrides: Optional[Dict[str, Any]] = None,
    env_json_overrides_var: str = "RISK_JSON_OVERRIDES",
    default_tz: str = "America/Chicago",
) -> RiskProfileConfig:
    """
    Standalone loader for RiskGuard profiles.

    Accepts the same arguments as the production helper but only reads a single
    YAML file (if provided).  Missing files fall back to a permissive baseline.
    """

    payload: Dict[str, Any] = {
        "profile": profile,
        "instrument": instrument,
        "tz": default_tz,
        "loss_limits": {
            "max_dollar_loss": None,
            "max_losses_per_day": None,
            "max_R_per_day": None,
        },
    }

    def _deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> None:
        for key, value in src.items():
            if isinstance(value, dict):
                current = dst.get(key)
                if not isinstance(current, dict):
                    current = {}
                dst[key] = current
                _deep_update(current, value)
            else:
                dst[key] = value

    def _resolve_profile(doc: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        profiles = doc.get("profiles")
        if isinstance(profiles, Mapping):
            candidate = profiles.get(profile)
            if isinstance(candidate, Mapping):
                return dict(candidate)
        candidate = doc.get(profile)
        if isinstance(candidate, Mapping):
            return dict(candidate)
        # Allow docs that are already a profile blob (contains guard keys)
        if any(k in doc for k in ("loss_limits", "state", "tz")):
            return dict(doc)
        return None

    def _merge_from_file(candidate: Optional[str]) -> None:
        if not candidate:
            return
        path = Path(candidate).expanduser()
        if not path.exists():
            return
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            return
        # Support both dedicated risk_guard.yaml files (top-level `profiles:`)
        # and master.yaml files that embed RiskGuard under `risk_guard:`.
        if "risk_guard" in raw and isinstance(raw.get("risk_guard"), Mapping):
            raw = raw["risk_guard"]  # type: ignore[assignment]
        # Merge shared settings first (everything except explicit profile map)
        shared = {k: v for k, v in raw.items() if k != "profiles"}
        if shared:
            _deep_update(payload, shared)
        prof_data = _resolve_profile(raw)
        if prof_data:
            _deep_update(payload, prof_data)
            return
        if risk_strict:
            raise ValueError(f"Risk profile '{profile}' not found in '{candidate}'")

    _merge_from_file(yaml_path)
    _merge_from_file(materialized_path)

    env_blob = os.getenv(env_json_overrides_var or "")
    if env_blob:
        try:
            env_cfg = json.loads(env_blob)
        except json.JSONDecodeError:
            env_cfg = None
        if isinstance(env_cfg, Mapping):
            _deep_update(payload, env_cfg)  # best-effort env override

    if cli_overrides:
        for key, value in cli_overrides.items():
            if "." in key:
                parts = [part for part in key.split(".") if part]
                if not parts:
                    continue
                cursor = payload
                for part in parts[:-1]:
                    nxt = cursor.get(part)
                    if not isinstance(nxt, dict):
                        nxt = {}
                    cursor[part] = nxt
                    cursor = nxt
                cursor[parts[-1]] = value
            else:
                payload[key] = value

    return RiskProfileConfig(name=profile, data=payload)


__all__ = ["RiskProfileConfig", "load_effective_profile"]
