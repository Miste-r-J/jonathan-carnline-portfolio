"""
Standalone configuration loader used by the ``trading_system.runtime_engine`` package.

Only a subset of the real project's configuration surface is implemented,
enough to power the training scripts and the CSV streamer.  YAML layouts are
kept intentionally small and ship with sane defaults so the gym works
out-of-the-box, while still allowing operators to override paths via env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

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

RUNTIME_ENV = "NA_SIMPLIFIED_RUNTIME_CONFIG"
MASTER_ENV = "NA_SIMPLIFIED_MASTER_CONFIG"
_RUNTIME_PATH = Path(__file__).with_name("runtime.yaml")
_MASTER_PATH = Path(__file__).with_name("master.yaml")


@dataclass(frozen=True)
class MasterAppConfig:
    instrument: str
    session_tz: str
    rth_start: str
    rth_end: str
    orb_min: int


@dataclass(frozen=True)
class MasterRiskConfig:
    per_stop_usd: float
    daily_stop_usd: float


@dataclass(frozen=True)
class MasterConfig:
    app: MasterAppConfig
    risk: MasterRiskConfig


def _resolve(path: str | os.PathLike[str] | None, env_key: str, default: Path) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    env_value = os.getenv(env_key)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return default


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise TypeError(f"Expected mapping at root of {path}, got {type(data).__name__}")
        return data


def _coerce_float(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_evolution(section: Mapping[str, Any] | None) -> EvolutionConfig:
    defaults = EvolutionThresholds()

    def _parse_thresholds(data: Mapping[str, Any] | None) -> EvolutionThresholds:
        if not data:
            return defaults
        return EvolutionThresholds(
            min_trades=int(data.get("min_trades", defaults.min_trades)),
            min_pnl=float(data.get("min_pnl", defaults.min_pnl)),
            max_drawdown=float(data.get("max_drawdown", defaults.max_drawdown)),
            min_winrate=float(data.get("min_winrate", defaults.min_winrate)),
        )

    section = section or {}
    return EvolutionConfig(
        sim_thresholds=_parse_thresholds(section.get("sim_thresholds")),
        live_thresholds=_parse_thresholds(section.get("live_thresholds")),
    )


def _parse_risk_profiles(section: Mapping[str, Mapping[str, Any]] | None) -> Dict[str, RiskProfileConfig]:
    profiles: Dict[str, RiskProfileConfig] = {}
    if not section:
        return profiles
    for name, payload in section.items():
        if not isinstance(payload, Mapping):
            continue
        profiles[name] = RiskProfileConfig(
            name=str(name),
            max_drawdown=float(payload.get("max_drawdown", 0.0)),
            profit_target=float(payload.get("profit_target", 0.0)),
            lockout_on_violation=bool(payload.get("lockout_on_violation", True)),
            max_trades=int(payload["max_trades"]) if payload.get("max_trades") is not None else None,
        )
    return profiles


def load_app_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    resolved = _resolve(path, RUNTIME_ENV, _RUNTIME_PATH)
    payload = _load_yaml(resolved)
    runtime = payload.get("runtime") or {}

    risk_section = runtime.get("risk") or {}
    per_inst = risk_section.get("per_instrument_risk") or {}
    per_inst_upper = {str(k).upper(): float(v) for k, v in per_inst.items() if v is not None}
    risk_cfg = RiskSection(
        daily_loss_limit=float(risk_section.get("daily_loss_limit", 1000.0)),
        max_intraday_dd=_coerce_float(risk_section.get("max_intraday_dd")),
        per_instrument_risk=per_inst_upper or None,
    )

    session_section = runtime.get("session") or {}
    session_cfg = SessionSection(
        tz=str(session_section.get("tz", "America/Chicago")),
        rth_start=str(session_section.get("rth_start", "07:30")),
        rth_end=str(session_section.get("rth_end", "14:00")),
    )

    router_section = runtime.get("router") or {}
    whitelist = router_section.get("instrument_whitelist")
    if whitelist:
        whitelist = [str(sym).upper() for sym in whitelist]
    else:
        whitelist = ["ES"]
    router_cfg = RouterSection(
        hmac_secret_env=str(router_section.get("hmac_secret_env", "ROUTER_HMAC_SECRET")),
        min_prob=float(router_section.get("min_prob", 0.55)),
        instrument_whitelist=tuple(whitelist),
    )

    labels_section = runtime.get("labels") or {}
    label_cfg = LabelSection(
        trend_ma_window=int(labels_section.get("trend_ma_window", 200)),
        trend_slope_window=int(labels_section.get("trend_slope_window", 20)),
        horizon_bars=int(labels_section.get("horizon_bars", 12)),
        domain=str(labels_section.get("domain", "ternary")),
        drop_flats=bool(labels_section.get("drop_flats", False)),
    )

    journal_section = runtime.get("journal") or {}
    journal_cfg = JournalSection(
        backend_url=journal_section.get("backend_url"),
        auth_header=str(journal_section.get("auth_header", "X-JOURNAL-TOKEN")),
        auth_token_env=journal_section.get("auth_token_env"),
        enable_http=bool(journal_section.get("enable_http", False)),
    )

    evolution_cfg = _parse_evolution(runtime.get("evolution"))
    risk_profiles = _parse_risk_profiles(runtime.get("risk_profiles") or payload.get("risk_profiles"))
    features_cfg_raw = runtime.get("features")
    if isinstance(features_cfg_raw, Mapping):
        features_cfg = dict(features_cfg_raw)
    else:
        features_cfg = {}

    return AppConfig(
        risk=risk_cfg,
        session=session_cfg,
        router=router_cfg,
        labels=label_cfg,
        journal=journal_cfg,
        evolution=evolution_cfg,
        risk_profiles=risk_profiles,
        features=features_cfg,
    )


def load_master(path: str | os.PathLike[str] | None = None) -> MasterConfig:
    resolved = _resolve(path, MASTER_ENV, _MASTER_PATH)
    payload = _load_yaml(resolved)
    app_section = payload.get("app") or {}
    risk_section = payload.get("risk") or {}

    app_cfg = MasterAppConfig(
        instrument=str(app_section.get("instrument", "ES")).upper(),
        session_tz=str(app_section.get("session_tz", "America/Chicago")),
        rth_start=str(app_section.get("rth_start", "07:30")),
        rth_end=str(app_section.get("rth_end", "14:00")),
        orb_min=int(app_section.get("orb_min", 15)),
    )
    risk_cfg = MasterRiskConfig(
        per_stop_usd=float(risk_section.get("per_stop_usd", 500.0)),
        daily_stop_usd=float(risk_section.get("daily_stop_usd", 1500.0)),
    )
    return MasterConfig(app=app_cfg, risk=risk_cfg)


__all__ = ["load_app_config", "load_master", "MasterConfig"]
