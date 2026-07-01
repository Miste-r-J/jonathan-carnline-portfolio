"""
Lightweight version of the project's registry loader.

The standalone gym still relies on presets, instrument metadata, and online
learning knobs, so we mirror the public API of ``trading_system.runtime_engine.runtime_config.registry`` while
parsing a drastically smaller ``master.yaml``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

MASTER_PATH = Path(__file__).resolve().with_name("master.yaml")
STRUCTURAL_KEYS = {
    "session_tz",
    "trade_window_start",
    "trade_window_end",
    "allowed_grades",
    "risk",
    "parameters",
    "instrument",
    "description",
    "max_trades_per_session",
}
RISK_FALLBACK_KEYS = (
    "max_daily_loss",
    "max_trades_per_day",
    "max_risk_per_trade_usd",
    "cooldown_after_loss",
    "loss_streak_limit",
    "probability_thresholds",
)


@dataclass(frozen=True)
class InstrumentConfig:
    name: str
    alias: str
    point_value: float
    tick_size: float
    round_lot: int = 1


@dataclass(frozen=True)
class RiskProbabilityThresholds:
    min_p_long: float
    min_p_short: float
    p_setup: float | None = None


@dataclass(frozen=True)
class RiskLimitsConfig:
    max_daily_loss: float
    max_trades_per_day: int
    max_risk_per_trade_usd: float
    cooldown_after_loss: int
    loss_streak_limit: int


@dataclass(frozen=True)
class RiskPresetConfig:
    name: str
    instrument: str
    session_tz: str
    trade_window_start: str
    trade_window_end: str
    allowed_grades: tuple[str, ...] = field(default_factory=tuple)
    max_trades_per_session: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    limits: RiskLimitsConfig | None = None
    probability_thresholds: RiskProbabilityThresholds | None = None

    def as_cli_overrides(self) -> Dict[str, Any]:
        payload = deepcopy(self.parameters)
        phase2_section = payload.pop("phase2", None)
        if isinstance(phase2_section, Mapping):
            for key in ("p_setup", "p_long", "p_short"):
                if key in phase2_section and key not in payload:
                    payload[key] = phase2_section[key]
        payload.setdefault("session_tz", self.session_tz)
        payload.setdefault("trade_window_start", self.trade_window_start)
        payload.setdefault("trade_window_end", self.trade_window_end)
        payload.setdefault("allowed_grades", tuple(self.allowed_grades))
        payload.setdefault("max_trades_per_session", self.max_trades_per_session)
        if self.limits:
            payload.setdefault("max_daily_loss", self.limits.max_daily_loss)
            payload.setdefault("max_trades_per_day", self.limits.max_trades_per_day)
            payload.setdefault("max_risk_per_trade_usd", self.limits.max_risk_per_trade_usd)
            payload.setdefault("cooldown_after_loss", self.limits.cooldown_after_loss)
            payload.setdefault("loss_streak_limit", self.limits.loss_streak_limit)
        if self.probability_thresholds:
            payload.setdefault(
                "probability_thresholds",
                {
                    "min_p_long": self.probability_thresholds.min_p_long,
                    "min_p_short": self.probability_thresholds.min_p_short,
                },
            )
            payload.setdefault("p_long", self.probability_thresholds.min_p_long)
            payload.setdefault("p_short", self.probability_thresholds.min_p_short)
            if self.probability_thresholds.p_setup is not None:
                payload.setdefault("p_setup", float(self.probability_thresholds.p_setup))
        # Legacy fallback for non-phase2 presets: if setup is unspecified, mirror p_buy.
        # Do not apply this when phase2 is enabled, or it can accidentally tighten setup
        # gating to p_buy (e.g., 0.70) when manifest thresholds are disabled.
        if "p_setup" not in payload and "p_buy" in payload and not bool(payload.get("phase2")):
            payload["p_setup"] = payload["p_buy"]
        payload.update(self.metadata)
        return payload


@dataclass(frozen=True)
class ManualPresetConfig:
    name: str
    symbol: str
    preset_id: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class ChampionConfig:
    symbol: str
    name: str
    preset: str
    artifact: str
    preset_file: Optional[str] = None


@dataclass(frozen=True)
class ExperienceConfig:
    directory: str
    file_pattern: str
    min_trades_for_retraining: int
    window_days: int
    max_samples: int


@dataclass(frozen=True)
class ShadowRegistryConfig:
    enabled: bool
    experience_dir: str
    max_models_per_instrument: int
    default_candidates: Dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ABEvalConfig:
    enabled: bool
    min_overlapping_trades: int
    min_delta_ev_for_promotion: float
    max_delta_drawdown: float


@dataclass(frozen=True)
class DriftConfig:
    enabled: bool
    window_trades: int
    max_allowed_feature_shift: float
    max_allowed_label_shift: float
    block_on_drift: bool


@dataclass(frozen=True)
class OnlineLearningRegistryConfig:
    retrain_window_days: int
    retrain_min_trades: int
    retrain_max_samples: int
    promotion_min_trades: int
    promotion_min_ev: float
    promotion_min_win_rate: float
    promotion_max_drawdown: float
    promotion_ev_margin: float
    models_dir: str
    champion_state_dir: str
    instruments: tuple[str, ...]
    watchdog_window_trades: int
    watchdog_min_ev_per_trade: float
    watchdog_max_drawdown: float
    watchdog_auto_rollback: bool
    shadow: ShadowRegistryConfig
    ab_eval: ABEvalConfig
    drift: DriftConfig


class Registry:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self._raw = data
        self.instruments: Dict[str, InstrumentConfig] = self._parse_instruments(data.get("instruments") or {})
        self.risk_presets: Dict[str, RiskPresetConfig] = self._parse_risk_presets(data.get("risk_presets") or {})
        self.manual_presets: Dict[str, ManualPresetConfig] = self._parse_manual_presets(data.get("manual_presets") or {})
        self.champions: Dict[str, ChampionConfig] = self._parse_champions(data.get("champions") or {})
        self.experience_config: ExperienceConfig = self._parse_experience(data.get("experience") or {})
        self.online_learning: OnlineLearningRegistryConfig = self._parse_online_learning(data.get("online_learning") or {})

    def _parse_instruments(self, section: Mapping[str, Any]) -> Dict[str, InstrumentConfig]:
        result: Dict[str, InstrumentConfig] = {}
        for name, payload in section.items():
            alias = str(payload.get("alias") or name).upper()
            result[name.upper()] = InstrumentConfig(
                name=name.upper(),
                alias=alias,
                point_value=float(payload.get("point_value", 0.0)),
                tick_size=float(payload.get("tick_size", 0.0)),
                round_lot=int(payload.get("round_lot", 1) or 1),
            )
        return result

    def _parse_risk_presets(self, section: Mapping[str, Any]) -> Dict[str, RiskPresetConfig]:
        presets: Dict[str, RiskPresetConfig] = {}
        for name, payload in section.items():
            if payload is None:
                continue
            allowed = payload.get("allowed_grades") or ()
            if isinstance(allowed, str):
                allowed = (allowed,)
            parameters = dict(payload.get("parameters") or {})
            for key, value in payload.items():
                if key in STRUCTURAL_KEYS:
                    continue
                if key in parameters:
                    continue
                parameters[key] = deepcopy(value)
            metadata: Dict[str, Any] = {}
            if "description" in payload:
                metadata["description"] = payload["description"]
            max_trades = int(payload.get("max_trades_per_session", parameters.get("max_trades_per_session", 0)) or 0)
            risk_section = dict(payload.get("risk") or {})
            for key in RISK_FALLBACK_KEYS:
                if key in payload and key not in risk_section:
                    risk_section[key] = deepcopy(payload[key])
            limits = self._parse_risk_limits(risk_section)
            prob_thresholds = self._parse_prob_thresholds(risk_section.get("probability_thresholds"))
            presets[name] = RiskPresetConfig(
                name=name,
                instrument=str(payload.get("instrument") or "").upper(),
                session_tz=str(payload.get("session_tz") or "America/Chicago"),
                trade_window_start=str(payload.get("trade_window_start") or "07:30"),
                trade_window_end=str(payload.get("trade_window_end") or "12:00"),
                allowed_grades=tuple(allowed),
                max_trades_per_session=max_trades,
                parameters=parameters,
                metadata=metadata,
                limits=limits,
                probability_thresholds=prob_thresholds,
            )
        return presets

    def _parse_risk_limits(self, section: Mapping[str, Any]) -> RiskLimitsConfig | None:
        if not section:
            return None
        return RiskLimitsConfig(
            max_daily_loss=float(section.get("max_daily_loss", 0.0) or 0.0),
            max_trades_per_day=int(section.get("max_trades_per_day", 0) or 0),
            max_risk_per_trade_usd=float(section.get("max_risk_per_trade_usd", 0.0) or 0.0),
            cooldown_after_loss=int(section.get("cooldown_after_loss", 0) or 0),
            loss_streak_limit=int(section.get("loss_streak_limit", 0) or 0),
        )

    def _parse_prob_thresholds(self, section: Mapping[str, Any] | None) -> RiskProbabilityThresholds | None:
        if not section:
            return None
        return RiskProbabilityThresholds(
            min_p_long=float(section.get("min_p_long", 0.0) or 0.0),
            min_p_short=float(section.get("min_p_short", 0.0) or 0.0),
            p_setup=(float(section.get("p_setup")) if section.get("p_setup") is not None else None),
        )

    def _parse_manual_presets(self, section: Mapping[str, Any]) -> Dict[str, ManualPresetConfig]:
        result: Dict[str, ManualPresetConfig] = {}
        for name, payload in section.items():
            params = dict(payload.get("params") or {})
            result[name] = ManualPresetConfig(
                name=name,
                symbol=str(payload.get("symbol") or "").upper(),
                preset_id=str(payload.get("preset_id") or ""),
                params=params,
            )
        return result

    def _parse_champions(self, section: Mapping[str, Any]) -> Dict[str, ChampionConfig]:
        champions: Dict[str, ChampionConfig] = {}
        for symbol, payload in section.items():
            presets = payload.get("presets") or {}
            active_name = payload.get("active")
            if not active_name:
                continue
            active_payload = presets.get(active_name)
            if not active_payload:
                continue
            champions[symbol.upper()] = ChampionConfig(
                symbol=symbol.upper(),
                name=active_name,
                preset=str(active_payload.get("preset") or ""),
                artifact=str(active_payload.get("artifact") or ""),
                preset_file=active_payload.get("preset_file"),
            )
        return champions

    def _parse_experience(self, payload: Mapping[str, Any]) -> ExperienceConfig:
        directory = str(payload.get("dir") or "journal_reports/experience")
        file_pattern = str(payload.get("file_pattern") or "{symbol}_experience.jsonl")
        min_trades = int(payload.get("min_trades_for_retraining", 50) or 50)
        window_days = int(payload.get("window_days", 5) or 5)
        max_samples = int(payload.get("max_samples", min_trades * 5) or min_trades * 5)
        if min_trades <= 0:
            raise ValueError("experience.min_trades_for_retraining must be positive")
        if window_days <= 0:
            raise ValueError("experience.window_days must be positive")
        return ExperienceConfig(
            directory=directory,
            file_pattern=file_pattern,
            min_trades_for_retraining=min_trades,
            window_days=window_days,
            max_samples=max_samples if max_samples > 0 else min_trades * 5,
        )

    def _parse_shadow(self, payload: Mapping[str, Any]) -> ShadowRegistryConfig:
        enabled = bool(payload.get("enabled", False))
        experience_dir = str(payload.get("experience_shadow_dir") or "journal_reports/experience_shadow")
        max_models = int(payload.get("max_shadow_models_per_instrument", 1) or 1)
        defaults_raw = payload.get("default_shadow_candidates") or {}
        defaults: Dict[str, tuple[str, ...]] = {}
        for sym, values in defaults_raw.items():
            if isinstance(values, str):
                defaults[sym.upper()] = (str(values),)
            elif isinstance(values, Iterable):
                defaults[sym.upper()] = tuple(str(v) for v in values if v)
            else:
                defaults[sym.upper()] = tuple()
        return ShadowRegistryConfig(
            enabled=enabled,
            experience_dir=experience_dir,
            max_models_per_instrument=max_models,
            default_candidates=defaults,
        )

    def _parse_ab_eval(self, payload: Mapping[str, Any]) -> ABEvalConfig:
        enabled = bool(payload.get("enabled", False))
        min_overlap = int(payload.get("min_overlapping_trades", 0) or 0)
        min_delta_ev = float(payload.get("min_delta_ev_for_promotion", 0.0) or 0.0)
        max_delta_dd = float(payload.get("max_delta_drawdown", 0.0) or 0.0)
        return ABEvalConfig(
            enabled=enabled,
            min_overlapping_trades=min_overlap,
            min_delta_ev_for_promotion=min_delta_ev,
            max_delta_drawdown=max_delta_dd,
        )

    def _parse_drift(self, payload: Mapping[str, Any]) -> DriftConfig:
        enabled = bool(payload.get("enabled", False))
        window = int(payload.get("window_trades", 0) or 0)
        feature_shift = float(payload.get("max_allowed_feature_shift", 0.0) or 0.0)
        label_shift = float(payload.get("max_allowed_label_shift", 0.0) or 0.0)
        block_on_drift = bool(payload.get("block_on_drift", True))
        return DriftConfig(
            enabled=enabled,
            window_trades=window,
            max_allowed_feature_shift=feature_shift,
            max_allowed_label_shift=label_shift,
            block_on_drift=block_on_drift,
        )

    def _parse_online_learning(self, payload: Mapping[str, Any]) -> OnlineLearningRegistryConfig:
        retrain_window_days = int(payload.get("retrain_window_days", 1) or 1)
        retrain_min_trades = int(payload.get("retrain_min_trades", 1) or 1)
        retrain_max_samples = int(payload.get("retrain_max_samples", retrain_min_trades * 5) or retrain_min_trades * 5)
        promo_min_trades = int(payload.get("promotion_min_trades", retrain_min_trades) or retrain_min_trades)
        promotion_min_ev = float(payload.get("promotion_min_ev", 0.0) or 0.0)
        promotion_min_win_rate = float(payload.get("promotion_min_win_rate", 0.0) or 0.0)
        promotion_max_drawdown = float(payload.get("promotion_max_drawdown", 0.0) or 0.0)
        promotion_ev_margin = float(payload.get("promotion_ev_margin", 0.0) or 0.0)
        models_dir = str(payload.get("models_dir") or "artifacts/models")
        champion_state_dir = str(payload.get("champion_state_dir") or "artifacts/champions_state")
        instruments_raw = payload.get("instruments") or ()
        if isinstance(instruments_raw, str):
            instruments = (instruments_raw.upper(),)
        else:
            instruments = tuple(str(sym).upper() for sym in instruments_raw)
        watchdog_section = payload.get("watchdog") or {}
        watchdog_window_trades = int(watchdog_section.get("window_trades", 0) or 0)
        watchdog_min_ev = float(watchdog_section.get("min_ev_per_trade", 0.0) or 0.0)
        watchdog_max_dd = float(watchdog_section.get("max_drawdown", 0.0) or 0.0)
        watchdog_auto_rb = bool(watchdog_section.get("auto_rollback", False))
        shadow_cfg = self._parse_shadow(payload.get("shadow") or {})
        ab_eval_cfg = self._parse_ab_eval(payload.get("ab_eval") or {})
        drift_cfg = self._parse_drift(payload.get("drift") or {})
        if retrain_window_days <= 0 or retrain_min_trades <= 0:
            raise ValueError("online_learning retrain settings must be positive")
        return OnlineLearningRegistryConfig(
            retrain_window_days=retrain_window_days,
            retrain_min_trades=retrain_min_trades,
            retrain_max_samples=retrain_max_samples,
            promotion_min_trades=promo_min_trades,
            promotion_min_ev=promotion_min_ev,
            promotion_min_win_rate=promotion_min_win_rate,
            promotion_max_drawdown=promotion_max_drawdown,
            promotion_ev_margin=promotion_ev_margin,
            models_dir=models_dir,
            champion_state_dir=champion_state_dir,
            instruments=instruments,
            watchdog_window_trades=watchdog_window_trades,
            watchdog_min_ev_per_trade=watchdog_min_ev,
            watchdog_max_drawdown=watchdog_max_dd,
            watchdog_auto_rollback=watchdog_auto_rb,
            shadow=shadow_cfg,
            ab_eval=ab_eval_cfg,
            drift=drift_cfg,
        )

    def get_instrument(self, name: str) -> InstrumentConfig:
        key = str(name).upper()
        if key not in self.instruments:
            raise KeyError(f"Unknown instrument '{name}'")
        return self.instruments[key]

    def get_risk_preset(self, name: str) -> RiskPresetConfig:
        key = str(name)
        if key not in self.risk_presets:
            raise KeyError(f"Unknown risk preset '{name}'")
        return self.risk_presets[key]

    def get_risk_preset_for_instrument(self, instrument: str, name: Optional[str] = None) -> RiskPresetConfig:
        if name:
            preset = self.get_risk_preset(name)
            if preset.instrument and preset.instrument != instrument.upper():
                raise ValueError(f"Risk preset '{name}' does not match instrument '{instrument}'")
            return preset
        for preset in self.risk_presets.values():
            if preset.instrument == instrument.upper():
                return preset
        raise KeyError(f"No risk preset found for instrument '{instrument}'")

    def get_manual_preset(self, name: str) -> ManualPresetConfig:
        key = str(name)
        if key not in self.manual_presets:
            raise KeyError(f"Unknown manual preset '{name}'")
        return self.manual_presets[key]

    def get_champion(self, symbol: str) -> ChampionConfig:
        key = str(symbol).upper()
        if key not in self.champions:
            raise KeyError(f"No champion configured for '{symbol}'")
        return self.champions[key]


class _DuplicateKeyError(ValueError):
    pass


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys at the same mapping level.

    PyYAML silently keeps the last of any duplicated key, which hides config
    bugs (a stray second ``max_daily_loss`` overriding the intended one). We
    fail fast instead so the duplicate is fixed rather than resolved by luck.
    """

    def construct_mapping(self, node, deep=False):
        seen: set = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                line = key_node.start_mark.line + 1
                raise _DuplicateKeyError(f"Duplicate key '{key}' at line {line}")
            seen.add(key)
        return super().construct_mapping(node, deep)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictSafeLoader) or {}
    except _DuplicateKeyError as exc:
        raise ValueError(f"Duplicate key in {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping at root of {path}")
    return data


@lru_cache(maxsize=None)
def get_registry(config_path: Optional[str] = None) -> Registry:
    target = Path(config_path).expanduser().resolve() if config_path else MASTER_PATH
    return Registry(_load_yaml(target))


__all__ = [
    "InstrumentConfig",
    "RiskPresetConfig",
    "ManualPresetConfig",
    "ChampionConfig",
    "ExperienceConfig",
    "ShadowRegistryConfig",
    "Registry",
    "get_registry",
]
