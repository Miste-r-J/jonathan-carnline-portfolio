from __future__ import annotations
"""
Production-ready configuration for the prop-firm backtest & live engine.
Extended with unified bundles (model/risk/online/portfolio/meta) and YAML loader.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Iterable, List, Any, TYPE_CHECKING
from pathlib import Path

import yaml

from ..config.registry import get_registry, MASTER_PATH

_REGISTRY = get_registry()

# =========================
# Probability grading bands
# =========================
PROB_BANDS: Dict[str, Dict[str, float]] = {
    # Calibrated for conservative A+/B+ gating; pair with EV gating when provided
    "long": {"A+": 0.66, "B+": 0.54},
    "short": {"A+": 0.34, "B+": 0.42},
}


def _validate_prob_bands(bands: Dict[str, Dict[str, float]]) -> None:
    for side in ("long", "short"):
        assert side in bands, f"PROB_BANDS missing '{side}'"
        assert {"A+", "B+"}.issubset(bands[side]), f"PROB_BANDS[{side}] must have A+ and B+"
        for k, v in bands[side].items():
            assert 0.0 <= v <= 1.0, f"PROB_BANDS[{side}][{k}] must be within [0,1]"
    assert bands["long"]["B+"] <= bands["long"]["A+"], "long bands must satisfy B+ <= A+"
    assert bands["short"]["A+"] <= bands["short"]["B+"], "short bands must satisfy A+ <= B+"


_validate_prob_bands(PROB_BANDS)

# =========================
# Column name conventions
# =========================
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOLUME_COL = "Volume"
MAX_LOSSES_PER_DAY = 5  # legacy backtest guard; kept for compatibility
HORIZON = 5
RET_THRESHOLD = 0.0005

# =========================
# Feature-engineering knobs
# =========================
VOL_WINDOW = 60
SMA_WINDOWS = (5, 10, 20, 50)
EMA_WINDOWS = (9, 20, 50)
RSI_WINDOW = 14

# =========================
# Cost models
# =========================
@dataclass(frozen=True)
class CostConfig:
    """Legacy equity-bps cost (kept for backward compat)."""
    fee_bps: float = 0.5  # per side
    slippage_bps: float = 0.5  # per side


@dataclass(frozen=True)
class ContractCostConfig:
    """Per-contract costs for futures (preferred in contract mode)."""
    commission_per_contract: float = 2.0  # USD, per side
    slippage_ticks_per_side: float = 1.0  # ticks per side (override in CLI as needed)


# =========================
# Instrument metadata
# =========================
@dataclass(frozen=True)
class InstrumentSpec:
    alias: str  # "ES", "NQ", "MES", "MNQ", ...
    point_value: float  # USD per 1.00 price point (ES=50, NQ=20, MES=5, MNQ=2)
    tick_size: float  # minimum price increment (ES=0.25)
    round_lot: int = 1  # contracts are integers

    @property
    def tick_value(self) -> float:
        return self.point_value * self.tick_size

    def round_to_tick(self, price: float) -> float:
        """Round a price to the nearest valid tick."""
        k = round(price / self.tick_size)
        return k * self.tick_size


INSTRUMENTS: Dict[str, InstrumentSpec] = {
    name: InstrumentSpec(
        alias=cfg.alias,
        point_value=cfg.point_value,
        tick_size=cfg.tick_size,
        round_lot=cfg.round_lot,
    )
    for name, cfg in _REGISTRY.instruments.items()
}
INSTRUMENTS.setdefault(
    "MES",
    InstrumentSpec(alias="MES", point_value=5.0, tick_size=0.25, round_lot=1),
)


def instrument_by_alias(alias: str) -> InstrumentSpec:
    a = str(alias).upper()
    if a in INSTRUMENTS:
        return INSTRUMENTS[a]
    raise KeyError(f"Unknown instrument alias '{alias}'. Available: {', '.join(sorted(INSTRUMENTS))}")


# =========================
# Engine defaults
# =========================
@dataclass(frozen=True)
class EngineDefaults:
    # thresholds & grading (fallback if EV thresholds not provided)
    p_buy: float = 0.60
    p_sell: float = 0.40
    allowed_grades: Tuple[str, ...] = ("A+", "B+")

    # session/window (Prop-friendly: MT / Denver and RTH sub-window)
    session_tz: str = "America/Denver"
    trade_window_start: str = "07:30"  # ES RTH open (MT)
    trade_window_end: str = "14:00"  # stop before lunch/liquidity fade

    # Feature builder day structure
    rth_start: str = "06:30"
    rth_end: str = "14:00"
    orb_min: int = 15

    # HARD cap
    max_trades_per_day: int = 3

    # In contract mode, equity acts as a notional ledger. Keep baseline
    # to translate % thresholds into USD (prop-firm style).
    account_scale_usd: float = 100_000.0
    initial_equity: float = 100_000.0

    # drawdown circuit (% of account_scale_usd)
    enable_dd_circuit: bool = True
    dd_limit: float = 0.10
    dd_resume_hysteresis: float = 0.03
    dd_disable_from_next_bar: bool = True

    # vol targeting (off by default; enable via preset/CLI)
    enable_vol_target: bool = False
    target_vol: Optional[float] = None
    vol_ema_span: int = 10
    vol_annualize_k: float = 19656.0  # 252 days * ~78 5m bars/day
    pos_cap: float = 3.0

    # LLM (optional)
    use_llm: bool = False
    llm_review_all: bool = False
    llm_max_risk_bps: int = 25
    llm_cooldown_min: int = 5
    symbol: str = "ES=F"
    instrument_alias: str = "ES"


ENGINE = EngineDefaults()
RISK_DEFAULTS = {
    "max_risk_per_trade": 0.003,  # 30 bps of account
    "max_trades_per_session": 3,
    "max_entries_per_signal": 1,
    "max_positions": 1,
    "max_r_giveback": 0.0,
    "fallback_stop_pct": 0.0035,
    "max_risk_per_day": 0.04,
    "max_risk_per_week": 0.10,
}
CONTRACT_COST = ContractCostConfig()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


EXTRA_PRESET_SOURCES: Dict[str, Path] = {}


def _load_extra_cli_presets() -> Dict[str, Dict[str, object]]:
    """
    Load operator-authored presets from configs/presets/*.yaml so CLI users can
    add bundles without touching the registry.
    """
    presets: Dict[str, Dict[str, object]] = {}
    sources: Dict[str, Path] = {}
    presets_dir = _repo_root() / "configs" / "presets"
    if not presets_dir.exists():
        return presets

    for pattern in ("*.yaml", "*.yml"):
        for path in presets_dir.glob(pattern):
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            entries = payload.get("presets") if isinstance(payload, dict) else None
            if not isinstance(entries, dict):
                continue
            for name, cfg in entries.items():
                if not isinstance(cfg, dict):
                    continue
                if name in presets:
                    raise ValueError(
                        f"Duplicate preset '{name}' in {path} (already defined in {sources[name]})"
                    )
                if name in _REGISTRY.risk_presets:
                    raise ValueError(
                        f"Duplicate preset '{name}' in {path} (already defined in {MASTER_PATH})"
                    )
                presets[name] = cfg
                sources[name] = path
    EXTRA_PRESET_SOURCES.update(sources)
    return presets


PRESETS: Dict[str, Dict[str, object]] = {
    name: preset.as_cli_overrides()
    for name, preset in _REGISTRY.risk_presets.items()
}
EXTRA_PRESETS = _load_extra_cli_presets()
if EXTRA_PRESETS:
    # Registry-defined presets take precedence; extras fill the remaining slots.
    PRESETS.update({k: v for k, v in EXTRA_PRESETS.items() if k not in PRESETS})
# Ensure late-session streaming works for the high-participation preset
_MAXPACK10 = PRESETS.get("es_maxpack_10_full_send")
if isinstance(_MAXPACK10, dict):
    # extend TOD gate so stream_live_csv doesn't block after lunch
    _MAXPACK10.setdefault("trade_window_end", "15:30")
    PRESETS["es_maxpack_10_full_send"] = _MAXPACK10

# =========================
# Unified bundles & YAML loader
# =========================
from .online_config import OnlineLearningConfig, default_online_config
from .meta_strategy import MetaStrategyConfig
from .risk_config import RiskConfig  # type: ignore  # circular-safe; imports this module for INSTRUMENTS
from .alerting import build_alert_sink_from_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .portfolio_backtest import PortfolioConfig


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    horizon: int = 5
    label_threshold: float = 0.0005
    p_buy: float = 0.62
    p_sell: float = 0.38


@dataclass
class RiskConfigBundle:
    symbol: str = "ES"
    risk: Optional[RiskConfig] = None


@dataclass
class OnlineConfigBundle:
    strategy_id: str = "base_online_default"
    online: OnlineLearningConfig = field(default_factory=default_online_config)



def _default_portfolio_config():
    from .portfolio_backtest import PortfolioConfig as _PortfolioConfig  # local import to avoid heavy deps at import time

    return _PortfolioConfig()


@dataclass
class PortfolioConfigBundle:
    symbols: List[str] = field(default_factory=lambda: ["ES"])
    portfolio: "PortfolioConfig" = field(default_factory=_default_portfolio_config)


@dataclass
class MetaStrategyConfigBundle:
    enabled: bool = False
    meta: MetaStrategyConfig = field(default_factory=MetaStrategyConfig)
    strategies: Dict[str, str] = field(default_factory=dict)  # id->description


@dataclass
class FullConfig:
    model: ModelConfig
    risk: RiskConfigBundle
    online: OnlineConfigBundle
    portfolio: PortfolioConfigBundle
    meta_strategies: MetaStrategyConfigBundle
    alerting_sink: Optional[str] = None


def load_config_from_yaml(path: str) -> FullConfig:
    cfg_dict = yaml.safe_load(Path(path).read_text()) or {}
    model_cfg = ModelConfig(**cfg_dict.get("model", {}))
    risk_sym = (cfg_dict.get("risk", {}) or {}).get("symbol", "ES")
    # risk config assembled lazily to avoid circular on import
    from .risk_config import default_risk_config  # local import
    rc = default_risk_config(risk_sym)
    for k, v in (cfg_dict.get("risk", {}) or {}).items():
        if hasattr(rc, k) and k != "symbol":
            setattr(rc, k, v)
    risk_bundle = RiskConfigBundle(symbol=risk_sym, risk=rc)

    online_cfg_raw = cfg_dict.get("online", {}) or {}
    online_cfg = default_online_config(risk_sym)
    for k, v in online_cfg_raw.items():
        if hasattr(online_cfg, k):
            setattr(online_cfg, k, v)
    online_bundle = OnlineConfigBundle(
        strategy_id=online_cfg_raw.get("strategy_id", "base_online_default"),
        online=online_cfg,
    )

    portfolio_cfg_raw = cfg_dict.get("portfolio", {}) or {}
    from .portfolio_backtest import PortfolioConfig as _PortfolioConfig  # local import, keep lazily loaded

    portfolio_cfg = _PortfolioConfig(
        **{k: v for k, v in portfolio_cfg_raw.items() if k in _PortfolioConfig.__annotations__}
    )
    portfolio_bundle = PortfolioConfigBundle(
        symbols=portfolio_cfg_raw.get("symbols", ["ES"]),
        portfolio=portfolio_cfg,
    )

    meta_cfg_raw = cfg_dict.get("meta_strategies", {}) or {}
    meta_cfg = MetaStrategyConfig(**{k: v for k, v in meta_cfg_raw.items() if k in MetaStrategyConfig.__annotations__})
    meta_bundle = MetaStrategyConfigBundle(
        enabled=bool(meta_cfg_raw.get("enabled", False)),
        meta=meta_cfg,
        strategies=meta_cfg_raw.get("strategies", {}) or {},
    )
    alert_sink = cfg_dict.get("alerting", {}).get("sink") if isinstance(cfg_dict.get("alerting"), dict) else None

    return FullConfig(
        model=model_cfg,
        risk=risk_bundle,
        online=online_bundle,
        portfolio=portfolio_bundle,
        meta_strategies=meta_bundle,
        alerting_sink=alert_sink,
    )


def default_config(symbol: str = "ES") -> FullConfig:
    from .risk_config import default_risk_config  # local import to avoid circular
    return FullConfig(
        model=ModelConfig(),
        risk=RiskConfigBundle(symbol=symbol, risk=default_risk_config(symbol)),
        online=OnlineConfigBundle(strategy_id="base_online_default", online=default_online_config(symbol)),
        portfolio=PortfolioConfigBundle(symbols=[symbol], portfolio=_default_portfolio_config()),
        meta_strategies=MetaStrategyConfigBundle(enabled=False, strategies={}),
    )


def build_runtime_alert_sink(full_cfg: FullConfig):
    """Convenience builder for AlertSink from FullConfig."""
    return build_alert_sink_from_config(full_cfg.alerting_sink)


__all__ = [
    "PROB_BANDS",
    "OPEN_COL",
    "HIGH_COL",
    "LOW_COL",
    "CLOSE_COL",
    "VOLUME_COL",
    "VOL_WINDOW",
    "SMA_WINDOWS",
    "EMA_WINDOWS",
    "RSI_WINDOW",
    "CostConfig",
    "ContractCostConfig",
    "InstrumentSpec",
    "INSTRUMENTS",
    "instrument_by_alias",
    "EngineDefaults",
    "ENGINE",
    "RISK_DEFAULTS",
    "MAX_LOSSES_PER_DAY",
    "CONTRACT_COST",
    "PRESETS",
    "ModelConfig",
    "RiskConfigBundle",
    "OnlineConfigBundle",
    "PortfolioConfigBundle",
    "MetaStrategyConfigBundle",
    "FullConfig",
    "load_config_from_yaml",
    "default_config",
]
