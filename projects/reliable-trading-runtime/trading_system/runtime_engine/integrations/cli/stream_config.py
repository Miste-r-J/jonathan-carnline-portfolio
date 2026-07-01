"""
stream_config.py — Immutable config and dependency-bundle dataclasses.

These replace the 309-parameter LiveCSVStreamer constructor incrementally.
The existing constructor remains fully intact for backward compatibility.
Phase 4 will wire StreamDependencies into LiveCSVStreamer as a bundle.

Usage (Phase 3 — configuration only):
    cfg = LiveStreamConfig.from_args(args)
    # cfg documents all validated config in one inspectable object

Usage (Phase 4 — full wiring, not yet implemented):
    deps = StreamDependencies.build(cfg)
    streamer = LiveCSVStreamer.from_config(cfg, deps)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Core immutable config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConnectionConfig:
    """Network and transport connection parameters."""
    fiber_port: int = 6012
    nt_port: int = 5023
    fiber_lookback_bars: int = 10000
    fiber_require_hist: bool = False
    input_mode: str = "csv"


@dataclass(frozen=True)
class SessionConfig:
    """Market session timing and timezone parameters."""
    session_tz: str = "America/Denver"
    rth_start: str = "08:30"
    rth_end: str = "15:00"
    trade_window_start: str = "09:30"
    trade_window_end: str = "14:30"
    session_start: str = "07:45"
    session_end: str = "14:30"
    pre_arm_until: str = "07:30"
    pre_unlock_start: str = "06:30"


@dataclass(frozen=True)
class TradingThresholdConfig:
    """Probability thresholds that gate trade entry and exit."""
    p_buy: Optional[float] = None
    p_sell: Optional[float] = None
    phase2_p_setup: float = 0.60
    phase2_p_long: float = 0.60
    phase2_p_short: float = 0.60
    phase2_close_threshold: float = 0.60
    cooldown_bars: int = 2
    min_target_r_multiple: float = 1.0
    allow_shorts: bool = True


@dataclass(frozen=True)
class RiskConfigPaths:
    """File paths to the three risk/guardrail config YAML files.

    These are loaded by _load_instrument_risk_params, _load_bridge_risk_limits,
    and _load_prop_guardrails respectively. When strict_mode=True (or env var
    NA_LIVE_STRICT_CONFIG_MODE=1), missing or malformed files raise immediately
    instead of silently using permissive defaults.
    """
    instrument_risk_path: Optional[str] = None
    bridge_risk_path: Optional[str] = None
    prop_guardrails_path: Optional[str] = None
    strict_mode: bool = field(
        default_factory=lambda: os.environ.get("NA_LIVE_STRICT_CONFIG_MODE", "").lower() in ("1", "true")
    )


@dataclass(frozen=True)
class Phase2Config:
    """Phase 2 model configuration."""
    enabled: bool = False
    fail_open: bool = False
    setup_model_path: Optional[str] = None
    dir_model_path: Optional[str] = None
    close_enabled: Optional[bool] = None
    close_model_path: Optional[str] = None
    tag: Optional[str] = None
    feature_hash: Optional[str] = None


@dataclass(frozen=True)
class LiveStreamConfig:
    """Top-level immutable runtime config for a single streaming session.

    Call LiveStreamConfig.from_args(args) to build from CLI-parsed argparse args.
    This object is the intended single source of truth for all config that drives
    a live trading run.
    """
    # Identity
    instrument_alias: str = "ES"
    csv_path: Optional[str] = None
    model_alias_or_path: str = ""
    preset_name: Optional[str] = None
    out_dir: str = ""

    # Sub-configs
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    thresholds: TradingThresholdConfig = field(default_factory=TradingThresholdConfig)
    risk: RiskConfigPaths = field(default_factory=RiskConfigPaths)
    phase2: Phase2Config = field(default_factory=Phase2Config)

    # Operational
    poll_sec: float = 1.0
    lookback_bars: int = 2000
    sim_mode: bool = False
    print_signals: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "LiveStreamConfig":
        """Build from argparse Namespace. Validates critical fields."""
        g = lambda name, default=None: getattr(args, name, default)

        strict = os.environ.get("NA_LIVE_STRICT_CONFIG_MODE", "").lower() in ("1", "true")

        return cls(
            instrument_alias=str(g("instrument_alias") or "ES"),
            csv_path=g("csv"),
            model_alias_or_path=str(g("model") or ""),
            preset_name=g("preset"),
            out_dir=str(g("out_dir") or ""),
            connection=ConnectionConfig(
                fiber_port=int(g("fiber_port") or 6012),
                nt_port=int(g("nt_port") or 5023),
                fiber_lookback_bars=int(g("fiber_lookback_bars") or 10000),
                fiber_require_hist=bool(g("fiber_require_hist") or False),
                input_mode=str(g("input_mode") or "csv"),
            ),
            session=SessionConfig(
                session_tz=str(g("session_tz") or "America/Denver"),
                rth_start=str(g("rth_start") or "08:30"),
                rth_end=str(g("rth_end") or "15:00"),
                trade_window_start=str(g("trade_window_start") or "09:30"),
                trade_window_end=str(g("trade_window_end") or "14:30"),
                session_start=str(g("session_start") or "07:45"),
                session_end=str(g("session_end") or "14:30"),
                pre_arm_until=str(g("pre_arm_until") or "07:30"),
                pre_unlock_start=str(g("pre_unlock_start") or "06:30"),
            ),
            thresholds=TradingThresholdConfig(
                p_buy=g("p_buy"),
                p_sell=g("p_sell"),
                phase2_p_setup=float(g("phase2_p_setup") or 0.60),
                phase2_p_long=float(g("phase2_p_long") or 0.60),
                phase2_p_short=float(g("phase2_p_short") or 0.60),
                phase2_close_threshold=float(g("phase2_close_threshold") or 0.60),
                cooldown_bars=int(g("cooldown_bars") or 2),
                min_target_r_multiple=float(g("min_target_r_multiple") or 1.0),
                allow_shorts=bool(g("allow_shorts") if g("allow_shorts") is not None else True),
            ),
            risk=RiskConfigPaths(
                instrument_risk_path=g("instrument_risk_config"),
                bridge_risk_path=g("bridge_risk_config"),
                prop_guardrails_path=g("prop_guardrails_config"),
                strict_mode=strict,
            ),
            phase2=Phase2Config(
                enabled=bool(g("phase2") or False),
                fail_open=bool(g("phase2_fail_open") or False),
                setup_model_path=g("phase2_setup_model"),
                dir_model_path=g("phase2_dir_model"),
                close_enabled=g("phase2_close_enabled"),
                close_model_path=g("phase2_close_model"),
                tag=g("phase2_tag"),
                feature_hash=g("phase2_feature_hash"),
            ),
            poll_sec=float(g("poll_sec") or 1.0),
            lookback_bars=int(g("lookback_bars") or 2000),
            sim_mode=bool(g("sim_mode") or False),
            print_signals=bool(g("print_signals") or False),
        )

    def validate_for_live(self) -> None:
        """Raise ValueError if this config is unsafe for live trading."""
        errors: list[str] = []
        if not self.model_alias_or_path:
            errors.append("model_alias_or_path must be set for live trading")
        if not self.out_dir:
            errors.append("out_dir must be set for live trading")
        if self.thresholds.p_buy is not None and not (0.0 < self.thresholds.p_buy < 1.0):
            errors.append(f"p_buy={self.thresholds.p_buy} is outside (0, 1)")
        if self.thresholds.p_sell is not None and not (0.0 < self.thresholds.p_sell < 1.0):
            errors.append(f"p_sell={self.thresholds.p_sell} is outside (0, 1)")
        if self.risk.strict_mode:
            for label, path in [
                ("instrument_risk_path", self.risk.instrument_risk_path),
                ("bridge_risk_path", self.risk.bridge_risk_path),
                ("prop_guardrails_path", self.risk.prop_guardrails_path),
            ]:
                if path and not Path(path).expanduser().exists():
                    errors.append(f"risk config {label}={path!r} does not exist on disk")
        if errors:
            raise ValueError("LiveStreamConfig validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# ---------------------------------------------------------------------------
# Service dependency bundle (Phase 4 target — not yet wired to LiveCSVStreamer)
# ---------------------------------------------------------------------------

@dataclass
class StreamDependencies:
    """
    Bundle of all external service objects that LiveCSVStreamer depends on.

    In Phase 4, LiveCSVStreamer.__init__ will accept a StreamDependencies
    instead of 309 individual keyword arguments.  For now this class acts as
    a dependency manifest and is not yet used in production code paths.
    """
    feature_builder: Any = None        # callable: (raw_df, ...) -> pd.DataFrame
    model_store: Any = None             # trading_system.runtime_engine.modeling.models loaded model
    publisher: Any = None              # DiscordEmitter / AlertSink
    execution_bridge: Any = None       # NTBridgeServer / MJTBridgeClient
    risk_policy: Any = None            # EnhancedGuardrails / PropRiskConfig
    telemetry: Any = None              # TradeTelemetryStore / DiscordTelemetryReporter
    fiber_server: Any = None           # FiberServer
    file_lock: Any = None              # FileLock
    extra: Dict[str, Any] = field(default_factory=dict)
