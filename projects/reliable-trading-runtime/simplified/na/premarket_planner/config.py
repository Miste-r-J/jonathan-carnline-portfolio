from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..discord_addons.route_config import default_openclaw_config_path, load_openclaw_routes

__all__ = [
    "DATA_DEFAULT_PATH",
    "PACKAGE_DEFAULT_PATH",
    "ENV_CONFIG_PATH",
    "WORKSPACE_ROOT",
    "BackfillConfig",
    "DataConfig",
    "MetricsConfig",
    "OutputConfig",
    "PlannerConfig",
    "PlannerTargetConfig",
    "WindowsConfig",
    "load_config",
    "load_planner_config",
]


DATA_DEFAULT_PATH = Path("/etc/na/premarket_planner.yaml")
MODULE_DIR = Path(__file__).resolve().parent
NA_ROOT = MODULE_DIR.parent
WORKSPACE_ROOT = NA_ROOT.parent
PACKAGE_DEFAULT_PATH = MODULE_DIR / "premarket_planner.yaml"
ENV_CONFIG_PATH = "PREMARKET_PLANNER_CONFIG"


@dataclass(frozen=True)
class DataConfig:
    csv_path: Path
    csv_timezone: Optional[str] = None


@dataclass(frozen=True)
class WindowsConfig:
    eth_start_hour: int
    eth_end_hour: int
    eth_end_minute: int
    min_eth_bars: int
    last_hours_fallback: int


@dataclass(frozen=True)
class MetricsConfig:
    atr_len: int
    atr_timeframe: str
    compute_rth_vwap: bool


@dataclass(frozen=True)
class OutputConfig:
    discord_webhook: str
    round_decimals: int


@dataclass(frozen=True)
class BackfillConfig:
    enabled: bool
    hours: int


@dataclass(frozen=True)
class PlannerTargetConfig:
    """
    Backwards-compatibility stub for legacy imports.
    """

    key: str
    instrument: str
    csv_path: Path
    signals_path: Optional[Path] = None
    notes: Optional[str] = None
    audience: str = "pro"
    channel_key: Optional[str] = None
    channel_id: Optional[int] = None


@dataclass(frozen=True)
class PlannerConfig:
    enabled: bool
    instrument: str
    session_tz: str
    rth_start: time
    rth_end: time
    emit_time_local: time
    data: DataConfig
    windows: WindowsConfig
    metrics: MetricsConfig
    output: OutputConfig
    backfill: BackfillConfig
    signals_path: Optional[Path] = None
    run_time: time = time(hour=6, minute=30)
    dispatch_interval_minutes: float = 60.0
    skip_weekends: bool = True
    command_trigger: str = "/planner"
    discord_token_env: str = "AUTOMATION_DISCORD_TOKEN"
    discord_channel_id: Optional[int] = None
    webhook_url: str = ""
    webhook_username: Optional[str] = None
    mention_role_id: Optional[int] = None
    authorized_users: Tuple[str, ...] = ()
    targets: Tuple[PlannerTargetConfig, ...] = ()
    display_timezone: Optional[str] = None
    openclaw_discord_config: Optional[Path] = None
    news_csv_path: Optional[Path] = None
    news_source_tz: Optional[str] = None
    news_blackout_before_min: int = 5
    news_blackout_after_min: int = 5

    @property
    def session_zone(self) -> ZoneInfo:
        return ZoneInfo(self.session_tz)

    @property
    def timezone(self) -> ZoneInfo:
        return self.session_zone

    @property
    def rth_start_str(self) -> str:
        return self.rth_start.strftime("%H:%M")

    @property
    def rth_end_str(self) -> str:
        return self.rth_end.strftime("%H:%M")

    @property
    def emit_time_str(self) -> str:
        return self.emit_time_local.strftime("%H:%M")


def load_config(path: Optional[str] = None) -> PlannerConfig:
    """
    Load and validate the planner configuration.
    Resolution order:
      1. Explicit `path` argument.
      2. `$PREMARKET_PLANNER_CONFIG` environment variable.
      3. `/etc/na/premarket_planner.yaml`.
      4. Packaged default alongside this module.
    """
    cfg_path = _resolve_config_path(path)
    payload = _read_yaml(cfg_path)

    enabled = bool(payload.get("enabled", True))
    instrument = str(payload.get("instrument") or "ES").upper()

    session_tz = str(payload.get("session_tz") or payload.get("timezone") or "America/Denver")
    _validate_timezone(session_tz)

    rth_start = _parse_time(payload.get("rth_start", "06:30"), key="rth_start")
    rth_end = _parse_time(payload.get("rth_end", "12:59"), key="rth_end")
    emit_time = _parse_time(payload.get("emit_time_local", payload.get("emit_time", "06:30")), key="emit_time_local")

    display_timezone = payload.get("display_timezone")
    if display_timezone:
        _validate_timezone(display_timezone)

    data_cfg = _load_data(payload.get("data"), base_dir=cfg_path.parent)
    windows_cfg = _load_windows(payload.get("windows"))
    metrics_cfg = _load_metrics(payload.get("metrics"))
    output_cfg = _load_output(payload.get("output"))
    backfill_cfg = _load_backfill(payload.get("backfill"))

    run_time_raw = payload.get("run_time")
    run_time = _parse_time(run_time_raw, key="run_time") if run_time_raw else emit_time

    dispatch_raw = payload.get("dispatch_interval_minutes", payload.get("dispatch_interval_min"))
    if dispatch_raw is None:
        dispatch_raw = 60
    dispatch_interval = _coerce_float(dispatch_raw, "dispatch_interval_minutes", min_value=1.0)

    skip_weekends = bool(payload.get("skip_weekends", True))

    command_trigger = str(payload.get("command_trigger") or "/planner").strip()
    if not command_trigger:
        command_trigger = "/planner"

    discord_token_env = str(payload.get("discord_token_env") or "AUTOMATION_DISCORD_TOKEN").strip() or "AUTOMATION_DISCORD_TOKEN"

    channel_raw = payload.get("discord_channel_id")
    if channel_raw is None:
        channel_raw = os.getenv("PREMARKET_DISCORD_CHANNEL_ID")
    discord_channel_id = _coerce_optional_int(channel_raw, "discord_channel_id")

    webhook_url = str(payload.get("webhook_url") or os.getenv("PREMARKET_PLANNER_WEBHOOK") or "").strip()
    webhook_username_raw = payload.get("webhook_username")
    webhook_username = (
        str(webhook_username_raw).strip() or None if webhook_username_raw is not None else None
    )

    mention_role_id = _coerce_optional_int(payload.get("mention_role_id"), "mention_role_id")
    authorized_users = _normalize_authorized_users(payload.get("authorized_users"))
    openclaw_config_raw = payload.get("openclaw_discord_config") or os.getenv("OPENCLAW_DISCORD_CONFIG")
    openclaw_config = (
        _resolve_path(Path(str(openclaw_config_raw)).expanduser(), base_dir=cfg_path.parent)
        if openclaw_config_raw
        else default_openclaw_config_path()
    )
    news_csv_raw = payload.get("news_csv_path")
    if news_csv_raw is None:
        news_csv_raw = os.getenv("AUTOMATION_NEWS_CSV") or os.getenv("NEWS_CSV")
    news_csv_path = _resolve_optional_path(_maybe_path(news_csv_raw), base_dir=cfg_path.parent) if news_csv_raw else None
    news_source_tz = str(payload.get("news_source_tz") or os.getenv("AUTOMATION_NEWS_SOURCE_TZ") or session_tz).strip() or session_tz
    _validate_timezone(news_source_tz)
    news_blackout_before_min = int(payload.get("news_blackout_before_min", os.getenv("AUTOMATION_NEWS_BLACKOUT_BEFORE", 5)))
    news_blackout_after_min = int(payload.get("news_blackout_after_min", os.getenv("AUTOMATION_NEWS_BLACKOUT_AFTER", 5)))

    signals_override = payload.get("signals_path")
    env_override = os.getenv("PREMARKET_PLANNER_SIGNALS")
    if signals_override is not None:
        signals_path = _maybe_path(signals_override)
    elif env_override:
        signals_path = _maybe_path(env_override)
    else:
        signals_path = _resolve_default_signals_path(instrument)
    signals_path = _resolve_optional_path(signals_path, base_dir=cfg_path.parent)
    targets = _load_targets(
        payload.get("targets"),
        default_instrument=instrument,
        default_csv=data_cfg.csv_path,
        default_signals=signals_path,
        base_dir=cfg_path.parent,
        openclaw_config=openclaw_config,
    )
    if discord_channel_id is None:
        for target in targets:
            if target.channel_id is not None:
                discord_channel_id = target.channel_id
                break

    return PlannerConfig(
        enabled=enabled,
        instrument=instrument,
        session_tz=session_tz,
        rth_start=rth_start,
        rth_end=rth_end,
        emit_time_local=emit_time,
        data=data_cfg,
        windows=windows_cfg,
        metrics=metrics_cfg,
        output=output_cfg,
        backfill=backfill_cfg,
        signals_path=signals_path,
        run_time=run_time,
        dispatch_interval_minutes=dispatch_interval,
        skip_weekends=skip_weekends,
        command_trigger=command_trigger,
        discord_token_env=discord_token_env,
        discord_channel_id=discord_channel_id,
        webhook_url=webhook_url,
        webhook_username=webhook_username,
        mention_role_id=mention_role_id,
        authorized_users=authorized_users,
        targets=targets,
        display_timezone=display_timezone,
        openclaw_discord_config=openclaw_config,
        news_csv_path=news_csv_path,
        news_source_tz=news_source_tz,
        news_blackout_before_min=news_blackout_before_min,
        news_blackout_after_min=news_blackout_after_min,
    )


def load_planner_config(path: Optional[str] = None) -> PlannerConfig:
    """
    Compatibility alias for legacy callers.
    """
    return load_config(path)


def _resolve_config_path(path: Optional[str]) -> Path:
    if path:
        target = Path(path).expanduser()
        if target.exists():
            return target
        raise FileNotFoundError(f"Planner config not found at {target}")

    candidates = []
    env = os.getenv(ENV_CONFIG_PATH)
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(DATA_DEFAULT_PATH)
    candidates.append(PACKAGE_DEFAULT_PATH)

    for target in candidates:
        if target.exists():
            return target

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Planner config not found. Checked: {searched}")


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Planner config root must be a mapping (received {type(data).__name__})")
    return data


def _load_data(section: Any, *, base_dir: Optional[Path] = None) -> DataConfig:
    if section is None:
        raise ValueError("config missing required `data` section")
    if not isinstance(section, dict):
        raise TypeError("`data` section must be a mapping")

    csv_path_raw = section.get("csv_path")
    if not csv_path_raw:
        raise ValueError("`data.csv_path` is required")
    csv_path = Path(str(csv_path_raw)).expanduser()
    csv_path = _resolve_path(csv_path, base_dir=base_dir)

    csv_tz_raw = section.get("csv_timezone")
    csv_tz = str(csv_tz_raw).strip() if csv_tz_raw else None
    if csv_tz:
        _validate_timezone(csv_tz)

    return DataConfig(csv_path=csv_path, csv_timezone=csv_tz)


def _load_windows(section: Any) -> WindowsConfig:
    defaults = {
        "eth_start_hour": 13,
        "eth_end_hour": 7,
        "eth_end_minute": 29,
        "min_eth_bars": 30,
        "last_hours_fallback": 8,
    }
    if section is None:
        section = defaults
    elif not isinstance(section, dict):
        raise TypeError("`windows` section must be a mapping")
    else:
        section = {**defaults, **section}

    eth_start_hour = _coerce_int(section.get("eth_start_hour"), "windows.eth_start_hour", min_value=0, max_value=23)
    eth_end_hour = _coerce_int(section.get("eth_end_hour"), "windows.eth_end_hour", min_value=0, max_value=23)
    eth_end_minute = _coerce_int(section.get("eth_end_minute"), "windows.eth_end_minute", min_value=0, max_value=59)
    min_eth_bars = _coerce_int(section.get("min_eth_bars"), "windows.min_eth_bars", min_value=1)
    last_hours_fallback = _coerce_int(section.get("last_hours_fallback"), "windows.last_hours_fallback", min_value=1)

    return WindowsConfig(
        eth_start_hour=eth_start_hour,
        eth_end_hour=eth_end_hour,
        eth_end_minute=eth_end_minute,
        min_eth_bars=min_eth_bars,
        last_hours_fallback=last_hours_fallback,
    )


def _load_metrics(section: Any) -> MetricsConfig:
    defaults = {
        "atr_len": 14,
        "atr_timeframe": "5m",
        "compute_rth_vwap": True,
    }
    if section is None:
        section = defaults
    elif not isinstance(section, dict):
        raise TypeError("`metrics` section must be a mapping")
    else:
        section = {**defaults, **section}

    atr_len = _coerce_int(section.get("atr_len"), "metrics.atr_len", min_value=1)
    atr_timeframe = str(section.get("atr_timeframe") or "5m")
    compute_rth_vwap = bool(section.get("compute_rth_vwap", True))

    return MetricsConfig(
        atr_len=atr_len,
        atr_timeframe=atr_timeframe,
        compute_rth_vwap=compute_rth_vwap,
    )


def _load_output(section: Any) -> OutputConfig:
    defaults = {"discord_webhook": "", "round_decimals": 2}
    if section is None:
        section = defaults
    elif not isinstance(section, dict):
        raise TypeError("`output` section must be a mapping")
    else:
        section = {**defaults, **section}

    discord_webhook = str(section.get("discord_webhook") or "").strip()
    round_decimals = _coerce_int(section.get("round_decimals"), "output.round_decimals", min_value=0, max_value=10)

    return OutputConfig(discord_webhook=discord_webhook, round_decimals=round_decimals)


def _load_backfill(section: Any) -> BackfillConfig:
    defaults = {"enabled": True, "hours": 24}
    if section is None:
        section = defaults
    elif not isinstance(section, dict):
        raise TypeError("`backfill` section must be a mapping")
    else:
        section = {**defaults, **section}

    enabled = bool(section.get("enabled", True))
    hours = _coerce_int(section.get("hours"), "backfill.hours", min_value=1, max_value=168)

    return BackfillConfig(enabled=enabled, hours=hours)


def _parse_time(value: Any, *, key: str) -> time:
    if isinstance(value, time):
        return value
    if not value:
        raise ValueError(f"`{key}` must be provided")
    try:
        return time.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Invalid time for `{key}`: {value!r}") from exc


def _coerce_int(value: Any, key: str, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    if value is None:
        raise ValueError(f"`{key}` must be provided")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{key}` must be an integer (got {value!r})") from exc
    if min_value is not None and result < min_value:
        raise ValueError(f"`{key}` must be >= {min_value} (got {result})")
    if max_value is not None and result > max_value:
        raise ValueError(f"`{key}` must be <= {max_value} (got {result})")
    return result


def _coerce_float(value: Any, key: str, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{key}` must be a float (got {value!r})") from exc
    if min_value is not None and result < min_value:
        raise ValueError(f"`{key}` must be >= {min_value} (got {result})")
    if max_value is not None and result > max_value:
        raise ValueError(f"`{key}` must be <= {max_value} (got {result})")
    return result


def _coerce_optional_int(value: Any, key: str, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return None
    return _coerce_int(value, key, min_value=min_value, max_value=max_value)


def _validate_timezone(tz_name: str) -> None:
    try:
        ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone `{tz_name}`") from exc


def _normalize_authorized_users(section: Any) -> Tuple[str, ...]:
    if section is None:
        return ()
    if isinstance(section, str):
        raw_items = [item.strip() for item in section.split(",")]
    elif isinstance(section, (list, tuple, set)):
        raw_items = [str(item).strip() for item in section]
    else:
        raise TypeError("`authorized_users` must be a list or comma-delimited string")

    normalized: list[str] = []
    for item in raw_items:
        if not item:
            continue
        normalized.append(item.lower())

    seen: set[str] = set()
    ordered: list[str] = []
    for token in normalized:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return tuple(ordered)


def _load_targets(
    section: Any,
    *,
    default_instrument: str,
    default_csv: Path,
    default_signals: Optional[Any],
    base_dir: Optional[Path] = None,
    openclaw_config: Optional[Path] = None,
) -> Tuple[PlannerTargetConfig, ...]:
    if section is None:
        inferred = _infer_targets_from_openclaw(
            default_instrument=default_instrument,
            default_csv=default_csv,
            default_signals=default_signals,
            openclaw_config=openclaw_config,
        )
        if inferred:
            return inferred
        return (
            PlannerTargetConfig(
                key=default_instrument,
                instrument=default_instrument,
                csv_path=default_csv,
                signals_path=_maybe_path(default_signals),
            ),
        )

    if not isinstance(section, (list, tuple)):
        raise TypeError("`targets` section must be a list of mappings")

    targets: list[PlannerTargetConfig] = []
    seen_keys: set[str] = set()

    for item in section:
        if not isinstance(item, dict):
            raise TypeError("Each target entry must be a mapping")

        key_source = item.get("key") or item.get("instrument") or default_instrument
        key = str(key_source).strip()
        if not key:
            raise ValueError("Planner target `key` is required")
        if key in seen_keys:
            raise ValueError(f"Duplicate planner target key `{key}`")
        seen_keys.add(key)

        instrument_raw = item.get("instrument") or default_instrument
        instrument = str(instrument_raw).strip().upper()
        if not instrument:
            raise ValueError(f"Planner target `{key}` missing `instrument`")

        csv_raw = item.get("csv_path") or default_csv
        csv_path = _resolve_path(Path(str(csv_raw)).expanduser(), base_dir=base_dir)

        signals_raw = item.get("signals_path")
        if signals_raw is None:
            signals_raw = default_signals
        signals_path = _resolve_optional_path(_maybe_path(signals_raw), base_dir=base_dir)

        notes_raw = item.get("notes")
        if isinstance(notes_raw, str):
            notes = notes_raw.strip() or None
        else:
            notes = None if notes_raw is None else str(notes_raw)
        audience = str(item.get("audience") or "pro").strip().lower() or "pro"
        channel_key_raw = item.get("channel_key")
        channel_key = str(channel_key_raw).strip() if channel_key_raw else None
        channel_id = _coerce_optional_int(item.get("channel_id"), f"targets[{key}].channel_id")

        targets.append(
            PlannerTargetConfig(
                key=key,
                instrument=instrument,
                csv_path=csv_path,
                signals_path=signals_path,
                notes=notes,
                audience=audience,
                channel_key=channel_key,
                channel_id=channel_id,
            )
        )

    if not targets:
        return (
            PlannerTargetConfig(
                key=default_instrument,
                instrument=default_instrument,
                csv_path=default_csv,
                signals_path=_maybe_path(default_signals),
            ),
        )

    return tuple(targets)


def _maybe_path(path_like: Any) -> Optional[Path]:
    if path_like in (None, ""):
        return None
    return Path(str(path_like)).expanduser()


def _infer_targets_from_openclaw(
    *,
    default_instrument: str,
    default_csv: Path,
    default_signals: Optional[Any],
    openclaw_config: Optional[Path],
) -> Tuple[PlannerTargetConfig, ...]:
    if openclaw_config is None or not Path(openclaw_config).exists():
        return ()
    try:
        routes = load_openclaw_routes(openclaw_config)
    except Exception:
        return ()
    targets: list[PlannerTargetConfig] = []
    for item in routes.planner_targets():
        channel_id = _coerce_optional_int(item.get("channel_id"), f"planner target {item.get('key')}.channel_id")
        targets.append(
            PlannerTargetConfig(
                key=str(item.get("key") or default_instrument),
                instrument=default_instrument,
                csv_path=default_csv,
                signals_path=_maybe_path(default_signals),
                audience=str(item.get("audience") or "pro"),
                channel_key=str(item.get("channel_key") or "premarket_planner"),
                channel_id=channel_id,
            )
        )
    return tuple(targets)


def _resolve_optional_path(path: Optional[Path], *, base_dir: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return _resolve_path(path, base_dir=base_dir)


def _resolve_path(path: Path, *, base_dir: Optional[Path]) -> Path:
    if path.is_absolute():
        return path
    if base_dir is not None:
        return (base_dir / path).resolve()
    return (WORKSPACE_ROOT / path).resolve()


def _resolve_default_signals_path(instrument: str) -> Optional[Path]:
    instrument_slug = instrument.lower()
    live_dir = WORKSPACE_ROOT / "runs" / "live"
    candidates: list[Path] = []

    if live_dir.exists():
        priority_dirs = [
            live_dir / f"elite_{instrument_slug}",
            live_dir / f"pro_{instrument_slug}",
            live_dir / instrument_slug,
        ]
        for directory in priority_dirs:
            candidates.append(directory / "signals.csv")

        for child in sorted(live_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name.endswith(f"_{instrument_slug}"):
                candidates.append(child / "signals.csv")

        candidates.append(live_dir / "signals.csv")

    candidates.append(NA_ROOT / "runs" / "l3" / "signals.csv")

    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.expanduser()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None
