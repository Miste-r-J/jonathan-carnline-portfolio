from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

OPENCLAW_CONFIG_ENV = "OPENCLAW_DISCORD_CONFIG"
OPENCLAW_WORKSPACE_ENV = "OPENCLAW_WORKSPACE"
OPENCLAW_ROUTE_MODE_ENV = "OPENCLAW_DISCORD_REPORT_MODE"


def default_openclaw_workspace() -> Path:
    raw = os.environ.get(OPENCLAW_WORKSPACE_ENV)
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".openclaw" / "workspace"


def default_openclaw_config_path() -> Path:
    raw = os.environ.get(OPENCLAW_CONFIG_ENV)
    if raw:
        return Path(raw).expanduser()
    return default_openclaw_workspace() / "openclaw_discord_config.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_discord_token(workspace_root: Optional[Path] = None) -> Optional[str]:
    return resolve_discord_token_details(workspace_root)["token"]


def resolve_automation_discord_token(workspace_root: Optional[Path] = None) -> Optional[str]:
    return resolve_automation_discord_token_details(workspace_root)["token"]


def _resolve_token_details(
    sources: Iterable[tuple[str, Optional[str]]],
    *,
    workspace: Path,
    env_file: Path,
) -> dict[str, Optional[str]]:
    token = None
    source = None
    available_sources: list[str] = []
    for key, value in sources:
        if value:
            available_sources.append(key)
        if value and token is None:
            token = value
            source = key
    if not token:
        return {
            "token": None,
            "source": None,
            "workspace_root": str(workspace),
            "env_file": str(env_file),
            "available_sources": available_sources,
        }
    normalized = token.removeprefix("Bot ").strip() or None
    return {
        "token": normalized,
        "source": source,
        "workspace_root": str(workspace),
        "env_file": str(env_file),
        "available_sources": available_sources,
    }


def resolve_discord_token_details(workspace_root: Optional[Path] = None) -> dict[str, Optional[str]]:
    workspace = workspace_root or default_openclaw_workspace()
    env_file = workspace / "discord-mcp.env"
    env_values = _load_env_file(env_file)
    sources = [
        (f"{env_file.name}:DISCORD_BOT_TOKEN", env_values.get("DISCORD_BOT_TOKEN")),
        (f"{env_file.name}:DISCORD_TOKEN", env_values.get("DISCORD_TOKEN")),
        ("DISCORD_BOT_TOKEN", os.environ.get("DISCORD_BOT_TOKEN")),
        ("DISCORD_TOKEN", os.environ.get("DISCORD_TOKEN")),
    ]
    return _resolve_token_details(sources, workspace=workspace, env_file=env_file)


def resolve_automation_discord_token_details(workspace_root: Optional[Path] = None) -> dict[str, Optional[str]]:
    workspace = workspace_root or default_openclaw_workspace()
    env_file = workspace / "discord-mcp.env"
    env_values = _load_env_file(env_file)
    sources = [
        (f"{env_file.name}:AUTOMATION_DISCORD_TOKEN", env_values.get("AUTOMATION_DISCORD_TOKEN")),
        (f"{env_file.name}:AUTOMATION_DISCORD_BOT_TOKEN", env_values.get("AUTOMATION_DISCORD_BOT_TOKEN")),
        (f"{env_file.name}:PREMARKET_DISCORD_TOKEN", env_values.get("PREMARKET_DISCORD_TOKEN")),
        ("AUTOMATION_DISCORD_TOKEN", os.environ.get("AUTOMATION_DISCORD_TOKEN")),
        ("AUTOMATION_DISCORD_BOT_TOKEN", os.environ.get("AUTOMATION_DISCORD_BOT_TOKEN")),
        ("PREMARKET_DISCORD_TOKEN", os.environ.get("PREMARKET_DISCORD_TOKEN")),
    ]
    return _resolve_token_details(sources, workspace=workspace, env_file=env_file)


def _dedupe(items: Iterable[Optional[str]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item:
            continue
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


@dataclass(frozen=True)
class OpenClawDiscordRoutes:
    config_path: Path
    workspace_root: Path
    mode: str
    guild_id: Optional[str]
    channels: dict[str, Any]
    escalation: dict[str, Any]
    token: Optional[str]

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "OpenClawDiscordRoutes":
        config_path = Path(path).expanduser() if path else default_openclaw_config_path()
        workspace_root = config_path.parent
        data = _load_json(config_path)
        section = data.get("openclaw_discord", data)
        if not isinstance(section, dict):
            section = {}
        return cls(
            config_path=config_path,
            workspace_root=workspace_root,
            mode=str(section.get("mode") or os.environ.get(OPENCLAW_ROUTE_MODE_ENV) or "shadow").strip().lower(),
            guild_id=str(section.get("guild_id")).strip() if section.get("guild_id") else None,
            channels=dict(section.get("channels") or {}),
            escalation=dict(section.get("escalation") or {}),
            token=resolve_automation_discord_token(workspace_root),
        )

    def resolve_channel_id(self, channel_key: Optional[str], *, audience: str = "pro") -> Optional[str]:
        if not channel_key:
            return None
        channels = dict(self.channels or {})
        if channel_key in channels and isinstance(channels[channel_key], str):
            return channels[channel_key]
        tiered = channels.get("tiered_channels")
        if isinstance(tiered, dict):
            audience_group = tiered.get(audience)
            if isinstance(audience_group, dict):
                if channel_key in audience_group and isinstance(audience_group[channel_key], str):
                    return audience_group[channel_key]
                if channel_key == "signals":
                    candidate = audience_group.get("signals_es")
                    if isinstance(candidate, str):
                        return candidate
        for group_name in ("education", "community", "admin"):
            group = channels.get(group_name)
            if isinstance(group, dict) and isinstance(group.get(channel_key), str):
                return group[channel_key]
        return None

    def resolve_signal_channels(self, instrument: Optional[str], *, include_recap: bool = False) -> list[str]:
        symbol = str(instrument or "ES").strip().upper()
        signal_key = f"signals_{symbol.lower()}"
        channels = [
            self.resolve_channel_id(signal_key, audience="pro"),
            self.resolve_channel_id(signal_key, audience="elite"),
        ]
        if include_recap:
            channels.extend(
                [
                    self.resolve_channel_id("signal_recap", audience="pro"),
                    self.resolve_channel_id("signal_recap", audience="elite"),
                ]
            )
        return _dedupe(channels)

    def planner_targets(self) -> list[dict[str, str]]:
        targets: list[dict[str, str]] = []
        planner_id = self.resolve_channel_id("premarket_planner", audience="pro")
        if planner_id:
            targets.append(
                {
                    "key": "planner-pro",
                    "audience": "pro",
                    "channel_key": "premarket_planner",
                    "channel_id": planner_id,
                }
            )
        elite_channel_key = "premarket_planner"
        elite_channel_id = self.resolve_channel_id(elite_channel_key, audience="elite")
        if not elite_channel_id:
            elite_channel_key = "chat"
            elite_channel_id = self.resolve_channel_id(elite_channel_key, audience="elite")
        if elite_channel_id:
            targets.append(
                {
                    "key": "planner-elite",
                    "audience": "elite",
                    "channel_key": elite_channel_key,
                    "channel_id": elite_channel_id,
                }
            )
        return targets

    def route_channels(
        self,
        *,
        event_type: Optional[str],
        instrument: Optional[str] = None,
        audience: Optional[str] = None,
        channel_key: Optional[str] = None,
        channel_keys: Optional[Iterable[str]] = None,
        include_recap: bool = False,
    ) -> list[str]:
        if channel_keys:
            return _dedupe(self.resolve_channel_id(item, audience=audience or "pro") for item in channel_keys)
        if channel_key:
            return _dedupe([self.resolve_channel_id(channel_key, audience=audience or "pro")])

        event_name = str(event_type or "signal").strip().lower()
        if event_name == "premarket_plan":
            elite_planner = self.resolve_channel_id("premarket_planner", audience="elite")
            return _dedupe(
                [
                    self.resolve_channel_id("premarket_planner", audience="pro"),
                    elite_planner or self.resolve_channel_id("chat", audience="elite"),
                ]
            )
        if event_name in {"signal", "order_ack"}:
            return self.resolve_signal_channels(instrument, include_recap=False)
        if event_name in {"fill", "signal_recap", "daily_summary", "performance_report"}:
            return self.resolve_signal_channels(instrument, include_recap=True)
        if event_name == "lockout":
            return _dedupe(
                [
                    self.resolve_channel_id("alerts", audience="ops"),
                    self.resolve_channel_id("ops_updates", audience="ops"),
                    self.resolve_channel_id("mod_log", audience="admin"),
                ]
            )
        if event_name == "health_update":
            return _dedupe(
                [
                    self.resolve_channel_id("health", audience="ops"),
                    self.resolve_channel_id("ops_updates", audience="ops"),
                ]
            )
        if event_name == "alert":
            return _dedupe(
                [
                    self.resolve_channel_id("alerts", audience="ops"),
                    self.resolve_channel_id("watchdog_logs", audience="admin"),
                    self.resolve_channel_id("admin_chat", audience="admin"),
                ]
            )
        if event_name == "audit_event":
            return _dedupe(
                [
                    self.resolve_channel_id("audit", audience="ops"),
                    self.resolve_channel_id("mod_log", audience="admin"),
                ]
            )
        if event_name == "admin_command":
            return _dedupe(
                [
                    self.resolve_channel_id("ops_updates", audience="admin"),
                    self.resolve_channel_id("claw_admin", audience="admin"),
                ]
            )
        return []


def load_openclaw_routes(path: Optional[str | Path] = None) -> OpenClawDiscordRoutes:
    return OpenClawDiscordRoutes.load(path)
