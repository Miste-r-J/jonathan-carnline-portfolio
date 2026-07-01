from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .config import PlannerConfig, load_planner_config
from ..integrations.route_config import load_openclaw_routes, resolve_automation_discord_token_details
from .scheduler import main as scheduler_main


def resolve_runtime_token(config: PlannerConfig) -> Optional[str]:
    details = resolve_runtime_token_details(config)
    return details["token"]


def resolve_runtime_token_details(config: PlannerConfig) -> Dict[str, Optional[str]]:
    route_details = resolve_automation_discord_token_details()
    if route_details.get("token"):
        return {
            "token": route_details.get("token"),
            "source": route_details.get("source"),
            "authoritative_file": route_details.get("env_file"),
            "available_sources": route_details.get("available_sources"),
        }
    sources = [
        (config.discord_token_env, os.getenv(config.discord_token_env)),
        ("AUTOMATION_DISCORD_BOT_TOKEN", os.getenv("AUTOMATION_DISCORD_BOT_TOKEN")),
        ("PREMARKET_DISCORD_TOKEN", os.getenv("PREMARKET_DISCORD_TOKEN")),
    ]
    token = None
    source = None
    available_sources: list[str] = []
    for key, value in sources:
        if value:
            available_sources.append(key)
        if value and token is None:
            token = value.removeprefix("Bot ").strip() or None
            source = key
    return {
        "token": token,
        "source": source,
        "authoritative_file": route_details.get("env_file"),
        "available_sources": available_sources,
    }


def probe_discord_token(token: Optional[str]) -> Dict[str, Any]:
    response = _safe_json_request("https://discord.com/api/v10/users/@me", token, timeout=10)
    if response["ok"]:
        return {"status": "ok", "detail": f"http {response.get('status_code', 200)}"}
    if response.get("detail") == "missing_token":
        return {"status": "fail", "detail": "missing"}
    if response.get("status_code") is not None:
        return {"status": "fail", "detail": f"http {response['status_code']}"}
    return {"status": "warning", "detail": str(response.get("detail") or "unknown")}


def _safe_json_request(url: str, token: Optional[str], *, timeout: int = 10) -> Dict[str, Any]:
    if not token:
        return {"ok": False, "status_code": None, "detail": "missing_token"}
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bot {token}", "User-Agent": "MisterJAutomationBot/1.0"},
            timeout=timeout,
        )
        payload = response.json() if response.text else {}
        return {
            "ok": response.ok,
            "status_code": response.status_code,
            "detail": "ok" if response.ok else f"http {response.status_code}",
            "payload": payload,
        }
    except requests.RequestException as exc:
        return {"ok": False, "status_code": None, "detail": str(exc)}


def _probe_target_access(token: Optional[str], channel_id: Optional[int]) -> Dict[str, Any]:
    if channel_id is None:
        return {"status": "unresolved", "detail": "missing_channel_id", "channel_id": channel_id}
    response = _safe_json_request(f"https://discord.com/api/v10/channels/{channel_id}", token)
    if response["ok"]:
        return {"status": "accessible", "detail": "ok", "channel_id": channel_id}
    status_code = response.get("status_code")
    if status_code == 403:
        return {"status": "missing access", "detail": "http 403", "channel_id": channel_id}
    if status_code == 404:
        return {"status": "not found", "detail": "http 404", "channel_id": channel_id}
    return {
        "status": "unresolved",
        "detail": response.get("detail") or "unknown",
        "channel_id": channel_id,
    }


def build_auth_diagnostics(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = load_planner_config(config_path)
    token_details = resolve_runtime_token_details(config)
    token = token_details["token"]
    identity = _safe_json_request("https://discord.com/api/v10/users/@me", token)
    routes = load_openclaw_routes(config.openclaw_discord_config) if config.openclaw_discord_config else None
    classification = "missing"
    if identity["ok"]:
        classification = "valid"
    elif identity.get("status_code") == 401:
        classification = "invalid_or_stale_token"
    elif identity.get("status_code") == 403:
        classification = "bot_auth_forbidden"
    elif identity.get("detail") != "missing_token":
        classification = "network_or_unknown"

    guild_diag: Dict[str, Any] = {
        "check_status": "skipped",
        "detail": "identity_failed",
        "guild_id": routes.guild_id if routes else None,
        "planner_channel_ids": [target.channel_id for target in config.targets if target.channel_id is not None],
    }
    if identity["ok"]:
        guilds = _safe_json_request("https://discord.com/api/v10/users/@me/guilds", token)
        guild_diag = {
            "check_status": "ok" if guilds["ok"] else "fail",
            "detail": guilds["detail"],
            "guild_id": routes.guild_id if routes else None,
            "planner_channel_ids": [target.channel_id for target in config.targets if target.channel_id is not None],
            "guild_present": bool(
                guilds["ok"]
                and routes
                and any(str(item.get("id")) == str(routes.guild_id) for item in (guilds.get("payload") or []))
            ),
        }
    target_access = [
        {
            "key": target.key,
            "audience": target.audience,
            "channel_key": target.channel_key,
            **(_probe_target_access(token, target.channel_id) if identity["ok"] else {
                "status": "unresolved",
                "detail": "identity_failed",
                "channel_id": target.channel_id,
            }),
        }
        for target in config.targets
    ]

    return {
        "token_present": bool(token),
        "token_source": token_details.get("source"),
        "authoritative_file": token_details.get("authoritative_file"),
        "available_sources": token_details.get("available_sources") or [],
        "discord_auth_classification": classification,
        "identity_probe": {
            "status": "ok" if identity["ok"] else "fail",
            "detail": identity["detail"],
            "status_code": identity.get("status_code"),
            "bot_id": (identity.get("payload") or {}).get("id") if identity["ok"] else None,
            "bot_username": (identity.get("payload") or {}).get("username") if identity["ok"] else None,
        },
        "guild_probe": guild_diag,
        "target_access": target_access,
    }


def build_readiness_report(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = load_planner_config(config_path)
    checks: list[dict[str, Any]] = []
    token_details = resolve_runtime_token_details(config)
    token = token_details["token"]
    token_probe = probe_discord_token(token)
    checks.append(
        {
            "name": "discord_token",
            "status": "ok" if token else "fail",
            "detail": token_details.get("source") or config.discord_token_env,
        }
    )
    checks.append(
        {
            "name": "discord_token_probe",
            "status": token_probe["status"],
            "detail": token_probe["detail"],
        }
    )
    checks.append(
        {
            "name": "planner_targets",
            "status": "ok" if config.targets else "fail",
            "detail": [target.key for target in config.targets],
        }
    )
    missing_target_channels = [target.key for target in config.targets if target.channel_id is None and not config.webhook_url]
    checks.append(
        {
            "name": "target_channels",
            "status": "ok" if not missing_target_channels else "fail",
            "detail": "all configured" if not missing_target_channels else missing_target_channels,
        }
    )
    checks.append(
        {
            "name": "planner_data_csv",
            "status": "ok" if config.data.csv_path.exists() else "fail",
            "detail": str(config.data.csv_path),
        }
    )
    if config.signals_path is not None:
        checks.append(
            {
                "name": "planner_signals_path",
                "status": "ok" if Path(config.signals_path).exists() else "warning",
                "detail": str(config.signals_path),
            }
        )
    if config.openclaw_discord_config is not None:
        checks.append(
            {
                "name": "openclaw_route_map",
                "status": "ok" if Path(config.openclaw_discord_config).exists() else "warning",
                "detail": str(config.openclaw_discord_config),
            }
        )
    diagnostics = build_auth_diagnostics(config_path)
    target_access = diagnostics.get("target_access") or []
    accessible_targets = [item["key"] for item in target_access if item.get("status") == "accessible"]
    blocked_targets = [f"{item.get('key')}:{item.get('status')}" for item in target_access if item.get("status") != "accessible"]
    target_access_status = "ok"
    target_access_detail: Any = "all accessible"
    if blocked_targets:
        target_access_status = "warning"
        target_access_detail = blocked_targets
    checks.append(
        {
            "name": "target_access",
            "status": target_access_status,
            "detail": target_access_detail,
        }
    )
    checks.append(
        {
            "name": "news_source",
            "status": "ok" if config.news_csv_path and Path(config.news_csv_path).exists() else "warning",
            "detail": str(config.news_csv_path) if config.news_csv_path else "AUTOMATION_NEWS_CSV not configured",
        }
    )
    planner_targets_configured = bool(config.targets)
    has_accessible_target = bool(accessible_targets)
    return {
        "ok": (
            all(item["status"] in {"ok", "warning"} for item in checks)
            and bool(token)
            and token_probe["status"] != "fail"
            and (not planner_targets_configured or has_accessible_target or bool(config.webhook_url))
        ),
        "command_trigger": config.command_trigger,
        "targets": [
            {
                "key": target.key,
                "audience": target.audience,
                "channel_key": target.channel_key,
                "channel_id": target.channel_id,
            }
            for target in config.targets
        ],
        "checks": checks,
        "diagnostics": diagnostics,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Premarket planner runtime wrapper with readiness checks.")
    parser.add_argument("--config", default=None, help="Optional planner config path")
    parser.add_argument("--readiness", action="store_true", help="Print readiness report and exit")
    parser.add_argument("--diagnostics", action="store_true", help="Print Discord auth diagnostics and exit")
    parser.add_argument("--log-level", default="INFO", help="Logging level passed through to scheduler")
    args = parser.parse_args(argv)

    config = load_planner_config(args.config)
    token = resolve_runtime_token(config)
    if args.diagnostics:
        print(json.dumps(build_auth_diagnostics(args.config), indent=2, sort_keys=True))
        return 0
    report = build_readiness_report(args.config)
    if args.readiness:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2
    if not report["ok"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    scheduler_main(["--log-level", args.log_level] + (["--token", token] if token else []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
