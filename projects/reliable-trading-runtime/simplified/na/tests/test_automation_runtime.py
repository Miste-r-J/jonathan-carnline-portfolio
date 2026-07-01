from __future__ import annotations

import json
from pathlib import Path

from na.automation_runtime import AUTOMATION_COMMANDS
from na.discord_addons.route_config import resolve_automation_discord_token_details
from na.premarket_planner.bot import PremarketPlannerBot
from na.premarket_planner.config import load_config
from na.premarket_planner.news import build_news_summary
from na.premarket_planner.runtime import build_readiness_report


def _write_openclaw_config(root: Path) -> Path:
    path = root / "openclaw_discord_config.json"
    path.write_text(
        json.dumps(
            {
                "openclaw_discord": {
                    "guild_id": "guild-123",
                    "channels": {
                        "tiered_channels": {
                            "pro": {"premarket_planner": "1001"},
                            "elite": {"chat": "2001"},
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_planner_yaml(root: Path) -> Path:
    path = root / "planner.yaml"
    path.write_text(
        "\n".join(
            [
                "enabled: true",
                "instrument: ES",
                "session_tz: America/Denver",
                "rth_start: '06:30'",
                "rth_end: '12:59'",
                "emit_time_local: '06:30'",
                "data:",
                "  csv_path: es.csv",
                "output:",
                "  discord_webhook: ''",
                "  round_decimals: 2",
                "backfill:",
                "  enabled: true",
                "  hours: 24",
                "news_csv_path: news.csv",
            ]
        ),
        encoding="utf-8",
    )
    (root / "es.csv").write_text("datetime,open,high,low,close,volume\n", encoding="utf-8")
    (root / "news.csv").write_text("time_local,impact,title,currency\n2026-04-22 08:00,high,CPI,USD\n", encoding="utf-8")
    return path


def test_automation_token_resolution_prefers_new_token(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / "discord-mcp.env"
    env_file.write_text(
        "\n".join(
            [
                "AUTOMATION_DISCORD_TOKEN=automation-secret",
                "DISCORD_TOKEN=chat-secret",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCLAW_WORKSPACE", str(tmp_path))

    details = resolve_automation_discord_token_details()

    assert details["token"] == "automation-secret"
    assert details["source"] == "discord-mcp.env:AUTOMATION_DISCORD_TOKEN"


def test_automation_bot_registers_planner_and_news_commands(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_openclaw_config(tmp_path)
    planner_path = _write_planner_yaml(tmp_path)
    monkeypatch.setenv("OPENCLAW_DISCORD_CONFIG", str(config_path))

    config = load_config(str(planner_path))
    bot = PremarketPlannerBot(config)

    assert sorted(command.name for command in bot.tree.get_commands()) == ["news", "planner"]


def test_news_summary_reports_missing_source(tmp_path: Path) -> None:
    planner_path = _write_planner_yaml(tmp_path)
    planner_path.write_text(planner_path.read_text(encoding="utf-8").replace("news_csv_path: news.csv", ""), encoding="utf-8")
    (tmp_path / "es.csv").write_text("datetime,open,high,low,close,volume\n", encoding="utf-8")
    config = load_config(str(planner_path))

    summary = build_news_summary(config)

    assert "No news source configured" in summary


def test_automation_readiness_includes_news_source(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_openclaw_config(tmp_path)
    planner_path = _write_planner_yaml(tmp_path)
    monkeypatch.setenv("OPENCLAW_DISCORD_CONFIG", str(config_path))
    monkeypatch.setattr(
        "na.premarket_planner.runtime._safe_json_request",
        lambda url, token, timeout=10: {"ok": True, "status_code": 200, "detail": "ok", "payload": {"id": "bot-1", "username": "auto"}} if url.endswith("/users/@me") else {"ok": True, "status_code": 200, "detail": "ok", "payload": [{"id": "guild-123"}]},
    )
    monkeypatch.setenv("AUTOMATION_DISCORD_TOKEN", "automation-secret")

    report = build_readiness_report(str(planner_path))

    assert any(check["name"] == "news_source" and check["status"] == "ok" for check in report["checks"])
    assert AUTOMATION_COMMANDS == ["/planner", "/news"]
