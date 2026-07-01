from __future__ import annotations

import json
from pathlib import Path

from trading_system.runtime_engine.premarket_planner import runtime


def _write_openclaw_config(root: Path) -> Path:
    path = root / "openclaw_discord_config.json"
    payload = {
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
    path.write_text(json.dumps(payload), encoding="utf-8")
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
            ]
        ),
        encoding="utf-8",
    )
    (root / "es.csv").write_text("datetime,open,high,low,close,volume\n", encoding="utf-8")
    return path


def test_build_auth_diagnostics_reports_source_without_token(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_openclaw_config(tmp_path)
    planner_path = _write_planner_yaml(tmp_path)
    monkeypatch.setenv("OPENCLAW_DISCORD_CONFIG", str(config_path))
    monkeypatch.setenv("PREMARKET_DISCORD_TOKEN", "secret-token-value")

    monkeypatch.setattr(
        runtime,
        "_safe_json_request",
        lambda url, token, timeout=10: {"ok": False, "status_code": 403, "detail": "http 403"},
    )

    report = runtime.build_auth_diagnostics(str(planner_path))

    assert report["token_present"] is True
    assert report["token_source"] in {
        "discord-mcp.env:AUTOMATION_DISCORD_TOKEN",
        "discord-mcp.env:DISCORD_TOKEN",
        "discord-mcp.env:PREMARKET_DISCORD_TOKEN",
        "PREMARKET_DISCORD_TOKEN",
    }
    assert "PREMARKET_DISCORD_TOKEN" in report["available_sources"]
    assert report["discord_auth_classification"] == "bot_auth_forbidden"
    assert "secret-token-value" not in json.dumps(report)


def test_build_auth_diagnostics_reports_target_access(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_openclaw_config(tmp_path)
    planner_path = _write_planner_yaml(tmp_path)
    monkeypatch.setenv("OPENCLAW_DISCORD_CONFIG", str(config_path))
    monkeypatch.setenv("AUTOMATION_DISCORD_TOKEN", "secret-token-value")

    def _fake_request(url: str, token: str | None, timeout: int = 10) -> dict:
        if url.endswith("/users/@me"):
            return {"ok": True, "status_code": 200, "detail": "ok", "payload": {"id": "bot-1", "username": "planner"}}
        if url.endswith("/users/@me/guilds"):
            return {"ok": True, "status_code": 200, "detail": "ok", "payload": [{"id": "guild-123"}]}
        if url.endswith("/channels/1001"):
            return {"ok": False, "status_code": 403, "detail": "http 403", "payload": {}}
        if url.endswith("/channels/2001"):
            return {"ok": True, "status_code": 200, "detail": "ok", "payload": {"id": "2001"}}
        raise AssertionError(url)

    monkeypatch.setattr(runtime, "_safe_json_request", _fake_request)

    report = runtime.build_auth_diagnostics(str(planner_path))

    assert report["guild_probe"]["guild_present"] is True
    assert report["target_access"] == [
        {
            "key": "planner-pro",
            "audience": "pro",
            "channel_key": "premarket_planner",
            "status": "missing access",
            "detail": "http 403",
            "channel_id": 1001,
        },
        {
            "key": "planner-elite",
            "audience": "elite",
            "channel_key": "chat",
            "status": "accessible",
            "detail": "ok",
            "channel_id": 2001,
        },
    ]


def test_readiness_reports_target_access_warning(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_openclaw_config(tmp_path)
    planner_path = _write_planner_yaml(tmp_path)
    monkeypatch.setenv("OPENCLAW_DISCORD_CONFIG", str(config_path))
    monkeypatch.setenv("AUTOMATION_DISCORD_TOKEN", "secret-token-value")

    def _fake_request(url: str, token: str | None, timeout: int = 10) -> dict:
        if url.endswith("/users/@me"):
            return {"ok": True, "status_code": 200, "detail": "ok", "payload": {"id": "bot-1", "username": "planner"}}
        if url.endswith("/users/@me/guilds"):
            return {"ok": True, "status_code": 200, "detail": "ok", "payload": [{"id": "guild-123"}]}
        if url.endswith("/channels/1001"):
            return {"ok": False, "status_code": 403, "detail": "http 403", "payload": {}}
        if url.endswith("/channels/2001"):
            return {"ok": True, "status_code": 200, "detail": "ok", "payload": {"id": "2001"}}
        raise AssertionError(url)

    monkeypatch.setattr(runtime, "_safe_json_request", _fake_request)

    report = runtime.build_readiness_report(str(planner_path))

    target_access = next(check for check in report["checks"] if check["name"] == "target_access")
    assert target_access["status"] == "warning"
    assert target_access["detail"] == ["planner-pro:missing access"]
    assert report["ok"] is True


def test_readiness_stays_ok_when_all_targets_blocked(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_openclaw_config(tmp_path)
    planner_path = _write_planner_yaml(tmp_path)
    monkeypatch.setenv("OPENCLAW_DISCORD_CONFIG", str(config_path))
    monkeypatch.setenv("AUTOMATION_DISCORD_TOKEN", "secret-token-value")

    def _fake_request(url: str, token: str | None, timeout: int = 10) -> dict:
        if url.endswith("/users/@me"):
            return {"ok": True, "status_code": 200, "detail": "ok", "payload": {"id": "bot-1", "username": "planner"}}
        if url.endswith("/users/@me/guilds"):
            return {"ok": True, "status_code": 200, "detail": "ok", "payload": [{"id": "guild-123"}]}
        if url.endswith("/channels/1001") or url.endswith("/channels/2001"):
            return {"ok": False, "status_code": 403, "detail": "http 403", "payload": {}}
        raise AssertionError(url)

    monkeypatch.setattr(runtime, "_safe_json_request", _fake_request)

    report = runtime.build_readiness_report(str(planner_path))

    target_access = next(check for check in report["checks"] if check["name"] == "target_access")
    assert target_access["status"] == "warning"
    assert target_access["detail"] == ["planner-pro:missing access", "planner-elite:missing access"]
    assert report["ok"] is True
