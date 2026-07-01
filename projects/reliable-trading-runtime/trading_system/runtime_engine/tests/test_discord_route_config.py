from __future__ import annotations

import json
from pathlib import Path

from trading_system.runtime_engine.integrations.discord_emitter import DiscordEmitter
from trading_system.runtime_engine.integrations.route_config import load_openclaw_routes
from trading_system.runtime_engine.premarket_planner.config import load_config


def _write_openclaw_config(root: Path) -> Path:
    path = root / "openclaw_discord_config.json"
    payload = {
        "openclaw_discord": {
            "mode": "shadow",
            "guild_id": "guild-1",
            "channels": {
                "signals": "signals-generic",
                "health": "health-1",
                "alerts": "alerts-1",
                "audit": "audit-1",
                "ops_updates": "ops-1",
                "tiered_channels": {
                    "pro": {
                        "signals_es": "1001",
                        "signal_recap": "1002",
                        "premarket_planner": "1003",
                    },
                    "elite": {
                        "signals_es": "2001",
                        "signal_recap": "2002",
                        "chat": "2003",
                    },
                },
                "admin": {
                    "mod_log": "3001",
                    "admin_chat": "3002",
                    "watchdog_logs": "3003",
                },
            },
            "escalation": {},
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


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict]] = []

    def post(self, url: str, json: dict | None = None, headers: dict | None = None, timeout: int | None = None):
        self.calls.append((url, json or {}, headers or {}))

        class _Resp:
            status_code = 200
            text = "{}"

        return _Resp()


def test_openclaw_routes_resolve_signal_and_planner_targets(tmp_path: Path) -> None:
    config_path = _write_openclaw_config(tmp_path)
    routes = load_openclaw_routes(config_path)

    assert routes.route_channels(event_type="signal", instrument="ES") == ["1001", "2001"]
    assert routes.route_channels(event_type="fill", instrument="ES") == ["1001", "2001", "1002", "2002"]
    assert [item["channel_id"] for item in routes.planner_targets()] == ["1003", "2003"]


def test_planner_config_infers_targets_from_openclaw_map(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_openclaw_config(tmp_path)
    planner_path = _write_planner_yaml(tmp_path)
    monkeypatch.setenv("OPENCLAW_DISCORD_CONFIG", str(config_path))

    config = load_config(str(planner_path))

    assert [target.key for target in config.targets] == ["planner-pro", "planner-elite"]
    assert config.targets[0].channel_id == 1003
    assert config.targets[0].channel_key == "premarket_planner"
    assert config.targets[1].channel_key == "chat"


def test_planner_routes_prefer_elite_premarket_channel_when_present(tmp_path: Path) -> None:
    config_path = _write_openclaw_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["openclaw_discord"]["channels"]["tiered_channels"]["elite"]["premarket_planner"] = "2004"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    routes = load_openclaw_routes(config_path)

    assert routes.route_channels(event_type="premarket_plan") == ["1003", "2004"]
    assert routes.planner_targets()[1]["channel_id"] == "2004"
    assert routes.planner_targets()[1]["channel_key"] == "premarket_planner"


def test_planner_config_uses_new_es_intraday_csv(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_openclaw_config(tmp_path)
    planner_path = _write_planner_yaml(tmp_path)
    planner_text = planner_path.read_text(encoding="utf-8").replace(
        "  csv_path: es.csv",
        "  csv_path: ../../../data/intraday/es/ES.csv",
    )
    planner_path.write_text(planner_text, encoding="utf-8")
    monkeypatch.setenv("OPENCLAW_DISCORD_CONFIG", str(config_path))

    config = load_config(str(planner_path))

    assert str(config.data.csv_path).endswith(r"data\intraday\es\ES.csv")
    assert all(target.csv_path == config.data.csv_path for target in config.targets)


def test_discord_emitter_posts_to_channel_routes_without_webhook(tmp_path: Path) -> None:
    config_path = _write_openclaw_config(tmp_path)
    routes = load_openclaw_routes(config_path)
    emitter = DiscordEmitter(webhook_url=None, routes=routes)
    fake = _FakeSession()
    emitter._client = ("requests", fake)
    emitter._discord_token = "token-123"

    emitter.publish_event({"event_type": "signal", "type": "OPEN", "side": "LONG", "instrument": "ES", "price": 5250.0})

    assert len(fake.calls) == 2
    urls = [item[0] for item in fake.calls]
    assert urls[0].endswith("/channels/1001/messages")
    assert urls[1].endswith("/channels/2001/messages")
    assert all(call[2]["Authorization"] == "Bot token-123" for call in fake.calls)
