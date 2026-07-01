from __future__ import annotations

from datetime import time
from pathlib import Path
from types import SimpleNamespace

import discord
import pytest

from na.premarket_planner.bot import PlannerTargetPublishError, PremarketPlannerBot
from na.premarket_planner.config import (
    BackfillConfig,
    DataConfig,
    MetricsConfig,
    OutputConfig,
    PlannerConfig,
    PlannerTargetConfig,
    WindowsConfig,
)
from na.premarket_planner.runtime import build_readiness_report


class _FakePlanEmbed:
    def to_discord_embed(self) -> discord.Embed:
        return discord.Embed(title="Planner")


class _FakePayload:
    def __init__(self) -> None:
        self.embed = _FakePlanEmbed()


class _FakeInteraction:
    def __init__(self) -> None:
        self.channel_id = 1001
        self.user = SimpleNamespace(id=1, name="tester", global_name="tester", display_name="tester")


def _make_config(tmp_path: Path) -> PlannerConfig:
    csv_path = tmp_path / "ES.csv"
    csv_path.write_text("datetime,open,high,low,close,volume\n", encoding="utf-8")
    return PlannerConfig(
        enabled=True,
        instrument="ES",
        session_tz="America/Denver",
        rth_start=time(hour=6, minute=30),
        rth_end=time(hour=12, minute=59),
        emit_time_local=time(hour=6, minute=30),
        data=DataConfig(csv_path=csv_path, csv_timezone="America/Denver"),
        windows=WindowsConfig(eth_start_hour=13, eth_end_hour=7, eth_end_minute=29, min_eth_bars=30, last_hours_fallback=8),
        metrics=MetricsConfig(atr_len=14, atr_timeframe="5m", compute_rth_vwap=True),
        output=OutputConfig(discord_webhook="", round_decimals=2),
        backfill=BackfillConfig(enabled=False, hours=24),
        dispatch_interval_minutes=5.0,
        discord_channel_id=1001,
        targets=(
            PlannerTargetConfig(key="planner-pro", instrument="ES", csv_path=csv_path, audience="pro", channel_key="premarket_planner", channel_id=1001),
            PlannerTargetConfig(key="planner-elite", instrument="ES", csv_path=csv_path, audience="elite", channel_key="chat", channel_id=2001),
        ),
    )


@pytest.mark.asyncio
async def test_dispatch_partial_success_when_one_target_forbidden(tmp_path: Path, monkeypatch) -> None:
    bot = PremarketPlannerBot(_make_config(tmp_path))
    replies: list[str] = []

    monkeypatch.setattr("na.premarket_planner.bot.build_plan_payload", lambda config, target: _FakePayload())

    async def _fake_resolve_channel(channel_id: int):
        return SimpleNamespace(id=channel_id)

    async def _fake_publish_plan(channel, target_cfg, content, embed):
        if target_cfg.key == "planner-pro":
            raise PlannerTargetPublishError("missing access", "Missing access to send planner message")

    async def _fake_reply(interaction, content: str, ephemeral: bool = True):
        replies.append(content)

    bot._resolve_channel = _fake_resolve_channel  # type: ignore[method-assign]
    bot._publish_plan = _fake_publish_plan  # type: ignore[method-assign]
    bot._reply_interaction = _fake_reply  # type: ignore[method-assign]

    success = await bot._dispatch_plan(trigger="/planner", interaction=_FakeInteraction())

    assert success is True
    assert replies == ["⚠️ Premarket plans refreshed partially. Failed targets: planner-pro (missing access)."]


@pytest.mark.asyncio
async def test_dispatch_returns_false_when_all_targets_fail(tmp_path: Path, monkeypatch) -> None:
    bot = PremarketPlannerBot(_make_config(tmp_path))
    replies: list[str] = []

    monkeypatch.setattr("na.premarket_planner.bot.build_plan_payload", lambda config, target: _FakePayload())

    async def _fake_resolve_channel(channel_id: int):
        return None

    async def _fake_reply(interaction, content: str, ephemeral: bool = True):
        replies.append(content)

    bot._resolve_channel = _fake_resolve_channel  # type: ignore[method-assign]
    bot._reply_interaction = _fake_reply  # type: ignore[method-assign]

    success = await bot._dispatch_plan(trigger="/planner", interaction=_FakeInteraction())

    assert success is False
    assert replies == ["⚠️ Unable to refresh ES: No target channel or webhook available"]


@pytest.mark.asyncio
async def test_on_ready_does_not_raise_when_targets_inaccessible(tmp_path: Path, monkeypatch) -> None:
    bot = PremarketPlannerBot(_make_config(tmp_path))
    monkeypatch.setattr("na.premarket_planner.bot.build_plan_payload", lambda config, target: _FakePayload())

    async def _fake_resolve_channel(channel_id: int):
        return None

    bot._resolve_channel = _fake_resolve_channel  # type: ignore[method-assign]
    bot._connection.user = SimpleNamespace(name="planner", id=123)

    await bot.on_ready()

    assert bot._startup_dispatched is True


@pytest.mark.asyncio
async def test_periodic_dispatch_does_not_raise_when_dispatch_fails(tmp_path: Path, monkeypatch) -> None:
    bot = PremarketPlannerBot(_make_config(tmp_path))
    bot._last_run_at = None
    monkeypatch.setattr("na.premarket_planner.bot.datetime", SimpleNamespace(
        now=lambda tz=None: __import__("datetime").datetime(2026, 4, 22, 7, 0, tzinfo=tz),
        combine=__import__("datetime").datetime.combine,
    ))

    async def _fake_dispatch_plan(*, trigger: str, interaction=None) -> bool:
        return False

    bot._dispatch_plan = _fake_dispatch_plan  # type: ignore[method-assign]

    await bot.periodic_dispatch.coro(bot)

    assert bot._last_run_at is None


def test_readiness_report_fails_when_all_targets_missing_access(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "premarket_planner.yaml"
    csv_path = tmp_path / "ES.csv"
    csv_path.write_text("datetime,open,high,low,close,volume\n", encoding="utf-8")
    config_path.write_text(
        f"""
enabled: true
instrument: ES
session_tz: America/Denver
rth_start: "06:30"
rth_end: "12:59"
emit_time_local: "06:30"
data:
  csv_path: {csv_path}
output:
  discord_webhook: ""
targets:
  - key: planner-pro
    instrument: ES
    audience: pro
    channel_key: premarket_planner
    channel_id: 1001
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "na.premarket_planner.runtime.probe_discord_token",
        lambda token: {"status": "ok", "detail": "http 200"},
    )
    monkeypatch.setattr(
        "na.premarket_planner.runtime.build_auth_diagnostics",
        lambda config_path=None: {
            "target_access": [
                {
                    "key": "planner-pro",
                    "status": "missing access",
                    "detail": "http 403",
                    "channel_id": 1001,
                }
            ]
        },
    )

    report = build_readiness_report(str(config_path))

    assert report["ok"] is False
