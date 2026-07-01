from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from trading_system.runtime_engine.integrations.cli.live_trading_runtime import LiveCSVStreamer
from trading_system.runtime_engine.integrations.discord_delivery import DiscordDeliveryError, DiscordDeliveryQueue
from trading_system.runtime_engine.integrations.trade_telemetry import DiscordTelemetryReporter, TradeTelemetryStore


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_discord_delivery_queue_retries_then_succeeds(tmp_path: Path) -> None:
    attempts = {"count": 0}

    def flaky_send(payload: dict) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise DiscordDeliveryError("rate limited", retry_after=0.01, retryable=True, status_code=429)

    queue = DiscordDeliveryQueue(
        send_callable=flaky_send,
        dead_letter_path=tmp_path / "dead_letter.jsonl",
        max_attempts=3,
        max_total_wait_sec=2.0,
        global_rate_per_sec=100.0,
        channel_rate_per_sec=100.0,
    )

    result = queue.submit(channel_id="chan-1", kind="message", payload={"content": "hello"}, dedupe_key="k1")

    assert result["status"] == "sent"
    assert result["attempt_count"] == 2
    assert attempts["count"] == 2
    assert not (tmp_path / "dead_letter.jsonl").exists()


def test_discord_delivery_queue_dead_letters_non_retryable_failure(tmp_path: Path) -> None:
    def hard_fail(_payload: dict) -> None:
        raise DiscordDeliveryError("bad request", retryable=False, status_code=400, body="invalid embed")

    queue = DiscordDeliveryQueue(
        send_callable=hard_fail,
        dead_letter_path=tmp_path / "dead_letter.jsonl",
        max_attempts=3,
        max_total_wait_sec=1.0,
        global_rate_per_sec=100.0,
        channel_rate_per_sec=100.0,
    )

    result = queue.submit(channel_id="chan-2", kind="embed", payload={"content": "bad"}, dedupe_key="k2")
    dead_letters = _read_jsonl(tmp_path / "dead_letter.jsonl")

    assert result["status"] == "dead_letter"
    assert len(dead_letters) == 1
    assert dead_letters[0]["error"]["status_code"] == 400


def test_trade_telemetry_store_persists_live_state_and_shadow_report(tmp_path: Path) -> None:
    reporter = DiscordTelemetryReporter(router=object(), out_dir=tmp_path, mode="shadow")
    store = TradeTelemetryStore(out_dir=tmp_path, reporter=reporter)

    doc = store.emit(
        "FILL",
        {
            "ts": "2026-04-20T20:00:00Z",
            "run_id": "RUN-1",
            "account": "Sim101",
            "instrument": "ES",
            "signal_id": "SIG-1",
            "client_order_id": "CID-1",
            "side": "LONG",
            "qty": 1,
            "price": 5250.25,
            "position_qty": 1,
            "realized_pnl": 125.0,
            "unrealized_pnl": 45.0,
            "bridge_status": "up",
            "source": "live_trading_runtime",
        },
    )
    store.update_health("nt_bridge", status="up", detail={"freshness_sec": 1})
    store.mark_smoke_test()

    telemetry_rows = _read_jsonl(tmp_path / "trade_telemetry.jsonl")
    shadow_rows = _read_jsonl(tmp_path / "discord_reporter_shadow.jsonl")
    live_state = json.loads((tmp_path / "live_pnl_state.json").read_text(encoding="utf-8"))
    health = json.loads((tmp_path / "system_health.json").read_text(encoding="utf-8"))

    assert doc["event_type"] == "FILL"
    assert len(telemetry_rows) == 1
    assert len(shadow_rows) == 1
    assert live_state["realized_pnl"] == 125.0
    assert live_state["position_qty"] == 1
    assert health["telemetry"]["status"] == "up"
    assert health["nt_bridge"]["status"] == "up"
    assert health["last_smoke_test_ts"] is not None


def test_trade_telemetry_store_forces_flat_side_when_qty_zero(tmp_path: Path) -> None:
    store = TradeTelemetryStore(out_dir=tmp_path, reporter=None)
    store.update_live_pnl_state(
        {
            "ts": "2026-05-07T20:55:46-06:00",
            "run_id": "RUN-FLAT",
            "account": "DEMO",
            "instrument": "ES JUN26",
            "position_qty": 0,
            "side": "SHORT",
            "realized_pnl": -25.0,
            "unrealized_pnl": 0.0,
            "bridge_status": "connected",
        }
    )
    live_state = json.loads((tmp_path / "live_pnl_state.json").read_text(encoding="utf-8"))
    assert live_state["position_qty"] == 0.0
    assert live_state["side"] == "FLAT"


def test_live_trading_runtime_trade_telemetry_helpers_write_expected_files(tmp_path: Path) -> None:
    reporter = DiscordTelemetryReporter(router=object(), out_dir=tmp_path, mode="shadow")
    store = TradeTelemetryStore(out_dir=tmp_path, reporter=reporter)
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.trade_telemetry_store = store
    streamer._emit_trade_telemetry = LiveCSVStreamer._emit_trade_telemetry.__get__(streamer, LiveCSVStreamer)
    streamer._update_live_pnl_from_state = LiveCSVStreamer._update_live_pnl_from_state.__get__(streamer, LiveCSVStreamer)
    streamer.run_id = "RUN-STREAM"
    streamer.instrument_alias = "ES"
    streamer.exec_instrument = "ES 06-26"
    streamer.nt_account = "Sim101"
    streamer.nt_bridge = SimpleNamespace(is_connected=True)
    streamer.state = SimpleNamespace(
        last_signal_id="SIG-STREAM",
        last_nt_client_order_id="CID-STREAM",
        last_entry_side="LONG",
        daily_usd=210.5,
    )

    streamer._emit_trade_telemetry(
        "SIGNAL_EMITTED",
        {
            "ts": "2026-04-20T20:05:00Z",
            "price": 5251.0,
            "qty": 1,
            "meta": {"status": "emitted"},
        },
    )
    streamer._update_live_pnl_from_state(position_qty=1, unrealized_pnl=18.0, bridge_status="connected")

    telemetry_rows = _read_jsonl(tmp_path / "trade_telemetry.jsonl")
    live_state = json.loads((tmp_path / "live_pnl_state.json").read_text(encoding="utf-8"))

    assert len(telemetry_rows) == 1
    assert telemetry_rows[0]["run_id"] == "RUN-STREAM"
    assert telemetry_rows[0]["signal_id"] == "SIG-STREAM"
    assert telemetry_rows[0]["client_order_id"] == "CID-STREAM"
    assert live_state["realized_pnl"] == 210.5
    assert live_state["unrealized_pnl"] == 18.0


def test_live_trading_runtime_flat_snapshot_telemetry_does_not_inherit_stale_signal(tmp_path: Path) -> None:
    store = TradeTelemetryStore(out_dir=tmp_path, reporter=None)
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.trade_telemetry_store = store
    streamer._emit_trade_telemetry = LiveCSVStreamer._emit_trade_telemetry.__get__(streamer, LiveCSVStreamer)
    streamer.run_id = "RUN-FLAT-SNAPSHOT"
    streamer.instrument_alias = "ES"
    streamer.exec_instrument = "ES 06-26"
    streamer.nt_account = "DEMO"
    streamer.nt_bridge = SimpleNamespace(is_connected=True)
    streamer.state = SimpleNamespace(
        last_signal_id="STALE-SIGNAL",
        last_nt_client_order_id="STALE-CID",
        last_entry_side="LONG",
        daily_usd=0.0,
    )

    streamer._emit_trade_telemetry(
        "POSITION_SNAPSHOT",
        {
            "ts": "2026-06-05T04:35:00Z",
            "position_qty": 0,
            "instrument": "ES JUN26",
            "bridge_status": "connected",
            "meta": {"snapshot_type": "POSITION_SNAPSHOT"},
        },
    )

    telemetry_rows = _read_jsonl(tmp_path / "trade_telemetry.jsonl")
    assert telemetry_rows[-1]["position_qty"] == 0.0
    assert telemetry_rows[-1]["side"] == "FLAT"
    assert telemetry_rows[-1]["signal_id"] is None
    assert telemetry_rows[-1]["client_order_id"] is None
