from __future__ import annotations

import csv
import json
from pathlib import Path

from na.discord_addons.cli.stream_live_csv import ExecutionIntent, LiveCSVStreamer
from tools.audit_live_backfill_parity import audit_live_backfill_parity


def _write_csv(path: Path, header: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_audit_live_backfill_parity_reports_mismatch(tmp_path: Path) -> None:
    live = tmp_path / "live"
    backfill = tmp_path / "backfill"
    out = tmp_path / "reports"
    header = [
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "action",
        "requested_action",
        "resolved_action",
        "display_action",
        "execution_intent_action",
        "side",
        "price",
        "prob",
        "transition_id",
        "signal_id",
        "client_order_id",
        "phase",
        "dedupe_key",
    ]
    live_rows = [
        {
            "Datetime": "2026-05-01T08:00:00-06:00",
            "action": "OPEN",
            "requested_action": "OPEN",
            "resolved_action": "OPEN",
            "display_action": "OPEN",
            "execution_intent_action": "OPEN",
            "side": "LONG",
            "price": "7000.0",
            "prob": "0.61",
            "transition_id": "T1",
            "signal_id": "S1",
            "client_order_id": "C1",
            "phase": "LIVE",
            "dedupe_key": "T1|S1|C1|OPEN|OPEN",
        },
        {
            "Datetime": "2026-05-01T08:05:00-06:00",
            "action": "CLOSE",
            "requested_action": "CLOSE",
            "resolved_action": "CLOSE",
            "display_action": "CLOSE",
            "execution_intent_action": "CLOSE",
            "side": "LONG",
            "price": "7002.0",
            "prob": "0.55",
            "transition_id": "T2",
            "signal_id": "S2",
            "client_order_id": "C2",
            "phase": "LIVE",
            "dedupe_key": "T2|S2|C2|CLOSE|CLOSE",
        },
    ]
    backfill_rows = [
        dict(live_rows[0]),
        {
            **live_rows[1],
            "price": "6998.0",
            "prob": "0.44",
            "dedupe_key": "T2|S2|C2|CLOSE|CLOSE",
        },
    ]
    _write_csv(live / "state.csv", header, live_rows)
    _write_csv(backfill / "state.csv", header, backfill_rows)
    _write_csv(live / "trades.csv", ["realized_points"], [{"realized_points": "2.0"}])
    _write_csv(backfill / "trades.csv", ["realized_points"], [{"realized_points": "0.0"}])
    (live / "status.json").write_text(json.dumps({"run_mode": "live", "nt_order_entry_total": 1, "model_emits_total": 1}), encoding="utf-8")
    (backfill / "status.json").write_text(json.dumps({"run_mode": "backfill", "nt_order_entry_total": 1, "model_emits_total": 1}), encoding="utf-8")

    report = audit_live_backfill_parity(live, backfill, out)
    assert report["live_decisions_total"] == 2
    assert report["backfill_decisions_total"] == 2
    assert report["mismatches_total"] >= 1
    assert (out / "parity_report.md").exists()
    assert (out / "parity_mismatches.csv").exists()
    assert (out / "lifecycle_replay_from_live.csv").exists()
    assert (out / "lifecycle_replay_from_backfill.csv").exists()
    assert (out / "order_intent_reconciliation.csv").exists()


def test_runtime_writes_decision_and_flip_order_intents(tmp_path: Path) -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer._ensure_compat_defaults = lambda: None
    streamer.run_mode = "live"
    streamer.instrument_alias = "ES"
    streamer.model_sha = "sha-test"
    streamer.model_path = "model-test"
    streamer._compute_config_hash = lambda: "cfg-test"
    streamer._lifecycle_event_dedupe = set()
    streamer._order_intent_dedupe = set()
    streamer._pos = 1
    streamer._open_trade = {"side": "LONG"}
    streamer.decision_events_path = tmp_path / "decision_events.csv"
    streamer.order_intents_path = tmp_path / "order_intents.csv"
    streamer.parity_mismatches_path = tmp_path / "parity_mismatches.csv"
    streamer.lifecycle_events_path = tmp_path / "lifecycle_events.jsonl"
    streamer.lifecycle_events_csv_path = tmp_path / "lifecycle_events.csv"
    streamer._ensure_lifecycle_paths = lambda: None
    streamer._handle_io_failure = lambda **_kwargs: None

    payload = {
        "ts": "2026-05-01T08:00:00-06:00",
        "bar_ts": "2026-05-01T08:00:00-06:00",
        "phase": "LIVE",
        "requested_action": "OPEN",
        "resolved_action": "OPEN",
        "display_action": "OPEN",
        "execution_intent_action": "OPEN",
        "side": "LONG",
        "price": 7000.0,
        "prob": 0.61,
        "emit_allowed": True,
        "publish_ready": True,
        "blocked_reason": "",
        "transition_id": "T-DECISION",
        "transition_step": "",
        "signal_id": "S-DECISION",
        "client_order_id": "C-DECISION",
        "source": "state_projection",
        "dedupe_key": "T-DECISION|S-DECISION|C-DECISION|OPEN|OPEN",
    }
    streamer._append_lifecycle_record(payload)
    decision_rows = list(csv.DictReader((tmp_path / "decision_events.csv").open("r", encoding="utf-8")))
    assert len(decision_rows) == 1
    assert decision_rows[0]["requested_action"] == "OPEN"
    assert decision_rows[0]["model_version"] == "sha-test"
    assert decision_rows[0]["config_hash"] == "cfg-test"

    intent = ExecutionIntent(
        intent_id="RUN|P|ES|2026-05-01T08:05:00-06:00|FLIP|SHORT|abc123",
        action="FLIP",
        side="SHORT",
        qty=1,
        instrument_raw="ES",
        exec_instrument="ES",
        account="SIM",
        bar_ts="2026-05-01T08:05:00-06:00",
        entry_price=7001.0,
        model_price=7001.0,
        stop_price=6998.0,
        target_price=7008.0,
    )
    streamer._open_trade = {"side": "LONG"}
    streamer._pos = 1
    streamer._append_order_intent_rows(intent=intent, decision="SENT", reason_code="ok", nt_order_ids=["BROKER-1"])
    order_rows = list(csv.DictReader((tmp_path / "order_intents.csv").open("r", encoding="utf-8")))
    assert len(order_rows) == 2
    assert [row["order_action"] for row in order_rows] == ["CLOSE", "OPEN"]
    assert order_rows[0]["parent_transition_id"] == "RUN|P|ES|2026-05-01T08:05:00-06:00|FLIP|SHORT|abc123"
    assert order_rows[1]["parent_transition_id"] == "RUN|P|ES|2026-05-01T08:05:00-06:00|FLIP|SHORT|abc123"
    assert order_rows[0]["client_order_id"] == intent.intent_id
