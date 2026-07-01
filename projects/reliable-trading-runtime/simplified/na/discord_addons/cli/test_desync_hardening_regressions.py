from __future__ import annotations

import csv
from pathlib import Path

from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer, STATE_CSV_COLUMNS


def _write_state_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=STATE_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        filtered_rows = [{k: row.get(k, "") for k in STATE_CSV_COLUMNS} for row in rows]
        writer.writerows(filtered_rows)


def test_state_reconcile_prefers_stable_ids(tmp_path: Path) -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.state_stream_csv = tmp_path / "state.csv"
    streamer._state_dedupe = set()
    streamer._load_state_dedupe = lambda: None
    streamer._log_exec_event = lambda payload: None

    ts = "2026-05-18T10:00:00-06:00"
    _write_state_csv(
        streamer.state_stream_csv,
        [
            {
                "datetime": ts,
                "action": "OPEN",
                "side": "LONG",
                "price": "100.0",
                "prob": "0.7",
                "entry_conf": "",
                "hold_conf": "",
                "gates": "",
                "stop": "",
                "target": "",
                "R": "",
                "grade": "",
                "size_hint": "",
                "hold_text": "",
                "success_prob": "",
                "context": "",
                "position": "",
                "client_order_id": "cid-older",
                "signal_id": "sig-older",
                "dedupe_key": f"{ts}|ES|OPEN|cid-older",
            },
            {
                "datetime": ts,
                "action": "OPEN",
                "side": "LONG",
                "price": "101.0",
                "prob": "0.8",
                "entry_conf": "",
                "hold_conf": "",
                "gates": "",
                "stop": "",
                "target": "",
                "R": "",
                "grade": "",
                "size_hint": "",
                "hold_text": "",
                "success_prob": "",
                "context": "",
                "position": "",
                "client_order_id": "cid-target",
                "signal_id": "sig-target",
                "dedupe_key": f"{ts}|ES|OPEN|cid-target",
            },
        ],
    )

    streamer._reconcile_state_action_for_suppressed_signal(
        ts_iso=ts,
        original_action="OPEN",
        side="LONG",
        reason="test",
        client_order_id="cid-target",
        signal_id="sig-target",
    )

    with streamer.state_stream_csv.open("r", encoding="utf-8", newline="\n") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["action"] == "OPEN"
    assert rows[1]["action"] == "HOLD"
    assert rows[1]["side"] == "LONG"


def test_emitted_entry_lifecycle_defers_until_ack_or_fill() -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.nt_enabled = True
    streamer._nt_order_state = {"cid-1": {"entry_acked": False, "entry_filled": False}}
    streamer._open_trade = {"client_order_id": "cid-1", "trades_csv_open_written": False}
    streamer._append_trade_open_row = lambda open_trade: (_ for _ in ()).throw(AssertionError("must not write"))
    streamer._emit_trades_csv_ignored_decision = lambda ev, action, reason: None

    streamer._record_emitted_entry_lifecycle({"type": "OPEN", "client_order_id": "cid-1"})
    assert streamer._open_trade["trades_csv_open_written"] is False
