from __future__ import annotations

import csv

import pytest

from na.discord_addons.cli.stream_live_csv import EmissionLedger, LiveCSVStreamer, StreamState


def _make_streamer(tmp_path):
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.state = StreamState()
    streamer.run_id = "RUNID"
    streamer.nt_instrument = "ES MAR26"
    streamer.exec_instrument = None
    streamer.instrument_alias = "ES"
    streamer.preset = "preset"
    streamer.active_preset_name = None
    streamer.active_strategy_tag = ""
    streamer.model_run_id = "model"
    streamer.model_sha = None
    streamer.tick_size = 0.25

    streamer._phase = "LIVE"
    streamer._emission_eval_calls = {"LIVE": 0}
    streamer._executor_stats = {}
    streamer._last_csv_bar = None
    streamer._last_bar_ts_guard = None
    streamer._blocked_past_bar_emit_count = 0

    streamer.out_dir = tmp_path
    streamer.signals_csv = tmp_path / "signals.csv"
    streamer.signals_jsonl = tmp_path / "signals.jsonl"
    streamer.emission_ledger_path = tmp_path / "emission_ledger.jsonl"
    streamer._emission_ledger = EmissionLedger(streamer.emission_ledger_path)

    streamer.contracts_per_trade = 1
    streamer.policy_guard_enabled = False
    streamer.nt_enabled = False
    streamer.leak_guard_strict = False

    streamer._io_failed = False
    streamer._active_close_correlation_id = None
    streamer._active_close_started_ts = None
    streamer._close_intents = {}
    streamer._exit_stuck_flatten_attempted = set()
    streamer._close_watchdog_resend_attempted = set()

    streamer._log_exec_event = lambda *args, **kwargs: None
    streamer._log_signal_to_order = lambda *args, **kwargs: None
    streamer._emit_block_event = lambda *args, **kwargs: None
    streamer._handle_io_failure = lambda *args, **kwargs: None
    streamer._write_status = lambda *args, **kwargs: None
    streamer._maybe_send_nt_order = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("_maybe_send_nt_order should not be called in these unit tests")
    )

    # Minimal header for parsing
    streamer.signals_csv.write_text(
        "datetime,type,side,price,prob,directional_prob,grade,stop,target,contracts,client_order_id,signal_id,"
        "override_confident_long,override_prob_min,override_hold_conf_min,override_applied,blocked,blocked_reason\n",
        encoding="utf-8",
    )
    return streamer


def test_position_uid_cleared_on_exit_to_flat(tmp_path):
    streamer = _make_streamer(tmp_path)
    streamer.state.position_uid = "pos|ES 03-26|some-stale-uid"
    streamer.state.position_state = "EXITING"

    streamer._emit_event = lambda *args, **kwargs: None
    streamer._flush_inflight_buffer = lambda *args, **kwargs: None

    streamer._set_position_state("FLAT")
    assert streamer.state.position_uid is None


def test_position_uid_hashes_entry_cid(tmp_path):
    streamer = _make_streamer(tmp_path)
    uid = streamer._ensure_position_uid(entry_cid="ENTRY|CID|WITH|PIPES", entry_order_id=None, source="test")
    assert uid is not None
    assert "ENTRY|CID|WITH|PIPES" not in uid
    assert "|cid:" in uid


def test_deduped_emission_writes_signals_csv_as_blocked(tmp_path):
    streamer = _make_streamer(tmp_path)

    cid = "RUNID|ES MAR26|pos|ES 03-26|abc123|CLOSE"
    streamer._emission_ledger.append(cid, {"ts": "2026-01-01T00:00:00-07:00", "type": "CLOSE", "side": "LONG"})
    streamer._active_close_correlation_id = cid

    ev = {
        "datetime": "2026-01-01T00:05:00-07:00",
        "type": "CLOSE",
        "side": "LONG",
        "price": 100.0,
        "contracts": 1,
        "instrument": "ES",
        "client_order_id": cid,
        "risk": {"stop": 99.0, "target": None},
        "prob": 0.5,
        "grade": "C",
        "gates_detail": {},
    }

    streamer._append_signal(ev)

    rows = list(csv.DictReader(streamer.signals_csv.read_text(encoding="utf-8").splitlines()))
    assert rows, "expected at least one row written"
    last = rows[-1]
    assert last["client_order_id"] == cid
    assert last["blocked"] == "1"
    assert last["blocked_reason"] == "deduped_already_emitted"


def test_directional_prob_written_for_short_signal(tmp_path):
    streamer = _make_streamer(tmp_path)

    ev = {
        "datetime": "2026-01-01T00:05:00-07:00",
        "type": "OPEN",
        "side": "SHORT",
        "price": 100.0,
        "contracts": 1,
        "instrument": "ES",
        "client_order_id": "CID",
        "risk": {"stop": 101.0, "target": None},
        "prob": 0.398,
        "grade": "B",
        "gates_detail": {},
    }

    streamer._append_signal(ev)

    rows = list(csv.DictReader(streamer.signals_csv.read_text(encoding="utf-8").splitlines()))
    last = rows[-1]
    assert last["side"] == "SHORT"
    assert float(last["prob"]) == pytest.approx(0.398)
    assert float(last["directional_prob"]) == pytest.approx(1.0 - 0.398)


def test_close_idempotency_key_sanitizes_run_and_instrument_delimiters(tmp_path):
    streamer = _make_streamer(tmp_path)
    streamer.run_id = "RUN|ID 01"
    streamer.nt_instrument = "ES|MAR26"

    key = streamer._close_idempotency_key(position_uid="pos|ES MAR26|cid:abc123")
    assert key is not None
    assert key.startswith("RUN_ID_01|ES_MAR26|")
    assert key.endswith("|CLOSE")


def test_ensure_close_intent_reuses_active_cid(tmp_path):
    streamer = _make_streamer(tmp_path)
    streamer._active_close_correlation_id = "RUNID|ES MAR26|pos|abc123|CLOSE"
    ev = {"type": "CLOSE", "side": "LONG"}

    close_id = streamer._ensure_close_intent_id(ev)
    assert close_id == "RUNID|ES MAR26|pos|abc123|CLOSE"
    assert ev["client_order_id"] == "RUNID|ES MAR26|pos|abc123|CLOSE"
