from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_system.runtime_engine.integrations.cli.live_trading_runtime import ExecutionIntent, LiveCSVStreamer


def _make_intent(signal_id: str) -> ExecutionIntent:
    return ExecutionIntent(
        intent_id="RUNID|OPEN|1",
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES",
        exec_instrument="ES 06-26",
        account=None,
        bar_ts="2026-04-01T08:30:00-06:00",
        model_price=6200.0,
        model_stop_price=6192.0,
        model_target_price=6208.0,
        signal_id=signal_id,
    )


def _lineage_streamer() -> LiveCSVStreamer:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.enforce_integrity_gate = True
    streamer.run_id = "RUNID"
    streamer._open_intents_sent = set()
    streamer._signal_lineage_by_client_order_id = {}
    streamer._signal_lineage_by_signal_id = {}
    return streamer


def _phase2_force_open_record(
    *,
    signal_id: str = "sig-force",
    side: str = "LONG",
    gate_open: bool = True,
    setup_pass: bool = True,
    direction_signal: int = 1,
) -> dict:
    return {
        "run_id": "RUNID",
        "ts": "2026-04-01T08:30:00-06:00",
        "event": "OPEN",
        "side": side,
        "signal_id": signal_id,
        "client_order_id": "RUNID|OPEN|1",
        "gates": {
            "gate_state": {
                "setup": True,
                "prob": False,
                "vwap": True,
                "ema": True,
                "tod": True,
            }
        },
        "ctx": {
            "phase2_force_open_applied": True,
            "phase2": {
                "gate_open": gate_open,
                "setup_pass": setup_pass,
                "direction_signal": direction_signal,
            },
        },
    }


def test_lineage_gate_blocks_missing_or_failed_signal() -> None:
    streamer = _lineage_streamer()
    streamer._signal_lineage_by_signal_id = {
        "seed": {
            "run_id": "RUNID",
            "bar_ts": "2026-04-01T08:30:00-06:00",
            "action": "OPEN",
            "side": "LONG",
            "gate_pass": True,
        }
    }

    missing_reason = streamer._validate_open_lineage(_make_intent("sig-missing"))
    assert missing_reason == "lineage_signal_not_found"

    streamer._signal_lineage_by_signal_id["sig-fail"] = {
        "run_id": "RUNID",
        "bar_ts": "2026-04-01T08:30:00-06:00",
        "action": "OPEN",
        "side": "LONG",
        "gate_pass": False,
    }
    failed_reason = streamer._validate_open_lineage(_make_intent("sig-fail"))
    assert failed_reason == "lineage_gate_not_passed"


def test_phase2_force_open_lineage_bypasses_legacy_prob_gate() -> None:
    streamer = _lineage_streamer()
    record = _phase2_force_open_record()

    streamer._index_signal_lineage(record)
    reason = streamer._validate_open_lineage(_make_intent("sig-force"))

    lineage = streamer._signal_lineage_by_signal_id["sig-force"]
    assert reason is None
    assert lineage["gate_pass"] is True
    assert lineage["phase2_force_open_lineage_gate_bypass"] is True
    assert lineage["phase2_force_open_lineage_gate_bypass_reason"] == "phase2_gate_open"


def test_phase2_force_open_lineage_requires_phase2_gate_open() -> None:
    streamer = _lineage_streamer()
    record = _phase2_force_open_record(gate_open=False)

    streamer._index_signal_lineage(record)
    reason = streamer._validate_open_lineage(_make_intent("sig-force"))

    assert reason == "lineage_gate_not_passed"


def test_phase2_force_open_lineage_requires_side_direction_match() -> None:
    streamer = _lineage_streamer()
    record = _phase2_force_open_record(side="SHORT", direction_signal=1)

    streamer._index_signal_lineage(record)
    reason = streamer._validate_open_lineage(_make_intent("sig-force"))

    assert reason == "lineage_side_mismatch"
    assert streamer._signal_lineage_by_signal_id["sig-force"]["gate_pass"] is False


def test_normal_open_with_failed_prob_gate_still_fails_lineage() -> None:
    streamer = _lineage_streamer()
    record = _phase2_force_open_record(signal_id="sig-normal")
    record["ctx"] = {}

    streamer._index_signal_lineage(record)
    reason = streamer._validate_open_lineage(_make_intent("sig-normal"))

    assert reason == "lineage_gate_not_passed"


def test_legacy_force_open_bypass_marker_sets_lineage_gate_pass() -> None:
    streamer = _lineage_streamer()
    record = _phase2_force_open_record(signal_id="sig-legacy")
    record["ctx"] = {
        "phase2_force_open_legacy_gate_bypass": True,
        "phase2_force_open_legacy_gate_bypass_reason": "directional_prob_live_bridge",
    }
    record["phase2_force_open_legacy_gate_bypass"] = True
    record["gates"]["gate_state"]["setup"] = False
    record["gates"]["failed_gate"] = "setup"

    streamer._index_signal_lineage(record)
    reason = streamer._validate_open_lineage(_make_intent("sig-legacy"))

    assert reason is None
    assert streamer._signal_lineage_by_signal_id["sig-legacy"]["gate_pass"] is True


def test_artifact_consistency_rejects_cross_run_mix(tmp_path: Path) -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.enforce_integrity_gate = True
    streamer.run_id = "RUNID"
    streamer.signals_jsonl = tmp_path / "signals.jsonl"
    streamer.signal_to_order_path = tmp_path / "signal_to_order.jsonl"
    streamer.order_events_path = tmp_path / "order_events.jsonl"
    streamer._log_exec_event = lambda _payload: None

    streamer.signals_jsonl.write_text(json.dumps({"run_id": "RUNID"}) + "\n", encoding="utf-8")
    streamer.signal_to_order_path.write_text(json.dumps({"run_id": "RUNID"}) + "\n", encoding="utf-8")
    streamer.order_events_path.write_text(json.dumps({"run_id": "OTHER"}) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="RUN_ARTIFACT_INTEGRITY_FAIL"):
        streamer._validate_artifact_run_id_consistency()
