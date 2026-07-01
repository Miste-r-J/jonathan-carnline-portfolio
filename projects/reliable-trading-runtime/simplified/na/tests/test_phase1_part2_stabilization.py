from __future__ import annotations

from pathlib import Path

import pandas as pd

from na.discord_addons.cli import stream_live_csv as mod
from na.discord_addons.cli.stream_live_csv import ExecutionIntent, LiveCSVStreamer


def test_parse_epoch_like_timestamp_seconds_millis_nanos() -> None:
    sec = mod._parse_epoch_like_timestamp_utc(1778564703)
    ms = mod._parse_epoch_like_timestamp_utc(1778564703000)
    ns = mod._parse_epoch_like_timestamp_utc(1778564703000000000)

    assert sec is not None
    assert ms is not None
    assert ns is not None
    assert sec.year == 2026
    assert ms.year == 2026
    assert ns.year == 2026
    assert sec.year != 1970


def test_setup_fail_send_guard_blocks_legacy_bypass() -> None:
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    events = []
    s.live_dry_run = False
    s.nt_exec_policy = "paper"
    s.phase2_force_open_allow_legacy_gate_bypass = False
    s._log_exec_event = lambda payload: events.append(payload)
    s._entry_block_pre_send_diagnostics = lambda intent, reason: None

    intent = ExecutionIntent(
        intent_id="t1",
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES",
        exec_instrument="ES 06-26",
        account="SIM",
        legacy_gate_bypassed=True,
    )
    result = s.execute_intent(intent)
    assert result.decision == "BLOCKED_SAFETY"
    assert result.reason_code == "setup_fail_final_send_guard"
    assert any(str(e.get("event")) == "ORDER_BLOCKED_SETUP_FAIL" for e in events)


def test_finalize_entry_emit_action_allows_setup_fail_entries_when_policy_enabled() -> None:
    final_action, blocked_by, block_reason, emit_allowed = mod._finalize_entry_emit_action(
        action="OPEN",
        phase2_setup_pass=False,
        override_applied=False,
        allow_setup_fail_entries=True,
        blocked_by=["setup"],
    )

    assert final_action == "OPEN"
    assert blocked_by == []
    assert block_reason is None
    assert emit_allowed is True


def test_bad_time_in_trade_blocks_overlay() -> None:
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    events = []
    s.pnl_overlay_enabled = True
    s._pos = 1
    s._open_trade = {"client_order_id": "cid1"}
    s._compute_trade_pnl_state = lambda **kwargs: {"bad_time_in_trade": True}
    s._log_exec_event = lambda payload: events.append(payload)

    row = pd.Series({"Close": 5000.0, "High": 5001.0, "Low": 4999.0})
    out = s._maybe_pnl_overlay_event(row=row, ts_dt=pd.Timestamp("2026-05-12T10:00:00"))
    assert out is None
    assert any(str(e.get("event")) == "BAD_TIME_IN_TRADE" for e in events)


def test_source_guard_no_direct_ns_timestamp_parse_on_exec_paths() -> None:
    source = Path(mod.__file__).read_text(encoding="utf-8")
    ns_hits = source.count('unit="ns"') + source.count("unit='ns'")
    assert ns_hits <= 1
    assert "1778564703" not in source


def test_restored_directional_bridge_allows_setup_fail_with_safety() -> None:
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    s.run_mode = "live"
    s.phase2_force_open_policy_enabled = True
    s.phase2_force_open_policy_mode = "directional_bridge_restored"
    s.phase2_force_open_allow_legacy_gate_bypass = True
    s.phase2_force_open_allow_setup_fail_entries = True
    s.phase2_force_open_restore_old_directional_bridge = True
    s.phase2_force_open_min_direction_prob_long = 0.57
    s.phase2_force_open_min_direction_prob_short = 0.57
    s.phase2_force_open_require_valid_geometry = True
    s.phase2_force_open_require_stop = True
    s.phase2_force_open_require_target = True
    s.phase2_force_open_allow_only_when_flat = False
    s.phase2_force_open_allow_flip = True
    s.phase2_p_long = 0.57
    s.phase2_p_short = 0.57
    s.tick_size = 0.25
    s._hard_lockout_active = False
    s.state = type("S", (), {"entries_disarmed_reason": None})()
    s._entries_disarmed_reason = None
    s._phase_allows_execution = lambda: True
    ev = {"type": "OPEN", "side": "LONG", "prob": 0.8, "price": 5000.0, "risk": {"stop": 4998.0, "target": 5004.0}}
    ok, reason, detail = s.restored_directional_bridge_allowed(ev=ev, open_allowed=False)
    assert ok is True
    assert reason == "restored_directional_bridge"
    assert float(detail.get("r_multiple") or 0) > 0


def test_restored_directional_bridge_blocks_bad_geometry() -> None:
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    s.run_mode = "live"
    s.phase2_force_open_policy_enabled = True
    s.phase2_force_open_policy_mode = "directional_bridge_restored"
    s.phase2_force_open_allow_legacy_gate_bypass = True
    s.phase2_force_open_allow_setup_fail_entries = True
    s.phase2_force_open_restore_old_directional_bridge = True
    s.phase2_force_open_min_direction_prob_long = 0.57
    s.phase2_force_open_min_direction_prob_short = 0.57
    s.phase2_force_open_require_valid_geometry = True
    s.phase2_force_open_require_stop = True
    s.phase2_force_open_require_target = True
    s.phase2_force_open_allow_only_when_flat = False
    s.phase2_force_open_allow_flip = True
    s.phase2_p_long = 0.57
    s.phase2_p_short = 0.57
    s.tick_size = 0.25
    s._hard_lockout_active = False
    s.state = type("S", (), {"entries_disarmed_reason": None})()
    s._entries_disarmed_reason = None
    s._phase_allows_execution = lambda: True
    ev = {"type": "OPEN", "side": "LONG", "prob": 0.8, "price": 5000.0, "risk": {"stop": 5001.0, "target": 5002.0}}
    ok, reason, _ = s.restored_directional_bridge_allowed(ev=ev, open_allowed=False)
    assert ok is False
    assert reason.startswith("geometry:")
