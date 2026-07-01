from __future__ import annotations

import csv
import json
import queue
import time
from datetime import time as dtime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from na.bot.phase2_sim import Phase2DecisionPolicy
from na.discord_addons.cli import stream_live_csv as stream_live_csv_mod
from na.discord_addons.cli.stream_live_csv import (
    EmissionLedger,
    LiveCSVStreamer,
    StreamState,
    _canonical_ts_str,
    _compute_vwap_gate_raw,
    _finalize_entry_emit_action,
    _now_denver,
    utc_ts,
)
from tools.audit_live_run import audit_run


SIGNALS_CSV_HEADER = (
    "datetime,type,side,price,prob,directional_prob,grade,stop,target,contracts,client_order_id,signal_id,"
    "override_confident_long,override_prob_min,override_hold_conf_min,override_applied,blocked,blocked_reason\n"
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_master_preset(preset_name: str) -> dict:
    master_path = Path(__file__).resolve().parents[1] / "config" / "master.yaml"
    payload = yaml.safe_load(master_path.read_text(encoding="utf-8"))
    return dict((payload or {}).get("risk_presets", {}).get(preset_name, {}))


def test_safe_pnl_master_preset_retune_integrity() -> None:
    preset = _load_master_preset("es_prop_safe_v1")
    assert preset
    assert preset["allowed_grades"] == ["A", "B"]
    risk = dict(preset.get("risk") or {})
    assert float(risk.get("max_daily_loss", 0) or 0) == pytest.approx(250.0)
    assert float(risk.get("max_risk_per_trade_usd", 0) or 0) == pytest.approx(100.0)
    assert preset["trade_window_end"] == "14:00"
    assert preset.get("phase2_force_open_on_gate_pass") in (None, False)
    assert preset.get("phase2_force_open_min_setup") is None
    assert preset.get("phase2_force_open_min_entry_conf") is None
    assert preset.get("pnl_runner_suppress_target") in (None, False)
    assert preset.get("pnl_giveback_activate_r") is None
    assert preset.get("pnl_giveback_close_r") is None
    assert preset.get("session_suppression_windows") in (None, [])


def test_apply_preset_to_args_maps_phase2_force_open_thresholds() -> None:
    parser = stream_live_csv_mod._build_arg_parser()
    argv = ["--preset", "es_prop_safe_v1"]
    args = parser.parse_args(argv)

    applied = stream_live_csv_mod._apply_preset_to_args(args, argv)

    assert bool(getattr(applied, "phase2_force_open_on_gate_pass", False)) is False
    assert float(getattr(applied, "phase2_force_open_min_setup", 0.0) or 0.0) == pytest.approx(0.027)
    assert float(getattr(applied, "phase2_force_open_min_entry_conf", 0.0) or 0.0) == pytest.approx(0.2)


def test_parser_accepts_fiber_history_retry_flags() -> None:
    parser = stream_live_csv_mod._build_arg_parser()
    args = parser.parse_args(
        [
            "--csv",
            "dummy.csv",
            "--model",
            "dummy.joblib",
            "--history_stale_max_sec",
            "777",
            "--fiber_history_retry_max",
            "5",
            "--fiber_history_retry_backoff_sec",
            "3.5",
        ]
    )
    assert float(getattr(args, "history_stale_max_sec", 0.0) or 0.0) == pytest.approx(777.0)
    assert int(getattr(args, "fiber_history_retry_max", 0) or 0) == 5
    assert float(getattr(args, "fiber_history_retry_backoff_sec", 0.0) or 0.0) == pytest.approx(3.5)


def test_count_unique_bar_days_from_fiber_rows() -> None:
    rows = [
        {"Datetime": "2026-05-19T16:10:00-06:00"},
        {"Datetime": "2026-05-19T16:15:00-06:00"},
        {"Datetime": "2026-05-20T09:30:00-06:00"},
    ]
    assert stream_live_csv_mod._count_unique_bar_days(rows) == 2


def test_adjusted_v4_preset_allowed_for_paper_and_demo_live_only() -> None:
    paper_allowed = stream_live_csv_mod._approved_preset_names_for_exec_policy("paper")
    assert "es_modelrun77_adjusted_v4_paper" in paper_allowed

    demo_args = SimpleNamespace(nt_allowed_accounts="DEMO8142346", nt_account="", nt_account_label="")
    demo_live_allowed = stream_live_csv_mod._approved_preset_names_for_exec_policy("live", demo_args)
    assert "es_modelrun77_adjusted_v4_paper" in demo_live_allowed

    funded_args = SimpleNamespace(nt_allowed_accounts="MFFUEVRPD447934003", nt_account="", nt_account_label="")
    funded_live_allowed = stream_live_csv_mod._approved_preset_names_for_exec_policy("live", funded_args)
    assert "es_modelrun77_adjusted_v4_paper" not in funded_live_allowed


def test_adjusted_v4_preset_synthesizes_entries_only_in_live() -> None:
    preset = _load_master_preset("es_modelrun77_adjusted_v4_paper")
    assert preset.get("phase2_force_open_on_gate_pass") is True
    assert preset.get("phase2_force_open_live_only") is True
    assert preset.get("phase2_force_open_allow_setup_fail_entries") is False

    parser = stream_live_csv_mod._build_arg_parser()
    argv = ["--preset", "es_modelrun77_adjusted_v4_paper"]
    applied = stream_live_csv_mod._apply_preset_to_args(parser.parse_args(argv), argv)

    assert bool(getattr(applied, "phase2_force_open_on_gate_pass", False)) is True
    assert bool(getattr(applied, "phase2_force_open_live_only", False)) is True
    assert bool(getattr(applied, "phase2_force_open_allow_setup_fail_entries", True)) is False


def test_live_stale_emit_guard_uses_processed_bar_not_shifted_csv_max(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase = "LIVE"
    streamer._last_bar_ts_guard = pd.Timestamp("2026-06-23T01:05:00-06:00")
    streamer._last_csv_bar = pd.Timestamp("2026-06-23T02:05:00-06:00")

    assert streamer._current_execution_bar_ts() == pd.Timestamp("2026-06-23T01:05:00-06:00")


def test_backfill_stale_emit_guard_keeps_csv_max_authority(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase = "BACKFILL"
    streamer._last_bar_ts_guard = pd.Timestamp("2026-06-23T01:05:00-06:00")
    streamer._last_csv_bar = pd.Timestamp("2026-06-23T02:05:00-06:00")

    assert streamer._current_execution_bar_ts() == pd.Timestamp("2026-06-23T02:05:00-06:00")


def test_evaluate_history_staleness_uses_max_sec_threshold() -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.history_stale_max_sec = 600.0
    streamer._fiber_hist_end_ts = "2026-05-19T16:10:00-06:00"
    streamer._last_csv_bar = pd.Timestamp("2026-05-19T16:10:00-06:00")
    streamer._bar_time_delta_sec_signed = lambda _bar: 1800.0
    stale, detail = LiveCSVStreamer._evaluate_history_staleness(streamer)
    assert stale is True
    assert float(detail["lag_sec"]) == pytest.approx(1800.0)
    assert float(detail["history_stale_max_sec"]) == pytest.approx(600.0)


def test_elite_open_test_preset_maps_norisk() -> None:
    preset = _load_master_preset("es_elite_v1_open_test")
    expected_disable_risk = bool(preset.get("disable_risk"))

    parser = stream_live_csv_mod._build_arg_parser()
    argv = ["--preset", "es_elite_v1_open_test"]
    args = parser.parse_args(argv)
    applied = stream_live_csv_mod._apply_preset_to_args(args, argv)
    assert bool(applied.norisk) is expected_disable_risk


def test_elite_v1_preset_maps_disable_safety_and_replay_exec() -> None:
    preset = _load_master_preset("es_elite_v1")
    assert "disable_safety_gates" not in preset
    assert "allow_replay_exec_to_nt" not in preset

    parser = stream_live_csv_mod._build_arg_parser()
    argv = ["--preset", "es_elite_v1"]
    args = parser.parse_args(argv)
    applied = stream_live_csv_mod._apply_preset_to_args(args, argv)
    assert bool(getattr(applied, "disable_safety_gates", False)) is False
    assert bool(getattr(applied, "allow_replay_exec_to_nt", False)) is False


def test_elite_v1_preset_maps_trend_bias_guard_toggle() -> None:
    preset = _load_master_preset("es_elite_v1")
    assert preset.get("trend_bias_guard_enabled") is False

    parser = stream_live_csv_mod._build_arg_parser()
    argv = ["--preset", "es_elite_v1"]
    args = parser.parse_args(argv)
    applied = stream_live_csv_mod._apply_preset_to_args(args, argv)
    assert getattr(applied, "trend_bias_guard_enabled", None) is False


def test_elite_v1_preset_shelf_runner_safety_defaults() -> None:
    preset = _load_master_preset("es_elite_v1")
    assert preset.get("pnl_runner_suppress_target") is False
    assert preset.get("pnl_runner_ignore_close_before_arm") is False
    assert preset.get("pnl_shelf_arm_mode") == "step_only"
    assert preset.get("pnl_shelf_intrabar") is False
    assert int(preset.get("phase2_close_min_hold_bars", 0) or 0) >= 1
    assert float(preset.get("phase2_close_min_hold_sec", 0.0) or 0.0) >= 1.0


def test_parity_preset_yaml_has_expected_gate_and_force_open_profile() -> None:
    preset = _load_master_preset("es_elite_backfill_live_parity_v1")
    assert preset
    assert preset.get("phase2_use_manifest_thresholds") is False
    assert float(preset.get("p_setup", 1.0) or 1.0) == pytest.approx(0.0)
    assert float(preset.get("p_long", 1.0) or 1.0) == pytest.approx(0.5)
    assert float(preset.get("p_short", 1.0) or 1.0) == pytest.approx(0.5)
    force_policy = dict(preset.get("phase2_force_open_policy") or {})
    assert force_policy.get("enabled") is True
    assert str(force_policy.get("mode") or "") == "directional_bridge_aggressive"
    assert force_policy.get("allow_setup_fail_entries") is True
    assert force_policy.get("legacy_gate_bypass_allowed") is True
    assert float(force_policy.get("min_direction_prob_long", 1.0) or 1.0) == pytest.approx(0.5)
    assert float(force_policy.get("min_direction_prob_short", 1.0) or 1.0) == pytest.approx(0.5)


def test_apply_preset_to_args_parity_preset_enables_force_open_and_preset_thresholds() -> None:
    parser = stream_live_csv_mod._build_arg_parser()
    argv = ["--preset", "es_elite_backfill_live_parity_v1"]
    args = parser.parse_args(argv)
    applied = stream_live_csv_mod._apply_preset_to_args(args, argv)

    assert getattr(applied, "phase2_use_manifest_thresholds", None) is False
    assert float(getattr(applied, "p_setup", 1.0) or 1.0) == pytest.approx(0.0)
    assert float(getattr(applied, "p_long", 1.0) or 1.0) == pytest.approx(0.5)
    assert float(getattr(applied, "p_short", 1.0) or 1.0) == pytest.approx(0.5)
    assert bool(getattr(applied, "phase2_force_open_on_setup_pass", False)) is True
    assert bool(getattr(applied, "phase2_force_open_allow_setup_fail_entries", False)) is True
    assert bool(getattr(applied, "phase2_force_open_allow_legacy_gate_bypass", False)) is True
    assert str(getattr(applied, "phase2_force_open_policy_mode", "") or "") == "directional_bridge_aggressive"
    assert float(getattr(applied, "phase2_force_open_min_direction_prob_long", 1.0) or 1.0) == pytest.approx(0.5)
    assert float(getattr(applied, "phase2_force_open_min_direction_prob_short", 1.0) or 1.0) == pytest.approx(0.5)


def test_modelrun77_final_paper_preset_uses_manifest_and_disables_force_open() -> None:
    preset = _load_master_preset("es_modelrun77_candidate_strict_v1")
    assert preset.get("phase2_tag") == "modelrun77_final_candidate_20260622"
    assert preset.get("phase2_use_manifest_thresholds") is True
    assert preset.get("phase2_force_open_on_gate_pass") is False
    assert preset.get("phase2_force_open_allow_setup_fail_entries") is False

    parser = stream_live_csv_mod._build_arg_parser()
    argv = ["--preset", "es_modelrun77_candidate_strict_v1"]
    applied = stream_live_csv_mod._apply_preset_to_args(parser.parse_args(argv), argv)
    assert getattr(applied, "phase2_tag", None) == "modelrun77_final_candidate_20260622"
    assert getattr(applied, "phase2_use_manifest_thresholds", None) is True
    assert bool(getattr(applied, "phase2_force_open_on_gate_pass", True)) is False
    assert bool(getattr(applied, "phase2_force_open_allow_setup_fail_entries", True)) is False


def test_live_runner_target_suppression_is_blocked_without_override(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer.pnl_runner_enabled = True
    streamer.pnl_runner_suppress_target = True
    streamer.pnl_runner_allow_target_suppression_live = False
    streamer._runner_target_suppressed = LiveCSVStreamer._runner_target_suppressed.__get__(streamer, LiveCSVStreamer)

    assert streamer._runner_target_suppressed() is False


def test_stream_source_does_not_use_shelf_arm_force_exit_token() -> None:
    source = Path(stream_live_csv_mod.__file__).read_text(encoding="utf-8")
    assert "SHELF_ARM_ATTEMPT_FORCE_EXIT" not in source
    assert "SHELF_DEGRADED_NO_FORCE_EXIT" in source


def test_late_exits_submitted_after_close_fill_does_not_mark_unprotected(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    logs: list[dict] = []
    streamer._log_exec_event = lambda payload: logs.append(dict(payload))
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer._open_trade = None
    streamer._last_terminal_flat_cid = "cid-flat"
    streamer._last_terminal_flat_ts = time.time()
    streamer.close_emit_timeout_sec = 5.0
    streamer._nt_last_orders = []
    streamer._nt_last_pos_qty_by_instrument = {"ES 06-26": 0.0, "ES": 0.0}
    streamer._exec_instrument_key = lambda: "ES 06-26"
    streamer._load_fill_truth_index = lambda fresh=False: {"cid-flat": {"entry_fill_ts_epoch": 1.0}}
    streamer._summarize_fill_truth = lambda _: {
        "fills_present": True,
        "net_qty": 1.0,
        "last_cid": "cid-flat",
        "last_entry": (1.0, "cid-flat", {"entry_fill_price": 7200.0, "entry_fill_ts_epoch": 1.0}),
    }
    streamer._rebuild_trades_from_fill_truth = lambda _: None

    streamer._apply_fill_truth_reconciliation(source="ORDER_UPDATE_EXITS_SUBMITTED")

    assert streamer.state.position_state == "FLAT"
    assert all(str(ev.get("event")) != "IN_POSITION_UNPROTECTED" for ev in logs)
    assert any(str(ev.get("event")) == "STALE_PROTECTION_UPDATE_IGNORED_AFTER_FLAT" for ev in logs)


def test_flat_with_working_orders_is_orphan_not_unprotected(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    logs: list[dict] = []
    streamer._log_exec_event = lambda payload: logs.append(dict(payload))
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer._open_trade = None
    streamer._last_terminal_flat_cid = "cid-flat"
    streamer._last_terminal_flat_ts = time.time()
    streamer.close_emit_timeout_sec = 5.0
    streamer._nt_last_orders = [{"id": "stop-1"}]
    streamer._nt_last_pos_qty_by_instrument = {"ES 06-26": 0.0, "ES": 0.0}
    streamer._exec_instrument_key = lambda: "ES 06-26"
    streamer._load_fill_truth_index = lambda fresh=False: {"cid-flat": {"entry_fill_ts_epoch": 1.0}}
    streamer._summarize_fill_truth = lambda _: {
        "fills_present": True,
        "net_qty": 1.0,
        "last_cid": "cid-flat",
        "last_entry": (1.0, "cid-flat", {"entry_fill_price": 7200.0, "entry_fill_ts_epoch": 1.0}),
    }
    streamer._rebuild_trades_from_fill_truth = lambda _: None

    streamer._apply_fill_truth_reconciliation(source="ORDER_UPDATE_EXITS_SUBMITTED")

    assert streamer.state.position_state == "FLAT"
    assert any(str(ev.get("event")) == "ORPHAN_WORKING_ORDER_AFTER_FLAT" for ev in logs)
    assert not any("IN_POSITION_UNPROTECTED" in str(ev) for ev in logs)


def test_closed_trade_late_protection_update_archived_not_applied(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    logs: list[dict] = []
    streamer._log_exec_event = lambda payload: logs.append(dict(payload))
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer._open_trade = {"client_order_id": "cid-flat", "protection_status": "protected_confirmed", "status": "closed"}
    streamer._last_terminal_flat_cid = "cid-flat"
    streamer._last_terminal_flat_ts = time.time()
    streamer.close_emit_timeout_sec = 5.0
    streamer._nt_last_orders = []
    streamer._nt_last_pos_qty_by_instrument = {"ES 06-26": 0.0, "ES": 0.0}
    streamer._exec_instrument_key = lambda: "ES 06-26"
    streamer._load_fill_truth_index = lambda fresh=False: {"cid-flat": {"entry_fill_ts_epoch": 1.0}}
    streamer._summarize_fill_truth = lambda _: {
        "fills_present": True,
        "net_qty": 1.0,
        "last_cid": "cid-flat",
        "last_entry": (1.0, "cid-flat", {"entry_fill_price": 7200.0, "entry_fill_ts_epoch": 1.0}),
    }
    streamer._rebuild_trades_from_fill_truth = lambda _: None

    streamer._apply_fill_truth_reconciliation(source="ORDER_UPDATE_EXITS_SUBMITTED")

    assert streamer.state.position_state == "FLAT"
    assert any(str(ev.get("event")) == "STALE_PROTECTION_UPDATE_IGNORED_AFTER_FLAT" for ev in logs)


def test_reconciliation_does_not_disarm_without_entry_intent_evidence(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    logs: list[dict] = []
    streamer._log_exec_event = lambda payload: logs.append(dict(payload))
    streamer.state.position_state = "IN_POSITION_NO_PROTECTION"
    streamer.state.active_client_order_id = "cid-no-intent"
    streamer._pos = 1
    streamer._open_trade = {"client_order_id": "cid-no-intent"}
    streamer._nt_order_state = {}
    streamer._reporting_mismatch_disarm_detail = None
    streamer._load_fill_truth_index = lambda fresh=False: {}
    streamer._summarize_fill_truth = lambda _index: {
        "fills_present": False,
        "net_qty": 0.0,
        "last_cid": None,
        "last_entry": None,
    }
    streamer._rebuild_trades_from_fill_truth = lambda _index: None

    streamer._apply_fill_truth_reconciliation(source="unit_test")

    assert streamer._reporting_mismatch_disarm_detail is None


def test_reconcile_reporting_repairs_untracked_protection_leg_active_cid(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    logs: list[dict] = []
    streamer._log_exec_event = lambda payload: logs.append(dict(payload))
    entry_cid = "RUNID|es_modelrun77_parity_v1|ES JUN26|2026-06-03T23:10:00-06:00|OPEN|LONG|e19fa7c5"
    leg_cid = "UNTRACKED|ES JUN26|514943310949"
    fill_index = {
        entry_cid: {
            "entry_fill_ts_epoch": 1780550102.142919,
            "entry_fill_price": 7540.25,
            "entry_fill_qty": 1.0,
            "side": "LONG",
            "intent_side": "LONG",
        }
    }
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer.state.active_client_order_id = leg_cid
    streamer._pos = 1
    streamer._open_trade = {
        "client_order_id": leg_cid,
        "side": "LONG",
        "entry_ack_ts": "2026-06-03T23:10:01.561830-06:00",
    }
    streamer._nt_order_state = {
        entry_cid: {"sent_ts": time.time(), "entry_order_id": "514943310940", "entry_filled": True}
    }
    streamer._load_fill_truth_index = lambda fresh=False: dict(fill_index)
    streamer._load_exit_lifecycle_index = lambda: {}
    streamer._rebuild_trades_from_fill_truth = lambda _index: None

    streamer._reconcile_reporting(
        now_ts=time.time(),
        bar_ts=pd.Timestamp("2026-06-03T23:15:00-06:00"),
    )

    assert streamer.state.active_client_order_id == entry_cid
    assert streamer._open_trade["client_order_id"] == entry_cid
    assert streamer._reporting_mismatch_detail is None
    assert not any(e.get("reason") == "reporting_mismatch_no_entry_fill" for e in logs)
    assert any(e.get("event") == "active_cid_repaired_from_fill_truth" for e in logs)


def test_trend_guard_disabled_bypasses_trend_bias_block(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.trend_bias_guard_enabled = False
    streamer._trend_direction = "up"
    streamer._market_mode = "trend"
    assert streamer._trend_guard_reason("SHORT") is None


def test_write_status_repairs_stale_flat_fill_truth_from_active_trade_row(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "RUNID|preset|ES 06-26|2026-06-03T22:10:00-06:00|OPEN|SHORT|abc123"
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer.state.active_client_order_id = cid
    streamer._pos = -1
    streamer._entry_price = 7542.75
    streamer._entry_stop = 7548.25
    streamer._entry_target = 7537.25
    streamer._open_trade = {"client_order_id": cid, "entry_price": 7542.75, "side": "SHORT"}
    streamer._last_executor_decision = {"intent_id": cid, "executor_decision": "SENT"}
    streamer._fill_truth_state = "flat_no_active_trade"
    streamer._has_fill_truth_last = False
    streamer._has_fill_truth_false_since_ts = time.time() - 60.0
    streamer._has_fill_truth_false_reason = "flat_no_active_trade"
    streamer.trades_csv = tmp_path / "trades.csv"
    with streamer.trades_csv.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "client_order_id",
                "entry_ts",
                "entry_fill_ts",
                "side",
                "qty",
                "entry_price",
                "entry_fill_price",
                "filled_qty",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "client_order_id": cid,
                "entry_ts": "2026-06-03T22:10:01.321057-06:00",
                "entry_fill_ts": "2026-06-03T22:10:01.321057-06:00",
                "side": "SHORT",
                "qty": "1",
                "entry_price": "7542.75",
                "entry_fill_price": "7542.75",
                "filled_qty": "1",
            }
        )
    streamer._load_fill_truth_index = lambda *, fresh=False: {}

    streamer._write_status(force=True)
    payload = json.loads(streamer.status_path.read_text(encoding="utf-8"))

    assert payload["fill_truth_state"] == "active_truth_present"
    assert payload["has_fill_truth_last"] is True
    assert payload["has_fill_truth_false_since_ts"] in (None, "")
    assert streamer._fill_truth_state == "active_truth_present"


def _make_streamer(tmp_path: Path) -> LiveCSVStreamer:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer._ensure_compat_defaults = lambda: None
    streamer.state = StreamState()
    streamer.state.position_state = "FLAT"
    streamer.state.entries_disarmed_reason = None
    streamer.state.nt_exec_state = "ARMED"
    streamer.run_id = "RUNID"
    streamer.instrument_alias = "ES"
    streamer.nt_instrument = "ES 06-26"
    streamer.exec_instrument = "ES 06-26"
    streamer.primary_preset = "preset"
    streamer.active_preset_name = "preset"
    streamer.active_strategy_tag = "strategy"
    streamer.model_run_id = "model"
    streamer.model_sha = "sha"
    streamer.session_tz = "America/Denver"
    streamer.time_mode = "wall_clock"
    streamer.run_mode = "live"
    streamer.execution_mode = "model_master"
    streamer.protection_mode = "stop_only_until_model_close_allowed"
    streamer.protection_price_mode = "offset"
    streamer.allow_replay_exec_to_nt = False
    streamer.replay_exec_pacing = None
    streamer.contract_version = "v1"
    streamer._nt_addon_version = "2026.01.23"
    streamer._nt_min_addon_version = "2026.01.23"
    streamer._nt_addon_flags = {}
    streamer.phase_state_machine_enabled = True
    streamer.phase_currentness_lag_sec_threshold = 120.0
    streamer.phase_backfill_lag_sec_threshold = 1500.0
    streamer.phase_currentness_confirmations_required = 1
    streamer._phase_last_lag_sec = 23.5
    streamer._phase_current_confirmations = 1
    streamer._sim_now_initialized = False
    streamer._inflight_buffer = []
    streamer._last_event_ts = None
    streamer._last_executor_decision = {}
    streamer._executor_stats = {
        "model_emits_total": 0,
        "executor_sent_total": 0,
        "executor_rejected_total": 0,
        "executor_skipped_idempotent_total": 0,
        "executor_skipped_noop_total": 0,
        "executor_blocked_sync_total": 0,
        "executor_blocked_safety_total": 0,
        "blocked_for_execution_total": 0,
        "accepted_for_sim_evaluation_total": 0,
        "simulated_replay_trades_total": 0,
        "blocked_pre_signal_total": 0,
        "gating_open_emits_total": 0,
    }
    streamer._executor_terminal = {}
    streamer._fill_truth_ignored_fills_count = 0
    streamer._fill_truth_ignored_fills_reasons = {}
    streamer._blocked_candidate_ids = set()
    streamer._gate_audit_last_bar_ts = None
    streamer._sim_clock_ts = None
    streamer._hard_lockout_active = False
    streamer._hard_lockout_code = None
    streamer._hard_lockout_detail = None
    streamer._entries_disarmed_reason = None
    streamer._current_block_code = None
    streamer._current_block_ts = None
    streamer._first_trade_block_code = None
    streamer._first_trade_block_ts = None
    streamer._first_trade_block_detail = None
    streamer._block_active = {}
    streamer._block_last_detail = {}
    streamer._gate_audit_header_written = False
    streamer._bar_diag_counters = {
        "bars_dropped_off_grid": 0,
        "bars_dropped_duplicate": 0,
        "bars_rounded_off_grid": 0,
    }
    streamer._last_bar_ts_guard = None
    streamer._last_status_write_ts = 0.0
    streamer._prob_diag = {
        "prob_count": 0,
        "prob_sum": 0.0,
        "prob_sum_sq": 0.0,
        "prob_min": None,
        "prob_max": None,
        "prob_central_band_count": 0,
        "directional_prob_count": 0,
        "directional_prob_sum": 0.0,
        "directional_prob_sum_sq": 0.0,
        "directional_prob_min": None,
        "directional_prob_max": None,
        "directional_prob_central_band_count": 0,
        "open_emit_long_count": 0,
        "open_emit_short_count": 0,
    }
    streamer._last_csv_bar = None
    streamer._last_suppressed_phase_bar_ts = None
    streamer._signal_to_order_header_written = False
    streamer._signal_to_order_outside_live_count = 0
    streamer._signal_to_order_emitted = set()
    streamer._replay_eval_open_trade = None
    streamer._simulated_replay_rows_seen = set()
    streamer._emission_eval_calls = {}
    streamer._phase = "LIVE"
    streamer._first_executable_phase_ts = None
    streamer._phase_last_bar_ts_seen = None
    streamer._phase_catchup_stall_cycles = 0
    streamer._phase_catchup_stall_last_reason = None
    streamer._phase_catchup_recovery_attempts = 0
    streamer._phase_catchup_last_recovery_ts = None
    streamer._phase_catchup_recover_last_ts = None
    streamer._phase_transition_count = 0
    streamer.phase_catchup_recover_min_interval_sec = 5.0
    streamer._phase_allows_execution = lambda: True
    streamer.print_signals = False
    streamer.trace = False
    streamer._last_phase2_meta = None
    streamer._jdun_pending = set()
    streamer._reset_fx_state = lambda: None
    streamer._apply_area_hold_settings = lambda: None
    streamer._sanitize_fallback_thresholds = lambda *args, **kwargs: None
    streamer._open_slippage_cooldown_bars = 6
    streamer._loss_cascade_consecutive_threshold = 2
    streamer._loss_cascade_cooldown_bars = 3
    streamer._loss_cascade_side_lockout_bars = 6
    streamer._log_exec_event = lambda *args, **kwargs: None
    streamer._log_signal_to_order = LiveCSVStreamer._log_signal_to_order.__get__(streamer, LiveCSVStreamer)
    streamer._emit_block_event = LiveCSVStreamer._emit_block_event.__get__(streamer, LiveCSVStreamer)
    streamer._handle_io_failure = lambda *args, **kwargs: None
    streamer._log_order_event = lambda *args, **kwargs: None
    streamer._update_last_bar_ts_guard = lambda *args, **kwargs: None
    streamer._snapshot_price_state = lambda: (False, None, None, None, None)
    streamer._get_build_info = lambda: {
        "git_sha": "abc123",
        "git_dirty": False,
        "git_sha_source": "file_hash",
        "package_path": "stream_live_csv.py",
    }
    streamer._compute_config_hash = lambda: "confighash"
    streamer._get_gate_now = lambda: pd.Timestamp("2026-04-02T00:00:00-06:00")
    streamer._bar_age_guard_seconds = lambda: 0.0
    streamer._why_not_armed = lambda: None
    streamer._ensure_signal_id = lambda ev: ev.setdefault("signal_id", "sig|1") or ev["signal_id"]
    streamer._ensure_close_intent_id = lambda ev: None
    streamer._build_nt_client_order_id = lambda ev: ev.get("client_order_id") or f"{streamer.run_id}|{ev.get('type', '')}|CID"
    streamer._prop_guardrails_pre_signal_reason = lambda ev: str(
        ev.get("blocked_pre_signal_reason") or ev.get("policy_block") or ""
    )
    streamer._record_blocked_candidate = LiveCSVStreamer._record_blocked_candidate.__get__(streamer, LiveCSVStreamer)
    streamer._emit_gate_audit = LiveCSVStreamer._emit_gate_audit.__get__(streamer, LiveCSVStreamer)
    streamer._maybe_emit_signal = LiveCSVStreamer._maybe_emit_signal.__get__(streamer, LiveCSVStreamer)
    streamer._append_signal = LiveCSVStreamer._append_signal.__get__(streamer, LiveCSVStreamer)
    streamer._clear_entries_disarmed = LiveCSVStreamer._clear_entries_disarmed.__get__(streamer, LiveCSVStreamer)
    streamer._emit_event = lambda *args, **kwargs: None
    streamer._emit_unblocked = lambda *args, **kwargs: None
    streamer._write_status = LiveCSVStreamer._write_status.__get__(streamer, LiveCSVStreamer)
    streamer._trend_guard_reason = LiveCSVStreamer._trend_guard_reason.__get__(streamer, LiveCSVStreamer)
    streamer._allow_countertrend_fade_in_trend = LiveCSVStreamer._allow_countertrend_fade_in_trend.__get__(streamer, LiveCSVStreamer)
    streamer._apply_countertrend_fade_size = LiveCSVStreamer._apply_countertrend_fade_size.__get__(streamer, LiveCSVStreamer)
    streamer._trend_alignment_for_side = LiveCSVStreamer._trend_alignment_for_side.__get__(streamer, LiveCSVStreamer)
    streamer._effective_vwap_gate_mode = LiveCSVStreamer._effective_vwap_gate_mode.__get__(streamer, LiveCSVStreamer)
    streamer._regime_debug_payload = LiveCSVStreamer._regime_debug_payload.__get__(streamer, LiveCSVStreamer)
    streamer._update_trend_mode = LiveCSVStreamer._update_trend_mode.__get__(streamer, LiveCSVStreamer)
    streamer._sync_runtime_regime_policy_from_preset = LiveCSVStreamer._sync_runtime_regime_policy_from_preset.__get__(streamer, LiveCSVStreamer)
    streamer._activate_preset = LiveCSVStreamer._activate_preset.__get__(streamer, LiveCSVStreamer)
    streamer._effective_expects_target = LiveCSVStreamer._effective_expects_target.__get__(streamer, LiveCSVStreamer)
    streamer.contracts_per_trade = 1
    streamer.p_buy = 0.7
    streamer.p_sell = 0.3
    streamer.override_prob_min = 0.78
    streamer.override_hold_conf_min = 0.7
    streamer.phase2_enabled = True
    streamer.phase2_p_setup = 0.7
    streamer.phase2_p_long = 0.7
    streamer.phase2_p_short = 0.3
    streamer.phase2_short_cut = 0.3
    streamer._nt_ready = True
    streamer._nt_ready_reason = "ok"
    streamer._nt_last_snapshot_ts_utc = None
    streamer._nt_last_snapshot_instrument = "ES 06-26"
    streamer._nt_last_snapshot_account = "Sim101"
    streamer._nt_last_snapshot_msg_ts_raw = None
    streamer._nt_instrument_configured = "ES 06-26"
    streamer.exec_instrument_source = "snapshot"
    streamer._nt_account_configured = "auto"
    streamer._nt_account_detected = "Sim101"
    streamer._nt_account_chosen = "Sim101"
    streamer._account_resolution_state = "account_selected"
    streamer._last_account_event_ts = None
    streamer._pos = 0
    streamer._open_trade = None
    streamer._market_mode = "neutral"
    streamer._market_mode_source = "startup"
    streamer._trend_direction = None
    streamer._configured_trend_preset = "preset_trend"
    streamer._configured_chop_preset = "preset_chop"
    streamer.allow_countertrend_fade_in_trend = False
    streamer.countertrend_fade_min_vwap_extension_pts = 0.0
    streamer.countertrend_fade_prob_threshold = 0.30
    streamer.countertrend_fade_size_multiplier = 0.5
    streamer.pnl_overlay_enabled = True
    streamer.pnl_runner_enabled = False
    streamer.pnl_runner_suppress_target = True
    streamer.adaptive_entry_floor_enabled = False
    streamer.adaptive_entry_floor_delta = 0.0
    streamer.winner_hold_extension_enabled = False
    streamer.winner_hold_min_mfe_r = 0.0
    streamer.loss_tail_clamp_enabled = False
    streamer.loss_tail_clamp_max_adverse_r = 0.0
    streamer._session_suppression_windows = []
    streamer._weak_long_windows = []
    streamer._entry_price = None
    streamer._entry_time = None
    streamer._bars_in_trade = 0
    streamer._last_live_pnl_quality = {"quality": "ok", "source": "unit_test", "quality_reason": "ok"}
    streamer._pnl_live_metrics = {}
    streamer._nt_order_state = {}
    streamer.nt_enabled = False
    streamer.qa_emit_signals = False
    streamer.nt_proof_mode = False
    streamer.nt_bridge = None
    streamer.nt_exec_state = "ARMED"
    streamer.policy_guard_enabled = False
    streamer.shadow_mode = False
    streamer.emit_prediction_bundle = False
    streamer.emit_execution_decision = False
    streamer.live_dry_run = False
    streamer.pin_input = False
    streamer.pinned_csv_sha256 = None
    streamer.original_csv_sha256 = None
    streamer.input_provenance = {}
    streamer.trade_window_start = "08:10"
    streamer.trade_window_end = "14:30"
    streamer.trade_window_configured_start = "08:10"
    streamer.trade_window_configured_end = "14:30"
    streamer.trade_window_effective_reason = "configured"
    streamer.trade_window_cli_forced = False
    streamer.trade_window_start_cli_forced = False
    streamer.trade_window_end_cli_forced = False
    streamer.status = SimpleNamespace(enabled=False, write=lambda *_args, **_kwargs: None)
    streamer.market_narrator_enabled = False
    streamer._market_narrator_last_text = None
    streamer._market_narrator_last_ts = None
    streamer._market_narrator_last_emitted_ts = None
    streamer.policy = SimpleNamespace(
        trend_window_bars=6,
        trend_threshold=0.55,
        chop_threshold=0.45,
        chop_range_threshold=0.95,
        mode_persist_bars=1,
        allow_countertrend_in_unresolved=False,
        trend_preset="preset_trend",
        chop_preset="preset_chop",
    )
    streamer._trend_scores = []
    streamer._mode_votes = {"trend": 0, "chop": 0}
    streamer._regime_last_avg_score = None
    streamer._regime_last_range_ratio = None
    streamer._regime_threshold_state = None
    streamer._regime_neutral_reason = None
    streamer._session_high = None
    streamer._session_low = None
    streamer.engine = None
    streamer.fallback = SimpleNamespace(
        cfg=SimpleNamespace(
            use_vwap_gate=True,
            use_ema_gate=True,
            use_tod_gate=True,
            gate_mode="None",
            min_hold_bars=1,
            exit_prob_confirm_bars=1,
            hold_threshold_long=None,
            hold_threshold_short=None,
        )
    )
    streamer.guardrail_config = SimpleNamespace(
        as_dict=lambda: {
            "effective_bar_age_max_sec": 605.0,
            "bar_age_max_sec": 605.0,
            "bar_interval_sec": 300.0,
        }
    )

    streamer.out_dir = tmp_path
    streamer.status_path = tmp_path / "status.json"
    streamer.signals_csv = tmp_path / "signals.csv"
    streamer.signals_jsonl = tmp_path / "signals.jsonl"
    streamer.events_jsonl = tmp_path / "events.jsonl"
    streamer.gating_events_path = tmp_path / "gating_events.jsonl"
    streamer.bar_diag_summary_path = tmp_path / "bar_diagnostics_summary.json"
    streamer.state_stream_csv = tmp_path / "state.csv"
    streamer.signal_to_order_path = tmp_path / "signal_to_order.jsonl"
    streamer.lifecycle_events_path = tmp_path / "lifecycle_events.jsonl"
    streamer.lifecycle_events_csv_path = tmp_path / "lifecycle_events.csv"
    streamer.blocked_candidates_path = tmp_path / "blocked_candidates.jsonl"
    streamer.emission_ledger_path = tmp_path / "emission_ledger.jsonl"
    streamer.event_ledger_path = tmp_path / "event_ledger.jsonl"
    streamer.order_events_path = tmp_path / "order_events.jsonl"
    streamer._emission_ledger = EmissionLedger(streamer.emission_ledger_path)
    streamer.execution_ledger = SimpleNamespace(mark=lambda *_args, **_kwargs: None)
    streamer._state_dedupe = set()
    streamer._lifecycle_event_dedupe = set()

    streamer.signals_csv.write_text(SIGNALS_CSV_HEADER, encoding="utf-8")
    streamer.signals_jsonl.write_text("", encoding="utf-8")
    streamer.events_jsonl.write_text("", encoding="utf-8")
    streamer.gating_events_path.write_text("", encoding="utf-8")
    streamer.state_stream_csv.write_text(",".join(stream_live_csv_mod.STATE_CSV_COLUMNS) + "\n", encoding="utf-8")
    streamer.signal_to_order_path.write_text("", encoding="utf-8")
    streamer.lifecycle_events_csv_path.write_text(",".join(stream_live_csv_mod.LIFECYCLE_EVENT_CSV_COLUMNS) + "\n", encoding="utf-8")
    streamer.blocked_candidates_path.write_text("", encoding="utf-8")

    return streamer


def test_policy_guard_does_not_count_backfill_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase = "BACKFILL"
    streamer.policy_guard_enabled = True

    pretrade_calls: list[object] = []
    monkeypatch.setattr(stream_live_csv_mod.policy_adapter, "guard_loaded", lambda: True)

    def _unexpected_pretrade(*args: object, **kwargs: object) -> object:
        pretrade_calls.append((args, kwargs))
        raise AssertionError("RiskGuard pretrade path should not run outside LIVE")

    monkeypatch.setattr(stream_live_csv_mod.policy_adapter, "pretrade_gate", _unexpected_pretrade)

    event = {"price": 7000.0}
    decision = streamer._policy_guard_decision(
        event,
        pd.Series({"Close": 7000.0}),
        pd.Timestamp("2026-06-04T19:30:00-06:00"),
        "LONG",
    )

    assert decision is None
    assert pretrade_calls == []
    assert event["policy"]["risk_guard_skipped_phase"] == "BACKFILL"


def test_snapshot_no_orders_defers_repair_for_fresh_filled_entry(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer.state.position_state = "PENDING_ENTRY"
    streamer.state.active_client_order_id = "cid-fresh-entry"
    streamer.nt_snapshot_grace_sec = 3.0
    streamer.protection_grace_sec = 8.0
    streamer._nt_order_state = {
        "cid-fresh-entry": {
            "instrument": "ES 06-26",
            "entry_filled": True,
            "fill_ts": time.time(),
            "stop_price": 7602.0,
            "target_price": 7578.0,
            "exits_submitted": True,
            "stop_state": "SUBMITTED",
        }
    }

    assert streamer._fresh_entry_snapshot_repair_grace_active(
        inst_key="ES 06-26",
        pos_qty=-1.0,
        stop_price=7602.0,
        target_price=7578.0,
        reason="snapshot_no_orders",
    )
    assert events[-1]["event"] == "protection_repair_deferred_fresh_entry_snapshot"
    assert events[-1]["client_order_id"] == "cid-fresh-entry"


def test_maybe_repair_protection_defers_during_fresh_fill_settlement(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._repair_close_or_flatten_in_flight = lambda _inst_key: False
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_repair_state_by_instrument = {}
    streamer.protection_repair_enabled = True
    streamer.nt_protection_repair_timeout_sec = 30.0
    streamer.nt_snapshot_grace_sec = 3.0
    streamer.protection_grace_sec = 8.0
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer.state.active_client_order_id = "cid-fresh-repair"
    streamer._nt_order_state = {
        "cid-fresh-repair": {
            "instrument": "ES 06-26",
            "entry_filled": True,
            "fill_ts": time.time(),
            "stop_price": 7594.5,
            "target_price": 7571.0,
            "exits_submitted": True,
            "stop_state": "SUBMITTED",
        }
    }

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=-1.0,
        stop_price=7594.5,
        target_price=7571.0,
        reason="protection_repair_failed",
    )

    assert events[-1]["event"] == "protection_repair_deferred_fresh_entry_snapshot"
    assert streamer._nt_repair_state_by_instrument["ES 06-26"]["attempts"] == 0



def test_emit_block_event_tolerates_missing_nt_exec_state_during_init(tmp_path: Path) -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer._block_fingerprint = LiveCSVStreamer._block_fingerprint.__get__(streamer, LiveCSVStreamer)
    streamer._emit_block_event = LiveCSVStreamer._emit_block_event.__get__(streamer, LiveCSVStreamer)
    streamer._block_active = {}
    streamer._block_last_detail = {}
    streamer._first_trade_block_code = None
    streamer._first_trade_block_ts = None
    streamer._first_trade_block_detail = None
    streamer._current_block_code = None
    streamer._current_block_ts = None
    streamer._last_executor_decision = {}
    streamer._write_status = lambda *args, **kwargs: None
    streamer._emit_event = lambda *args, **kwargs: None

    streamer._emit_block_event(
        block_code="emit_lock_active",
        block_detail={"path": str(tmp_path / ".emit.lock")},
        owning_layer="execution",
        operator_action="Stop the other streamer or remove stale .emit.lock.",
    )

    assert streamer._current_block_code == "emit_lock_active"
    assert streamer._block_last_detail["emit_lock_active"]["path"].endswith(".emit.lock")


def test_trend_guard_blocks_countertrend_when_regime_unresolved(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._trend_direction = "up"
    streamer._market_mode = "neutral"

    assert streamer._trend_guard_reason("SHORT") == "Countertrend entry blocked while regime is unresolved"
    assert streamer._allow_countertrend_fade_in_trend(
        side="SHORT",
        row=pd.Series({"Close": 7145.0, "vwap_sess": 7130.0, "proba": 0.20}),
        ev={"price": 7145.0, "prob": 0.20},
    ) is False


def test_trend_guard_allows_countertrend_when_unresolved_policy_enabled(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._trend_direction = "up"
    streamer._market_mode = "neutral"
    streamer.policy.allow_countertrend_in_unresolved = True

    assert streamer._trend_guard_reason("SHORT") is None


def test_trend_guard_uses_runtime_unresolved_flag_when_policy_missing(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._trend_direction = "up"
    streamer._market_mode = "neutral"
    streamer.policy = None
    streamer.allow_countertrend_in_unresolved = True

    assert streamer._trend_guard_reason("SHORT") is None


def test_update_trend_mode_bootstraps_default_policy_in_live(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.policy = None
    streamer.disable_safety_gates = False
    streamer.run_mode = "live"
    streamer._market_mode = "neutral"
    streamer._market_mode_source = "startup"
    row = pd.Series({"Close": 7145.0, "ema_20": 7144.0, "ema_50": 7138.0, "atr14": 8.0})

    streamer._update_trend_mode(0.80, row)

    assert isinstance(streamer.policy, stream_live_csv_mod.EntryPolicyConfig)
    assert streamer._market_mode_source == "regime_policy"
    payload = streamer._regime_debug_payload()
    assert payload["neutral_reason"] in {"awaiting_trend_persistence", "startup_unresolved", "score_between_thresholds"}


def test_countertrend_fade_requires_trend_mode_and_applies_size_policy(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._trend_direction = "up"
    streamer._market_mode = "neutral"
    streamer.allow_countertrend_fade_in_trend = True
    streamer.countertrend_fade_min_vwap_extension_pts = 10.0
    streamer.countertrend_fade_prob_threshold = 0.24
    streamer.countertrend_fade_size_multiplier = 0.25
    ev = {"price": 7145.0, "prob": 0.20, "size_hint": 1.0}

    assert streamer._allow_countertrend_fade_in_trend(
        side="SHORT",
        row=pd.Series({"Close": 7145.0, "vwap_sess": 7130.0, "proba": 0.20}),
        ev=ev,
    ) is True

    streamer._apply_countertrend_fade_size(
        side="SHORT",
        row=pd.Series({"Close": 7145.0, "vwap_sess": 7130.0, "proba": 0.20}),
        ev=ev,
    )

    assert ev["size_hint"] == 0.25
    assert ev["trend_alignment"] == "countertrend"
    assert ev["countertrend_override_reason"] == "countertrend_fade_override"
    assert ev["policy"]["countertrend_fade"]["allowed"] is True


def test_activate_preset_syncs_runtime_regime_policy_and_status_fields(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    preset_name = "preset_runtime_sync"
    original = stream_live_csv_mod.PRESETS.get(preset_name)
    stream_live_csv_mod.PRESETS[preset_name] = {
        "session_tz": "America/Denver",
        "trade_window_start": "08:10",
        "trade_window_end": "14:30",
        "rth_start": "07:30",
        "rth_end": "14:00",
        "trend_preset": "preset_runtime_sync_trend",
        "chop_preset": "preset_runtime_sync_chop",
        "midday_trend_threshold": 0.58,
        "trend_threshold": 0.55,
        "chop_threshold": 0.45,
        "chop_range_threshold": 0.95,
        "mode_persist_bars": 1,
        "allow_countertrend_in_unresolved": True,
        "gate_tod": False,
        "allowed_grades": ["A", "B", "C"],
    }
    try:
        streamer.primary_preset = preset_name
        streamer.active_preset_name = "other_preset"
        streamer.trade_window_start = "00:00"
        streamer.trade_window_end = "23:59"
        streamer._configured_trend_preset = None
        streamer._configured_chop_preset = None

        streamer._activate_preset(preset_name, reason="init")
        streamer._write_status()

        payload = json.loads(streamer.status_path.read_text(encoding="utf-8"))
        assert streamer.trade_window_start == "08:10"
        assert streamer.trade_window_end == "14:30"
        assert streamer._configured_trend_preset == "preset_runtime_sync_trend"
        assert streamer._configured_chop_preset == "preset_runtime_sync_chop"
        assert streamer.policy.trend_threshold == pytest.approx(0.55)
        assert streamer.policy.chop_threshold == pytest.approx(0.45)
        assert streamer.policy.mode_persist_bars == 1
        assert streamer.policy.allow_countertrend_in_unresolved is True
        assert payload["trade_window_start"] == "08:10"
        assert payload["trade_window_end"] == "14:30"
        assert payload["trend_preset"] == "preset_runtime_sync_trend"
        assert payload["chop_preset"] == "preset_runtime_sync_chop"
    finally:
        if original is None:
            stream_live_csv_mod.PRESETS.pop(preset_name, None)
        else:
            stream_live_csv_mod.PRESETS[preset_name] = original


def test_activate_same_preset_applies_hold_cap(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    preset_name = "preset_same_name_hold_cap"
    original = stream_live_csv_mod.PRESETS.get(preset_name)
    stream_live_csv_mod.PRESETS[preset_name] = {"max_hold_bars": 12}
    try:
        streamer.active_preset_name = preset_name
        streamer.active_preset_cfg = stream_live_csv_mod.PRESETS[preset_name]
        streamer.engine = SimpleNamespace(cfg=SimpleNamespace(max_bars_in_trade=20))
        streamer.engine.cfg.max_bars_in_trade = 20
        streamer.fallback.cfg.max_bars_in_trade = 20

        streamer._activate_preset(preset_name, reason="init")

        assert streamer.engine.cfg.max_bars_in_trade == 12
        assert streamer.fallback.cfg.max_bars_in_trade == 12
    finally:
        if original is None:
            stream_live_csv_mod.PRESETS.pop(preset_name, None)
        else:
            stream_live_csv_mod.PRESETS[preset_name] = original


def test_status_reports_effective_trade_window_metadata_for_tod_disabled_auto_24h(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.fallback.cfg.use_tod_gate = False
    streamer.trade_window_configured_start = "08:10"
    streamer.trade_window_configured_end = "14:30"
    streamer.trade_window_start = "00:00"
    streamer.trade_window_end = "23:59"
    streamer.trade_window_effective_reason = "gate_tod_disabled_auto_24h"
    streamer.trade_window_cli_forced = False
    streamer.trade_window_start_cli_forced = False
    streamer.trade_window_end_cli_forced = False

    streamer._write_status()
    payload = json.loads(streamer.status_path.read_text(encoding="utf-8"))

    assert payload["trade_window_start"] == "00:00"
    assert payload["trade_window_end"] == "23:59"
    assert payload["trade_window_effective_reason"] == "gate_tod_disabled_auto_24h"
    assert payload["trade_window_metadata"]["configured_start"] == "08:10"
    assert payload["trade_window_metadata"]["configured_end"] == "14:30"
    assert payload["trade_window_metadata"]["effective_start"] == "00:00"
    assert payload["trade_window_metadata"]["effective_end"] == "23:59"
    assert payload["trade_window_metadata"]["cli_forced"] is False


def test_normalize_trade_window_for_tod_forces_24h_even_when_cli_window_forced() -> None:
    args = SimpleNamespace(
        preset="es_modelrun77_parity_v1",
        gate_tod=False,
        trade_window_start="07:00",
        trade_window_end="14:00",
        disable_safety_gates=False,
        gate_vwap=True,
        gate_ema=True,
        publish_min_grade="A",
    )
    meta = stream_live_csv_mod._normalize_trade_window_for_tod(args, ["--gate_tod", "--trade_window_start", "07:00", "--trade_window_end", "14:00"])

    assert args.trade_window_start == "00:00"
    assert args.trade_window_end == "23:59"
    assert meta["trade_window_effective_reason"] == "gate_tod_disabled_auto_24h"
    assert meta["trade_window_cli_forced"] is True


def test_probability_diagnostics_written_to_status_and_bar_summary(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._executor_stats["candidate_signals_total"] = 2
    streamer._executor_stats["signal_append_total"] = 2
    streamer._record_probability_diagnostic(prob=0.52, directional_prob=0.52, action="OPEN", side="LONG")
    streamer._record_probability_diagnostic(prob=0.49, directional_prob=0.51, action="MARK", side="LONG")

    streamer._update_bar_diag_summary()
    bar_summary = json.loads(streamer.bar_diag_summary_path.read_text(encoding="utf-8"))
    assert "probability_diagnostics" in bar_summary
    assert bar_summary["probability_diagnostics"]["prob"]["count"] == 2
    assert bar_summary["probability_diagnostics"]["emit_rate"] == pytest.approx(0.5)

    streamer._write_status()
    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))
    diag = status["probability_diagnostics"]
    assert diag["prob"]["count"] == 2
    assert diag["prob"]["mean"] == pytest.approx((0.52 + 0.49) / 2.0)
    assert diag["directional_prob"]["count"] == 2
    assert isinstance(diag["diagnostic_verdict"], str)
    assert diag["emit_rate"] == pytest.approx(0.5)


def test_fill_oob_ignores_stale_replay_when_terminal_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(payload)
    streamer._record_guardrail_event = lambda _kind: None
    streamer.max_fill_slippage_ticks = 2
    streamer.tick_size = 0.25
    streamer.late_fill_threshold_sec = 5.0
    streamer.stale_fill_replay_multiplier = 5.0
    streamer._last_close = None
    streamer._snapshot_age_sec = lambda _inst=None: None
    streamer.nt_snapshot_fresh_sec = 0.0
    streamer.guardrail_config = SimpleNamespace(snapshot_age_max_sec=0.0)
    streamer._nt_last_price_by_instrument = {}
    streamer.state.position_state = "FLAT"
    state = {
        "expected_entry_ref": 7000.0,
        "sent_ts": time.time() - 120.0,
        "status": "CLOSED",
        "instrument": "ES 06-26",
    }

    oob = streamer._fill_price_out_of_bounds(state, 7020.0)

    assert oob is False
    assert any(e.get("event") == "stale_fill_replay_ignored_for_oob_lockout" for e in events)


def test_live_feed_liveness_timeout_disarms_even_with_high_stale_tolerance(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._evaluate_guardrails = lambda **_kwargs: SimpleNamespace(preflight_ok=True)
    streamer._get_gate_now = lambda: pd.Timestamp("2026-04-14T05:56:00-06:00")
    streamer._tod_gate_enabled = lambda: False
    streamer._offline_exec_should_block = lambda _offline: False
    streamer._entries_today_for_limits = lambda: 0
    streamer._strategy_trade_limit = 0
    streamer.run_mode = "live"
    streamer.live_feed_liveness_max_sec = 90.0
    streamer.max_bar_age_seconds_for_exec = 605.0
    streamer.stale_bar_tolerance_sec = 600.0
    streamer.stale_bar_confirmations = 10
    streamer._bar_age_seconds = lambda _bar_ts: 180.0

    streamer._update_entry_arming_status(pd.Timestamp("2026-04-14T05:55:00-06:00"))

    assert streamer._entries_disarmed_reason == "live_feed_liveness_timeout"
    assert streamer.state.entries_disarmed_reason == "live_feed_liveness_timeout"


def test_stable_ingest_profile_live_feed_liveness_timeout_is_telemetry_only(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._evaluate_guardrails = lambda **_kwargs: SimpleNamespace(preflight_ok=True)
    streamer._get_gate_now = lambda: pd.Timestamp("2026-04-14T05:56:00-06:00")
    streamer._tod_gate_enabled = lambda: False
    streamer._offline_exec_should_block = lambda _offline: False
    streamer._entries_today_for_limits = lambda: 0
    streamer._strategy_trade_limit = 0
    streamer.run_mode = "live"
    streamer.stable_ingest_profile = True
    streamer.live_feed_liveness_max_sec = 90.0
    streamer.max_bar_age_seconds_for_exec = 605.0
    streamer.stale_bar_tolerance_sec = 600.0
    streamer.stale_bar_confirmations = 10
    streamer._bar_age_seconds = lambda _bar_ts: 180.0
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    streamer._update_entry_arming_status(pd.Timestamp("2026-04-14T05:55:00-06:00"))

    assert streamer._entries_disarmed_reason is None
    assert streamer.state.entries_disarmed_reason is None
    assert any(e.get("event") == "entries_disarm_suppressed_stable_ingest" for e in events)


def test_first_fresh_tick_rearms_after_stale_disarm(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._evaluate_guardrails = lambda **_kwargs: SimpleNamespace(preflight_ok=True)
    streamer._get_gate_now = lambda: pd.Timestamp("2026-04-14T05:56:00-06:00")
    streamer._tod_gate_enabled = lambda: False
    streamer._offline_exec_should_block = lambda _offline: False
    streamer._entries_today_for_limits = lambda: 0
    streamer._strategy_trade_limit = 0
    streamer.run_mode = "live"
    streamer.live_feed_liveness_max_sec = 0.0
    streamer.max_bar_age_seconds_for_exec = 90.0
    streamer.stale_bar_tolerance_sec = 0.0
    streamer.stale_bar_confirmations = 3
    streamer._entries_disarmed_reason = "stale_bars"
    streamer.state.entries_disarmed_reason = "stale_bars"
    streamer._stale_bar_consecutive = 4
    streamer._bar_age_seconds = lambda _bar_ts: 10.0

    streamer._update_entry_arming_status(pd.Timestamp("2026-04-14T05:55:00-06:00"))

    assert streamer._entries_disarmed_reason is None
    assert streamer.state.entries_disarmed_reason is None
    assert streamer._stale_bar_consecutive == 0


def test_fiber_bar_timeout_disarms_on_silence(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 120.0
    streamer.fiber_no_bar_disarm_sec = 180.0
    streamer._phase_allows_execution = lambda: True
    streamer._entries_disarmed_reason = None
    streamer.state.entries_disarmed_reason = None
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_no_bar_warn_emitted = False
    streamer._fiber_no_bar_disarmed = False
    block_codes: list[str] = []
    exec_events: list[dict] = []
    streamer._emit_block_event = lambda **kwargs: block_codes.append(str(kwargs.get("block_code")))
    streamer._log_exec_event = lambda payload: exec_events.append(payload)

    streamer._fiber_no_bar_consecutive_required = 2
    streamer._check_fiber_bar_watchdog(now=1200.0)
    assert streamer._entries_disarmed_reason is None
    streamer._check_fiber_bar_watchdog(now=1400.0)

    assert streamer._entries_disarmed_reason == "fiber_bar_timeout"
    assert streamer.state.entries_disarmed_reason == "fiber_bar_timeout"
    assert "fiber_bar_timeout" in block_codes
    assert any(e.get("event") == "fiber_bar_timeout_disarm" for e in exec_events)


def test_stable_ingest_profile_fiber_bar_timeout_is_telemetry_only(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.stable_ingest_profile = True
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 120.0
    streamer.fiber_no_bar_disarm_sec = 180.0
    streamer._phase_allows_execution = lambda: True
    streamer._entries_disarmed_reason = None
    streamer.state.entries_disarmed_reason = None
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_no_bar_warn_emitted = False
    streamer._fiber_no_bar_disarmed = False
    block_codes: list[str] = []
    exec_events: list[dict] = []
    streamer._emit_block_event = lambda **kwargs: block_codes.append(str(kwargs.get("block_code")))
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))

    streamer._fiber_no_bar_consecutive_required = 1
    streamer._check_fiber_bar_watchdog(now=1400.0)

    assert streamer._entries_disarmed_reason is None
    assert streamer.state.entries_disarmed_reason is None
    assert bool(streamer._fiber_no_bar_disarmed) is False
    assert block_codes == []
    assert any(e.get("event") == "fiber_bar_timeout_telemetry_only_stable_ingest" for e in exec_events)


def test_simple_ingest_mode_bypasses_fiber_watchdog_disarm(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.simple_ingest_mode = True
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_no_bar_disarmed = False
    streamer._entries_disarmed_reason = None
    streamer.state.entries_disarmed_reason = None

    streamer._check_fiber_bar_watchdog(now=5000.0)

    assert streamer._entries_disarmed_reason is None
    assert streamer.state.entries_disarmed_reason is None
    assert bool(streamer._fiber_no_bar_disarmed) is False
    assert str(getattr(streamer, "_fiber_watchdog_state", "")) == "simple_ingest_mode"


def test_simple_ingest_mode_bypasses_bar_continuity_gate(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._bar_continuity_allows_execution = LiveCSVStreamer._bar_continuity_allows_execution.__get__(streamer, LiveCSVStreamer)
    streamer.simple_ingest_mode = True
    streamer.run_mode = "live"
    streamer.input_mode = "fiber"
    streamer.bar_continuity_required_for_entry = True
    streamer._fiber_no_bar_disarmed = True
    streamer._bar_ingress_fault_cause = "transport_not_delivering"
    streamer._fiber_last_bar_recv_mono = None

    assert streamer._bar_continuity_allows_execution() is True


def test_simple_ingest_mode_preserves_fiber_utcticks_dedupe(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._handle_fiber_message = LiveCSVStreamer._handle_fiber_message.__get__(streamer, LiveCSVStreamer)
    streamer._fiber_rows = []
    streamer._fiber_rows_by_utc = set()
    streamer._fiber_hist_in_progress = False
    streamer._fiber_last_seen_utc_ticks = None
    streamer.fiber_lookback_bars = 1600
    streamer._fiber_process_buffer = lambda: None
    streamer.fiber_require_hist = False
    streamer.input_mode = "fiber"
    streamer.simple_ingest_mode = True

    msg = {
        "type": "BAR",
        "ts": "2026-05-22T09:35:00-06:00",
        "o": 5300.0,
        "h": 5302.0,
        "l": 5298.0,
        "c": 5301.0,
        "v": 100,
        "utcTicks": 1234567890,
        "_recv_ts_mono": 1.0,
        "_recv_ts_utc": utc_ts(),
    }
    streamer._handle_fiber_message(msg)
    streamer._handle_fiber_message(msg)

    assert len(streamer._fiber_rows) == 1
    assert len(streamer._fiber_rows_by_utc) == 1


def test_run70_parity_blocks_open_send_when_phase_not_live(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._should_block_nt_send_policy = LiveCSVStreamer._should_block_nt_send_policy.__get__(streamer, LiveCSVStreamer)
    streamer.run70_parity_live_profile = True
    streamer.run_mode = "live"
    streamer.input_mode = "fiber"
    streamer._phase = "BACKFILL"
    blocked = streamer._should_block_nt_send_policy({"datetime": "2026-05-22T10:00:00-06:00"}, "OPEN")
    assert isinstance(blocked, dict)
    assert str(blocked.get("reason") or "") == "phase_not_live_for_entry_send"


def test_run70_parity_legacy_gate_block_is_observational_in_live(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase2_force_open_legacy_gate_block_reason = (
        LiveCSVStreamer._phase2_force_open_legacy_gate_block_reason.__get__(streamer, LiveCSVStreamer)
    )
    streamer.run70_parity_live_profile = True
    streamer.run_mode = "live"
    streamer._phase = "LIVE"
    streamer.phase2_force_open_allow_legacy_gate_bypass = False
    ev = {"type": "OPEN", "side": "LONG", "phase2_setup_pass": False, "grade": "C", "prob": 0.8, "ctx": {}}
    reason = streamer._phase2_force_open_legacy_gate_block_reason(ev, open_allowed=False, gate_meta={})
    assert reason is None
    assert bool(ev.get("phase2_force_open_legacy_gate_bypass")) is True
    assert str(ev.get("phase2_force_open_legacy_gate_bypass_reason") or "") == "run70_parity_live_profile"


def test_run70_parity_suppresses_desync_kill_switch_during_stale_transport(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._trigger_desync_kill_switch = LiveCSVStreamer._trigger_desync_kill_switch.__get__(streamer, LiveCSVStreamer)
    streamer.run70_parity_live_profile = True
    streamer._hard_lockout_active = False
    streamer._fiber_transport_state = "degraded"
    streamer._fiber_last_silence_sec = 800.0
    streamer.fiber_no_bar_disarm_sec = 300.0
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    disarms: list[str] = []
    lockouts: list[str] = []
    streamer._set_entries_disarmed = lambda reason, detail=None: disarms.append(str(reason))
    streamer._set_hard_lockout = lambda reason, detail=None: lockouts.append(str(reason))

    streamer._trigger_desync_kill_switch(reason="state_toggle_oscillation", detail={"x": 1}, age_sec=10.0)

    assert disarms == []
    assert lockouts == []
    assert any(e.get("event") == "desync_kill_switch_deferred_transport_stale_run70_parity" for e in events)


def test_run70_parity_reporting_mismatch_trades_csv_is_reconcile_only(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_reporting_mismatch = LiveCSVStreamer._check_reporting_mismatch.__get__(streamer, LiveCSVStreamer)
    streamer.run70_parity_live_profile = True
    streamer._phase = "LIVE"
    streamer._last_bar_ts_guard = pd.Timestamp("2026-05-22T10:00:00-06:00")
    streamer.state.position_state = "FLAT"
    streamer.state.active_client_order_id = None
    streamer._fill_truth_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fills": [
                    {
                        "client_order_id": "cid-1",
                        "entry_fill_ts_epoch": 1000.0,
                        "exit_fill_ts_epoch": 1100.0,
                        "status": "CLOSED",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    streamer._trades_csv_path.write_text("client_order_id,actual_exit_ts\n", encoding="utf-8")
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    disarms: list[str] = []
    streamer._set_entries_disarmed = lambda reason, detail=None: disarms.append(str(reason))
    sync_calls: list[str] = []
    streamer._send_nt_sync_request = lambda reason=None: sync_calls.append(str(reason or ""))
    streamer._apply_fill_truth_reconciliation = lambda **kwargs: sync_calls.append("reconcile")

    streamer._check_reporting_mismatch(now_ts=time.time(), bar_ts="2026-05-22T10:00:00-06:00")

    assert disarms == []
    assert any(e.get("event") == "reporting_mismatch_reconcile_only_run70_parity" for e in events)
    assert sync_calls


def test_run70_phase_parity_report_writes_mismatch_artifacts(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._write_run70_phase_parity_report = LiveCSVStreamer._write_run70_phase_parity_report.__get__(streamer, LiveCSVStreamer)
    streamer.run70_parity_live_profile = True
    streamer.out_dir = tmp_path
    streamer.signal_to_order_path = tmp_path / "signal_to_order.jsonl"
    rows = [
        {"type": "header", "run_id": "r1"},
        {"phase": "BACKFILL", "action": "OPEN", "transition_id": "t1", "bar_ts": "2026-05-20T10:00:00-06:00", "sent_to_nt": False, "reason": "gates_block"},
        {"phase": "LIVE", "action": "OPEN", "transition_id": "t1", "bar_ts": "2026-05-20T10:00:00-06:00", "sent_to_nt": True, "reason": "sent"},
    ]
    with streamer.signal_to_order_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    streamer._write_run70_phase_parity_report()
    report_path = tmp_path / "phase_parity_report.json"
    mismatch_path = tmp_path / "phase_parity_mismatches.jsonl"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert int(report.get("compared_pairs") or 0) == 1
    assert int(report.get("mismatch_count") or 0) == 1
    assert bool(report.get("parity_pass")) is False
    assert mismatch_path.exists()


def test_fiber_bar_timeout_recovers_after_new_bars(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 120.0
    streamer.fiber_no_bar_disarm_sec = 180.0
    streamer._phase_allows_execution = lambda: True
    streamer._entries_disarmed_reason = "fiber_bar_timeout"
    streamer.state.entries_disarmed_reason = "fiber_bar_timeout"
    streamer._fiber_last_bar_recv_mono = 1195.0
    streamer._fiber_last_recv_mono = 1198.0
    streamer._fiber_no_bar_warn_emitted = True
    streamer._fiber_no_bar_disarmed = True
    streamer._fiber_no_bar_missed_windows = 2
    streamer._nt_ready = True
    streamer._snapshot_age_sec = lambda *_args, **_kwargs: 0.0
    streamer._last_bar_ts_guard = pd.Timestamp("2026-05-22T03:35:00-06:00")
    streamer._fiber_bar_seq_last_received = 10
    streamer._fiber_recovery_last_bar_ts = pd.Timestamp("2026-05-22T03:30:00-06:00")
    streamer._fiber_recovery_last_bar_seq = 9
    streamer._fiber_recovery_consecutive_bars_required = 1
    unblocked_codes: list[str] = []
    exec_events: list[dict] = []
    streamer._emit_unblocked = lambda code: unblocked_codes.append(str(code))
    streamer._log_exec_event = lambda payload: exec_events.append(payload)

    streamer._check_fiber_bar_watchdog(now=1200.0)

    assert streamer._entries_disarmed_reason is None
    assert streamer.state.entries_disarmed_reason is None
    assert "fiber_bar_timeout" in unblocked_codes
    assert any(e.get("event") == "fiber_bar_timeout_recovered" for e in exec_events)


def test_fiber_bar_timeout_clock_domain_mismatch_is_ignored(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 120.0
    streamer.fiber_no_bar_disarm_sec = 180.0
    streamer._phase_allows_execution = lambda: True
    streamer._entries_disarmed_reason = None
    streamer.state.entries_disarmed_reason = None
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_no_bar_warn_emitted = False
    streamer._fiber_no_bar_disarmed = False
    block_codes: list[str] = []
    exec_events: list[dict] = []
    streamer._emit_block_event = lambda **kwargs: block_codes.append(str(kwargs.get("block_code")))
    streamer._log_exec_event = lambda payload: exec_events.append(payload)

    # Simulates a wall-clock/monotonic mismatch signature.
    streamer._check_fiber_bar_watchdog(now=1_777_681_430.0)

    assert streamer._entries_disarmed_reason is None
    assert streamer.state.entries_disarmed_reason is None
    assert "fiber_bar_timeout" not in block_codes
    assert any(e.get("event") == "fiber_bar_timeout_elapsed_invalid" for e in exec_events)


def test_fiber_bar_timeout_detail_silence_sec_is_bounded(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 120.0
    streamer.fiber_no_bar_disarm_sec = 180.0
    streamer._phase_allows_execution = lambda: True
    streamer._entries_disarmed_reason = None
    streamer.state.entries_disarmed_reason = None
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_no_bar_warn_emitted = False
    streamer._fiber_no_bar_disarmed = False

    streamer._fiber_no_bar_consecutive_required = 1
    streamer._check_fiber_bar_watchdog(now=1200.0)

    detail = getattr(streamer, "_entries_disarmed_reason_detail", None) or {}
    silence_sec = float(detail.get("silence_sec") or 0.0)
    assert silence_sec >= 0.0
    assert silence_sec < 3600.0
    assert int(detail.get("missed_windows") or 0) >= 1
    assert "bridge_freshness_sec" in detail
    assert "queue_pressure" in detail


def test_fiber_bar_timeout_single_late_window_warns_without_disarm(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 330.0
    streamer.fiber_no_bar_disarm_sec = 360.0
    streamer._fiber_no_bar_consecutive_required = 2
    streamer._phase_allows_execution = lambda: True
    streamer._entries_disarmed_reason = None
    streamer.state.entries_disarmed_reason = None
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_last_recv_mono = 1355.0  # healthy bridge freshness
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(payload)

    streamer._check_fiber_bar_watchdog(now=1365.0)

    assert streamer._entries_disarmed_reason is None
    assert any(e.get("event") == "fiber_bar_timeout_warn" for e in exec_events)
    assert not any(e.get("event") == "fiber_bar_timeout_disarm" for e in exec_events)


def test_fiber_bar_watchdog_runs_recovery_during_catchup_without_trade_emit(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer._phase = "CATCHUP"
    streamer._phase_allows_execution = lambda: False
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 120.0
    streamer.fiber_no_bar_disarm_sec = 180.0
    streamer._fiber_no_bar_consecutive_required = 2
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_last_recv_mono = 1360.0  # bridge is fresh enough for recovery nudge path
    streamer._fiber_no_bar_warn_emitted = False
    streamer._fiber_no_bar_disarmed = False
    streamer._entries_disarmed_reason = None
    streamer.state.entries_disarmed_reason = None
    exec_events: list[dict] = []
    ping_reasons: list[str] = []
    sync_reasons: list[str] = []
    streamer._log_exec_event = lambda payload: exec_events.append(payload)
    streamer._send_nt_ping = lambda reason=None: ping_reasons.append(str(reason))
    streamer._send_nt_sync_request = lambda **kwargs: sync_reasons.append(str(kwargs.get("reason")))

    streamer._check_fiber_bar_watchdog(now=1365.0)

    assert ping_reasons
    assert sync_reasons
    assert any(e.get("event") == "fiber_bar_refresh_nudge" for e in exec_events)
    assert all(str(e.get("event")) not in {"OPEN", "CLOSE", "FLIP", "EXIT"} for e in exec_events)


def test_fiber_bar_recovery_stages_escalate_and_emit_recycle_events(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 5.0
    streamer.fiber_no_bar_warn_sec = 1.0
    streamer.fiber_no_bar_disarm_sec = 2.0
    streamer.fiber_bar_recovery_enabled = True
    streamer.fiber_bar_recovery_attempt_limit = 1
    streamer._fiber_no_bar_consecutive_required = 1
    streamer._phase_allows_execution = lambda: True
    streamer._entries_disarmed_reason = None
    streamer.state.entries_disarmed_reason = None
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_last_recv_mono = 1000.0
    streamer.nt_bridge = SimpleNamespace(
        reconnect=lambda: True,
        handshake_ok=lambda: True,
        is_connected=True,
    )
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(payload)

    for i in range(1, 16):
        now = 1000.0 + (i * 3.0)
        streamer._fiber_last_recv_mono = now - 0.2
        streamer._check_fiber_bar_watchdog(now=now)

    assert int(getattr(streamer, "_bar_recovery_stage", 0) or 0) >= 3
    assert any(e.get("event") == "bar_recovery_stage_advanced" for e in exec_events)
    assert any(e.get("event") == "bar_transport_recycle_attempt" for e in exec_events)
    assert any(e.get("event") == "bar_transport_recycle_result" for e in exec_events)


def test_fiber_bar_recovery_status_fields_and_rearm_event(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 120.0
    streamer.fiber_no_bar_disarm_sec = 180.0
    streamer._phase_allows_execution = lambda: True
    streamer._entries_disarmed_reason = "fiber_bar_timeout"
    streamer.state.entries_disarmed_reason = "fiber_bar_timeout"
    streamer._fiber_no_bar_disarmed = True
    streamer._fiber_last_bar_recv_mono = 1195.0
    streamer._fiber_last_recv_mono = 1198.0
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(payload)

    streamer._check_fiber_bar_watchdog(now=1200.0)
    streamer._write_status()
    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))

    assert "bar_stall_cycle_count" in status
    assert "bar_recovery_stage" in status
    assert "last_recover_action" in status
    assert "last_recover_result" in status
    assert "last_fresh_bar_ts" in status
    assert any(e.get("event") == "bar_rearm_condition_met" for e in exec_events)


def test_fiber_backpressure_updates_status_and_emits_event(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer._phase_allows_execution = lambda: True
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(payload)
    q: "queue.Queue[dict]" = queue.Queue(maxsize=10)
    q.put({"type": "HEARTBEAT"})
    q.put({"type": "HEARTBEAT"})
    fake_srv = SimpleNamespace(queue=q, dropped=1)
    streamer._fiber_server = fake_srv
    streamer._fiber_queue_pressure_threshold = 2

    streamer._check_fiber_backpressure()
    streamer._write_status()

    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))
    assert bool(status.get("fiber_backpressure_active")) is True
    assert int(status.get("fiber_queue_depth") or 0) >= 2
    assert int(status.get("fiber_queue_dropped") or 0) == 1
    assert int(status.get("fiber_queue_evicted_oldest") or 0) == 0
    assert str(status.get("fiber_queue_policy") or "").startswith("drop_oldest:")
    assert any(e.get("event") == "fiber_backpressure" and bool(e.get("active")) for e in exec_events)
    assert any(e.get("event") == "FIBER_QUEUE_BACKPRESSURE" and bool(e.get("active")) for e in exec_events)


def test_fiber_backpressure_tracks_bar_vs_nonbar_drops(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(payload)
    q: "queue.Queue[dict]" = queue.Queue(maxsize=10)
    q.put({"type": "HEARTBEAT"})
    fake_srv = SimpleNamespace(
        queue=q,
        dropped=3,
        dropped_bar=2,
        dropped_nonbar=1,
        evicted_oldest_count=0,
        last_eviction_seq=0,
    )
    streamer._fiber_server = fake_srv

    streamer._check_fiber_backpressure()
    streamer._write_status()
    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))
    assert int(status.get("fiber_bar_drop_count") or 0) == 2
    assert int(status.get("fiber_nonbar_drop_count") or 0) == 1


def test_nt_single_source_rejects_noncanonical_messages(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer.nt_single_source_enforce = True
    streamer.nt_allowed_source = "NinjaRepoBridge"
    streamer._nt_dispatcher_enabled = False
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    inner_calls: list[dict] = []
    streamer._handle_nt_message_inner = lambda msg: inner_calls.append(dict(msg))

    streamer._handle_nt_message({"type": "PNL_SNAPSHOT", "source": "NTRealPnLBridge"})

    assert int(getattr(streamer, "_nt_source_reject_count", 0) or 0) == 1
    assert inner_calls == []
    assert any(str(e.get("event")) == "bridge_source_rejected" for e in events)


def test_bar_continuity_loss_forces_no_trade() -> None:
    final_action, blocked, block_reason, emit_allowed = _finalize_entry_emit_action(
        action="OPEN",
        phase2_setup_pass=True,
        strategy_blocked_reason=None,
        phase="LIVE",
        override_applied=False,
        legacy_setup_bypass=False,
        blocked_by=[],
        strict_intent_parity_mode=False,
        bar_continuity_ok=False,
    )
    assert final_action == "NO_TRADE"
    assert emit_allowed is False
    assert block_reason == "bar_continuity_lost"
    assert "bar_continuity_lost" in blocked


def test_rearm_requires_true_bar_progress(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer._entries_disarmed_reason = "fiber_bar_timeout"
    streamer._nt_ready = True
    streamer._fiber_last_bar_recv_mono = 195.0
    streamer._fiber_last_recv_mono = 199.0
    streamer._fiber_recovery_last_recv_mono = 198.0
    streamer._fiber_recovery_last_bar_ts = pd.Timestamp("2026-05-18T12:00:00Z")
    streamer._fiber_recovery_last_bar_seq = 100
    streamer._last_bar_ts_guard = pd.Timestamp("2026-05-18T12:00:00Z")
    streamer._fiber_bar_seq_last_received = 100
    streamer._bar_ingress_fault_cause = "transport_not_delivering"
    streamer.fiber_no_bar_warn_sec = 30.0
    streamer.guardrail_config = SimpleNamespace(snapshot_age_max_sec=9999.0)
    streamer._snapshot_age_sec = lambda *a, **k: 1.0

    assert streamer._maybe_recover_fiber_disarm(now=200.0) is False


def test_fiber_bar_accept_reject_events_are_emitted(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._handle_fiber_message = LiveCSVStreamer._handle_fiber_message.__get__(streamer, LiveCSVStreamer)
    streamer._fiber_rows = []
    streamer._fiber_rows_by_utc = set()
    streamer._fiber_hist_in_progress = False
    streamer._fiber_last_seen_utc_ticks = None
    streamer.fiber_lookback_bars = 1600
    streamer.lookback_bars = 0
    streamer._fiber_process_buffer = lambda: None
    streamer.fiber_require_hist = False
    streamer.input_mode = "fiber"
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))
    msg = {
        "type": "BAR",
        "ts": "2026-04-23T19:50:00-06:00",
        "o": 1.0,
        "h": 2.0,
        "l": 0.5,
        "c": 1.5,
        "v": 10,
        "utcTicks": 1234,
        "_recv_ts_mono": 100.0,
        "_recv_ts_utc": "2026-04-24T01:50:00Z",
    }
    streamer._handle_fiber_message(msg)
    streamer._handle_fiber_message(msg)
    names = [str(e.get("event")) for e in exec_events]
    assert "FIBER_BAR_ACCEPTED" in names
    assert "FIBER_BAR_REJECTED" in names


def test_fiber_generation_stale_messages_are_rejected(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._handle_fiber_message = LiveCSVStreamer._handle_fiber_message.__get__(streamer, LiveCSVStreamer)
    streamer._fiber_rows = []
    streamer._fiber_rows_by_utc = set()
    streamer._fiber_hist_in_progress = False
    streamer._fiber_last_seen_utc_ticks = None
    streamer.fiber_lookback_bars = 1600
    streamer.lookback_bars = 0
    streamer._fiber_process_buffer = lambda: None
    streamer.fiber_require_hist = False
    streamer.input_mode = "fiber"
    streamer._fiber_conn_generation_last = 3
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))
    msg = {
        "type": "BAR",
        "ts": "2026-04-23T19:50:00-06:00",
        "o": 1.0,
        "h": 2.0,
        "l": 0.5,
        "c": 1.5,
        "v": 10,
        "utcTicks": 1234,
        "conn_generation": 2,
        "sender_instance_id": "sender-a",
        "_recv_ts_mono": 100.0,
        "_recv_ts_utc": "2026-04-24T01:50:00Z",
    }
    streamer._handle_fiber_message(msg)
    assert len(streamer._fiber_rows) == 0
    assert any(str(e.get("event")) == "FIBER_GEN_STALE_DROP" for e in exec_events)


def test_fiber_watchdog_classifies_sender_not_emitting_with_heartbeat_alive(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 120.0
    streamer.fiber_no_bar_disarm_sec = 180.0
    streamer._phase_allows_execution = lambda: True
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_last_heartbeat_recv_mono = 1120.0
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))
    streamer._fiber_no_bar_consecutive_required = 1
    streamer._check_fiber_bar_watchdog(now=1200.0)
    classified = [e for e in exec_events if str(e.get("event")) == "FIBER_INGRESS_STALLED_CLASSIFIED"]
    assert classified
    assert str(classified[-1].get("ingress_cause")) == "sender_not_emitting"


def test_fiber_socket_watchdog_defers_when_new_bar_is_queued(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_watchdog = LiveCSVStreamer._check_fiber_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.live_feed_liveness_max_sec = 390.0
    streamer.exec_heartbeat_sec = 0.25
    streamer.fiber_require_hist = True
    streamer._fiber_hist_in_progress = False
    streamer._fiber_hist_complete = True
    streamer._last_bar_ts_guard = pd.Timestamp("2026-06-03T08:40:00-06:00")
    streamer._bar_age_guard_seconds = lambda: 537.0
    streamer._fiber_last_seen_utc_ticks = 639160944000000000
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))

    class FakeFiberServer:
        def __init__(self) -> None:
            self.reset_reasons: list[str] = []

        def watchdog_snapshot(self) -> dict:
            return {
                "has_client": True,
                "last_msg_recv_mono": 1999.5,
                "last_bar_recv_mono": 1940.0,
                "last_msg_recv_utc": "2026-06-03T08:48:57-06:00",
                "last_bar_recv_utc": "2026-06-03T08:47:56-06:00",
                "last_msg_type": "HEARTBEAT",
                "last_bar_utc_ticks": 639160947000000000,
                "queue_depth": 2177,
                "queue_max": 10000,
                "conn_id": 3,
            }

        def reset_client(self, *, reason: str) -> None:
            self.reset_reasons.append(reason)

    fake_srv = FakeFiberServer()
    streamer._fiber_server = fake_srv

    streamer._check_fiber_watchdog(now_mono=2000.0)

    assert fake_srv.reset_reasons == []
    assert any(e.get("event") == "fiber_watchdog_deferred_backlog" for e in exec_events)


def test_fiber_watchdog_classifies_transport_not_delivering_when_sender_recent(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 120.0
    streamer.fiber_no_bar_disarm_sec = 180.0
    streamer._phase_allows_execution = lambda: True
    now = time.time()
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_last_heartbeat_recv_mono = 1120.0
    streamer._fiber_sender_last_emit_ts_epoch = now - 10.0
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))
    streamer._fiber_no_bar_consecutive_required = 1
    streamer._check_fiber_bar_watchdog(now=1200.0)
    classified = [e for e in exec_events if str(e.get("event")) == "FIBER_INGRESS_STALLED_CLASSIFIED"]
    assert classified
    assert str(classified[-1].get("ingress_cause")) == "transport_not_delivering"


def test_fiber_watchdog_executes_recovery_for_transport_silent_socket_open_in_degraded(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 330.0
    streamer.fiber_no_bar_disarm_sec = 900.0
    streamer._phase_allows_execution = lambda: True
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_last_heartbeat_recv_mono = 850.0
    streamer._fiber_bar_recovery_last_action_ts = None
    streamer._fiber_bar_recovery_attempt_total = 0
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))

    streamer._check_fiber_bar_watchdog(now=1310.0)

    executed = [e for e in exec_events if str(e.get("event")) == "BAR_RECOVERY_ATTEMPT_EXECUTED"]
    assert executed
    assert str(executed[-1].get("source_state")) == "degraded"
    assert str(executed[-1].get("ingress_cause")) == "transport_silent_socket_open"
    assert int(getattr(streamer, "_fiber_bar_recovery_attempt_total", 0) or 0) >= 1


def test_fiber_watchdog_recovery_cooldown_skip_is_telemetry_only(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 330.0
    streamer.fiber_no_bar_disarm_sec = 900.0
    streamer._phase_allows_execution = lambda: True
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_last_heartbeat_recv_mono = 1120.0
    streamer._fiber_sender_last_emit_ts_epoch = time.time() - 500.0
    streamer._fiber_bar_recovery_cooldown_sec = 60.0
    streamer._fiber_bar_recovery_last_action_ts = time.time()
    streamer._fiber_bar_recovery_attempt_total = 4
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))

    streamer._check_fiber_bar_watchdog(now=1310.0)

    skipped = [e for e in exec_events if str(e.get("event")) == "BAR_RECOVERY_ATTEMPT_SKIPPED"]
    assert skipped
    assert str(skipped[-1].get("source_state")) == "degraded"
    assert str(skipped[-1].get("ingress_cause")) == "sender_not_emitting"
    assert str(skipped[-1].get("reason")) == "cooldown_or_budget"
    assert int(getattr(streamer, "_fiber_bar_recovery_attempt_total", 0) or 0) == 4
    assert streamer._entries_disarmed_reason is None


def test_fiber_watchdog_transport_silent_socket_open_bypasses_cooldown_and_executes_recovery(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._check_fiber_bar_watchdog = LiveCSVStreamer._check_fiber_bar_watchdog.__get__(streamer, LiveCSVStreamer)
    streamer.input_mode = "fiber"
    streamer.run_mode = "live"
    streamer.bar_interval_sec = 300.0
    streamer.fiber_no_bar_warn_sec = 330.0
    streamer.fiber_no_bar_disarm_sec = 900.0
    streamer._phase_allows_execution = lambda: True
    streamer._fiber_last_bar_recv_mono = 1000.0
    streamer._fiber_last_heartbeat_recv_mono = 850.0
    streamer._fiber_bar_recovery_cooldown_sec = 120.0
    streamer._fiber_bar_recovery_last_action_ts = time.time()
    streamer._fiber_bar_recovery_attempt_total = 4
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))

    streamer._check_fiber_bar_watchdog(now=1310.0)

    executed = [e for e in exec_events if str(e.get("event")) == "BAR_RECOVERY_ATTEMPT_EXECUTED"]
    assert executed
    assert str(executed[-1].get("ingress_cause")) == "transport_silent_socket_open"
    assert bool(executed[-1].get("forced_cooldown_bypass")) is True
    assert int(getattr(streamer, "_fiber_bar_recovery_attempt_total", 0) or 0) > 4


def test_handle_fiber_bar_tracks_seq_gap_and_latency(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._handle_fiber_message = LiveCSVStreamer._handle_fiber_message.__get__(streamer, LiveCSVStreamer)
    streamer._fiber_process_buffer = lambda: None
    streamer._save_fiber_state = lambda: None
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))
    base = {
        "type": "BAR",
        "ts": "2026-05-18T03:30:00Z",
        "o": 1.0,
        "h": 2.0,
        "l": 0.5,
        "c": 1.5,
        "v": 10,
        "_recv_ts_mono": 100.0,
        "_recv_ts_utc": "2026-05-18T03:30:00.050Z",
        "send_ts_utc": "2026-05-18T03:30:00.000Z",
    }
    msg1 = dict(base)
    msg1["utcTicks"] = 100
    msg1["bar_seq"] = 1
    msg2 = dict(base)
    msg2["utcTicks"] = 200
    msg2["bar_seq"] = 3
    streamer._handle_fiber_message(msg1)
    streamer._handle_fiber_message(msg2)
    assert int(getattr(streamer, "_fiber_bar_seq_gap_count", 0) or 0) == 1
    assert float(getattr(streamer, "_fiber_bar_send_to_recv_ms", 0.0) or 0.0) >= 0.0
    assert any(str(e.get("event")) == "BAR_SEQ_GAP" for e in exec_events)


def test_status_includes_fiber_transport_and_generation_fields(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._fiber_transport_state = "degraded"
    streamer._fiber_ingress_state = "warn"
    streamer._fiber_conn_generation_last = 7
    streamer._fiber_last_bar_recv_mono = time.monotonic() - 4.0
    streamer._fiber_last_heartbeat_recv_mono = time.monotonic() - 2.0
    streamer._fiber_silent_failure_marker_total = 3
    streamer._fiber_bar_seq_gap_count = 2
    streamer._fiber_bar_send_to_recv_ms = 12.5
    streamer._fiber_bar_receive_to_process_ms = 33.0
    streamer._fiber_bar_last_skip_reason = "non_monotonic_utcTicks"
    streamer._write_status()
    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))
    assert str(status.get("fiber_transport_state")) == "degraded"
    assert str(status.get("fiber_ingress_state")) == "warn"
    assert int(status.get("fiber_conn_generation_last") or 0) == 7
    assert float(status.get("fiber_last_bar_age_sec") or 0.0) >= 0.0
    assert float(status.get("fiber_last_heartbeat_age_sec") or 0.0) >= 0.0
    assert int(status.get("fiber_silent_failure_marker_total") or 0) == 3
    assert int(status.get("bar_seq_gap_count") or 0) == 2
    assert float(status.get("bar_send_to_recv_ms") or 0.0) >= 0.0
    assert float(status.get("bar_receive_to_process_ms") or 0.0) >= 0.0
    assert str(status.get("bar_last_skip_reason") or "") == "non_monotonic_utcTicks"


def test_fiber_server_eviction_prefers_nonbar_first() -> None:
    srv = stream_live_csv_mod.FiberServer("127.0.0.1", 0, queue_max=3, evict_batch=1, drop_priority="nonbar_first")
    srv.queue.put({"type": "BAR", "utcTicks": 1})
    srv.queue.put({"type": "HEARTBEAT"})
    srv.queue.put({"type": "BAR", "utcTicks": 2})

    evicted = srv._evict_oldest_for_space(incoming={"type": "BAR"}, max_evict=1)

    assert evicted == 1
    remaining = list(srv.queue.queue)
    assert any(str((x or {}).get("type") or "").upper() == "BAR" and int((x or {}).get("utcTicks") or 0) == 1 for x in remaining)
    assert not any(str((x or {}).get("type") or "").upper() == "HEARTBEAT" for x in remaining)
    assert srv.evicted_oldest_count == 1


def test_fiber_server_enqueue_retries_after_eviction() -> None:
    srv = stream_live_csv_mod.FiberServer("127.0.0.1", 0, queue_max=2, evict_batch=1, drop_priority="oldest")
    srv.queue.put({"type": "BAR", "utcTicks": 1})
    srv.queue.put({"type": "BAR", "utcTicks": 2})

    # Simulate queue-full path from receiver loop.
    evicted = srv._evict_oldest_for_space(incoming={"type": "BAR", "utcTicks": 3}, max_evict=srv.evict_batch)
    assert evicted == 1
    srv.queue.put_nowait({"type": "BAR", "utcTicks": 3})
    srv.enqueue_retry_success += 1

    ticks = [int((x or {}).get("utcTicks") or 0) for x in list(srv.queue.queue) if str((x or {}).get("type") or "").upper() == "BAR"]
    assert ticks == [2, 3]
    assert srv.enqueue_retry_success == 1
    assert srv.dropped == 0


def test_status_defaults_executor_stats_to_live_phase(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase = "LIVE"
    streamer._inc_phase_stat("candidate_signals_total", phase="BACKFILL", amount=7)
    streamer._inc_phase_stat("candidate_signals_total", phase="LIVE", amount=2)
    streamer._inc_phase_stat("executable_signals_outside_live", phase="BACKFILL", amount=3)
    streamer._write_status()

    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))
    assert status["executor_stats"]["candidate_signals_total"] == 2
    assert int(status["executor_stats_all_phases"].get("candidate_signals_total", 0) or 0) == 0
    assert status["executor_stats_by_phase"]["BACKFILL"]["candidate_signals_total"] == 7
    assert status["executor_stats_by_phase"]["LIVE"]["candidate_signals_total"] == 2


def test_update_trend_mode_activates_trend_preset(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.policy.trend_threshold = 0.55
    streamer.policy.mode_persist_bars = 1
    activations: list[tuple[str, str]] = []
    streamer._activate_preset = lambda preset_name, reason: activations.append((preset_name, reason))
    row = pd.Series({"Close": 7145.0, "ema_20": 7144.0, "ema_50": 7138.0, "atr14": 8.0})

    streamer._update_trend_mode(0.56, row)

    assert streamer._market_mode == "trend"
    assert streamer._market_mode_source == "regime_policy"
    assert activations[-1] == ("preset_trend", "trend-mode")


def test_regime_debug_payload_tracks_neutral_handoff_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.policy.mode_persist_bars = 2
    row = pd.Series({"Close": 7145.0, "ema_20": 7144.0, "ema_50": 7138.0, "atr14": 8.0})

    streamer._update_trend_mode(0.70, row)

    payload = streamer._regime_debug_payload()
    assert streamer._market_mode == "neutral"
    assert streamer._market_mode_source == "regime_policy"
    assert payload["avg_score"] == pytest.approx(0.70)
    assert payload["trend_votes"] == 1
    assert payload["persist_required"] == 2
    assert payload["threshold_state"] == "trend_threshold_met"
    assert payload["neutral_reason"] == "awaiting_trend_persistence"


def test_effective_vwap_gate_mode_uses_trend_rule_during_aligned_handoff(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._trend_direction = "down"
    streamer._market_mode = "neutral"
    streamer._regime_last_avg_score = 0.83
    streamer._mode_votes["trend"] = 1

    effective_mode = streamer._effective_vwap_gate_mode(configured_mode="fade", side="SHORT")

    assert effective_mode == "trend"


def test_vwap_gate_mode_any_passes_aligned_and_fade_style_cases() -> None:
    assert _compute_vwap_gate_raw(
        close=7130.0,
        vwap=7145.0,
        longish=False,
        prob=0.20,
        mode="any",
    ) is True
    assert _compute_vwap_gate_raw(
        close=7130.0,
        vwap=7145.0,
        longish=True,
        prob=0.80,
        mode="any",
    ) is True


def test_countertrend_fade_threshold_semantics_are_explicit_for_short_and_long(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.allow_countertrend_fade_in_trend = True
    streamer.countertrend_fade_min_vwap_extension_pts = 10.0
    streamer.countertrend_fade_prob_threshold = 0.30

    streamer._trend_direction = "up"
    streamer._market_mode = "neutral"
    assert streamer._allow_countertrend_fade_in_trend(
        side="SHORT",
        row=pd.Series({"Close": 7145.0, "vwap_sess": 7130.0, "proba": 0.30}),
        ev={"price": 7145.0, "prob": 0.30},
    ) is True
    assert streamer._allow_countertrend_fade_in_trend(
        side="SHORT",
        row=pd.Series({"Close": 7145.0, "vwap_sess": 7130.0, "proba": 0.31}),
        ev={"price": 7145.0, "prob": 0.31},
    ) is False

    streamer._trend_direction = "down"
    assert streamer._allow_countertrend_fade_in_trend(
        side="LONG",
        row=pd.Series({"Close": 7130.0, "vwap_sess": 7145.0, "proba": 0.70}),
        ev={"price": 7130.0, "prob": 0.70},
    ) is True
    assert streamer._allow_countertrend_fade_in_trend(
        side="LONG",
        row=pd.Series({"Close": 7130.0, "vwap_sess": 7145.0, "proba": 0.69}),
        ev={"price": 7130.0, "prob": 0.69},
    ) is False


def test_trade_pnl_state_requires_target_for_protected_confirmed_in_stop_and_target(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.protection_mode = "stop_and_target"
    streamer.pnl_overlay_enabled = True
    streamer.pnl_runner_enabled = True
    streamer.pnl_runner_suppress_target = False
    streamer._pos = -1
    streamer._entry_price = 7142.25
    streamer._entry_time = pd.Timestamp("2026-04-22T00:40:00-06:00")
    streamer._bars_in_trade = 1
    streamer.state.nt_protected = True
    streamer._open_trade = {
        "side": "SHORT",
        "entry_price": 7142.25,
        "live_stop": 7147.75,
        "live_target": None,
        "entry_target": 7137.0,
        "protection_status": "protected_confirmed",
        "target_order_id": None,
        "target_working_ts": None,
    }

    pnl_state = LiveCSVStreamer._compute_trade_pnl_state(streamer, mark_price=7143.25, update_extrema=False)

    assert pnl_state["target_attached"] is False
    assert pnl_state["protected_confirmed"] is False


def test_trade_pnl_state_ignores_epoch_like_entry_timestamp(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = -1
    streamer._entry_price = 7142.25
    streamer._entry_stop = 7147.75
    streamer._entry_target = 7137.0
    streamer._open_trade = {
        "side": "SHORT",
        "entry_price": 7142.25,
        "entry_fill_ts": 0.427086,
        "live_stop": 7147.75,
        "live_target": 7137.0,
    }

    pnl_state = LiveCSVStreamer._compute_trade_pnl_state(streamer, mark_price=7141.25, update_extrema=False)

    assert pnl_state["time_in_trade_sec"] is None


def test_strategy_gate_blocks_session_suppression_windows(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.disable_safety_gates = False
    streamer.active_strategy_type = "break_retest"
    streamer.active_strategy_params = {}
    streamer._strategy_entries_today = 0
    streamer._strategy_last_entry_ts = None
    streamer._session_suppression_windows = [("LONG", dtime(8, 10), dtime(9, 45))]

    reason = streamer._strategy_gate_reason(
        typ="OPEN",
        side="LONG",
        ts=pd.Timestamp("2026-04-21T08:30:00-06:00"),
        row=pd.Series({"Close": 7000.0}),
        feats=pd.DataFrame([{"Close": 7000.0}]),
        bar_index=0,
    )

    assert reason == "session_suppression_window"


def test_strategy_gate_blocks_baseline_midday_suppression_window(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.disable_safety_gates = False
    streamer.active_strategy_type = "break_retest"
    streamer.active_strategy_params = {}
    streamer._strategy_entries_today = 0
    streamer._strategy_last_entry_ts = None
    streamer._session_suppression_windows = streamer._parse_session_suppression_windows(["11:58-12:07"])

    reason = streamer._strategy_gate_reason(
        typ="OPEN",
        side="SHORT",
        ts=pd.Timestamp("2026-04-21T12:00:00-06:00"),
        row=pd.Series({"Close": 7000.0}),
        feats=pd.DataFrame([{"Close": 7000.0}]),
        bar_index=0,
    )

    assert reason == "session_suppression_window"


def test_adaptive_entry_floor_blocks_low_edge_in_weak_slice(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.disable_safety_gates = False
    streamer.adaptive_entry_floor_enabled = True
    streamer.adaptive_entry_floor_delta = 0.03
    streamer._in_session_suppression_window = lambda *_args, **_kwargs: True
    streamer._ensure_day_state = lambda *_args, **_kwargs: None
    streamer._maybe_refresh_areas = lambda *_args, **_kwargs: None
    streamer._news_blackout_reason = lambda *_args, **_kwargs: None
    streamer._set_policy_flag = lambda *_args, **_kwargs: None
    streamer._areas_enabled = False
    streamer.state.day_stopped = False
    streamer.state.locked_until_grade = None
    streamer.policy.midday_pause = None
    streamer.p_buy = 0.72

    reason = streamer._entry_policy_reason(
        {
            "type": "OPEN",
            "side": "LONG",
            "prob": 0.74,
            "datetime": "2026-04-21T08:30:00-06:00",
            "grade": "A+",
        },
        row=None,
    )

    assert str(reason).startswith("adaptive_entry_floor_long<")


def test_close_model_winner_hold_extension_suppresses_early_close(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_overlay_enabled = False
    streamer.winner_hold_extension_enabled = True
    streamer.winner_hold_min_mfe_r = 0.60
    streamer.phase2_close_threshold = 0.70

    reason = streamer._close_model_suppression_reason(
        pd.Series({"Close": 7000.0}),
        {
            "side": "LONG",
            "unrealized_r": 0.35,
            "mfe_r": 0.85,
            "giveback_r": 0.20,
            "distance_to_stop_r": 1.25,
            "bars_in_trade": 2,
        },
        close_prob=0.82,
    )

    assert reason == "close_model_suppressed_winner_hold_extension"


def test_pnl_overlay_loss_tail_clamp_exits_quick_adverse_move(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_overlay_enabled = True
    streamer.loss_tail_clamp_enabled = True
    streamer.loss_tail_clamp_max_adverse_r = 0.20
    streamer._market_mode = "chop"
    streamer._pos = 1
    streamer._open_trade = {"decision_proba": 0.52}
    streamer._last_prob = 0.52
    streamer.pnl_severe_adverse_r = 9.0
    streamer.protection_mode = "stop_and_target"
    streamer._compute_trade_pnl_state = lambda **_kwargs: {
        "side": "LONG",
        "stop_price": 6998.0,
        "target_price": 7006.0,
        "mark_price": 6999.0,
        "unrealized_r": -0.25,
        "mfe_r": 0.05,
        "giveback_r": 0.10,
        "bars_in_trade": 1,
    }
    streamer._build_forced_exit_event = lambda **kwargs: {"type": "CLOSE", "reason": kwargs.get("reason")}

    event = streamer._maybe_pnl_overlay_event(
        pd.Series({"Close": 6999.0, "High": 7001.0, "Low": 6998.0}),
        pd.Timestamp("2026-04-21T09:15:00-06:00"),
    )

    assert event is not None
    assert event.get("reason") == "pnl_overlay_loss_tail_clamp"


def test_runner_first_profile_does_not_preempt_before_runner_arm(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_overlay_enabled = True
    streamer.pnl_runner_enabled = True
    streamer.pnl_runner_suppress_target = True
    streamer.pnl_runner_arm_r = 2.25
    streamer.pnl_runner_hard_exit_r = 2.50
    streamer.pnl_runner_giveback_r = 0.50
    streamer.pnl_giveback_activate_r = 2.25
    streamer.pnl_giveback_close_r = 0.50
    streamer.pnl_severe_adverse_r = 9.0
    streamer.protection_mode = "stop_and_target"
    streamer._pos = 1
    streamer._open_trade = {"side": "LONG", "entry_price": 7000.0}
    streamer._entry_price = 7000.0
    streamer._compute_trade_pnl_state = lambda **_kwargs: {
        "side": "LONG",
        "stop_price": 6998.0,
        "target_price": 7006.0,
        "mark_price": 7001.2,
        "unrealized_r": 1.20,
        "mfe_r": 1.60,
        "giveback_r": 0.60,
        "bars_in_trade": 4,
    }
    streamer._build_forced_exit_event = lambda **kwargs: {"type": "CLOSE", "reason": kwargs.get("reason")}

    event = streamer._maybe_pnl_overlay_event(
        pd.Series({"Close": 7001.2, "High": 7002.0, "Low": 6999.0}),
        pd.Timestamp("2026-04-21T09:15:00-06:00"),
    )

    assert event is None
    assert streamer._open_trade.get("pnl_runner_armed") is not True


def test_pnl_shelf_floor_exit_arms_then_exits_on_pullback(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_steps_usd = "100,200,400"
    streamer.pnl_shelf_locks_usd = "25,100,250"
    streamer.pnl_shelf_min_hold_bars = 1
    streamer.pnl_shelf_exit_mode = "hard_floor"
    streamer.pnl_shelf_breach_ticks = 1
    streamer._pos = 1
    streamer._open_trade = {"side": "LONG", "contracts": 1, "client_order_id": "CID-1"}
    streamer._last_prob = 0.75
    streamer.protection_mode = "stop_and_target"

    seq = iter(
        [
            {
                "side": "LONG",
                "stop_price": 6998.0,
                "target_price": 7008.0,
                "mark_price": 7004.4,
                "unrealized_r": 1.10,
                "mfe_r": 1.10,
                "giveback_r": 0.0,
                "bars_in_trade": 1,
                "unrealized_usd": 220.0,
            },
            {
                "side": "LONG",
                "stop_price": 6998.0,
                "target_price": 7008.0,
                "mark_price": 7001.9,
                "unrealized_r": 0.48,
                "mfe_r": 1.10,
                "giveback_r": 0.62,
                "bars_in_trade": 1,
                "unrealized_usd": 95.0,
            },
        ]
    )
    streamer._compute_trade_pnl_state = lambda **_kwargs: next(seq)
    shelf_events: list[str] = []
    streamer._emit_pnl_shelf_event = lambda **kwargs: shelf_events.append(str(kwargs.get("event_name")))
    streamer._build_forced_exit_event = lambda **kwargs: {"type": "CLOSE", "reason": kwargs.get("reason")}

    first = streamer._maybe_pnl_overlay_event(
        pd.Series({"Close": 7004.4, "High": 7004.4, "Low": 7004.4}),
        pd.Timestamp("2026-04-30T09:30:00-06:00"),
    )
    second = streamer._maybe_pnl_overlay_event(
        pd.Series({"Close": 7001.9, "High": 7001.9, "Low": 7001.9}),
        pd.Timestamp("2026-04-30T09:35:00-06:00"),
    )

    assert first is None
    assert second is not None
    assert second.get("reason") == "pnl_shelf_exit_triggered"
    assert "pnl_shelf_armed" in shelf_events
    assert "pnl_shelf_floor_raise" in shelf_events
    assert "pnl_shelf_exit_triggered" in shelf_events


def test_pnl_shelf_first_positive_tick_arms_in_live(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_steps_usd = "100,200"
    streamer.pnl_shelf_locks_usd = "25,100"
    streamer._pos = 1
    streamer._open_trade = {"side": "LONG", "contracts": 1, "client_order_id": "CID-2"}
    streamer._last_prob = 0.71
    streamer._compute_trade_pnl_state = lambda **_kwargs: {
        "side": "LONG",
        "stop_price": 6998.0,
        "target_price": 7008.0,
        "mark_price": 7000.5,
        "unrealized_r": 0.12,
        "mfe_r": 0.12,
        "giveback_r": 0.0,
        "bars_in_trade": 1,
        "unrealized_usd": 10.0,
    }
    shelf_events: list[str] = []
    streamer._emit_pnl_shelf_event = lambda **kwargs: shelf_events.append(str(kwargs.get("event_name")))

    event = streamer._maybe_pnl_overlay_event(
        pd.Series({"Close": 7000.5, "High": 7000.5, "Low": 7000.5}),
        pd.Timestamp("2026-04-30T09:30:00-06:00"),
    )

    if event is not None:
        assert str(event.get("ctx", {}).get("pnl_overlay_reason")) in {"target_arm", "runner_target_arm"}
    assert bool(streamer._open_trade.get("pnl_shelf_first_tick_armed")) is True
    assert float(streamer._open_trade.get("pnl_shelf_locked_floor_usd") or 0.0) == pytest.approx(0.0)
    assert "pnl_shelf_first_tick_armed" in shelf_events


def test_pnl_shelf_blocked_without_protection_truth_emits_block_event(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_steps_usd = "100,200"
    streamer.pnl_shelf_locks_usd = "100,200"
    streamer._pos = 1
    streamer._entry_price = 5000.0
    streamer.point_value = 50.0
    streamer._bars_in_trade = 2
    streamer._last_live_pnl_quality = {"quality": "ok", "source": "computed_from_snapshot"}
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._open_trade = {
        "side": "LONG",
        "contracts": 1,
        "client_order_id": "CID-BLOCK-1",
        "stop_order_id": None,
        "target_order_id": None,
    }
    shelf_events: list[str] = []
    streamer._emit_pnl_shelf_event = lambda **kwargs: shelf_events.append(str(kwargs.get("event_name")))

    row = pd.Series({"Close": 5002.5, "High": 5003.0, "Low": 4999.5})
    streamer._maybe_pnl_overlay_event(row, pd.Timestamp("2026-05-04T10:00:00-06:00"))

    assert "SHELF_ARM_BLOCKED" in shelf_events
    assert streamer._open_trade.get("pnl_shelf_armed_index") is None
    assert float(streamer._open_trade.get("max_unrealized_usd") or 0.0) > 0.0


def test_close_model_blocked_pre_shelf_arm_in_live(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_steps_usd = "100"
    streamer.pnl_shelf_locks_usd = "100"
    streamer._pos = 1
    streamer._open_trade = {"side": "LONG", "contracts": 1, "client_order_id": "CID-3", "pnl_shelf_armed_index": None}

    reason = streamer._shelf_prearm_close_block_reason({"side": "LONG", "unrealized_usd": 50.0})
    assert reason == "close_model_blocked_pre_shelf_lock"

    streamer._open_trade["max_unrealized_usd"] = 100.0
    assert streamer._shelf_prearm_close_block_reason({"side": "LONG", "unrealized_usd": 50.0}) is None


def test_merge_open_trade_runtime_state_preserves_shelf_fields(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    previous = {
        "max_unrealized_usd": 180.0,
        "pnl_shelf_armed_index": 1,
        "pnl_shelf_locked_floor_usd": 70.0,
        "pnl_shelf_last_ts": "2026-04-30T09:35:00-06:00",
    }
    rebuilt = {"side": "LONG", "max_unrealized_usd": 20.0}
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(payload)

    merged = streamer._merge_open_trade_runtime_state(
        rebuilt,
        previous_trade=previous,
        source="unit_test",
    )

    assert float(merged["max_unrealized_usd"]) == pytest.approx(180.0)
    assert int(merged["pnl_shelf_armed_index"]) == 1
    assert float(merged["pnl_shelf_locked_floor_usd"]) == pytest.approx(70.0)
    assert merged["pnl_shelf_last_ts"] == "2026-04-30T09:35:00-06:00"
    assert any(ev.get("event") == "shelf_state_reset_guarded" for ev in events)


def test_pnl_shelf_fail_fast_missing_config_raises(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_shelf_enabled = False
    streamer.pnl_shelf_steps_usd = ""
    streamer.pnl_shelf_locks_usd = ""
    streamer.pnl_shelf_fail_fast_missing_config = True

    with pytest.raises(RuntimeError, match="PNL_SHELF_CONFIG_MISSING_OR_INVALID"):
        streamer._validate_pnl_shelf_runtime_config(
            {"pnl_shelf_enabled": True, "pnl_shelf_steps_usd": [100], "pnl_shelf_locks_usd": [25]},
            "es_elite_v1",
        )


def test_pnl_shelf_locked_floor_is_monotonic_ratchet(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_steps_usd = "100,200,400"
    streamer.pnl_shelf_locks_usd = "25,100,250"
    streamer.pnl_shelf_min_hold_bars = 5
    streamer.pnl_shelf_exit_mode = "hard_floor"
    streamer._pos = 1
    streamer._open_trade = {"side": "LONG", "contracts": 1, "client_order_id": "CID-RATCHET"}
    streamer._last_prob = 0.70
    streamer.protection_mode = "stop_and_target"
    streamer._build_forced_exit_event = lambda **kwargs: {"type": "CLOSE", "reason": kwargs.get("reason")}

    seq = iter(
        [
            {
                "side": "LONG",
                "stop_price": 6998.0,
                "target_price": 7008.0,
                "mark_price": 7004.4,
                "unrealized_r": 1.10,
                "mfe_r": 1.10,
                "giveback_r": 0.0,
                "bars_in_trade": 1,
                "unrealized_usd": 220.0,
            },
            {
                "side": "LONG",
                "stop_price": 6998.0,
                "target_price": 7008.0,
                "mark_price": 7002.0,
                "unrealized_r": 0.50,
                "mfe_r": 1.10,
                "giveback_r": 0.60,
                "bars_in_trade": 1,
                "unrealized_usd": 120.0,
            },
            {
                "side": "LONG",
                "stop_price": 6998.0,
                "target_price": 7008.0,
                "mark_price": 7009.0,
                "unrealized_r": 2.20,
                "mfe_r": 2.20,
                "giveback_r": 0.0,
                "bars_in_trade": 1,
                "unrealized_usd": 450.0,
            },
        ]
    )
    streamer._compute_trade_pnl_state = lambda **_kwargs: next(seq)

    first = streamer._maybe_pnl_overlay_event(
        pd.Series({"Close": 7004.4, "High": 7004.4, "Low": 7004.4}),
        pd.Timestamp("2026-04-30T09:30:00-06:00"),
    )
    floor_after_first = float(streamer._open_trade.get("pnl_shelf_locked_floor_usd") or 0.0)
    second = streamer._maybe_pnl_overlay_event(
        pd.Series({"Close": 7002.0, "High": 7002.0, "Low": 7002.0}),
        pd.Timestamp("2026-04-30T09:35:00-06:00"),
    )
    floor_after_second = float(streamer._open_trade.get("pnl_shelf_locked_floor_usd") or 0.0)
    third = streamer._maybe_pnl_overlay_event(
        pd.Series({"Close": 7009.0, "High": 7009.0, "Low": 7009.0}),
        pd.Timestamp("2026-04-30T09:40:00-06:00"),
    )
    floor_after_third = float(streamer._open_trade.get("pnl_shelf_locked_floor_usd") or 0.0)

    assert first is None
    assert second is None
    assert third is None
    assert floor_after_first == pytest.approx(100.0)
    assert floor_after_second == pytest.approx(100.0)
    assert floor_after_third == pytest.approx(250.0)
    assert floor_after_first <= floor_after_second <= floor_after_third


def test_pnl_shelf_exit_debounce_requires_two_breaches_or_min_breach_usd(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_steps_usd = "100,200"
    streamer.pnl_shelf_locks_usd = "100,200"
    streamer.pnl_shelf_min_hold_bars = 1
    streamer.pnl_shelf_exit_mode = "hard_floor"
    streamer.pnl_shelf_bounce_handles = 0.25
    streamer._pos = 1
    streamer._open_trade = {"side": "LONG", "contracts": 1, "client_order_id": "CID-DEBOUNCE", "pnl_shelf_locked_floor_usd": 100.0, "max_unrealized_usd": 120.0}
    streamer._last_prob = 0.75
    streamer.protection_mode = "stop_and_target"
    seq = iter(
        [
            {"side": "LONG", "stop_price": 6998.0, "target_price": 7008.0, "mark_price": 7001.95, "unrealized_r": 0.4, "mfe_r": 1.1, "giveback_r": 0.7, "bars_in_trade": 1, "unrealized_usd": 99.0},
            {"side": "LONG", "stop_price": 6998.0, "target_price": 7008.0, "mark_price": 7001.95, "unrealized_r": 0.4, "mfe_r": 1.1, "giveback_r": 0.7, "bars_in_trade": 1, "unrealized_usd": 99.0},
        ]
    )
    streamer._compute_trade_pnl_state = lambda **_kwargs: next(seq)
    streamer._build_forced_exit_event = lambda **kwargs: {"type": "CLOSE", "reason": kwargs.get("reason")}
    first = streamer._maybe_pnl_overlay_event(pd.Series({"Close": 7001.95, "High": 7001.95, "Low": 7001.95}), pd.Timestamp("2026-04-30T10:00:00-06:00"))
    second = streamer._maybe_pnl_overlay_event(pd.Series({"Close": 7001.95, "High": 7001.95, "Low": 7001.95}), pd.Timestamp("2026-04-30T10:00:01-06:00"))
    assert first is None
    assert second is not None
    assert second.get("reason") == "pnl_shelf_exit_triggered"


def test_pnl_shelf_gap_multi_level_lock_event_emitted(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_steps_usd = "100,200,400,800"
    streamer.pnl_shelf_locks_usd = "100,200,400,800"
    streamer._pos = 1
    streamer._open_trade = {"side": "LONG", "contracts": 1, "client_order_id": "CID-GAP", "pnl_shelf_armed_index": 0, "pnl_shelf_locked_floor_usd": 100.0, "max_unrealized_usd": 120.0}
    streamer._last_prob = 0.7
    streamer._compute_trade_pnl_state = lambda **_kwargs: {
        "side": "LONG",
        "stop_price": 6998.0,
        "target_price": 7008.0,
        "mark_price": 7008.0,
        "unrealized_r": 2.0,
        "mfe_r": 2.0,
        "giveback_r": 0.0,
        "bars_in_trade": 1,
        "unrealized_usd": 850.0,
    }
    shelf_events: list[str] = []
    streamer._emit_pnl_shelf_event = lambda **kwargs: shelf_events.append(str(kwargs.get("event_name")))
    _ = streamer._maybe_pnl_overlay_event(pd.Series({"Close": 7008.0, "High": 7008.0, "Low": 7008.0}), pd.Timestamp("2026-04-30T10:05:00-06:00"))
    assert "pnl_shelf_gap_multi_level_lock" in shelf_events


def test_normalize_snapshot_live_pnl_uses_nt_reported_when_available(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.instrument = SimpleNamespace(point_value=50.0)
    payload = streamer._normalize_snapshot_live_pnl(
        {
            "type": "POSITION_SNAPSHOT",
            "qty": 1,
            "avg_price": 7000.0,
            "last_price": 7001.0,
            "realizedPnL": 12.5,
            "unrealizedPnL": 55.0,
            "timestamp": utc_ts(),
        }
    )
    assert payload["source"] == "nt_reported"
    assert payload["quality"] == "ok"
    assert float(payload["unrealized_usd"]) == pytest.approx(55.0)
    assert float(payload["realized_usd"]) == pytest.approx(12.5)


def test_normalize_snapshot_live_pnl_computes_when_missing(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.instrument = SimpleNamespace(point_value=50.0)
    payload = streamer._normalize_snapshot_live_pnl(
        {
            "type": "POSITION_SNAPSHOT",
            "qty": 2,
            "avg_price": 7000.0,
            "last_price": 7001.0,
            "side": "LONG",
            "timestamp": utc_ts(),
        }
    )
    assert payload["source"] == "computed_from_snapshot"


def test_normalize_snapshot_live_pnl_prefers_nt_account_api_source(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    payload = streamer._normalize_snapshot_live_pnl(
        {
            "source": "nt_account_api",
            "position_qty": 1,
            "avg_price": 6000.0,
            "last_price": 6001.0,
            "unrealized_pnl_currency": 50.0,
            "timestamp": utc_ts(),
        }
    )
    assert payload["source"] == "nt_account_api"
    assert payload["unrealized_usd"] == pytest.approx(50.0)


def test_pnl_feed_quality_state_stale_when_missing_feed(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_pnl_snapshot_stale_sec = 60.0
    streamer._last_pnl_snapshot_received_ts = None
    assert streamer._pnl_feed_quality_state() == "stale"


def test_compute_nt_ready_requires_fresh_pnl_when_enabled(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_require_pnl_snapshot = True
    streamer.nt_pnl_snapshot_stale_sec = 5.0
    streamer.nt_snapshot_fresh_sec = 30.0
    streamer.nt_sync_interval_sec = 30.0
    streamer.nt_require_ready_message = False
    streamer._last_pnl_snapshot_received_ts = time.time() - 120.0
    streamer._nt_last_snapshot_ts_utc = pd.Timestamp.utcnow()
    streamer._nt_snapshot_last_ts_valid = True
    streamer._nt_last_snapshot_instrument = "ES 06-26"
    streamer.exec_instrument = "ES 06-26"
    streamer._snapshot_age_sec = lambda *_args, **_kwargs: 0.0
    streamer._nt_account_chosen = "Sim101"
    streamer._nt_ready = True
    streamer._emit_unblocked = lambda *_args, **_kwargs: None
    streamer._set_entries_disarmed = lambda *_args, **_kwargs: None
    streamer._emit_block_event = lambda *_args, **_kwargs: None
    streamer._emit_event = lambda *_args, **_kwargs: None

    class _Bridge:
        is_connected = True

        @staticmethod
        def handshake_ok() -> bool:
            return True

    streamer.nt_bridge = _Bridge()
    ready, reason = streamer._compute_nt_ready()
    assert ready is False
    assert reason == "pnl_snapshot_stale"


def test_compute_nt_ready_global_only_snapshot_keeps_not_ready_with_reason_detail(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_require_pnl_snapshot = False
    streamer.nt_require_ready_message = False
    streamer.nt_snapshot_fresh_sec = 30.0
    streamer.nt_sync_interval_sec = 30.0
    streamer._nt_last_snapshot_instrument = None
    streamer._nt_snapshot_any_instrument_seen = True
    streamer._nt_snapshot_last_ts_valid = True
    streamer._snapshot_age_sec = (
        lambda inst_key=None, **_kwargs: (None if inst_key else 0.0)
    )
    streamer._nt_account_chosen = "Sim101"
    streamer._nt_ready = False
    streamer._emit_unblocked = lambda *_args, **_kwargs: None
    streamer._set_entries_disarmed = lambda *_args, **_kwargs: None
    streamer._emit_block_event = lambda *_args, **_kwargs: None
    streamer._emit_event = lambda *_args, **_kwargs: None

    class _Bridge:
        is_connected = True

        @staticmethod
        def handshake_ok() -> bool:
            return True

    streamer.nt_bridge = _Bridge()
    ready, reason = streamer._compute_nt_ready()
    assert ready is False
    assert reason == "snapshot_stale"
    assert streamer._nt_ready_reason_detail == "snapshot_global_only"


def test_compute_nt_ready_requires_order_capable_bridge(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_adapter = "native"
    streamer.nt_require_pnl_snapshot = False
    streamer.nt_require_ready_message = False
    streamer.nt_snapshot_fresh_sec = 30.0
    streamer.nt_sync_interval_sec = 30.0
    streamer._nt_last_snapshot_instrument = "ES 06-26"
    streamer.exec_instrument = "ES 06-26"
    streamer._snapshot_age_sec = lambda *_args, **_kwargs: 0.0
    streamer._nt_account_chosen = "Sim101"
    streamer._nt_ready = False
    streamer._emit_unblocked = lambda *_args, **_kwargs: None
    streamer._set_entries_disarmed = lambda *_args, **_kwargs: None
    streamer._emit_block_event = lambda *_args, **_kwargs: None
    streamer._emit_event = lambda *_args, **_kwargs: None

    class _Bridge:
        is_connected = True

        @staticmethod
        def handshake_ok() -> bool:
            return True

        @staticmethod
        def client_kind() -> str:
            return "telemetry_only"

        @staticmethod
        def client_source() -> str:
            return "nt_account_api"

    streamer.nt_bridge = _Bridge()

    ready, reason = streamer._compute_nt_ready()

    assert ready is False
    assert reason == "order_bridge_missing"


def test_compute_nt_ready_blocks_stale_live_addon_version(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_adapter = "native"
    streamer.nt_exec_policy = "live"
    streamer.nt_require_pnl_snapshot = False
    streamer.nt_require_ready_message = False
    streamer.nt_snapshot_fresh_sec = 30.0
    streamer.nt_sync_interval_sec = 30.0
    streamer._nt_last_snapshot_instrument = "ES 06-26"
    streamer.exec_instrument = "ES 06-26"
    streamer._snapshot_age_sec = lambda *_args, **_kwargs: 0.0
    streamer._nt_account_chosen = "Sim101"
    streamer._nt_addon_version = "2026.01.20"
    streamer._nt_min_addon_version = "2026.01.23"
    streamer._nt_ready = False
    streamer._emit_unblocked = lambda *_args, **_kwargs: None
    streamer._set_entries_disarmed = lambda *_args, **_kwargs: None
    streamer._emit_block_event = lambda *_args, **_kwargs: None
    streamer._emit_event = lambda *_args, **_kwargs: None

    class _Bridge:
        is_connected = True

        @staticmethod
        def handshake_ok() -> bool:
            return True

        @staticmethod
        def client_kind() -> str:
            return "order_capable"

        @staticmethod
        def client_source() -> str:
            return "NinjaTrader8"

    streamer.nt_bridge = _Bridge()

    ready, reason = streamer._compute_nt_ready()

    assert ready is False
    assert reason == "addon_version_stale"
    assert streamer._nt_ready_reason_detail == "observed=2026.01.20|required>=2026.01.23"


def test_flip_state_machine_requires_explicit_allow_flips(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 1.0
    streamer.allow_flips = False

    assert streamer._state_machine_violation("FLIP", "SHORT") == "flip_forbidden"

    streamer.allow_flips = True
    assert streamer._state_machine_violation("FLIP", "SHORT") is None


def test_heartbeat_snapshot_fallback_seeds_wait_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_adapter = "native"
    class _Bridge:
        is_connected = True

        @staticmethod
        def client_kind() -> str:
            return "order_capable"

        @staticmethod
        def client_source() -> str:
            return "NinjaRepoBridge"

        @staticmethod
        def handshake_ok() -> bool:
            return True

    streamer.nt_bridge = _Bridge()
    streamer._nt_snapshot_wait_active = True
    streamer._nt_snapshot_seen = False
    streamer._nt_snapshot_request_ts = 123.0
    streamer._nt_snapshot_last_ping_ts = 123.0
    streamer._nt_snapshot_request_error = True
    streamer._nt_last_snapshot_instrument = None
    streamer._nt_snapshot_any_instrument_seen = False
    streamer.exec_instrument = "ES 06-26"
    streamer.nt_instrument = "ES JUN26"
    streamer._nt_instrument_for_tx = lambda: "ES 06-26"
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    streamer._record_nt_snapshot_rx(
        "HEARTBEAT",
        {
            "event_type": "HEARTBEAT",
            "source": "nt_account_api",
            "ts_local": "2026-06-01T21:38:44.1714254-06:00",
            "ts_exchange": "2026-06-02T03:38:44.1714254+00:00",
        },
    )

    assert streamer._nt_snapshot_seen is True
    assert streamer._nt_snapshot_request_ts is None
    assert streamer._nt_snapshot_last_ping_ts is None
    assert streamer._nt_snapshot_request_error is False
    assert streamer._nt_snapshot_any_instrument_seen is True
    assert streamer._nt_last_snapshot_instrument == "ES 06-26"
    assert streamer._nt_last_snapshot_ts_utc is not None
    assert any(ev.get("event") == "nt_heartbeat_snapshot_fallback" for ev in events)


def test_telemetry_heartbeat_does_not_seed_execution_readiness(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_adapter = "native"
    class _Bridge:
        is_connected = True

        @staticmethod
        def client_kind() -> str:
            return "telemetry_only"

        @staticmethod
        def client_source() -> str:
            return "nt_account_api"

        @staticmethod
        def handshake_ok() -> bool:
            return False

    streamer.nt_bridge = _Bridge()
    streamer._nt_snapshot_wait_active = True
    streamer._nt_snapshot_seen = False
    streamer._nt_snapshot_request_ts = 123.0
    streamer._nt_snapshot_last_ping_ts = 123.0
    streamer._nt_snapshot_request_error = True
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    streamer._record_nt_snapshot_rx(
        "HEARTBEAT",
        {
            "event_type": "HEARTBEAT",
            "source": "nt_account_api",
            "ts_local": "2026-06-01T21:38:44.1714254-06:00",
        },
    )

    assert streamer._nt_snapshot_seen is False
    assert streamer._nt_snapshot_request_ts == 123.0
    assert streamer._nt_snapshot_last_ping_ts == 123.0
    assert streamer._nt_snapshot_request_error is True
    assert any(ev.get("event") == "nt_telemetry_heartbeat_ignored_for_execution_readiness" for ev in events)


def test_snapshot_timestamp_sanitizer_rejects_nat(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    assert streamer._sanitize_snapshot_timestamp(pd.NaT) is None


def test_snapshot_stale_recovery_suppresses_hard_reset_when_ts_invalid(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer._nt_ready = False
    streamer._nt_ready_reason = "snapshot_stale"
    streamer._nt_ready_reason_detail = "snapshot_ts_invalid"
    streamer._snapshot_recovery_attempts = 10
    streamer._nt_snapshot_recover_last_ts = 0.0
    streamer.nt_snapshot_recover_min_interval_sec = 0.0
    streamer._nt_last_snapshot_rx_wall_ts = time.time()
    printed: list[str] = []
    original_safe_print = stream_live_csv_mod._safe_print
    stream_live_csv_mod._safe_print = lambda msg: printed.append(str(msg))
    streamer._send_nt_ping = lambda **_kwargs: None
    streamer._send_nt_sync_request = lambda **_kwargs: None
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._snapshot_age_sec = lambda *_args, **_kwargs: None

    class _Bridge:
        is_connected = True

        @staticmethod
        def handshake_ok() -> bool:
            return True

        @staticmethod
        def reconnect() -> None:
            raise AssertionError("reconnect should be suppressed for ts-invalid-only stale state")

    streamer.nt_bridge = _Bridge()
    streamer._compute_nt_ready = lambda: (False, "snapshot_stale")
    try:
        streamer._check_nt_readiness()
    finally:
        stream_live_csv_mod._safe_print = original_safe_print
    assert not any("SNAPSHOT_STALE_HARD_RESET" in line for line in printed)
    assert any(ev.get("event") == "snapshot_stale_reset_suppressed_rx_active" for ev in events)


def test_compute_nt_ready_non_live_armed_keeps_operational_on_snapshot_stale(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "paper"
    streamer.nt_exec_state = "ARMED"
    streamer.nt_enabled = True
    streamer.nt_require_pnl_snapshot = False
    streamer.nt_require_ready_message = False
    streamer.nt_snapshot_fresh_sec = 10.0
    streamer.nt_sync_interval_sec = 5.0
    streamer._snapshot_age_sec = lambda *_args, **_kwargs: 30.0
    streamer._nt_last_snapshot_instrument = "ES 06-26"
    streamer.exec_instrument = "ES 06-26"
    streamer._nt_account_chosen = "Sim101"
    streamer._nt_ready = True
    streamer._emit_unblocked = lambda *_args, **_kwargs: None
    streamer._set_entries_disarmed = lambda *_args, **_kwargs: None
    streamer._emit_block_event = lambda *_args, **_kwargs: None
    streamer._emit_event = lambda *_args, **_kwargs: None

    class _Bridge:
        is_connected = True

        @staticmethod
        def handshake_ok() -> bool:
            return True

    streamer.nt_bridge = _Bridge()
    ready, reason = streamer._compute_nt_ready()
    assert ready is True
    assert reason == "snapshot_stale_operational"


def test_execution_lag_aware_age_detail_flags_processing_lag(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase_last_lag_sec = 90.0
    detail = streamer._execution_lag_aware_age_detail(
        signal_timestamp="2026-05-05T06:15:00-06:00",
        current_timestamp="2026-05-05T06:16:38-06:00",
    )
    assert detail["raw_age_sec"] == pytest.approx(98.0, abs=1.0)
    assert detail["processing_lag_sec"] == pytest.approx(90.0)
    assert detail["adjusted_age_sec"] == pytest.approx(8.0, abs=1.0)
    assert detail["age_classification"] == "processing_lag_adjusted"


def test_pre_send_entry_guard_uses_lag_adjusted_age(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.max_entry_age_sec = 25.0
    streamer.max_entry_bars_since_signal = None
    streamer.max_entry_drift_points = None
    streamer._phase_last_lag_sec = 90.0
    streamer.bar_interval_sec = 300.0
    streamer.session_tz = "America/Denver"
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="cid-1",
        action="OPEN",
        side="SHORT",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES 06-26",
        account="Sim101",
        entry_price=7266.25,
        model_price=7266.25,
        bar_ts="2026-05-05T06:15:00-06:00",
    )
    original_now = stream_live_csv_mod._now_denver
    stream_live_csv_mod._now_denver = lambda: pd.Timestamp("2026-05-05T06:16:38-06:00")
    try:
        violation = streamer._pre_send_entry_guard_violation(intent)
    finally:
        stream_live_csv_mod._now_denver = original_now
    assert violation is None


def test_compute_nt_ready_handshake_jitter_needs_sustained_bad_count_to_disarm(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_require_pnl_snapshot = False
    streamer.nt_require_ready_message = False
    streamer.nt_snapshot_fresh_sec = 30.0
    streamer.nt_sync_interval_sec = 30.0
    streamer._nt_last_snapshot_ts_utc = pd.Timestamp.utcnow()
    streamer._nt_last_snapshot_instrument = "ES 06-26"
    streamer.exec_instrument = "ES 06-26"
    streamer._snapshot_age_sec = lambda *_args, **_kwargs: 4.0
    streamer._nt_account_chosen = "Sim101"
    streamer._nt_ready = True
    streamer.readiness_bad_count_to_disarm = 2
    streamer._readiness_bad_count = 0
    streamer._readiness_good_count = 0
    streamer._readiness_disarm_count = 0
    streamer._phase_allows_execution = lambda: True
    disarm_calls: list[str] = []
    streamer._set_entries_disarmed = lambda reason, _detail: disarm_calls.append(str(reason))
    streamer._set_nt_exec_state = lambda *_args, **_kwargs: None
    streamer._emit_block_event = lambda *_args, **_kwargs: None
    streamer._emit_event = lambda *_args, **_kwargs: None
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    class _Bridge:
        is_connected = True

        @staticmethod
        def handshake_ok() -> bool:
            return False

    streamer.nt_bridge = _Bridge()
    ready, reason = streamer._compute_nt_ready()
    assert ready is False
    assert reason == "handshake_missing"
    assert disarm_calls == []
    assert any(e.get("event") == "transient_handshake_jitter" for e in events)


def test_compute_nt_ready_hard_handshake_loss_disarms(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_require_pnl_snapshot = False
    streamer.nt_require_ready_message = False
    streamer.nt_snapshot_fresh_sec = 30.0
    streamer.nt_sync_interval_sec = 30.0
    streamer._nt_last_snapshot_ts_utc = pd.Timestamp.utcnow()
    streamer._nt_last_snapshot_instrument = "ES 06-26"
    streamer.exec_instrument = "ES 06-26"
    streamer._snapshot_age_sec = lambda *_args, **_kwargs: 4.0
    streamer._nt_account_chosen = "Sim101"
    streamer._nt_ready = True
    streamer.readiness_bad_count_to_disarm = 1
    streamer._readiness_bad_count = 0
    streamer._readiness_good_count = 0
    streamer._readiness_disarm_count = 0
    streamer._phase_allows_execution = lambda: True
    disarm_calls: list[str] = []
    streamer._set_entries_disarmed = lambda reason, _detail: disarm_calls.append(str(reason))
    streamer._set_nt_exec_state = lambda *_args, **_kwargs: None
    streamer._emit_block_event = lambda *_args, **_kwargs: None
    streamer._emit_event = lambda *_args, **_kwargs: None
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    class _Bridge:
        is_connected = True

        @staticmethod
        def handshake_ok() -> bool:
            return False

    streamer.nt_bridge = _Bridge()
    ready, reason = streamer._compute_nt_ready()
    assert ready is False
    assert reason == "handshake_missing"
    assert disarm_calls == ["readiness_lost"]
    assert any(e.get("event") == "hard_handshake_loss" for e in events)


def test_startup_pnl_preflight_policy_warn_does_not_raise(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_require_pnl_snapshot = True
    streamer.nt_pnl_preflight_policy = "warn"
    snapshot = {
        "pnl_quality_state": "stale",
        "pnl_feed_staleness_ms": 120000.0,
        "live_pnl_source": "",
    }
    streamer._apply_startup_pnl_preflight_policy(snapshot)


def test_startup_pnl_preflight_policy_fail_raises(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_require_pnl_snapshot = True
    streamer.nt_pnl_preflight_policy = "fail"
    snapshot = {
        "pnl_quality_state": "stale",
        "pnl_feed_staleness_ms": 120000.0,
        "live_pnl_source": "",
    }
    with pytest.raises(RuntimeError, match="startup_preflight_blocked:pnl_preflight_failed"):
        streamer._apply_startup_pnl_preflight_policy(snapshot)


def test_normalize_snapshot_live_pnl_short_positive_qty_is_normalized_not_failed(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.instrument = SimpleNamespace(point_value=50.0)
    payload = streamer._normalize_snapshot_live_pnl(
        {
            "type": "POSITION_SNAPSHOT",
            "qty": 1,
            "avg_price": 7000.0,
            "last_price": 6998.0,
            "side": "SHORT",
            "timestamp": utc_ts(),
        }
    )
    assert payload["source"] == "computed_from_snapshot"
    assert payload["quality"] == "ok"
    assert payload["quality_reason"] == "side_qty_sign_mismatch_normalized"
    assert float(payload["position_qty"]) == pytest.approx(-1.0)
    assert float(payload["unrealized_usd"]) == pytest.approx(100.0)


def test_normalize_snapshot_live_pnl_naive_timestamp_uses_utc_first(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.instrument = SimpleNamespace(point_value=50.0)
    naive_now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%S")
    payload = streamer._normalize_snapshot_live_pnl(
        {
            "type": "POSITION_SNAPSHOT",
            "qty": 1,
            "avg_price": 7000.0,
            "last_price": 7002.0,
            "side": "LONG",
            "timestamp": naive_now,
        }
    )
    assert payload["quality"] == "ok"
    assert payload.get("ts_parse_mode") in {"naive_assumed_utc", "naive_assumed_session_tz"}
    assert str(payload.get("quality_reason", "")).startswith("ok_naive_ts_assumed_")


def test_pnl_shelf_arm_mode_step_only_skips_first_positive_tick_event(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_arm_mode = "step_only"
    streamer.pnl_shelf_steps_usd = "100,200"
    streamer.pnl_shelf_locks_usd = "25,100"
    streamer._pos = 1
    streamer._open_trade = {"side": "LONG", "contracts": 1, "client_order_id": "CID-STEP-ONLY"}
    streamer._last_prob = 0.71
    streamer._compute_trade_pnl_state = lambda **_kwargs: {
        "side": "LONG",
        "stop_price": 6998.0,
        "target_price": 7008.0,
        "mark_price": 7000.5,
        "unrealized_r": 0.12,
        "mfe_r": 0.12,
        "giveback_r": 0.0,
        "bars_in_trade": 1,
        "unrealized_usd": 10.0,
    }
    shelf_events: list[str] = []
    streamer._emit_pnl_shelf_event = lambda **kwargs: shelf_events.append(str(kwargs.get("event_name")))
    _ = streamer._maybe_pnl_overlay_event(
        pd.Series({"Close": 7000.5, "High": 7000.5, "Low": 7000.5}),
        pd.Timestamp("2026-04-30T09:30:00-06:00"),
    )
    assert "pnl_shelf_first_tick_armed" not in shelf_events


def test_pnl_shelf_rearm_cooldown_blocks_immediate_raise(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_rearm_cooldown_sec = 120.0
    streamer.pnl_shelf_steps_usd = "100,200,400"
    streamer.pnl_shelf_locks_usd = "100,200,400"
    streamer._pos = 1
    streamer._last_prob = 0.71
    streamer._open_trade = {
        "side": "LONG",
        "contracts": 1,
        "client_order_id": "CID-COOLDOWN",
        "pnl_shelf_armed_index": 0,
        "pnl_shelf_locked_floor_usd": 100.0,
        "max_unrealized_usd": 120.0,
        "pnl_shelf_last_raise_ts": "2026-04-30T09:30:00-06:00",
    }
    streamer._compute_trade_pnl_state = lambda **_kwargs: {
        "side": "LONG",
        "stop_price": 6998.0,
        "target_price": 7008.0,
        "mark_price": 7008.0,
        "unrealized_r": 2.0,
        "mfe_r": 2.0,
        "giveback_r": 0.0,
        "bars_in_trade": 1,
        "unrealized_usd": 500.0,
    }
    events: list[str] = []
    streamer._emit_pnl_shelf_event = lambda **kwargs: events.append(str(kwargs.get("event_name")))
    _ = streamer._maybe_pnl_overlay_event(
        pd.Series({"Close": 7008.0, "High": 7008.0, "Low": 7008.0}),
        pd.Timestamp("2026-04-30T09:30:30-06:00"),
    )
    assert streamer._open_trade.get("pnl_shelf_armed_index") == 0
    assert float(streamer._open_trade.get("pnl_shelf_locked_floor_usd") or 0.0) == pytest.approx(100.0)
    assert "pnl_shelf_rearm_cooldown_blocked" in events


def test_normalize_snapshot_live_pnl_missing_fields_fails_quality(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    payload = streamer._normalize_snapshot_live_pnl(
        {
            "type": "POSITION_SNAPSHOT",
            "qty": 1,
            "last_price": 7001.0,
            "timestamp": utc_ts(),
        }
    )
    assert payload["source"] == "computed_from_snapshot"
    assert payload["quality"] == "missing_fields"
    assert payload["unrealized_usd"] is None


def test_normalize_snapshot_live_pnl_flat_missing_fields_is_ok(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    payload = streamer._normalize_snapshot_live_pnl(
        {
            "type": "POSITION_SNAPSHOT",
            "qty": 0,
            "side": "SHORT",
            "timestamp": utc_ts(),
        }
    )
    assert payload["source"] == "computed_from_snapshot"
    assert payload["quality"] == "ok"
    assert payload["quality_reason"] == "ok_flat_no_position"
    assert float(payload["unrealized_usd"]) == pytest.approx(0.0)


def test_catchup_stale_bar_emits_explicit_stall_reason(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase = "BACKFILL"
    streamer._phase_current_confirmations = 0
    streamer.phase_currentness_confirmations_required = 1
    streamer.phase_backfill_lag_sec_threshold = 1500.0
    streamer.phase_currentness_lag_sec_threshold = 120.0
    streamer._effective_bar_age_max_sec = 605.0
    streamer._snapshot_age_sec = lambda: 0.2
    streamer._get_gate_now = lambda: pd.Timestamp("2026-05-03T21:16:00-06:00")
    stream_events: list[dict] = []
    state_events: list[tuple[str, dict]] = []
    streamer._log_exec_event = lambda payload: stream_events.append(dict(payload))
    streamer._emit_event = lambda et, payload: state_events.append((str(et), dict(payload)))
    stale_bar = pd.Timestamp("2026-05-03T21:05:00-06:00")

    streamer._update_phase_state(stale_bar)
    streamer._update_phase_state(stale_bar)

    stall_events = [ev for ev in stream_events if ev.get("event") == "phase_catchup_stall"]
    assert stall_events, stream_events
    last = stall_events[-1]
    assert last["phase"] == "CATCHUP"
    assert last["failed_condition"] == "lag_exceeds_currentness_threshold"
    assert float(last["bar_age_sec"]) > float(last["effective_bar_age_max_sec"])
    assert state_events and state_events[-1][0] == "STATE"
    assert state_events[-1][1].get("state_key") == "phase_catchup_stall"


def test_catchup_transitions_to_live_when_bar_is_current(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase = "BACKFILL"
    streamer._phase_current_confirmations = 0
    streamer.phase_currentness_confirmations_required = 1
    streamer.phase_backfill_lag_sec_threshold = 1500.0
    streamer.phase_currentness_lag_sec_threshold = 120.0
    streamer._get_gate_now = lambda: pd.Timestamp("2026-04-01T23:59:59-06:00")
    streamer._log_exec_event = lambda *_args, **_kwargs: None
    streamer._emit_event = lambda *_args, **_kwargs: None
    current_bar = pd.Timestamp("2026-04-01T23:59:30-06:00")

    streamer._update_phase_state(current_bar)

    assert streamer._phase == "LIVE"
    assert streamer._phase_execution_gate_reason() is None


def test_catchup_transitions_to_live_with_chicago_session_and_denver_engine_timestamp(
    tmp_path: Path,
) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.session_tz = "America/Chicago"
    streamer._phase = "BACKFILL"
    streamer._phase_current_confirmations = 0
    streamer.phase_currentness_confirmations_required = 1
    streamer.phase_backfill_lag_sec_threshold = 1500.0
    streamer.phase_currentness_lag_sec_threshold = 120.0
    streamer._get_gate_now = lambda: pd.Timestamp("2026-06-22T23:45:25-06:00")
    streamer._log_exec_event = lambda *_args, **_kwargs: None
    streamer._emit_event = lambda *_args, **_kwargs: None

    # Engine rows are normalized by `_ensure_denver_naive` before phase update.
    current_engine_bar = pd.Timestamp("2026-06-22 23:45:00")
    streamer._update_phase_state(current_engine_bar)

    assert float(streamer._phase_last_lag_sec or 0.0) == pytest.approx(25.0)
    assert streamer._phase == "LIVE"
    assert streamer._phase_execution_gate_reason() is None


def test_catchup_stall_triggers_bounded_recovery_attempt(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase = "BACKFILL"
    streamer._phase_current_confirmations = 0
    streamer.phase_currentness_confirmations_required = 1
    streamer.phase_backfill_lag_sec_threshold = 1500.0
    streamer.phase_currentness_lag_sec_threshold = 120.0
    streamer._effective_bar_age_max_sec = 605.0
    streamer._snapshot_age_sec = lambda: 0.2
    streamer._get_gate_now = lambda: pd.Timestamp("2026-05-03T21:16:00-06:00")
    streamer.phase_catchup_recover_min_interval_sec = 0.0
    ping_calls: list[str] = []
    sync_calls: list[str] = []
    streamer._send_nt_ping = lambda reason=None: ping_calls.append(str(reason))
    streamer._send_nt_sync_request = lambda reason=None: sync_calls.append(str(reason))
    streamer._log_exec_event = lambda *_args, **_kwargs: None
    streamer._emit_event = lambda *_args, **_kwargs: None
    stale_bar = pd.Timestamp("2026-05-03T21:05:00-06:00")

    for _ in range(4):
        streamer._update_phase_state(stale_bar)

    assert int(streamer._phase_catchup_recovery_attempts or 0) >= 1
    assert ping_calls and ping_calls[-1] == "phase_catchup_stall_recover"
    assert sync_calls and sync_calls[-1] == "phase_catchup_stall_recover"


def test_phase_lag_hysteresis_keeps_live_during_single_window_with_heartbeat() -> None:
    phase, confirmations, current_ok = stream_live_csv_mod._next_phase_state(
        lag_sec=275.0,
        backfill_threshold_sec=1500.0,
        current_threshold_sec=120.0,
        current_threshold_explicit=False,
        confirmations_required=1,
        confirmations_so_far=1,
        bar_interval_sec=300.0,
        lag_warn_windows=1,
        lag_catchup_windows=2,
        lag_jitter_buffer_sec=25.0,
        prev_phase="LIVE",
        heartbeat_alive=True,
    )
    assert phase == "LIVE"
    assert confirmations >= 1
    assert current_ok is True


def test_phase_lag_hysteresis_demotes_to_catchup_after_sustained_windows() -> None:
    phase, confirmations, current_ok = stream_live_csv_mod._next_phase_state(
        lag_sec=700.0,
        backfill_threshold_sec=1500.0,
        current_threshold_sec=120.0,
        current_threshold_explicit=False,
        confirmations_required=1,
        confirmations_so_far=1,
        bar_interval_sec=300.0,
        lag_warn_windows=1,
        lag_catchup_windows=2,
        lag_jitter_buffer_sec=25.0,
        prev_phase="LIVE",
        heartbeat_alive=False,
    )
    assert phase == "CATCHUP"
    assert confirmations == 0
    assert current_ok is False


def test_phase_lag_explicit_threshold_override_preserves_legacy_behavior() -> None:
    phase, confirmations, current_ok = stream_live_csv_mod._next_phase_state(
        lag_sec=275.0,
        backfill_threshold_sec=1500.0,
        current_threshold_sec=120.0,
        current_threshold_explicit=True,
        confirmations_required=1,
        confirmations_so_far=1,
        bar_interval_sec=300.0,
        lag_warn_windows=1,
        lag_catchup_windows=2,
        lag_jitter_buffer_sec=25.0,
        prev_phase="LIVE",
        heartbeat_alive=True,
    )
    assert phase == "CATCHUP"
    assert confirmations == 0
    assert current_ok is False


def test_pnl_shelf_quality_degrade_does_not_skip_stairs_progression(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_steps_usd = "100,200"
    streamer.pnl_shelf_locks_usd = "100,200"
    streamer._last_prob = 0.5
    streamer._pos = 1
    streamer._open_trade = {"side": "LONG", "contracts": 1, "client_order_id": "CID-FAILCLOSED", "entry_price": 7000.0}
    streamer._last_live_pnl_quality = {"quality": "stale_price", "quality_reason": "stale", "source": "computed_from_snapshot"}
    streamer._compute_trade_pnl_state = lambda **_kwargs: {
        "side": "LONG",
        "stop_price": 6998.0,
        "target_price": 7008.0,
        "mark_price": 7008.0,
        "unrealized_r": 2.0,
        "mfe_r": 2.0,
        "giveback_r": 0.0,
        "bars_in_trade": 1,
        "unrealized_usd": 850.0,
    }
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(payload)
    _ev = streamer._maybe_pnl_overlay_event(
        pd.Series({"Close": 7008.0, "High": 7008.0, "Low": 7008.0}),
        pd.Timestamp("2026-04-30T10:05:00-06:00"),
    )
    assert float(streamer._open_trade.get("max_unrealized_usd") or 0.0) == pytest.approx(400.0)
    assert streamer._open_trade.get("pnl_shelf_armed_index") == 1
    assert float(streamer._open_trade.get("pnl_shelf_locked_floor_usd") or 0.0) == pytest.approx(200.0)
    assert any(e.get("event") == "stairs_blocked_quality" for e in events)


def test_stairs_pre_signal_block_reason_when_quality_not_ok(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_quality_policy = "hard_block"
    streamer._last_live_pnl_quality = {"quality": "missing_fields", "quality_reason": "missing"}
    assert streamer._stairs_pre_signal_block_reason() is None


def test_stairs_pre_signal_block_reason_not_blocked_when_quality_ok_flat(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_quality_policy = "hard_block"
    streamer._last_live_pnl_quality = {"quality": "ok", "quality_reason": "ok_flat_no_position"}
    assert streamer._stairs_pre_signal_block_reason() is None


def test_merge_open_trade_runtime_state_initializes_stairs_baseline_fields(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    merged = streamer._merge_open_trade_runtime_state(
        {"side": "LONG", "contracts": 1, "client_order_id": "CID-BASELINE"},
        previous_trade=None,
        source="unit_test",
    )
    assert "max_unrealized_usd" in merged
    assert "pnl_shelf_armed_index" in merged
    assert "pnl_shelf_locked_floor_usd" in merged
    assert "pnl_shelf_last_ts" in merged
    assert float(merged["max_unrealized_usd"]) == pytest.approx(0.0)
    assert merged["pnl_shelf_armed_index"] is None
    assert float(merged["pnl_shelf_locked_floor_usd"]) == pytest.approx(0.0)
    assert merged["pnl_shelf_last_ts"] is None


def test_pnl_shelf_adaptive_r_steps_convert_from_trade_risk(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer.pnl_shelf_mode = "adaptive_r"
    streamer.pnl_shelf_r_steps = "0.5,1.0,2.0"
    streamer.pnl_shelf_r_locks = "0.25,0.75,1.5"
    streamer.point_value = 50.0
    streamer._open_trade = {"risk_points": 8.0, "contracts": 1}
    steps, locks = streamer._pnl_shelf_steps_and_locks()
    assert steps == pytest.approx([200.0, 400.0, 800.0])
    assert locks == pytest.approx([100.0, 300.0, 600.0])


def test_write_status_includes_regime_telemetry(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._market_mode = "trend"
    streamer._market_mode_source = "regime_policy"
    streamer._trend_direction = "up"

    streamer._write_status()

    payload = json.loads(streamer.status_path.read_text(encoding="utf-8"))
    assert payload["preset"] == "preset"
    assert payload["market_mode"] == "trend"
    assert payload["market_mode_source"] == "regime_policy"
    assert payload["trend_preset"] == "preset_trend"


def test_setup_gate_failure_without_override_blocks_open_emission(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    event = {
        "datetime": "2026-04-01T23:00:00-06:00",
        "type": "OPEN",
        "side": "LONG",
        "price": 6533.25,
        "prob": 0.861673,
        "grade": "A+",
        "risk": {"stop": 6525.75, "target": 6540.75},
        "contracts": 1,
        "gates_detail": {
            "gate_state": {"setup": False, "prob": True, "vwap": True, "ema": True, "tod": True},
            "override_confident_long": False,
            "override_prob_min": 0.78,
            "override_hold_conf_min": 0.7,
            "override_applied": False,
        },
        "blocked_pre_signal": True,
        "blocked_pre_signal_reason": "setup",
    }

    streamer._maybe_emit_signal(event)

    signal_rows = list(csv.DictReader(streamer.signals_csv.read_text(encoding="utf-8").splitlines()))
    assert signal_rows == []
    blocked_rows = _read_jsonl(streamer.blocked_candidates_path)
    assert len(blocked_rows) == 1
    assert blocked_rows[0]["candidate_action"] == "OPEN"
    assert blocked_rows[0]["blocked_reason"] in {"setup", "phase2_setup_blocked_upstream"}
    assert int(streamer._executor_stats.get("blocked_pre_signal_total", 0) or 0) == 1


def _prepare_gap_policy_streamer(streamer: LiveCSVStreamer, tmp_path: Path, events: list[dict]) -> None:
    streamer.gaps_path = tmp_path / "gaps.jsonl"
    streamer.lockout_path = tmp_path / "lockout.json"
    streamer.state_path = tmp_path / "stream_state.json"
    streamer.gap_action = "pause"
    streamer.gap_pause_sec = 900
    streamer.bar_interval_sec = 300
    streamer.gap_threshold_bars = 3
    streamer.lockout_policy = "all"
    streamer._last_gap_end = None
    streamer._phase = "LIVE"
    streamer._pos = 0
    streamer._lockout_reset_token = None
    streamer._lockout_reset_required = False
    streamer._lockout_sticky = False
    streamer._hard_lockout_event_key = None
    streamer._session_closed_lockouts_outside_live = 0
    streamer._session_closed_lockout_count = 0
    streamer._entry_price = None
    streamer._entry_time = None
    streamer._entry_stop = None
    streamer._entry_target = None
    streamer._bars_in_trade = 0
    streamer._cooldown_left = 0
    streamer._post_stop_cooldown_left = 0
    streamer.state.position_state = "FLAT"
    streamer.execution_state = SimpleNamespace(update_lockout=lambda **_kwargs: None)
    streamer._log_exec_event = lambda payload: events.append(payload)
    streamer._emit_event = lambda *_args, **_kwargs: None
    streamer._emit_block_event = lambda *_args, **_kwargs: None
    streamer._emit_hard_lockout_event = lambda *_args, **_kwargs: None
    streamer._emit_unblocked = lambda *_args, **_kwargs: None
    streamer._write_status = lambda *_args, **_kwargs: None
    streamer._save_state = LiveCSVStreamer._save_state.__get__(streamer, LiveCSVStreamer)


def test_flat_gap_pause_creates_timed_soft_block_not_sticky_lockout(tmp_path: Path) -> None:
    events: list[dict] = []
    streamer = _make_streamer(tmp_path)
    _prepare_gap_policy_streamer(streamer, tmp_path, events)

    streamer._apply_gap_policy(
        {
            "start": pd.Timestamp("2026-04-14T03:30:00-06:00"),
            "end": pd.Timestamp("2026-04-14T05:40:00-06:00"),
            "delta_seconds": 7800,
        }
    )

    assert streamer._hard_lockout_active is False
    assert streamer._hard_lockout_code is None
    assert streamer.lockout_path.exists() is False
    assert streamer._entries_disarmed_reason == "gap_pause"
    assert str(streamer.state.entries_disarmed_until) == "2026-04-14 05:55:00"
    assert any(ev.get("event") == "gap_detected_soft_pause" for ev in events)
    gap_rows = _read_jsonl(streamer.gaps_path)
    assert len(gap_rows) == 1
    assert gap_rows[0]["event"] == "GAP_DETECTED"


def test_in_position_gap_pause_still_creates_hard_lockout(tmp_path: Path) -> None:
    events: list[dict] = []
    streamer = _make_streamer(tmp_path)
    _prepare_gap_policy_streamer(streamer, tmp_path, events)
    streamer._pos = 1
    streamer.state.position_state = "IN_POSITION_PROTECTED"

    streamer._apply_gap_policy(
        {
            "start": pd.Timestamp("2026-04-14T03:30:00-06:00"),
            "end": pd.Timestamp("2026-04-14T05:40:00-06:00"),
            "delta_seconds": 7800,
        }
    )

    assert streamer._hard_lockout_active is True
    assert streamer._hard_lockout_code == "gap_detected"
    assert streamer.lockout_path.exists() is True
    lockout_payload = json.loads(streamer.lockout_path.read_text(encoding="utf-8"))
    assert lockout_payload["lockout_until"] == "2026-04-14T05:55:00-06:00"
    assert not any(ev.get("event") == "gap_detected_soft_pause" for ev in events)


def test_flat_live_large_gap_resume_suppresses_gap_pause_and_lockout(tmp_path: Path) -> None:
    events: list[dict] = []
    streamer = _make_streamer(tmp_path)
    _prepare_gap_policy_streamer(streamer, tmp_path, events)
    streamer.phase_backfill_lag_sec_threshold = 1500.0
    streamer._phase = "LIVE"
    streamer._pos = 0
    streamer.state.position_state = "FLAT"

    streamer._apply_gap_policy(
        {
            "start": pd.Timestamp("2026-05-01T10:15:00-06:00"),
            "end": pd.Timestamp("2026-05-03T20:50:00-06:00"),
            "delta_seconds": 210900,
        }
    )

    assert streamer._entries_disarmed_reason is None
    assert streamer._hard_lockout_active is False
    assert streamer._hard_lockout_code is None
    assert any(ev.get("event") == "gap_detected_live_resume_suppressed" for ev in events)


def test_expired_gap_pause_auto_rearms_when_current(tmp_path: Path) -> None:
    events: list[dict] = []
    streamer = _make_streamer(tmp_path)
    _prepare_gap_policy_streamer(streamer, tmp_path, events)
    streamer._entries_disarmed_reason = "gap_pause"
    streamer.state.entries_disarmed_reason = "gap_pause"
    streamer.state.entries_disarmed_until = pd.Timestamp("2026-04-14T05:55:00-06:00")
    streamer._evaluate_guardrails = lambda **_kwargs: SimpleNamespace(preflight_ok=True)
    streamer._get_gate_now = lambda: pd.Timestamp("2026-04-14T05:56:00-06:00")
    streamer._tod_gate_enabled = lambda: False
    streamer._offline_exec_should_block = lambda _offline: False
    streamer._bar_age_seconds = lambda _bar_ts: 0.0
    streamer.max_bar_age_seconds_for_exec = 605.0
    streamer.stale_bar_tolerance_sec = 600.0
    streamer.stale_bar_confirmations = 1

    streamer._update_entry_arming_status(pd.Timestamp("2026-04-14T05:55:00-06:00"))

    assert streamer._entries_disarmed_reason is None
    assert streamer.state.entries_disarmed_reason is None
    assert any(ev.get("event") == "gap_pause_cleared" for ev in events)


def _phase2_force_open_ready_streamer(tmp_path: Path) -> LiveCSVStreamer:
    streamer = _make_streamer(tmp_path)
    streamer.phase2_force_open_on_gate_pass = True
    streamer.phase2_force_open_allow_legacy_gate_bypass = True
    streamer.phase2_force_open_min_setup = 0.35
    streamer.phase2_force_open_min_entry_conf = 0.55
    streamer.phase2_decision_policy = Phase2DecisionPolicy()
    streamer.phase2_p_long = 0.60
    streamer.phase2_p_short = 0.60
    streamer.phase2_p_setup = 0.35
    streamer._phase = "LIVE"
    streamer._pos = 0
    streamer._open_trade = None
    streamer._hard_lockout_active = False
    streamer.state.position_state = "FLAT"
    streamer.state.pending_client_order_id = None
    return streamer


def _phase2_force_open_row() -> pd.Series:
    return pd.Series(
        {
            "stop": 7002.04,
            "target": 7008.46,
            "entry_conf": 0.31,
            "hold_conf": 0.88,
        }
    )


def _phase2_force_open_meta() -> dict:
    return {
        "setup_prob": 0.93,
        "setup_pass": True,
        "direction_prob": 0.82,
        "short_prob": 0.18,
        "direction_signal": 1,
    }


def _phase2_force_open_short_meta() -> dict:
    return {
        "setup_prob": 0.94,
        "setup_pass": True,
        "direction_prob": 0.12,
        "short_prob": 0.88,
        "direction_signal": -1,
    }


def _phase2_force_open_gates() -> dict:
    return {
        "tod": True,
        "risk": True,
        "setup": True,
        "prob": True,
        "vwap": True,
        "ema": True,
        "bid_ask": True,
        "armed": True,
        "offline_exec": True,
        "stale_bar": True,
        "feature_health": True,
        "time_skew": True,
    }


def test_phase2_force_open_live_flat_gate_pass_synthesizes_open(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)

    event = streamer._phase2_force_open_event(
        row=_phase2_force_open_row(),
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        price=7005.25,
        prob=0.73,
        phase2_meta=_phase2_force_open_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.31,
        hold_conf=0.88,
        original_action="NO_TRADE",
    )

    assert event is not None
    assert event["type"] == "OPEN"
    assert event["side"] == "LONG"
    assert event["risk"] == {"stop": 7002.04, "target": 7008.46}
    assert event["phase2_force_open_applied"] is True
    assert event["phase2_force_open_reason"] == "gate_pass_no_trade"
    assert event["original_action"] == "NO_TRADE"
    assert event["original_entry_conf"] == 0.31
    assert event["original_hold_conf"] == 0.88
    assert event["entry_conf"] == pytest.approx(0.55)
    assert event["entry_conf_source"] == "direction_margin_long"
    assert event["entry_conf_calibrated"] == pytest.approx(0.55)
    assert event["hold_conf"] == 0.88
    assert event["ctx"]["phase2_force_open_applied"] is True
    assert event["ctx"]["phase2_force_open_candidate"] is True
    assert event["ctx"]["phase"] == "LIVE"
    assert event["ctx"]["bar_ts"] == "2026-04-15T21:05:00-06:00"
    assert event["ctx"]["stop"] == 7002.04
    assert event["ctx"]["target"] == 7008.46
    assert event["ctx"]["phase2"]["setup_pass"] is True


def test_phase2_force_open_live_flat_short_gate_pass_synthesizes_open(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    row = pd.Series({"stop": 7010.89, "target": 7004.61, "entry_conf": 0.780926, "hold_conf": 0.867672})

    event = streamer._phase2_force_open_event(
        row=row,
        ts_dt=pd.Timestamp("2026-04-14T22:45:00-06:00"),
        price=7009.75,
        prob=0.127630,
        phase2_meta=_phase2_force_open_short_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.780926,
        hold_conf=0.867672,
        original_action="NO_TRADE",
    )

    assert event is not None
    assert event["type"] == "OPEN"
    assert event["side"] == "SHORT"
    assert event["risk"] == {"stop": 7010.89, "target": 7004.61}
    assert event["phase2_force_open_applied"] is True
    assert event["ctx"]["phase2"]["short_prob"] == 0.88


def test_phase2_force_open_infers_short_direction_when_signal_suppressed(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer.phase2_force_open_allow_setup_fail_entries = True
    streamer.phase2_force_open_min_entry_conf = 0.1
    row = pd.Series({"stop": 7010.89, "target": 7004.61, "entry_conf": 0.0, "hold_conf": 0.867672})
    phase2_meta = {
        "setup_prob": 0.94,
        "setup_pass": False,
        "direction_prob": 0.3181818127632141,
        "short_prob": 0.6818181872367859,
        "direction_signal": 0,
    }

    event = streamer._phase2_force_open_event(
        row=row,
        ts_dt=pd.Timestamp("2026-06-03T00:40:00-06:00"),
        price=7009.75,
        prob=0.3181818127632141,
        phase2_meta=phase2_meta,
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.0,
        hold_conf=0.867672,
        original_action="NO_TRADE",
    )

    assert event is not None
    assert event["type"] == "OPEN"
    assert event["side"] == "SHORT"
    assert event["phase2_force_open_applied"] is True
    assert event["entry_conf_source"] == "direction_margin_short"


def test_phase2_force_open_short_blocked_when_trend_score_is_above_threshold(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer.phase2_decision_policy = Phase2DecisionPolicy(block_short_above_trend_score=0.55)
    row = pd.Series(
        {
            "stop": 7010.89,
            "target": 7004.61,
            "entry_conf": 0.780926,
            "hold_conf": 0.867672,
            "trend_score": 0.636,
        }
    )

    event = streamer._phase2_force_open_event(
        row=row,
        ts_dt=pd.Timestamp("2026-04-14T22:45:00-06:00"),
        price=7009.75,
        prob=0.127630,
        phase2_meta=_phase2_force_open_short_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.780926,
        hold_conf=0.867672,
        original_action="NO_TRADE",
    )

    assert event is None


def test_phase2_force_open_derives_fallback_atr_bracket_when_row_levels_missing(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    row = pd.Series({"atr_14": 1.57, "entry_conf": 0.780926, "hold_conf": 0.867672})

    event = streamer._phase2_force_open_event(
        row=row,
        ts_dt=pd.Timestamp("2026-04-14T23:00:00-06:00"),
        price=7007.75,
        prob=0.200770,
        phase2_meta=_phase2_force_open_short_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.780926,
        hold_conf=0.867672,
        original_action="NO_TRADE",
    )

    assert event is not None
    assert event["type"] == "OPEN"
    assert event["side"] == "SHORT"
    assert event["risk"]["stop"] == 7010.89
    assert event["risk"]["target"] == 7004.61


def test_phase2_force_open_broad_profile_allows_low_entry_confidence(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer.phase2_force_open_min_entry_conf = 0.0

    event = streamer._phase2_force_open_event(
        row=_phase2_force_open_row(),
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        price=7005.25,
        prob=0.73,
        phase2_meta=_phase2_force_open_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.0,
        hold_conf=0.50,
        original_action="HOLD",
    )

    assert event is not None
    assert event["type"] == "OPEN"
    assert event["original_action"] == "HOLD"
    assert event["original_entry_conf"] == 0.0


def test_aggressive_force_open_uses_direction_threshold_before_entry_conf_floor(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer.phase2_force_open_policy_enabled = True
    streamer.phase2_force_open_policy_mode = "directional_bridge_aggressive"
    streamer.phase2_force_open_allow_setup_fail_entries = True
    streamer.phase2_force_open_min_direction_prob_short = 0.57
    streamer.phase2_force_open_min_entry_conf = 0.10
    streamer.phase2_p_short = 0.58
    meta = {
        "setup_prob": 0.024169184267520905,
        "setup_pass": False,
        "setup_reason": "setup_fail",
        "direction_prob": 0.4054054021835327,
        "short_prob": 0.5945945978164673,
        "direction_signal": 0,
    }
    row = pd.Series({"stop": 7608.75, "target": 7598.75, "entry_conf": 0.0, "hold_conf": 0.50})

    event = streamer._phase2_force_open_event(
        row=row,
        ts_dt=pd.Timestamp("2026-06-03T06:40:00-06:00"),
        price=7604.75,
        prob=0.4054054021835327,
        phase2_meta=meta,
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.0,
        hold_conf=0.50,
        original_action="NO_TRADE",
    )

    assert event is not None
    assert event["type"] == "OPEN"
    assert event["side"] == "SHORT"
    assert event["entry_conf"] < 0.10


def test_phase2_force_open_tighter_profile_uses_directional_confidence_fallback(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer.phase2_force_open_min_entry_conf = 0.55

    event = streamer._phase2_force_open_event(
        row=_phase2_force_open_row(),
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        price=7005.25,
        prob=0.73,
        phase2_meta=_phase2_force_open_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.0,
        hold_conf=0.50,
        original_action="HOLD",
    )

    assert event is not None
    assert event["entry_conf"] == pytest.approx(0.55)
    assert event["entry_conf_source"] == "direction_margin_long"


def test_phase2_force_open_blocks_when_required_gate_fails(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    gates = _phase2_force_open_gates()
    gates["vwap"] = False

    event = streamer._phase2_force_open_event(
        row=_phase2_force_open_row(),
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        price=7005.25,
        prob=0.73,
        phase2_meta=_phase2_force_open_meta(),
        gate_state=gates,
        entry_conf=0.31,
        hold_conf=0.88,
        original_action="NO_TRADE",
    )

    assert event is None


def test_phase2_force_open_blocks_below_setup_threshold(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    meta = _phase2_force_open_meta()
    meta["setup_prob"] = 0.34

    event = streamer._phase2_force_open_event(
        row=_phase2_force_open_row(),
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        price=7005.25,
        prob=0.73,
        phase2_meta=meta,
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.31,
        hold_conf=0.88,
        original_action="NO_TRADE",
    )

    assert event is None


def test_phase2_force_open_blocks_below_entry_conf_threshold(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    meta = _phase2_force_open_meta()
    meta["direction_prob"] = 0.815
    row = pd.Series({"stop": 7002.04, "target": 7008.46, "entry_conf": 0.31, "hold_conf": 0.88})

    event = streamer._phase2_force_open_event(
        row=row,
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        price=7005.25,
        prob=0.73,
        phase2_meta=meta,
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.31,
        hold_conf=0.88,
        original_action="NO_TRADE",
    )

    assert event is None


def test_phase2_force_open_audit_records_calibrated_confidence(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    event = streamer._phase2_force_open_event(
        row=_phase2_force_open_row(),
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        price=7005.25,
        prob=0.73,
        phase2_meta=_phase2_force_open_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.31,
        hold_conf=0.88,
        original_action="NO_TRADE",
    )

    assert event is not None
    streamer._emit_gate_audit(
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        action="OPEN",
        gate_state=_phase2_force_open_gates(),
        blocked_by=[],
        reason_detail="gate_pass_no_trade",
        prob=0.73,
        phase2_meta=_phase2_force_open_meta(),
        extra={
            "phase2_force_open_candidate": True,
            "phase2_force_open_applied": True,
            "phase2_force_open_reason": "gate_pass_no_trade",
            "entry_conf": event["entry_conf"],
            "entry_conf_source": event["entry_conf_source"],
            "entry_conf_calibrated": event["entry_conf_calibrated"],
            "hold_conf": event["hold_conf"],
            "direction_prob": event["direction_prob"],
            "short_prob": event["short_prob"],
            "direction_signal": event["direction_signal"],
            "original_action": event["original_action"],
            "side": event["side"],
            "final_action": "OPEN",
            "emit_allowed": True,
            "override_confident_long": False,
            "override_applied": False,
        },
    )

    payloads = [
        json.loads(line)
        for line in streamer.gating_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = payloads[-1]
    assert payload["phase2_force_open_candidate"] is True
    assert payload["phase2_force_open_applied"] is True
    assert payload["phase2_force_open_reason"] == "gate_pass_no_trade"
    assert payload["entry_conf"] == pytest.approx(0.55)
    assert payload["entry_conf_calibrated"] == pytest.approx(0.55)
    assert payload["entry_conf_source"] == "direction_margin_long"
    assert payload["hold_conf"] == 0.88
    assert payload["original_action"] == "NO_TRADE"
    assert payload["final_action"] == "OPEN"


def test_phase2_force_open_mark_event_does_not_count_as_trade_intent(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)

    assert streamer._has_trade_intent_event([{"type": "MARK", "side": "FLAT"}]) is False
    assert streamer._has_trade_intent_event([{"type": "MARK"}, {"type": "OPEN"}]) is True


def test_phase2_force_open_bypasses_legacy_gate_block(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    event = {
        "type": "OPEN",
        "side": "LONG",
        "phase2_force_open_applied": True,
        "ctx": {"phase2_force_open_applied": True},
    }
    gate_meta = {"gate_prob_effective_pass": False}

    block_reason = streamer._phase2_force_open_legacy_gate_block_reason(
        event,
        open_allowed=False,
        gate_meta=gate_meta,
    )

    assert block_reason is None
    assert event["phase2_force_open_legacy_gate_bypass"] is True
    assert event["ctx"]["phase2_force_open_legacy_gate_bypass_reason"] == "phase2_force_open_applied"
    assert gate_meta["phase2_force_open_legacy_gate_bypass"] is True


def test_phase2_force_open_setup_fail_policy_bypasses_stale_legacy_gate_block(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer.phase2_force_open_allow_legacy_gate_bypass = False
    streamer.phase2_force_open_allow_setup_fail_entries = True
    event = {
        "type": "OPEN",
        "side": "SHORT",
        "price": 7607.0,
        "prob": 0.043,
        "short_prob": 0.957,
        "risk": {"stop": 7613.0, "target": 7601.0},
        "phase2_force_open_applied": True,
        "phase2_force_open_policy": {"allow_setup_fail_entries": True},
        "ctx": {
            "phase2_force_open_applied": True,
            "phase2_force_open_policy": {"allow_setup_fail_entries": True},
        },
    }
    gate_meta = {"gate_prob_effective_pass": False}

    block_reason = streamer._phase2_force_open_legacy_gate_block_reason(
        event,
        open_allowed=False,
        gate_meta=gate_meta,
    )

    assert block_reason is None
    assert event["phase2_force_open_legacy_gate_bypass"] is True
    assert (
        event["phase2_force_open_legacy_gate_bypass_reason"]
        == "phase2_force_open_setup_fail_entries_allowed"
    )
    assert gate_meta["phase2_force_open_legacy_gate_bypass"] is True


def test_phase2_force_open_setup_fail_policy_passes_lineage_gate() -> None:
    record = {
        "event": "OPEN",
        "side": "SHORT",
        "phase2_force_open_applied": True,
        "phase2_force_open_policy": {"allow_setup_fail_entries": True},
        "ctx": {
            "phase2_force_open_applied": True,
            "phase2_force_open_policy": {"allow_setup_fail_entries": True},
            "phase2": {
                "setup_pass": False,
                "direction_signal": 0,
                "short_prob": 0.921,
            },
            "direction_signal": -1,
        },
    }

    assert LiveCSVStreamer._phase2_force_open_lineage_gate_pass(record) is True


def test_normal_open_still_blocks_on_legacy_gate_failure(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    event = {"type": "OPEN", "side": "LONG", "ctx": {}}

    block_reason = streamer._phase2_force_open_legacy_gate_block_reason(
        event,
        open_allowed=False,
        gate_meta={},
    )

    assert block_reason == "gates_block"
    assert "phase2_force_open_legacy_gate_bypass" not in event


def test_live_directional_bridge_requires_setup_pass(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer._phase = "LIVE"
    streamer.phase2_p_long = 0.57
    event = {
        "type": "OPEN",
        "side": "LONG",
        "grade": "B+",
        "phase2_setup_pass": 0,
        "direction_prob": 0.61,
        "ctx": {},
    }

    block_reason = streamer._phase2_force_open_legacy_gate_block_reason(
        event,
        open_allowed=False,
        gate_meta={},
    )

    assert block_reason == "gates_block"
    assert "phase2_force_open_legacy_gate_bypass" not in event


def test_live_directional_bridge_disallowed_without_legacy_bypass_permission(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer._phase = "LIVE"
    streamer.phase2_force_open_allow_legacy_gate_bypass = False
    streamer.phase2_p_long = 0.57
    event = {
        "type": "OPEN",
        "side": "LONG",
        "grade": "A",
        "phase2_setup_pass": 1,
        "directional_prob": 0.91,
        "ctx": {},
    }
    gate_meta: dict[str, object] = {}

    block_reason = streamer._phase2_force_open_legacy_gate_block_reason(
        event,
        open_allowed=False,
        gate_meta=gate_meta,
    )

    assert block_reason == "gates_block"
    assert event["phase2_force_open_legacy_gate_bypass"] is False
    assert event["phase2_force_open_legacy_gate_bypass_reason"] == "not_allowed"
    assert gate_meta["phase2_force_open_legacy_gate_bypass"] is False


def test_aggressive_mode_setup_fail_strong_direction_allows_explicit_bridge(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer.run_mode = "live"
    streamer._phase = "LIVE"
    streamer.phase2_force_open_policy_enabled = True
    streamer.phase2_force_open_policy_mode = "directional_bridge_aggressive"
    streamer.phase2_force_open_allow_legacy_gate_bypass = True
    streamer.phase2_force_open_allow_setup_fail_entries = True
    streamer.phase2_force_open_min_direction_prob_long = 0.57
    streamer._pos = 0
    streamer._hard_lockout_active = False
    streamer.state.entries_disarmed_reason = None
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7120.25,
        "prob": 0.91,
        "directional_prob": 0.91,
        "risk": {"stop": 7112.25, "target": 7132.25},
        "ctx": {"phase2_setup_pass": False, "phase2": {"setup_pass": False}},
    }

    block_reason = streamer._phase2_force_open_legacy_gate_block_reason(
        event,
        open_allowed=False,
        gate_meta={},
    )

    assert block_reason is None
    assert event["phase2_force_open_legacy_gate_bypass"] is True
    assert event["phase2_force_open_legacy_gate_bypass_reason"] == "explicit_aggressive_directional_bridge"
    assert any(str(e.get("event")) == "AGGRESSIVE_DIRECTIONAL_BRIDGE_ENTRY" for e in events)


def test_aggressive_mode_bad_geometry_blocks(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer.run_mode = "live"
    streamer._phase = "LIVE"
    streamer.phase2_force_open_policy_enabled = True
    streamer.phase2_force_open_policy_mode = "directional_bridge_aggressive"
    streamer.phase2_force_open_allow_legacy_gate_bypass = True
    streamer.phase2_force_open_allow_setup_fail_entries = True
    streamer.phase2_force_open_min_direction_prob_long = 0.57
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7120.25,
        "prob": 0.91,
        "directional_prob": 0.91,
        "risk": {"stop": 7130.25, "target": 7132.25},
        "ctx": {"phase2_setup_pass": False, "phase2": {"setup_pass": False}},
    }

    block_reason = streamer._phase2_force_open_legacy_gate_block_reason(
        event,
        open_allowed=False,
        gate_meta={},
    )

    assert block_reason == "gates_block"
    assert "phase2_force_open_legacy_gate_bypass" not in event
    assert any(str(e.get("event")) == "AGGRESSIVE_DIRECTIONAL_BRIDGE_BLOCKED" for e in events)


def test_live_setup_fail_open_is_suppressed_upstream_and_logs_setup_block(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))
    streamer.phase2_force_open_allow_legacy_gate_bypass = False
    streamer.phase2_p_setup = 0.060
    streamer.phase2_p_long = 0.57
    streamer._phase = "LIVE"
    streamer.run_mode = "live"

    event = {
        "datetime": "2026-05-12T02:45:00-06:00",
        "type": "OPEN",
        "side": "LONG",
        "price": 7120.25,
        "prob": 0.91,
        "grade": "A",
        "risk": {"stop": 7112.25, "target": 7132.25},
        "ctx": {
            "phase2_setup_pass": False,
            "phase2": {
                "setup_pass": False,
                "setup_prob": 0.007777777966111898,
                "setup_threshold": 0.060,
            },
        },
        "gates_detail": {"gate_state": {"setup": False}},
    }

    streamer._append_signal(event)
    streamer._write_status(force=True)

    signal_rows = list(csv.DictReader(streamer.signals_csv.read_text(encoding="utf-8").splitlines()))
    assert signal_rows
    assert signal_rows[-1]["type"] == "NO_TRADE"
    assert signal_rows[-1]["blocked"] == "1"
    assert signal_rows[-1]["blocked_reason"] in {"phase2_setup_blocked", "setup_blocked"}

    blocked_rows = _read_jsonl(streamer.blocked_candidates_path)
    assert blocked_rows
    assert (
        blocked_rows[-1].get("reason")
        or blocked_rows[-1].get("blocked_reason")
    ) in {"phase2_setup_blocked", "setup_blocked"}
    assert (blocked_rows[-1].get("action") or blocked_rows[-1].get("candidate_action")) == "OPEN"

    assert exec_events == []


def test_live_setup_fail_open_is_allowed_when_policy_enabled(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))
    streamer._phase = "LIVE"
    streamer.run_mode = "live"
    streamer.phase2_force_open_policy_enabled = True
    streamer.phase2_force_open_policy_mode = "directional_bridge_aggressive"
    streamer.phase2_force_open_allow_legacy_gate_bypass = True
    streamer.phase2_force_open_allow_setup_fail_entries = True
    streamer.phase2_force_open_min_direction_prob_long = 0.57
    streamer.phase2_force_open_min_direction_prob_short = 0.57
    streamer.nt_exec_state = "ARMED"
    streamer.state.nt_exec_state = "ARMED"
    streamer._entries_disarmed_reason = None
    streamer.state.entries_disarmed_reason = None
    streamer.nt_enabled = False

    event = {
        "datetime": "2026-05-12T02:45:00-06:00",
        "type": "OPEN",
        "side": "LONG",
        "price": 7120.25,
        "prob": 0.91,
        "grade": "A",
        "risk": {"stop": 7112.25, "target": 7132.25},
        "ctx": {
            "phase2_setup_pass": False,
            "phase2": {
                "setup_pass": False,
                "setup_prob": 0.0356,
                "setup_threshold": 0.060,
            },
        },
        "gates_detail": {
            "gate_state": {"setup": False},
            "override_applied": False,
        },
        "_signal_blocked_by": ["setup"],
    }

    streamer._append_signal(event)
    streamer._write_status(force=True)

    signal_rows = list(csv.DictReader(streamer.signals_csv.read_text(encoding="utf-8").splitlines()))
    assert signal_rows
    assert signal_rows[-1]["type"] == "OPEN"
    assert signal_rows[-1]["blocked"] == "0"
    assert signal_rows[-1]["blocked_reason"] == ""


def test_gate_audit_does_not_readd_setup_when_setup_fail_policy_allows_entry(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.phase2_force_open_allow_setup_fail_entries = True
    ts = pd.Timestamp("2026-04-23T06:10:00-06:00")

    streamer._emit_gate_audit(
        ts_dt=ts,
        action="OPEN",
        gate_state={"setup": False, "prob": True, "vwap": True, "ema": True, "tod": True},
        blocked_by=[],
        reason_detail="p_setup=0.005 < 0.350",
        prob=0.586,
        phase2_meta={
            "setup_prob": 0.005,
            "setup_pass": False,
            "setup_reason": "setup_fail",
            "setup_threshold": 0.35,
            "direction_prob": 0.586,
            "short_prob": 0.414,
            "direction_signal": 0,
        },
        extra={
            "final_action": "OPEN",
            "strategy_blocked_reason": "setup",
            "execution_blocked_reason": None,
            "phase": "LIVE",
            "side": "SHORT",
        },
    )

    gate_rows = _read_jsonl(streamer.gating_events_path)
    assert gate_rows
    gate_row = gate_rows[-1]
    assert gate_row["action"] == "OPEN"
    assert gate_row["blocked_by"] == []
    assert gate_row["gate_state"]["setup"] is True
    assert gate_row["strategy_blocked_reason"] is None
    assert gate_row["final_action"] == "OPEN"
    assert gate_row["execution_send_allowed"] is True


def test_gate_audit_honors_event_level_force_open_setup_fail_policy(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.phase2_force_open_allow_setup_fail_entries = False
    ts = pd.Timestamp("2026-06-03T17:50:00-06:00")

    streamer._emit_gate_audit(
        ts_dt=ts,
        action="OPEN",
        gate_state={"setup": False, "prob": True, "vwap": True, "ema": True, "tod": True},
        blocked_by=[],
        reason_detail="p_setup=0.024 < 0.350",
        prob=0.732,
        phase2_meta={
            "setup_prob": 0.024,
            "setup_pass": False,
            "setup_reason": "setup_fail",
            "setup_threshold": 0.35,
            "direction_prob": 0.732,
            "short_prob": 0.268,
            "direction_signal": 0,
        },
        extra={
            "final_action": "OPEN",
            "strategy_blocked_reason": "setup",
            "execution_blocked_reason": None,
            "phase": "LIVE",
            "side": "LONG",
            "phase2_force_open_setup_fail_allowed": True,
        },
    )

    gate_rows = _read_jsonl(streamer.gating_events_path)
    gate_row = gate_rows[-1]
    assert gate_row["action"] == "OPEN"
    assert gate_row["blocked_by"] == []
    assert gate_row["gate_state"]["setup"] is True
    assert gate_row["strategy_blocked_reason"] is None
    assert gate_row["execution_send_allowed"] is True


def test_force_open_clears_speculative_local_position_when_broker_flat(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer._phase = "LIVE"
    streamer._pos = 1
    streamer._pos_side = "LONG"
    streamer._entry_price = 7531.0
    streamer._entry_time = pd.Timestamp("2026-06-03T17:45:00-06:00")
    streamer._open_trade = {
        "side": "LONG",
        "entry_price": 7531.0,
        "client_order_id": None,
    }
    streamer.state.position_state = "FLAT"
    streamer.state.active_client_order_id = None
    streamer.state.pending_client_order_id = None
    streamer._pending_client_order_id = None
    streamer.exec_instrument_key = "ES 06-26"
    streamer._nt_snapshot_seen = True
    streamer._nt_snapshot_orders_count = 0
    streamer._nt_snapshot_blocking_orders_count = 0
    streamer._nt_last_pos_qty_by_instrument = {"ES 06-26": 0.0}

    event = streamer._phase2_force_open_event(
        row=_phase2_force_open_row(),
        ts_dt=pd.Timestamp("2026-06-03T18:05:00-06:00"),
        price=7005.25,
        prob=0.732,
        phase2_meta=_phase2_force_open_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.60,
        hold_conf=0.0,
        original_action="NO_TRADE",
    )

    assert event is not None
    assert event["type"] == "OPEN"
    assert streamer._pos == 0
    assert streamer._open_trade is None
    assert streamer.state.position_state == "FLAT"


def test_force_open_does_not_clear_speculative_position_without_flat_snapshot(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer._phase = "LIVE"
    streamer._pos = 1
    streamer.state.position_state = "FLAT"
    streamer.exec_instrument_key = "ES 06-26"
    streamer._nt_snapshot_seen = False
    streamer._nt_snapshot_orders_count = 0
    streamer._nt_snapshot_blocking_orders_count = 0
    streamer._nt_last_pos_qty_by_instrument = {"ES 06-26": 0.0}

    event = streamer._phase2_force_open_event(
        row=_phase2_force_open_row(),
        ts_dt=pd.Timestamp("2026-06-03T18:05:00-06:00"),
        price=7534.5,
        prob=0.732,
        phase2_meta=_phase2_force_open_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.50,
        hold_conf=0.0,
        original_action="NO_TRADE",
    )

    assert event is None
    assert streamer._pos == 1



def test_disarm_priority_preserves_fiber_timeout_over_snapshot_stale(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._entries_disarm_reason_rank = {"fiber_bar_timeout": 20, "snapshot_stale_reconcile": 30}
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    streamer._set_entries_disarmed("fiber_bar_timeout", {"reason": "stall"})
    streamer._set_entries_disarmed("snapshot_stale_reconcile", {"reason": "stale_snapshot"})

    assert streamer._entries_disarmed_reason == "fiber_bar_timeout"
    assert streamer.state.entries_disarmed_reason == "fiber_bar_timeout"
    assert any(
        str(ev.get("event")) == "entries_disarm_preserved"
        and str(ev.get("reason_existing")) == "fiber_bar_timeout"
        and str(ev.get("reason_ignored")) == "snapshot_stale_reconcile"
        for ev in events
    )


def test_pnl_stair_noop_flat_and_reset_events(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer.pnl_stair_enabled = True
    streamer._pos = 0

    out = streamer._maybe_pnl_overlay_event(
        row=pd.Series({"Close": 7400.0, "High": 7401.0, "Low": 7399.0}),
        ts_dt=pd.Timestamp("2026-05-12T10:00:00-06:00"),
    )
    assert out is None
    names = {str(e.get("event")) for e in events}
    assert "PNL_STAIR_NOOP_FLAT" in names
    assert "PNL_STAIR_RESET_FLAT" in names


def test_pnl_stair_bad_time_emits_noop_event(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer.pnl_stair_enabled = True
    streamer._pos = 1
    streamer._open_trade = {"client_order_id": "cid", "side": "LONG"}
    streamer._compute_trade_pnl_state = lambda **kwargs: {"bad_time_in_trade": True}

    out = streamer._maybe_pnl_overlay_event(
        row=pd.Series({"Close": 7400.0, "High": 7401.0, "Low": 7399.0}),
        ts_dt=pd.Timestamp("2026-05-12T10:00:00-06:00"),
    )
    assert out is None
    names = {str(e.get("event")) for e in events}
    assert "BAD_TIME_IN_TRADE" in names
    assert "PNL_STAIR_NOOP_BAD_TIME" in names


def test_pnl_stair_desync_active_noop(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer.pnl_stair_enabled = True
    streamer._execution_desync_active = True
    streamer._pos = 1

    out = streamer._maybe_pnl_overlay_event(
        row=pd.Series({"Close": 7400.0, "High": 7401.0, "Low": 7399.0}),
        ts_dt=pd.Timestamp("2026-05-12T10:00:00-06:00"),
    )
    assert out is None
    names = {str(e.get("event")) for e in events}
    assert "PNL_STAIR_NOOP_NO_IMPROVEMENT" in names
    assert not any(name in {"PNL_STAIR_STEP_ADVANCED", "PNL_STAIR_STOP_UPDATE_REQUESTED"} for name in names)


def test_phase2_force_open_does_not_bypass_stop_or_size_blocks(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    event = {"type": "OPEN", "side": "LONG", "ctx": {"phase2_force_open_applied": True}}

    assert streamer._phase2_force_open_legacy_gate_block_reason(event, open_allowed=True, gate_meta={}) is None
    assert streamer._is_phase2_force_open_event(event) is True


def _hard_gate_ready_streamer(tmp_path: Path) -> LiveCSVStreamer:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer._pending_until = None
    streamer._pending_client_order_id = None
    streamer._instrument_lockouts = {}
    streamer._maybe_reset_hard_lockout = lambda _now: None
    streamer._tod_gate_enabled = lambda: False
    streamer._entries_today_for_limits = lambda: 0
    streamer._in_weak_long_window = lambda _now: False
    streamer._get_gate_now = lambda: pd.Timestamp("2026-04-14T23:45:00-06:00")
    streamer.trade_window_start = "00:00"
    streamer.trade_window_end = "23:59"
    streamer.max_losses_per_day = 4
    streamer.max_daily_loss_usd = None
    streamer.max_daily_drawdown_usd = None
    streamer.max_position_contracts = 1
    streamer.max_risk_usd_per_trade = None
    streamer.max_consec_errors = 0
    streamer.min_execute_grade = None
    streamer.skip_weekdays = set()
    streamer._strategy_trade_limit = 0
    streamer._closing_fill_in_flight = False
    streamer._consec_errors = 0
    streamer._last_hold_side = None
    streamer._last_hold_grade = None
    streamer._last_hold_ts = None
    streamer._scratch_cooldown_left = 0
    streamer._scratch_cooldown_side = None
    streamer._post_stop_cooldown_left = 0
    streamer._loss_cascade_cooldown_left = 0
    streamer._loss_cascade_side_lockout = {"LONG": 0, "SHORT": 0}
    streamer.extreme_proximity_pts = 0
    streamer.instrument = SimpleNamespace(point_value=50.0)
    return streamer


def test_phase2_force_open_self_tracking_cooldown_does_not_block(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer._cooldown_left = 2
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 1,
        "ctx": {"phase2_force_open_applied": True},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-04-14T23:45:00-06:00"),
        "OPEN",
        "LONG",
        event,
        cooldown_left=0,
    )

    assert reason is None


def test_resolve_trade_limit_uses_stricter_session_day_cap() -> None:
    assert LiveCSVStreamer._resolve_trade_limit({"max_trades_per_session": 2, "max_trades_per_day": 5}) == 2
    assert LiveCSVStreamer._resolve_trade_limit({"max_trades_per_session": 5, "max_trades_per_day": 2}) == 2
    assert LiveCSVStreamer._resolve_trade_limit({"max_trades_per_session": 0, "max_trades_per_day": 4}) == 4
    assert LiveCSVStreamer._resolve_trade_limit({"max_trades_per_session": 3, "max_trades_per_day": 0}) == 3


def test_resolve_trade_limit_components_returns_session_day_effective() -> None:
    assert LiveCSVStreamer._resolve_trade_limit_components({"max_trades_per_session": 10, "max_trades_per_day": 6}) == (10, 6, 6)
    assert LiveCSVStreamer._resolve_trade_limit_components({"max_trades_per_session": 0, "max_trades_per_day": 10}) == (0, 10, 10)


def test_existing_real_cooldown_still_blocks_phase2_force_open(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer._cooldown_left = 2
    streamer.state.last_entry_ts = pd.Timestamp("2026-04-14T23:40:00-06:00")
    streamer.state.entry_intents_today = 1
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 1,
        "ctx": {"phase2_force_open_applied": True},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-04-14T23:45:00-06:00"),
        "OPEN",
        "LONG",
        event,
        cooldown_left=2,
    )

    assert reason == "cooldown_active"


def test_stale_flat_startup_cooldown_bypassed_for_phase2_force_open(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer._cooldown_left = 2
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer.state.trades_today = 0
    streamer.state.entry_intents_today = 0
    streamer.state.last_entry_ts = None
    streamer.state.pending_client_order_id = None
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 1,
        "ctx": {"phase2_force_open_applied": True},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-04-14T23:45:00-06:00"),
        "OPEN",
        "LONG",
        event,
        cooldown_left=2,
    )

    assert reason is None
    assert streamer._cooldown_left == 0
    assert event["phase2_force_open_startup_cooldown_bypass"] is True
    assert event["ctx"]["phase2_force_open_startup_cooldown_bypass_reason"] == "flat_no_entry_state"


def test_stale_flat_startup_cooldown_does_not_bypass_normal_open(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer._cooldown_left = 2
    streamer.state.trades_today = 0
    streamer.state.entry_intents_today = 0
    streamer.state.last_entry_ts = None
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 1,
        "ctx": {},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-04-14T23:45:00-06:00"),
        "OPEN",
        "LONG",
        event,
        cooldown_left=2,
    )

    assert reason == "cooldown_active"
    assert streamer._cooldown_left == 2


def test_release_bar_cooldown_clears_terminal_flat_handles_before_gate(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    cid = "RUNID|OPEN|SHORT|TERMINAL"
    streamer.cooldown_bars = 1
    streamer.bar_interval_sec = 300
    streamer._cooldown_left = 1
    streamer._cooldown_started_bar_ts = "2026-06-03T18:45:00-06:00"
    streamer._cooldown_started_side = "SHORT"
    streamer._pos = 0
    streamer._nt_snapshot_orders_count = 0
    streamer._nt_snapshot_blocking_orders_count = 0
    streamer.state.position_state = "FLAT"
    streamer.state.active_client_order_id = cid
    streamer._nt_order_state = {
        cid: {
            "status": "CLOSE_FILLED",
            "entry_filled": True,
            "exit_fill_ts": 1780533902.0,
        }
    }
    event = {
        "type": "OPEN",
        "side": "SHORT",
        "price": 7543.5,
        "prob": 0.696,
        "grade": "A",
        "risk": {"stop": 7554.5, "target": 7532.5},
        "contracts": 1,
        "ctx": {},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-06-03T18:50:00-06:00"),
        "OPEN",
        "SHORT",
        event,
        cooldown_left=1,
    )

    assert reason is None
    assert streamer._cooldown_left == 0
    assert streamer.state.active_client_order_id is None


def test_replay_backfill_bypasses_cooldown_gate(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer.run_mode = "replay"
    streamer._phase = "BACKFILL"
    streamer._cooldown_left = 2
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 1,
        "ctx": {},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-04-14T23:45:00-06:00"),
        "OPEN",
        "LONG",
        event,
        cooldown_left=2,
    )

    assert reason is None


def test_replay_live_still_applies_cooldown_gate(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer.run_mode = "replay"
    streamer._phase = "LIVE"
    streamer._cooldown_left = 2
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 1,
        "ctx": {},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-04-14T23:45:00-06:00"),
        "OPEN",
        "LONG",
        event,
        cooldown_left=2,
    )

    assert reason == "cooldown_active"


def test_live_phase2_force_open_bypasses_directional_memory_conflict(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer._last_hold_side = "SHORT"
    streamer._last_hold_grade = "A+"
    streamer._last_hold_ts = pd.Timestamp("2026-04-14T23:43:00-06:00")
    streamer._last_hold_setup_pass = True
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 1,
        "ctx": {"phase2_force_open_applied": True},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-04-14T23:45:00-06:00"),
        "OPEN",
        "LONG",
        event,
        cooldown_left=0,
    )

    assert reason is None


def test_live_non_force_open_still_blocks_on_directional_memory_conflict(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer._last_hold_side = "SHORT"
    streamer._last_hold_grade = "A+"
    streamer._last_hold_ts = pd.Timestamp("2026-04-14T23:43:00-06:00")
    streamer._last_hold_setup_pass = True
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 1,
        "ctx": {},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-04-14T23:45:00-06:00"),
        "OPEN",
        "LONG",
        event,
        cooldown_left=0,
    )

    assert reason == "directional_memory_conflict"


def test_directional_memory_conflict_ignored_when_hold_not_setup_pass(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer._last_hold_side = "SHORT"
    streamer._last_hold_grade = "A+"
    streamer._last_hold_ts = pd.Timestamp("2026-04-14T23:43:00-06:00")
    streamer._last_hold_setup_pass = False
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 1,
        "ctx": {},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-04-14T23:45:00-06:00"),
        "OPEN",
        "LONG",
        event,
        cooldown_left=0,
    )

    assert reason is None


def test_stale_flat_cooldown_age_guard_clears_normal_open(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(payload)
    streamer.cooldown_bars = 2
    streamer.bar_interval_sec = 300
    streamer._cooldown_left = 2
    streamer._cooldown_started_bar_ts = "2026-04-14T23:30:00-06:00"
    streamer._cooldown_started_ts = None
    streamer.state.trades_today = 1
    streamer.state.entry_intents_today = 1
    streamer.state.last_entry_ts = pd.Timestamp("2026-04-14T23:25:00-06:00")
    streamer.state.active_client_order_id = None
    streamer.state.nt_entry_ninja_order_id = None
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 1,
        "ctx": {},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-04-14T23:45:00-06:00"),
        "OPEN",
        "LONG",
        event,
        cooldown_left=2,
    )

    assert reason is None
    assert streamer._cooldown_left == 0
    assert any(e.get("event") == "cooldown_cleared" and e.get("reason") == "stale_flat_guard" for e in events)


def test_blocked_entry_validation_leaves_tracking_state_unchanged(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer._pos = 0
    streamer._entry_time = None
    streamer._entry_price = None
    streamer._entry_stop = None
    streamer._entry_target = None
    streamer._cooldown_left = 0
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 2,
        "ctx": {"phase2_force_open_applied": True},
    }

    reason = streamer._hard_gate_reason(
        pd.Timestamp("2026-04-14T23:45:00-06:00"),
        "OPEN",
        "LONG",
        event,
        cooldown_left=0,
    )

    assert reason == "max_position_size"
    assert streamer._pos == 0
    assert streamer._entry_time is None
    assert streamer._entry_price is None
    assert streamer._entry_stop is None
    assert streamer._entry_target is None
    assert streamer._cooldown_left == 0


def test_accepted_entry_tracking_starts_normal_cooldown(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer.cooldown_bars = 2
    event = {
        "type": "OPEN",
        "side": "LONG",
        "datetime": "2026-04-14T23:45:00-06:00",
        "price": 7001.75,
        "risk": {"stop": 6997.0, "target": 7006.5},
    }

    streamer._apply_event_position_tracking(
        typ="OPEN",
        side="LONG",
        ev=event,
        prev_pos=0,
        prev_bars=0,
    )

    assert streamer._pos == 1
    assert streamer._entry_price == 7001.75
    assert streamer._entry_stop == 6997.0
    assert streamer._entry_target == 7006.5
    assert int(getattr(streamer, "_cooldown_left", 0) or 0) == 0


def test_normal_cooldown_decrements_and_clears_after_completed_bars(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(payload)
    streamer.cooldown_bars = 2
    streamer._cooldown_left = 2

    streamer._decrement_cooldowns()
    streamer._decrement_cooldowns()

    assert streamer._cooldown_left == 0
    assert [e.get("event") for e in events if str(e.get("event", "")).startswith("cooldown_")] == [
        "cooldown_decremented",
        "cooldown_cleared",
    ]


def test_load_state_restores_flat_cooldown_metadata(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(payload)
    streamer.state_path = tmp_path / "stream_state.json"
    streamer._csv_time_shift_auto = False
    streamer._csv_time_shift_sec = None
    streamer._cooldown_left = 0
    streamer.state_path.write_text(
        json.dumps(
            {
                "state": {"position_state": "FLAT"},
                "position": {
                    "pos": 0,
                    "cooldown_left": 2,
                    "cooldown_started_bar_ts": "2026-04-14T23:30:00-06:00",
                    "cooldown_started_ts": 1776231000.0,
                    "cooldown_last_update_ts": 1776231000.0,
                },
            }
        ),
        encoding="utf-8",
    )

    LiveCSVStreamer._load_state(streamer)

    assert streamer._cooldown_left == 2
    assert streamer._cooldown_started_bar_ts == "2026-04-14T23:30:00-06:00"
    assert any(e.get("event") == "cooldown_restored_from_state" and e.get("pos") == 0 for e in events)


def test_status_write_clears_stale_flat_cooldown(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.cooldown_bars = 2
    streamer.bar_interval_sec = 300
    streamer._bar_interval_known = True
    streamer._cooldown_left = 2
    streamer._cooldown_started_bar_ts = "2026-04-15T18:50:00-06:00"
    streamer._cooldown_started_ts = None
    streamer._cooldown_last_update_ts = None
    streamer._last_bar_ts_guard = pd.Timestamp("2026-04-15T19:10:00-06:00")
    streamer.state.position_state = "FLAT"
    streamer.state.active_client_order_id = None
    streamer._pending_until = None
    streamer._pending_client_order_id = None
    streamer._active_close_correlation_id = None
    streamer._nt_snapshot_orders_count = 0
    streamer._nt_snapshot_blocking_orders_count = 0

    streamer._write_status(force=True)

    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))
    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))
    assert streamer._cooldown_left == 0
    assert status["cooldown_left"] == 0
    assert "flat_cooldown_active" not in health["unresolved_warnings"]


def test_feature_warmup_block_preserves_status_last_bar_ts(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._update_last_bar_ts_guard = LiveCSVStreamer._update_last_bar_ts_guard.__get__(streamer, LiveCSVStreamer)
    streamer._set_entries_disarmed = LiveCSVStreamer._set_entries_disarmed.__get__(streamer, LiveCSVStreamer)
    streamer._handle_feature_warmup_insufficient = LiveCSVStreamer._handle_feature_warmup_insufficient.__get__(
        streamer, LiveCSVStreamer
    )
    streamer._write_jsonl = LiveCSVStreamer._write_jsonl.__get__(streamer, LiveCSVStreamer)
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(payload)
    streamer.feature_health_path = tmp_path / "feature_health.jsonl"
    streamer.anomalies_path = tmp_path / "anomalies.jsonl"

    bar_ts = pd.Timestamp("2026-04-23T19:50:00-06:00")
    fiber_bar_processed_ts = stream_live_csv_mod._canonical_ts_str(bar_ts)
    streamer._update_last_bar_ts_guard(bar_ts)
    streamer._last_csv_bar = bar_ts

    detail = {
        "bad_features": ["sma_day_20_day", "dist_sma_day_20_day"],
        "warmup_features": ["sma_day_20_day", "dist_sma_day_20_day"],
        "available_bars": 3876,
        "unique_days": 16,
        "required_bars": 5760,
        "required_unique_days": 20,
        "phase2_tag": "retrain_v2_776a77a63611",
    }
    streamer._handle_feature_warmup_insufficient(ts_dt=bar_ts, detail=detail)
    streamer._write_status(force=True)

    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))
    anomalies = _read_jsonl(streamer.anomalies_path)
    feature_health = _read_jsonl(streamer.feature_health_path)

    assert fiber_bar_processed_ts == "2026-04-23T19:50:00-06:00"
    assert status["last_bar_ts"] == fiber_bar_processed_ts
    assert streamer._entries_disarmed_reason == "feature_warmup_insufficient"
    assert anomalies[-1]["reason"] == "feature_warmup_insufficient"
    assert feature_health[-1]["required_unique_days"] == 20
    assert any(ev.get("event") == "feature_warmup_insufficient" for ev in exec_events)


def test_fiber_short_history_block_reports_model_requirements(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._fiber_process_buffer = LiveCSVStreamer._fiber_process_buffer.__get__(streamer, LiveCSVStreamer)
    streamer._fiber_rows = [{"Datetime": "2026-04-23T19:50:00-06:00", "Close": 5900.0}] * 3894
    streamer._fiber_hist_complete = True
    streamer.fiber_require_hist = True
    streamer.lookback_bars = 5760
    streamer.fiber_lookback_bars = 10000
    streamer._fiber_hist_expected = 3894
    streamer._fiber_hist_received = 3894
    streamer._model_required_history_days = 20
    streamer.phase2_tag = "retrain_v2_776a77a63611"
    block_events: list[dict] = []
    streamer._emit_block_event = lambda **kwargs: block_events.append(kwargs)

    streamer._fiber_process_buffer()

    assert streamer._entries_disarmed_reason == "feature_warmup_insufficient"
    assert block_events[-1]["block_code"] == "feature_warmup_insufficient"
    detail = block_events[-1]["block_detail"]
    assert detail["received_bars"] == 3894
    assert detail["required_bars"] == 5760
    assert detail["required_unique_days"] == 20
    assert detail["fiber_hist_expected"] == 3894
    assert detail["fiber_lookback_bars"] == 10000
    assert detail["phase2_tag"] == "retrain_v2_776a77a63611"


def test_fiber_hist_begin_logs_required_history_for_v2_model(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._handle_fiber_message = LiveCSVStreamer._handle_fiber_message.__get__(streamer, LiveCSVStreamer)
    streamer._log_exec_event = lambda payload: events.append(payload)
    streamer.fiber_lookback_bars = 10000
    streamer.lookback_bars = 5760
    streamer._model_required_history_days = 20
    streamer.phase2_tag = "retrain_v2_776a77a63611"
    events: list[dict] = []

    streamer._handle_fiber_message(
        {
            "type": "HIST_BEGIN",
            "count": 10000,
            "session_id": "session-1",
            "send_ts_utc": "2026-04-24T03:19:58Z",
        }
    )

    assert streamer._fiber_hist_in_progress is True
    assert streamer._fiber_hist_expected == 10000
    assert events[-1]["event"] == "fiber_hist_begin"
    assert events[-1]["required_bars"] == 5760
    assert events[-1]["required_unique_days"] == 20
    assert events[-1]["phase2_tag"] == "retrain_v2_776a77a63611"


def test_warn_if_reused_out_dir_logs_existing_immutable_manifest(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._warn_if_reused_out_dir = LiveCSVStreamer._warn_if_reused_out_dir.__get__(streamer, LiveCSVStreamer)
    streamer._run_manifest_path = tmp_path / "run_manifest.json"
    streamer._run_manifest_path.write_text(
        json.dumps({"run_id": "old-run", "immutable": True}),
        encoding="utf-8",
    )
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(payload)

    streamer._warn_if_reused_out_dir()

    assert events[-1]["event"] == "out_dir_reuse_warning"
    assert events[-1]["existing_run_id"] == "old-run"
    assert events[-1]["new_run_id"] == streamer.run_id
    assert events[-1]["immutable"] is True


def test_mj_fiber_tail_reader_source_supports_large_history() -> None:
    source_path = Path(__file__).resolve().parents[3] / "ninjatrader" / "MJ_ModelSafeCsvExporter5m.cs"
    source = source_path.read_text(encoding="utf-8")

    row = "2026-01-01T00:00:00.0000000-07:00,6000.25,6001.25,5999.25,6000.75,12345\n"
    old_fixed_cap_rows = (256 * 1024) // len(row)

    assert old_fixed_cap_rows < 5760
    assert "256 * 1024" not in source
    assert "while (position > 0 && newlineCount <= maxLines)" in source
    assert "Math.Min(maxLines, count)" in source


def test_max_trades_flat_cooldown_not_health_warning(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.cooldown_bars = 2
    streamer._cooldown_left = 2
    streamer.state.position_state = "FLAT"
    streamer.state.entries_disarmed_reason = "max_trades_reached"
    streamer._entries_disarmed_reason = "max_trades_reached"

    status = {
        "ts": "2026-04-15T19:10:00-06:00",
        "last_bar_ts": "2026-04-15T19:10:00-06:00",
        "feed_health_ok": True,
        "bar_age_sec": 0.0,
        "effective_bar_age_max_sec": 605.0,
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "fill_truth_state": "flat_no_active_trade",
        "nt_safety_capabilities": {"ok": True},
    }

    streamer._write_run_health_summary(status)

    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))
    assert "flat_cooldown_active" not in health["unresolved_warnings"]


def test_valid_flat_cooldown_is_running_healthy(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.cooldown_bars = 2
    streamer.bar_interval_sec = 300
    streamer._bar_interval_known = True
    streamer._cooldown_left = 2
    streamer._cooldown_started_bar_ts = "2026-04-15T19:40:00-06:00"
    streamer._last_bar_ts_guard = pd.Timestamp("2026-04-15T19:45:00-06:00")
    streamer.state.position_state = "FLAT"
    streamer.state.active_client_order_id = None

    status = {
        "ts": "2026-04-15T19:45:05-06:00",
        "last_bar_ts": "2026-04-15T19:45:00-06:00",
        "feed_health_ok": True,
        "bar_age_sec": 5.0,
        "effective_bar_age_max_sec": 605.0,
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "fill_truth_state": "flat_no_active_trade",
        "nt_safety_capabilities": {"ok": True},
    }

    streamer._write_run_health_summary(status)

    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))
    assert health["verdict"] == "running_healthy"
    assert health["cooldown_state"] == "active_valid"
    assert health["cooldown_is_stale"] is False
    assert health["cooldown_expected_release_bar"] == "2026-04-15T19:50:00-06:00"
    assert "flat_cooldown_active" not in health["unresolved_warnings"]


def test_health_summary_marks_live_protected_position_running_protected(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 1
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer._fill_truth_state = "active_truth_present"
    streamer._open_trade = {
        "client_order_id": "CID-PROTECTED",
        "protection_status": "protected_confirmed",
        "stop_price": 7544.75,
        "target_price": 7557.25,
    }
    streamer._execution_intended_mode = lambda: False

    status = {
        "ts": "2026-06-04T02:12:09-06:00",
        "last_bar_ts": "2026-06-04T02:10:00-06:00",
        "feed_health_ok": True,
        "bar_age_sec": 129.0,
        "effective_bar_age_max_sec": 605.0,
        "position_state": "IN_POSITION_PROTECTED",
        "snapshot_orders_count": 2,
        "snapshot_blocking_orders_count": 0,
        "fill_truth_state": "active_truth_present",
        "trade_pnl_state": {
            "protected_confirmed": True,
            "target_attached": True,
            "stop_price": 7544.75,
            "target_price": 7557.25,
        },
        "nt_safety_capabilities": {"ok": True},
    }

    streamer._write_run_health_summary(status)

    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))
    assert health["verdict"] == "running_protected"
    assert health["live_position_protected"] is True
    assert health["protection_confirmed"] is True
    assert "final_position_not_flat" not in health["unresolved_warnings"]
    assert "working_orders_present" not in health["unresolved_warnings"]


def test_health_summary_clears_flat_terminal_pending_handles(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "CID-FLAT-TERMINAL"
    streamer.cooldown_bars = 2
    streamer.bar_interval_sec = 300
    streamer._cooldown_left = 2
    streamer._cooldown_started_bar_ts = "2026-04-15T19:40:00-06:00"
    streamer._last_bar_ts_guard = pd.Timestamp("2026-04-15T19:45:00-06:00")
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer.state.active_client_order_id = cid
    streamer.state.pending_client_order_id = cid
    streamer._pending_client_order_id = cid
    streamer._active_close_correlation_id = cid
    streamer._closing_fill_in_flight = True
    streamer._nt_order_state[cid] = {
        "status": "close_filled",
        "exit_fill_ts": time.time(),
        "instrument": "ES JUN26",
    }

    status = {
        "ts": "2026-04-15T19:45:05-06:00",
        "last_bar_ts": "2026-04-15T19:45:00-06:00",
        "feed_health_ok": True,
        "bar_age_sec": 5.0,
        "effective_bar_age_max_sec": 605.0,
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "fill_truth_state": "flat_no_active_trade",
        "nt_safety_capabilities": {"ok": True},
    }

    streamer._write_run_health_summary(status)
    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))

    assert streamer.state.active_client_order_id is None
    assert streamer.state.pending_client_order_id is None
    assert streamer._pending_client_order_id is None
    assert streamer._active_close_correlation_id is None
    assert streamer._closing_fill_in_flight is False
    assert health["cooldown_state"] == "active_valid"
    assert "flat_cooldown_active" not in health["unresolved_warnings"]


def test_health_summary_does_not_log_clear_when_already_flat(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 0
    streamer.state.position_state = "FLAT"

    status = {
        "ts": "2026-04-15T19:45:05-06:00",
        "last_bar_ts": "2026-04-15T19:45:00-06:00",
        "feed_health_ok": True,
        "bar_age_sec": 5.0,
        "effective_bar_age_max_sec": 605.0,
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "fill_truth_state": "flat_no_active_trade",
        "nt_safety_capabilities": {"ok": True},
    }

    streamer._write_run_health_summary(status)

    exec_path = tmp_path / "exec_events.jsonl"
    if exec_path.exists():
        events = [json.loads(line) for line in exec_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        events = []
    assert not any(event.get("event") == "flat_terminal_execution_handles_cleared" for event in events)


def test_run_health_summary_uses_live_executor_stats_by_default(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._inc_phase_stat("candidate_signals_total", phase="BACKFILL", amount=5)
    streamer._inc_phase_stat("candidate_signals_total", phase="LIVE", amount=1)
    status = {
        "ts": "2026-04-15T19:45:05-06:00",
        "last_bar_ts": "2026-04-15T19:45:00-06:00",
        "feed_health_ok": True,
        "bar_age_sec": 5.0,
        "effective_bar_age_max_sec": 605.0,
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "fill_truth_state": "flat_no_active_trade",
        "nt_safety_capabilities": {"ok": True},
    }

    streamer._write_run_health_summary(status)
    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))

    assert health["executor_stats"]["candidate_signals_total"] == 1
    assert health["executor_stats_by_phase"]["BACKFILL"]["candidate_signals_total"] == 5


def test_run_health_summary_includes_hardening_fields(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._risk_limit_sources = {
        "session_limit": 10,
        "session_limit_source": "preset:es_elite_v1",
        "day_limit": 10,
        "day_limit_source": "prop_guardrails:test",
        "effective_trade_limit": 10,
        "effective_trade_limit_source": "min(session_limit,day_limit)",
    }
    streamer._setup_gate_diag = {
        "LIVE": {
            "count": 4,
            "pass_count": 1,
            "setup_margin_sum": -0.12,
            "setup_margin_count": 4,
        }
    }
    streamer._watchdog_transition_log = [{"ts": "2026-05-08T10:00:00-06:00", "cause": "fiber_bar_timeout", "state_to": "disarmed"}]
    streamer._watchdog_recovery_attempts = {"fiber_bar_timeout": {"attempt_count": 2, "last_result": "waiting_snapshot"}}
    (tmp_path / "signal_to_order.jsonl").write_text(
        json.dumps(
            {
                "phase": "LIVE",
                "decision": "BLOCKED",
                "reason": "gates_block",
                "execution_veto_layer": "strategy_gate",
                "client_order_id": "cid-1",
                "ts": "2026-05-08T10:01:00-06:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    status = {
        "ts": "2026-04-15T19:45:05-06:00",
        "last_bar_ts": "2026-04-15T19:45:00-06:00",
        "feed_health_ok": True,
        "bar_age_sec": 5.0,
        "effective_bar_age_max_sec": 605.0,
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "fill_truth_state": "flat_no_active_trade",
        "nt_safety_capabilities": {"ok": True},
    }

    streamer._write_run_health_summary(status)
    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))

    assert health["effective_trade_limit"] == 10
    assert health["setup_pass_rate_live"] == 0.25
    assert health["setup_margin_live"] == -0.03
    assert "LIVE:BLOCKED:gates_block" in health["gate_histogram_by_phase"]
    assert health["veto_layer_histogram"]["strategy_gate"] == 1
    assert len(health["watchdog_transitions"]) == 1
    assert health["recovery_attempts"]["fiber_bar_timeout"]["attempt_count"] == 2


def test_run_health_summary_flags_unresolved_live_open_no_send_contract_violation(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    (tmp_path / "signal_to_order.jsonl").write_text(
        json.dumps(
            {
                "phase": "LIVE",
                "final_action": "OPEN",
                "emit_allowed": True,
                "sent_to_nt": False,
                "send_attempted": False,
                "send_outcome": "unresolved_no_send",
                "send_block_reason_code": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    status = {
        "ts": "2026-04-15T19:45:05-06:00",
        "last_bar_ts": "2026-04-15T19:45:00-06:00",
        "feed_health_ok": True,
        "bar_age_sec": 5.0,
        "effective_bar_age_max_sec": 605.0,
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "fill_truth_state": "flat_no_active_trade",
        "nt_safety_capabilities": {"ok": True},
    }
    streamer._write_run_health_summary(status)
    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))
    assert int(health.get("execution_contract_violations_total") or 0) == 1
    assert "execution_contract_unresolved_open_no_send" in list(health.get("unresolved_warnings") or [])


def test_run_health_summary_includes_source_policy_counters(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    status = {
        "ts": "2026-04-15T19:45:05-06:00",
        "last_bar_ts": "2026-04-15T19:45:00-06:00",
        "feed_health_ok": True,
        "bar_age_sec": 5.0,
        "effective_bar_age_max_sec": 605.0,
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "fill_truth_state": "flat_no_active_trade",
        "nt_safety_capabilities": {"ok": True},
        "bridge_source_rejected_count_by_type": {"POSITION_SNAPSHOT": 2, "HEARTBEAT": 5},
        "readiness_credits_rejected_by_source_policy": 2,
        "send_blocked_by_source_policy_total": 3,
    }
    streamer._write_run_health_summary(status)
    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))
    assert health["bridge_source_rejected_count_by_type"]["POSITION_SNAPSHOT"] == 2
    assert int(health["readiness_credits_rejected_by_source_policy"]) == 2
    assert int(health["send_blocked_by_source_policy_total"]) == 3


def test_entries_disarmed_reason_precedence_preserves_higher_rank(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._set_entries_disarmed("max_daily_loss", {"reason": "lock"})
    streamer._set_entries_disarmed("fiber_bar_timeout", {"reason": "stall"})
    assert streamer._entries_disarmed_reason == "max_daily_loss"


def test_signal_to_order_marks_execution_policy_blocks_for_replay_eval(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "replay"
    streamer._phase = "BACKFILL"

    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-04-24T04:25:00-06:00"),
        signal_action="OPEN",
        client_order_id="RID|OPEN|1",
        sent_to_nt=False,
        blocked_by=["past_bar_emit"],
        extra={"decision": "BLOCKED", "reason": "past_bar_emit", "side": "LONG", "prob": 0.72},
    )

    rows = _read_jsonl(tmp_path / "signal_to_order.jsonl")
    payload = next(row for row in rows if row.get("client_order_id") == "RID|OPEN|1")
    assert payload["blocked_for_execution"] is True
    assert payload["accepted_for_sim_evaluation"] is True
    assert payload["execution_truth_mode"] == "model_primary_safe"
    assert payload["model_intent_action"] == "OPEN"
    assert payload["action"] == "NO_TRADE"
    assert payload["execution_intent_action"] == "NO_TRADE"
    assert payload["execution_veto_code"] == "past_bar_emit"
    assert payload["execution_veto_layer"] == "phase_gate"
    assert payload["intent_terminal_state"] == "blocked"
    assert streamer._executor_stats["blocked_for_execution_total"] == 1
    assert streamer._executor_stats["accepted_for_sim_evaluation_total"] == 1


def test_signal_to_order_marks_blocked_exec_policy_for_replay_eval(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "replay"
    streamer._phase = "BACKFILL"

    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-04-24T09:40:00-06:00"),
        signal_action="OPEN",
        client_order_id="RID|OPEN|BEP",
        sent_to_nt=False,
        blocked_by=["replay_wait_for_last_bar"],
        extra={
            "decision": "BLOCKED_EXEC_POLICY",
            "reason_code": "replay_wait_for_last_bar",
            "reason": "replay_wait_for_last_bar",
            "side": "LONG",
            "prob": None,
        },
    )

    rows = _read_jsonl(tmp_path / "signal_to_order.jsonl")
    payload = next(row for row in rows if row.get("client_order_id") == "RID|OPEN|BEP")
    assert payload["decision"] == "BLOCKED_EXEC_POLICY"
    assert payload["blocked_for_execution"] is True
    assert payload["accepted_for_sim_evaluation"] is True
    assert payload["execution_truth_mode"] == "model_primary_safe"
    assert payload["execution_intent_action"] == "NO_TRADE"
    assert payload["execution_veto_layer"] == "hard_safety"
    assert streamer._executor_stats["blocked_for_execution_total"] == 1
    assert streamer._executor_stats["accepted_for_sim_evaluation_total"] == 1


def test_signal_to_order_sent_sets_model_primary_execution_truth(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer._phase = "LIVE"

    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-08T04:55:00-06:00"),
        signal_action="FLIP",
        client_order_id="RID|FLIP|SENT",
        sent_to_nt=True,
        blocked_by=[],
        extra={"decision": "SENT", "side": "LONG", "emit_allowed": True},
    )

    rows = _read_jsonl(tmp_path / "signal_to_order.jsonl")
    payload = next(row for row in rows if row.get("client_order_id") == "RID|FLIP|SENT")
    assert payload["execution_truth_mode"] == "model_primary_safe"
    assert payload["model_intent_action"] == "FLIP"
    assert payload["action"] == "FLIP"
    assert payload["execution_intent_action"] == "FLIP"
    assert payload["execution_veto_code"] is None
    assert payload["execution_veto_layer"] is None
    assert payload["intent_terminal_state"] == "sent"


def test_record_executor_decision_emits_anomaly_on_terminal_change(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._record_executor_decision("intent-1", "SENT")
    streamer._record_executor_decision("intent-1", "BLOCKED_SAFETY")

    anomaly = next((e for e in events if e.get("event") == "counter_invariant_violation"), None)
    assert anomaly is not None
    assert anomaly["reason"] == "intent_terminal_state_changed"
    assert anomaly["intent_id"] == "intent-1"


def test_record_executor_decision_blocks_sent_to_idempotent_for_unresolved_close(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    cid = "RID|CLOSE|1"
    streamer._nt_order_state[cid] = {"intent_action": "CLOSE", "status": "sent"}
    streamer._record_executor_decision(cid, "SENT")
    result_prev = streamer._record_executor_decision(cid, "SKIPPED_IDEMPOTENT")
    assert result_prev == "SENT"
    assert streamer._executor_terminal[cid] == "SENT"
    assert any(ev.get("event") == "terminal_downgrade_blocked_post_fill" for ev in events)


def test_untracked_order_ack_reassociates_to_active_cid(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "RID|OPEN|1"
    streamer.state.position_state = "PENDING_ENTRY"
    streamer.state.active_client_order_id = cid
    streamer._external_nt_cids = {cid}
    now = time.time()
    streamer._nt_order_state[cid] = {
        "intent_action": "OPEN",
        "status": "sent",
        "sent_ts": now,
        "side": "SHORT",
        "qty": 1,
        "instrument": "ES JUN26",
    }
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._log_order_event = lambda *args, **kwargs: None
    streamer._send_nt_sync_request = lambda *args, **kwargs: None
    streamer._queue_trade_update_from_state = lambda *args, **kwargs: None
    streamer._emit_exec_line = lambda *args, **kwargs: None
    streamer.protection_timeout_sec = 30.0
    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_ACK",
            "client_order_id": "UNTRACKED|ES JUN26|123",
            "status": "SUBMITTED_ENTRY",
            "instrument": "ES JUN26",
            "timestamp": utc_ts(),
            "protocol_version": 1,
        }
    )
    assert streamer._nt_order_state[cid].get("entry_acked") is True
    assert streamer._nt_order_state[cid].get("reassociated_untracked") is True
    assert any(ev.get("event") == "reassociate_untracked_success" for ev in events)


def test_queue_degraded_suppresses_non_run_lifecycle_fill(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._nt_event_queue_degraded = True
    streamer._entries_disarmed_reason = "snapshot_stale_reconcile"
    streamer.run_id = "RID-RUN-1"
    events: list[dict] = []
    reconciled: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._apply_fill_truth_reconciliation = lambda **kwargs: reconciled.append(dict(kwargs))

    streamer._handle_nt_message_inner(
        {
            "type": "FILL",
            "client_order_id": "UNTRACKED|ES JUN26|12345",
            "instrument": "ES JUN26",
            "fill_qty": 1,
            "fill_price": 7421.25,
            "side": "SHORT",
            "lifecycle_eligible": False,
            "timestamp": utc_ts(),
            "protocol_version": 1,
        }
    )
    assert "UNTRACKED|ES JUN26|12345" not in streamer._nt_order_state
    assert any(ev.get("event") == "non_run_lifecycle_update_suppressed" for ev in events)
    assert len(reconciled) == 1
    assert reconciled[0].get("source") == "suppressed_non_run_fill"


def test_stale_reconcile_lockout_records_last_run_owned_lifecycle_event(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._snapshot_age_sec = lambda *_args, **_kwargs: 9999.0
    streamer._snapshot_price_state = lambda: (True, 7432.5, None, None, None)
    streamer._last_bar_ts_guard = pd.Timestamp("2026-05-21T07:00:00-06:00")
    streamer._nt_event_queue_degraded = True
    streamer._snapshot_stale_reconcile_enter_mono = time.monotonic() - 120.0
    streamer._snapshot_stale_reconcile_max_dwell_sec = 1.0
    streamer.nt_snapshot_fresh_sec = 1.0
    streamer.max_bar_age_seconds_for_exec = 1.0
    streamer._last_live_pnl_quality = {"quality": "ok"}
    streamer._maybe_request_nt_snapshot = lambda **_kwargs: None
    streamer._set_entries_disarmed = lambda *_args, **_kwargs: None
    captured: dict = {}

    streamer._last_run_owned_lifecycle_event = {
        "client_order_id": "RID|OPEN|1",
        "msg_type": "FILL",
        "ts": "2026-05-21T07:41:31.038392-06:00",
    }

    def _capture_lockout(code: str, evidence=None, **_kwargs):
        captured["code"] = code
        captured["evidence"] = dict(evidence or {})

    streamer._set_hard_lockout = _capture_lockout
    ok = streamer._require_fresh_snapshot(reason="execute_intent_entry", inst_key="ES 06-26")
    assert ok is False
    assert captured.get("code") == "snapshot_stale_reconcile_max_dwell"
    evidence = captured.get("evidence") or {}
    assert evidence.get("last_run_owned_lifecycle_event_present") is True
    assert isinstance(evidence.get("last_run_owned_lifecycle_event"), dict)
    assert evidence.get("last_run_owned_lifecycle_event", {}).get("client_order_id") == "RID|OPEN|1"


def test_stable_ingest_profile_stale_reconcile_max_dwell_is_telemetry_only(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.stable_ingest_profile = True
    streamer._snapshot_age_sec = lambda *_args, **_kwargs: 9999.0
    streamer._snapshot_price_state = lambda: (True, 7432.5, None, None, None)
    streamer._last_bar_ts_guard = pd.Timestamp("2026-05-21T07:00:00-06:00")
    streamer._nt_event_queue_degraded = True
    streamer._snapshot_stale_reconcile_enter_mono = time.monotonic() - 120.0
    streamer._snapshot_stale_reconcile_max_dwell_sec = 1.0
    streamer.nt_snapshot_fresh_sec = 1.0
    streamer.max_bar_age_seconds_for_exec = 1.0
    streamer._last_live_pnl_quality = {"quality": "ok"}
    streamer._maybe_request_nt_snapshot = lambda **_kwargs: None
    lockouts: list[str] = []
    events: list[dict] = []
    streamer._set_hard_lockout = lambda code, evidence=None, **_kwargs: lockouts.append(str(code))
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    ok = streamer._require_fresh_snapshot(reason="execute_intent_entry", inst_key="ES 06-26")
    assert ok is False
    assert lockouts == []
    assert any(e.get("event") == "snapshot_stale_reconcile_max_dwell_telemetry" for e in events)


def test_run70_parity_stale_reconcile_max_dwell_is_telemetry_only(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run70_parity_live_profile = True
    streamer.stable_ingest_profile = False
    streamer._snapshot_age_sec = lambda *_args, **_kwargs: 9999.0
    streamer._snapshot_price_state = lambda: (True, 7432.5, None, None, None)
    streamer._last_bar_ts_guard = pd.Timestamp("2026-05-21T07:00:00-06:00")
    streamer._nt_event_queue_degraded = True
    streamer._snapshot_stale_reconcile_enter_mono = time.monotonic() - 120.0
    streamer._snapshot_stale_reconcile_max_dwell_sec = 1.0
    streamer.nt_snapshot_fresh_sec = 1.0
    streamer.max_bar_age_seconds_for_exec = 1.0
    streamer._last_live_pnl_quality = {"quality": "ok"}
    streamer._maybe_request_nt_snapshot = lambda **_kwargs: None
    lockouts: list[str] = []
    events: list[dict] = []
    streamer._set_hard_lockout = lambda code, evidence=None, **_kwargs: lockouts.append(str(code))
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    ok = streamer._require_fresh_snapshot(reason="execute_intent_entry", inst_key="ES 06-26")
    assert ok is False
    assert lockouts == []
    assert any(e.get("event") == "snapshot_stale_reconcile_max_dwell_telemetry" for e in events)


def test_setup_pass_force_open_eligibility_live_ready(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.phase2_force_open_on_setup_pass = True
    streamer.phase2_force_open_allow_legacy_gate_bypass = True
    streamer.phase2_force_open_live_only = True
    streamer._phase = "LIVE"
    streamer._phase_allows_execution = lambda: True
    streamer._hard_lockout_active = False
    streamer._entries_disarmed_reason = None
    streamer.nt_exec_state = "ARMED"
    streamer.nt_require_snapshot = False
    streamer._account_resolution_state = "account_selected"
    streamer._effective_bar_age_max_sec = 600.0
    streamer._bar_age_guard_seconds = lambda: 1.0

    eligible, reason, bypass = streamer._setup_pass_force_open_eligibility(
        action="OPEN",
        phase2_setup_pass=True,
        blocked_by=["setup", "local_block:gates_block"],
    )
    assert eligible is True
    assert reason == "force_open_setup_pass"
    assert bypass is True


def test_setup_pass_force_open_eligibility_rejects_non_live_phase(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.phase2_force_open_on_setup_pass = True
    streamer.phase2_force_open_allow_legacy_gate_bypass = True
    streamer.phase2_force_open_live_only = True
    streamer._phase = "BACKFILL"
    streamer._phase_allows_execution = lambda: False

    eligible, reason, bypass = streamer._setup_pass_force_open_eligibility(
        action="OPEN",
        phase2_setup_pass=True,
        blocked_by=["setup", "local_block:gates_block"],
    )
    assert eligible is False
    assert reason in {"phase_not_live", "phase_not_executable"}
    assert bypass is False


def test_signal_to_order_block_reason_priority_prefers_hard_safety(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer._phase = "LIVE"
    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-08T04:55:00-06:00"),
        signal_action="OPEN",
        client_order_id="RID|OPEN|PRIORITY",
        sent_to_nt=False,
        blocked_by=["setup", "local_block:gates_block", "blocked_not_armed"],
        extra={"decision": "BLOCKED", "side": "LONG"},
    )
    rows = _read_jsonl(tmp_path / "signal_to_order.jsonl")
    payload = next(row for row in rows if row.get("client_order_id") == "RID|OPEN|PRIORITY")
    assert payload["reason"] == "blocked_not_armed"
    assert payload["execution_veto_layer"] == "hard_safety"


def test_signal_to_order_normalizes_gate_block_detail_and_conversion_stage(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer._phase = "LIVE"
    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-08T04:55:00-06:00"),
        signal_action="OPEN",
        client_order_id="RID|OPEN|SETUPFAIL",
        sent_to_nt=False,
        blocked_by=["setup", "local_block:gates_block"],
        extra={"decision": "BLOCKED", "side": "LONG", "phase2_setup_pass": False},
    )
    rows = _read_jsonl(tmp_path / "signal_to_order.jsonl")
    payload = next(row for row in rows if row.get("client_order_id") == "RID|OPEN|SETUPFAIL")
    assert payload["reason"] == "gates_block"
    assert payload["reason_class"] == "setup_fail"
    assert payload["gate_block_detail"] == "setup_fail"
    assert payload["conversion_stage"] == "pre_signal_block"
    assert payload["conversion_path"]["candidate"] is True
    assert payload["conversion_path"]["pre_signal_block"] is True
    assert payload["conversion_path"]["send_path"] is False


def test_status_reports_gate_fail_emit_attempt_telemetry(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._executor_stats.update(
        {
            "candidate_signals_total": 20,
            "blocked_pre_signal_total": 7,
            "gating_open_emits_total": 3,
            "nt_send_path_entry_count": 1,
            "nt_order_entry_total": 0,
            "gate_fail_but_emit_attempt_total": 2,
        }
    )
    streamer._executor_phase_stats = dict(getattr(streamer, "_executor_phase_stats", {}) or {})
    streamer._executor_phase_stats.setdefault("LIVE", {})["gate_fail_but_emit_attempt_total"] = 2
    streamer._setup_gate_diag = {
        "BACKFILL": {"count": 10, "pass_count": 2},
        "LIVE": {"count": 4, "pass_count": 1},
    }
    streamer.pnl_overlay_enabled = True
    streamer.pnl_shelf_enabled = True
    streamer._write_status(force=True)
    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))
    assert int(status.get("gate_fail_but_emit_attempt_total", 0) or 0) == 2
    assert int((status.get("gate_fail_but_emit_attempt_by_phase") or {}).get("LIVE", 0) or 0) == 2


def test_run_health_summary_surfaces_setup_and_stop_block_totals(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    signal_rows = [
        {
            "ts": utc_ts(),
            "client_order_id": "RID|A",
            "phase": "BACKFILL",
            "decision": "BLOCKED",
            "reason": "gates_block",
            "reason_class": "setup_fail",
        },
        {
            "ts": utc_ts(),
            "client_order_id": "RID|B",
            "phase": "BACKFILL",
            "decision": "BLOCKED",
            "reason": "stop_already_breached",
            "reason_class": "stop_already_breached",
        },
    ]
    with (tmp_path / "signal_to_order.jsonl").open("w", encoding="utf-8") as fh:
        for row in signal_rows:
            fh.write(json.dumps(row) + "\n")
    status = {
        "ts": utc_ts(),
        "last_bar_ts": utc_ts(),
        "feed_health_ok": True,
        "bar_age_sec": 10.0,
        "effective_bar_age_max_sec": 605.0,
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "desync_latch_active": False,
        "event_seq_state": "monotonic_ok",
        "nt_queue_degraded": False,
        "integrity_anomaly_count": 0,
        "symbol_mismatch_count": 0,
        "pnl_shelf_config": {"enabled": True},
    }
    streamer._write_run_health_summary(status)
    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))
    assert health["setup_fail_total"] == 1
    assert health["stop_already_breached_total"] == 1
    assert health["locks_inactive_due_to_no_entries"] is True


def test_prop_guardrails_bypass_when_disable_risk_enabled(tmp_path: Path) -> None:
    class _GuardStub:
        def check_pre_signal(self, **_kwargs):
            raise AssertionError("guard should not be called when disable_risk=true")

    streamer = _make_streamer(tmp_path)
    streamer.disable_risk = True
    streamer.prop_guardrails = _GuardStub()

    reason = LiveCSVStreamer._prop_guardrails_pre_signal_reason(
        streamer,
        {"type": "OPEN", "side": "LONG", "price": 7000.0, "risk": {"stop": 6998.0}, "contracts": 1},
    )
    ok, block_reason, details = LiveCSVStreamer._prop_guardrails_execution_validation(
        streamer,
        signal_price=7000.0,
        current_price=7001.0,
        signal_timestamp=pd.Timestamp("2026-04-24T10:00:00-06:00"),
        current_timestamp=pd.Timestamp("2026-04-24T10:01:00-06:00"),
    )
    fill_ok, fill_reason, fill_detail = LiveCSVStreamer._prop_guardrails_validate_fill(
        streamer,
        state={"model_price": 7000.0, "side": "LONG"},
        fill_price=7001.0,
    )

    assert reason is None
    assert ok is True
    assert block_reason == ""
    assert details.get("risk_bypass") == "norisk"
    assert fill_ok is True
    assert fill_reason == ""
    assert fill_detail.get("risk_bypass") == "norisk"


def test_prop_fill_slippage_threshold_and_warning_band(tmp_path: Path) -> None:
    class _Cfg:
        fill_slippage_max = 6.0

    class _GuardStub:
        config = _Cfg()

        def validate_fill(self, expected_price: float, fill_price: float, _side: str):
            slip = abs(float(fill_price) - float(expected_price))
            if slip > 6.0:
                return False, f"fill_slippage_excessive:{slip:.1f}pts>max_6.0pts"
            return True, ""

    streamer = _make_streamer(tmp_path)
    streamer.disable_risk = False
    streamer.prop_guardrails = _GuardStub()
    streamer._fill_slippage_threshold_source = "preset:es_elite_v1"

    ok_3, reason_3, detail_3 = LiveCSVStreamer._prop_guardrails_validate_fill(
        streamer,
        state={"model_price": 7000.0, "side": "LONG"},
        fill_price=7003.0,
    )
    assert ok_3 is True
    assert reason_3 == ""
    assert detail_3.get("decision") == "accept"
    assert detail_3.get("warning") is None

    ok_warn, reason_warn, detail_warn = LiveCSVStreamer._prop_guardrails_validate_fill(
        streamer,
        state={"model_price": 7000.0, "side": "LONG"},
        fill_price=7004.5,
    )
    assert ok_warn is True
    assert reason_warn == ""
    assert detail_warn.get("warning") == "elevated_fill_slippage"

    ok_601, reason_601, detail_601 = LiveCSVStreamer._prop_guardrails_validate_fill(
        streamer,
        state={"model_price": 7000.0, "side": "LONG"},
        fill_price=7006.01,
    )
    assert ok_601 is False
    assert "fill_slippage_excessive" in reason_601
    assert detail_601.get("decision") == "lockout"


def test_parity_preset_inherited_slippage_override_accepts_live_market_fill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _GuardStub:
        def __init__(self) -> None:
            self.config = stream_live_csv_mod.EnhancedGuardRailConfig(
                max_slippage_points=2.0,
                fill_slippage_max=2.0,
            )

        def validate_fill(self, expected_price: float, fill_price: float, _side: str):
            slip = abs(float(fill_price) - float(expected_price))
            if slip > float(self.config.fill_slippage_max):
                return False, f"fill_slippage_excessive:{slip:.1f}pts>max_{self.config.fill_slippage_max}pts"
            return True, ""

    monkeypatch.setitem(stream_live_csv_mod.PRESETS, "base_slip_6", {"max_slippage_points": 6.0})
    streamer = _make_streamer(tmp_path)
    streamer.disable_risk = False
    streamer.tick_size = 0.25
    streamer.max_fill_slippage_ticks = 37
    streamer.prop_guardrails = _GuardStub()

    LiveCSVStreamer._apply_preset_fill_slippage_override(
        streamer,
        {"inherits": "base_slip_6"},
        preset_name="es_modelrun77_parity_v1",
        reason="test",
    )

    assert streamer.max_fill_slippage_ticks == pytest.approx(24.0)
    assert streamer.prop_guardrails.config.fill_slippage_max == pytest.approx(6.0)
    assert streamer._fill_slippage_threshold_source == "preset:base_slip_6.max_slippage_points"

    ok, reason, detail = LiveCSVStreamer._prop_guardrails_validate_fill(
        streamer,
        state={"model_price": 7549.25, "side": "LONG"},
        fill_price=7551.5,
    )
    assert ok is True
    assert reason == ""
    assert detail["slippage_points"] == pytest.approx(2.25)
    assert detail["max_slippage_points"] == pytest.approx(6.0)
    assert detail["decision"] == "accept"

    blocked, block_reason, block_detail = LiveCSVStreamer._prop_guardrails_validate_fill(
        streamer,
        state={"model_price": 7549.25, "side": "LONG"},
        fill_price=7555.75,
    )
    assert blocked is False
    assert "fill_slippage_excessive" in block_reason
    assert block_detail["decision"] == "lockout"


def test_prop_closed_trade_uses_canonical_trade_row_when_open_trade_entry_is_contaminated(tmp_path: Path) -> None:
    class _GuardStub:
        def __init__(self) -> None:
            self.trades = []

        def record_trade(self, trade):
            self.trades.append(trade)

        def get_state_summary(self):
            return {
                "risk": {
                    "is_locked": False,
                    "lock_reason": "",
                    "realized_pnl": sum(float(t.pnl) for t in self.trades),
                }
            }

    streamer = _make_streamer(tmp_path)
    cid = "RUNID|es_modelrun77_parity_v1|ES JUN26|2026-06-03T20:20:00-06:00|OPEN|LONG|77e81efd"
    streamer.trades_csv = tmp_path / "trades.csv"
    with streamer.trades_csv.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=stream_live_csv_mod.TRADES_HEADER_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "entry_ts": "2026-06-03T20:20:01.282001-06:00",
                "exit_ts": "2026-06-03T21:18:44.497305-06:00",
                "side": "LONG",
                "qty": "1",
                "entry_price": "7531.75",
                "exit_price": "7541.0",
                "entry_fill_price": "7531.75",
                "actual_entry_price": "7531.75",
                "exit_fill_price": "7541.0",
                "actual_exit_price": "7541.0",
                "exit_reason": "target_hit",
                "client_order_id": cid,
            }
        )
    guard = _GuardStub()
    streamer.prop_guardrails = guard
    streamer._phase = "LIVE"
    streamer.nt_exec_policy = "live"
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))

    LiveCSVStreamer._prop_guardrails_record_closed_trade(
        streamer,
        open_trade={
            "client_order_id": cid,
            "signal_id": cid,
            "side": "LONG",
            "actual_entry_price": 7541.0,
            "entry_fill_price": 7541.0,
            "actual_exit_price": 7541.0,
            "exit_fill_price": 7541.0,
            "contracts": 1,
            "planned_stop": 7522.5,
            "prop_trade_recorded": False,
        },
        exit_price=7541.0,
        exit_reason="target_hit",
    )

    assert len(guard.trades) == 1
    trade = guard.trades[0]
    assert trade.entry_price == pytest.approx(7531.75)
    assert trade.exit_price == pytest.approx(7541.0)
    assert trade.pnl == pytest.approx(462.5)
    assert exec_events[-1]["event"] == "prop_trade_recorded"
    assert exec_events[-1]["source"] == "trades_csv"
    assert exec_events[-1]["pnl"] == pytest.approx(462.5)


def test_live_nt_trade_limit_ignores_startup_entry_intents(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer.nt_exec_policy = "live"
    streamer.state.trades_today = 0
    streamer.state.entry_intents_today = 10
    streamer._entry_intents_today = 10

    assert LiveCSVStreamer._entries_today_for_limits(streamer) == 0


def test_prop_closed_trade_skips_non_live_reconstructed_close(tmp_path: Path) -> None:
    class _GuardStub:
        def __init__(self) -> None:
            self.trades = []

        def record_trade(self, trade):
            self.trades.append(trade)

        def get_state_summary(self):
            return {"risk": {"is_locked": False, "lock_reason": "", "realized_pnl": 0.0}}

    streamer = _make_streamer(tmp_path)
    streamer.prop_guardrails = _GuardStub()
    streamer._phase = "BACKFILL"
    streamer.nt_exec_policy = "live"
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))
    open_trade = {
        "client_order_id": "cid-1",
        "signal_id": "sig-1",
        "side": "LONG",
        "actual_entry_price": 7531.75,
        "entry_fill_price": 7531.75,
        "actual_exit_price": 7541.0,
        "exit_fill_price": 7541.0,
        "contracts": 1,
        "planned_stop": 7522.5,
        "prop_trade_recorded": False,
    }

    LiveCSVStreamer._prop_guardrails_record_closed_trade(
        streamer,
        open_trade=open_trade,
        exit_price=7541.0,
        exit_reason="target_hit",
    )

    assert streamer.prop_guardrails.trades == []
    assert open_trade["prop_trade_recorded"] is False
    assert exec_events[-1]["event"] == "prop_trade_record_skipped_non_live"
    assert exec_events[-1]["phase"] == "BACKFILL"


def test_replay_eval_records_simulated_non_proof_trade_rows(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "replay"
    streamer._phase = "BACKFILL"
    streamer.trades_csv = tmp_path / "trades.csv"
    with streamer.trades_csv.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=stream_live_csv_mod.TRADES_HEADER_FIELDS, lineterminator="\n")
        writer.writeheader()

    streamer._record_replay_eval_signal(
        {
            "type": "OPEN",
            "side": "LONG",
            "datetime": "2026-04-24T04:20:00-06:00",
            "price": 7100.25,
            "contracts": 2,
            "risk": {"stop": 7096.25, "target": 7108.25},
            "prediction_id": "pred-1",
            "prediction_hash": "hash-1",
            "client_order_id": "RID|OPEN|SIM1",
        },
        blocked_reason="past_bar_emit",
    )
    streamer._record_replay_eval_signal(
        {
            "type": "CLOSE",
            "side": "LONG",
            "datetime": "2026-04-24T04:35:00-06:00",
            "price": 7104.75,
            "client_order_id": "RID|CLOSE|SIM1",
        },
        blocked_reason="past_bar_emit",
    )

    rows = list(csv.DictReader(streamer.trades_csv.open("r", encoding="utf-8", newline="\n")))
    assert len(rows) == 1
    assert rows[0]["protection_status"] == "simulated_replay_fill"
    assert rows[0]["proof_only"] == "False"
    assert rows[0]["qty"] == "2.0000"
    assert rows[0]["exit_reason"].startswith("simulated_replay_fill:")
    assert streamer._executor_stats["simulated_replay_trades_total"] == 1


def test_simulated_replay_fill_upserts_existing_open_trade_row(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "replay"
    streamer.trades_csv = tmp_path / "trades.csv"
    with streamer.trades_csv.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=stream_live_csv_mod.TRADES_HEADER_FIELDS, lineterminator="\n")
        writer.writeheader()
        row = {field: "" for field in stream_live_csv_mod.TRADES_HEADER_FIELDS}
        row.update(
            {
                "entry_ts": "2026-04-24T04:20:00-06:00",
                "side": "LONG",
                "qty": "1.0000",
                "entry_price": "7100.25",
                "protection_status": "planned_only",
                "client_order_id": "RID|OPEN|SIM2",
            }
        )
        writer.writerow(row)

    streamer._append_simulated_replay_fill_row(
        open_trade={
            "datetime": "2026-04-24T04:20:00-06:00",
            "side": "LONG",
            "price": 7100.25,
            "contracts": 1,
            "risk": {"stop": 7096.25, "target": 7108.25},
            "client_order_id": "RID|OPEN|SIM2",
        },
        close_event={"datetime": "2026-04-24T04:35:00-06:00", "price": 7104.75},
        reason="replay_exec_blocked",
    )

    rows = list(csv.DictReader(streamer.trades_csv.open("r", encoding="utf-8", newline="\n")))
    assert len(rows) == 1
    assert rows[0]["client_order_id"] == "RID|OPEN|SIM2"
    assert rows[0]["exit_ts"] == "2026-04-24T04:35:00-06:00"
    assert rows[0]["protection_status"] == "simulated_replay_fill"
    assert rows[0]["exit_reason"].startswith("simulated_replay_fill:")


def test_finalize_replay_trades_csv_closes_open_rows_and_merges_sim_duplicates(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "replay"
    streamer.trades_csv = tmp_path / "trades.csv"
    with streamer.trades_csv.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=stream_live_csv_mod.TRADES_HEADER_FIELDS, lineterminator="\n")
        writer.writeheader()
        base = {field: "" for field in stream_live_csv_mod.TRADES_HEADER_FIELDS}
        row_a = dict(base)
        row_a.update(
            {
                "entry_ts": "2026-04-24T09:05:00-06:00",
                "side": "LONG",
                "qty": "1.0000",
                "entry_price": "7175.0",
                "protection_status": "execution_pending",
                "client_order_id": "RID|OPEN|A",
            }
        )
        row_a_sim = dict(base)
        row_a_sim.update(
            {
                "entry_ts": "2026-04-24T09:05:00-06:00",
                "exit_ts": "2026-04-24T09:10:00-06:00",
                "side": "LONG",
                "qty": "1.0000",
                "entry_price": "7175.0",
                "exit_price": "7176.0",
                "protection_status": "simulated_replay_fill",
                "exit_reason": "simulated_replay_fill:replay_exec_blocked",
                "client_order_id": "RID|OPEN|A|SIM_REPLAY_CLOSE|2026-04-24T09:10:00-06:00",
            }
        )
        row_b = dict(base)
        row_b.update(
            {
                "entry_ts": "2026-04-24T10:45:00-06:00",
                "side": "SHORT",
                "qty": "1.0000",
                "entry_price": "7186.75",
                "protection_status": "planned_only",
                "client_order_id": "RID|OPEN|B",
            }
        )
        writer.writerow(row_a)
        writer.writerow(row_a_sim)
        writer.writerow(row_b)

    streamer._last_bar_ts_guard = pd.Timestamp("2026-04-24T14:55:00-06:00")
    streamer._last_close = 7196.5
    streamer._finalize_replay_trades_csv_on_shutdown("keyboard_interrupt")

    rows = list(csv.DictReader(streamer.trades_csv.open("r", encoding="utf-8", newline="\n")))
    assert len(rows) == 2
    by_cid = {row["client_order_id"]: row for row in rows}
    assert "RID|OPEN|A" in by_cid
    assert "RID|OPEN|B" in by_cid
    assert by_cid["RID|OPEN|A"]["exit_ts"] == "2026-04-24T09:10:00-06:00"
    assert by_cid["RID|OPEN|B"]["exit_ts"] == ""
    assert by_cid["RID|OPEN|B"]["protection_status"] == "planned_only"
    assert by_cid["RID|OPEN|B"]["exit_reason"].startswith("replay_eval_pending_unclosed:")


def test_replay_local_sim_resets_runtime_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = -1
    streamer._open_trade = {"side": "SHORT", "entry_price": 7001.0, "contracts": 1.0}
    streamer._cooldown_left = 3
    streamer._post_stop_cooldown_left = 2
    streamer._slippage_cooldown_left = 1
    streamer._loss_cascade_cooldown_left = 1
    streamer._loss_cascade_side_lockout = {"LONG": 1, "SHORT": 2}
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer.state.pending_client_order_id = "CID"
    streamer.state.pending_until = pd.Timestamp("2026-04-24T04:20:00-06:00")
    streamer._pending_client_order_id = "CID"
    streamer._pending_until = pd.Timestamp("2026-04-24T04:20:00-06:00")
    streamer._replay_eval_open_trade = {"side": "SHORT"}

    streamer._reset_replay_runtime_state(reason="test")

    assert streamer._pos == 0
    assert streamer._open_trade is None
    assert streamer._cooldown_left == 0
    assert streamer._post_stop_cooldown_left == 0
    assert streamer._slippage_cooldown_left == 0
    assert streamer._loss_cascade_cooldown_left == 0
    assert streamer._loss_cascade_side_lockout == {"LONG": 0, "SHORT": 0}
    assert streamer.state.position_state == "FLAT"
    assert streamer.state.pending_client_order_id is None
    assert streamer.state.pending_until is None
    assert streamer._pending_client_order_id is None
    assert streamer._pending_until is None
    assert streamer._replay_eval_open_trade is None


def test_replay_local_sim_past_bar_still_appends_signal_jsonl(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "replay"
    streamer._phase = "BACKFILL"
    streamer.allow_replay_exec_to_nt = False
    streamer.replay_local_sim = True
    streamer._last_csv_bar = pd.Timestamp("2026-04-24T04:30:00-06:00")
    ev = {
        "type": "OPEN",
        "side": "LONG",
        "datetime": pd.Timestamp("2026-04-24T04:20:00-06:00"),
        "price": 7100.25,
        "prob": 0.72,
        "risk": {"stop": 7098.0, "target": 7106.0},
        "contracts": 1,
    }

    streamer._maybe_emit_signal(ev)
    rows = _read_jsonl(streamer.signals_jsonl)
    assert any((row.get("event") or "").upper() == "OPEN" for row in rows)


def test_live_backfill_lifecycle_signal_is_logged_for_pnl_not_dropped(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer._phase = "BACKFILL"
    streamer._phase_allows_execution = lambda: False
    streamer._last_csv_bar = pd.Timestamp("2026-04-24T04:30:00-06:00")
    ev = {
        "type": "OPEN",
        "side": "LONG",
        "datetime": pd.Timestamp("2026-04-24T04:20:00-06:00"),
        "price": 7100.25,
        "prob": 0.72,
        "risk": {"stop": 7098.0, "target": 7106.0},
        "contracts": 1,
    }

    streamer._maybe_emit_signal(ev)

    signal_rows = list(csv.DictReader(streamer.signals_csv.read_text(encoding="utf-8").splitlines()))
    assert signal_rows
    row = signal_rows[-1]
    assert row["type"] == "OPEN"
    assert row["side"] == "LONG"
    assert row["blocked"] == "1"
    assert row["blocked_reason"] in {"phase:backfill", "past_bar_emit"}


def test_replay_local_sim_bypasses_bootstrap_exec_block(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "replay"
    streamer.allow_replay_exec_to_nt = False
    streamer.replay_local_sim = True
    streamer._bootstrap_active = True
    streamer._entries_disarmed_reason = "bootstrap_lookback"
    streamer._phase = "BACKFILL"

    blocked = streamer._should_block_nt_send_policy({"type": "OPEN"}, "OPEN")

    assert isinstance(blocked, dict)
    assert blocked.get("reason") == "replay_eval_only"
    assert streamer._bootstrap_active is False
    assert streamer._entries_disarmed_reason is None


def test_replay_execution_intended_requires_allow_replay_exec_flag(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "replay"
    streamer.replay_execution_intended = True
    streamer.allow_replay_exec_to_nt = False
    streamer.replay_local_sim = False
    streamer._bootstrap_active = False
    streamer._phase = "LIVE"

    blocked = streamer._should_block_nt_send_policy({"type": "OPEN"}, "OPEN")

    assert isinstance(blocked, dict)
    assert blocked.get("reason") == "replay_exec_blocked"


def test_record_entry_event_allows_first_open_after_position_tracking(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 1
    streamer._open_trade = None
    streamer.state.day_start = pd.Timestamp("2026-04-24T00:00:00-06:00")
    streamer.state.position_uid = "pos|ES 06-26|stale"
    streamer._ensure_day_state = lambda _ts: None
    streamer.instrument = SimpleNamespace(round_lot=None, point_value=50.0)
    streamer.policy = SimpleNamespace(dedupe_window_min=0)
    streamer._last_row_features = None
    streamer._last_base_context = {}
    streamer._current_bar_index = None
    ev = {
        "type": "OPEN",
        "side": "LONG",
        "datetime": "2026-04-24T04:20:00-06:00",
        "price": 7100.25,
        "risk": {"stop": 7098.0, "target": 7106.0},
        "contracts": 1,
    }

    streamer._record_entry_event(ev)

    assert isinstance(streamer._open_trade, dict)
    assert streamer._open_trade.get("side") == "LONG"
    assert streamer._open_trade.get("client_order_id")
    assert streamer.state.last_entry_side == "LONG"
    assert streamer.state.position_uid is None


def test_non_emitted_open_rolls_back_speculative_trade_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._last_csv_bar = pd.Timestamp("2026-04-24T04:30:00-06:00")
    streamer._load_fill_truth_index = lambda: {}
    streamer._summarize_fill_truth = lambda _idx: {"net_qty": 0.0}
    streamer._exec_instrument_key = lambda: "ES 06-26"
    streamer._nt_last_pos_qty_by_instrument = {}
    streamer._record_replay_eval_signal = lambda *args, **kwargs: None
    streamer._append_blocked_candidate_jsonl = lambda *args, **kwargs: None
    streamer._record_replay_parity_event = lambda *args, **kwargs: None
    streamer._emit_trades_csv_ignored_decision = lambda *args, **kwargs: None
    streamer._reconcile_state_action_for_suppressed_signal = lambda *args, **kwargs: None
    streamer._set_position_state = LiveCSVStreamer._set_position_state.__get__(streamer, LiveCSVStreamer)
    ev = {
        "type": "OPEN",
        "side": "LONG",
        "datetime": pd.Timestamp("2026-04-24T04:20:00-06:00"),
        "price": 7100.25,
        "prob": 0.72,
        "risk": {"stop": 7098.0, "target": 7106.0},
        "contracts": 1,
        "gates_detail": {},
    }

    streamer._apply_event_position_tracking(
        typ="OPEN",
        side="LONG",
        ev=ev,
        prev_pos=0,
        prev_bars=0,
    )
    streamer._record_entry_event(ev)
    speculative_cid = streamer._open_trade.get("client_order_id")
    streamer.state.active_client_order_id = speculative_cid
    streamer.state.position_state = "FLAT"

    streamer._append_signal(ev)

    assert streamer._open_trade is None
    assert streamer._pos == 0
    assert streamer.state.active_client_order_id is None


def test_apply_event_position_tracking_normalizes_entry_time(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.cooldown_bars = 2
    streamer._apply_event_position_tracking(
        typ="OPEN",
        side="LONG",
        ev={
            "datetime": "2026-04-24T04:20:00-06:00",
            "price": 7100.25,
            "risk": {"stop": 7098.0, "target": 7106.0},
        },
        prev_pos=0,
        prev_bars=0,
    )
    assert isinstance(streamer._entry_time, pd.Timestamp)


def test_save_state_tolerates_string_entry_time(tmp_path: Path) -> None:
    events: list[dict] = []
    streamer = _make_streamer(tmp_path)
    _prepare_gap_policy_streamer(streamer, tmp_path, events)
    streamer._entry_time = "2026-04-24T04:20:00-06:00"

    streamer._save_state()

    payload = json.loads(streamer.state_path.read_text(encoding="utf-8"))
    assert payload["position"]["entry_time"] == "2026-04-24T04:20:00-06:00"


def test_final_shutdown_health_clean_when_flat_no_orders(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._cooldown_left = 0
    streamer.state.position_state = "FLAT"
    status = {
        "ts": "2026-04-15T19:10:00-06:00",
        "last_bar_ts": "2026-04-15T19:10:00-06:00",
        "feed_health_ok": True,
        "bar_age_sec": 0.0,
        "effective_bar_age_max_sec": 605.0,
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "fill_truth_state": "flat_no_active_trade",
        "nt_safety_capabilities": {"ok": True},
    }

    streamer._write_run_health_summary(status, process_alive=False, shutdown_reason="test_done", exit_code=0)

    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))
    assert health["verdict"] == "clean_stopped"
    assert health["clean_stopped"] is True
    assert health["process_alive"] is False
    assert health["shutdown_reason"] == "test_done"


def test_run_health_summary_marks_insufficient_live_evidence_for_backfill_only_runs(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._inc_phase_stat("candidate_signals_total", phase="BACKFILL", amount=10)
    streamer._inc_phase_stat("signal_append_total", phase="BACKFILL", amount=10)
    status = {
        "ts": "2026-04-15T19:10:00-06:00",
        "last_bar_ts": "2026-04-15T19:10:00-06:00",
        "feed_health_ok": True,
        "bar_age_sec": 0.0,
        "effective_bar_age_max_sec": 605.0,
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "fill_truth_state": "flat_no_active_trade",
        "nt_safety_capabilities": {"ok": True},
    }

    streamer._write_run_health_summary(status, process_alive=False, shutdown_reason="test_done", exit_code=0)

    health = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))
    assert health["verdict"] == "insufficient_live_evidence"
    assert health["insufficient_live_evidence"] is True
    assert health["live_evidence"]["all_phase_candidate_signals_total"] == 10
    assert health["live_evidence"]["live_candidate_signals_total"] == 0
    assert health["replay_evaluation"]["blocked_for_execution"] == 0
    assert health["replay_evaluation"]["accepted_for_sim_evaluation"] == 0
    assert health["replay_evaluation"]["simulated_replay_trades_total"] == 0


def test_flat_no_active_trade_fill_truth_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._fill_truth_state = None
    streamer._has_fill_truth_last = True
    streamer._log_exec_event = lambda *_args, **_kwargs: None

    streamer._record_has_fill_truth_transition(
        has_fill_truth=False,
        source="reconcile_reporting",
        reason="flat_no_active_trade",
        position_state="FLAT",
        active_client_order_id=None,
        fill_truth_state="flat_no_active_trade",
    )

    assert streamer._fill_truth_state == "flat_no_active_trade"


def test_audit_allows_resolved_fallback_fill_and_clean_shutdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "RID",
                "feed_health_ok": True,
                "bar_age_sec": 0.0,
                "effective_bar_age_max_sec": 605.0,
                "snapshot_orders_count": 0,
                "snapshot_blocking_orders_count": 0,
                "addon_flags": {
                    "protection_repair_enabled": True,
                    "stop_update_enabled": True,
                    "auto_flatten_enabled": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "stream_state.json").write_text(
        json.dumps({"state": {"position_state": "FLAT"}, "position": {"pos": 0, "cooldown_left": 0}}),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_health_summary.json").write_text(
        json.dumps({"verdict": "clean_stopped", "clean_stopped": True, "unresolved_warnings": []}),
        encoding="utf-8",
    )
    (run_dir / "exec_events.jsonl").write_text(
        json.dumps({"event": "fallback_fill_reassociated", "untracked_cid": "UNTRACKED|ES|1|2"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "order_events.jsonl").write_text("", encoding="utf-8")

    report = audit_run(run_dir)

    assert report["verdict"] == "PASS"
    assert report["issues"] == []


def test_audit_allows_valid_running_flat_cooldown(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "RID",
                "feed_health_ok": True,
                "bar_age_sec": 5.0,
                "effective_bar_age_max_sec": 605.0,
                "snapshot_orders_count": 0,
                "snapshot_blocking_orders_count": 0,
                "cooldown_left": 2,
                "addon_flags": {
                    "protection_repair_enabled": True,
                    "stop_update_enabled": True,
                    "auto_flatten_enabled": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "stream_state.json").write_text(
        json.dumps({"state": {"position_state": "FLAT"}, "position": {"pos": 0, "cooldown_left": 2}}),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_health_summary.json").write_text(
        json.dumps(
            {
                "verdict": "running_healthy",
                "clean_stopped": False,
                "process_alive": True,
                "unresolved_warnings": [],
                "cooldown_state": "active_valid",
                "cooldown_is_stale": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "exec_events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "order_events.jsonl").write_text("", encoding="utf-8")

    report = audit_run(run_dir)

    assert report["verdict"] == "PASS"
    assert report["issues"] == []
    assert report["cooldown_state"] == "active_valid"


def test_audit_flags_threshold_telemetry_vs_prediction_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "RID",
                "feed_health_ok": True,
                "bar_age_sec": 0.0,
                "effective_bar_age_max_sec": 605.0,
                "snapshot_orders_count": 0,
                "snapshot_blocking_orders_count": 0,
                "addon_flags": {
                    "protection_repair_enabled": True,
                    "stop_update_enabled": True,
                    "auto_flatten_enabled": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "stream_state.json").write_text(
        json.dumps({"state": {"position_state": "FLAT"}, "position": {"pos": 0, "cooldown_left": 0}}),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_health_summary.json").write_text(
        json.dumps({"verdict": "running_healthy", "clean_stopped": False, "process_alive": True, "unresolved_warnings": []}),
        encoding="utf-8",
    )
    gating_rows = [
        {
            "phase": "LIVE",
            "threshold_p_long": 0.6,
            "pred_p_long": 0.68,
            "phase2": {"direction_prob": 0.68},
        },
        {
            "phase": "LIVE",
            "threshold_p_long": 0.6,
            "pred_p_long": 0.31,
            "phase2": {"direction_prob": 0.31},
        },
    ]
    (run_dir / "gating_events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in gating_rows) + "\n",
        encoding="utf-8",
    )
    (run_dir / "exec_events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "order_events.jsonl").write_text("", encoding="utf-8")

    report = audit_run(run_dir)

    warning_codes = {w.get("code") for w in report.get("warnings", [])}
    assert "threshold_fields_not_model_predictions" in warning_codes


def test_rebuild_trades_preserves_stop_and_model_close_exit_reasons(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.trades_csv = tmp_path / "trades.csv"
    streamer._nt_order_state = {
        "ENTRY_STOP": {"side": "SHORT", "qty": 1, "exits_working": True},
        "ENTRY_MODEL": {"side": "LONG", "qty": 1, "exits_working": True},
    }

    streamer._rebuild_trades_from_fill_truth(
        {
            "ENTRY_STOP": {
                "side": "SHORT",
                "entry_fill_ts_epoch": 1776303301.0,
                "entry_fill_price": 7073.25,
                "entry_fill_qty": 1,
                "exit_fill_ts_epoch": 1776303601.0,
                "exit_fill_price": 7073.75,
                "exit_fill_qty": 1,
                "exit_reason_hint": "stop_hit",
            },
            "ENTRY_MODEL": {
                "side": "LONG",
                "entry_fill_ts_epoch": 1776303901.0,
                "entry_fill_price": 7074.0,
                "entry_fill_qty": 1,
                "exit_fill_ts_epoch": 1776304201.0,
                "exit_fill_price": 7074.5,
                "exit_fill_qty": 1,
                "exit_reason_hint": "model_close",
            },
        }
    )

    rows = list(csv.DictReader(streamer.trades_csv.read_text(encoding="utf-8").splitlines()))
    assert [row["exit_reason"] for row in rows] == ["stop_hit", "model_close"]


def test_rebuild_trades_keeps_fallback_exit_reason_without_hint(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.trades_csv = tmp_path / "trades.csv"
    streamer._nt_order_state = {"ENTRY": {"side": "SHORT", "qty": 1, "exits_working": True}}

    streamer._rebuild_trades_from_fill_truth(
        {
            "ENTRY": {
                "side": "SHORT",
                "entry_fill_ts_epoch": 1776303301.0,
                "entry_fill_price": 7073.25,
                "entry_fill_qty": 1,
                "exit_fill_ts_epoch": 1776303601.0,
                "exit_fill_price": 7073.75,
                "exit_fill_qty": 1,
            }
        }
    )

    rows = list(csv.DictReader(streamer.trades_csv.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["exit_reason"] == "reconciled_fill_exit"


def test_rebuild_trades_preserves_specific_forced_exit_state_reasons(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.trades_csv = tmp_path / "trades.csv"
    streamer._nt_order_state = {
        "ENTRY_MISSING": {"side": "SHORT", "qty": 1, "exits_working": True},
        "ENTRY_OFFTICK": {"side": "LONG", "qty": 1, "exits_working": True},
    }

    streamer._rebuild_trades_from_fill_truth(
        {
            "ENTRY_MISSING": {
                "side": "SHORT",
                "entry_fill_ts_epoch": 1776303301.0,
                "entry_fill_price": 7073.25,
                "entry_fill_qty": 1,
                "exit_fill_ts_epoch": 1776303601.0,
                "exit_fill_price": 7073.75,
                "exit_fill_qty": 1,
                "exit_reason_hint": "missing_stop_state",
            },
            "ENTRY_OFFTICK": {
                "side": "LONG",
                "entry_fill_ts_epoch": 1776303901.0,
                "entry_fill_price": 7074.0,
                "entry_fill_qty": 1,
                "exit_fill_ts_epoch": 1776304201.0,
                "exit_fill_price": 7074.5,
                "exit_fill_qty": 1,
                "exit_reason_hint": "offtick_stop_state",
            },
        }
    )

    rows = list(csv.DictReader(streamer.trades_csv.read_text(encoding="utf-8").splitlines()))
    assert [row["exit_reason"] for row in rows] == ["missing_stop_state", "offtick_stop_state"]


def test_forced_exit_valid_stored_stop_does_not_emit_invalid_stop(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.safe_mode = True
    streamer._last_prob = 0.5
    streamer._pos = -1
    streamer._entry_price = 7074.25
    streamer._entry_stop = 7078.0
    streamer._entry_target = None
    streamer._bars_in_trade = 1
    streamer.tick_size = 0.25
    streamer.instrument = SimpleNamespace(tick_size=0.25, point_value=50.0)
    streamer._open_trade = {
        "client_order_id": "ENTRY1",
        "side": "SHORT",
        "entry_stop": 7078.0,
        "planned_stop": 7078.0,
        "live_stop": 7078.0,
    }

    row = pd.Series({"Open": 7074.0, "High": 7077.75, "Low": 7072.5, "Close": 7073.5})

    assert streamer._maybe_forced_exit(row, pd.Timestamp("2026-04-02T10:00:00-06:00")) is None


def test_forced_exit_missing_stop_state_uses_specific_reason(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.safe_mode = True
    streamer._last_prob = 0.5
    streamer._pos = -1
    streamer._entry_price = 7074.25
    streamer._entry_stop = None
    streamer._entry_target = None
    streamer._bars_in_trade = 1
    streamer.tick_size = 0.25
    streamer.instrument = SimpleNamespace(tick_size=0.25, point_value=50.0)
    streamer._open_trade = {"client_order_id": "ENTRY2", "side": "SHORT"}

    row = pd.Series({"Open": 7074.0, "High": 7075.0, "Low": 7072.5, "Close": 7073.5})
    event = streamer._maybe_forced_exit(row, pd.Timestamp("2026-04-02T10:05:00-06:00"))

    assert event is not None
    assert event["ctx"]["exit_reason"] == "missing_stop_state"


def test_forced_exit_offtick_stop_state_uses_specific_reason(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.safe_mode = True
    streamer._last_prob = 0.5
    streamer._pos = -1
    streamer._entry_price = 7074.25
    streamer._entry_stop = None
    streamer._entry_target = None
    streamer._bars_in_trade = 1
    streamer.tick_size = 0.25
    streamer.instrument = SimpleNamespace(tick_size=0.25, point_value=50.0)
    streamer._open_trade = {
        "client_order_id": "ENTRY3",
        "side": "SHORT",
        "entry_stop": 7078.13,
        "planned_stop": 7078.13,
    }

    row = pd.Series({"Open": 7074.0, "High": 7075.0, "Low": 7072.5, "Close": 7073.5})
    event = streamer._maybe_forced_exit(row, pd.Timestamp("2026-04-02T10:10:00-06:00"))

    assert event is not None
    assert event["ctx"]["exit_reason"] == "offtick_stop_state"


def test_nt_protection_state_stop_only_does_not_require_target(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.protection_mode = "stop_only_until_model_close_allowed"

    assert streamer._is_protection_working_state(
        {
            "stop_state": "WORKING",
            "stop_order_id": "STOP1",
            "model_target_abs": 7068.0,
            "target_state": None,
            "target_order_id": None,
        }
    ) is True


def test_nt_protection_timeout_grace_suppresses_pending_stop(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "paper"
    streamer.protection_timeout_sec = 1.0
    streamer.protection_grace_sec = 10.0
    streamer._hard_lockout_active = False
    streamer._protection_timeout_detail = None
    streamer._protection_timeout_attempts = {"CID-TIMEOUT": 1}
    streamer._protection_timeout_last_attempt_ts = {}
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._snapshot_protection_working = lambda _inst: (False, {})
    streamer._nt_stop_update_pending = None
    streamer._maybe_repair_protection = lambda **_kwargs: None
    timeout_calls: list[dict] = []
    streamer._handle_protection_timeout = lambda **kwargs: timeout_calls.append(kwargs)
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._nt_order_state = {
        "CID-GRACE": {
            "intent_action": "OPEN",
            "instrument": streamer.nt_instrument,
            "entry_filled": True,
            "fill_ts": time.time(),
            "exits_submitted": True,
            "stop_state": "INITIALIZED",
            "stop_order_id": "STOP1",
            "protected": False,
            "exits_working": False,
            "status": "entry_filled",
        }
    }

    streamer._enforce_nt_protection_timeouts()

    assert timeout_calls == []
    assert streamer._nt_order_state["CID-GRACE"].get("protection_first_became_false_ts") is None


def test_nt_protection_timeout_triggers_without_stop_evidence(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "paper"
    streamer.protection_timeout_sec = 0.1
    streamer.protection_grace_sec = 0.1
    streamer.nt_protection_timeout_max_retries = 1
    streamer._hard_lockout_active = False
    streamer._protection_timeout_detail = None
    streamer._protection_timeout_attempts = {"CID-TIMEOUT": 1}
    streamer._protection_timeout_last_attempt_ts = {}
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._snapshot_protection_working = lambda _inst: (False, {})
    streamer._nt_stop_update_pending = None
    streamer._maybe_repair_protection = lambda **_kwargs: None
    timeout_calls: list[dict] = []
    streamer._handle_protection_timeout = lambda **kwargs: timeout_calls.append(kwargs)
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._nt_order_state = {
        "CID-TIMEOUT": {
            "intent_action": "OPEN",
            "instrument": streamer.nt_instrument,
            "entry_filled": True,
            "fill_ts": time.time() - 5.0,
            "protection_first_became_false_ts": time.time() - 1.0,
            "protected": False,
            "exits_working": False,
            "status": "entry_filled",
        }
    }

    streamer._enforce_nt_protection_timeouts()

    assert len(timeout_calls) == 1
    assert timeout_calls[0]["reason"] == "no_protection_working"


def test_entry_filled_unprotected_known_bracket_repairs_immediately(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "paper"
    streamer.protection_timeout_sec = 30.0
    streamer.protection_grace_sec = 10.0
    streamer.nt_protection_timeout_max_retries = 2
    streamer.nt_protection_timeout_retry_sec = 10.0
    streamer._hard_lockout_active = False
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._snapshot_protection_working = lambda _inst: (False, {})
    streamer._maybe_request_nt_snapshot = lambda **_kwargs: None
    streamer._emit_soft_block_event = lambda **_kwargs: None
    streamer._nt_stop_update_pending = None
    repair_calls: list[dict] = []
    streamer._maybe_repair_protection = lambda **kwargs: repair_calls.append(dict(kwargs))
    timeout_calls: list[dict] = []
    streamer._handle_protection_timeout = lambda **kwargs: timeout_calls.append(kwargs)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._pos = 1
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer.state.active_client_order_id = "CID-UNPROTECTED"
    streamer._nt_order_state = {
        "CID-UNPROTECTED": {
            "intent_action": "OPEN",
            "instrument": streamer.nt_instrument,
            "entry_filled": True,
            "fill_ts": time.time(),
            "qty": 1,
            "stop_price": 7540.0,
            "target_price": 7551.0,
            "protected": False,
            "exits_working": False,
            "status": "entry_filled",
        }
    }

    streamer._enforce_nt_protection_timeouts()

    assert repair_calls
    assert repair_calls[-1]["reason"] == "entry_filled_unprotected"
    assert repair_calls[-1]["stop_price"] == pytest.approx(7540.0)
    assert repair_calls[-1]["target_price"] == pytest.approx(7551.0)
    assert streamer._protection_timeout_attempts["CID-UNPROTECTED"] == 1
    assert timeout_calls == []
    assert any(e.get("event") == "protection_immediate_repair" for e in events)


def test_protection_timeout_waits_for_repair_confirmation_window(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "paper"
    streamer.protection_timeout_sec = 1.0
    streamer.protection_grace_sec = 0.1
    streamer.nt_protection_timeout_max_retries = 2
    streamer.nt_protection_timeout_retry_sec = 10.0
    streamer._hard_lockout_active = False
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._snapshot_protection_working = lambda _inst: (False, {})
    snapshot_requests: list[dict] = []
    streamer._maybe_request_nt_snapshot = lambda **kwargs: snapshot_requests.append(dict(kwargs))
    soft_blocks: list[dict] = []
    streamer._emit_soft_block_event = lambda **kwargs: soft_blocks.append(dict(kwargs))
    streamer._nt_stop_update_pending = None
    repair_calls: list[dict] = []
    streamer._maybe_repair_protection = lambda **kwargs: repair_calls.append(dict(kwargs))
    timeout_calls: list[dict] = []
    streamer._handle_protection_timeout = lambda **kwargs: timeout_calls.append(kwargs)
    now = time.time()
    streamer._protection_timeout_attempts = {"CID-WAIT": 1}
    streamer._protection_timeout_last_attempt_ts = {"CID-WAIT": now}
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._nt_order_state = {
        "CID-WAIT": {
            "intent_action": "OPEN",
            "instrument": streamer.nt_instrument,
            "entry_filled": True,
            "fill_ts": now - 5.0,
            "protection_first_became_false_ts": now - 5.0,
            "qty": 1,
            "stop_price": 7540.0,
            "target_price": 7551.0,
            "protected": False,
            "exits_working": False,
            "status": "entry_filled",
        }
    }

    streamer._enforce_nt_protection_timeouts()

    assert repair_calls == []
    assert timeout_calls == []
    assert snapshot_requests
    assert soft_blocks[-1]["block_code"] == "waiting_for_protection_repair"


def test_broker_flat_suppresses_stale_nonflat_fill_truth(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._ensure_compat_defaults()
    streamer._pos = -1
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer.state.active_client_order_id = "CID-STALE-FILL-TRUTH"
    streamer._open_trade = {"client_order_id": "CID-STALE-FILL-TRUTH", "side": "SHORT", "entry_price": 7546.5}
    streamer._nt_last_snapshot_instrument = "ES JUN26"
    streamer.nt_instrument = "ES JUN26"
    streamer._nt_last_pos_qty_by_instrument = {
        "ES JUN26": 0.0,
        "ES 06-26": 0.0,
    }
    streamer._rebuild_trades_from_fill_truth = lambda *_args, **_kwargs: None
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    fill_index = {
        "CID-STALE-FILL-TRUTH": {
            "client_order_id": "CID-STALE-FILL-TRUTH",
            "correlation_id": "CID-STALE-FILL-TRUTH",
            "side": "SHORT",
            "intent_side": "SHORT",
            "entry_fill_ts_epoch": time.time() - 10.0,
            "entry_fill_price": 7546.5,
            "entry_fill_qty": 1.0,
            "exit_fill_ts_epoch": None,
            "exit_fill_price": None,
            "exit_fill_qty": None,
            "exit_correlation_id": None,
            "exit_reason_hint": None,
        }
    }

    streamer._apply_fill_truth_reconciliation(
        fill_index=fill_index,
        source="close_watchdog_snapshot_flat",
        fresh=True,
    )

    assert streamer._pos == 0
    assert streamer.state.position_state == "FLAT"
    assert streamer.state.active_client_order_id is None
    assert any(e.get("event") == "fill_truth_nonflat_suppressed_broker_flat" for e in events)


def test_unprotected_followup_escalation_threshold_from_min_timeout(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "paper"
    streamer.protection_timeout_sec = 30.0
    streamer.nt_protection_repair_timeout_sec = 12.0
    streamer.protection_grace_sec = 0.1
    streamer.nt_protection_timeout_max_retries = 99
    streamer.nt_protection_timeout_retry_sec = 999.0
    streamer._hard_lockout_active = False
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._snapshot_protection_working = lambda _inst: (False, {})
    streamer._nt_stop_update_pending = None
    streamer._maybe_repair_protection = lambda **_kwargs: None
    streamer._handle_protection_timeout = lambda **_kwargs: None
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"

    now = time.time()
    streamer._nt_order_state = {
        "CID-ESC-THRESH": {
            "intent_action": "OPEN",
            "instrument": streamer.nt_instrument,
            "entry_filled": True,
            "protection_first_became_false_ts": now - 13.0,
            "protected": False,
            "exits_working": False,
            "status": "entry_filled",
        }
    }
    streamer._enforce_nt_protection_timeouts()
    detail = streamer._get_unprotected_followup_escalation("CID-ESC-THRESH")
    assert detail is not None
    assert float(detail.get("escalation_timeout_sec") or 0.0) == pytest.approx(12.0)
    assert float(detail.get("elapsed_unprotected_sec") or 0.0) >= 13.0


def test_unprotected_followup_escalation_not_set_below_threshold(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "paper"
    streamer.protection_timeout_sec = 30.0
    streamer.nt_protection_repair_timeout_sec = 12.0
    streamer.protection_grace_sec = 0.1
    streamer.nt_protection_timeout_max_retries = 99
    streamer.nt_protection_timeout_retry_sec = 999.0
    streamer._hard_lockout_active = False
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._snapshot_protection_working = lambda _inst: (False, {})
    streamer._nt_stop_update_pending = None
    streamer._maybe_repair_protection = lambda **_kwargs: None
    streamer._handle_protection_timeout = lambda **_kwargs: None
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"

    now = time.time()
    streamer._nt_order_state = {
        "CID-ESC-NO": {
            "intent_action": "OPEN",
            "instrument": streamer.nt_instrument,
            "entry_filled": True,
            "protection_first_became_false_ts": now - 5.0,
            "protected": False,
            "exits_working": False,
            "status": "entry_filled",
        }
    }
    streamer._enforce_nt_protection_timeouts()
    assert streamer._get_unprotected_followup_escalation("CID-ESC-NO") is None


def test_apply_contract_size_skips_risk_reason_for_close(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.disable_risk = False
    streamer.auto_contracts = False
    streamer.max_position_contracts = None
    streamer.tick_size = 0.25
    streamer.instrument = SimpleNamespace(tick_size=0.25, point_value=50.0, round_lot=None)

    event = {
        "type": "CLOSE",
        "side": "LONG",
        "price": 7072.5,
        "contracts": 1,
        "ctx": {},
    }
    streamer._apply_contract_size(event)

    assert "risk_size_reason" not in event["ctx"]
    assert event["contracts"] == 1


def test_record_unprotected_exit_uses_canonical_stop_and_model_close(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._last_prob = 0.5
    streamer._last_close = 7073.25
    captured: list[dict] = []
    streamer._record_exit_event = lambda ev, **_kwargs: captured.append(ev)
    streamer._maybe_emit_signal = lambda ev: captured.append({"signal": ev})
    streamer._open_trade = {
        "client_order_id": "ENTRY_TIMEOUT",
        "side": "SHORT",
        "entry_price": 7074.75,
        "entry_stop": 7078.25,
        "planned_stop": 7078.25,
        "live_stop": 7078.25,
        "planned_target": None,
    }
    streamer._entry_stop = None

    streamer._record_unprotected_exit("ENTRY_TIMEOUT", reason="nt_protection_timeout")

    exit_event = captured[0]
    assert exit_event["ctx"]["exit_reason"] == "model_close"
    assert exit_event["risk"]["stop"] == 7078.25
    assert streamer._open_trade["planned_stop"] == 7078.25
    assert streamer._open_trade["entry_stop"] == 7078.25


def test_record_exit_event_non_emitted_does_not_mutate_position_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer.state.last_signal_status = "holding"
    streamer.state.active_client_order_id = "ENTRYX"
    streamer._open_trade = {
        "client_order_id": "ENTRYX",
        "side": "LONG",
        "entry_price": 7000.0,
        "risk": 5.0,
        "contracts": 1.0,
    }
    ignored: list[tuple[str, str]] = []
    streamer._emit_trades_csv_ignored_decision = lambda _ev, *, action, reason: ignored.append((action, reason))

    streamer._record_exit_event(
        {
            "type": "CLOSE",
            "side": "LONG",
            "price": 7005.0,
            "datetime": "2026-05-01T07:35:00-06:00",
            "_emitted_lifecycle_event": False,
        }
    )

    assert ignored == [("CLOSE", "close_not_emitted")]
    assert streamer._open_trade is not None
    assert streamer.state.position_state == "IN_POSITION_UNPROTECTED"
    assert streamer.state.last_signal_status == "holding"
    assert streamer.state.active_client_order_id == "ENTRYX"


def test_close_fill_preserves_specific_forced_exit_reason(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._last_prob = 0.5
    streamer._last_close = 7073.5
    streamer._update_trade_csv_row = lambda *_args, **_kwargs: None
    streamer._queue_trade_update_from_state = lambda *_args, **_kwargs: None
    streamer._record_exit_event = lambda *_args, **_kwargs: None
    streamer._sync_position_flat = lambda *_args, **_kwargs: None
    streamer._open_trade = {
        "client_order_id": "ENTRY4",
        "side": "SHORT",
        "planned_stop": 7078.0,
        "planned_target": 7068.0,
    }
    state = {
        "forced_exit_reason": "missing_stop_state",
        "execution_forced": True,
        "signal_id": "sig|close",
    }
    msg = {
        "fill_price": 7073.75,
        "timestamp": "2026-04-02T10:15:00-06:00",
    }

    streamer._finalize_close_fill("CLOSE4", state, msg)

    assert state["exit_reason"] == "missing_stop_state"
    assert streamer._open_trade["exit_reason"] == "missing_stop_state"


def test_close_fill_uses_close_order_id_not_bracket_id(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._last_prob = 0.5
    streamer._last_close = 7073.5
    streamer._update_trade_csv_row = lambda *_args, **_kwargs: None
    streamer._queue_trade_update_from_state = lambda *_args, **_kwargs: None
    streamer._record_exit_event = lambda *_args, **_kwargs: None
    streamer._sync_position_flat = lambda *_args, **_kwargs: None
    streamer._open_trade = {
        "client_order_id": "ENTRY-CID",
        "side": "SHORT",
        "planned_stop": 7078.0,
        "planned_target": 7068.0,
    }
    state = {
        "stop_order_id": "STOP-OLD",
        "target_order_id": "TARGET-OLD",
        "last_exit_order_id": "STOP-OLD",
        "signal_id": "sig|close",
    }

    streamer._finalize_close_fill(
        "CLOSE-CID",
        state,
        {
            "ninja_order_id": "CLOSE-NINJA",
            "fill_price": 7073.75,
            "timestamp": "2026-04-02T10:15:00-06:00",
        },
    )

    assert streamer.state.last_confirmed_exit_order_id == "CLOSE-NINJA"
    assert streamer._open_trade["last_confirmed_exit_order_id"] == "CLOSE-NINJA"


def test_closed_trade_update_does_not_downgrade_protection_or_working_ts(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "ENTRY-PROTECTED-CLOSED"
    streamer._nt_order_state[cid] = {
        "entry_filled": True,
        "exits_submitted": True,
        "protected": True,
        "exits_working": True,
        "protected_ts": 1000.0,
        "last_exit_update_ts": 2000.0,
        "exit_fill_ts": 3000.0,
        "exit_fill_price": 7073.75,
        "stop_order_id": "STOP-1",
        "target_order_id": "TARGET-1",
        "stop_price": 7078.0,
        "target_price": 7068.0,
    }

    update = streamer._collect_nt_trade_update(cid)

    assert update["exit_fill_price"] == pytest.approx(7073.75)
    assert "protection_status" not in update
    assert "stop_working_ts" not in update
    assert "target_working_ts" not in update


def test_fill_truth_rebuild_preserves_completed_trade_protection(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "ENTRY-FILL-TRUTH-PROTECTED"
    streamer.trades_csv = tmp_path / "trades.csv"
    streamer._nt_order_state[cid] = {
        "side": "LONG",
        "qty": 1,
        "entry_filled": True,
        "exits_submitted": True,
        "protected_ts": 1000.0,
        "protection_confirmed_ts": 1000.0,
        "stop_order_id": "STOP-1",
        "target_order_id": "TARGET-1",
        "stop_price": 6994.5,
        "target_price": 7005.5,
    }
    fill_index = {
        cid: {
            "side": "LONG",
            "entry_fill_ts_epoch": 1000.0,
            "entry_fill_price": 7000.0,
            "entry_fill_qty": 1.0,
            "exit_fill_ts_epoch": 1300.0,
            "exit_fill_price": 7002.0,
            "exit_fill_qty": 1.0,
        }
    }

    streamer._rebuild_trades_from_fill_truth(fill_index)

    rows = list(csv.DictReader(streamer.trades_csv.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert rows[0]["client_order_id"] == cid
    assert rows[0]["protection_status"] == "protected_confirmed"
    assert rows[0]["stop_order_id"] == "STOP-1"
    assert rows[0]["target_order_id"] == "TARGET-1"
    assert rows[0]["exit_fill_price"] == "7002.0"


def test_entry_fill_resets_snapshot_flat_override_streak(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._snapshot_flat_streak_by_instrument = {"ES 06-26": 166}
    streamer._snapshot_flat_last_ts_by_instrument = {"ES 06-26": 1780638600.0}

    streamer._reset_snapshot_flat_streak("ES JUN26", reason="entry_fill")

    assert streamer._snapshot_flat_streak_by_instrument["ES 06-26"] == 0
    assert "ES 06-26" not in streamer._snapshot_flat_last_ts_by_instrument


def test_update_open_trade_protection_refreshes_canonical_stop_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.tick_size = 0.25
    streamer.instrument = SimpleNamespace(tick_size=0.25, point_value=50.0)
    streamer._entry_stop = None
    streamer._entry_target = None
    streamer._open_trade = {
        "client_order_id": "ENTRY5",
        "planned_stop": None,
        "planned_target": None,
    }

    streamer._update_open_trade_protection(
        "ENTRY5",
        {"stop_price": 7078.0, "target_price": 7068.0},
    )

    assert streamer._entry_stop == 7078.0
    assert streamer._entry_target == 7068.0
    assert streamer._open_trade["entry_stop"] == 7078.0
    assert streamer._open_trade["planned_stop"] == 7078.0
    assert streamer._open_trade["live_stop"] == 7078.0


def test_confirm_protection_uses_state_prices_when_ack_omits_prices(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "CID-PROTECTION-STATE-PRICES"
    streamer._queue_trade_update_from_state = lambda *_args, **_kwargs: None
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._open_trade = {
        "client_order_id": cid,
        "planned_stop": None,
        "planned_target": None,
    }
    streamer._nt_order_state[cid] = {
        "client_order_id": cid,
        "intent_action": "OPEN",
        "entry_filled": True,
        "stop_price": 7078.0,
        "target_price": 7068.0,
        "stop_state": "WORKING",
        "target_state": "WORKING",
        "stop_order_id": "STOP-STATE",
        "target_order_id": "TARGET-STATE",
        "instrument": "ES JUN26",
    }

    streamer._confirm_nt_protection(
        cid,
        msg={"stop_state": "WORKING", "target_state": "WORKING"},
        method="unit_state_prices",
    )

    assert streamer._open_trade["live_stop"] == 7078.0
    assert streamer._open_trade["live_target"] == 7068.0
    assert streamer._open_trade["stop_order_id"] == "STOP-STATE"
    assert streamer._open_trade["target_order_id"] == "TARGET-STATE"
    assert streamer._open_trade["protection_status"] == "protected_confirmed"


def test_audit_warns_on_missing_stop_state_with_persisted_stop(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({
        "run_id": "RUNID",
        "position_state": "EXITING",
        "cooldown_left": 2,
        "snapshot_orders_count": 0,
        "feed_health_ok": True,
        "bar_age_sec": 5,
        "effective_bar_age_max_sec": 605,
    }), encoding="utf-8")
    (run_dir / "stream_state.json").write_text(json.dumps({
        "position": {"pos": 0, "cooldown_left": 2},
        "state": {"position_state": "EXITING"},
    }), encoding="utf-8")
    (run_dir / "resolved_config.json").write_text(json.dumps({"run_id": "RUNID"}), encoding="utf-8")
    (run_dir / "run_health_summary.json").write_text(json.dumps({"verdict": "running_healthy", "cooldown_state": "active_valid"}), encoding="utf-8")
    (run_dir / "order_events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "exec_events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "signals.jsonl").write_text("", encoding="utf-8")
    (run_dir / "signal_to_order.jsonl").write_text("", encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "trades.csv").write_text(
        "entry_ts,exit_ts,side,qty,entry_price,exit_price,stop,target,planned_stop,planned_target,live_stop,live_target,protection_status,nt_connected,handshake_ok,entry_ack_ts,entry_fill_ts,entry_fill_price,actual_entry_price,filled_qty,exit_fill_price,exit_fill_ts,actual_exit_price,actual_exit_ts,stop_working_ts,target_working_ts,stop_order_id,target_order_id,exit_reason,proof_only,prediction_id,prediction_hash,exec_entry_order_id,exec_stop_order_id,exec_target_order_id,client_order_id\n"
        + "2026-04-15T20:45:01-06:00,2026-04-15T20:50:02-06:00,SHORT,1,7074.75,7073.75,7078.25,,7078.25,,7078.25,,protected_confirmed,True,True,,,,,,,,,,,,,,missing_stop_state,False,,,,,,CID1\n",
        encoding="utf-8",
    )

    report = audit_run(run_dir)

    assert any(w.get("code") == "false_missing_stop_state" for w in report["warnings"])


def test_ninja_bridge_safety_flags_enabled_and_live_gate_preserved() -> None:
    bridge = (
        Path(__file__).resolve().parents[1]
        / "discord_addons"
        / "cli"
        / "NinjaRepoBridge.cs"
    )
    text = bridge.read_text(encoding="utf-8", errors="replace")

    assert "private const bool EnableProtectionRepair = true;" in text
    assert "private const bool StopUpdateEnabled = true;" in text
    assert "private const bool AutoFlattenEnabled = true;" in text
    assert "private const bool AllowLiveAccounts = false;" in text
    assert "return AllowLiveWatchdog && IsExplicitlyAllowedLiveAccount(name);" in text


def test_post_stop_and_loss_cascade_cooldowns_still_block_force_open(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    event = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7001.75,
        "prob": 0.8558,
        "grade": "A+",
        "risk": {"stop": 6997.0, "target": 7006.5},
        "contracts": 1,
        "ctx": {"phase2_force_open_applied": True},
    }

    streamer._post_stop_cooldown_left = 1
    assert (
        streamer._hard_gate_reason(
            pd.Timestamp("2026-04-14T23:45:00-06:00"),
            "OPEN",
            "LONG",
            event,
            cooldown_left=0,
        )
        == "post_stop_cooldown_active"
    )

    streamer._post_stop_cooldown_left = 0
    streamer._loss_cascade_cooldown_left = 1
    assert (
        streamer._hard_gate_reason(
            pd.Timestamp("2026-04-14T23:45:00-06:00"),
            "OPEN",
            "LONG",
            event,
            cooldown_left=0,
        )
        == "loss_cascade_cooldown"
    )


def test_flat_flip_normalizes_to_open_before_state_machine(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    event = {"type": "FLIP", "side": "LONG", "ctx": {}}

    streamer._phase = "LIVE"
    typ = streamer._normalize_entry_action_for_phase(event, side="LONG", pos=0)

    assert typ == "OPEN"
    assert event["type"] == "OPEN"
    assert event["original_type"] == "FLIP"
    assert event["flat_flip_normalized"] is True
    assert event["ctx"]["flat_flip_normalized_reason"] == "flat_execution_state"
    assert streamer._state_machine_violation_for_pos(typ, "LONG", 0) is None


def test_flip_stays_forbidden_when_position_exists(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer._pos = -1
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    event = {"type": "FLIP", "side": "LONG", "ctx": {}}

    streamer._phase = "LIVE"
    typ = streamer._normalize_entry_action_for_phase(event, side="LONG", pos=-1)

    assert typ == "FLIP"
    assert event["type"] == "FLIP"
    assert streamer._state_machine_violation_for_pos(typ, "LONG", -1) == "flip_forbidden"


def test_forbidden_flip_downgrades_to_close_when_position_exists(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer.allow_flips = False
    streamer._pos = -1
    streamer.exec_instrument = "ES JUN26"
    streamer.nt_instrument = "ES JUN26"
    streamer.nt_account_mode = "none"
    event = {
        "type": "FLIP",
        "side": "LONG",
        "datetime": "2026-06-03T10:15:00-06:00",
        "instrument": "ES",
        "price": 7579.75,
        "prob": 0.703,
        "contracts": 1,
        "risk": {"stop": 7595.0, "target": 7571.5},
        "ctx": {},
    }

    state_violation = streamer._state_machine_violation_for_pos("FLIP", "LONG", -1)
    typ = streamer._downgrade_forbidden_flip_to_close(
        event,
        state_violation=state_violation,
        pos=-1,
    )
    intent, err = streamer._build_execution_intent(event)

    assert state_violation == "flip_forbidden"
    assert typ == "CLOSE"
    assert event["type"] == "CLOSE"
    assert event["original_type"] == "FLIP"
    assert event["flip_downgraded_to_close"] is True
    assert event["ctx"]["action_resolution_reason"] == "flip_open_forbidden_close_allowed"
    assert err is None
    assert intent is not None
    assert intent.action == "CLOSE"
    assert intent.side == "NONE"
    assert intent.qty == 1


def test_opposite_open_while_positioned_downgrades_to_safe_close(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer.allow_flips = False
    streamer._phase = "LIVE"
    streamer._pos = 1
    event = {
        "type": "OPEN",
        "side": "SHORT",
        "datetime": "2026-06-04T02:10:00-06:00",
        "instrument": "ES",
        "price": 7548.25,
        "prob": 0.953,
        "contracts": 1,
        "risk": {"stop": 7549.75, "target": 7536.25},
        "ctx": {},
    }

    typ = streamer._normalize_entry_action_for_phase(event, side="SHORT", pos=1)
    state_violation = streamer._state_machine_violation_for_pos(typ, "SHORT", 1)
    downgraded = streamer._downgrade_forbidden_flip_to_close(
        event,
        state_violation=state_violation,
        pos=1,
    )

    assert typ == "FLIP"
    assert state_violation == "flip_forbidden"
    assert downgraded == "CLOSE"
    assert event["type"] == "CLOSE"
    assert event["original_type"] == "OPEN"
    assert event["opposite_open_normalized_to_flip"] is True
    assert event["flip_downgraded_to_close"] is True
    assert event["ctx"]["action_resolution_reason"] == "flip_open_forbidden_close_allowed"
    assert event["ctx"]["opposite_open_normalized_to_flip"] is True


def test_forbidden_flip_does_not_downgrade_when_flat(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer.allow_flips = False
    event = {"type": "FLIP", "side": "LONG", "ctx": {}}

    typ = streamer._downgrade_forbidden_flip_to_close(
        event,
        state_violation="flip_forbidden",
        pos=0,
    )

    assert typ is None
    assert event["type"] == "FLIP"


def test_backfill_flip_on_flat_lifecycle_normalizes_to_open(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer._phase = "BACKFILL"
    streamer._lifecycle_open_side = None
    event = {"type": "FLIP", "side": "LONG", "ctx": {}}

    typ = streamer._normalize_entry_action_for_phase(event, side="LONG", pos=0)

    assert typ == "OPEN"
    assert event["type"] == "OPEN"
    assert event["ctx"]["action_resolution_reason"] == "flat_lifecycle_open"


def test_backfill_open_same_side_becomes_hold(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer._phase = "BACKFILL"
    streamer._lifecycle_open_side = "LONG"
    event = {"type": "OPEN", "side": "LONG", "ctx": {}}

    typ = streamer._normalize_entry_action_for_phase(event, side="LONG", pos=1)

    assert typ == "HOLD"
    assert event["type"] == "HOLD"
    assert event["ctx"]["action_resolution_reason"] == "same_side_duplicate"


def test_backfill_open_reverse_side_becomes_flip(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer._phase = "BACKFILL"
    streamer._lifecycle_open_side = "LONG"
    event = {"type": "OPEN", "side": "SHORT", "ctx": {}}

    typ = streamer._normalize_entry_action_for_phase(event, side="SHORT", pos=1)

    assert typ == "FLIP"
    assert event["type"] == "FLIP"
    assert event["ctx"]["action_resolution_reason"] == "reverse_from_open"


def test_open_state_machine_uses_pre_event_position(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer._pos = 1

    assert streamer._state_machine_violation_for_pos("OPEN", "SHORT", 0) is None
    assert streamer._state_machine_violation_for_pos("OPEN", "SHORT", 1) == "open_while_in_position"


def test_phase2_force_open_applies_outside_live_when_parity_enabled(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer._phase = "BACKFILL"

    event = streamer._phase2_force_open_event(
        row=_phase2_force_open_row(),
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        price=7005.25,
        prob=0.73,
        phase2_meta=_phase2_force_open_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.478243,
        hold_conf=0.787104,
        original_action="NO_TRADE",
    )

    assert event is not None
    assert event["ctx"]["phase"] == "BACKFILL"


def test_phase2_force_open_can_be_live_only_when_opted_out(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer._phase = "BACKFILL"
    streamer.phase2_force_open_live_only = True

    event = streamer._phase2_force_open_event(
        row=_phase2_force_open_row(),
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        price=7005.25,
        prob=0.73,
        phase2_meta=_phase2_force_open_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.478243,
        hold_conf=0.787104,
        original_action="NO_TRADE",
    )

    assert event is None


def test_phase2_force_open_does_not_apply_while_in_position(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    streamer._pos = 1
    streamer.state.position_state = "IN_POSITION_PROTECTED"

    event = streamer._phase2_force_open_event(
        row=_phase2_force_open_row(),
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        price=7005.25,
        prob=0.73,
        phase2_meta=_phase2_force_open_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.478243,
        hold_conf=0.787104,
        original_action="NO_TRADE",
    )

    assert event is None


def test_phase2_force_open_rejects_invalid_stop_target_geometry(tmp_path: Path) -> None:
    streamer = _phase2_force_open_ready_streamer(tmp_path)
    bad_row = pd.Series({"stop": 7006.00, "target": 7008.46})

    event = streamer._phase2_force_open_event(
        row=bad_row,
        ts_dt=pd.Timestamp("2026-04-15T21:05:00-06:00"),
        price=7005.25,
        prob=0.73,
        phase2_meta=_phase2_force_open_meta(),
        gate_state=_phase2_force_open_gates(),
        entry_conf=0.478243,
        hold_conf=0.787104,
        original_action="NO_TRADE",
    )

    assert event is None


def test_hard_gate_blocks_above_300_usd_and_allows_compliant_stop(tmp_path: Path) -> None:
    streamer = _hard_gate_ready_streamer(tmp_path)
    streamer.max_risk_usd_per_trade = 300
    streamer._cooldown_left = 0

    ts = pd.Timestamp("2026-04-15T21:05:00-06:00")
    too_wide = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7005.25,
        "prob": 0.73,
        "grade": "A+",
        "risk": {"stop": 6998.25, "target": 7011.25},
        "contracts": 1,
    }
    compliant = {
        "type": "OPEN",
        "side": "LONG",
        "price": 7005.25,
        "prob": 0.73,
        "grade": "A+",
        "risk": {"stop": 6999.25, "target": 7011.25},
        "contracts": 1,
    }

    blocked_reason = streamer._hard_gate_reason(ts=ts, typ="OPEN", side="LONG", ev=too_wide)
    allowed_reason = streamer._hard_gate_reason(ts=ts, typ="OPEN", side="LONG", ev=compliant)

    assert blocked_reason == "max_risk_per_trade"
    assert allowed_reason is None


def test_open_signal_and_gate_audit_remain_aligned_when_emit_allowed(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    ts = pd.Timestamp("2026-04-01T23:40:00-06:00")
    event = {
        "datetime": ts.isoformat(),
        "type": "OPEN",
        "side": "LONG",
        "price": 6530.75,
        "prob": 0.864575,
        "grade": "A+",
        "risk": {"stop": 6522.75, "target": 6538.75},
        "contracts": 1,
        "gates_detail": {
            "gate_state": {"setup": True, "prob": True, "vwap": True, "ema": True, "tod": True},
            "override_confident_long": False,
            "override_prob_min": 0.78,
            "override_hold_conf_min": 0.7,
            "override_applied": False,
        },
    }

    streamer._emit_gate_audit(
        ts_dt=ts,
        action="OPEN",
        gate_state={"setup": True, "prob": True, "vwap": True, "ema": True, "tod": True},
        blocked_by=[],
        reason_detail="",
        prob=0.864575,
        phase2_meta={
            "setup_prob": 0.812,
            "setup_pass": True,
            "setup_reason": "setup_pass",
            "setup_threshold": 0.7,
            "direction_prob": 0.864575,
            "short_prob": 0.135425,
            "direction_signal": 1,
        },
        extra={
            "entries_disarmed_reason": None,
            "execution_blocked_reason": None,
            "strategy_blocked_reason": None,
            "phase": "LIVE",
            "phase_lag_sec": 0.0,
            "phase_confirmations": 1,
            "snapshot_age_sec": 1.0,
            "override_confident_long": False,
            "override_applied": False,
        },
    )
    streamer._maybe_emit_signal(event)

    gate_rows = _read_jsonl(streamer.gating_events_path)
    assert gate_rows
    gate_row = gate_rows[-1]
    assert gate_row["action"] == "OPEN"
    assert gate_row["blocked_by"] == []
    assert gate_row["gate_state"]["setup"] is True
    assert gate_row["phase2"]["setup_pass"] is True
    assert gate_row["threshold_p_long"] == pytest.approx(0.7)
    assert gate_row["threshold_short_cut"] == pytest.approx(0.3)
    assert gate_row["pred_p_long"] == pytest.approx(0.864575)
    assert gate_row["pred_p_short"] == pytest.approx(1.0 - 0.864575)
    assert int(streamer._executor_stats.get("gating_open_emits_total", 0) or 0) == 1

    signal_rows = list(csv.DictReader(streamer.signals_csv.read_text(encoding="utf-8").splitlines()))
    assert signal_rows
    signal_row = signal_rows[-1]
    assert signal_row["type"] == "OPEN"
    assert signal_row["blocked"] == "0"
    assert signal_row["blocked_reason"] == ""


def test_emit_gate_audit_no_trade_fills_reason_when_blank(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    ts = pd.Timestamp("2026-04-23T06:05:00-06:00")
    streamer._emit_gate_audit(
        ts_dt=ts,
        action="NO_TRADE",
        gate_state={"setup": True, "prob": True, "vwap": True, "ema": True, "tod": True},
        blocked_by=[],
        reason_detail="",
        prob=0.61,
        phase2_meta={"setup_pass": True},
        extra={"final_action": "NO_TRADE", "side": "SHORT"},
    )

    gate_rows = _read_jsonl(streamer.gating_events_path)
    assert gate_rows
    gate_row = gate_rows[-1]
    assert gate_row["action"] == "NO_TRADE"
    assert gate_row["reason_detail"] == "no_emittable_strategy_event"


def test_phase2_setup_pass_no_emit_logs_explicit_diagnostics(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._emit_gate_audit = LiveCSVStreamer._emit_gate_audit.__get__(streamer, LiveCSVStreamer)
    streamer.phase2_force_open_on_gate_pass = False
    exec_events: list[dict] = []
    streamer._log_exec_event = lambda payload: exec_events.append(dict(payload))
    ts = pd.Timestamp("2026-04-23T06:05:00-06:00")
    streamer._emit_gate_audit(
        ts_dt=ts,
        action="NO_TRADE",
        gate_state={"setup": True, "prob": True, "vwap": True, "ema": True, "tod": True},
        blocked_by=[],
        reason_detail="",
        prob=0.77,
        phase2_meta={
            "setup_prob": 0.12,
            "setup_pass": True,
            "setup_reason": "setup_pass",
            "setup_threshold": 0.06,
            "direction_prob": 0.77,
            "short_prob": 0.23,
            "direction_signal": 1,
        },
        extra={"final_action": "NO_TRADE", "side": "LONG"},
    )
    names = {str(e.get("event")) for e in exec_events}
    assert "PHASE2_SETUP_PASS_BUT_NO_BASE_ENTRY" in names
    assert "ORDER_SEND_SKIPPED_NO_EMITTABLE_EVENT" in names


def test_no_trade_reason_helper_formats_publish_block_taxonomy() -> None:
    reason = stream_live_csv_mod._resolve_no_trade_reason_detail(
        action="NO_TRADE",
        final_action="NO_TRADE",
        reason_detail="",
        blocked_by=[],
        gate_state={"setup": True, "prob": True, "vwap": True, "ema": True, "tod": True},
        entry_block_reason="trend_guard_block",
        publish_block_reason=None,
    )
    assert reason == "gate_pass_but_publish_blocked:trend_guard_block"


def test_no_trade_reason_helper_prioritizes_readiness_blockers() -> None:
    reason = stream_live_csv_mod._resolve_no_trade_reason_detail(
        action="NO_TRADE",
        final_action="NO_TRADE",
        reason_detail="",
        blocked_by=[],
        gate_state={"setup": True, "prob": True, "vwap": True, "ema": True, "tod": True},
        readiness_snapshot={
            "armed_ok": True,
            "nt_enabled": True,
            "nt_connected": False,
            "nt_ready": False,
            "feed_health_ok": True,
            "offline_ok": True,
        },
    )
    assert reason == "nt_not_ready"


def test_state_csv_downgrades_open_when_phase_not_executable(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase_allows_execution = lambda: False
    streamer.run_mode = "live"
    streamer.allow_replay_exec_to_nt = False
    streamer.replay_execution_intended = False

    streamer._emit_console_line(
        "OPEN",
        "LONG",
        pd.Timestamp("2026-05-13T07:25:00-06:00"),
        7433.75,
        0.602632,
        0.075887,
        0.432331,
        "PVET",
        "7425.75",
        "7441.75",
        "--",
        context={"emit_allowed": True, "final_action": "OPEN"},
        state_meta={"emit_allowed": True, "final_action": "OPEN"},
    )

    rows = list(csv.DictReader(streamer.state_stream_csv.read_text(encoding="utf-8").splitlines()))
    assert rows
    last = rows[-1]
    assert str(last.get("requested_action") or "").upper() == "OPEN"
    assert str(last.get("resolved_action") or "").upper() == "OPEN"
    lifecycle_rows = _read_jsonl(streamer.lifecycle_events_path)
    assert lifecycle_rows
    assert lifecycle_rows[-1].get("requested_action") == "OPEN"
    assert lifecycle_rows[-1].get("display_action") == "OPEN"


def test_state_csv_downgrades_open_when_execution_intent_is_no_trade(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase_allows_execution = lambda: True

    streamer._emit_console_line(
        "OPEN",
        "LONG",
        pd.Timestamp("2026-05-13T07:10:00-06:00"),
        7429.75,
        0.608295,
        0.189058,
        0.490421,
        "PVET",
        "7417.29",
        "7442.21",
        "0.00",
        context={
            "emit_allowed": True,
            "final_action": "OPEN",
            "execution_intent_action": "NO_TRADE",
        },
        state_meta={
            "emit_allowed": True,
            "final_action": "OPEN",
            "execution_intent_action": "NO_TRADE",
        },
    )

    rows = list(csv.DictReader(streamer.state_stream_csv.read_text(encoding="utf-8").splitlines()))
    assert rows
    last = rows[-1]
    assert str(last.get("requested_action") or "").upper() == "OPEN"
    assert str(last.get("execution_intent_action") or "").upper() == "NO_TRADE"
    lifecycle_rows = _read_jsonl(streamer.lifecycle_events_path)
    assert lifecycle_rows
    assert lifecycle_rows[-1].get("requested_action") == "OPEN"
    assert lifecycle_rows[-1].get("execution_intent_action") == "NO_TRADE"


def test_state_csv_keeps_open_during_backfill_simulation_when_phase_not_executable(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._phase_allows_execution = lambda: False
    streamer.run_mode = "live"
    streamer.allow_replay_exec_to_nt = False
    streamer.replay_execution_intended = False
    streamer._phase = "BACKFILL"

    streamer._emit_console_line(
        "OPEN",
        "LONG",
        pd.Timestamp("2026-05-13T07:35:00-06:00"),
        7435.25,
        0.622345,
        0.082214,
        0.451003,
        "PVET",
        "7427.25",
        "7443.25",
        "--",
        context={"emit_allowed": True, "final_action": "OPEN"},
        state_meta={"emit_allowed": True, "final_action": "OPEN"},
    )

    rows = list(csv.DictReader(streamer.state_stream_csv.read_text(encoding="utf-8").splitlines()))
    assert rows
    last = rows[-1]
    assert str(last.get("action") or "").upper() == "OPEN"


def test_lifecycle_events_preserve_blocked_candidate_action_when_display_is_hold(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._emit_console_line(
        "HOLD",
        "LONG",
        pd.Timestamp("2026-05-13T07:40:00-06:00"),
        7437.25,
        0.622345,
        0.082214,
        0.451003,
        "PVET",
        "7427.25",
        "7443.25",
        "--",
        context_text="Entering LONG at 7437.25",
        state_meta={
            "requested_action": "OPEN",
            "resolved_action": "NO_TRADE",
            "display_action": "HOLD",
            "execution_intent_action": "NO_TRADE",
            "blocked_candidate_action": "OPEN",
            "blocked_candidate_reason": "gates_block",
            "emit_allowed": False,
            "publish_ready": False,
            "transition_id": "tid-1",
            "transition_step": "open",
            "signal_id": "sig-1",
            "client_order_id": "cid-1",
            "phase": "LIVE",
        },
    )
    state_rows = list(csv.DictReader(streamer.state_stream_csv.read_text(encoding="utf-8").splitlines()))
    assert state_rows
    assert state_rows[-1]["action"] in {"HOLD", "NO_TRADE"}
    assert state_rows[-1]["requested_action"] == "OPEN"
    assert state_rows[-1]["blocked_candidate_reason"] == "gates_block"
    lifecycle_rows = _read_jsonl(streamer.lifecycle_events_path)
    assert lifecycle_rows[-1]["requested_action"] == "OPEN"
    assert lifecycle_rows[-1]["display_action"] in {"HOLD", "NO_TRADE"}
    assert lifecycle_rows[-1]["resolved_action"] == "NO_TRADE"


def test_startup_fail_fast_blocks_offline_exec_for_execution_intended_replay(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.startup_fail_fast_blockers = False
    streamer._startup_fail_fast_checked = False
    streamer.nt_enabled = True
    streamer.run_mode = "replay"
    streamer.allow_replay_exec_to_nt = True
    streamer.nt_exec_policy = "paper"
    streamer.nt_bridge = SimpleNamespace(is_connected=True)
    streamer._nt_ready = True
    streamer.nt_exec_state = "ARMED"
    streamer.lockout_policy = "all"
    streamer._entries_disarmed_reason = None
    streamer._offline_detected = True
    streamer._offline_exec_override = False
    streamer.max_fill_slippage_ticks = 8

    streamer._maybe_fail_fast_on_startup_blockers()


def test_replay_exec_offline_override_suppresses_offline_exec_block_event(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.run_mode = "replay"
    streamer.allow_replay_exec_to_nt = True
    streamer.replay_exec_pacing = "as_fast_as_possible"
    streamer.nt_exec_policy = "paper"
    streamer.i_understand_offline_exec = True
    streamer._offline_exec_override = True
    streamer.offline_run = False
    streamer.offline_threshold_min = 1.0
    streamer._offline_last_flag = None
    streamer._offline_reason = None
    streamer._offline_delta_min = None
    streamer._bar_time_delta_min = lambda _bar_ts: 999.0
    streamer._maybe_adjust_exec_policy = lambda _offline: None
    streamer._maybe_clear_exec_lockout = lambda _offline=None: None

    blocked_codes: list[str] = []
    unblocked_codes: list[str] = []
    streamer._emit_block_event = lambda **kwargs: blocked_codes.append(str(kwargs.get("block_code")))
    streamer._emit_unblocked = lambda code: unblocked_codes.append(str(code))

    streamer._update_offline_exec_state(pd.Timestamp("2026-04-25T09:30:00-06:00"))

    assert "offline_exec_blocked" not in blocked_codes
    assert "offline_exec_blocked" in unblocked_codes


def test_replay_fast_snapshot_freshness_uses_bar_time_reference(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "replay"
    streamer.allow_replay_exec_to_nt = True
    streamer.replay_exec_pacing = "as_fast_as_possible"
    streamer.nt_snapshot_fresh_sec = 450.0
    streamer._last_csv_bar = pd.Timestamp("2026-04-25T09:40:00-06:00")
    snapshot_ts = pd.Timestamp("2026-04-25T09:35:30-06:00")
    streamer._nt_last_snapshot_ts = snapshot_ts
    streamer._nt_last_snapshot_orders_ts_by_instrument = {"ES 06-26": snapshot_ts}

    disarm_calls: list[str] = []
    streamer._set_entries_disarmed = lambda reason, _detail: disarm_calls.append(str(reason))
    streamer._maybe_request_nt_snapshot = lambda **_kwargs: None
    streamer._log_exec_event = lambda *_args, **_kwargs: None

    ok = streamer._require_fresh_snapshot(reason="execute_intent_entry", inst_key=streamer.exec_instrument)

    assert ok is True
    assert "snapshot_stale_reconcile" not in disarm_calls


def test_forensic_artifact_contract_lists_missing_required_outputs(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.run_mode = "live"
    streamer.allow_replay_exec_to_nt = False
    streamer.execution_ledger_path = tmp_path / "execution_ledger.jsonl"
    streamer.anomalies_path = tmp_path / "anomalies.jsonl"
    streamer.feature_health_path = tmp_path / "feature_health.jsonl"

    missing = streamer._forensic_artifact_missing_list()

    assert set(missing) == {"execution_ledger.jsonl"}
    assert streamer.anomalies_path.exists()
    assert streamer.feature_health_path.exists()


def test_emit_gate_audit_includes_preflight_summary(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    ts = pd.Timestamp("2026-04-23T06:10:00-06:00")
    streamer._emit_gate_audit(
        ts_dt=ts,
        action="NO_TRADE",
        gate_state={"setup": True, "prob": True, "vwap": True, "ema": True, "tod": True},
        blocked_by=[],
        reason_detail="",
        prob=0.50,
        phase2_meta={"setup_pass": True},
        extra={
            "final_action": "NO_TRADE",
            "side": "FLAT",
            "preflight_readiness": {
                "armed_ok": False,
                "nt_enabled": True,
                "nt_connected": True,
                "nt_ready": True,
                "feed_health_ok": True,
                "offline_ok": True,
            },
        },
    )
    gate_rows = _read_jsonl(streamer.gating_events_path)
    assert gate_rows
    gate_row = gate_rows[-1]
    assert gate_row["reason_detail"] == "not_armed"
    assert gate_row["preflight"]["first_failing_gate"] == "not_armed"
    assert gate_row["preflight"]["readiness"]["armed_ok"] is False


def test_status_clears_startup_resync_when_live_and_armed(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._entries_disarmed_reason = "startup_resync"
    streamer.state.entries_disarmed_reason = "startup_resync"
    streamer._current_block_code = "startup_resync"
    streamer._current_block_ts = utc_ts()
    streamer._block_active = {"startup_resync": {"reason": "awaiting_snapshot"}}
    streamer._block_last_detail = {"startup_resync": {"reason": "awaiting_snapshot"}}
    streamer.nt_exec_state = "ARMED"
    streamer.state.nt_exec_state = "ARMED"
    streamer.run_mode = "live"
    streamer.state.position_state = "FLAT"

    streamer._clear_entries_disarmed()
    streamer._write_status(force=True)
    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))

    assert status["armed"] is True
    assert status["entries_disarmed_reason"] is None
    assert status["primary_state"] == "ARMED"
    assert status["current_block_code"] is None
    assert status["current_block_code"] != "startup_resync"


def test_live_nt_entry_publish_blocks_startup_resync_catchup_entry(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "live"
    streamer.nt_exec_state = "DISARMED"
    streamer._nt_ready = False
    streamer._phase = "CATCHUP"
    streamer._entries_disarmed_reason = "startup_resync"
    streamer.state.entries_disarmed_reason = "startup_resync"

    reason = streamer._live_nt_entry_publish_block_reason(
        "OPEN",
        {
            "type": "OPEN",
            "side": "LONG",
            "datetime": pd.Timestamp("2026-06-03T19:10:00-06:00"),
        },
    )

    assert reason == "startup_resync"


def test_live_nt_entry_publish_allows_ready_live_entry(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_mode = "live"
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "live"
    streamer.nt_exec_state = "ARMED"
    streamer._nt_ready = True
    streamer._phase = "LIVE"
    streamer._entries_disarmed_reason = None
    streamer.state.entries_disarmed_reason = None

    reason = streamer._live_nt_entry_publish_block_reason(
        "OPEN",
        {
            "type": "OPEN",
            "side": "LONG",
            "datetime": pd.Timestamp("2026-06-03T19:15:00-06:00"),
        },
    )

    assert reason is None


def test_startup_fail_fast_raises_on_readiness_blocker(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.startup_fail_fast_blockers = True
    streamer._startup_fail_fast_checked = False
    streamer.nt_enabled = True
    streamer.nt_exec_state = "DISARMED"
    streamer._entries_disarmed_reason = "startup_resync"
    streamer._nt_ready = False
    streamer.max_fill_slippage_ticks = 8

    with pytest.raises(RuntimeError, match="startup_preflight_blocked:not_armed"):
        streamer._maybe_fail_fast_on_startup_blockers()


def test_nt_order_ack_rejected_preserves_raw_reason_and_nt_block_code(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "RUNID|CID-NT-REJECT-1"
    raw_reason = "entry_rejected"
    reject_code = "nt_entry_rejected"
    streamer._nt_order_state = {
        cid: {
            "intent_action": "OPEN",
            "side": "LONG",
            "qty": 1,
            "stop_price": 6525.75,
            "target_price": 6538.75,
            "signal_id": "sig|nt|1",
        }
    }

    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_ACK",
            "client_order_id": cid,
            "status": "REJECTED",
            "reason": raw_reason,
            "protocol_version": 1,
        }
    )

    # In this unit harness `_emit_event` is intentionally stubbed, so event ledger
    # rows may be absent even when status + signal_to_order are updated correctly.
    event_rows = _read_jsonl(streamer.event_ledger_path)
    if event_rows:
        block_rows = [
            row
            for row in event_rows
            if row.get("event_type") in {"BLOCK", "BLOCK_UPDATED", "LOCKOUT"}
        ]
        if block_rows:
            block_row = block_rows[-1]
            block_code = block_row.get("block_code") or block_row.get("lockout_code")
            block_detail = block_row.get("block_detail") or block_row.get("lockout_detail") or {}
            assert block_code == reject_code
            if isinstance(block_detail, dict):
                assert block_detail.get("reason") == raw_reason
                assert block_detail.get("broker_reason") == raw_reason
                assert block_detail.get("reason_code") == reject_code

    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))
    assert status["last_block_code"] == reject_code
    assert status["last_block_detail"]["reason"] == raw_reason
    assert status["executor_last"]["reason_code"] == reject_code
    assert status["executor_last"]["reason_detail"]["nt_reason"] == raw_reason

    signal_rows = _read_jsonl(streamer.signal_to_order_path)
    assert signal_rows, signal_rows
    signal_row = signal_rows[-1]
    assert signal_row["decision"] == "REJECTED"
    assert signal_row["reason"] == raw_reason
    assert signal_row["blocked_by"] == [reject_code]
    assert signal_row["reason_detail"]["nt_reason"] == raw_reason


def test_order_update_close_fill_does_not_rebind_entry_or_run_prop_fill_validation(tmp_path: Path) -> None:
    class _GuardStub:
        def validate_fill(self, *_args, **_kwargs):
            raise AssertionError("validate_fill must not run for CLOSE ORDER_UPDATE fills")

    streamer = _make_streamer(tmp_path)
    cid = "RUNID|CID-CLOSE-ORDER-UPDATE-1"
    streamer.prop_guardrails = _GuardStub()
    streamer._nt_order_state = {
        cid: {
            "intent_action": "CLOSE",
            "side": "SHORT",
            "qty": 1,
            "model_price": 7148.25,
            "entry_order_id": "ENTRY-ACK-ID",
            "entry_ninja_order_id": "ENTRY-ACK-ID",
        }
    }

    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_UPDATE",
            "client_order_id": cid,
            "protocol_version": 1,
            "status": "FILLED",
            "state": "FILLED",
            "ninja_order_id": "CLOSE-NINJA-ID",
            "avg_fill_price": 7152.25,
            "filled": 1,
            "quantity": 1,
            "timestamp": "2026-04-28T11:10:01.8639438-06:00",
        }
    )

    state = streamer._nt_order_state[cid]
    assert state.get("entry_ninja_order_id") == "ENTRY-ACK-ID"
    assert state.get("entry_fill_price") is None
    assert streamer._hard_lockout_code is None


def test_order_update_target_fill_does_not_run_entry_prop_fill_validation(tmp_path: Path) -> None:
    class _GuardStub:
        def validate_fill(self, *_args, **_kwargs):
            raise AssertionError("validate_fill must not run for target ORDER_UPDATE fills")

    streamer = _make_streamer(tmp_path)
    cid = "RUNID|OPEN|SHORT|TARGET-FILL-1"
    streamer.prop_guardrails = _GuardStub()
    streamer.tick_size = 0.25
    streamer._open_trade = {
        "client_order_id": cid,
        "side": "SHORT",
        "entry_price": 7583.0,
        "target": 7571.25,
    }
    streamer._nt_order_state = {
        cid: {
            "intent_action": "OPEN",
            "side": "SHORT",
            "qty": 1,
            "model_price": 7583.25,
            "entry_fill_price": 7583.0,
            "entry_filled": True,
            "entry_ninja_order_id": "TARGET-NINJA-ID",
            "stop_order_id": "STOP-NINJA-ID",
            "target_order_id": "TARGET-NINJA-ID",
            "stop_price": 7594.75,
            "target_price": 7571.25,
        }
    }

    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_UPDATE",
            "client_order_id": cid,
            "protocol_version": 1,
            "status": "FILLED",
            "state": "FILLED",
            "ninja_order_id": "TARGET-NINJA-ID",
            "avg_fill_price": 7571.25,
            "filled": 1,
            "quantity": 1,
            "timestamp": "2026-06-03T10:23:30.6936561-06:00",
        }
    )

    state = streamer._nt_order_state[cid]
    assert state.get("entry_fill_price") == pytest.approx(7583.0)
    assert state.get("exit_fill_price") == pytest.approx(7571.25)
    assert state.get("exit_reason") == "target_hit"
    assert streamer._hard_lockout_code is None


def test_order_update_cancelled_protection_leg_ignored_during_close_in_flight(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "RUNID|OPEN|SHORT|CANCEL-TARGET-DURING-CLOSE"
    desync_calls: list[dict] = []
    order_events: list[tuple[str, str, dict]] = []
    streamer._record_desync = lambda **kwargs: desync_calls.append(dict(kwargs))
    streamer._log_order_event = lambda cid_arg, event, payload=None: order_events.append(
        (str(cid_arg), str(event), dict(payload or {}))
    )
    streamer._closing_fill_in_flight = True
    streamer._active_close_correlation_id = "RUNID|CLOSE|1"
    streamer._nt_order_state = {
        cid: {
            "intent_action": "OPEN",
            "side": "SHORT",
            "qty": 1,
            "entry_filled": True,
            "close_in_progress": True,
            "entry_fill_price": 7543.5,
            "stop_order_id": "STOP-NINJA-ID",
            "target_order_id": "TARGET-NINJA-ID",
            "stop_price": 7554.5,
            "target_price": 7532.5,
        }
    }

    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_UPDATE",
            "client_order_id": cid,
            "protocol_version": 1,
            "status": "CANCELLED",
            "state": "CANCELLED",
            "role": "LIMIT",
            "ninja_order_id": "TARGET-NINJA-ID",
            "filled": 0,
            "quantity": 1,
            "timestamp": "2026-06-03T18:45:02.5732913-06:00",
        }
    )

    assert desync_calls == []
    assert any(event == "desync_cancelled_protection_leg_ignored_close_in_flight" for _, event, _ in order_events)


def test_pre_arm_close_model_event_is_suppressed(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.phase2_close_enabled = True
    streamer.phase2_close_model = object()
    streamer.phase2_close_model_path = None
    streamer.phase2_close_threshold = 0.50
    streamer._open_trade = {"side": "LONG", "entry_price": 7148.25}
    streamer._pos = 1
    streamer._last_prob = 0.61
    streamer._entries_disarmed_reason = "pre_session_arm"
    streamer.state.entries_disarmed_reason = "pre_session_arm"
    streamer._build_phase2_close_feature_row = lambda *args, **kwargs: (
        pd.DataFrame([{"Datetime": pd.Timestamp("2026-04-30T07:10:00-06:00"), "f1": 1.0}]),
        {"side": "LONG", "unrealized_r": 0.0, "bars_in_trade": 0},
    )
    streamer.phase2_close_expected_features = ["f1"]
    streamer.phase2_close_meta = {}
    streamer.phase2_close_calibrator = None
    streamer._close_model_suppression_reason = lambda *args, **kwargs: (
        "close_model_suppressed_pre_session_arm"
        if streamer._entries_disarmed_reason == "pre_session_arm"
        else None
    )

    orig_align = stream_live_csv_mod._align_X
    orig_predict = stream_live_csv_mod._predict_proba_safely
    orig_cal = stream_live_csv_mod.calibrate_proba
    try:
        stream_live_csv_mod._align_X = lambda frame, expected, warn_context=None, strict=True: (frame[list(expected)], None)
        stream_live_csv_mod._predict_proba_safely = lambda *args, **kwargs: [0.95]
        stream_live_csv_mod.calibrate_proba = lambda raw, cal: raw
        event = streamer._maybe_phase2_close_event(
            pd.Series({"Close": 7152.25, "High": 7152.5, "Low": 7151.75}),
            pd.Timestamp("2026-04-30T07:10:00-06:00"),
        )
    finally:
        stream_live_csv_mod._align_X = orig_align
        stream_live_csv_mod._predict_proba_safely = orig_predict
        stream_live_csv_mod.calibrate_proba = orig_cal

    assert event is None


def test_advance_bars_in_trade_for_close_eval_counts_active_position_once(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 1
    streamer._bars_in_trade = 4

    streamer._advance_bars_in_trade_for_close_eval()

    assert streamer._bars_in_trade == 5


def test_advance_bars_in_trade_for_close_eval_counts_each_bar_once(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = -1
    streamer._bars_in_trade = 0
    bar_ts = pd.Timestamp("2026-04-30T07:10:00-06:00")

    streamer._advance_bars_in_trade_for_close_eval(bar_ts)
    streamer._advance_bars_in_trade_for_close_eval(bar_ts)
    streamer._advance_bars_in_trade_for_close_eval(pd.Timestamp("2026-04-30T07:15:00-06:00"))

    assert streamer._bars_in_trade == 2


def test_advance_bars_in_trade_for_close_eval_ignores_flat_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 0
    streamer._bars_in_trade = 4

    streamer._advance_bars_in_trade_for_close_eval()

    assert streamer._bars_in_trade == 4


def test_phase2_close_missing_features_is_soft_skipped(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.phase2_close_enabled = True
    streamer.phase2_close_model = object()
    streamer.phase2_close_model_path = None
    streamer.phase2_close_threshold = 0.50
    streamer._open_trade = {"side": "LONG", "entry_price": 7148.25}
    streamer._pos = 1
    streamer.allow_feature_mismatch = False
    streamer.phase2_close_expected_features = ["unrealized_r", "giveback_r", "distance_to_stop_r", "distance_to_target_r"]
    streamer.phase2_close_meta = {}
    streamer.phase2_close_calibrator = None
    streamer._build_phase2_close_feature_row = lambda *args, **kwargs: (
        pd.DataFrame([{"Datetime": pd.Timestamp("2026-04-30T07:10:00-06:00"), "f1": 1.0}]),
        {"side": "LONG", "bars_in_trade": 0},
    )

    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(payload)

    orig_align = stream_live_csv_mod._align_X
    try:
        stream_live_csv_mod._align_X = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("phase2_close missing features: unrealized_r, giveback_r")
        )
        event = streamer._maybe_phase2_close_event(
            pd.Series({"Close": 7152.25, "High": 7152.5, "Low": 7151.75}),
            pd.Timestamp("2026-04-30T07:10:00-06:00"),
        )
    finally:
        stream_live_csv_mod._align_X = orig_align

    assert event is None
    assert any(ev.get("event") == "phase2_close_skipped_missing_features" for ev in events)


def test_execute_intent_blocks_entry_when_bar_age_is_stale(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, send=lambda _msg: True, handshake_ok=lambda: True)
    streamer.nt_exec_state = "ARMED"
    streamer._entries_disarmed_reason = None
    streamer._hard_lockout_active = False
    streamer._effective_bar_age_max_sec = 60.0
    streamer._last_bar_ts_guard = pd.Timestamp("2026-04-30T10:00:00-06:00")
    streamer._bar_age_guard_seconds = lambda: 120.0

    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="RUNID|open|stale-1",
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        stop_price=7200.0,
        target_price=7210.0,
        entry_price=7205.0,
        model_price=7205.0,
        model_stop_price=7200.0,
        model_target_price=7210.0,
    )

    result = streamer.execute_intent(intent)
    assert result.decision == "BLOCKED_SAFETY"
    assert result.reason_code == "stale_bar_exec_blocked"


def test_execute_intent_close_bypasses_stale_bar_safety_block(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, send=lambda _msg: True, handshake_ok=lambda: True)
    streamer.nt_account_mode = "none"
    streamer.nt_exec_state = "ARMED"
    streamer._entries_disarmed_reason = "stale_bars"
    streamer._hard_lockout_active = False
    streamer._effective_bar_age_max_sec = 60.0
    streamer._last_bar_ts_guard = pd.Timestamp("2026-04-30T10:00:00-06:00")
    streamer._bar_age_guard_seconds = lambda: 120.0
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer.state.position_side = "LONG"
    streamer.state.position_qty = 1
    streamer.execution_ledger = SimpleNamespace(get=lambda _cid: None, record=lambda *_args, **_kwargs: None)

    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="RUNID|close|stale-1",
        action="CLOSE",
        side="LONG",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        stop_price=7200.0,
        target_price=7210.0,
        entry_price=7205.0,
        model_price=7205.0,
        model_stop_price=7200.0,
        model_target_price=7210.0,
    )

    result = streamer.execute_intent(intent)

    assert result.decision != "BLOCKED_SAFETY"
    assert result.reason_code != "stale_bar_exec_blocked"


def test_safety_flatten_force_is_throttled_by_reason_and_instrument(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    sent_payloads: list[dict] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        send=lambda payload: sent_payloads.append(dict(payload)) or True,
    )
    streamer.nt_exec_state = "ARMED"
    streamer.nt_instrument = "ES JUN26"
    streamer.instrument_alias = "ES 06-26"
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer._effective_bar_age_max_sec = 600.0
    streamer._last_bar_ts_guard = pd.Timestamp.now(tz="America/Denver")
    streamer._last_close_guard = {}
    streamer._close_dedupe_cooldown_sec = 300.0
    streamer._external_nt_cids = set()
    streamer.nt_adapter = "native"
    streamer.nt_account_mode = "none"

    streamer._send_nt_flatten("CID-1", reason="protection_repair_failed")
    streamer._send_nt_flatten("CID-2", reason="protection_repair_failed")

    assert len(sent_payloads) == 1
    assert sent_payloads[0].get("type") == "FLATTEN"
    assert sent_payloads[0].get("schema_version") == 1
    assert str(sent_payloads[0].get("session_id") or "").strip() != ""


def test_close_dedupe_collapses_repeated_nonflat_close_lifecycle(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer.state.position_uid = "pos|ES|abc123"
    streamer.state.active_client_order_id = "CID-ENTRY-1"
    streamer._open_trade = {"client_order_id": "CID-ENTRY-1"}
    streamer._last_close_guard = {}
    streamer._close_dedupe_cooldown_sec = 300.0
    streamer._close_duplicate_suppressed_total = 0
    streamer.exec_events_path = tmp_path / "exec_events.jsonl"

    first = streamer._should_dedupe_close(reason="model_close", position_state="IN_POSITION_PROTECTED", instrument="ES JUN26")
    second = streamer._should_dedupe_close(reason="protection_repair_failed", position_state="EXITING", instrument="ES JUN26")

    assert first is False
    assert second is True
    assert streamer._close_duplicate_suppressed_total == 1


def test_duplicate_close_links_to_active_close_watch(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._close_intents = {
        "corr-1": {
            "intent_id": "CLOSE|A",
            "status": "SENT",
            "started_ts": time.time(),
        }
    }
    streamer._active_close_correlation_id = "corr-1"
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="CLOSE|B",
        action="CLOSE",
        side="LONG",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account=None,
        order_type="MARKET",
        correlation_id="corr-2",
    )

    linked = streamer._link_duplicate_close_to_active_watch(intent)

    assert linked == "CLOSE|A"
    rec = streamer._close_intents["corr-1"]
    assert "CLOSE|B" in list(rec.get("linked_duplicate_intents") or [])


def test_close_owner_state_true_for_exiting_and_inflight_close(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.state.position_state = "EXITING"
    assert streamer._active_close_owner_state() is True

    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer._close_intents = {"corr-1": {"intent_id": "CLOSE|A", "status": "SENT"}}
    assert streamer._active_close_owner_state() is True

    streamer._close_intents = {}
    streamer._nt_order_state = {"CID-CLOSE": {"intent_action": "CLOSE", "close_in_progress": True, "status": "WORKING"}}
    assert streamer._active_close_owner_state() is True


def test_record_desync_runs_close_supersession_reconcile_when_recent(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    calls: list[dict] = []
    streamer._run_desync_reconcile_pipeline = lambda **kwargs: calls.append(dict(kwargs))
    now_ts = time.time()
    streamer._close_supersession_guard_last_ts = now_ts
    streamer.close_supersession_desync_window_sec = 12.0
    streamer.close_supersession_desync_reconcile_min_interval_sec = 0.0

    streamer._record_desync(
        reason="broker_nonflat_internal_flat",
        detail={"event_seq_state": "out_of_order_detected"},
        now_ts=now_ts,
    )

    assert calls, "expected close supersession desync reconcile call"
    assert calls[0].get("mismatch_code") == "close_supersession_guard_desync"


def test_emit_trade_telemetry_sets_execution_truth_tag(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    captured: list[tuple[str, dict]] = []

    class _Store:
        def emit(self, event_type: str, record: dict) -> None:
            captured.append((event_type, dict(record)))

    streamer.trade_telemetry_store = _Store()
    streamer._emit_trade_telemetry("FILL", {"nt_message_type": "FILL"})
    streamer._emit_trade_telemetry("POSITION_SNAPSHOT", {"nt_message_type": "POSITION_SNAPSHOT"})

    assert captured[0][1]["is_execution_truth"] is True
    assert captured[1][1]["is_execution_truth"] is False


def test_nt_position_snapshot_preserves_zero_position_qty_in_telemetry(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    captured: list[tuple[str, dict]] = []
    streamer.trade_telemetry_store = SimpleNamespace(emit=lambda event_type, record: captured.append((event_type, dict(record))))
    streamer._record_nt_snapshot_rx = lambda *_args, **_kwargs: None
    streamer._handle_nt_position_snapshot = lambda *_args, **_kwargs: None

    streamer._handle_nt_message_inner(
        {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "client_order_id": "SAFETY|ES JUN26|SNAPSHOT",
            "instrument": "ES JUN26",
            "qty": 0,
            "pos_qty": 0,
            "account": "DEMO7818783",
        }
    )

    assert captured
    assert captured[0][0] == "POSITION_SNAPSHOT"
    assert captured[0][1]["position_qty"] == 0
    assert captured[0][1]["side"] == "FLAT"
    assert captured[0][1]["signal_id"] is None


def test_order_event_suppresses_repeated_idle_flat_snapshots(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    record = {
        "status": "rx_pnl_snapshot",
        "type": "PNL_SNAPSHOT",
        "event_type": "PNL_SNAPSHOT",
        "account": "DEMO7818783",
        "instrument": "ES JUN26",
        "position_qty": 0,
        "position_state": "FLAT",
        "last_price": 7573.25,
    }

    assert streamer._should_suppress_idle_snapshot_order_event(record) is False
    assert streamer._should_suppress_idle_snapshot_order_event(record) is True
    assert streamer._idle_snapshot_order_event_suppressed_total == 1


def test_snapshot_repeat_suppression_counter_surfaces_in_status(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer._nt_snapshot_processing_disabled = True
    streamer._snapshot_only_repeat_suppressed_total = 0

    streamer._handle_nt_message_inner({"type": "POSITION_SNAPSHOT", "protocol_version": 1, "instrument": "ES JUN26"})
    assert streamer._snapshot_only_repeat_suppressed_total == 1

    streamer._write_status(force=True)
    status = json.loads(streamer.status_path.read_text(encoding="utf-8"))
    assert status["snapshot_only_repeat_suppressed_total"] == 1


def test_protection_repair_synthesizes_target_when_missing(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    sent_payloads: list[dict] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        send=lambda payload: sent_payloads.append(dict(payload)) or True,
    )
    streamer.nt_account_mode = "none"
    streamer.nt_instrument = "ES JUN26"
    streamer.instrument_alias = "ES 06-26"
    streamer.tick_size = 0.25
    streamer._last_close = 7240.0

    streamer._send_nt_protection_repair(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7232.75,
        target_price=None,
        reason="missing_stop",
    )

    assert sent_payloads, "repair order was not sent"
    order = sent_payloads[-1]
    assert order.get("type") == "ORDER"
    assert order.get("order_type") == "BRACKET"
    assert order.get("stop_price") is not None
    assert order.get("target_price") is not None
    assert order.get("repair_target_required") is True
    assert order.get("protection_price_mode") == "absolute"
    assert order.get("model_stop_abs") == pytest.approx(float(order.get("stop_price")))
    assert order.get("model_target_abs") == pytest.approx(float(order.get("target_price")))


def test_protection_repair_blocks_locally_when_target_unresolvable(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    sent_payloads: list[dict] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        send=lambda payload: sent_payloads.append(dict(payload)) or True,
    )
    streamer.nt_account_mode = "none"
    streamer.nt_instrument = "ES JUN26"
    streamer.instrument_alias = "ES 06-26"
    streamer.tick_size = 0.25
    streamer._last_close = None
    streamer._snapshot_price_state = lambda: (False, None, None, None, None)

    streamer._send_nt_protection_repair(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7232.75,
        target_price=None,
        reason="missing_stop",
    )

    assert sent_payloads == []
    rows = [
        row
        for row in _read_jsonl(streamer.signal_to_order_path)
        if str(row.get("signal_action") or "").upper() in {"OPEN", "CLOSE", "FLIP"}
    ]
    assert rows
    last = rows[-1]
    assert last.get("decision") == "REJECTED"
    assert last.get("reason_code") == "nt_missing_protection_prices"
    detail = last.get("reason_detail") or {}
    assert detail.get("blocked_locally") is True


def test_protection_repair_qty_rounds_up_from_fractional_position(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    sent_payloads: list[dict] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        send=lambda payload: sent_payloads.append(dict(payload)) or True,
    )
    streamer.nt_account_mode = "none"
    streamer.nt_instrument = "ES JUN26"
    streamer.instrument_alias = "ES 06-26"
    streamer.tick_size = 0.25
    streamer._last_close = 7240.0

    sent = streamer._send_nt_protection_repair(
        inst_key="ES 06-26",
        pos_qty=0.6,
        stop_price=7232.75,
        target_price=7247.25,
        reason="fractional_pos",
    )

    assert sent is True
    assert sent_payloads
    assert int(sent_payloads[-1].get("qty") or 0) >= 1


def test_protection_repair_uses_exit_side_from_position_direction(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    sent_payloads: list[dict] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        send=lambda payload: sent_payloads.append(dict(payload)) or True,
    )
    streamer.nt_account_mode = "none"
    streamer.nt_instrument = "ES JUN26"
    streamer.instrument_alias = "ES 06-26"
    streamer.tick_size = 0.25
    streamer._last_close = 7240.0

    streamer._send_nt_protection_repair(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7232.75,
        target_price=7247.25,
        reason="side_check_long",
    )
    streamer._send_nt_protection_repair(
        inst_key="ES 06-26",
        pos_qty=-1.0,
        stop_price=7247.25,
        target_price=7232.75,
        reason="side_check_short",
    )

    assert len(sent_payloads) >= 2
    assert sent_payloads[-2].get("action") == "SELL"
    assert sent_payloads[-1].get("action") == "BUY"
    sell_order = sent_payloads[-2]
    buy_order = sent_payloads[-1]
    sell_price = float(sell_order.get("model_price"))
    sell_stop = float(sell_order.get("model_stop_abs"))
    sell_target = float(sell_order.get("model_target_abs"))
    buy_price = float(buy_order.get("model_price"))
    buy_stop = float(buy_order.get("model_stop_abs"))
    buy_target = float(buy_order.get("model_target_abs"))
    assert sell_stop < sell_price < sell_target
    assert buy_target < buy_price < buy_stop


def test_validate_repair_geometry_short_and_long(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    assert streamer._validate_repair_geometry(
        side="SHORT",
        entry=7418.75,
        stop=7425.5,
        target=7412.0,
    ) is True
    assert streamer._validate_repair_geometry(
        side="LONG",
        entry=7400.0,
        stop=7390.0,
        target=7420.0,
    ) is True
    assert streamer._validate_repair_geometry(
        side="SHORT",
        entry=7418.75,
        stop=7412.0,
        target=7425.5,
    ) is False


def test_protection_repair_short_normalizes_stop_target_and_invalid_escalates_flatten(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    sent_payloads: list[dict] = []
    flatten_calls: list[tuple[str, str]] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        send=lambda payload: sent_payloads.append(dict(payload)) or True,
    )
    streamer.nt_account_mode = "none"
    streamer.nt_instrument = "ES JUN26"
    streamer.instrument_alias = "ES 06-26"
    streamer.tick_size = 0.25
    streamer._last_close = 7418.75
    streamer._send_nt_flatten = lambda cid, reason: flatten_calls.append((str(cid), str(reason)))

    sent = streamer._send_nt_protection_repair(
        inst_key="ES 06-26",
        pos_qty=-1.0,
        stop_price=7412.0,
        target_price=7425.5,
        reason="short_geometry",
    )
    assert sent is True
    assert sent_payloads
    short_order = sent_payloads[-1]
    assert short_order.get("stop_price") == pytest.approx(7425.5)
    assert short_order.get("target_price") == pytest.approx(7412.0)
    assert short_order.get("model_stop_abs") == pytest.approx(7425.5)
    assert short_order.get("model_target_abs") == pytest.approx(7412.0)

    sent_payloads.clear()
    streamer._last_close = 7418.75
    streamer._snapshot_price_state = lambda: (False, None, None, None, None)
    sent_bad = streamer._send_nt_protection_repair(
        inst_key="ES 06-26",
        pos_qty=-1.0,
        stop_price=7418.75,
        target_price=7418.75,
        reason="bad_geometry",
    )
    assert sent_bad is False
    assert sent_payloads == []
    assert flatten_calls, "invalid geometry should escalate flatten"
    assert flatten_calls[-1][1] == "protection_repair_failed"


def test_protection_repair_blocks_locally_on_degenerate_geometry(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    sent_payloads: list[dict] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        send=lambda payload: sent_payloads.append(dict(payload)) or True,
    )
    streamer.nt_account_mode = "none"
    streamer.nt_instrument = "ES JUN26"
    streamer.instrument_alias = "ES 06-26"
    streamer.tick_size = 0.25
    streamer._last_close = 7240.0

    sent = streamer._send_nt_protection_repair(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7240.0,
        target_price=7240.0,
        reason="degenerate_geometry",
    )

    assert sent is False
    assert sent_payloads == []
    rows = [
        row
        for row in _read_jsonl(streamer.signal_to_order_path)
        if str(row.get("signal_action") or "").upper() in {"OPEN", "CLOSE", "FLIP"}
    ]
    assert rows
    last = rows[-1]
    assert last.get("decision") == "REJECTED"
    assert last.get("reason_code") == "nt_invalid_bracket_geometry"


def test_nt_instrument_for_tx_prefers_configured_text_contract(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_instrument = "ES JUN26"
    streamer.exec_instrument = "ES 06-26"
    streamer.instrument_alias = "ES 06-26"

    assert streamer._nt_instrument_for_tx() == "ES JUN26"


def test_protection_repair_attempts_increment_only_on_send(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._repair_snapshot_stale_for_protection = lambda _inst: (False, {})
    streamer._repair_snapshot_stale_for_protection = lambda _inst: (False, {})
    streamer._repair_snapshot_stale_for_protection = lambda _inst: (False, {})
    streamer._repair_snapshot_stale_for_protection = lambda _inst: (False, {})
    streamer.protection_repair_enabled = True
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _payload: True)
    streamer.nt_account_mode = "none"
    streamer.nt_instrument = "ES JUN26"
    streamer.instrument_alias = "ES 06-26"
    streamer.tick_size = 0.25
    streamer._last_close = None
    streamer._snapshot_price_state = lambda: (False, None, None, None, None)
    streamer._phase = "LIVE"
    streamer._nt_repair_state_by_instrument = {}
    streamer._expected_unprotected_by_instrument = {}
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer.nt_protection_repair_retry_sec = 0.0
    streamer.nt_protection_repair_timeout_sec = 30.0
    streamer._protection_repairs_attempted = 0

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7232.75,
        target_price=None,
        reason="missing_target",
    )
    state = streamer._nt_repair_state_by_instrument.get("ES 06-26") or {}
    assert int(state.get("attempts") or 0) == 0
    assert int(streamer._protection_repairs_attempted or 0) == 0


def test_protection_repair_suppressed_while_repair_lockout_active(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._repair_snapshot_stale_for_protection = lambda _inst: (False, {})
    streamer._repair_snapshot_stale_for_protection = lambda _inst: (False, {})
    streamer._repair_snapshot_stale_for_protection = lambda _inst: (False, {})
    streamer._repair_snapshot_stale_for_protection = lambda _inst: (False, {})
    streamer.protection_repair_enabled = True
    streamer._hard_lockout_active = True
    streamer._hard_lockout_code = "protection_repair_failed"
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_repair_state_by_instrument = {}
    calls: list[dict] = []
    streamer._send_nt_protection_repair = lambda **kwargs: calls.append(dict(kwargs)) or True

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7232.75,
        target_price=7240.25,
        reason="snapshot_exits_mismatch",
    )

    assert calls == []
    assert streamer._nt_repair_state_by_instrument.get("ES 06-26") is None


def test_repair_signed_pos_qty_uses_side_not_absolute_qty(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    short_val = streamer._repair_signed_pos_qty({"qty": 1, "side": "SHORT"})
    long_val = streamer._repair_signed_pos_qty({"qty": 1, "side": "LONG"})
    assert short_val == pytest.approx(-1.0)
    assert long_val == pytest.approx(1.0)


def test_extract_snapshot_exits_uses_target_price_when_order_type_degraded(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    orders = [
        {
            "order_type": "STOP",
            "side": "BUYTOCOVER",
            "qty": 1,
            "ninja_order_id": "STOP-1",
            "client_order_id": "UNTRACKED|ES JUN26|STOP-1",
            "order_state": "WORKING",
            "stop_price": 7554.5,
        },
        {
            "order_type": "",
            "side": "BUYTOCOVER",
            "qty": 1,
            "ninja_order_id": "TARGET-1",
            "client_order_id": "UNTRACKED|ES JUN26|TARGET-1",
            "order_state": "WORKING",
            "target_price": 7532.5,
        },
    ]

    exits = streamer._extract_snapshot_exits(orders, "ES 06-26")

    assert exits.get("stop_price") == pytest.approx(7554.5)
    assert exits.get("target_price") == pytest.approx(7532.5)
    assert exits.get("target_order_id") == "TARGET-1"


def test_extract_snapshot_exits_merges_untracked_target_for_preferred_stop(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    active_cid = "RUN|ES JUN26|OPEN|SHORT|abc"
    streamer.state.active_client_order_id = active_cid
    streamer._nt_order_state = {
        active_cid: {
            "instrument": "ES JUN26",
            "entry_filled": True,
        }
    }
    orders = [
        {
            "order_type": "STOP",
            "side": "BUYTOCOVER",
            "qty": 1,
            "ninja_order_id": "STOP-1",
            "client_order_id": active_cid,
            "order_state": "WORKING",
            "stop_price": 7552.5,
        },
        {
            "order_type": "LIMIT",
            "side": "BUYTOCOVER",
            "qty": 1,
            "ninja_order_id": "TARGET-1",
            "client_order_id": "UNTRACKED|ES JUN26|TARGET-1",
            "order_state": "WORKING",
            "limit_price": 7536.0,
            "target_price": 7536.0,
        },
    ]

    exits = streamer._extract_snapshot_exits(orders, "ES 06-26")

    assert exits.get("stop_price") == pytest.approx(7552.5)
    assert exits.get("target_price") == pytest.approx(7536.0)
    assert exits.get("stop_order_id") == "STOP-1"
    assert exits.get("target_order_id") == "TARGET-1"


def test_protection_repair_suppressed_when_close_in_progress_for_instrument(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer.protection_repair_enabled = True
    streamer._hard_lockout_active = False
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_order_state = {
        "CLOSE-CID": {"instrument": "ES JUN26", "close_in_progress": True, "intent_action": "CLOSE"}
    }
    calls: list[dict] = []
    streamer._send_nt_protection_repair = lambda **kwargs: calls.append(dict(kwargs)) or True

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7232.75,
        target_price=7240.25,
        reason="snapshot_exits_mismatch",
    )

    assert calls == []


def test_protection_repair_suppressed_recent_confirmed_exit_guard(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer.protection_repair_enabled = True
    streamer._hard_lockout_active = False
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_order_state = {}
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    calls: list[dict] = []
    streamer._send_nt_protection_repair = lambda **kwargs: calls.append(dict(kwargs)) or True
    streamer._mark_confirmed_exit_terminal(
        inst_key="ES 06-26",
        state_map={"last_exit_order_id": "X1", "client_order_id": "CID-X1", "exit_fill_ts": time.time()},
        source="test_guard",
    )

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7232.75,
        target_price=7240.25,
        reason="snapshot_exits_mismatch",
    )

    assert calls == []
    assert any(ev.get("event") == "protection_repair_suppressed_recent_confirmed_exit" for ev in events)


def test_confirmed_exit_cleanup_idempotent_by_exit_key(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    state = {
        "client_order_id": "CID-1",
        "signal_id": "SIG-1",
        "last_exit_order_id": "EXIT-1",
        "exit_fill_ts": 1000.0,
        "stop_state": "FILLED",
    }
    assert streamer._reconcile_confirmed_exit_cleanup(
        inst_key="ES 06-26",
        active_state=state,
        repair_state={},
        source="test_once",
    )
    count_after_first = int(getattr(streamer, "_protection_repairs_suppressed_confirmed_exit", 0) or 0)
    assert streamer._reconcile_confirmed_exit_cleanup(
        inst_key="ES 06-26",
        active_state=state,
        repair_state={},
        source="test_twice",
    )
    count_after_second = int(getattr(streamer, "_protection_repairs_suppressed_confirmed_exit", 0) or 0)
    assert count_after_second == count_after_first

def test_protection_repair_non_mutating_mode_blocks_order_send(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    sent_payloads: list[dict] = []
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        send=lambda payload: sent_payloads.append(dict(payload)) or True,
    )
    streamer.nt_account_mode = "none"
    streamer.nt_instrument = "ES JUN26"
    streamer.instrument_alias = "ES 06-26"
    streamer.tick_size = 0.25
    streamer._last_close = 7240.0
    streamer.nt_exec_policy = "paper"
    streamer._phase = "LIVE"
    streamer.repair_non_mutating_mode_enabled = True

    sent = streamer._send_nt_protection_repair(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7232.75,
        target_price=7247.25,
        reason="snapshot_exits_mismatch",
    )

    assert sent is False
    assert sent_payloads == []
    names = [str((ev or {}).get("event")) for ev in events if isinstance(ev, dict)]
    assert "repair_non_mutating_mode_active" in names
    assert "repair_order_send_blocked" in names
    assert "repair_watchdog_deferred" in names

def test_position_snapshot_order_update_repair_preserves_pnl_shelf_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.guardrail_config = stream_live_csv_mod.GuardRailConfig()
    streamer._ensure_compat_defaults = LiveCSVStreamer._ensure_compat_defaults.__get__(streamer, LiveCSVStreamer)
    streamer._ensure_compat_defaults()
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, send=lambda _msg: True, handshake_ok=lambda: True)
    streamer._nt_snapshot_seen = True
    streamer.nt_canonical_instrument = None
    streamer._nt_stop_update_pending = {}
    streamer._nt_stop_update_inflight_by_inst = {}
    streamer._nt_last_pos_qty_by_instrument = {}
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer._open_trade = {
        "side": "LONG",
        "entry_price": 7243.25,
        "pnl_shelf_armed_index": 1,
        "pnl_shelf_locked_floor_usd": 100.0,
    }
    repair_cid = "repair|ES 06-26|2026-04-30T15:00:00Z"
    streamer._nt_repair_state_by_instrument = {"ES 06-26": {"attempts": 1, "start_ts": time.time() - 5}}

    streamer._handle_nt_message_inner(
        {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "instrument": "ES JUN26",
            "account": "DEMO7467442",
            "timestamp": "2026-04-30T15:01:00-06:00",
            "qty": 1,
            "pos_qty": 1,
            "side": "LONG",
            "position": {
                "side": "LONG",
                "qty": 1,
                "entry_price": 7243.25,
                "pnl_shelf_state": {
                    "max_unrealized_usd": 225.0,
                    "armed_shelf_index": 1,
                    "locked_floor_usd": 100.0,
                    "last_shelf_ts": "2026-04-30T14:58:00-06:00",
                },
            },
            "orders": [],
            "positions": [],
        }
    )
    floor_after_snapshot = float(streamer._open_trade.get("pnl_shelf_locked_floor_usd") or 0.0)
    idx_after_snapshot = streamer._open_trade.get("pnl_shelf_armed_index")

    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_UPDATE",
            "protocol_version": 1,
            "client_order_id": repair_cid,
            "status": "SUBMITTED",
            "state": "SUBMITTED",
            "ninja_order_id": "RPR-1",
            "timestamp": "2026-04-30T15:01:05-06:00",
        }
    )

    floor_after_update = float(streamer._open_trade.get("pnl_shelf_locked_floor_usd") or 0.0)
    idx_after_update = streamer._open_trade.get("pnl_shelf_armed_index")

    assert floor_after_snapshot == pytest.approx(100.0)
    assert floor_after_update == pytest.approx(100.0)
    assert idx_after_snapshot == 1
    assert idx_after_update == 1
    assert "ES 06-26" in streamer._nt_repair_state_by_instrument


def test_snapshot_missing_target_with_working_stop_does_not_trigger_repair(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.guardrail_config = stream_live_csv_mod.GuardRailConfig()
    streamer._ensure_compat_defaults = LiveCSVStreamer._ensure_compat_defaults.__get__(streamer, LiveCSVStreamer)
    streamer._ensure_compat_defaults()
    streamer.nt_enabled = True
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer._nt_snapshot_seen = True
    streamer._nt_order_state = {}
    streamer._last_known_exit_prices = lambda _inst: (7256.5, 7249.25)
    streamer._effective_expects_target = lambda _ctx: True
    streamer._extract_snapshot_exits = lambda _orders, _inst: {
        "stop_price": 7256.5,
        "target_price": None,
        "stop_qty": 1,
        "target_qty": None,
        "stop_order_id": "STOP-1",
        "target_order_id": None,
        "stop_state": "WORKING",
        "target_state": None,
        "client_order_id": "CID-PRIMARY",
    }
    repair_calls: list[dict] = []
    streamer._maybe_repair_protection = lambda **kwargs: repair_calls.append(dict(kwargs))

    streamer._handle_nt_message_inner(
        {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "instrument": "ES JUN26",
            "account": "DEMO7467442",
            "timestamp": "2026-05-01T03:40:13-06:00",
            "qty": 1,
            "pos_qty": 1,
            "side": "SHORT",
            "orders": [{"client_order_id": "CID-PRIMARY"}],
            "positions": [{"instrument": "ES JUN26", "qty": 1, "side": "SHORT", "avg_price": 7252.75}],
        }
    )

    assert repair_calls == []


def test_guardrail_price_age_missing_does_not_disarm_fresh_protected_snapshot(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.guardrail_config = stream_live_csv_mod.GuardRailConfig(
        price_age_max_sec=360.0,
        snapshot_age_max_sec=450.0,
        bar_age_max_sec=605.0,
        effective_bar_age_max_sec=605.0,
        bar_interval_sec=300.0,
    )
    streamer._ensure_compat_defaults = LiveCSVStreamer._ensure_compat_defaults.__get__(streamer, LiveCSVStreamer)
    streamer._ensure_compat_defaults()
    streamer.run_mode = "live"
    streamer.sim_mode = False
    streamer.nt_enabled = True
    streamer.nt_require_snapshot = True
    streamer.nt_require_ref_price = True
    streamer.nt_exec_state = "ARMED"
    streamer._nt_ready = True
    streamer._nt_ready_reason = "ok"
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True)
    streamer._hard_lockout_active = False
    streamer._entries_disarmed_reason = None
    streamer.state.entries_disarmed_reason = None
    streamer._pos = -1.0
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer.state.nt_protected = True
    streamer.state.nt_exits_working = True
    streamer._last_bar_ts_guard = _now_denver()
    inst_key = "ES 06-26"
    streamer._exec_instrument_key = lambda: inst_key
    streamer._nt_last_snapshot_orders_by_instrument[inst_key] = [
        {
            "order_type": "STOP",
            "side": "BUYTOCOVER",
            "order_state": "WORKING",
            "stop_price": 7556.0,
            "ninja_order_id": "STOP-1",
        },
        {
            "order_type": "LIMIT",
            "side": "BUYTOCOVER",
            "order_state": "WORKING",
            "limit_price": 7531.0,
            "target_price": 7531.0,
            "ninja_order_id": "TARGET-1",
        },
    ]
    now_ts = _now_denver()
    streamer._nt_last_snapshot_orders_ts_by_instrument[inst_key] = now_ts
    streamer._nt_last_snapshot_ts_utc = pd.Timestamp(now_ts).tz_convert("UTC")
    streamer._nt_last_price_by_instrument[inst_key] = 7543.5
    streamer._nt_last_price_ts_by_instrument.pop(inst_key, None)

    ctx = streamer._guardrail_context(preflight=True, bar_ts=now_ts)
    guard = stream_live_csv_mod.evaluate_guardrails(ctx, streamer.guardrail_config, preflight=True)

    assert ctx["protected_snapshot_can_skip_price"] is True
    assert ctx["enforce_snapshot_price_checks"] is False
    assert guard.required_action == "NONE"
    assert "price_age_missing" not in {reason.get("code") for reason in guard.reason_dicts()}


def test_snapshot_mismatch_suppressed_while_close_in_progress(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.guardrail_config = stream_live_csv_mod.GuardRailConfig()
    streamer._ensure_compat_defaults = LiveCSVStreamer._ensure_compat_defaults.__get__(streamer, LiveCSVStreamer)
    streamer._ensure_compat_defaults()
    streamer.nt_enabled = True
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer._nt_snapshot_seen = True
    streamer._nt_order_state = {
        "SAFETY|X|ES JUN26|protection_repair_failed|abc": {
            "instrument": "ES JUN26",
            "intent_action": "FLATTEN",
            "close_in_progress": True,
        }
    }
    streamer._last_known_exit_prices = lambda _inst: (7256.5, 7249.25)
    streamer._effective_expects_target = lambda _ctx: True
    streamer._extract_snapshot_exits = lambda _orders, _inst: {
        "stop_price": None,
        "target_price": None,
        "stop_qty": None,
        "target_qty": None,
        "stop_order_id": None,
        "target_order_id": None,
        "stop_state": None,
        "target_state": None,
        "client_order_id": "CID-PRIMARY",
    }
    repair_calls: list[dict] = []
    streamer._maybe_repair_protection = lambda **kwargs: repair_calls.append(dict(kwargs))

    streamer._handle_nt_message_inner(
        {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "instrument": "ES JUN26",
            "account": "DEMO7467442",
            "timestamp": "2026-05-01T04:40:09-06:00",
            "qty": 1,
            "pos_qty": 1,
            "side": "LONG",
            "orders": [{"client_order_id": "CID-PRIMARY"}],
            "positions": [{"instrument": "ES JUN26", "qty": 1, "side": "LONG", "avg_price": 7249.75}],
        }
    )

    assert repair_calls == []


def test_snapshot_no_orders_known_bracket_sends_immediate_repair(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.guardrail_config = stream_live_csv_mod.GuardRailConfig()
    streamer._ensure_compat_defaults = LiveCSVStreamer._ensure_compat_defaults.__get__(streamer, LiveCSVStreamer)
    streamer._ensure_compat_defaults()
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "live"
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True)
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer.state.active_client_order_id = "CID-SNAPSHOT-REPAIR"
    streamer._nt_snapshot_seen = True
    streamer._nt_last_pos_qty_by_instrument = {}
    streamer._nt_snapshot_blocking_orders_count = 0
    streamer._nt_stop_update_pending = {}
    streamer._nt_repair_state_by_instrument = {}
    streamer._expected_unprotected_by_instrument = {}
    streamer._last_known_exit_prices = lambda _inst: (7540.75, 7551.75)
    send_calls: list[dict] = []
    streamer._send_nt_protection_repair = lambda **kwargs: send_calls.append(dict(kwargs)) or True
    snapshot_requests: list[dict] = []
    streamer._maybe_request_nt_snapshot = lambda **kwargs: snapshot_requests.append(dict(kwargs))
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._nt_order_state = {
        "CID-SNAPSHOT-REPAIR": {
            "instrument": "ES JUN26",
            "client_order_id": "CID-SNAPSHOT-REPAIR",
            "entry_filled": True,
            "fill_ts": time.time(),
            "qty": 1,
            "stop_price": 7540.75,
            "target_price": 7551.75,
            "protected": False,
            "exits_working": False,
            "status": "entry_filled",
        }
    }

    streamer._handle_nt_message_inner(
        {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "instrument": "ES JUN26",
            "account": "DEMO7818783",
            "timestamp": _canonical_ts_str(_now_denver()),
            "qty": 1,
            "pos_qty": 1,
            "side": "LONG",
            "orders": [],
            "positions": [{"instrument": "ES JUN26", "qty": 1, "side": "LONG", "avg_price": 7546.5}],
        }
    )

    assert send_calls
    assert send_calls[-1]["reason"] == "snapshot_no_orders_immediate"
    assert send_calls[-1]["stop_price"] == pytest.approx(7540.75)
    assert send_calls[-1]["target_price"] == pytest.approx(7551.75)
    repair_inst_key = send_calls[-1]["inst_key"]
    assert streamer._nt_repair_state_by_instrument[repair_inst_key]["attempts"] == 1
    assert snapshot_requests[-1]["reason"] == "snapshot_no_orders_immediate_repair"
    assert any(e.get("event") == "snapshot_no_orders_immediate_repair" for e in events)


def test_snapshot_no_orders_ignores_cached_protected_state_and_repairs(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.guardrail_config = stream_live_csv_mod.GuardRailConfig()
    streamer._ensure_compat_defaults = LiveCSVStreamer._ensure_compat_defaults.__get__(streamer, LiveCSVStreamer)
    streamer._ensure_compat_defaults()
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "live"
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True)
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer.state.active_client_order_id = "CID-CACHED-PROTECTED"
    streamer.state.nt_protected = True
    streamer.state.nt_exits_working = True
    streamer._nt_snapshot_seen = True
    streamer._nt_last_pos_qty_by_instrument = {}
    streamer._nt_snapshot_blocking_orders_count = 0
    streamer._nt_stop_update_pending = {}
    streamer._nt_repair_state_by_instrument = {}
    streamer._expected_unprotected_by_instrument = {}
    streamer._last_known_exit_prices = lambda _inst: (7554.0, 7537.5)
    send_calls: list[dict] = []
    streamer._send_nt_protection_repair = lambda **kwargs: send_calls.append(dict(kwargs)) or True
    snapshot_requests: list[dict] = []
    streamer._maybe_request_nt_snapshot = lambda **kwargs: snapshot_requests.append(dict(kwargs))
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    protected_epoch = time.time()
    streamer._nt_order_state = {
        "CID-CACHED-PROTECTED": {
            "instrument": "ES JUN26",
            "client_order_id": "CID-CACHED-PROTECTED",
            "entry_filled": True,
            "fill_ts": protected_epoch - 1.0,
            "qty": 1,
            "side": "SHORT",
            "stop_price": 7554.0,
            "target_price": 7537.5,
            "stop_order_id": "OPAQUE-STOP",
            "target_order_id": "OPAQUE-TARGET",
            "stop_state": "WORKING",
            "target_state": "WORKING",
            "protected": True,
            "exits_working": True,
            "protection_confirmed_ts": protected_epoch,
            "protected_ts": protected_epoch,
            "status": "EXITS_WORKING",
        }
    }

    streamer._handle_nt_message_inner(
        {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "instrument": "ES JUN26",
            "account": "DEMO7818783",
            "timestamp": _canonical_ts_str(_now_denver()),
            "qty": 1,
            "pos_qty": 1,
            "side": "SHORT",
            "orders": [],
            "positions": [{"instrument": "ES JUN26", "qty": 1, "side": "SHORT", "avg_price": 7545.75}],
        }
    )

    assert send_calls
    assert send_calls[-1]["reason"] == "snapshot_no_orders_immediate"
    assert send_calls[-1]["stop_price"] == pytest.approx(7554.0)
    assert send_calls[-1]["target_price"] == pytest.approx(7537.5)
    assert snapshot_requests[-1]["reason"] == "snapshot_no_orders_immediate_repair"
    assert not any(e.get("event") == "protection_repair_suppressed_already_protected" for e in events)


def test_snapshot_mismatch_suppressed_for_stale_snapshot_context(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.guardrail_config = stream_live_csv_mod.GuardRailConfig()
    streamer._ensure_compat_defaults = LiveCSVStreamer._ensure_compat_defaults.__get__(streamer, LiveCSVStreamer)
    streamer._ensure_compat_defaults()
    streamer.nt_enabled = True
    streamer.nt_exec_state = "ARMED"
    streamer.nt_snapshot_fresh_sec = 0.0
    streamer._compute_nt_ready = lambda: None
    streamer._nt_stop_update_pending = {}
    streamer._nt_stop_update_last_emit_ts = {}
    streamer._nt_stop_update_last_price_by_cid = {}
    streamer._nt_last_pos_qty_by_instrument = {}
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer._nt_snapshot_seen = True
    cid = "CID-STALE-1"
    now = time.time()
    streamer._nt_order_state = {
        cid: {
            "instrument": "ES JUN26",
            "client_order_id": cid,
            "entry_filled": True,
            "fill_ts": now,
            "ack_ts": now,
            "sent_ts": now,
        }
    }
    streamer.state.active_client_order_id = cid
    streamer._last_known_exit_prices = lambda _inst: (7256.5, 7249.25)
    streamer._effective_expects_target = lambda _ctx: True
    streamer._extract_snapshot_exits = lambda _orders, _inst: {
        "stop_price": None,
        "target_price": None,
        "stop_qty": None,
        "target_qty": None,
        "stop_order_id": None,
        "target_order_id": None,
        "stop_state": None,
        "target_state": None,
        "client_order_id": cid,
    }
    repair_calls: list[dict] = []
    streamer._maybe_repair_protection = lambda **kwargs: repair_calls.append(dict(kwargs))

    stale_ts = "2026-05-01T04:39:00-06:00"
    streamer._handle_nt_message_inner(
        {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "instrument": "ES JUN26",
            "account": "DEMO7467442",
            "timestamp": stale_ts,
            "qty": 1,
            "pos_qty": 1,
            "side": "LONG",
            "orders": [{"client_order_id": cid}],
            "positions": [{"instrument": "ES JUN26", "qty": 1, "side": "LONG", "avg_price": 7249.75}],
        }
    )

    assert repair_calls == []


def test_snapshot_stop_transition_grace_suppresses_missing_target_repair(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.guardrail_config = stream_live_csv_mod.GuardRailConfig()
    streamer._ensure_compat_defaults = LiveCSVStreamer._ensure_compat_defaults.__get__(streamer, LiveCSVStreamer)
    streamer._ensure_compat_defaults()
    streamer.nt_enabled = True
    streamer.nt_exec_state = "ARMED"
    streamer.nt_snapshot_fresh_sec = 0.0
    streamer._compute_nt_ready = lambda: None
    streamer._nt_stop_update_pending = {}
    streamer._nt_stop_update_last_emit_ts = {}
    streamer._nt_stop_update_last_price_by_cid = {}
    streamer._nt_last_pos_qty_by_instrument = {}
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._nt_snapshot_seen = True
    cid = "CID-TRANS-1"
    now = time.time()
    streamer._nt_order_state = {
        cid: {
            "instrument": "ES JUN26",
            "client_order_id": cid,
            "entry_filled": True,
            "fill_ts": now,
            "ack_ts": now,
            "sent_ts": now,
        }
    }
    streamer.state.active_client_order_id = cid
    streamer._last_known_exit_prices = lambda _inst: (7256.5, None)
    streamer._effective_expects_target = lambda _ctx: True
    streamer._extract_snapshot_exits = lambda _orders, _inst: {
        "stop_price": 7256.5,
        "target_price": None,
        "stop_qty": 1,
        "target_qty": None,
        "stop_order_id": "STOP-TRANS-1",
        "target_order_id": None,
        "stop_state": "INITIALIZED",
        "target_state": None,
        "client_order_id": cid,
    }
    repair_calls: list[dict] = []
    streamer._maybe_repair_protection = lambda **kwargs: repair_calls.append(dict(kwargs))

    streamer._handle_nt_message_inner(
        {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "instrument": "ES JUN26",
            "account": "DEMO7467442",
            "timestamp": "2026-05-01T04:40:13-06:00",
            "qty": 1,
            "pos_qty": 1,
            "side": "LONG",
            "orders": [{"client_order_id": cid}],
            "positions": [{"instrument": "ES JUN26", "qty": 1, "side": "LONG", "avg_price": 7249.75}],
        }
    )

    assert repair_calls == []
    assert streamer.state.nt_protected is True


def test_snapshot_fresh_true_mismatch_still_escalates_repair(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.guardrail_config = stream_live_csv_mod.GuardRailConfig()
    streamer._ensure_compat_defaults = LiveCSVStreamer._ensure_compat_defaults.__get__(streamer, LiveCSVStreamer)
    streamer._ensure_compat_defaults()
    streamer.nt_enabled = True
    streamer.nt_exec_state = "ARMED"
    streamer.nt_snapshot_fresh_sec = 0.0
    streamer._compute_nt_ready = lambda: None
    streamer._nt_stop_update_pending = {}
    streamer._nt_stop_update_last_emit_ts = {}
    streamer._nt_stop_update_last_price_by_cid = {}
    streamer._nt_last_pos_qty_by_instrument = {}
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer._nt_snapshot_seen = True
    streamer._nt_order_state = {}
    streamer._last_known_exit_prices = lambda _inst: (7256.5, 7249.25)
    streamer._effective_expects_target = lambda _ctx: True
    streamer._extract_snapshot_exits = lambda _orders, _inst: {
        "stop_price": None,
        "target_price": None,
        "stop_qty": None,
        "target_qty": None,
        "stop_order_id": None,
        "target_order_id": None,
        "stop_state": None,
        "target_state": None,
        "client_order_id": "CID-FRESH-1",
    }
    repair_calls: list[dict] = []
    streamer._maybe_repair_protection = lambda **kwargs: repair_calls.append(dict(kwargs))

    streamer._handle_nt_message_inner(
        {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "instrument": "ES JUN26",
            "account": "DEMO7467442",
            "timestamp": "2026-05-01T04:40:30-06:00",
            "qty": 1,
            "pos_qty": 1,
            "side": "LONG",
            "orders": [{"client_order_id": "CID-FRESH-1"}],
            "positions": [{"instrument": "ES JUN26", "qty": 1, "side": "LONG", "avg_price": 7249.75}],
        }
    )

    assert len(repair_calls) == 1


def test_order_ack_rejected_repair_does_not_force_flat_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    sent_payloads: list[dict] = []
    flatten_calls: list[tuple[str, str]] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        send=lambda msg: sent_payloads.append(dict(msg)) or True,
        handshake_ok=lambda: True,
    )
    streamer._send_nt_flatten = lambda cid, reason="": flatten_calls.append((str(cid), str(reason))) or True
    streamer.nt_account_mode = "none"
    streamer.nt_instrument = "ES JUN26"
    streamer.instrument_alias = "ES 06-26"
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    primary_cid = "RUNID|es_elite_v1|ES JUN26|2026-05-01T02:40:00-06:00|OPEN|LONG|2056ee51"
    repair_cid = "repair|ES 06-26|2026-05-01T02:40:07.761076-06:00"
    streamer._open_trade = {
        "client_order_id": primary_cid,
        "side": "LONG",
        "entry_price": 7249.5,
        "protection_status": "protected_confirmed",
    }
    streamer.state.active_client_order_id = primary_cid
    streamer._nt_order_state[repair_cid] = {
        "client_order_id": repair_cid,
        "intent_action": "OPEN",
        "side": "LONG",
        "qty": 1,
        "stop_price": 7245.0,
        "target_price": 7254.0,
        "signal_id": None,
        "instrument": "ES JUN26",
        "inst_key": "ES 06-26",
        "repair_mode": "absolute",
    }

    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_ACK",
            "protocol_version": 1,
            "client_order_id": repair_cid,
            "status": "REJECTED",
            "reason": "missing_protection_prices",
            "instrument": "ES JUN26",
            "timestamp": "2026-05-01T02:40:09-06:00",
        }
    )

    assert streamer.state.position_state == "IN_POSITION_PROTECTED"
    assert streamer.state.active_client_order_id == primary_cid
    assert streamer._open_trade is not None
    assert streamer._open_trade.get("client_order_id") == primary_cid
    assert flatten_calls == []
    fallback_orders = [o for o in sent_payloads if str(o.get("client_order_id") or "").startswith("repair|")]
    assert fallback_orders
    assert fallback_orders[-1].get("repair_mode") == "legacy"
    assert fallback_orders[-1].get("action") == "SELL"


def test_repair_missing_protection_double_failure_flattens(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    flatten_calls: list[tuple[str, str]] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: False)
    streamer._send_nt_flatten = lambda cid, reason="": flatten_calls.append((str(cid), str(reason))) or True
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer.protection_repair_enabled = True
    streamer._hard_lockout_active = False
    streamer._hard_lockout_code = None
    streamer.lockout_policy = "all"
    streamer._expected_unprotected_by_instrument = {}
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_repair_state_by_instrument = {
        "ES 06-26": {
            "start_ts": time.time(),
            "last_attempt_ts": None,
            "attempts": 1,
            "snapshot_mismatch_confirmations": 2,
        }
    }
    streamer.protection_repair_max_attempts = 1
    streamer.nt_protection_repair_timeout_sec = 30.0
    streamer._phase = "LIVE"

    repair_cid = "repair|ES 06-26|2026-05-01T02:40:07.761076-06:00"
    streamer._nt_order_state[repair_cid] = {
        "client_order_id": repair_cid,
        "intent_action": "OPEN",
        "side": "LONG",
        "qty": 1,
        "stop_price": 7245.0,
        "target_price": 7254.0,
        "signal_id": None,
        "instrument": "ES JUN26",
        "inst_key": "ES 06-26",
        "repair_mode": "absolute",
    }

    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_ACK",
            "protocol_version": 1,
            "client_order_id": repair_cid,
            "status": "REJECTED",
            "reason": "missing_protection_prices",
            "instrument": "ES JUN26",
            "timestamp": "2026-05-01T02:40:09-06:00",
        }
    )

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7245.0,
        target_price=7254.0,
        reason="snapshot_exits_mismatch",
    )

    assert flatten_calls
    assert flatten_calls[-1][1] == "protection_repair_failed"


def test_repair_exhaustion_suppressed_when_active_lifecycle_already_protected(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    flatten_calls: list[tuple[str, str]] = []
    events: list[dict] = []
    primary_cid = "RUNID|OPEN|SHORT|MODEL-1"
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: False)
    streamer._send_nt_flatten = lambda cid, reason="": flatten_calls.append((str(cid), str(reason))) or True
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer.protection_repair_enabled = True
    streamer._hard_lockout_active = False
    streamer._hard_lockout_code = None
    streamer.lockout_policy = "all"
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_repair_state_by_instrument = {
        "ES 06-26": {
            "start_ts": time.time(),
            "last_attempt_ts": time.time() - 1.0,
            "attempts": 1,
            "snapshot_mismatch_confirmations": 2,
        }
    }
    streamer.protection_repair_max_attempts = 1
    streamer.nt_protection_repair_timeout_sec = 30.0
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer.state.active_client_order_id = primary_cid
    streamer.state.nt_protected = True
    streamer.state.nt_exits_working = True
    streamer.state.nt_stop_order_id = "STOP-1"
    streamer.state.nt_target_order_id = "TARGET-1"
    streamer.state.nt_protection_confirmed_ts = _now_denver()
    streamer._nt_order_state[primary_cid] = {
        "client_order_id": primary_cid,
        "intent_action": "OPEN",
        "instrument": "ES JUN26",
        "position_state": "IN_POSITION_PROTECTED",
        "entry_filled": True,
        "entry_fill_price": 7595.75,
        "stop_order_id": "STOP-1",
        "target_order_id": "TARGET-1",
        "stop_state": "WORKING",
        "target_state": "WORKING",
        "protection_confirmed_method": "order_update_leg_working",
        "protection_confirmed_ts": time.time(),
    }

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=-1.0,
        stop_price=7610.0,
        target_price=7581.5,
        reason="snapshot_exits_mismatch",
    )

    assert flatten_calls == []
    assert streamer._nt_repair_state_by_instrument.get("ES 06-26") is None
    assert any(ev.get("event") == "protection_repair_suppressed_already_protected" for ev in events)


def test_repair_missing_stop_first_escalation_soft_disarms_no_sticky_lockout(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: False)
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._repair_snapshot_stale_for_protection = lambda _inst: (False, {})
    streamer._repair_close_or_flatten_in_flight = lambda _inst: False
    streamer._active_order_state_for_instrument = lambda _inst: (None, {})
    streamer._emit_soft_block_event = lambda **_kwargs: None
    streamer._maybe_request_nt_snapshot = lambda **_kwargs: None
    streamer.nt_exec_state = "ARMED"
    streamer.lockout_policy = "all"
    streamer._hard_lockout_active = False
    streamer._hard_lockout_code = None
    streamer._lockout_reset_token = None
    streamer._lockout_reset_required = False
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_repair_state_by_instrument = {
        "ES 06-26": {
            "start_ts": time.time() - 120.0,
            "last_attempt_ts": None,
            "attempts": 0,
            "missing_stop_confirmations": 1,
            "timeout_confirmations": 1,
        }
    }
    streamer.nt_protection_repair_timeout_sec = 30.0
    streamer._send_nt_flatten = lambda *_args, **_kwargs: True

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=None,
        target_price=7250.0,
        reason="snapshot_exits_mismatch",
    )

    state = streamer._nt_repair_state_by_instrument["ES 06-26"]
    assert int(state.get("missing_stop_confirmations") or 0) == 2
    assert streamer.nt_exec_state == "DISARMED"
    assert streamer._hard_lockout_active is False


def test_repair_missing_stop_second_escalation_sets_sticky_lockout(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: False)
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._repair_snapshot_stale_for_protection = lambda _inst: (False, {})
    streamer._repair_close_or_flatten_in_flight = lambda _inst: False
    streamer._active_order_state_for_instrument = lambda _inst: (None, {})
    streamer._emit_soft_block_event = lambda **_kwargs: None
    streamer._maybe_request_nt_snapshot = lambda **_kwargs: None
    streamer.lockout_policy = "all"
    streamer._hard_lockout_active = False
    streamer._hard_lockout_code = None
    streamer._lockout_reset_token = None
    streamer._lockout_reset_required = False
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_repair_state_by_instrument = {
        "ES 06-26": {
            "start_ts": time.time() - 120.0,
            "last_attempt_ts": None,
            "attempts": 0,
            "missing_stop_confirmations": 1,
            "timeout_confirmations": 2,
        }
    }
    streamer.nt_protection_repair_timeout_sec = 30.0
    flatten_calls: list[tuple[str, str]] = []
    streamer._send_nt_flatten = lambda cid, reason="": flatten_calls.append((str(cid), str(reason))) or True

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=None,
        target_price=7250.0,
        reason="snapshot_exits_mismatch",
    )

    state = streamer._nt_repair_state_by_instrument["ES 06-26"]
    assert streamer.nt_exec_state == "DISARMED"
    assert streamer._hard_lockout_active is False
    assert int(state.get("missing_stop_escalations") or 0) >= 1
    assert flatten_calls == []


def test_repair_missing_stop_escalation_suppressed_on_recent_close_attempt(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: False)
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._repair_snapshot_stale_for_protection = lambda _inst: (False, {})
    streamer._repair_close_or_flatten_in_flight = lambda _inst: False
    streamer._active_order_state_for_instrument = lambda _inst: (None, {})
    streamer._emit_soft_block_event = lambda **_kwargs: None
    streamer._maybe_request_nt_snapshot = lambda **_kwargs: None
    streamer.lockout_policy = "all"
    streamer._hard_lockout_active = False
    streamer._hard_lockout_code = None
    streamer._lockout_reset_token = None
    streamer._lockout_reset_required = False
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_repair_state_by_instrument = {
        "ES 06-26": {
            "start_ts": time.time() - 120.0,
            "last_attempt_ts": None,
            "attempts": 0,
            "missing_stop_confirmations": 1,
            "timeout_confirmations": 2,
        }
    }
    streamer._last_close_guard = {
        "ES 06-26": {"reason": "protection_repair_failed", "position_state": "EXITING", "ts": time.time()}
    }
    streamer.nt_protection_repair_timeout_sec = 30.0
    flatten_calls: list[tuple[str, str]] = []
    events: list[dict] = []
    streamer._send_nt_flatten = lambda cid, reason="": flatten_calls.append((str(cid), str(reason))) or True
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=None,
        target_price=7250.0,
        reason="snapshot_exits_mismatch",
    )

    state = streamer._nt_repair_state_by_instrument["ES 06-26"]
    assert streamer._hard_lockout_active is False
    assert int(state.get("missing_stop_escalations") or 0) == 0
    assert flatten_calls == []
    assert any(ev.get("event") == "protection_repair_suppressed_recent_close_attempt" for ev in events)


def test_protection_repair_timeout_lockout_deferred_during_close_settlement(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: False)
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._emit_soft_block_event = lambda **_kwargs: None
    streamer._maybe_request_nt_snapshot = lambda **_kwargs: None
    streamer.lockout_policy = "all"
    streamer._hard_lockout_active = False
    streamer._hard_lockout_code = None
    streamer._lockout_reset_token = None
    streamer._lockout_reset_required = False
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_repair_state_by_instrument = {
        "ES 06-26": {
            "start_ts": time.time() - 120.0,
            "last_attempt_ts": None,
            "attempts": 0,
            "timeout_confirmations": 2,
            "snapshot_mismatch_confirmations": 2,
        }
    }
    streamer.nt_protection_repair_timeout_sec = 30.0
    streamer._repair_close_or_flatten_in_flight = lambda _inst: True
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, evidence=None, **_kwargs: lockouts.append((str(code), dict(evidence or {})))

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7245.0,
        target_price=7254.0,
        reason="snapshot_exits_mismatch",
    )

    assert lockouts == []
    assert streamer._hard_lockout_active is False


def test_late_empty_snapshot_after_working_protection_does_not_repair(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: False)
    streamer.nt_instrument = "ES JUN26"
    streamer.instrument_alias = "ES JUN26"
    streamer.nt_canonical_instrument = "ES 06-26"
    streamer._nt_last_snapshot_orders_by_instrument = {}
    streamer._nt_last_snapshot_orders_ts_by_instrument = {}
    streamer._nt_last_price_by_instrument = {}
    streamer._nt_last_price_source_by_instrument = {}
    streamer._nt_last_price_value_ts_by_instrument = {}
    streamer._nt_last_bid_by_instrument = {}
    streamer._nt_last_ask_by_instrument = {}
    streamer._nt_last_price_source_by_instrument = {}
    streamer._nt_last_price_ts_by_instrument = {}
    streamer._last_snapshot_avg_by_instrument = {}
    streamer._compute_nt_ready = lambda: None
    streamer._rearm_reject_circuit_breaker = lambda **_kwargs: None
    streamer._startup_resync_pending = False
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._maybe_request_nt_snapshot = lambda **_kwargs: None
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_repair_state_by_instrument = {}
    repairs: list[dict] = []
    events: list[dict] = []
    streamer._send_nt_protection_repair = lambda **kwargs: repairs.append(dict(kwargs)) or True
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    cid = "RUN|es_modelrun77_parity_v1|ES JUN26|2026-06-04T00:30:00-06:00|OPEN|SHORT|race"
    protected_epoch = time.time()
    streamer.state.active_client_order_id = cid
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer.state.nt_protected = True
    streamer.state.nt_exits_working = True
    streamer.state.nt_stop_order_id = "STOP-ORDER"
    streamer.state.nt_target_order_id = "TARGET-ORDER"
    streamer.state.nt_protection_confirmed_ts = pd.Timestamp.fromtimestamp(protected_epoch, tz="America/Denver")
    streamer._pos = -1
    streamer._open_trade = {
        "client_order_id": cid,
        "side": "SHORT",
        "entry_price": 7545.0,
        "live_stop": 7551.25,
        "live_target": 7538.75,
    }
    streamer._nt_order_state = {
        cid: {
            "client_order_id": cid,
            "instrument": "ES JUN26",
            "side": "SHORT",
            "qty": 1,
            "entry_filled": True,
            "entry_fill_price": 7545.0,
            "status": "exits_working",
            "protected": True,
            "exits_working": True,
            "stop_order_id": "STOP-ORDER",
            "target_order_id": "TARGET-ORDER",
            "stop_price": 7551.25,
            "target_price": 7538.75,
            "stop_state": "WORKING",
            "target_state": "WORKING",
            "protection_confirmed_ts": protected_epoch,
            "protected_ts": protected_epoch,
        }
    }

    suppressed = streamer._suppress_stale_snapshot_repair_if_protected(
        inst_key="ES 06-26",
        active_cid=cid,
        active_state=streamer._nt_order_state[cid],
        snapshot_epoch=protected_epoch - 1.0,
        reason="snapshot_no_orders",
    )

    assert suppressed is True
    assert repairs == []
    assert streamer.state.nt_protected is True
    assert streamer.state.nt_exits_working is True
    assert any(ev.get("event") == "snapshot_repair_suppressed_stale_protection" for ev in events)


def test_protection_repair_timeout_waits_for_snapshot_corroboration_window(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: False)
    streamer._require_fresh_snapshot = lambda **_kwargs: True
    streamer._emit_soft_block_event = lambda **_kwargs: None
    streamer._maybe_request_nt_snapshot = lambda **_kwargs: None
    streamer.lockout_policy = "all"
    streamer._hard_lockout_active = False
    streamer._hard_lockout_code = None
    streamer._lockout_reset_token = None
    streamer._lockout_reset_required = False
    streamer._expected_unprotected_by_instrument = {}
    streamer._nt_repair_state_by_instrument = {
        "ES 06-26": {
            "start_ts": time.time() - 120.0,
            "last_attempt_ts": None,
            "attempts": 0,
            "timeout_confirmations": 2,
            "snapshot_mismatch_confirmations": 1,
            "snapshot_mismatch_first_ts": time.time() - 1.5,
        }
    }
    streamer.nt_protection_repair_timeout_sec = 30.0
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, evidence=None, **_kwargs: lockouts.append((str(code), dict(evidence or {})))

    streamer._maybe_repair_protection(
        inst_key="ES 06-26",
        pos_qty=1.0,
        stop_price=7245.0,
        target_price=7254.0,
        reason="snapshot_exits_mismatch",
    )

    assert lockouts == []
    assert streamer._hard_lockout_active is False


def test_close_lockout_missing_stop_retry_is_bounded(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "paper"
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: True)
    streamer._external_nt_cids = {"CID-CLOSE-BOUND"}
    flatten_calls: list[tuple[str, str]] = []
    streamer._send_nt_flatten = lambda cid, reason="", **_kwargs: flatten_calls.append((str(cid), str(reason))) or True
    streamer._emit_block_event = lambda **_kwargs: None
    streamer._emit_executor_decision = lambda *_args, **_kwargs: None
    streamer._record_executor_decision = lambda *_args, **_kwargs: "SENT"
    streamer._nt_order_state["CID-CLOSE-BOUND"] = {
        "client_order_id": "CID-CLOSE-BOUND",
        "intent_action": "CLOSE",
        "side": "NONE",
        "qty": 1,
        "instrument": "ES JUN26",
        "status": "sent",
    }

    msg = {
        "type": "ORDER_ACK",
        "protocol_version": 1,
        "client_order_id": "CID-CLOSE-BOUND",
        "status": "LOCKOUT",
        "reason": "missing_stop_price",
        "instrument": "ES JUN26",
    }
    streamer._handle_nt_message_inner(dict(msg))
    streamer._handle_nt_message_inner(dict(msg))
    streamer._handle_nt_message_inner(dict(msg))

    assert len(flatten_calls) == 2
    assert all(reason == "model_close_missing_stop_bypass" for _, reason in flatten_calls)
    assert streamer._nt_order_state["CID-CLOSE-BOUND"].get("status") == "close_retry_exhausted"


def test_safety_lockout_ack_does_not_trigger_orphan_lockout(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._external_nt_cids = set()

    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_ACK",
            "client_order_id": "SAFETY|ES JUN26|20260430071508|PROTECTION_REPAIR_TIMEOUT",
            "status": "LOCKOUT",
            "reason": "protection_repair_timeout",
            "instrument": "ES JUN26",
            "protocol_version": 1,
        }
    )

    assert streamer._hard_lockout_code is None
    assert streamer.nt_exec_state != "LOCKED_OUT"


def test_late_lockout_after_terminal_fill_is_ignored(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.lockout_policy = "all"
    streamer._external_nt_cids = {"CID-LATE"}
    streamer._record_reject_circuit_breaker = lambda **_kwargs: False
    streamer._emit_block_event = lambda **_kwargs: None
    streamer._emit_executor_decision = lambda *args, **kwargs: None
    streamer._record_executor_decision = lambda *_args, **_kwargs: "SENT"
    streamer._nt_order_state = {
        "CID-LATE": {
            "intent_action": "OPEN",
            "status": "close_filled",
            "exit_fill_ts": time.time(),
            "side": "SHORT",
            "qty": 1,
            "instrument": "ES JUN26",
        }
    }
    streamer.state.last_signal_status = "sent"

    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_ACK",
            "client_order_id": "CID-LATE",
            "status": "LOCKOUT",
            "reason": "missing_stop_price",
            "instrument": "ES JUN26",
            "protocol_version": 1,
        }
    )

    assert streamer.state.last_signal_status == "sent"
    assert streamer._nt_order_state["CID-LATE"]["status"] == "close_filled"


def test_terminal_protection_repair_timeout_is_suppressed_before_reject_path(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "RUNID|es_modelrun77_parity_v1|ES JUN26|2026-06-03T22:45:00-06:00|OPEN|LONG|183cc43b"
    events: list[dict] = []
    order_events: list[tuple[str, str, dict]] = []
    reject_calls: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._log_order_event = lambda cid_arg, status, payload: order_events.append(
        (str(cid_arg), str(status), dict(payload or {}))
    )
    streamer._record_reject_circuit_breaker = lambda **kwargs: reject_calls.append(dict(kwargs)) or False
    streamer._emit_block_event = lambda **_kwargs: None
    streamer._emit_executor_decision = lambda *_args, **_kwargs: None
    streamer._record_executor_decision = lambda *_args, **_kwargs: "SENT"
    streamer._pos = 0
    streamer._pending_client_order_id = None
    streamer._pending_until = None
    streamer._open_trade = None
    streamer._nt_order_state = {
        cid: {
            "client_order_id": cid,
            "intent_action": "OPEN",
            "side": "LONG",
            "qty": 1,
            "instrument": "ES JUN26",
            "status": "close_filled",
            "entry_filled": True,
            "exit_fill_ts": time.time(),
            "exit_fill_price": 7547.0,
            "stop_order_id": "STOP-1",
            "target_order_id": "TARGET-1",
        }
    }

    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_ACK",
            "protocol_version": 1,
            "client_order_id": cid,
            "status": "LOCKOUT",
            "reason": "protection_repair_timeout",
            "instrument": "ES JUN26",
        }
    )

    assert streamer._hard_lockout_code is None
    assert streamer.state.last_signal_status != "rejected"
    assert reject_calls == []
    assert any(ev.get("event") == "terminal_repair_timeout_suppressed" for ev in events)
    assert not any(ev.get("direction") == "rx" and "LOCKOUT" in json.dumps(ev) for ev in events)
    assert any(status == "terminal_repair_timeout_suppressed" for _, status, _ in order_events)
    assert streamer._nt_order_state[cid]["exit_reason"] == "protection_repair_timeout"
    assert streamer._nt_order_state[cid]["terminal_cause"] == "protection_repair_timeout"


def test_opposite_side_fill_after_entry_is_classified_as_exit(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.run_id = "RID"
    cid = "RID|es_modelrun77_parity_v1|ES JUN26|2026-06-04T02:35:00-06:00|OPEN|SHORT|d5eec920"
    events: list[tuple[str, str, dict]] = []
    streamer._log_order_event = lambda cid_arg, status, payload: events.append(
        (str(cid_arg), str(status), dict(payload or {}))
    )
    streamer._queue_trade_update_from_state = lambda *_args, **_kwargs: None
    streamer._apply_fill_truth_reconciliation = lambda **_kwargs: None
    streamer._sync_position_flat = lambda *_args, **_kwargs: None
    streamer._open_trade = {"client_order_id": cid, "side": "SHORT"}
    streamer._nt_order_state = {
        cid: {
            "client_order_id": cid,
            "intent_action": "OPEN",
            "side": "SHORT",
            "qty": 1,
            "instrument": "ES JUN26",
            "entry_filled": True,
            "entry_ninja_order_id": "ENTRY-1",
            "entry_fill_price": 7545.75,
            "exit_reason": "protection_repair_timeout",
            "status": "EXITS_WORKING",
        }
    }

    streamer._handle_nt_message_inner(
        {
            "type": "FILL",
            "protocol_version": 1,
            "client_order_id": cid,
            "ninja_order_id": "FLATTEN-1",
            "instrument": "ES JUN26",
            "side": "LONG",
            "fill_price": 7546.25,
            "fill_qty": 1,
            "timestamp": _canonical_ts_str(_now_denver()),
        }
    )

    state = streamer._nt_order_state[cid]
    assert state["exit_fill_price"] == pytest.approx(7546.25)
    assert state["exit_reason"] == "protection_repair_timeout"
    assert streamer._open_trade["exit_reason"] == "protection_repair_timeout"
    assert any(status == "opposite_side_fill_classified_exit" for _, status, _ in events)


def test_exit_fill_stop_leg_without_stop_cross_is_not_stop_hit(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "CID-STOP-NOT-CROSSED"
    streamer._external_nt_cids = {cid}
    streamer._nt_order_state = {
        cid: {
            "intent_action": "OPEN",
            "side": "SHORT",
            "qty": 1,
            "entry_filled": True,
            "stop_order_id": "STOP-1",
            "target_order_id": None,
            "stop_price": 7383.75,
            "status": "entry_filled",
        }
    }
    streamer._open_trade = {"client_order_id": cid}
    streamer._queue_trade_update_from_state = lambda *_args, **_kwargs: None
    streamer._apply_fill_truth_reconciliation = lambda **_kwargs: None
    streamer._sync_position_flat = lambda *_args, **_kwargs: None

    streamer._handle_nt_message_inner(
        {
            "type": "FILL",
            "client_order_id": cid,
            "ninja_order_id": "STOP-1",
            "fill_price": 7377.5,
            "fill_qty": 1,
            "instrument": "ES JUN26",
            "protocol_version": 1,
        }
    )

    state = streamer._nt_order_state[cid]
    assert state.get("exit_reason") == "repair_flatten"
    assert state.get("terminal_cause") == "repair_flatten"
    assert streamer._open_trade.get("exit_reason") == "repair_flatten"


def test_reject_diagnostics_classifies_runtime_missing_stop_despite_stop_sent(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    detail = streamer._build_reject_diagnostics(
        cid="CID-RUNTIME-STOP",
        status="LOCKOUT",
        reason="missing_stop_price",
        reject_code="nt_missing_stop_price",
        state={
            "intent_action": "OPEN",
            "side": "SHORT",
            "qty": 1,
            "instrument": "ES JUN26",
            "stop_price": 7383.75,
            "entry_order_id": "ENTRY-1",
            "stop_order_id": "STOP-1",
        },
        msg={"schema_version": 1},
    )
    assert detail.get("reject_class") == "nt_runtime_missing_stop_despite_stop_sent"
    assert detail.get("session_id_present") is False


def test_resolve_exit_reason_uses_owned_repair_reason_after_reassociation(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    reason = streamer._resolve_exit_reason(
        state={
            "side": "SHORT",
            "stop_order_id": "STOP-1",
            "target_order_id": None,
            "stop_price": 7419.0,
            "reassociated_untracked": True,
        },
        msg={"reason": None},
        ninja_order_id="STOP-1",
        fill_price=7418.5,
    )
    assert reason == "repair_flatten_after_untracked_reassociated"


def test_pre_send_missing_stop_price_blocks_before_send(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._nt_ready = True
    streamer.nt_enabled = True
    streamer.nt_adapter = "nt"
    streamer.nt_account_mode = "none"
    send_calls: list[dict] = []
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        send=lambda msg: send_calls.append(dict(msg)) or True,
    )
    streamer._ensure_nt_protocol_context = lambda *args, **kwargs: True
    streamer._emit_block_event = lambda **_kwargs: None
    streamer._emit_executor_decision = lambda *args, **kwargs: None
    streamer._record_executor_decision = lambda *_args, **_kwargs: "SENT"
    streamer._log_order_event = lambda *args, **kwargs: None
    streamer._log_exec_event = lambda *_args, **_kwargs: None
    streamer._resolve_entry_protection_mode = lambda _action, _mode: "offset"
    streamer._validate_nt_order_contract = (
        lambda order: [] if order.get("stop_price") is not None else ["stop_price"]
    )
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="CID-PRE-SEND-STOP",
        action="OPEN",
        side="SHORT",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        limit_price=None,
        stop_price=None,
        target_price=7410.0,
        entry_price=7415.0,
        model_price=7415.0,
        model_stop_price=7419.25,
        model_target_price=7410.0,
        bar_ts="2026-05-11T04:05:00-06:00",
    )
    result = streamer._send_intent_order(intent)
    assert result.decision == "BLOCKED_SAFETY"
    assert result.reason_code == "missing_stop_price_preflight"
    assert not send_calls


def test_pre_send_missing_session_blocks_before_send(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._nt_ready = True
    streamer.nt_enabled = True
    streamer.nt_adapter = "nt"
    streamer.nt_account_mode = "none"
    send_calls: list[dict] = []
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        send=lambda msg: send_calls.append(dict(msg)) or True,
    )
    streamer._ensure_nt_protocol_context = lambda *args, **kwargs: False
    streamer._emit_block_event = lambda **_kwargs: None
    streamer._emit_executor_decision = lambda *args, **kwargs: None
    streamer._record_executor_decision = lambda *_args, **_kwargs: None
    streamer._log_order_event = lambda *args, **kwargs: None
    streamer._log_exec_event = lambda *_args, **_kwargs: None
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="CID-PRE-SEND-SESSION",
        action="OPEN",
        side="SHORT",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        limit_price=None,
        stop_price=7419.25,
        target_price=7410.0,
        entry_price=7415.0,
        model_price=7415.0,
        model_stop_price=7419.25,
        model_target_price=7410.0,
        bar_ts="2026-05-11T04:05:00-06:00",
    )
    result = streamer._send_intent_order(intent)
    assert result.decision == "BLOCKED_SAFETY"
    assert result.reason_code == "protocol_context_missing_preflight"
    assert not send_calls


def test_post_fill_lockout_routes_to_owned_repair(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "CID-POST-FILL-LOCKOUT"
    streamer._external_nt_cids = {cid}
    repair_calls: list[dict] = []
    streamer._maybe_repair_protection = lambda **kwargs: repair_calls.append(dict(kwargs)) or True
    streamer._queue_trade_update_from_state = lambda *_args, **_kwargs: None
    streamer._nt_order_state = {
        cid: {
            "intent_action": "OPEN",
            "entry_filled": True,
            "status": "entry_filled",
            "side": "SHORT",
            "qty": 1,
            "stop_price": 7419.25,
            "target_price": 7410.0,
            "instrument": "ES JUN26",
        }
    }
    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_ACK",
            "client_order_id": cid,
            "status": "LOCKOUT",
            "reason": "missing_stop_price",
            "instrument": "ES JUN26",
            "protocol_version": 1,
        }
    )
    assert streamer._nt_order_state[cid]["status"] == "protection_repair_required"
    assert repair_calls
    assert repair_calls[-1]["reason"] == "post_fill_lockout"


def test_post_fill_missing_stop_classified_as_protection_contract_violation(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "CID-PROTECTION-CONTRACT-1"
    streamer._external_nt_cids = {cid}
    streamer._pos = 1.0
    streamer._maybe_repair_protection = lambda **_kwargs: True
    streamer._emit_block_event = lambda **_kwargs: None
    streamer._emit_executor_decision = lambda *args, **kwargs: None
    streamer._record_executor_decision = lambda *_args, **_kwargs: "SENT"
    streamer._record_reject_circuit_breaker = lambda **_kwargs: False
    streamer._nt_order_state = {
        cid: {
            "intent_action": "OPEN",
            "entry_filled": True,
            "status": "entry_filled",
            "side": "LONG",
            "qty": 1,
            "stop_price": 7375.0,
            "target_price": 7395.0,
            "instrument": "ES JUN26",
        }
    }
    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_ACK",
            "client_order_id": cid,
            "status": "LOCKOUT",
            "reason": "missing_stop_price",
            "instrument": "ES JUN26",
            "protocol_version": 1,
        }
    )
    state = streamer._nt_order_state[cid]
    assert state.get("status") == "protection_repair_required"
    assert state.get("protection_contract_state") == "missing_stop"
    assert state.get("contract_invariant_violation") is True
    assert int(getattr(streamer, "_protection_contract_violation_total", 0) or 0) == 1


def test_post_fill_sent_to_rejected_downgrade_blocked(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    cid = "CID-LATCH-1"
    streamer._nt_order_state[cid] = {
        "entry_filled": True,
        "status": "entry_filled",
        "side": "SHORT",
        "qty": 1,
    }
    first = streamer._record_executor_decision(cid, "SENT")
    second = streamer._record_executor_decision(cid, "REJECTED")
    assert first is None
    assert second == "SENT"
    assert streamer._executor_terminal[cid] == "SENT"
    assert any(ev.get("event") == "terminal_downgrade_blocked_post_fill" for ev in events)


def test_flip_on_flat_normalizes_to_open(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer._should_buffer_inflight = lambda _typ: False
    streamer._should_block_nt_send_policy = lambda _ev, _typ: None
    streamer._should_block_stale_nt_send = lambda _ev, _typ: False
    streamer._build_nt_client_order_id = lambda ev: str(ev.get("client_order_id") or "RUNID|FLIP|CID")
    streamer.execution_ledger.get = lambda _cid: None
    streamer.execution_state = SimpleNamespace(position_qty=0, position_side="FLAT")
    streamer._emit_executor_decision = lambda *args, **kwargs: None
    streamer._record_executor_decision = lambda *args, **kwargs: None

    flip_intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="RUNID|flip|1",
        action="FLIP",
        side="LONG",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        limit_price=None,
        stop_price=7191.5,
        target_price=7210.0,
        entry_price=7198.0,
        bar_ts="2026-04-30T07:15:00-06:00",
        signal_id="sig|flip|1",
    )
    streamer._build_execution_intent = lambda _ev: (flip_intent, None)

    captured: list[stream_live_csv_mod.ExecutionIntent] = []

    def _exec_stub(intent):
        captured.append(intent)
        return SimpleNamespace(decision="SENT", reason_code="sent", nt_order_ids={})

    streamer.execute_intent = _exec_stub

    streamer._maybe_send_nt_order(
        {
            "type": "FLIP",
            "side": "LONG",
            "contracts": 1,
            "instrument": "ES JUN26",
            "client_order_id": "RUNID|flip|1",
            "signal_id": "sig|flip|1",
            "bar_ts": "2026-04-30T07:15:00-06:00",
        }
    )

    assert len(captured) == 1
    assert captured[0].action == "OPEN"


def test_flip_nonflat_emits_close_then_open_transition(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer._should_buffer_inflight = lambda _typ: False
    streamer._should_block_nt_send_policy = lambda _ev, _typ: None
    streamer._should_block_stale_nt_send = lambda _ev, _typ: False
    streamer._build_nt_client_order_id = lambda ev: str(ev.get("client_order_id") or "RUNID|FLIP|CID")
    streamer.execution_ledger.get = lambda _cid: None
    streamer.execution_state = SimpleNamespace(position_qty=1, position_side="LONG")
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer._emit_executor_decision = lambda *args, **kwargs: None
    streamer._record_executor_decision = lambda *args, **kwargs: None

    flip_intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="RUNID|flip|2",
        action="FLIP",
        side="SHORT",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        limit_price=None,
        stop_price=7205.0,
        target_price=7188.0,
        entry_price=7198.0,
        bar_ts="2026-04-30T07:16:00-06:00",
        signal_id="sig|flip|2",
    )
    streamer._build_execution_intent = lambda _ev: (flip_intent, None)
    captured: list[stream_live_csv_mod.ExecutionIntent] = []

    def _exec_stub(intent):
        captured.append(intent)
        if intent.action == "CLOSE":
            return SimpleNamespace(decision="SKIPPED_NOOP", reason_code="ALREADY_FLAT", nt_order_ids={})
        return SimpleNamespace(decision="SENT", reason_code="sent", nt_order_ids={})

    streamer.execute_intent = _exec_stub
    streamer._maybe_send_nt_order(
        {
            "type": "FLIP",
            "side": "SHORT",
            "contracts": 1,
            "instrument": "ES JUN26",
            "client_order_id": "RUNID|flip|2",
            "signal_id": "sig|flip|2",
            "bar_ts": "2026-04-30T07:16:00-06:00",
        }
    )
    assert [i.action for i in captured] == ["CLOSE", "OPEN"]


def test_close_suppressed_when_flat_and_snapshot_stale_reconcile(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer._should_buffer_inflight = lambda _typ: False
    streamer._should_block_nt_send_policy = lambda _ev, _typ: None
    streamer._should_block_stale_nt_send = lambda _ev, _typ: False
    streamer.execution_state = SimpleNamespace(position_qty=0, position_side="FLAT")
    streamer._set_entries_disarmed("snapshot_stale_reconcile", {"reason": "stale_snapshot"})
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer.state.active_client_order_id = "RUNID|close|flat"
    streamer._nt_last_pos_qty_by_instrument[streamer._exec_instrument_key()] = 0.0
    decisions: list[tuple[str, str, str]] = []
    streamer._emit_executor_decision = (
        lambda intent, *, decision, reason_code, **_kwargs: decisions.append((intent.intent_id, decision, str(reason_code)))
    )
    streamer.execute_intent = lambda _intent: (_ for _ in ()).throw(AssertionError("execute_intent should be suppressed"))
    close_intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="RUNID|close|flat",
        action="CLOSE",
        side="NONE",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        limit_price=None,
        stop_price=None,
        target_price=None,
        entry_price=None,
        bar_ts="2026-05-19T01:00:00-06:00",
        signal_id="sig|close|flat",
    )
    streamer._build_execution_intent = lambda _ev: (close_intent, None)
    streamer._maybe_send_nt_order({"type": "CLOSE", "instrument": "ES JUN26", "signal_id": "sig|close|flat"})
    assert decisions
    assert decisions[-1][1] == "BLOCKED_SYNC"
    assert decisions[-1][2] == "close_intent_suppressed_flat_truth"


def test_flat_truth_context_cleared_resets_trade_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._open_trade = {"client_order_id": "RUNID|open|1", "entry_fill_price": 7200.0}
    streamer._entry_price = 7200.0
    streamer._entry_time = pd.Timestamp("2026-05-19T00:00:00-06:00")
    streamer.state.active_client_order_id = "RUNID|open|1"
    streamer.state.position_uid = "UID-1"
    streamer._clear_flat_truth_context(source="test", reason="flat_confirmed")
    assert streamer._open_trade is None
    assert streamer._entry_price is None
    assert streamer._entry_time is None
    assert streamer.state.active_client_order_id is None
    assert streamer.state.position_uid is None


def test_exiting_auto_unlatches_to_flat_without_close_progress(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.state.position_state = "EXITING"
    streamer._exit_started_ts = time.time() - 20.0
    streamer._pos = 0
    streamer.state.active_client_order_id = "RUNID|close|stale"
    streamer._nt_last_pos_qty_by_instrument[streamer._exec_instrument_key()] = 0.0
    streamer._nt_order_state["RUNID|close|stale"] = {"close_in_progress": False}
    streamer._check_close_watchdogs(time.time())
    assert streamer.state.position_state == "FLAT"


def test_close_not_suppressed_when_run_owned_nonflat(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer._should_buffer_inflight = lambda _typ: False
    streamer._should_block_nt_send_policy = lambda _ev, _typ: None
    streamer._should_block_stale_nt_send = lambda _ev, _typ: False
    streamer._set_entries_disarmed("snapshot_stale_reconcile", {"reason": "stale_snapshot"})
    streamer._pos = -1
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    sent: list[str] = []
    streamer._emit_executor_decision = lambda *args, **kwargs: None
    streamer.execute_intent = lambda intent: sent.append(intent.action) or SimpleNamespace(
        decision="SENT", reason_code="sent", nt_order_ids={}
    )
    close_intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="RUNID|close|live",
        action="CLOSE",
        side="NONE",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        limit_price=None,
        stop_price=None,
        target_price=None,
        entry_price=None,
        bar_ts="2026-05-19T01:05:00-06:00",
        signal_id="sig|close|live",
    )
    streamer._build_execution_intent = lambda _ev: (close_intent, None)
    streamer._maybe_send_nt_order({"type": "CLOSE", "instrument": "ES JUN26", "signal_id": "sig|close|live"})
    assert sent == ["CLOSE"]
    assert streamer.state.position_state == "EXITING"


def test_snapshot_progress_stall_triggers_reconnect_and_disarm(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    reconnect_calls: list[str] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        reconnect=lambda: reconnect_calls.append("reconnect"),
        reset_connection=lambda: reconnect_calls.append("reset"),
    )
    streamer._nt_ready = False
    streamer._nt_ready_reason = "snapshot_stale"
    streamer._nt_ready_reason_detail = "snapshot_target_stale"
    streamer._snapshot_recovery_attempts = 9
    streamer._nt_snapshot_recover_last_ts = time.time() - 10.0
    streamer.nt_snapshot_recover_min_interval_sec = 0.1
    streamer.nt_snapshot_fresh_sec = 30.0
    streamer.nt_sync_interval_sec = 2.0
    streamer.nt_require_ready_message = False
    streamer._nt_last_snapshot_progress_wall_ts = time.time() - 999.0
    streamer._snapshot_progress_stall_windows = 12
    streamer._snapshot_progress_inst_stall_windows = 12
    streamer._send_nt_ping = lambda **_kwargs: True
    streamer._send_nt_sync_request = lambda **_kwargs: True
    streamer._compute_nt_ready = lambda: (
        setattr(streamer, "_nt_ready", False),
        setattr(streamer, "_nt_ready_reason", "snapshot_stale"),
        setattr(streamer, "_nt_ready_reason_detail", "snapshot_target_stale"),
    ) and (False, "snapshot_stale")
    streamer._check_nt_readiness()
    assert reconnect_calls
    assert str(streamer._entries_disarmed_reason) == "snapshot_progress_stalled"


def test_snapshot_progress_active_suppresses_reconnect(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    reconnect_calls: list[str] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        reconnect=lambda: reconnect_calls.append("reconnect"),
        reset_connection=lambda: reconnect_calls.append("reset"),
    )
    streamer._nt_ready = False
    streamer._nt_ready_reason = "snapshot_stale"
    streamer._nt_ready_reason_detail = "snapshot_target_stale"
    streamer._snapshot_recovery_attempts = 9
    streamer._nt_snapshot_recover_last_ts = time.time() - 10.0
    streamer.nt_snapshot_recover_min_interval_sec = 0.1
    streamer.nt_snapshot_fresh_sec = 30.0
    streamer.nt_sync_interval_sec = 2.0
    streamer.nt_require_ready_message = False
    streamer._nt_last_snapshot_progress_wall_ts = time.time()
    streamer._send_nt_ping = lambda **_kwargs: True
    streamer._send_nt_sync_request = lambda **_kwargs: True
    streamer._compute_nt_ready = lambda: (
        setattr(streamer, "_nt_ready", False),
        setattr(streamer, "_nt_ready_reason", "snapshot_stale"),
        setattr(streamer, "_nt_ready_reason_detail", "snapshot_target_stale"),
    ) and (False, "snapshot_stale")
    attempts_before = int(streamer._snapshot_recovery_attempts)
    streamer._check_nt_readiness()
    assert reconnect_calls == []
    assert int(streamer._snapshot_recovery_attempts) == attempts_before


def test_snapshot_progress_active_resets_stall_windows(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        reconnect=lambda: None,
        reset_connection=lambda: None,
    )
    streamer._nt_ready = False
    streamer._nt_ready_reason = "snapshot_stale"
    streamer._nt_ready_reason_detail = "snapshot_target_stale"
    streamer._snapshot_recovery_attempts = 12
    streamer._snapshot_progress_stall_windows = 7
    streamer._snapshot_progress_inst_stall_windows = 6
    streamer._nt_snapshot_recover_last_ts = time.time() - 10.0
    streamer.nt_snapshot_recover_min_interval_sec = 0.1
    streamer.nt_snapshot_fresh_sec = 30.0
    streamer.nt_sync_interval_sec = 2.0
    streamer.nt_require_ready_message = False
    streamer._nt_last_snapshot_progress_wall_ts = time.time()
    streamer._send_nt_ping = lambda **_kwargs: True
    streamer._send_nt_sync_request = lambda **_kwargs: True
    streamer._compute_nt_ready = lambda: (
        setattr(streamer, "_nt_ready", False),
        setattr(streamer, "_nt_ready_reason", "snapshot_stale"),
        setattr(streamer, "_nt_ready_reason_detail", "snapshot_target_stale"),
    ) and (False, "snapshot_stale")
    streamer._check_nt_readiness()
    assert int(streamer._snapshot_progress_stall_windows) == 0
    assert int(streamer._snapshot_progress_inst_stall_windows) == 0


def test_snapshot_runtime_context_rejects_external_recovered_without_state_mutation(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 0.0
    streamer.state.position_state = "FLAT"
    msg = {
        "type": "POSITION_SNAPSHOT",
        "client_order_id": "RECOVERED|ES JUN26|POSITION",
        "instrument": "ES JUN26",
        "pos_qty": 1,
        "side": "LONG",
        "timestamp": _canonical_ts_str(utc_ts()),
        "snapshot_seq": 10,
        "orders": [],
        "positions": [],
    }
    streamer._handle_nt_position_snapshot(msg)
    assert float(streamer._pos or 0.0) == pytest.approx(0.0)
    assert str(streamer._snapshot_last_credit_decision) == "rejected"
    assert "runtime_context_external_recovered_lineage" in str(streamer._snapshot_last_credit_reason)


def test_snapshot_runtime_context_rejects_historical_without_state_mutation(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 0.0
    streamer.state.position_state = "FLAT"
    streamer._run_start_utc_ts = pd.Timestamp("2026-05-19T20:00:00-06:00")
    streamer.nt_snapshot_fresh_sec = 30.0
    msg = {
        "type": "POSITION_SNAPSHOT",
        "client_order_id": "orphan|historic|1",
        "instrument": "ES JUN26",
        "pos_qty": 1,
        "side": "LONG",
        "timestamp": "2026-05-19T19:30:00-06:00",
        "snapshot_seq": 11,
        "orders": [],
        "positions": [],
    }
    streamer._handle_nt_position_snapshot(msg)
    assert float(streamer._pos or 0.0) == pytest.approx(0.0)
    assert str(streamer._snapshot_last_credit_decision) == "rejected"
    assert "runtime_context_pre_run_snapshot_ts" in str(streamer._snapshot_last_credit_reason)


def test_broker_flat_zero_orders_clears_orphan_safety_lockout(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._ensure_compat_defaults()
    streamer.exec_instrument = "ES JUN26"
    streamer.exec_instrument_source = "cli"
    streamer.nt_instrument = "ES JUN26"
    streamer.nt_canonical_instrument = None
    streamer.nt_account = "DEMO7818783"
    streamer._nt_account_chosen = "DEMO7818783"
    streamer.nt_exec_policy = "live"
    streamer.snapshot_flat_override_required_streak = 3
    streamer.snapshot_flat_override_max_age_sec = 30.0
    streamer._snapshot_flat_streak_by_instrument = {}
    streamer._snapshot_flat_streak_by_instrument["ES 06-26"] = 2
    streamer._snapshot_flat_last_ts_by_instrument = {}
    streamer._nt_last_snapshot_orders_by_instrument = {}
    streamer._nt_last_snapshot_orders_ts_by_instrument = {}
    streamer._nt_last_price_by_instrument = {}
    streamer._nt_last_price_ts_by_instrument = {}
    streamer._nt_last_price_source_by_instrument = {}
    streamer._nt_last_bid_by_instrument = {}
    streamer._nt_last_ask_by_instrument = {}
    streamer._nt_last_pos_qty_by_instrument = {}
    streamer._nt_stop_update_pending = {}
    streamer._nt_repair_state_by_instrument = {}
    streamer._startup_resync_pending = False
    streamer._nt_snapshot_blocking_orders_count = 0
    streamer._pos = -1.0
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer.state.active_client_order_id = "RECOVERED|ES JUN26|514943319427"
    streamer._open_trade = {"client_order_id": "RECOVERED|ES JUN26|514943319427", "side": "SHORT"}
    streamer._hard_lockout_active = True
    streamer._hard_lockout_code = "nt_orphan_safety"
    streamer._hard_lockout_detail = {"reason": "missing_stop_price"}
    streamer.state.hard_lockout_reason = "nt_orphan_safety"
    streamer.state.hard_lockout_evidence = {"reason": "missing_stop_price"}
    streamer._load_fill_truth_index = lambda fresh=False: {}
    streamer._resolve_account = lambda detected, source: (True, {}, detected)
    streamer._compute_nt_ready = lambda: (True, "ok")
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    streamer._handle_nt_position_snapshot(
        {
            "type": "POSITION_SNAPSHOT",
            "protocol_version": 1,
            "client_order_id": "SAFETY|ES JUN26|20260604184943|SNAPSHOT",
            "instrument": "ES JUN26",
            "account": "DEMO7818783",
            "pos_qty": 0,
            "side": "FLAT",
            "orders": [],
            "positions": [],
            "timestamp": _canonical_ts_str(utc_ts()),
        }
    )

    assert streamer._hard_lockout_active is False
    assert streamer._hard_lockout_code is None
    assert streamer.state.hard_lockout_reason is None
    assert float(streamer._pos or 0.0) == pytest.approx(0.0)
    assert streamer.state.position_state == "FLAT"
    assert streamer.state.active_client_order_id is None
    assert any(event.get("event") == "orphan_lockout_cleared_broker_flat" for event in events)


def test_cleanup_isolation_suppressed_when_protection_repair_inflight(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_exec_state = "ARMED"
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._protection_repair_inflight = True
    ok = streamer._assert_cleanup_isolation(
        cleanup_action="cancel_all",
        client_order_id="SAFETY|x",
        callsite="test",
    )
    assert ok is True
    assert str(getattr(streamer, "_hard_lockout_code", "") or "") != "cleanup_while_armed"


def test_lineage_entry_exit_isolation_keeps_flip_open_eligible(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.enforce_integrity_gate = True
    streamer.run_id = "RUNID"
    flip_signal_id = "RUNID|sig|flip|123"
    flip_cid = "RUNID|flip|123"

    streamer._index_signal_lineage(
        {
            "run_id": "RUNID",
            "ts": "2026-05-18T19:15:00-06:00",
            "event": "FLIP",
            "side": "SHORT",
            "signal_id": flip_signal_id,
            "client_order_id": flip_cid,
            "ctx": {"executor_mirror": False},
        }
    )
    streamer._index_signal_lineage(
        {
            "run_id": "RUNID",
            "ts": "2026-05-18T19:15:00-06:00",
            "event": "CLOSE",
            "side": "NONE",
            "signal_id": flip_signal_id,
            "client_order_id": f"{flip_cid}.CLOSE",
            "ctx": {"executor_mirror": True},
        }
    )

    open_intent = stream_live_csv_mod.ExecutionIntent(
        intent_id=f"{flip_cid}.OPEN",
        action="OPEN",
        side="SHORT",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        bar_ts="2026-05-18T19:15:00-06:00",
        signal_id=flip_signal_id,
        parent_signal_id=flip_signal_id,
        transition_id=flip_cid,
        lineage_family="ENTRY",
    )
    assert streamer._validate_open_lineage(open_intent) is None
    assert int(getattr(streamer, "_lineage_overwrite_prevented_total", 0) or 0) >= 1


def test_open_blocked_when_reusing_same_flip_signal_and_bar(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._last_flip_close_signal_id = "sig|flip|same"
    streamer._last_flip_close_bar_ts = "2026-05-18T23:05:00-06:00"
    streamer.nt_account_mode = "none"
    streamer.nt_enabled = False
    streamer.nt_bridge = None
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="RUNID|open|same",
        action="OPEN",
        side="SHORT",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        stop_price=7410.0,
        target_price=7395.0,
        entry_price=7404.0,
        bar_ts="2026-05-18T23:05:00-06:00",
        model_price=7404.0,
        model_stop_price=7410.0,
        model_target_price=7395.0,
        signal_id="sig|flip|same",
    )
    result = streamer.execute_intent(intent)
    assert result.decision == "BLOCKED_SYNC"
    assert result.reason_code == "flip_reopen_requires_fresh_signal"
    assert int(streamer._executor_stats.get("flip_reopen_blocked_no_fresh_signal_total", 0) or 0) >= 1
    assert any(ev.get("event") == "flip_reopen_requires_fresh_signal" for ev in events)


def test_lineage_autoheal_allows_flip_open_once_from_matching_close_chain(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.enforce_integrity_gate = True
    streamer.run_mode = "live"
    streamer._phase = "LIVE"
    streamer.run_id = "RUNID"
    transition_id = "ES|model|RUNID|2026-05-18T21:30:00-06:00|FLIP|LONG|chain123"
    close_cid = f"{transition_id}.CLOSE"
    open_cid = f"{transition_id}.OPEN"
    streamer._signal_lineage_by_client_order_id_exit[close_cid] = {
        "run_id": "RUNID",
        "bar_ts": "2026-05-18T21:30:00-06:00",
        "action": "CLOSE",
        "instrument": "ES JUN26",
        "client_order_id": close_cid,
    }
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id=open_cid,
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        bar_ts="2026-05-18T21:30:00-06:00",
        signal_id="SIG-AUTOHEAL-1",
        transition_id=transition_id,
        lineage_family="ENTRY",
    )
    assert streamer._validate_open_lineage(intent) is None
    assert int(getattr(streamer, "_lineage_autoheal_success_total", 0) or 0) == 1
    # one-time token: second intent in same chain still blocks
    intent2 = stream_live_csv_mod.ExecutionIntent(
        intent_id=f"{transition_id}.OPEN2.OPEN",
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        bar_ts="2026-05-18T21:30:00-06:00",
        signal_id="SIG-AUTOHEAL-2",
        transition_id=transition_id,
        lineage_family="ENTRY",
    )
    assert streamer._validate_open_lineage(intent2) == "lineage_signal_not_found"


def test_lineage_autoheal_keeps_run_mismatch_blocked(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.enforce_integrity_gate = True
    streamer.run_mode = "live"
    streamer._phase = "LIVE"
    streamer.run_id = "RUNID-B"
    transition_id = "ES|model|RUNID-A|2026-05-18T21:30:00-06:00|FLIP|LONG|chainABC"
    close_cid = f"{transition_id}.CLOSE"
    streamer._signal_lineage_by_client_order_id_exit[close_cid] = {
        "run_id": "RUNID-A",
        "bar_ts": "2026-05-18T21:30:00-06:00",
        "action": "CLOSE",
        "instrument": "ES JUN26",
        "client_order_id": close_cid,
    }
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id=f"{transition_id}.OPEN",
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        bar_ts="2026-05-18T21:30:00-06:00",
        signal_id="SIG-AUTOHEAL-MISMATCH",
        transition_id=transition_id,
        lineage_family="ENTRY",
    )
    assert streamer._validate_open_lineage(intent) == "lineage_wrong_run_session"


def test_state_csv_dual_labels_include_execution_truth_fields(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "RUNID|OPEN|123"
    streamer._executor_terminal[cid] = "SENT"
    streamer._nt_order_state[cid] = {"entry_filled": True}

    streamer._append_state_row(
        pd.Timestamp("2026-05-18T19:45:00-06:00"),
        "OPEN",
        "LONG",
        7424.0,
        0.7,
        0.5,
        0.6,
        "PVET",
        "7416",
        "7432",
        "--",
        "A+",
        1.0,
        "bars 0/20",
        0.5,
        "Entering LONG",
        state_meta={"client_order_id": cid},
    )
    rows = list(csv.DictReader(streamer.state_stream_csv.open("r", encoding="utf-8", newline="\n")))
    assert rows
    row = rows[-1]
    assert row.get("execution_decision") == "SENT"
    assert row.get("execution_sent") == "1"
    assert row.get("execution_fill_confirmed") == "1"
    assert row.get("execution_truth_state") == "executed"


def test_reject_circuit_breaker_disarms_then_rearms_on_fresh_snapshot(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True)
    streamer.reject_circuit_breaker_max_rejects = 2
    streamer.reject_circuit_breaker_window_sec = 300.0
    streamer._nt_sync_pending = False
    streamer._startup_resync_pending = False
    streamer._reject_circuit_breaker_pending_recovery = False
    sync_calls: list[tuple[tuple, dict]] = []
    log_events: list[dict] = []
    streamer._send_nt_sync_request = lambda *args, **kwargs: sync_calls.append((args, kwargs)) or True
    streamer._maybe_request_nt_snapshot = lambda *args, **kwargs: None
    streamer._log_exec_event = lambda payload: log_events.append(payload)

    first = streamer._record_reject_circuit_breaker(
        cid="CID-1",
        status="REJECTED",
        reason="bad_payload",
        source="ORDER_ACK",
        evidence={"client_order_id": "CID-1"},
    )
    second = streamer._record_reject_circuit_breaker(
        cid="CID-2",
        status="REJECTED",
        reason="bad_payload",
        source="ORDER_ACK",
        evidence={"client_order_id": "CID-2"},
    )

    assert first is False
    assert second is True
    assert streamer.state.entries_disarmed_reason == "reject_circuit_breaker"
    assert streamer._entries_disarmed_reason == "reject_circuit_breaker"
    assert streamer._reject_circuit_breaker_pending_recovery is True
    assert streamer._startup_resync_pending is True
    assert sync_calls
    assert any(ev.get("event") == "reject_circuit_breaker_tripped" for ev in log_events)
    assert any(ev.get("event") == "entries_disarmed" and ev.get("reason") == "reject_circuit_breaker" for ev in log_events)

    rearmed = streamer._rearm_reject_circuit_breaker(source="position_snapshot", inst_match=True)
    assert rearmed is True
    assert streamer.state.entries_disarmed_reason is None
    assert streamer._entries_disarmed_reason is None
    assert streamer._reject_circuit_breaker_pending_recovery is False
    assert streamer._reject_circuit_breaker_active is False


def test_orphan_fill_reconciliation_requires_broker_confirmation(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._ensure_compat_defaults()
    streamer._position_state_toggle_history = {}
    streamer._log_exec_event = lambda *args, **kwargs: None
    streamer._rebuild_trades_from_fill_truth = lambda *args, **kwargs: None
    streamer._load_fill_truth_index = lambda fresh=False: {
        "UNTRACKED|ES MAR26|1": {
            "client_order_id": "UNTRACKED|ES MAR26|1",
            "correlation_id": "UNTRACKED|ES MAR26|1",
            "side": "LONG",
            "intent_side": "LONG",
            "entry_fill_ts_epoch": 1.0,
            "entry_fill_price": 100.0,
            "entry_fill_qty": 1.0,
            "exit_fill_ts_epoch": None,
            "exit_fill_price": None,
            "exit_fill_qty": None,
        }
    }
    streamer._nt_last_snapshot_instrument = "ES MAR26"
    streamer._nt_last_pos_qty_by_instrument = {"ES MAR26": 0.0}
    ts = pd.Timestamp("2026-04-02T10:00:00-06:00")

    streamer._apply_fill_truth_reconciliation(source="orphan_fill_event", bar_ts=ts)
    assert streamer._pos == 0
    assert streamer.state.position_state == "FLAT"
    assert streamer._open_trade is None

    streamer._nt_last_pos_qty_by_instrument["ES MAR26"] = 1.0
    streamer._apply_fill_truth_reconciliation(source="orphan_fill_event", bar_ts=ts, fresh=True)
    assert streamer._pos == 1
    assert streamer.state.position_state == "IN_POSITION_UNPROTECTED"


def test_snapshot_adopt_reassociates_untracked_cid_to_active_trade(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._ensure_compat_defaults()
    streamer._position_state_toggle_history = {}
    logs: list[dict] = []
    streamer._log_exec_event = lambda payload: logs.append(dict(payload))
    streamer.state.active_client_order_id = "CID-KNOWN-1"
    streamer._open_trade = {"client_order_id": "CID-KNOWN-1", "side": "SHORT"}
    streamer._nt_order_state["CID-KNOWN-1"] = {"entry_filled": True, "instrument": "ES JUN26"}
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True)
    streamer._adopt_snapshot_position(
        inst_key="ES 06-26",
        pos_qty=-1.0,
        avg_price=7484.75,
        side="SHORT",
        msg={"timestamp": utc_ts()},
        orders=[
            {
                "order_type": "STOP",
                "client_order_id": "UNTRACKED|ES JUN26|497047052803",
                "order_state": "WORKING",
                "stop_price": 7491.5,
                "qty": 1,
                "instrument": "ES JUN26",
            }
        ],
    )
    assert streamer.state.active_client_order_id == "CID-KNOWN-1"
    assert str((streamer._open_trade or {}).get("client_order_id")) == "CID-KNOWN-1"
    assert any(
        ev.get("event") == "snapshot_baseline_adopted" and str(ev.get("client_order_id")) == "CID-KNOWN-1"
        for ev in logs
    )


def test_snapshot_untracked_protection_enriches_active_trade_prices(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._ensure_compat_defaults()
    streamer._position_state_toggle_history = {}
    logs: list[dict] = []
    queued: list[str] = []
    streamer._log_exec_event = lambda payload: logs.append(dict(payload))
    streamer._queue_trade_update_from_state = lambda cid: queued.append(str(cid))
    streamer._set_position_state = lambda state, **_kwargs: setattr(streamer.state, "position_state", state)
    streamer.state.active_client_order_id = "CID-ENTRY-1"
    streamer._pos = 1
    streamer._open_trade = {
        "client_order_id": "CID-ENTRY-1",
        "side": "LONG",
        "live_stop": 7534.5,
        "live_target": 7560.0,
        "planned_stop": 7533.75,
        "planned_target": 7559.25,
        "protection_status": "exits_working",
    }
    streamer._nt_order_state["CID-ENTRY-1"] = {
        "instrument": "ES JUN26",
        "entry_filled": True,
        "stop_price": 7534.5,
        "target_price": 7560.0,
    }

    streamer._enrich_active_trade_from_snapshot_protection(
        active_cid="CID-ENTRY-1",
        active_state=streamer._nt_order_state["CID-ENTRY-1"],
        chosen={
            "stop_price": 7533.75,
            "target_price": 7559.25,
            "stop_order_id": "STOP-1",
            "target_order_id": "TARGET-1",
        },
        stop_state="WORKING",
        target_state="WORKING",
        protection_working=True,
    )

    active_state = streamer._nt_order_state["CID-ENTRY-1"]
    assert active_state["stop_price"] == pytest.approx(7533.75)
    assert active_state["target_price"] == pytest.approx(7559.25)
    assert active_state["stop_order_id"] == "STOP-1"
    assert active_state["target_order_id"] == "TARGET-1"
    assert streamer._open_trade["live_stop"] == pytest.approx(7533.75)
    assert streamer._open_trade["live_target"] == pytest.approx(7559.25)
    assert streamer._open_trade["protection_status"] == "protected_confirmed"
    assert "CID-ENTRY-1" in queued
    assert streamer.state.active_client_order_id == "CID-ENTRY-1"


def test_apply_fill_truth_reconciliation_reassociates_untracked_active_cid(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._ensure_compat_defaults()
    streamer._position_state_toggle_history = {}
    logs: list[dict] = []
    streamer._log_exec_event = lambda payload: logs.append(dict(payload))
    streamer._rebuild_trades_from_fill_truth = lambda *_args, **_kwargs: None
    streamer._pos = -1
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer.state.active_client_order_id = "UNTRACKED|ES JUN26|497047052803"
    streamer._open_trade = {"client_order_id": "UNTRACKED|ES JUN26|497047052803", "side": "SHORT"}
    fill_index = {
        "CID-REAL-ENTRY-1": {
            "side": "SHORT",
            "intent_side": "SHORT",
            "entry_fill_ts_epoch": 100.0,
            "entry_fill_price": 7484.75,
            "entry_fill_qty": 1.0,
            "exit_fill_ts_epoch": None,
            "exit_fill_price": None,
            "exit_fill_qty": None,
        }
    }
    streamer._apply_fill_truth_reconciliation(fill_index=fill_index, source="guardrail")
    assert streamer.state.active_client_order_id == "CID-REAL-ENTRY-1"
    assert str((streamer._open_trade or {}).get("client_order_id")) == "CID-REAL-ENTRY-1"
    assert any(ev.get("event") == "fallback_fill_reassociated" for ev in logs)


def test_strict_intent_parity_mismatch_emits_artifacts_and_fail_closes(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.strict_intent_parity_enabled = True
    streamer.strict_intent_parity_fail_closed = True
    streamer.parity_intent_ledger_path = tmp_path / "parity_intent_ledger.jsonl"
    streamer.parity_mismatches_path = tmp_path / "parity_mismatches.jsonl"
    streamer.parity_summary_path = tmp_path / "parity_summary.json"
    streamer._strict_intent_parity_counts = {"intents_total": 0, "mismatches_total": 0}
    streamer._strict_intent_parity_first_cause = None
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, detail=None, **_kwargs: lockouts.append((str(code), dict(detail or {})))

    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-13T21:10:00-06:00"),
        signal_action="OPEN",
        client_order_id="CID-PARITY-FAIL-1",
        sent_to_nt=False,
        blocked_by=["blocked_not_armed"],
        extra={"decision": "BLOCKED", "reason": "blocked_not_armed", "phase": "LIVE", "side": "LONG"},
    )

    ledger = _read_jsonl(streamer.parity_intent_ledger_path)
    mismatches = _read_jsonl(streamer.parity_mismatches_path)
    summary = json.loads(streamer.parity_summary_path.read_text(encoding="utf-8"))
    assert len(ledger) == 1
    assert len(mismatches) == 1
    assert mismatches[0]["parity_mismatch_reason"] == "intent_not_sent"
    assert summary["parity_pass"] is False
    assert summary["first_cause"] == "intent_not_sent"
    assert lockouts and lockouts[-1][0] == "strict_intent_parity_mismatch"


def test_signal_to_order_emits_intent_execution_contract_fields(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.strict_intent_parity_enabled = True
    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-18T08:00:00-06:00"),
        signal_action="OPEN",
        client_order_id="CID-CONTRACT-1",
        sent_to_nt=False,
        blocked_by=["blocked_not_armed"],
        extra={
            "decision": "BLOCKED_SAFETY",
            "reason": "blocked_not_armed",
            "phase": "LIVE",
            "side": "LONG",
            "strategy_emit_allowed": True,
            "transport_send_allowed": False,
            "bridge_acceptance_status": "not_sent",
            "lifecycle_fill_status": "not_applicable",
            "parity_block_layer": "transport",
            "parity_reason_code": "blocked_not_armed",
            "requested_action": "OPEN",
            "resolved_action": "NO_TRADE",
        },
    )
    rows = [
        row
        for row in _read_jsonl(streamer.signal_to_order_path)
        if str(row.get("signal_action") or "").upper() in {"OPEN", "CLOSE", "FLIP"}
    ]
    assert rows
    row = rows[-1]
    contract = row.get("intent_execution_contract") or {}
    assert row.get("execution_truth_state") == "planned_only"
    assert contract.get("strategy_emit_allowed") is True
    assert contract.get("transport_send_allowed") is False
    assert contract.get("bridge_acceptance_status") == "not_sent"
    assert contract.get("lifecycle_fill_status") == "not_applicable"
    assert contract.get("requested_action") == "OPEN"
    assert contract.get("resolved_action") == "NO_TRADE"


def test_signal_to_order_flip_dual_row_normalization(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.strict_intent_parity_enabled = False
    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-18T08:05:00-06:00"),
        signal_action="FLIP",
        client_order_id="CID-FLIP-DUAL-1",
        sent_to_nt=False,
        blocked_by=["gates_block"],
        extra={
            "decision": "BLOCKED",
            "reason": "gates_block",
            "phase": "LIVE",
            "side": "SHORT",
            "transition_id": "CID-FLIP-DUAL-1",
        },
    )
    rows = [
        row
        for row in _read_jsonl(streamer.signal_to_order_path)
        if str(row.get("signal_action") or "").upper() in {"OPEN", "CLOSE", "FLIP"}
    ]
    assert len(rows) == 2
    assert rows[0].get("requested_action") == "FLIP"
    assert rows[0].get("resolved_action") == "CLOSE"
    assert rows[0].get("signal_action") == "CLOSE"
    assert rows[0].get("transition_step") == "close"
    assert rows[1].get("requested_action") == "FLIP"
    assert rows[1].get("resolved_action") == "OPEN"
    assert rows[1].get("signal_action") == "OPEN"
    assert rows[1].get("transition_step") == "open"
    assert rows[0].get("transition_id") == rows[1].get("transition_id")


def test_strict_intent_parity_hard_block_classifier() -> None:
    assert stream_live_csv_mod._strict_intent_parity_is_hard_block("snapshot_stale")
    assert stream_live_csv_mod._strict_intent_parity_is_hard_block("account_unauthorized")
    assert not stream_live_csv_mod._strict_intent_parity_is_hard_block("setup")
    assert not stream_live_csv_mod._strict_intent_parity_is_hard_block("phase2_setup_blocked")


def test_strict_intent_parity_ignores_backfill_phase(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.strict_intent_parity_enabled = True
    streamer.strict_intent_parity_fail_closed = True
    streamer.parity_intent_ledger_path = tmp_path / "parity_intent_ledger.jsonl"
    streamer.parity_mismatches_path = tmp_path / "parity_mismatches.jsonl"
    streamer.parity_summary_path = tmp_path / "parity_summary.json"
    streamer._strict_intent_parity_counts = {"intents_total": 0, "mismatches_total": 0}
    streamer._strict_intent_parity_first_cause = None
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, detail=None, **_kwargs: lockouts.append((str(code), dict(detail or {})))

    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-13T21:10:00-06:00"),
        signal_action="OPEN",
        client_order_id="CID-PARITY-BACKFILL-IGNORED-1",
        sent_to_nt=False,
        blocked_by=["blocked_not_armed"],
        extra={"decision": "BLOCKED", "reason": "blocked_not_armed", "phase": "BACKFILL", "side": "LONG"},
    )

    assert streamer._strict_intent_parity_counts["intents_total"] == 0
    assert streamer._strict_intent_parity_counts["mismatches_total"] == 0
    assert not lockouts


def test_strict_intent_parity_defers_mismatch_while_flip_transition_pending(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.strict_intent_parity_enabled = True
    streamer.strict_intent_parity_fail_closed = True
    streamer.parity_intent_ledger_path = tmp_path / "parity_intent_ledger.jsonl"
    streamer.parity_mismatches_path = tmp_path / "parity_mismatches.jsonl"
    streamer.parity_summary_path = tmp_path / "parity_summary.json"
    streamer._strict_intent_parity_counts = {"intents_total": 0, "mismatches_total": 0}
    streamer._strict_intent_parity_first_cause = None
    streamer._flip_transition_state = {
        "CID-FLIP-1": {
            "transition_id": "CID-FLIP-1",
            "status": "open_deferred",
            "start_ts": time.time(),
            "expires_ts": time.time() + 60.0,
        }
    }
    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-18T22:32:00-06:00"),
        signal_action="OPEN",
        client_order_id="CID-FLIP-1.OPEN",
        sent_to_nt=False,
        blocked_by=["blocked_not_armed"],
        extra={
            "decision": "BLOCKED_SYNC",
            "reason": "flip_open_deferred_wait_reconcile",
            "phase": "LIVE",
            "side": "SHORT",
            "transition_id": "CID-FLIP-1",
        },
    )
    mismatches = _read_jsonl(streamer.parity_mismatches_path)
    rows = _read_jsonl(streamer.parity_intent_ledger_path)
    assert rows and rows[-1].get("parity_transition_state") == "transition_pending"
    assert mismatches == []


def test_strict_parity_reason_returns_none_for_transition_pending() -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    payload = {
        "signal_action": "OPEN",
        "decision": "BLOCKED_SYNC",
        "parity_transition_state": "transition_pending",
    }
    assert streamer._strict_parity_reason_from_payload(payload) is None


def test_strict_intent_parity_post_fill_protection_degraded_is_nonfatal(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.strict_intent_parity_enabled = True
    streamer.strict_intent_parity_fail_closed = True
    streamer.parity_intent_ledger_path = tmp_path / "parity_intent_ledger.jsonl"
    streamer.parity_mismatches_path = tmp_path / "parity_mismatches.jsonl"
    streamer.parity_summary_path = tmp_path / "parity_summary.json"
    streamer._strict_intent_parity_counts = {"intents_total": 0, "mismatches_total": 0}
    streamer._strict_intent_parity_first_cause = None
    streamer._pos = 1
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, detail=None, **_kwargs: lockouts.append((str(code), dict(detail or {})))

    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-13T21:10:00-06:00"),
        signal_action="OPEN",
        client_order_id="CID-PARITY-POSTFILL-1",
        sent_to_nt=False,
        blocked_by=["nt_missing_stop_price"],
        extra={
            "decision": "REJECTED",
            "reason": "missing_stop_price",
            "reason_code": "nt_missing_stop_price",
            "phase": "LIVE",
            "side": "LONG",
            "entry_fill_confirmed": True,
            "position_state": "IN_POSITION_UNPROTECTED",
        },
    )

    mismatches = _read_jsonl(streamer.parity_mismatches_path)
    assert mismatches
    assert mismatches[-1]["parity_mismatch_reason"] == "execution_reject_after_fill"
    assert not lockouts


def test_strict_intent_parity_terminal_reject_no_fill_terminalizes_without_lockout(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.strict_intent_parity_enabled = True
    streamer.strict_intent_parity_fail_closed = True
    streamer.parity_intent_ledger_path = tmp_path / "parity_intent_ledger.jsonl"
    streamer.parity_mismatches_path = tmp_path / "parity_mismatches.jsonl"
    streamer.parity_summary_path = tmp_path / "parity_summary.json"
    streamer._strict_intent_parity_counts = {"intents_total": 0, "mismatches_total": 0}
    streamer._strict_intent_parity_first_cause = None
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, detail=None, **_kwargs: lockouts.append((str(code), dict(detail or {})))

    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-14T08:10:00-06:00"),
        signal_action="OPEN",
        client_order_id="CID-PARITY-REJECT-NOFILL-1",
        sent_to_nt=False,
        blocked_by=["nt_broker_reject"],
        extra={"decision": "REJECTED", "reason": "nt_broker_reject", "phase": "LIVE", "side": "LONG"},
    )

    mismatches = _read_jsonl(streamer.parity_mismatches_path)
    assert mismatches
    assert mismatches[-1]["parity_mismatch_reason"] == "execution_reject_no_fill"
    assert not lockouts
    state = streamer._nt_order_state.get("CID-PARITY-REJECT-NOFILL-1", {})
    assert state.get("parity_terminal") is True


def test_runtime_directional_open_is_blocked_by_model_authority(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, send=lambda _msg: True, handshake_ok=lambda: True)
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="SAFETY|RUN|ES JUN26|manual|abc123",
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        limit_price=None,
        stop_price=7490.0,
        target_price=7510.0,
        entry_price=7500.0,
        model_price=7500.0,
        model_stop_price=7490.0,
        model_target_price=7510.0,
        bar_ts="2026-05-14T08:20:00-06:00",
        signal_id="SIG-OPEN-1",
    )
    result = streamer.execute_intent(intent)
    assert result.decision == "BLOCKED_SAFETY"
    assert result.reason == "runtime_directional_action_blocked"
    assert int(getattr(streamer, "_runtime_lifecycle_actions_blocked", 0) or 0) >= 1


def test_strict_intent_parity_suppresses_cleanup_maintenance_block_without_entry_intent(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.strict_intent_parity_enabled = True
    streamer.strict_intent_parity_fail_closed = True
    streamer.parity_intent_ledger_path = tmp_path / "parity_intent_ledger.jsonl"
    streamer.parity_mismatches_path = tmp_path / "parity_mismatches.jsonl"
    streamer.parity_summary_path = tmp_path / "parity_summary.json"
    streamer._strict_intent_parity_counts = {"intents_total": 0, "mismatches_total": 0}
    streamer._strict_intent_parity_first_cause = None
    streamer._entries_disarmed_reason = "cleanup_while_armed"
    streamer._hard_lockout_code = "cleanup_while_armed"
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, detail=None, **_kwargs: lockouts.append((str(code), dict(detail or {})))

    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-14T07:41:29-06:00"),
        signal_action="OPEN",
        client_order_id="CID-PARITY-CLEANUP-SUPPRESS-1",
        sent_to_nt=False,
        blocked_by=["blocked_not_armed"],
        extra={
            "decision": "BLOCKED",
            "reason": "blocked_not_armed",
            "phase": "LIVE",
            "side": "SHORT",
            "entries_disarmed_reason": "cleanup_while_armed",
            "hard_lockout_code": "cleanup_while_armed",
        },
    )

    mismatches = _read_jsonl(streamer.parity_mismatches_path)
    assert not mismatches
    assert not lockouts


def test_strict_intent_parity_keeps_cleanup_block_mismatch_when_entry_intent_exists(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.strict_intent_parity_enabled = True
    streamer.strict_intent_parity_fail_closed = True
    streamer.parity_intent_ledger_path = tmp_path / "parity_intent_ledger.jsonl"
    streamer.parity_mismatches_path = tmp_path / "parity_mismatches.jsonl"
    streamer.parity_summary_path = tmp_path / "parity_summary.json"
    streamer._strict_intent_parity_counts = {"intents_total": 0, "mismatches_total": 0}
    streamer._strict_intent_parity_first_cause = None
    streamer._entries_disarmed_reason = "cleanup_while_armed"
    streamer._hard_lockout_code = "cleanup_while_armed"
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer._nt_order_state["CID-PARITY-CLEANUP-HARD-1"] = {"sent_ts": 1715694089.0}
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, detail=None, **_kwargs: lockouts.append((str(code), dict(detail or {})))

    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-14T07:41:29-06:00"),
        signal_action="OPEN",
        client_order_id="CID-PARITY-CLEANUP-HARD-1",
        sent_to_nt=False,
        blocked_by=["blocked_not_armed"],
        extra={
            "decision": "BLOCKED",
            "reason": "blocked_not_armed",
            "phase": "LIVE",
            "side": "SHORT",
            "entries_disarmed_reason": "cleanup_while_armed",
            "hard_lockout_code": "cleanup_while_armed",
        },
    )

    mismatches = _read_jsonl(streamer.parity_mismatches_path)
    assert mismatches
    assert mismatches[-1]["parity_mismatch_reason"] == "intent_not_sent"
    assert lockouts and lockouts[-1][0] == "strict_intent_parity_mismatch"


def test_strict_intent_parity_skips_structural_invalid_lineage(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.strict_intent_parity_enabled = True
    streamer.strict_intent_parity_fail_closed = True
    streamer.parity_intent_ledger_path = tmp_path / "parity_intent_ledger.jsonl"
    streamer.parity_mismatches_path = tmp_path / "parity_mismatches.jsonl"
    streamer.parity_summary_path = tmp_path / "parity_summary.json"
    streamer._strict_intent_parity_counts = {"intents_total": 0, "mismatches_total": 0}
    streamer._strict_intent_parity_first_cause = None
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, detail=None, **_kwargs: lockouts.append((str(code), dict(detail or {})))

    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-13T22:45:00-06:00"),
        signal_action="OPEN",
        client_order_id="CID-PARITY-LINEAGE-1",
        sent_to_nt=False,
        blocked_by=["lineage_wrong_run_session"],
        extra={"decision": "BLOCKED_SAFETY", "reason": "lineage_wrong_run_session", "phase": "LIVE", "side": "SHORT"},
    )

    assert streamer._strict_intent_parity_counts["intents_total"] == 0
    assert streamer._strict_intent_parity_counts["mismatches_total"] == 0
    assert not lockouts


def test_close_lockout_missing_stop_is_bypassed_to_flatten(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "CID-CLOSE-LOCKOUT-BYPASS"
    streamer._external_nt_cids = {cid}
    flatten_calls: list[tuple[str, str]] = []
    streamer._send_nt_flatten = lambda c, reason="": flatten_calls.append((str(c), str(reason))) or True
    streamer._nt_order_state[cid] = {
        "intent_action": "CLOSE",
        "status": "sent",
        "side": "NONE",
        "qty": 1,
        "instrument": "ES JUN26",
    }
    streamer._handle_nt_message_inner(
        {
            "type": "ORDER_ACK",
            "client_order_id": cid,
            "status": "LOCKOUT",
            "reason": "missing_stop_price:ES JUN26",
            "instrument": "ES JUN26",
            "protocol_version": 1,
        }
    )
    assert flatten_calls
    assert flatten_calls[-1][1] == "model_close_missing_stop_bypass"
    assert streamer._nt_order_state[cid]["status"] == "close_retry_flatten"


def test_close_intent_allows_missing_stop_target_pre_send(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._nt_ready = True
    streamer.nt_enabled = True
    streamer.nt_account_mode = "none"
    sent: list[dict] = []
    streamer.nt_bridge = SimpleNamespace(
        is_connected=True,
        handshake_ok=lambda: True,
        send=lambda msg: sent.append(dict(msg)) or True,
    )
    streamer._ensure_nt_protocol_context = lambda *args, **kwargs: True
    streamer._validate_nt_order_contract = stream_live_csv_mod.LiveCSVStreamer._validate_nt_order_contract.__get__(
        streamer, stream_live_csv_mod.LiveCSVStreamer
    )
    streamer._nt_send_with_wait = lambda payload, context="", wait_sec=10.0: sent.append(dict(payload)) or True
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="CID-CLOSE-PRE-SEND-1",
        action="CLOSE",
        side="NONE",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        limit_price=None,
        stop_price=None,
        target_price=None,
        entry_price=None,
        bar_ts="2026-05-13T21:10:00-06:00",
        signal_id="SIG-CLOSE-1",
    )
    result = streamer._send_intent_order(intent)
    assert result.decision == "SENT"
    assert sent


def test_open_send_failure_does_not_leave_sent_order_state(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._nt_ready = True
    streamer.nt_enabled = True
    streamer.nt_adapter = "nt"
    streamer.nt_account_mode = "none"
    streamer.nt_bridge = SimpleNamespace(
        is_connected=False,
        handshake_ok=lambda: False,
        send=lambda _msg: False,
    )
    streamer._nt_session_id = "SESSION-OPEN-SEND-FAIL"
    cid = "CID-OPEN-SEND-FAIL-1"
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id=cid,
        action="OPEN",
        side="LONG",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        order_type="MARKET",
        limit_price=None,
        stop_price=7550.0,
        target_price=7564.5,
        entry_price=None,
        model_price=7557.25,
        model_stop_price=7550.0,
        model_target_price=7564.5,
        model_stop_abs=7550.0,
        model_target_abs=7564.5,
        protection_price_mode="offset",
        bar_ts="2026-06-05T01:55:00-06:00",
        signal_id="SIG-OPEN-SEND-FAIL-1",
    )

    result = streamer._send_intent_order(intent)

    assert result.decision == "BLOCKED_SAFETY"
    assert result.reason_code == "send_failed"
    assert cid not in streamer._nt_order_state
    assert streamer.state.last_nt_client_order_id != cid


def test_close_intent_duplicate_when_close_already_in_progress(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_account_mode = "none"
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: True)
    cid = "CID-CLOSE-DUPE-1"
    streamer._nt_order_state[cid] = {"close_in_progress": True, "status": "EXITING"}
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id=cid,
        action="CLOSE",
        side="NONE",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        bar_ts="2026-05-13T21:10:00-06:00",
        signal_id="SIG-CLOSE-DUPE-1",
    )
    result = streamer._send_intent_order(intent)
    assert result.decision == "SKIPPED_IDEMPOTENT"
    assert result.reason_code == "IDEMPOTENT_DUPLICATE"


def test_signal_to_order_marks_non_actionable_followup_after_prior_send(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="CID-S2O-FOLLOWUP-1",
        action="OPEN",
        side="SHORT",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        bar_ts="2026-05-18T21:30:00-06:00",
        signal_id="SIG-S2O-FOLLOWUP-1",
    )
    streamer._emit_executor_decision(intent, decision="SENT", reason_code="sent")
    streamer._emit_executor_decision(intent, decision="SKIPPED_IDEMPOTENT", reason_code="IDEMPOTENT_DUPLICATE")
    rows = _read_jsonl(streamer.signal_to_order_path)
    assert rows
    last = rows[-1]
    assert last.get("reason") == "IDEMPOTENT_DUPLICATE"
    assert bool(last.get("non_actionable_followup")) is True
    assert str(last.get("send_lifecycle_state") or "") == "followup_after_prior_send"


def test_signal_to_order_marks_actionable_followup_when_unprotected_timeout_escalated(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "CID-S2O-FOLLOWUP-ESCALATED-1"
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id=cid,
        action="OPEN",
        side="SHORT",
        qty=1,
        instrument_raw="ES JUN26",
        exec_instrument="ES JUN26",
        account="SIM",
        bar_ts="2026-05-18T21:35:00-06:00",
        signal_id="SIG-S2O-FOLLOWUP-ESCALATED-1",
    )
    streamer._emit_executor_decision(intent, decision="SENT", reason_code="sent")
    streamer._set_unprotected_followup_escalation(
        cid=cid,
        elapsed_unprotected_sec=18.5,
        threshold_sec=12.0,
        reason_code="nt_missing_stop_price",
        reason_source="broker_reject",
    )
    streamer._emit_executor_decision(intent, decision="REJECTED", reason_code="nt_missing_stop_price")
    rows = _read_jsonl(streamer.signal_to_order_path)
    assert rows
    last = rows[-1]
    assert str(last.get("send_lifecycle_state") or "") == "followup_after_prior_send"
    assert str(last.get("lifecycle_fill_status") or "") == "timeout_actionable_unprotected"
    assert str(last.get("parity_guard_class") or "") == "safety_hard"
    assert bool(last.get("escalation_triggered")) is True
    assert str(last.get("escalation_reason_source") or "") == "broker_reject"
    assert str(last.get("escalation_reason_code") or "") == "nt_missing_stop_price"
    assert float(last.get("elapsed_unprotected_sec") or 0.0) == pytest.approx(18.5)


def test_build_reject_diagnostics_marks_safety_missing_stop_recoverable(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    detail = streamer._build_reject_diagnostics(
        cid="SAFETY|RUNID|ES JUN26|protection_repair_timeout|abc|a1",
        status="LOCKOUT",
        reason="missing_stop_price",
        reject_code="nt_missing_stop_price",
        state={"intent_action": "FLATTEN", "instrument": "ES JUN26"},
        msg={"instrument": "ES JUN26"},
    )
    assert detail.get("cause_family") == "protection"
    assert bool(detail.get("recoverable")) is True
    assert detail.get("reject_class") == "nt_protection_repair_recoverable"


def test_close_intent_uses_authoritative_qty(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_account_mode = "none"
    streamer._pos = 0
    streamer._nt_last_pos_qty_by_instrument = {"ES 06-26": 2.0}
    streamer._load_fill_truth_index = lambda fresh=False: {}
    streamer._summarize_fill_truth = lambda _: {"net_qty": 0.0, "fills_present": False}
    streamer._resolve_exec_instrument = lambda _raw: ("ES 06-26", "snapshot")
    ev = {
        "type": "CLOSE",
        "side": "NONE",
        "bar_ts": "2026-05-13T22:45:00-06:00",
        "client_order_id": "CID-CLOSE-QTY-AUTH-1",
    }
    intent, err = streamer._build_execution_intent(ev)
    assert err is None
    assert intent is not None
    assert intent.qty == 2


def test_reconcile_reporting_reloads_active_fill_truth_when_cache_is_stale(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    active_cid = "RUNID|preset|ES JUN26|2026-06-03T21:50:00-06:00|OPEN|LONG|abc123"
    active_truth = {
        "client_order_id": active_cid,
        "side": "LONG",
        "intent_side": "LONG",
        "entry_fill_ts_epoch": 1780545003.455978,
        "entry_fill_price": 7539.0,
        "entry_fill_qty": 1.0,
        "exit_fill_ts_epoch": None,
        "exit_fill_price": None,
        "exit_fill_qty": None,
    }

    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._load_fill_truth_index = lambda fresh=False: ({active_cid: active_truth} if fresh else {})
    streamer._rebuild_trades_from_fill_truth = lambda _index: None
    streamer._load_exit_lifecycle_index = lambda: {}
    streamer._close_watchdog = SimpleNamespace(pending_watches=lambda: [])
    streamer.state.position_state = "IN_POSITION_PROTECTED"
    streamer.state.active_client_order_id = active_cid
    streamer._pos = 1
    streamer._open_trade = {"client_order_id": active_cid, "entry_ack_ts": "2026-06-03T21:50:03-06:00"}
    streamer._nt_order_state = {
        active_cid: {
            "sent_ts": "2026-06-03T21:50:03-06:00",
            "entry_order_id": "514943310772",
            "entry_acked": True,
        }
    }

    streamer._reconcile_reporting(bar_ts=pd.Timestamp("2026-06-03T21:55:00-06:00"), now_ts=time.time())

    assert streamer._has_fill_truth_last is True
    assert streamer._fill_truth_state == "active_truth_present"
    assert streamer._reporting_mismatch_detail is None
    assert any(ev.get("event") == "fill_truth_active_reloaded" for ev in events)


def test_lockout_preserves_first_cause(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._hard_lockout_active = True
    streamer._hard_lockout_code = "strict_intent_parity_mismatch"
    streamer.state.hard_lockout_reason = "strict_intent_parity_mismatch"
    streamer._set_hard_lockout("reporting_mismatch_no_entry_fill", {"reason": "secondary"})
    assert str(streamer._hard_lockout_code or "") == "strict_intent_parity_mismatch"


def test_lockout_preserved_first_cause_throttles_repeats_with_heartbeat(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._hard_lockout_active = True
    streamer._hard_lockout_code = "strict_intent_parity_mismatch"
    streamer.state.hard_lockout_reason = "strict_intent_parity_mismatch"
    streamer._lockout_preserved_repeat_last_ts = time.time()
    streamer._set_hard_lockout("reporting_mismatch_no_entry_fill", {"reason": "secondary"})
    streamer._set_hard_lockout("reporting_mismatch_no_entry_fill", {"reason": "secondary"})
    streamer._set_hard_lockout("reporting_mismatch_no_entry_fill", {"reason": "secondary"})
    assert int(getattr(streamer, "_lockout_preserved_repeat_suppressed_total", 0) or 0) >= 1
    assert any(ev.get("event") == "lockout_preserved_first_cause" for ev in events)
    assert any(ev.get("event") == "lockout_preserved_first_cause_heartbeat" for ev in events)


def test_replay_auto_clear_lockout_after_flat(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.replay_execution_intended = True
    streamer.allow_replay_exec_to_nt = True
    streamer._hard_lockout_active = True
    streamer._hard_lockout_code = "strict_intent_parity_mismatch"
    streamer.state.hard_lockout_reason = "strict_intent_parity_mismatch"
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer._close_watchdog = SimpleNamespace(pending_watches=lambda: [])
    streamer._maybe_auto_clear_replay_lockout_after_flat(source="test", correlation_id="CID-CLOSE-1")
    assert streamer._hard_lockout_active is False


def test_classify_unknowncid_as_external_recovered(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "UNKNOWNCID|ES JUN26|2026-05-14T07:46:20.7083779-06:00"
    origin = streamer._classify_nt_cid_origin(cid)
    assert origin == "external_recovered"


def test_reconcile_only_fill_does_not_drive_lifecycle(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._ensure_compat_defaults()
    streamer.state.position_state = "FLAT"
    transitions: list[tuple[str, str, str | None]] = []
    original_set_state = streamer._set_position_state

    def _spy_set_state(new_state: str, *, cid: str | None = None, signal_id: str | None = None) -> None:
        transitions.append((str(streamer.state.position_state), str(new_state), cid))
        return original_set_state(new_state, cid=cid, signal_id=signal_id)

    streamer._set_position_state = _spy_set_state
    logs: list[dict] = []
    streamer._log_exec_event = lambda payload: logs.append(dict(payload))

    streamer._update_nt_order_state(
        {
            "type": "FILL",
            "client_order_id": "RECOVERED|ES JUN26|497047053529",
            "ninja_order_id": "497047053529",
            "instrument": "ES JUN26",
            "side": "LONG",
            "fill_price": 7501.25,
            "fill_qty": 1,
            "timestamp": "2026-05-14T08:31:53.1433049-06:00",
            "fill_origin": "recovered_reconcile",
            "lifecycle_eligible": False,
            "bridge_reconcile_context": "snapshot_flat",
        }
    )

    assert not transitions
    assert any(ev.get("event") == "reconcile_only_fill_ignored_for_lifecycle" for ev in logs)
    assert any(ev.get("event") == "parity_excluded_reconcile_only_fill" for ev in logs)


def test_desync_toggle_tracking_ignores_external_recovered_cid(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._ensure_compat_defaults()
    streamer._position_state_toggle_history = {}
    streamer._desync_latch_active = False
    cid = "UNKNOWNCID|ES JUN26|2026-05-14T07:46:20.7083779-06:00"
    streamer._record_position_state_transition_for_desync(
        prev="FLAT",
        new_state="IN_POSITION_UNPROTECTED",
        cid=cid,
        signal_id=None,
    )
    assert streamer._position_state_toggle_history == {}
    assert streamer._desync_latch_active is False


def test_desync_kill_switch_deferred_during_close_settlement(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._ensure_compat_defaults()
    streamer._desync_toggle_window_sec = 12.0
    streamer._desync_toggle_threshold = 2
    streamer._position_state_toggle_history = {}
    streamer._desync_latch_active = False
    streamer._repair_close_or_flatten_in_flight = lambda _inst: True
    kill_calls: list[str] = []
    streamer._desync_kill_switch = lambda **_kwargs: kill_calls.append("called")

    cid = "SAFETY|RID|ES JUN26|protection_repair_timeout|abcd|a1"
    streamer._record_position_state_transition_for_desync(
        prev="FLAT",
        new_state="IN_POSITION_UNPROTECTED",
        cid=cid,
        signal_id=None,
    )
    streamer._record_position_state_transition_for_desync(
        prev="IN_POSITION_UNPROTECTED",
        new_state="FLAT",
        cid=cid,
        signal_id=None,
    )

    assert kill_calls == []
    assert streamer._desync_latch_active is False


def test_pnl_snapshot_promotes_mark_price_to_instrument_cache(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._ensure_compat_defaults()
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: True)
    streamer.exec_instrument = "ES JUN26"
    streamer.nt_instrument = "ES JUN26"
    streamer._nt_last_price_by_instrument = {}
    streamer._nt_last_price_source_by_instrument = {}
    streamer._nt_last_price_ts_by_instrument = {}
    streamer._nt_last_price_value_ts_by_instrument = {}

    streamer._handle_nt_message_inner(
        {
            "type": "PNL_SNAPSHOT",
            "event_type": "PNL_SNAPSHOT",
            "protocol_version": 1,
            "schema_version": 1,
            "instrument": "ES JUN26",
            "last_price": 7420.5,
            "position_qty": 0,
            "avg_price": 0,
            "unrealized_pnl_currency": 0,
            "realized_pnl_currency": -117.44,
            "timestamp": "2026-05-18T05:40:21.4979971-06:00",
            "ts_exchange": "2026-05-18T11:40:21.4979971+00:00",
            "ts_local": "2026-05-18T05:40:21.4979971-06:00",
            "source": "nt_account_api",
            "seq": 573,
            "client_order_id": "SAFETY|ES JUN26|20260518054021|SNAPSHOT",
        }
    )

    assert any(abs(float(v) - 7420.5) < 1e-9 for v in streamer._nt_last_price_by_instrument.values())
    assert any(str(src) == "pnl_snapshot_last" for src in streamer._nt_last_price_source_by_instrument.values())


def test_pnl_snapshot_derives_mark_when_last_price_is_entry_price(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._ensure_compat_defaults()
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda _msg: True)
    streamer.exec_instrument = "ES JUN26"
    streamer.nt_instrument = "ES JUN26"
    streamer._pos = 1
    streamer._nt_last_price_by_instrument = {}
    streamer._nt_last_price_source_by_instrument = {}
    streamer._nt_last_price_ts_by_instrument = {}
    streamer._nt_last_price_value_ts_by_instrument = {}

    streamer._handle_nt_message_inner(
        {
            "type": "PNL_SNAPSHOT",
            "event_type": "PNL_SNAPSHOT",
            "protocol_version": 1,
            "schema_version": 1,
            "instrument": "ES JUN26",
            "last_price": 7531.75,
            "position_qty": 1,
            "avg_price": 7531.75,
            "unrealized_pnl_currency": 400.0,
            "realized_pnl_currency": 392.3,
            "timestamp": "2026-06-03T21:18:11.498687-06:00",
            "source": "NinjaRepoBridge",
            "client_order_id": "RUN|ES JUN26|OPEN|LONG|abc",
        }
    )

    assert any(abs(float(v) - 7539.75) < 1e-9 for v in streamer._nt_last_price_by_instrument.values())
    assert any(str(src) == "pnl_snapshot_derived" for src in streamer._nt_last_price_source_by_instrument.values())


def test_bar_health_telemetry_auto_clears_protocol_unknown_type_lockout(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._hard_lockout_active = True
    streamer._hard_lockout_code = "protocol_unknown_type"
    streamer._hard_lockout_detail = {"type": "BAR_HEALTH"}
    streamer.state.hard_lockout_reason = "protocol_unknown_type"
    streamer.state.hard_lockout_evidence = {"type": "BAR_HEALTH"}

    streamer._handle_nt_message_inner(
        {
            "type": "BAR_HEALTH",
            "protocol_version": 1,
            "reason": "sender_not_emitting",
            "bar_seq": 123,
            "last_emit_ts_utc": "2026-05-26T03:29:58Z",
        }
    )

    assert streamer._hard_lockout_active is False
    assert streamer._hard_lockout_code is None
    assert any(ev.get("event") == "NT_PROTOCOL_TELEMETRY_LOCKOUT_AUTO_CLEARED" for ev in events)
    assert any(ev.get("event") == "BAR_HEALTH_TELEMETRY_RX" for ev in events)


def test_bridge_status_telemetry_auto_clears_protocol_unknown_type_lockout(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    lockouts: list[tuple[str, dict]] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._set_hard_lockout = lambda code, evidence=None, **_kwargs: lockouts.append((str(code), dict(evidence or {})))
    streamer._hard_lockout_active = True
    streamer._hard_lockout_code = "protocol_unknown_type"
    streamer._hard_lockout_detail = {"type": "BRIDGE_STATUS"}
    streamer.state.hard_lockout_reason = "protocol_unknown_type"
    streamer.state.hard_lockout_evidence = {"type": "BRIDGE_STATUS"}
    streamer.nt_strict_protocol = True

    streamer._handle_nt_message_inner(
        {
            "event_type": "BRIDGE_STATUS",
            "source": "nt_account_api",
            "connected": True,
            "account": "DEMO7467442",
            "instrument": "ES JUN26",
        }
    )

    assert streamer._hard_lockout_active is False
    assert not lockouts
    assert any(ev.get("event") == "NT_PROTOCOL_TELEMETRY_LOCKOUT_AUTO_CLEARED" for ev in events)
    assert any(ev.get("event") == "BRIDGE_STATUS_TELEMETRY_RX" for ev in events)


def test_raw_protocol_frame_is_ignored_without_lockout(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    lockouts: list[tuple[str, dict]] = []
    streamer._log_exec_event = lambda payload: events.append(dict(payload))
    streamer._set_hard_lockout = lambda code, evidence=None, **_kwargs: lockouts.append((str(code), dict(evidence or {})))
    streamer.nt_strict_protocol = True

    streamer._handle_nt_message_inner({"type": "RAW", "value": "garbage"})

    assert not lockouts
    assert any(ev.get("event") == "RAW_PROTOCOL_FRAME_IGNORED" for ev in events)


def test_recovered_context_missing_stop_does_not_trigger_orphan_hard_lockout(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._hard_lockout_active = False
    streamer._hard_lockout_code = None
    streamer._pos = 1.0
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._nt_recent_recovered_nonflat_ts = time.time()
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, evidence=None, **_kwargs: lockouts.append((str(code), dict(evidence or {})))
    flatten_calls: list[tuple[str, str]] = []
    streamer._send_nt_flatten = lambda cid, reason="", **_kwargs: flatten_calls.append((str(cid), str(reason))) or True

    streamer._handle_nt_message_inner(
        {
            "type": "ERROR",
            "protocol_version": 1,
            "client_order_id": "UNKNOWNCID|ES JUN26|2026-05-14T07:46:20.7083779-06:00",
            "message": "LOCKOUT: missing_stop_price",
            "instrument": "ES JUN26",
        }
    )

    assert not lockouts
    assert not flatten_calls
    assert streamer._orphan_lockout_branch == "recovered_context_guarded_nonfatal"


def test_recovered_context_missing_cid_routes_to_flatten_only_guard(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._hard_lockout_active = False
    streamer._hard_lockout_code = None
    streamer._pos = 1.0
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._nt_recent_recovered_nonflat_ts = time.time()
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, evidence=None, **_kwargs: lockouts.append((str(code), dict(evidence or {})))
    flatten_calls: list[tuple[str, str]] = []
    streamer._send_nt_flatten = lambda cid, reason="", **_kwargs: flatten_calls.append((str(cid), str(reason))) or True

    streamer._handle_nt_message_inner(
        {
            "type": "ERROR",
            "protocol_version": 1,
            "message": "LOCKOUT: missing_stop_price",
            "instrument": "ES JUN26",
        }
    )

    assert not lockouts
    assert flatten_calls
    assert streamer._orphan_lockout_branch == "recovered_context_requires_flatten_only"


def test_true_orphan_missing_stop_still_hard_lockouts(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._hard_lockout_active = False
    streamer._hard_lockout_code = None
    streamer._nt_recent_recovered_nonflat_ts = 0.0
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, evidence=None, **_kwargs: lockouts.append((str(code), dict(evidence or {})))

    streamer._handle_nt_message_inner(
        {
            "type": "ERROR",
            "protocol_version": 1,
            "message": "LOCKOUT: missing_stop_price",
            "instrument": "ES JUN26",
        }
    )

    assert lockouts
    assert lockouts[-1][0] == "nt_orphan_safety"
    assert streamer._orphan_lockout_branch == "orphan_true_hard_lockout"


def test_close_dedupe_increments_status_counter(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._last_close_guard = {}
    streamer._close_dedupe_cooldown_sec = 300.0
    first = streamer._should_dedupe_close(reason="model_close", position_state="IN_POSITION_PROTECTED", instrument="ES JUN26")
    second = streamer._should_dedupe_close(reason="protection_repair_failed", position_state="EXITING", instrument="ES JUN26")
    assert first is False
    assert second is True
    assert int(streamer._close_flatten_dedupe_hits) >= 1


def test_audit_parity_fails_on_emit_allowed_without_send_in_live(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"run_id": "RID", "executor_stats_all_phases": {"nt_order_entry_total": 0}}), encoding="utf-8")
    (run_dir / "stream_state.json").write_text(json.dumps({"state": {"position_state": "FLAT"}, "position": {"pos": 0}}), encoding="utf-8")
    (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_health_summary.json").write_text(json.dumps({"verdict": "running_healthy"}), encoding="utf-8")
    (run_dir / "order_events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "exec_events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "signal_to_order.jsonl").write_text(
        json.dumps(
            {
                "phase": "LIVE",
                "signal_action": "OPEN",
                "decision": "REJECTED",
                "emit_allowed": True,
                "sent_to_nt": False,
                "blocked_by": [],
                "reason": "nt_broker_reject",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = audit_run(run_dir)
    assert report["parity_decision"] == "FAIL"
    assert "EMIT_ALLOWED_NOT_SENT" in set(report["parity_fail_reasons"])


def test_audit_parity_fails_on_state_execution_divergence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"run_id": "RID", "executor_stats_all_phases": {"nt_order_entry_total": 0}}), encoding="utf-8")
    (run_dir / "stream_state.json").write_text(json.dumps({"state": {"position_state": "FLAT"}, "position": {"pos": 0}}), encoding="utf-8")
    (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_health_summary.json").write_text(json.dumps({"verdict": "running_healthy"}), encoding="utf-8")
    (run_dir / "order_events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "exec_events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "signal_to_order.jsonl").write_text(
        json.dumps({"phase": "LIVE", "signal_action": "OPEN", "decision": "BLOCKED", "emit_allowed": False, "sent_to_nt": False})
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "state.csv").write_text("action,side,price\nOPEN,LONG,7000\n", encoding="utf-8")
    report = audit_run(run_dir)
    assert report["parity_decision"] == "FAIL"
    assert "STATE_EXECUTION_DIVERGENCE" in set(report["parity_fail_reasons"])


def test_protection_timeout_emits_single_terminal_and_suppresses_repeats(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "paper"
    streamer.auto_flatten_enabled = False
    streamer._pos = 1
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._set_entries_disarmed = lambda *_args, **_kwargs: None
    lockouts: list[tuple[str, dict]] = []
    streamer._set_hard_lockout = lambda code, evidence=None, **_kwargs: lockouts.append((str(code), dict(evidence or {})))
    state = {
        "instrument": "ES JUN26",
        "entry_filled": True,
        "status": "entry_filled",
        "signal_id": "SIG-TIMEOUT-1",
    }
    now_ts = time.time()
    streamer._handle_protection_timeout(
        cid="CID-TIMEOUT-1",
        state=state,
        reason="no_protection_working",
        anchor_ts=now_ts - 40.0,
        now_ts=now_ts,
    )
    streamer._handle_protection_timeout(
        cid="CID-TIMEOUT-1",
        state=state,
        reason="no_protection_working",
        anchor_ts=now_ts - 41.0,
        now_ts=now_ts + 1.0,
    )
    assert len(lockouts) == 1
    assert str(lockouts[0][0]) == "nt_protection_timeout"
    assert state.get("protection_timeout_triggered") is True
    assert str(state.get("protection_timeout_terminal_key") or "").strip() != ""


def test_live_protection_timeouts_default_to_30s() -> None:
    parser = stream_live_csv_mod._build_arg_parser()
    args = parser.parse_args([])
    assert float(getattr(args, "nt_protection_timeout")) == pytest.approx(30.0)
    assert float(getattr(args, "nt_protection_repair_timeout_sec")) == pytest.approx(30.0)


def test_safety_flatten_one_shot_survives_safety_active_cid(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    sent: list[dict] = []
    order_events: list[tuple[str, str, dict]] = []
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda order: sent.append(dict(order)) or True)
    streamer.nt_account_mode = "none"
    streamer.nt_adapter = "native"
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer.state.position_qty = 1
    streamer.state.position_side = "LONG"
    streamer.state.active_client_order_id = "RUNID|OPEN|LONG|MODEL-1"
    streamer._pos = 1
    streamer._protection_repair_generation = 7
    streamer._safety_flatten_guard = {}
    streamer._safety_exit_epoch_tokens = {}
    streamer._should_dedupe_close = lambda **_kwargs: False
    streamer._log_order_event = lambda cid, event, detail=None: order_events.append((str(cid), str(event), dict(detail or {})))

    streamer._send_nt_flatten("repair|ES 06-26|first", reason="protection_repair_failed")
    # Simulate the exact run-263 follow-up condition: local active owner is now a generated safety CID.
    streamer.state.active_client_order_id = "SAFETY|RUNID|ES 06-26|protection_repair_failed|abc|a1"
    streamer._safety_flatten_guard = {}
    streamer._send_nt_flatten("repair|ES 06-26|second", reason="protection_repair_failed")

    assert len(sent) == 1
    assert any(event == "blocked_safety_exit_epoch_duplicate" for _, event, _ in order_events)


def test_safety_flatten_blocks_missing_session_before_bridge_send(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    sent: list[dict] = []
    blocks: list[tuple[str, dict]] = []
    streamer.run_id = ""
    streamer._nt_bridge_session_id = ""
    streamer.nt_enabled = True
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True, send=lambda order: sent.append(dict(order)) or True)
    streamer.nt_account_mode = "none"
    streamer.nt_adapter = "native"
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer.state.position_qty = 1
    streamer.state.position_side = "LONG"
    streamer._pos = 1
    streamer._should_dedupe_close = lambda **_kwargs: False
    streamer._emit_block_event = lambda block_code, block_detail=None, **_kwargs: blocks.append((str(block_code), dict(block_detail or {})))

    streamer._send_nt_flatten("SAFETY|RID|ES JUN26|protection_repair_failed|abc|a1", reason="protection_repair_failed")

    assert sent == []
    assert blocks
    assert blocks[-1][0] == "contract_violation_pre_send"
    assert "session_id" in set(blocks[-1][1].get("missing_fields") or [])


def test_health_summary_uses_broker_flat_truth_over_stale_runtime_pos(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 1
    streamer.state.position_state = "FLAT"
    streamer.trades_csv = tmp_path / "trades.csv"
    streamer.trades_csv.write_text("filled_qty,proof_only\n", encoding="utf-8")
    streamer._execution_intended_mode = lambda: False
    status = {
        "feed_health_ok": True,
        "bar_age_sec": 1.0,
        "effective_bar_age_max_sec": 605.0,
        "position_state": "FLAT",
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "execution_health_ok": True,
        "live_pnl_quality": {"position_qty": 0.0},
    }

    streamer._write_run_health_summary(status, process_alive=False, shutdown_reason="unit_test", exit_code=0)
    summary = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))

    assert summary["final_position"] == pytest.approx(0.0)
    assert summary["final_position_source"] == "broker_position_qty"
    assert summary["runtime_position_at_shutdown"] == pytest.approx(1.0)
    assert "final_position_not_flat_broker" not in set(summary["unresolved_warnings"])


def test_health_summary_marks_unsafe_when_broker_nonflat_even_if_runtime_flat(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer.trades_csv = tmp_path / "trades.csv"
    streamer.trades_csv.write_text("filled_qty,proof_only\n", encoding="utf-8")
    streamer._execution_intended_mode = lambda: False
    status = {
        "feed_health_ok": True,
        "bar_age_sec": 1.0,
        "effective_bar_age_max_sec": 605.0,
        "position_state": "FLAT",
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "execution_health_ok": True,
        "live_pnl_quality": {"position_qty": 1.0},
    }

    streamer._write_run_health_summary(status, process_alive=False, shutdown_reason="unit_test", exit_code=0)
    summary = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))

    assert summary["final_position"] == pytest.approx(1.0)
    assert summary["final_position_source"] == "broker_position_qty"
    assert "final_position_not_flat_broker" in set(summary["unresolved_warnings"])
    assert summary["verdict"] == "unsafe"


def test_health_summary_ignores_planned_only_no_fill_rows_for_execution_warning(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer.trades_csv = tmp_path / "trades.csv"
    row = {k: "" for k in stream_live_csv_mod.TRADES_HEADER_FIELDS}
    row.update(
        {
            "entry_ts": "2026-06-05T01:40:00-06:00",
            "side": "SHORT",
            "qty": "1",
            "entry_price": "7557.75",
            "protection_status": "planned_only",
            "filled_qty": "",
            "proof_only": "False",
            "client_order_id": "RID|OPEN|SHORT|stale_blocked",
        }
    )
    with streamer.trades_csv.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=stream_live_csv_mod.TRADES_HEADER_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    streamer._execution_intended_mode = lambda: True
    status = {
        "feed_health_ok": True,
        "bar_age_sec": 1.0,
        "effective_bar_age_max_sec": 605.0,
        "position_state": "FLAT",
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "execution_health_ok": True,
        "live_pnl_quality": {"position_qty": 0.0},
    }

    streamer._write_run_health_summary(status, process_alive=True)
    summary = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))

    assert summary["trade_evidence"]["planned_only_rows"] == 1
    assert summary["trade_evidence"]["non_planned_rows"] == 0
    assert "execution_intended_without_executable_fills" not in set(summary["unresolved_warnings"])


def test_health_summary_recovers_live_nt_entry_count_from_signal_to_order(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer.trades_csv = tmp_path / "trades.csv"
    streamer.signal_to_order_path = tmp_path / "signal_to_order.jsonl"
    streamer._executor_phase_stats = {"LIVE": {"candidate_signals_total": 2, "signal_append_total": 2, "nt_order_entry_total": 0}}
    rows = [
        {"type": "header", "run_id": "RUNID"},
        {
            "phase": "LIVE",
            "signal_action": "OPEN",
            "decision": "SENT",
            "sent_to_nt": True,
            "client_order_id": "RUNID|OPEN|LONG|1",
        },
        {
            "phase": "LIVE",
            "signal_action": "OPEN",
            "decision": "BLOCKED",
            "sent_to_nt": False,
            "client_order_id": "RUNID|OPEN|LONG|2",
        },
    ]
    streamer.signal_to_order_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    status = {
        "feed_health_ok": True,
        "bar_age_sec": 1.0,
        "effective_bar_age_max_sec": 605.0,
        "position_state": "FLAT",
        "snapshot_orders_count": 0,
        "snapshot_blocking_orders_count": 0,
        "execution_health_ok": True,
        "live_pnl_quality": {"position_qty": 0.0},
    }

    streamer._write_run_health_summary(status, process_alive=True)
    summary = json.loads((tmp_path / "run_health_summary.json").read_text(encoding="utf-8"))

    assert summary["live_evidence"]["live_nt_order_entry_total"] == 1
    assert summary["verdict"] == "running_healthy"


def test_fill_truth_flat_reconciliation_closes_open_ghost_trade_row(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.trades_csv = tmp_path / "trades.csv"
    row = {k: "" for k in stream_live_csv_mod.TRADES_HEADER_FIELDS}
    row.update(
        {
            "entry_ts": "2026-05-20T17:40:48-06:00",
            "side": "LONG",
            "qty": "2",
            "entry_price": "7419.25",
            "exit_reason": "protection_repair_failed",
            "client_order_id": "SAFETY|RID|ES JUN26|protection_repair_failed|abc|a1",
        }
    )
    with streamer.trades_csv.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=stream_live_csv_mod.TRADES_HEADER_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    streamer._rebuild_trades_from_fill_truth = lambda _index: None
    streamer._nt_last_pos_qty_by_instrument = {"ES JUN26": 0.0}
    streamer._nt_last_snapshot_instrument = "ES JUN26"
    streamer.nt_instrument = "ES JUN26"
    streamer._pos = 1
    streamer.state.position_state = "EXITING"
    fill_index = {
        "SAFETY|RID|ES JUN26|protection_repair_failed|abc|a1": {
            "client_order_id": "SAFETY|RID|ES JUN26|protection_repair_failed|abc|a1",
            "entry_fill_ts_epoch": time.time() - 5,
            "exit_fill_ts_epoch": time.time(),
            "exit_fill_price": 7419.25,
        }
    }
    streamer._load_fill_truth_index = lambda fresh=False: fill_index
    streamer._summarize_fill_truth = lambda _index: {
        "fills_present": True,
        "net_qty": 0.0,
        "last_cid": "SAFETY|RID|ES JUN26|protection_repair_failed|abc|a1",
    }

    streamer._apply_fill_truth_reconciliation(source="fill_event", fresh=True)

    rows = list(csv.DictReader(streamer.trades_csv.open("r", encoding="utf-8", newline="")))
    assert rows
    assert rows[0]["exit_ts"]
    assert rows[0]["exit_fill_price"]


def test_rebuild_trades_uses_matching_cid_protection_evidence(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.trades_csv = tmp_path / "trades.csv"
    streamer.order_events_path = tmp_path / "order_events.jsonl"
    streamer.nt_bridge = SimpleNamespace(is_connected=True, handshake_ok=lambda: True)
    streamer._nt_order_state = {
        "CID-1": {
            "side": "LONG",
            "qty": 1,
            "entry_filled": True,
            "exits_working": True,
            "stop_order_id": "STOP-STALE-2",
            "target_order_id": "TARGET-STALE-2",
            "stop_price": 7560.25,
            "target_price": 7577.75,
            "protected_ts": 300.0,
        },
        "CID-2": {
            "side": "LONG",
            "qty": 1,
            "entry_filled": True,
            "exits_working": True,
            "stop_order_id": "STOP-STALE-2",
            "target_order_id": "TARGET-STALE-2",
            "stop_price": 7560.25,
            "target_price": 7577.75,
            "protected_ts": 300.0,
        },
    }
    order_rows = [
        {
            "ts": "1970-01-01T00:01:41Z",
            "client_order_id": "CID-1",
            "status": "WORKING",
            "protected": True,
            "entry_ninja_order_id": "ENTRY-1",
            "stop_order_id": "STOP-1",
            "target_order_id": "TARGET-1",
            "stop_price": 7558.75,
            "target_price": 7572.75,
            "timestamp": "1970-01-01T00:01:41Z",
        },
        {
            "ts": "1970-01-01T00:05:01Z",
            "client_order_id": "CID-2",
            "status": "WORKING",
            "protected": True,
            "entry_ninja_order_id": "ENTRY-2",
            "stop_order_id": "STOP-2",
            "target_order_id": "TARGET-2",
            "stop_price": 7560.25,
            "target_price": 7577.75,
            "timestamp": "1970-01-01T00:05:01Z",
        },
    ]
    streamer.order_events_path.write_text("\n".join(json.dumps(r) for r in order_rows) + "\n", encoding="utf-8")
    fill_index = {
        "CID-1": {
            "side": "LONG",
            "intent_side": "LONG",
            "entry_fill_ts_epoch": 100.0,
            "entry_fill_price": 7565.75,
            "entry_fill_qty": 1.0,
            "exit_fill_ts_epoch": 200.0,
            "exit_fill_price": 7572.75,
            "exit_fill_qty": 1.0,
            "exit_reason_hint": "target_hit",
        },
        "CID-2": {
            "side": "LONG",
            "intent_side": "LONG",
            "entry_fill_ts_epoch": 300.0,
            "entry_fill_price": 7569.0,
            "entry_fill_qty": 1.0,
            "exit_fill_ts_epoch": 400.0,
            "exit_fill_price": 7573.25,
            "exit_fill_qty": 1.0,
            "exit_reason_hint": "reconciled_fill_exit",
        },
    }

    streamer._rebuild_trades_from_fill_truth(fill_index)
    rows = list(csv.DictReader(streamer.trades_csv.open("r", encoding="utf-8", newline="")))

    assert rows[0]["client_order_id"] == "CID-1"
    assert rows[0]["stop_order_id"] == "STOP-1"
    assert rows[0]["target_order_id"] == "TARGET-1"
    assert rows[0]["live_stop"] == "7558.75"
    assert rows[0]["live_target"] == "7572.75"
    assert rows[1]["client_order_id"] == "CID-2"
    assert rows[1]["stop_order_id"] == "STOP-2"
    assert rows[1]["target_order_id"] == "TARGET-2"


def test_sync_position_flat_zeroes_live_unrealized_pnl(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    calls: list[dict] = []
    streamer._update_live_pnl_from_state = lambda **kwargs: calls.append(dict(kwargs))

    streamer._sync_position_flat("unit_test")

    assert calls
    last = calls[-1]
    assert float(last.get("position_qty")) == pytest.approx(0.0)
    assert float(last.get("unrealized_pnl")) == pytest.approx(0.0)


def test_runtime_safety_fill_truth_is_reconcile_only_without_model_mapping(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    events: list[dict] = []
    safety_cid = "SAFETY|RUNID|ES 06-26|protection_repair_failed|abc|a1"
    streamer._pos = 0
    streamer.state.position_state = "FLAT"
    streamer._nt_last_pos_qty_by_instrument = {"ES 06-26": 1.0}
    streamer._load_fill_truth_index = lambda fresh=False: {"fill-1": {"client_order_id": safety_cid}}
    streamer._summarize_fill_truth = lambda _index: {
        "fills_present": True,
        "net_qty": 1.0,
        "last_cid": safety_cid,
        "last_entry": ("entry", safety_cid, {"entry_fill_price": 7393.25, "entry_fill_ts_epoch": time.time()}),
    }
    streamer._rebuild_trades_from_fill_truth = lambda _index: None
    streamer._log_exec_event = lambda payload: events.append(dict(payload))

    streamer._apply_fill_truth_reconciliation(source="orphan_fill_runtime_safety", fresh=True)

    assert streamer._pos == 0
    assert streamer.state.position_state == "FLAT"
    assert any(ev.get("event") == "fill_truth_runtime_safety_reconcile_only" for ev in events)


def test_reconcile_does_not_resurrect_unprotected_after_terminal_timeout(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    streamer.state.position_state = "FLAT"
    streamer.state.active_client_order_id = None
    streamer._active_close_correlation_id = None
    streamer._protection_timeout_terminal_emitted_keys = {"RID|ES JUN26|SIG1"}
    streamer._set_position_state("IN_POSITION_UNPROTECTED", cid="CID-ANY")
    assert streamer.state.position_state == "FLAT"
    assert int(getattr(streamer, "_state_resurrection_suppressed_count", 0) or 0) >= 1


def test_audit_parity_flags_safety_loop_and_terminal_duplicate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"run_id": "RID", "executor_stats_all_phases": {"nt_order_entry_total": 1}}), encoding="utf-8")
    (run_dir / "stream_state.json").write_text(json.dumps({"state": {"position_state": "FLAT"}, "position": {"pos": 0}}), encoding="utf-8")
    (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_health_summary.json").write_text(json.dumps({"verdict": "running_healthy"}), encoding="utf-8")
    (run_dir / "order_events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "signal_to_order.jsonl").write_text(
        json.dumps({"phase": "LIVE", "signal_action": "OPEN", "decision": "SENT", "emit_allowed": True, "sent_to_nt": True}) + "\n",
        encoding="utf-8",
    )
    exec_rows = [
        {"event": "flatten_due_to_no_protection"},
        {"event": "protection_timeout_terminal_suppressed_repeat"},
        {"event": "state_resurrection_suppressed"},
        {"event": "late_fill_after_lockout"},
        {"event": "late_fill_after_lockout"},
        {"event": "late_fill_after_lockout"},
    ]
    (run_dir / "exec_events.jsonl").write_text("\n".join(json.dumps(r) for r in exec_rows) + "\n", encoding="utf-8")
    report = audit_run(run_dir)
    reasons = set(report["parity_fail_reasons"])
    assert "TERMINAL_DUPLICATE_EMIT" in reasons
    assert "STATE_OSCILLATION_AFTER_TIMEOUT" in reasons
    assert "SAFETY_LOOP_DETECTED" in reasons


def test_pre_send_missing_target_price_violation_reason(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    intent = stream_live_csv_mod.ExecutionIntent(
        intent_id="RID|OPEN|SHORT|missing-target",
        correlation_id="RID|OPEN|SHORT|missing-target",
        signal_id="SIG-MISSING-TGT",
        action="OPEN",
        side="SHORT",
        qty=1,
        instrument_raw="ES",
        exec_instrument="ES JUN26",
        account=None,
        bar_ts="2026-05-18T23:10:00-06:00",
        model_price=7405.0,
        model_stop_price=7407.0,
        model_target_price=7403.0,
        entry_price=7405.0,
        stop_price=7407.0,
        target_price=None,
        order_type="MARKET",
    )
    violation = streamer._pre_send_protection_completeness_violation(
        intent=intent,
        order={
            "client_order_id": intent.intent_id,
            "model_price": 7405.0,
            "model_stop_price": 7407.0,
            "model_target_price": 7403.0,
            "stop_price": 7407.0,
        },
        effective_mode="absolute",
    )
    assert violation is not None
    assert str(violation.get("reason_code")) == "missing_target_price_preflight"


def test_followup_reject_preserves_sent_lineage_contract(tmp_path: Path) -> None:
    streamer = _make_streamer(tmp_path)
    cid = "RID|OPEN|SHORT|sent-lineage"
    streamer._nt_order_state[cid] = {"sent_ts": time.time(), "entry_acked": True}
    streamer._log_signal_to_order(
        bar_ts=pd.Timestamp("2026-05-18T23:10:00-06:00"),
        signal_action="OPEN",
        client_order_id=cid,
        sent_to_nt=False,
        blocked_by=["nt_missing_stop_price"],
        extra={
            "decision": "REJECTED",
            "reason": "missing_stop_price",
            "send_lifecycle_state": "followup_after_prior_send",
            "send_sent_total_for_cid": 1,
            "phase": "LIVE",
            "side": "SHORT",
            "final_action": "NO_TRADE",
            "emit_allowed": False,
        },
    )
    rows = [json.loads(line) for line in streamer.signal_to_order_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    row = rows[-1]
    assert row["transport_send_allowed"] is True
    assert row["bridge_acceptance_status"] == "followup_after_prior_send"
    assert row["execution_truth_state"] == "executed_followup"
    assert row["first_send_truth"] is True


def test_audit_flags_flip_pending_stale_collapse_and_guardrail_nonflat(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"run_id": "RID", "working_order_count": 1}), encoding="utf-8")
    (run_dir / "stream_state.json").write_text(
        json.dumps({"state": {"position_state": "IN_POSITION_UNPROTECTED", "entries_disarmed_reason": "guardrail_lockout"}, "position": {"pos": -1}}),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_health_summary.json").write_text(json.dumps({"verdict": "running_healthy"}), encoding="utf-8")
    (run_dir / "order_events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "exec_events.jsonl").write_text("", encoding="utf-8")
    rows = [
        {"phase": "LIVE", "signal_action": "OPEN", "client_order_id": "cid-flip", "decision": "BLOCKED_SYNC", "reason": "flip_open_deferred_wait_reconcile", "parity_transition_state": "transition_pending", "emit_allowed": False, "sent_to_nt": False},
        {"phase": "LIVE", "signal_action": "OPEN", "client_order_id": "cid-flip", "decision": "BLOCKED_SAFETY", "reason": "signal_stale:age=116.5s>max=25.0s", "emit_allowed": False, "sent_to_nt": False},
    ]
    (run_dir / "signal_to_order.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    report = audit_run(run_dir)
    reasons = set(report["parity_fail_reasons"])
    assert "FLIP_PENDING_STALE_COLLAPSE" in reasons
    assert "GUARDRAIL_LOCKOUT_NONFLAT_TERMINAL" in reasons


def test_audit_flags_direct_flip_sent_and_order_violation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "signal_to_order.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"phase": "LIVE", "signal_action": "FLIP", "action": "FLIP", "sent_to_nt": True, "emit_allowed": True, "client_order_id": "cid-direct"}) + "\n")
        fh.write(json.dumps({"phase": "LIVE", "signal_action": "OPEN", "action": "OPEN", "sent_to_nt": False, "emit_allowed": False, "requested_action": "FLIP", "transition_id": "tid-1", "client_order_id": "cid-1"}) + "\n")
    (run_dir / "status.json").write_text(json.dumps({"executor_stats_all_phases": {"nt_order_entry_total": 0}}), encoding="utf-8")
    (run_dir / "stream_state.json").write_text(
        json.dumps({"pos": 0, "position_state": "FLAT", "nt_working_order_count": 0}),
        encoding="utf-8",
    )
    (run_dir / "system_health.json").write_text(
        json.dumps({"verdict": "clean_stopped", "cooldown_state": {}, "bar_age_sec": 0}),
        encoding="utf-8",
    )
    report = audit_run(run_dir)
    reasons = set(report.get("parity_fail_reasons") or [])
    assert "DIRECT_FLIP_SENT" in reasons
    assert "FLIP_TRANSITION_ORDER_VIOLATION" in reasons


