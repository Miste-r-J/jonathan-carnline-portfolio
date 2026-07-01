from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from na.bot.EnhancedGuardrails import EnhancedGuardRailConfig, EnhancedGuardrails
from na.bot.PropRiskManager import PropRiskConfig
from na.bot.VolatilityFilter import VolatilityConfig
from na.bot.config import PRESETS
from na.discord_addons.cli.stream_live_csv import (
    FallbackPolicyCfg,
    FallbackTransition,
    LiveCSVStreamer,
    StreamState,
    TRADES_HEADER_FIELDS,
    _apply_phase2_manifest_overrides,
    _log_effective_run_config,
)


def _write_trades_csv(path, *, cid: str, side: str, entry_price: float, entry_ts: str) -> None:
    path.write_text(",".join(TRADES_HEADER_FIELDS) + "\n", encoding="utf-8")
    row = {k: "" for k in TRADES_HEADER_FIELDS}
    row.update(
        {
            "entry_ts": entry_ts,
            "side": side,
            "qty": "1",
            "entry_price": str(entry_price),
            "client_order_id": cid,
            "nt_connected": "True",
            "handshake_ok": "True",
            "proof_only": "False",
        }
    )
    with path.open("a", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=TRADES_HEADER_FIELDS, lineterminator="\n")
        w.writerow(row)


def _make_streamer(tmp_path):
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    s.state = StreamState()
    s.state.position_state = "EXITING"
    s.run_id = "RUNID"
    s.instrument_alias = "ES"
    s.nt_instrument = "ES MAR26"
    s.exec_instrument = "ES MAR26"
    s.session_tz = "America/Denver"

    s.nt_enabled = True
    s.nt_exec_policy = "paper"
    s.nt_bridge = None

    s._pos = 1
    s._bars_in_trade = 1
    s._entry_time = None
    s._entry_price = 6882.0
    s._entry_stop = 6872.0
    s._entry_target = 6892.0
    s._last_prob = 0.70
    s._last_close = 6886.0

    s._active_close_correlation_id = None
    s._active_close_started_ts = None
    s._close_intents = {}
    s._exit_stuck_flatten_attempted = set()
    s._close_watchdog_resend_attempted = set()

    s._emit_event = lambda *args, **kwargs: None
    s._flush_inflight_buffer = lambda *args, **kwargs: None
    s._log_exec_event = lambda *args, **kwargs: None
    s._log_order_event = lambda *args, **kwargs: None
    s._emit_event_explanation = lambda *args, **kwargs: None
    s._emit_block_event = lambda *args, **kwargs: None
    s._handle_io_failure = lambda *args, **kwargs: None
    s._save_state = lambda *args, **kwargs: None
    s._write_jsonl = lambda *args, **kwargs: None
    s._apply_fill_truth_reconciliation = lambda *args, **kwargs: None

    s._instrument_lockouts = {}
    s._nt_stop_update_pending = {}
    s._pending_trade_updates = {}
    s._fx_state = SimpleNamespace(active_trades={})
    s.policy = None
    s.sim_mode = False
    s.leak_guard_strict = False
    s._commission_per_contract = 0.0
    s._slippage_ticks_per_side = 0.0

    s.out_dir = tmp_path
    s.trades_csv = tmp_path / "trades.csv"
    s.metrics_log_path = tmp_path / "live_metrics.jsonl"
    s.emission_ledger_path = tmp_path / "emission_ledger.jsonl"
    s.entry_blocks_path = tmp_path / "entry_blocks.jsonl"
    s.signal_to_order_path = tmp_path / "signal_to_order.jsonl"

    s.fallback = SimpleNamespace(pos=1, bars_in_trade=1, entry_time=None, entry_price=6882.0)
    s.instrument = SimpleNamespace(alias="ES", point_value=50.0, tick_size=0.25)
    return s


def test_A_close_fill_correlation_end_to_end(tmp_path, capsys):
    s = _make_streamer(tmp_path)
    entry_cid = "ENTRYCID"
    close_cid = "CLOSECID"
    s.state.active_client_order_id = close_cid

    _write_trades_csv(s.trades_csv, cid=entry_cid, side="LONG", entry_price=6882.0, entry_ts="2026-02-24T09:35:00-07:00")
    s._open_trade = {
        "side": "LONG",
        "entry_price": 6882.0,
        "risk": 10.0,
        "contracts": 1.0,
        "client_order_id": entry_cid,
        "signal_id": "SIG",
        "datetime": pd.Timestamp("2026-02-24T09:35:00-07:00"),
    }
    s._nt_order_state = {
        close_cid: {
            "intent_action": "CLOSE",
            "status": "SENT",
            "close_in_progress": True,
            "instrument": "ES MAR26",
            "signal_id": "SIG",
            "stop_price": 6872.0,
            "target_price": 6892.0,
        }
    }

    msg = {
        "type": "FILL",
        "client_order_id": "UNTRACKED|ES MAR26|406099040287|1771951203638",
        "instrument": "ES MAR26",
        "fill_qty": 1,
        "fill_price": 6886.0,
        "timestamp": "2026-02-24T09:40:03.638575-07:00",
        "role": "",
        "status": "FILLED",
    }
    s._update_nt_order_state(msg)
    out = capsys.readouterr().out
    assert "REASSOCIATED untracked fill" in out

    rows = list(csv.DictReader(s.trades_csv.read_text(encoding="utf-8").splitlines()))
    row = [r for r in rows if r.get("client_order_id") == entry_cid][0]
    assert row["exit_fill_price"] not in ("", None)
    assert float(row["exit_fill_price"]) == pytest.approx(6886.0)
    assert row["exit_reason"] == "model_close"
    assert s._nt_order_state[close_cid]["exit_reason"] == "model_close"
    assert s.state.daily_r != 0.0
    assert s.state.position_state == "FLAT"
    assert s._pos == 0
    assert s.fallback.pos == 0
    assert s._open_trade is None


def test_B_reentry_after_close_sync_flat(tmp_path):
    s = _make_streamer(tmp_path)
    s.state.position_state = "IN_POSITION_PROTECTED"
    s._pos = 1
    s.fallback.pos = 1
    s._open_trade = {"client_order_id": "ENTRYCID", "side": "LONG", "entry_price": 100.0, "risk": 5.0, "contracts": 1.0}

    s._sync_position_flat("close_fill")
    assert s.state.position_state == "FLAT"
    assert s._pos == 0
    assert s.fallback.pos == 0
    assert s._open_trade is None

    cfg = FallbackPolicyCfg(
        p_buy=0.75,
        p_sell=0.50,
        proba_cut_bad=0.70,
        atr_k=2.0,
        max_bars_in_trade=20,
        trade_window_start="07:30",
        trade_window_end="14:30",
        session_tz="America/Denver",
        hold_threshold_long=0.60,
        hold_threshold_short=0.40,
        exit_prob_confirm_bars=1,
        persist_prob_bars=1,
        min_hold_bars=3,
    )
    s.fallback = FallbackTransition(s.instrument, cfg, trace=False, status=None, emit=None, model_run_id=None, contracts_per_trade=1.0)

    row = pd.Series({"Datetime": pd.Timestamp("2026-02-24T10:00:00-07:00"), "Close": 100.0, "proba": 0.78})
    ev = s.fallback.step(row)
    assert ev is not None
    assert ev["type"] == "OPEN"
    assert ev["side"] == "LONG"
    assert s._should_publish(ev) is True


def test_C_hold_threshold_enforcement_min_hold_bars():
    inst = SimpleNamespace(alias="ES", point_value=50.0, tick_size=0.25)
    cfg = FallbackPolicyCfg(
        p_buy=0.75,
        p_sell=0.50,
        proba_cut_bad=0.70,
        atr_k=2.0,
        max_bars_in_trade=20,
        trade_window_start="00:00",
        trade_window_end="23:59",
        session_tz="America/Denver",
        hold_threshold_long=0.60,
        hold_threshold_short=0.40,
        exit_prob_confirm_bars=1,
        persist_prob_bars=1,
        min_hold_bars=3,
    )
    fb = FallbackTransition(inst, cfg, trace=False, status=None, emit=None, model_run_id=None, contracts_per_trade=1.0)

    # Bar 0: OPEN long
    ev0 = fb.step(pd.Series({"Datetime": pd.Timestamp("2026-02-24T09:35:00-07:00"), "Close": 100.0, "proba": 0.80}))
    assert ev0 is not None and ev0["type"] == "OPEN"
    assert fb.pos == 1

    # Bar 1: p above hold threshold -> no CLOSE
    ev1 = fb.step(pd.Series({"Datetime": pd.Timestamp("2026-02-24T09:40:00-07:00"), "Close": 101.0, "proba": 0.68}))
    assert ev1 is None
    assert fb.bars_in_trade == 1

    # Bar 2: p below hold threshold but min_hold_bars not met -> no CLOSE
    ev2 = fb.step(pd.Series({"Datetime": pd.Timestamp("2026-02-24T09:45:00-07:00"), "Close": 100.5, "proba": 0.55}))
    assert ev2 is None
    assert fb.bars_in_trade == 2

    # Bar 3: still below hold threshold and min_hold_bars met -> CLOSE
    ev3 = fb.step(pd.Series({"Datetime": pd.Timestamp("2026-02-24T09:50:00-07:00"), "Close": 100.5, "proba": 0.55}))
    assert ev3 is not None
    assert ev3["type"] == "CLOSE"


def test_sim_close_while_flat_writes_trades_csv_row(tmp_path):
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    s.trades_csv = tmp_path / "trades.csv"
    s.trades_csv.write_text(",".join(TRADES_HEADER_FIELDS) + "\n", encoding="utf-8")
    s.instrument_alias = "ES"
    s.nt_instrument = "ES MAR26"
    s.nt_enabled = False
    s.nt_bridge = None
    s.nt_exec_policy = "disabled"

    ev = {
        "type": "CLOSE",
        "instrument": "ES MAR26",
        "datetime": pd.Timestamp("2026-02-24T09:40:00-07:00"),
        "price": 6886.0,
        "prob": 0.5,
        "side": "",
        "ctx": {"exit_reason": "unknown"},
    }
    s._append_sim_close_row(ev, reason="flat_side_unresolved")
    s._append_sim_close_row(ev, reason="flat_side_unresolved")  # idempotent

    rows = list(csv.DictReader(s.trades_csv.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    row = rows[0]
    assert row["client_order_id"].startswith("SIM_CLOSE|")
    assert row["exit_reason"].startswith("sim_close:flat_side_unresolved")
    assert row["proof_only"] in {"True", "true", "1"}
    assert row["exit_ts"] != ""
    assert row["actual_exit_price"] != ""


def test_sim_close_suppressed_in_replay_local_sim_mode(tmp_path):
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    s.trades_csv = tmp_path / "trades.csv"
    s.trades_csv.write_text(",".join(TRADES_HEADER_FIELDS) + "\n", encoding="utf-8")
    s.instrument_alias = "ES"
    s.nt_instrument = "ES MAR26"
    s.nt_enabled = False
    s.nt_bridge = None
    s.run_mode = "replay"
    s.allow_replay_exec_to_nt = False
    s.replay_local_sim = True
    s.replay_emit_legacy_sim_close = False

    ev = {
        "type": "CLOSE",
        "instrument": "ES MAR26",
        "datetime": pd.Timestamp("2026-02-24T09:40:00-07:00"),
        "price": 6886.0,
        "prob": 0.5,
        "side": "",
        "ctx": {"exit_reason": "unknown"},
    }
    s._append_sim_close_row(ev, reason="flat_side_unresolved")

    rows = list(csv.DictReader(s.trades_csv.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 0


def test_input_pinning_skips_non_file_sources(tmp_path):
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    s.pin_input = True
    s.run_mode = "replay"
    s.pinned_input_dir = tmp_path / "pin"
    s.pinned_input_path = s.pinned_input_dir / "pinned_input.csv"
    s.require_pinned_hash_match = True

    out = s._apply_input_pinning(Path("fiber"))
    assert str(out).lower().endswith("fiber")
    assert s.original_csv_sha256 is None
    assert s.pinned_csv_sha256 is None


def test_D_double_fill_idempotency(tmp_path):
    s = _make_streamer(tmp_path)
    entry_cid = "ENTRYCID"
    close_cid = "CLOSECID"
    s.state.active_client_order_id = close_cid
    _write_trades_csv(s.trades_csv, cid=entry_cid, side="LONG", entry_price=6882.0, entry_ts="2026-02-24T09:35:00-07:00")
    s._open_trade = {"side": "LONG", "entry_price": 6882.0, "risk": 10.0, "contracts": 1.0, "client_order_id": entry_cid, "signal_id": "SIG"}
    s._nt_order_state = {close_cid: {"intent_action": "CLOSE", "status": "SENT", "close_in_progress": True, "instrument": "ES MAR26"}}

    msg = {
        "type": "FILL",
        "client_order_id": "UNTRACKED|ES MAR26|406099040287|1771951203638",
        "instrument": "ES MAR26",
        "fill_qty": 1,
        "fill_price": 6886.0,
        "timestamp": "2026-02-24T09:40:03.638575-07:00",
        "role": "",
        "status": "FILLED",
    }
    s._update_nt_order_state(msg)
    r1 = float(s.state.daily_r)
    s._update_nt_order_state(msg)
    r2 = float(s.state.daily_r)
    assert r2 == pytest.approx(r1)


def test_D1_reassociated_untracked_fill_does_not_trigger_prop_entry_lockout(tmp_path):
    s = _make_streamer(tmp_path)
    entry_cid = "ENTRYCID"
    s.state.active_client_order_id = entry_cid
    _write_trades_csv(
        s.trades_csv,
        cid=entry_cid,
        side="LONG",
        entry_price=6608.25,
        entry_ts="2026-03-26T07:50:00-06:00",
    )
    s._open_trade = {
        "side": "LONG",
        "entry_price": 6608.25,
        "risk": 8.0,
        "contracts": 1.0,
        "client_order_id": entry_cid,
        "signal_id": "SIG",
        "datetime": pd.Timestamp("2026-03-26T07:50:00-06:00"),
    }
    s.prop_guardrails = EnhancedGuardrails(
        EnhancedGuardRailConfig(fill_slippage_max=2.0),
        VolatilityConfig(),
        PropRiskConfig(),
    )
    s._nt_order_state = {
        entry_cid: {
            "intent_action": "OPEN",
            "status": "entry_filled",
            "close_in_progress": True,
            "instrument": "ES MAR26",
            "side": "LONG",
            "model_price": 6608.25,
            "entry_filled": True,
            "signal_id": "SIG",
            "stop_order_id": "STOP1",
        }
    }

    msg = {
        "type": "FILL",
        "client_order_id": entry_cid,
        "untracked_cid": "UNTRACKED|ES MAR26|444459900244|1774533300262",
        "ninja_order_id": "STOP1",
        "instrument": "ES MAR26",
        "fill_qty": 1,
        "fill_price": 6617.0,
        "side": "SHORT",
        "timestamp": "2026-03-26T07:55:00.262271-06:00",
        "role": "",
        "status": "FILLED",
    }
    s._update_nt_order_state(msg)

    assert bool(getattr(s, "_hard_lockout_active", False)) is False
    assert getattr(s, "_hard_lockout_code", None) is None
    assert s.state.position_state == "FLAT"
    assert s._nt_order_state[entry_cid]["exit_reason"] == "stop_hit"
    assert s._open_trade is None
    rows = list(csv.DictReader(s.trades_csv.read_text(encoding="utf-8").splitlines()))
    row = [r for r in rows if r.get("client_order_id") == entry_cid][0]
    assert row["exit_fill_price"] not in ("", None)
    assert float(row["exit_fill_price"]) == pytest.approx(6617.0)


def test_E_position_consistency_check_repairs_divergence(tmp_path, capsys):
    s = _make_streamer(tmp_path)
    s.state.position_state = "FLAT"
    s._pos = 1
    s.fallback.pos = 1
    s._open_trade = {"client_order_id": "ENTRYCID", "side": "LONG", "entry_price": 100.0, "risk": 5.0, "contracts": 1.0}

    s._check_position_consistency()
    out = capsys.readouterr().out
    assert "POSITION_DIVERGENCE" in out
    assert s.state.position_state == "FLAT"
    assert s._pos == 0
    assert s.fallback.pos == 0
    assert s._open_trade is None


def test_F_config_propagation_phase2_manifest_overrides(tmp_path):
    preset = PRESETS["es_maxpack_10_full_send_prop_safe_pnl"]
    assert preset["hold_threshold_long"] == pytest.approx(0.60)
    assert preset["hold_threshold_short"] == pytest.approx(0.40)
    assert int(preset["min_hold_bars"]) >= 1
    assert int(preset["exit_prob_confirm_bars"]) >= 1

    manifest_path = tmp_path / "manifest.json"
    repo_manifest = "simplified/artifacts/phase2/candidates/retrain_v2_full/manifest.json"
    manifest = {}
    try:
        manifest = json.loads(open(repo_manifest, "r", encoding="utf-8").read())
    except Exception:
        manifest = {}
    thresholds = dict(manifest.get("thresholds") or {}) if isinstance(manifest.get("thresholds"), dict) else {}
    # Ensure thresholds we assert on exist even if the repo manifest omits them.
    thresholds.setdefault("p_setup", 0.35)
    thresholds.setdefault("p_long", 0.75)
    thresholds.setdefault("p_short", 0.50)
    manifest = dict(manifest)
    manifest["thresholds"] = thresholds
    manifest.setdefault("config", {"tz": "America/Denver"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = argparse.Namespace(
        disable_safety_gates=False,
        phase2=False,
        phase2_tag="retrain_v2_full",
        phase2_use_manifest_thresholds=True,
        _preset_fields=set(preset.keys()),
        p_buy=float(preset["p_buy"]),
        p_sell=float(preset["p_sell"]),
        p_setup=None,
        p_long=None,
        p_short=None,
        setup_model_path=None,
        dir_model_path=None,
        model=None,
    )
    args = _apply_phase2_manifest_overrides(args, manifest_path, manifest, argv_tokens=[])
    eff = _log_effective_run_config(args)
    assert eff["phase2"] is True
    assert args.p_long is not None
    assert args.p_short is not None
    assert eff["p_buy"] == pytest.approx(float(args.p_long))
    assert eff["p_sell"] == pytest.approx(1.0 - float(args.p_short))

    # Ensure fallback cfg retains hold/min-hold knobs from preset and is accessible.
    cfg = FallbackPolicyCfg(
        p_buy=eff["p_buy"],
        p_sell=eff["p_sell"],
        proba_cut_bad=float(preset["proba_cut_bad"]),
        atr_k=float(preset.get("atr_k", 2.0) or 2.0),
        max_bars_in_trade=int(preset["max_hold_bars"]),
        trade_window_start=str(preset["trade_window_start"]),
        trade_window_end=str(preset["trade_window_end"]),
        session_tz="America/Denver",
        hold_threshold_long=float(preset["hold_threshold_long"]),
        hold_threshold_short=float(preset["hold_threshold_short"]),
        exit_prob_confirm_bars=int(preset["exit_prob_confirm_bars"]),
        persist_prob_bars=int(preset["persist_prob_bars"]),
        min_hold_bars=max(1, int(preset["min_hold_bars"])),
    )
    assert cfg.hold_threshold_long == pytest.approx(0.60)
    assert cfg.hold_threshold_short == pytest.approx(0.40)
    assert cfg.min_hold_bars >= 1
    assert cfg.exit_prob_confirm_bars >= 1
