from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from trading_system.runtime_engine.integrations.cli.live_trading_runtime import LiveCSVStreamer


def _make_streamer() -> LiveCSVStreamer:
    s = object.__new__(LiveCSVStreamer)
    s.state = SimpleNamespace(active_client_order_id=None, position_state="EXITING")
    s._open_trade = None
    s._nt_order_state = {}
    s._exec_instrument_key = lambda: "ES"
    s._event_ts_epoch = lambda ts: float(ts or 0.0)
    s.late_fill_threshold_sec = 30.0
    s._log_exec_event = MagicMock()
    return s


def test_terminal_fill_fallback_uses_single_open_trade_cardinality():
    s = _make_streamer()
    s._nt_order_state = {
        "cid-open-1": {
            "instrument": "ES",
            "entry_filled": True,
            "exit_fill_ts": None,
            "side": "LONG",
            "qty": 1,
            "close_in_progress": True,
            "sent_ts": 100.0,
            "last_update_event_ts": 110.0,
        }
    }
    msg = {"instrument": "ES", "side": "SELL", "fill_qty": 1, "timestamp": 115.0}

    cid, detail = s._resolve_terminal_fill_cid_fallback(msg=msg, untracked_cid="UNTRACKED|x")

    assert cid == "cid-open-1"
    assert detail.get("match_rule") in {"single_open_trade_cardinality", "active_context_match"}


def test_non_fillable_snapshot_noise_detected_for_missing_qty():
    s = _make_streamer()
    is_noise, reason = s._is_non_fillable_snapshot_noise(
        {"type": "FILL", "side": "SELL", "fill_price": 5200.25, "fill_qty": None}
    )

    assert is_noise is True
    assert reason == "missing_or_zero_fill_qty"


def test_snapshot_adopt_does_not_mutate_active_trade_on_snapshot_only_orphan_cid():
    s = _make_streamer()
    s._open_trade = {"client_order_id": "run|OPEN|123"}
    s._pos = 1
    s.state.active_client_order_id = None
    s._extract_snapshot_exits = lambda orders, inst_key: {"client_order_id": None}
    s._resolve_canonical_position = lambda pos_qty, side, source: {"canonical_side": "LONG", "signed_pos": 1}
    s._round_to_tick = lambda v, label=None: v
    s._ensure_position_uid = MagicMock()

    s._adopt_snapshot_position(
        inst_key="ES",
        pos_qty=1.0,
        avg_price=5201.0,
        side="LONG",
        msg={"timestamp": "2026-05-10T12:00:00Z"},
        orders=[],
    )

    assert s.state.active_client_order_id is None
    assert s._open_trade["client_order_id"] == "run|OPEN|123"
    s._ensure_position_uid.assert_not_called()
