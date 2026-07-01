import time

import pytest

from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer, StreamState


def _make_streamer():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.nt_enabled = True
    streamer.nt_exec_policy = "paper"
    streamer.state = StreamState()
    streamer.state.position_state = "IN_POSITION_UNPROTECTED"
    streamer._hard_lockout_active = False
    streamer._protection_timeout_detail = None
    streamer._nt_order_state = {}
    streamer._protection_timeout_attempts = {}
    streamer._protection_timeout_last_attempt_ts = {}

    streamer.protection_timeout_sec = 10.0
    # Disable retries in the timeout scanner for deterministic unit tests.
    # Note: the production code treats 0 as "use default" via `or 2`.
    streamer.nt_protection_timeout_max_retries = -1
    streamer.nt_protection_timeout_retry_sec = 999.0
    streamer.protection_repair_enabled = False

    streamer._ensure_compat_defaults = lambda *args, **kwargs: None
    streamer._is_protection_working_state = lambda *args, **kwargs: False
    streamer._require_fresh_snapshot = lambda *args, **kwargs: True
    streamer._snapshot_protection_working = lambda *args, **kwargs: (False, {})
    streamer._confirm_nt_protection = lambda *args, **kwargs: None
    streamer._check_nt_stop_update_timeout = lambda *args, **kwargs: None
    streamer._log_exec_event = lambda *args, **kwargs: None
    streamer._maybe_request_nt_snapshot = lambda *args, **kwargs: None
    return streamer


def test_protection_timeout_skipped_during_close_in_progress(monkeypatch):
    streamer = _make_streamer()
    streamer._nt_order_state["CID"] = {
        "instrument": "ES MAR26",
        "intent_action": "OPEN",
        "entry_filled": True,
        "close_in_progress": True,
        "protection_first_became_false_ts": None,
    }

    called = {"count": 0}

    def _handle(*args, **kwargs):
        called["count"] += 1

    streamer._handle_protection_timeout = _handle

    streamer._enforce_nt_protection_timeouts()
    assert called["count"] == 0
    assert streamer._nt_order_state["CID"]["protection_first_became_false_ts"] is None


def test_protection_timeout_anchors_to_unprotected_time(monkeypatch):
    streamer = _make_streamer()
    streamer._nt_order_state["CID"] = {
        "instrument": "ES MAR26",
        "intent_action": "OPEN",
        "entry_filled": True,
        "close_in_progress": False,
        "protection_first_became_false_ts": None,
    }

    seen = {}

    def _handle(*, cid, state, reason, anchor_ts, now_ts):
        seen["cid"] = cid
        seen["anchor_ts"] = anchor_ts
        seen["now_ts"] = now_ts
        seen["reason"] = reason

    streamer._handle_protection_timeout = _handle

    t = {"now": 1000.0}

    def _fake_time():
        return t["now"]

    monkeypatch.setattr(time, "time", _fake_time)

    streamer._enforce_nt_protection_timeouts()
    assert streamer._nt_order_state["CID"]["protection_first_became_false_ts"] == pytest.approx(1000.0)
    assert seen == {}

    t["now"] = 1011.0
    streamer._enforce_nt_protection_timeouts()
    assert seen["cid"] == "CID"
    assert seen["anchor_ts"] == pytest.approx(1000.0)
    assert seen["now_ts"] == pytest.approx(1011.0)
    assert seen["reason"] == "no_protection_working"


def test_protection_timeout_skipped_when_exit_leg_already_filled():
    streamer = _make_streamer()
    streamer._nt_order_state["CID"] = {
        "instrument": "ES MAR26",
        "intent_action": "OPEN",
        "entry_filled": True,
        "close_in_progress": False,
        "stop_order_id": "STOP123",
        "stop_state": "FILLED",
        "last_update_order_id": "STOP123",
        "last_update_state": "FILLED",
        "protection_first_became_false_ts": 1000.0,
    }

    called = {"count": 0}

    def _handle(*args, **kwargs):
        called["count"] += 1

    streamer._handle_protection_timeout = _handle
    streamer._enforce_nt_protection_timeouts()

    assert called["count"] == 0
    assert streamer._nt_order_state["CID"]["protection_first_became_false_ts"] is None
