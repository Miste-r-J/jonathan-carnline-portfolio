from typing import Any, Dict, List


def _make_streamer(order_events: List[Dict[str, Any]]):
    from trading_system.runtime_engine.integrations.cli.live_trading_runtime import LiveCSVStreamer

    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)

    def _ensure():
        if not hasattr(streamer, "_order_events_cache"):
            streamer._order_events_cache = None
        if not hasattr(streamer, "_order_events_mtime"):
            streamer._order_events_mtime = None
        if not hasattr(streamer, "_fill_truth_cache"):
            streamer._fill_truth_cache = {}
        if not hasattr(streamer, "_fill_truth_mtime"):
            streamer._fill_truth_mtime = None

    streamer._ensure_compat_defaults = _ensure
    streamer._nt_order_state = {}
    streamer._load_order_events = lambda fresh=False: list(order_events)
    streamer._order_events_mtime = 123.0
    return streamer, LiveCSVStreamer


def test_fill_truth_classifies_manual_flatten_as_exit():
    cid = "RUN|preset|ES MAR26|2026-02-17T08:00:00-07:00|OPEN|LONG|abc123"
    events = [
        {
            "type": "FILL",
            "client_order_id": cid,
            "ninja_order_id": "ENTRY1",
            "side": "LONG",
            "fill_qty": 1,
            "fill_price": 100.0,
            "timestamp": "2026-02-17T08:00:01-07:00",
        },
        # Manual flatten: market order fill, no role/action metadata, opposite side.
        {
            "type": "FILL",
            "client_order_id": cid,
            "ninja_order_id": "MKT_FLAT_1",
            "side": "SHORT",
            "fill_qty": 1,
            "fill_price": 101.0,
            "timestamp": "2026-02-17T09:00:00-07:00",
        },
    ]
    streamer, cls = _make_streamer(events)
    idx = cls._load_fill_truth_index(streamer, fresh=True)
    rec = idx[cid]
    assert float(rec.get("entry_fill_qty") or 0.0) == 1.0
    assert float(rec.get("exit_fill_qty") or 0.0) == 1.0


def test_summarize_fill_truth_last_cid_prefers_most_recent_fill():
    from trading_system.runtime_engine.integrations.cli.live_trading_runtime import LiveCSVStreamer

    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    idx = {
        "t1": {"intent_side": "LONG", "entry_fill_qty": 1, "entry_fill_ts_epoch": 100.0, "exit_fill_qty": 1, "exit_fill_ts_epoch": 110.0},
        "t2": {"intent_side": "LONG", "entry_fill_qty": 1, "entry_fill_ts_epoch": 200.0, "exit_fill_qty": None, "exit_fill_ts_epoch": None},
    }
    summary = LiveCSVStreamer._summarize_fill_truth(streamer, idx)
    assert summary["last_cid"] == "t2"
