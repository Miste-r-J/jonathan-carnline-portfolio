from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer, StreamState


def _make_streamer(active_cid: str | None):
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.state = StreamState()
    streamer.state.active_client_order_id = active_cid
    streamer._nt_order_state = {}
    return streamer


def test_extract_snapshot_exits_prefers_active_cid():
    streamer = _make_streamer(active_cid="CID_B")
    orders = [
        {
            "instrument": "ES MAR26",
            "order_type": "STOP",
            "order_state": "WORKING",
            "stop_price": 100.0,
            "qty": 1,
            "ninja_order_id": "A_STOP",
            "client_order_id": "CID_A",
        },
        {
            "instrument": "ES MAR26",
            "order_type": "LIMIT",
            "order_state": "WORKING",
            "limit_price": 110.0,
            "qty": 1,
            "ninja_order_id": "A_TGT",
            "client_order_id": "CID_A",
        },
        {
            "instrument": "ES MAR26",
            "order_type": "STOP",
            "order_state": "WORKING",
            "stop_price": 200.0,
            "qty": 1,
            "ninja_order_id": "B_STOP",
            "client_order_id": "CID_B",
        },
        {
            "instrument": "ES MAR26",
            "order_type": "LIMIT",
            "order_state": "WORKING",
            "limit_price": 210.0,
            "qty": 1,
            "ninja_order_id": "B_TGT",
            "client_order_id": "CID_B",
        },
    ]

    out = streamer._extract_snapshot_exits(orders, "ES MAR26")
    assert out["stop_order_id"] == "B_STOP"
    assert out["target_order_id"] == "B_TGT"
    assert out["stop_price"] == 200.0
    assert out["target_price"] == 210.0
    assert out["client_order_id"] == "CID_B"


def test_extract_snapshot_exits_falls_back_when_no_active_cid():
    streamer = _make_streamer(active_cid=None)
    orders = [
        {
            "instrument": "ES MAR26",
            "order_type": "STOP",
            "order_state": "WORKING",
            "stop_price": 100.0,
            "qty": 1,
            "ninja_order_id": "A_STOP",
            "client_order_id": "CID_A",
        },
        {
            "instrument": "ES MAR26",
            "order_type": "LIMIT",
            "order_state": "WORKING",
            "limit_price": 110.0,
            "qty": 1,
            "ninja_order_id": "A_TGT",
            "client_order_id": "CID_A",
        },
    ]

    out = streamer._extract_snapshot_exits(orders, "ES MAR26")
    assert out["stop_order_id"] == "A_STOP"
    assert out["target_order_id"] == "A_TGT"
    assert out["client_order_id"] == "CID_A"

