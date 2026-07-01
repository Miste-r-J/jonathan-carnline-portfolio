import json
import time
from pathlib import Path

import pandas as pd

from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer, StreamState


class _FakeBridge:
    def __init__(self):
        self.sent = []
        self.is_connected = True
        self._handshake = True

    def handshake_ok(self):
        return self._handshake

    def send(self, obj):
        self.sent.append(obj)
        return True


class _Ledger:
    def __init__(self):
        self.store = {}

    def get(self, cid):
        return self.store.get(cid)

    def mark(self, rec):
        self.store[rec.client_order_id] = rec


def _minimal_streamer(out_dir: Path) -> LiveCSVStreamer:
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)  # type: ignore
    s.nt_enabled = True
    s.nt_bridge = _FakeBridge()
    s.nt_instrument = "ES 03-26"
    s.nt_account = "Sim101"
    s.nt_order_type = "MARKET"
    s.nt_qty = 1
    s.nt_flatten_on_close = True
    s._consec_errors = 0
    s.nt_proof_mode = False
    s.pending_timeout_sec = 30
    s._pending_until = None
    s._pending_client_order_id = None
    s.tick_size = 0.25
    s._save_state = lambda: None  # type: ignore
    s.state = StreamState()
    s.preset = "replay"
    s.instrument_alias = "ES"
    s.trade_window_start = "00:00"
    s.trade_window_end = "23:59"
    s.session_tz = "America/Denver"
    s._cooldown_left = 0
    s._strategy_trade_limit = 0
    s._emission_ledger = type("L", (), {"seen": lambda *_: False, "append": lambda *_: None})()
    s.execution_ledger = _Ledger()
    s._nt_order_state = {}
    s.order_events_path = out_dir / "order_events.jsonl"
    s.nt_protection_timeout_sec = 0.5
    s._last_close = 4500.0
    return s


def main() -> None:
    out_dir = Path("tmp/nt_protection_replay")
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _minimal_streamer(out_dir)

    ev = {
        "type": "OPEN",
        "side": "LONG",
        "price": 4500.25,
        "risk": {"stop": 4498.0, "target": 4505.0},
        "datetime": pd.Timestamp("2025-01-02T09:35:00-07:00"),
    }
    s._maybe_send_nt_order(ev)
    cid = s.nt_bridge.sent[-1]["client_order_id"]
    s._update_nt_order_state({"type": "ORDER_ACK", "client_order_id": cid, "status": "ENTRY_ACKED"})
    s._update_nt_order_state({"type": "FILL", "client_order_id": cid, "status": "FILLED", "role": "ENTRY"})
    time.sleep(0.6)
    s._update_nt_order_state({"type": "ORDER_ACK", "client_order_id": cid, "status": "ENTRY_ACKED"})

    print("Replay complete.")
    print(f"order_events: {out_dir / 'order_events.jsonl'}")
    print("NT TX payloads:")
    print(json.dumps(s.nt_bridge.sent, indent=2))


if __name__ == "__main__":
    main()
