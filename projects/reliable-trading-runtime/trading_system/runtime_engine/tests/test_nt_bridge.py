from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, List

from trading_system.runtime_engine.integrations.nt_bridge import NTBridgeServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_nt_bridge_decodes_concatenated_json_frames() -> None:
    bridge = NTBridgeServer("127.0.0.1", _free_port(), logger=None, log_path=None)
    payload = (
        json.dumps({"schema_version": 1, "type": "POSITION_SNAPSHOT", "account": "SIM"})
        + json.dumps({"schema_version": 1, "type": "ACCOUNT_SNAPSHOT", "account": "SIM"})
    )

    msgs = bridge._decode_json_messages(payload)
    assert len(msgs) == 2
    assert msgs[0].get("type") == "POSITION_SNAPSHOT"
    assert msgs[1].get("type") == "ACCOUNT_SNAPSHOT"


def test_nt_bridge_flushes_unterminated_order_capable_json_on_eof() -> None:
    port = _free_port()
    received: List[Dict[str, Any]] = []

    bridge = NTBridgeServer("127.0.0.1", port, logger=None, log_path=None)
    bridge.add_callback(received.append)
    bridge.start()

    client = socket.create_connection(("127.0.0.1", port), timeout=2.0)
    try:
        payload = json.dumps(
            {
                "schema_version": 1,
                "type": "HEARTBEAT",
                "source": "NinjaRepoBridge",
                "platform": "NinjaTrader8",
            }
        )
        client.sendall(payload.encode("utf-8"))
        client.shutdown(socket.SHUT_WR)

        deadline = time.time() + 2.0
        while time.time() < deadline and not received:
            time.sleep(0.01)

        assert len(received) == 1
        assert received[0].get("type") == "HEARTBEAT"
    finally:
        try:
            client.close()
        finally:
            bridge.shutdown()


def test_nt_bridge_rejects_duplicate_client_when_current_socket_is_healthy() -> None:
    bridge = NTBridgeServer("127.0.0.1", _free_port(), logger=None, log_path=None)
    bridge.is_connected = True
    bridge._handshake_event.set()
    bridge.last_rx_ts = time.time()
    bridge.remote_addr = ("127.0.0.1", 54016)
    bridge.local_addr = ("127.0.0.1", 5019)
    bridge._client_kind = "order_capable"
    bridge._client = object()  # type: ignore[assignment]

    assert bridge._should_reject_duplicate_client() is True


def test_nt_bridge_send_refuses_stale_disconnected_client() -> None:
    class _Client:
        def __init__(self) -> None:
            self.sent = False

        def sendall(self, _data: bytes) -> None:
            self.sent = True

    client = _Client()
    bridge = NTBridgeServer("127.0.0.1", _free_port(), logger=None, log_path=None)
    bridge._client = client  # type: ignore[assignment]
    bridge.is_connected = False

    assert bridge.send({"type": "ORDER", "client_order_id": "CID-STALE"}) is False
    assert client.sent is False
    assert bridge.last_err == "send_no_connected_client"


def test_nt_bridge_does_not_reject_duplicate_when_current_socket_is_telemetry_only() -> None:
    bridge = NTBridgeServer("127.0.0.1", _free_port(), logger=None, log_path=None)
    bridge.is_connected = True
    bridge._handshake_event.set()
    bridge.last_rx_ts = time.time()
    bridge._client_kind = "telemetry_only"
    bridge._client = object()  # type: ignore[assignment]

    assert bridge._should_reject_duplicate_client() is False


def test_nt_bridge_rejects_telemetry_only_preview_even_without_active_client() -> None:
    bridge = NTBridgeServer("127.0.0.1", _free_port(), logger=None, log_path=None)

    assert bridge._should_reject_previewed_client(
        [{"event_type": "HEARTBEAT", "source": "nt_account_api", "ts_local": "2026-06-03T01:20:00-06:00"}]
    ) is True


def test_nt_bridge_allows_order_capable_preview_even_with_active_client() -> None:
    bridge = NTBridgeServer("127.0.0.1", _free_port(), logger=None, log_path=None)
    bridge.is_connected = True
    bridge._handshake_event.set()
    bridge.last_rx_ts = time.time()
    bridge._client_kind = "order_capable"
    bridge._client = object()  # type: ignore[assignment]

    assert bridge._should_reject_previewed_client(
        [{"type": "HELLO", "protocol_version": 1, "platform": "NinjaTrader8", "addon_version": "2026.01.20"}]
    ) is False


def test_nt_bridge_telemetry_only_heartbeat_does_not_mark_handshake() -> None:
    bridge = NTBridgeServer("127.0.0.1", _free_port(), logger=None, log_path=None)

    ok = bridge._validate_and_mark_handshake(
        {"event_type": "HEARTBEAT", "source": "nt_account_api", "ts_local": "2026-06-03T01:20:00-06:00"}
    )

    assert ok is True
    assert bridge.handshake_ok() is False
    assert bridge.client_kind() == "telemetry_only"


def test_nt_bridge_hello_marks_order_capable_handshake() -> None:
    bridge = NTBridgeServer("127.0.0.1", _free_port(), logger=None, log_path=None)

    ok = bridge._validate_and_mark_handshake(
        {"type": "HELLO", "protocol_version": 1, "platform": "NinjaTrader8", "addon_version": "1.0"}
    )

    assert ok is True
    assert bridge.handshake_ok() is True
    assert bridge.client_kind() == "order_capable"


def test_nt_bridge_ninjarepo_heartbeat_marks_order_capable_handshake() -> None:
    bridge = NTBridgeServer("127.0.0.1", _free_port(), logger=None, log_path=None)

    ok = bridge._validate_and_mark_handshake(
        {
            "type": "HEARTBEAT",
            "protocol_version": 1,
            "source": "NinjaRepoBridge",
            "platform": "NinjaTrader8",
        }
    )

    assert ok is True
    assert bridge.handshake_ok() is True
    assert bridge.client_kind() == "order_capable"


def test_nt_bridge_duplicate_reject_does_not_start_read_loop() -> None:
    bridge = NTBridgeServer("127.0.0.1", _free_port(), logger=None, log_path=None)
    calls: List[str] = []

    def fake_replace_client(client: Any, addr: Any, *, preview_remainder: bytes = b"") -> bool:
        calls.append("replace")
        return False

    def fake_start_read_loop() -> None:
        calls.append("read_loop")

    def fake_wait_for_handshake() -> bool:
        calls.append("wait")
        return True

    bridge._replace_client = fake_replace_client  # type: ignore[assignment]
    bridge._start_read_loop = fake_start_read_loop  # type: ignore[assignment]
    bridge._wait_for_handshake = fake_wait_for_handshake  # type: ignore[assignment]

    assert bridge._accept_client(object(), ("127.0.0.1", 12345)) is False
    assert calls == ["replace"]


def test_nt_bridge_accept_does_not_block_on_handshake_wait() -> None:
    bridge = NTBridgeServer("127.0.0.1", _free_port(), logger=None, log_path=None)
    calls: List[str] = []

    def fake_replace_client(client: Any, addr: Any, *, preview_remainder: bytes = b"") -> bool:
        calls.append("replace")
        return True

    def fake_start_read_loop() -> None:
        calls.append("read_loop")

    def fake_wait_for_handshake() -> bool:
        calls.append("wait")
        return True

    bridge._replace_client = fake_replace_client  # type: ignore[assignment]
    bridge._start_read_loop = fake_start_read_loop  # type: ignore[assignment]
    bridge._wait_for_handshake = fake_wait_for_handshake  # type: ignore[assignment]

    assert bridge._accept_client(object(), ("127.0.0.1", 12345)) is True
    assert calls == ["replace", "read_loop"]
