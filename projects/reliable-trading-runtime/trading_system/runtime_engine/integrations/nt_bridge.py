from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional ntplib import for clock-skew correction (Fix 3).
# Falls back to system time if ntplib is not installed.
# ---------------------------------------------------------------------------
try:
    import ntplib as _ntplib
    _NTPLIB_AVAILABLE = True
except ImportError:
    _ntplib = None  # type: ignore
    _NTPLIB_AVAILABLE = False

_ntp_logger = logging.getLogger(__name__)


class _NTPClock:
    """Thread-safe NTP-synchronized clock.

    Queries pool.ntp.org at startup and every sync_interval_sec (default 3600 s).
    All callers use ``ntp_clock.now()`` to get a timezone-aware datetime
    adjusted by the measured offset.
    """

    NTP_HOST = "pool.ntp.org"
    NTP_VERSION = 3

    def __init__(self, sync_interval_sec: float = 3600.0) -> None:
        self._offset_sec: float = 0.0          # positive = our clock is behind NTP
        self._last_sync_ts: Optional[float] = None
        self._skew_sec: float = 0.0
        self._sync_interval = float(sync_interval_sec)
        self._lock = threading.Lock()
        self._sync_timer: Optional[threading.Timer] = None

    def sync(self) -> None:
        """Synchronize with NTP server; safe to call from any thread."""
        if not _NTPLIB_AVAILABLE:
            return
        try:
            c = _ntplib.NTPClient()
            resp = c.request(self.NTP_HOST, version=self.NTP_VERSION)
            offset = resp.offset           # seconds our clock is behind NTP
            skew = resp.delay / 2.0        # one-way delay estimate
            with self._lock:
                self._offset_sec = offset
                self._skew_sec = skew
                self._last_sync_ts = time.time()
            _ntp_logger.info(
                "NTP sync OK  host=%s  offset=%.3fs  delay=%.3fs",
                self.NTP_HOST, offset, skew,
            )
        except Exception as exc:
            _ntp_logger.warning("NTP sync failed (%s); using previous offset=%.3fs", exc, self._offset_sec)
        finally:
            # Schedule next sync
            self._cancel_timer()
            self._sync_timer = threading.Timer(self._sync_interval, self.sync)
            self._sync_timer.daemon = True
            self._sync_timer.start()

    def start(self) -> None:
        """Perform initial sync and schedule recurring resyncs."""
        t = threading.Thread(target=self.sync, name="ntp-initial-sync", daemon=True)
        t.start()

    def now(self) -> datetime:
        """Return the current NTP-corrected time as a timezone-aware datetime."""
        with self._lock:
            offset = self._offset_sec
        corrected = datetime.now(timezone.utc) + timedelta(seconds=offset)
        return corrected.astimezone(ZoneInfo(DENVER_TZ))

    @property
    def skew_sec(self) -> float:
        with self._lock:
            return self._skew_sec

    @property
    def offset_sec(self) -> float:
        with self._lock:
            return self._offset_sec

    def _cancel_timer(self) -> None:
        if self._sync_timer is not None:
            try:
                self._sync_timer.cancel()
            except Exception:
                pass
            self._sync_timer = None

    def stop(self) -> None:
        self._cancel_timer()


# Module-level singleton — shared by utc_ts() and NTBridgeServer
ntp_clock = _NTPClock(sync_interval_sec=3600.0)


@dataclass
class NTBridgeHealth:
    is_connected: bool
    last_rx_ts: Optional[float]
    last_tx_ts: Optional[float]


DENVER_TZ = "America/Denver"


def utc_ts() -> str:
    # NTP-corrected execution-layer timestamp (America/Denver) for logs/records.
    return ntp_clock.now().isoformat()


def _to_denver_iso(value: Any) -> Any:
    if value is None:
        return value
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc).astimezone(ZoneInfo(DENVER_TZ)).isoformat()
        if isinstance(value, str):
            ts = value
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            dt_val = datetime.fromisoformat(ts)
            if dt_val.tzinfo is None:
                dt_val = dt_val.replace(tzinfo=timezone.utc)
            return dt_val.astimezone(ZoneInfo(DENVER_TZ)).isoformat()
    except Exception:
        return value
    return value


def _normalize_msg_ts(msg: dict) -> dict:
    if not isinstance(msg, dict):
        return msg
    out = dict(msg)
    for key in ("timestamp", "ts", "snapshot_ts", "snapshotTimestamp", "snapshot_time"):
        if key in out:
            out[key] = _to_denver_iso(out[key])
    return out


class NTBridgeServer:
    def __init__(
        self,
        host: str,
        port: int,
        logger: Optional[Any] = None,
        log_path: Optional[str] = None,
        handshake_timeout_sec: float = 60.0,
        require_newline_framing: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.logger = logger
        self.log_path = log_path
        self.handshake_timeout_sec = float(handshake_timeout_sec)
        self.require_newline_framing = bool(require_newline_framing)

        self._server: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._client_lock = threading.Lock()
        self._accept_thread: Optional[threading.Thread] = None
        self._read_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected_event = threading.Event()
        self._handshake_event = threading.Event()
        self._hello_event = threading.Event()
        self._snapshot_event = threading.Event()
        self._listening = False

        self._callbacks: List[Callable[[dict], None]] = []
        self._log_lock = threading.Lock()
        self._rx_buffer: bytearray = bytearray()

        self.is_connected = False
        self.last_rx_ts: Optional[float] = None
        self.last_tx_ts: Optional[float] = None
        self.last_err: Optional[str] = None
        self.remote_addr: Optional[Tuple[str, int]] = None
        self.local_addr: Optional[Tuple[str, int]] = None
        self.hello_seen = False
        self.snapshot_seen = False
        self._client_kind = "unknown"
        self._client_source: Optional[str] = None
        self._client_first_type: Optional[str] = None
        self._client_last_type: Optional[str] = None
        self._last_handshake_wait_log_ts: float = 0.0

    def add_callback(self, cb: Callable[[dict], None]) -> None:
        self._callbacks.append(cb)

    def start(self) -> None:
        if self._server is not None:
            return
        # Start NTP clock synchronization (Fix 3) — non-blocking background thread.
        ntp_clock.start()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(0.5)
        srv.bind((self.host, self.port))
        srv.listen(1)
        self._server = srv
        self._listening = True
        pid = os.getpid()
        self._log_info(
            "NT bridge listening on %s:%s (pid=%s)",
            self.host,
            self.port,
            pid,
        )
        self._log_status_snapshot("listening")
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="nt-bridge-accept",
            daemon=True,
        )
        self._accept_thread.start()

    def send(self, obj: dict) -> bool:
        payload = json.dumps(obj, ensure_ascii=True)
        data = (payload + "\n").encode("utf-8")
        with self._client_lock:
            if self._client is None or not bool(self.is_connected):
                self.last_err = "send_no_connected_client"
                self._log("TX", obj, warn="no_connected_client")
                return False
            try:
                self._client.sendall(data)
            except Exception as exc:
                self.last_err = f"send_failed:{exc}"
                self.is_connected = False
                self._connected_event.clear()
                self._handshake_event.clear()
                self._log("TX", obj, warn=f"send_failed:{exc}")
                return False
        self.last_tx_ts = time.time()
        self._log("TX", obj)
        return True

    def send_with_validation(self, obj: dict, recv_window_ms: int = 5000) -> bool:
        """Send an order message, rejecting it if its timestamp is too stale.

        Compares the message's ``timestamp`` / ``ts`` field against the current
        NTP-corrected time.  If the delta exceeds *recv_window_ms* the message
        is dropped and CRITICAL_LATENCY_REJECTION is logged (Fix 4).

        Args:
            obj: The order payload dict.
            recv_window_ms: Maximum allowed age of the message timestamp in ms
                            before the order is rejected. Default 5000 ms.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        msg_ts_raw = obj.get("timestamp") or obj.get("ts")
        if msg_ts_raw is not None:
            try:
                ts_str = str(msg_ts_raw)
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                msg_dt = datetime.fromisoformat(ts_str)
                if msg_dt.tzinfo is None:
                    msg_dt = msg_dt.replace(tzinfo=timezone.utc)
                now_dt = ntp_clock.now()
                delta_ms = abs((now_dt - msg_dt).total_seconds() * 1000)
                if delta_ms > recv_window_ms:
                    err = (
                        "CRITICAL_LATENCY_REJECTION: msg_ts=%s delta_ms=%.0f "
                        "exceeds recvWindow=%dms — order NOT sent.",
                        msg_ts_raw, delta_ms, recv_window_ms,
                    )
                    if self.logger:
                        self.logger.critical(*err)
                    else:
                        _ntp_logger.critical(*err)
                    return False
            except Exception:
                pass  # unparseable timestamp — proceed with send
        return self.send(obj)

    def health(self) -> NTBridgeHealth:
        return NTBridgeHealth(
            is_connected=self.is_connected,
            last_rx_ts=self.last_rx_ts,
            last_tx_ts=self.last_tx_ts,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "listening": self._listening,
            "connected": self.is_connected,
            "handshake_ok": self._handshake_event.is_set(),
            "remote": f"{self.remote_addr[0]}:{self.remote_addr[1]}" if self.remote_addr else None,
            "local": f"{self.local_addr[0]}:{self.local_addr[1]}" if self.local_addr else None,
            "last_rx_ts": self.last_rx_ts,
            "last_tx_ts": self.last_tx_ts,
            "last_err": self.last_err,
            "client_kind": self._client_kind,
            "client_source": self._client_source,
            "client_first_type": self._client_first_type,
            "client_last_type": self._client_last_type,
        }

    def handshake_ok(self) -> bool:
        return self._handshake_event.is_set()

    def client_kind(self) -> str:
        with self._client_lock:
            return str(self._client_kind or "unknown")

    def client_source(self) -> Optional[str]:
        with self._client_lock:
            return self._client_source

    def hello_ok(self) -> bool:
        return self._hello_event.is_set()

    def snapshot_ok(self) -> bool:
        return self._snapshot_event.is_set()

    def request_snapshot(self, account: Optional[str], instrument: Optional[str], request_id: Optional[str] = None) -> bool:
        base = {
            "protocol_version": 1,
            "timestamp": utc_ts(),
            "client_order_id": request_id or f"SNAPSHOT|{utc_ts()}",
        }
        payload = {
            "type": "SYNC",
            **base,
            "account": account,
            "instrument": instrument,
        }
        return self.send(payload)

    def shutdown(self) -> None:
        self._stop.set()
        self._close_client(reason="shutdown")
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
        self._listening = False
        if self._accept_thread and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=2.0)
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                client, addr = self._server.accept() if self._server is not None else (None, None)
            except socket.timeout:
                continue
            except Exception as exc:
                if self._stop.is_set() or self._server is None:
                    break
                self.last_err = f"accept_error:{exc}"
                self._log("STATUS", {"event": "accept_error", "error": str(exc)})
                continue
            if client is None:
                continue
            if not self._accept_client(client, addr):
                continue

    def _accept_client(self, client: socket.socket, addr: Tuple[str, int]) -> bool:
        """Install a newly accepted client if it is not a duplicate.

        Duplicate rejection must return early so the accept path does not start
        a read loop against the existing healthy socket.
        """
        preview_msgs, preview_remainder = self._preview_client_messages(client)
        if self._should_reject_previewed_client(preview_msgs):
            try:
                client.close()
            except Exception:
                pass
            self._log_info("NT bridge rejecting telemetry-only client; waiting for order-capable NinjaRepoBridge.")
            status_event = "telemetry_client_rejected"
            try:
                if self._client_kind == "order_capable" and self.is_connected:
                    status_event = "telemetry_client_rejected_order_bridge_preserved"
            except Exception:
                status_event = "telemetry_client_rejected"
            self._log_status_snapshot(status_event)
            return False
        if not self._replace_client(client, addr, preview_remainder=preview_remainder):
            return False
        for msg in preview_msgs:
            if not self._handle_rx_message(msg):
                return False
        self._start_read_loop()
        return True

    def _preview_client_messages(self, client: socket.socket) -> Tuple[List[Dict[str, Any]], bytes]:
        """Read the first line before replacing the current client.

        The NT account telemetry client and the order-capable NinjaRepoBridge can
        both connect to the same port. Classifying the first frame before socket
        replacement prevents telemetry from evicting a healthy order bridge.
        """
        original_timeout = None
        try:
            original_timeout = client.gettimeout()
        except Exception:
            pass
        buf = bytearray()
        try:
            client.settimeout(0.35)
            while len(buf) < 8192 and b"\n" not in buf:
                chunk = client.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
        except socket.timeout:
            pass
        except Exception as exc:
            self.last_err = f"preview_recv_error:{exc}"
            self._log("STATUS", {"event": "preview_recv_error", "error": str(exc)})
        finally:
            try:
                client.settimeout(original_timeout)
            except Exception:
                pass
        if not buf:
            return [], b""
        if b"\n" in buf:
            raw, remainder = bytes(buf).split(b"\n", 1)
        else:
            raw, remainder = bytes(buf), b""
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            return [], remainder
        msgs = self._decode_json_messages(line)
        if not msgs:
            self._log("RX", {"raw": line, "error": "json_decode_preview"})
            return [], remainder
        return msgs, remainder

    def _should_reject_previewed_client(self, preview_msgs: List[Dict[str, Any]]) -> bool:
        if not preview_msgs:
            return self._should_reject_duplicate_client()
        kinds = {self._classify_client_kind(msg) for msg in preview_msgs}
        if "order_capable" in kinds:
            return False
        if kinds.issubset({"telemetry_only"}):
            return True
        return self._should_reject_duplicate_client() and kinds.issubset({"telemetry_only", "unknown"})

    def _should_reject_duplicate_client(self) -> bool:
        """Keep a healthy current client instead of hot-swapping on duplicate accepts."""
        with self._client_lock:
            has_client = self._client is not None
            handshake_ok = self._handshake_event.is_set()
            connected = self.is_connected
            last_rx_ts = self.last_rx_ts
            client_kind = self._client_kind
        if not has_client or not handshake_ok or not connected:
            return False
        if client_kind != "order_capable":
            return False
        if last_rx_ts is None:
            return False
        try:
            return (time.time() - float(last_rx_ts)) <= 120.0
        except Exception:
            return False

    def _replace_client(self, client: socket.socket, addr: Tuple[str, int], *, preview_remainder: bytes = b"") -> bool:
        if self._should_reject_duplicate_client():
            try:
                client.close()
            except Exception:
                pass
            self._log_info("NT bridge rejecting duplicate connection; keeping healthy client.")
            self._log_status_snapshot("duplicate_rejected")
            return False
        old_client_to_close = None
        with self._client_lock:
            if self._client is not None:
                self._log_info("NT bridge replacing existing connection.")
                old_client_to_close = self._client
                self._client = None
            self._client = client
            self.is_connected = True
            self._connected_event.set()
            self._handshake_event.clear()
            self._hello_event.clear()
            self._snapshot_event.clear()
            self.hello_seen = False
            self.snapshot_seen = False
            self._client_kind = "unknown"
            self._client_source = None
            self._client_first_type = None
            self._client_last_type = None
            self._rx_buffer = bytearray(preview_remainder or b"")
        if old_client_to_close is not None:
            try:
                old_client_to_close.close()
            except Exception:
                pass
        try:
            client.settimeout(1.0)
        except Exception:
            pass
        try:
            self.remote_addr = client.getpeername()
        except Exception:
            self.remote_addr = addr
        try:
            self.local_addr = client.getsockname()
        except Exception:
            self.local_addr = None
        self._log_info(
            "NT bridge ACCEPT remote=%s local=%s",
            f"{self.remote_addr[0]}:{self.remote_addr[1]}" if self.remote_addr else "unknown",
            f"{self.local_addr[0]}:{self.local_addr[1]}" if self.local_addr else "unknown",
        )
        self._log_status_snapshot("accept")
        return True

    def _start_read_loop(self) -> None:
        self._read_thread = threading.Thread(
            target=self._read_loop,
            name="nt-bridge-read",
            daemon=True,
        )
        self._read_thread.start()

    def _send_handshake_ping(self) -> None:
        ping = {"type": "PING", "protocol_version": 1, "ts": utc_ts()}
        self.send(ping)

    def _wait_for_handshake(self) -> bool:
        if self._handshake_event.wait(timeout=self.handshake_timeout_sec):
            return True
        self._log_info(
            "NT handshake timeout after %.1fs; keeping client open for legacy startup flow.",
            self.handshake_timeout_sec,
        )
        return True

    def _read_loop(self) -> None:
        MAX_RX_BUFFER_SIZE = 1048576
        sock = None
        while not self._stop.is_set():
            try:
                with self._client_lock:
                    sock = self._client
                if sock is None:
                    break
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            except Exception as exc:
                if self._stop.is_set():
                    break
                self.last_err = f"recv_error:{exc}"
                self._log("STATUS", {"event": "recv_error", "error": str(exc)})
                break
            if not chunk:
                self._flush_pending_rx_buffer()
                self._log_info("NT socket closed by peer.")
                break
            if len(self._rx_buffer) + len(chunk) > MAX_RX_BUFFER_SIZE:
                self.last_err = "buffer_overflow"
                self._log("STATUS", {"event": "buffer_overflow", "size": len(self._rx_buffer) + len(chunk)})
                self._close_client(reason="buffer_overflow")
                break
            self._rx_buffer.extend(chunk)
            while b"\n" in self._rx_buffer:
                raw, remainder = self._rx_buffer.split(b"\n", 1)
                self._rx_buffer = bytearray(remainder)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                msgs = self._decode_json_messages(line)
                if not msgs:
                    self._log("RX", {"raw": line, "error": "json_decode"})
                    continue
                for msg in msgs:
                    if not self._handle_rx_message(msg):
                        return
        self._close_client(reason="peer_closed", sock=sock)

    def _decode_json_messages(self, line: str) -> List[Dict[str, Any]]:
        """Decode one or more JSON objects from a single text frame.

        Some legacy NT add-ons coalesce multiple JSON objects into one socket
        write without inserting a newline between them. We keep the newline
        framing as the primary contract, but we also split concatenated JSON
        objects here so a single malformed frame does not tear down the bridge.
        """
        text = line.strip()
        if not text:
            return []
        decoder = json.JSONDecoder()
        messages: List[Dict[str, Any]] = []
        idx = 0
        length = len(text)
        while idx < length:
            while idx < length and text[idx].isspace():
                idx += 1
            if idx >= length:
                break
            try:
                obj, end = decoder.raw_decode(text, idx)
            except Exception:
                break
            if isinstance(obj, dict):
                messages.append(obj)
            else:
                messages.append({"type": "RAW", "value": obj})
            idx = end
        return messages

    def _flush_pending_rx_buffer(self) -> None:
        """Process any buffered bytes that were not newline-terminated.

        Some senders close the socket after writing a JSON payload without a
        trailing newline. We still want to preserve that last frame if it is
        decodable, rather than dropping it on EOF.
        """
        if not self._rx_buffer:
            return
        raw = bytes(self._rx_buffer)
        self._rx_buffer.clear()
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            return
        msgs = self._decode_json_messages(line)
        if not msgs:
            self._log("RX", {"raw": line, "error": "json_decode"})
            return
        for msg in msgs:
            if not self._handle_rx_message(msg):
                return

    def _handle_rx_message(self, msg: Dict[str, Any]) -> bool:
        self.last_rx_ts = time.time()
        if not self._handshake_event.is_set():
            if not self._validate_and_mark_handshake(msg):
                return False
        self._update_client_metadata(msg)
        msg_type = self._message_type(msg)
        if msg_type == "HELLO":
            self.hello_seen = True
            self._hello_event.set()
            self._connected_event.set()
            self._handshake_event.set()
        elif msg_type in {"POSITION_SNAPSHOT", "SNAPSHOT", "POSITION", "ACCOUNT_SNAPSHOT", "STATE_SNAPSHOT"}:
            self.snapshot_seen = True
            self._snapshot_event.set()
            self._connected_event.set()
        self._log("RX", msg)
        for cb in list(self._callbacks):
            try:
                cb(msg)
            except Exception as exc:
                self.last_err = f"callback_error:{exc}"
                if self.logger:
                    self.logger.warning("NT bridge callback failed: %s", exc, exc_info=True)
        return True

    @staticmethod
    def _message_type(msg: Dict[str, Any]) -> str:
        msg_type = ""
        raw_type = msg.get("type")
        if raw_type is None:
            raw_type = msg.get("Type")
        if raw_type is None:
            raw_type = msg.get("msg_type")
        if raw_type is not None:
            try:
                msg_type = str(raw_type).upper()
            except Exception:
                msg_type = ""
        if not msg_type:
            raw_evt = msg.get("event_type")
            if raw_evt is None:
                raw_evt = msg.get("eventType")
            if raw_evt is not None:
                try:
                    msg_type = str(raw_evt).upper()
                except Exception:
                    msg_type = ""
        return msg_type

    @staticmethod
    def _message_source(msg: Dict[str, Any]) -> Optional[str]:
        for key in ("source", "bridge_source", "addon", "platform"):
            val = msg.get(key)
            if val is None:
                continue
            try:
                text = str(val).strip()
            except Exception:
                continue
            if text:
                return text
        return None

    @classmethod
    def _classify_client_kind(cls, msg: Dict[str, Any]) -> str:
        msg_type = cls._message_type(msg)
        source = (cls._message_source(msg) or "").lower()
        platform = str(msg.get("platform") or "").lower()
        if (
            msg_type == "HELLO"
            or msg_type in {"ORDER_ACK", "ORDER_UPDATE", "FILL", "READY"}
            or "ninjarepo" in source
            or "ninjatrader8" in platform
            or "ninjatrader 8" in platform
            or msg.get("addon_version") is not None
        ):
            return "order_capable"
        if source == "nt_account_api" or msg_type == "BRIDGE_STATUS":
            return "telemetry_only"
        if msg_type in {"HEARTBEAT", "PNL_SNAPSHOT"}:
            return "telemetry_only"
        return "unknown"

    def _update_client_metadata(self, msg: Dict[str, Any]) -> None:
        msg_type = self._message_type(msg)
        source = self._message_source(msg)
        kind = self._classify_client_kind(msg)
        with self._client_lock:
            if msg_type:
                if self._client_first_type is None:
                    self._client_first_type = msg_type
                self._client_last_type = msg_type
            if source and self._client_source is None:
                self._client_source = source
            if kind == "order_capable" or self._client_kind in {"", "unknown"}:
                self._client_kind = kind

    def _validate_and_mark_handshake(self, msg: Dict[str, Any]) -> bool:
        msg_type = self._message_type(msg)
        client_kind = self._classify_client_kind(msg)
        self._update_client_metadata(msg)
        if client_kind == "telemetry_only":
            now = time.time()
            if now - self._last_handshake_wait_log_ts >= 5.0:
                self._last_handshake_wait_log_ts = now
                self._log_info("NT HANDSHAKE WAIT (telemetry-only type=%s source=%s)", msg_type or "missing", self._message_source(msg) or "")
            return True
        if msg_type in {"POSITION_SNAPSHOT", "SNAPSHOT", "ACCOUNT_SNAPSHOT", "STATE_SNAPSHOT", "HEARTBEAT", "PNL_SNAPSHOT", "PING", "SYNC", "ORDER", "FLATTEN"}:
            # Legacy add-ons may not emit an explicit HELLO/ACK before they start
            # streaming useful traffic. Treat any valid inbound application frame
            # as proof that the socket contract is alive.
            self._handshake_event.set()
            self._log_info("NT HANDSHAKE OK (legacy type=%s)", msg_type)
            self._log_status_snapshot("handshake_ok")
            return True
        if msg_type in {"POSITION_SNAPSHOT", "SNAPSHOT", "ACCOUNT_SNAPSHOT", "STATE_SNAPSHOT", "HEARTBEAT", "PNL_SNAPSHOT"}:
            # Snapshot/heartbeat traffic can arrive before a formal HELLO/ACK on
            # some adapters. Keep the connection open and wait for a later frame.
            now = time.time()
            if now - self._last_handshake_wait_log_ts >= 5.0:
                self._last_handshake_wait_log_ts = now
                self._log_info("NT HANDSHAKE WAIT (legacy message before HELLO/ACK: %s)", msg_type)
            return True
        if msg_type not in {"HELLO", "ACK"}:
            # Pre-handshake chatter can arrive first on some adapters; wait for HELLO/ACK.
            now = time.time()
            if now - self._last_handshake_wait_log_ts >= 5.0:
                self._last_handshake_wait_log_ts = now
                self._log_info("NT HANDSHAKE WAIT (type=%s)", msg_type or "missing")
            return True

        pv = msg.get("protocol_version")
        if pv is None:
            pv = msg.get("protocolVersion")
        if pv is None:
            pv = msg.get("ProtocolVersion")
        if pv is None:
            pv = msg.get("protocol")
        if pv is None:
            for k, v in msg.items():
                try:
                    kn = str(k).replace("_", "").lower()
                except Exception:
                    continue
                if kn == "protocolversion":
                    pv = v
                    break
        if pv is None:
            # Compatibility fallback: treat HELLO/ACK with omitted protocol as v1.
            self._log_info("NT HANDSHAKE WARN (protocol_version missing; assuming v1)")
            pv = 1
        try:
            pv_int = int(pv)
        except Exception:
            self._log_info("NT HANDSHAKE FAIL (protocol_version invalid)")
            self._close_client(reason="handshake_fail:protocol_version_invalid")
            return False
        if pv_int != 1:
            self._log_info("NT HANDSHAKE FAIL (protocol_version)")
            self._close_client(reason=f"handshake_fail:protocol_version:{pv_int}")
            return False
        self._handshake_event.set()
        self._log_info("NT HANDSHAKE OK")
        self._log_status_snapshot("handshake_ok")
        return True

    def _close_client(self, *, reason: str, sock: Optional[socket.socket] = None) -> None:
        with self._client_lock:
            if sock is not None and self._client is not sock:
                return
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None
            self.is_connected = False
            self._connected_event.clear()
            self._handshake_event.clear()
        if reason:
            self._log_info("NT client closed (%s).", reason)
        self._log_status_snapshot("disconnected")

    def _log_status_snapshot(self, event: str) -> None:
        payload = self.get_status()
        payload["event"] = event
        self._log("STATUS", payload)

    def _log_info(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def _log(self, direction: str, msg: dict, warn: Optional[str] = None) -> None:
        if warn and self.logger:
            self.logger.warning("NT bridge %s: %s", warn, msg)
        msg_out = _normalize_msg_ts(msg)
        record = {
            "ts": utc_ts(),
            "dir": direction,
            "msg": msg_out,
        }
        if self.log_path:
            try:
                with self._log_lock:
                    with open(self.log_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                if self.logger:
                    self.logger.debug("Failed to write NT bridge log", exc_info=True)
