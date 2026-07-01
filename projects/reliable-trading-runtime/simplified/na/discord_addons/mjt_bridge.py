from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class MJTBridgeHealth:
    is_connected: bool
    last_rx_ts: Optional[float]
    last_tx_ts: Optional[float]


class MJTBridgeClient:
    def __init__(
        self,
        host: str,
        port: int,
        logger: Optional[Any] = None,
        log_path: Optional[str] = None,
        require_newline_framing: bool = True,
        reconnect_backoff_sec: float = 1.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.logger = logger
        self.log_path = log_path
        self.require_newline_framing = bool(require_newline_framing)
        self.reconnect_backoff_sec = float(reconnect_backoff_sec or 1.0)

        self._socket: Optional[socket.socket] = None
        self._sock_lock = threading.Lock()
        self._reader: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._callbacks: List[Callable[[dict], None]] = []
        self._rx_buffer: bytearray = bytearray()

        self.is_connected = False
        self.last_rx_ts: Optional[float] = None
        self.last_tx_ts: Optional[float] = None
        self.last_err: Optional[str] = None
        self.remote_addr: Optional[Tuple[str, int]] = None

    def add_callback(self, cb: Callable[[dict], None]) -> None:
        self._callbacks.append(cb)

    def start(self) -> None:
        if self._reader is not None:
            return
        self._reader = threading.Thread(target=self._read_loop, name="mjt-bridge-read", daemon=True)
        self._reader.start()
        if self.logger is not None:
            try:
                self.logger.info("MJT bridge reader thread started (%s:%s)", self.host, self.port)
            except Exception:
                pass

    def shutdown(self) -> None:
        self._stop.set()
        with self._sock_lock:
            if self._socket:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None
        if self._reader and self._reader.is_alive():
            self._reader.join(timeout=2.0)

    def send(self, obj: dict) -> bool:
        payload = json.dumps(obj, ensure_ascii=True)
        data = (payload + "\n").encode("utf-8")
        with self._sock_lock:
            if not self._socket:
                self.last_err = "send_no_socket"
                return False
            try:
                self._socket.sendall(data)
            except Exception as exc:
                self.last_err = f"send_failed:{exc}"
                self._socket = None
                self.is_connected = False
                return False
        self.last_tx_ts = time.time()
        return True

    def health(self) -> MJTBridgeHealth:
        return MJTBridgeHealth(
            is_connected=self.is_connected,
            last_rx_ts=self.last_rx_ts,
            last_tx_ts=self.last_tx_ts,
        )

    def handshake_ok(self) -> bool:
        return self.is_connected

    def _connect(self) -> None:
        with self._sock_lock:
            if self._socket:
                return
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((self.host, self.port))
            s.settimeout(1.0)
            self._socket = s
            self.is_connected = True
            try:
                self.remote_addr = s.getpeername()
            except Exception:
                self.remote_addr = None
        if self.logger is not None:
            try:
                self.logger.info("MJT bridge connected to %s:%s", self.host, self.port)
            except Exception:
                pass

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            if not self._socket:
                try:
                    self._connect()
                except Exception:
                    time.sleep(self.reconnect_backoff_sec)
                    continue
            try:
                chunk = self._socket.recv(4096)
                if not chunk:
                    raise ConnectionError("socket closed")
            except socket.timeout:
                # A read timeout is expected during quiet periods. Do not
                # tear down the connection; just keep polling.
                continue
            except Exception as exc:
                self.last_err = f"recv_failed:{exc}"
                if self.logger is not None:
                    try:
                        self.logger.warning("MJT recv failed; reconnecting: %s", exc)
                    except Exception:
                        pass
                with self._sock_lock:
                    if self._socket:
                        try:
                            self._socket.close()
                        except Exception:
                            pass
                        self._socket = None
                self.is_connected = False
                time.sleep(self.reconnect_backoff_sec)
                continue
            self._rx_buffer.extend(chunk)
            while b"\n" in self._rx_buffer:
                line, _, rest = self._rx_buffer.partition(b"\n")
                self._rx_buffer = bytearray(rest)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                self.last_rx_ts = time.time()
                for cb in list(self._callbacks):
                    try:
                        cb(msg)
                    except Exception:
                        continue
