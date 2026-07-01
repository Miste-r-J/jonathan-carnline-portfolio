import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Callable, Dict, List, Optional


def _utc_ts() -> str:
    return datetime.now(ZoneInfo("America/Denver")).isoformat()


class MockNTTransport:
    """In-memory NT transport that emits READY/ACK/FILL for replay and CI."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        account: str,
        instruments: Optional[List[str]] = None,
        protocol_version: int = 1,
        emit_ready: bool = True,
        lockout: bool = False,
        lockout_reason: str = "",
    ) -> None:
        self.host = host
        self.port = port
        self.account = account
        self.instruments = instruments or []
        self.protocol_version = int(protocol_version)
        self.emit_ready = bool(emit_ready)
        self.lockout = bool(lockout)
        self.lockout_reason = str(lockout_reason or "")
        self.is_connected = False
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._handshake_ok = False

    def add_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        if cb not in self._callbacks:
            self._callbacks.append(cb)

    def start(self) -> None:
        self.is_connected = True
        self._handshake_ok = True
        if self.emit_ready:
            self._emit(
                {
                    "type": "READY",
                    "protocol_version": self.protocol_version,
                    "account": self.account,
                    "instruments": self.instruments,
                    "snapshot_ts": _utc_ts(),
                    "lockout": self.lockout,
                    "lockout_reason": self.lockout_reason,
                }
            )
            if self.instruments:
                # Provide a baseline snapshot so readiness checks can pass in mock mode.
                self._emit(
                    {
                        "type": "POSITION_SNAPSHOT",
                        "protocol_version": self.protocol_version,
                        "instrument": self.instruments[0],
                        "account": self.account,
                        "qty": 0,
                        "pos_qty": 0,
                        "side": "FLAT",
                        "avg_price": 0,
                        "orders": [],
                        "positions": [],
                        "timestamp": _utc_ts(),
                    }
                )

    def shutdown(self) -> None:
        self.is_connected = False
        self._handshake_ok = False

    def handshake_ok(self) -> bool:
        return bool(self._handshake_ok)

    def send(self, payload: Dict[str, Any]) -> bool:
        msg_type = str(payload.get("type") or "").upper()
        if msg_type == "ORDER":
            cid = payload.get("client_order_id") or payload.get("clientOrderId")
            correlation_id = payload.get("correlation_id") or cid
            signal_id = payload.get("signal_id")
            if self.lockout:
                self._emit(
                    {
                        "type": "ORDER_ACK",
                        "protocol_version": self.protocol_version,
                        "client_order_id": cid,
                        "correlation_id": correlation_id,
                        "signal_id": signal_id,
                        "status": "LOCKOUT",
                        "reason": self.lockout_reason,
                        "timestamp": _utc_ts(),
                    }
                )
                self._emit(
                    {
                        "type": "ERROR",
                        "protocol_version": self.protocol_version,
                        "client_order_id": cid,
                        "correlation_id": correlation_id,
                        "signal_id": signal_id,
                        "status": "LOCKOUT",
                        "reason": self.lockout_reason,
                        "timestamp": _utc_ts(),
                    }
                )
                return True
            self._emit(
                {
                    "type": "ORDER_ACK",
                    "protocol_version": self.protocol_version,
                    "client_order_id": cid,
                    "correlation_id": correlation_id,
                    "signal_id": signal_id,
                    "status": "ACK",
                    "timestamp": _utc_ts(),
                }
            )
            self._emit(
                {
                    "type": "FILL",
                    "protocol_version": self.protocol_version,
                    "client_order_id": cid,
                    "correlation_id": correlation_id,
                    "signal_id": signal_id,
                    "fill_price": payload.get("price") or payload.get("limit_price"),
                    "qty": payload.get("qty"),
                    "timestamp": _utc_ts(),
                }
            )
            self._emit(
                {
                    "type": "ORDER_UPDATE",
                    "protocol_version": self.protocol_version,
                    "client_order_id": cid,
                    "correlation_id": correlation_id,
                    "signal_id": signal_id,
                    "status": "EXITS_SUBMITTED",
                    "timestamp": _utc_ts(),
                }
            )
        elif msg_type == "FLATTEN":
            cid = payload.get("client_order_id") or payload.get("clientOrderId")
            correlation_id = payload.get("correlation_id") or cid
            signal_id = payload.get("signal_id")
            self._emit(
                {
                    "type": "ORDER_ACK",
                    "protocol_version": self.protocol_version,
                    "client_order_id": cid,
                    "correlation_id": correlation_id,
                    "signal_id": signal_id,
                    "status": "ACK",
                    "timestamp": _utc_ts(),
                }
            )
        return True

    def _emit(self, msg: Dict[str, Any]) -> None:
        for cb in list(self._callbacks):
            try:
                cb(msg)
            except Exception:
                pass
