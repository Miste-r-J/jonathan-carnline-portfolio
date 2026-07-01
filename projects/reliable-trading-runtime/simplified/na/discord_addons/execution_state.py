from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExecutionState:
    position_qty: int = 0
    position_side: str = "FLAT"
    nt_connected: bool = False
    handshake_ok: bool = False
    nt_ready: bool = False
    snapshot_age_sec: Optional[float] = None
    last_snapshot_ts: Optional[str] = None
    lockout_active: bool = False
    lockout_reason: Optional[str] = None
    working_orders: Dict[str, Any] = field(default_factory=dict)
    fills: Dict[str, Any] = field(default_factory=dict)
    guardrail_state: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def update_position(self, qty: int, side: str) -> None:
        self.position_qty = int(qty)
        self.position_side = str(side)

    def update_nt(
        self,
        *,
        connected: bool,
        handshake_ok: bool,
        ready: bool,
        snapshot_age_sec: Optional[float],
        snapshot_ts: Optional[str] = None,
    ) -> None:
        self.nt_connected = bool(connected)
        self.handshake_ok = bool(handshake_ok)
        self.nt_ready = bool(ready)
        self.snapshot_age_sec = snapshot_age_sec
        if snapshot_ts is not None:
            self.last_snapshot_ts = snapshot_ts

    def update_lockout(self, *, active: bool, reason: Optional[str]) -> None:
        self.lockout_active = bool(active)
        self.lockout_reason = reason
