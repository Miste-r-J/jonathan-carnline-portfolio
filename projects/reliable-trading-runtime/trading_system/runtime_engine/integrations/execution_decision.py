from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from .execution_state import ExecutionState


def build_execution_decision(
    *,
    prediction_id: str,
    decision: str,
    reason_code: str,
    reason_detail: Optional[Mapping[str, Any]] = None,
    nt_state: Optional[Mapping[str, Any]] = None,
    lockout_state: Optional[Mapping[str, Any]] = None,
    size_final: int,
    stop_sent: Optional[float],
    target_sent: Optional[float],
    mode: str,
    decision_ts: str,
    tx_ts: Optional[str] = None,
    rx_ts: Optional[str] = None,
    trade_pnl_state: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
    decision_upper = str(decision or "").upper()
    reason_detail_payload = dict(reason_detail or {})
    if decision_upper not in {"SEND", "SENT"} and not reason_detail_payload:
        reason_detail_payload = {
            "reason": "blocked_no_detail",
            "reason_code": str(reason_code),
        }
    return {
        "prediction_id": str(prediction_id),
        "decision": str(decision),
        "reason_code": str(reason_code),
        "reason_detail": reason_detail_payload,
        "nt_state": dict(nt_state or {}),
        "lockout_state": dict(lockout_state or {}),
        "trade_pnl_state": dict(trade_pnl_state or {}),
        "size_final": int(size_final),
        "stop_sent": (float(stop_sent) if stop_sent is not None else None),
        "target_sent": (float(target_sent) if target_sent is not None else None),
        "mode": str(mode),
        "decision_ts": str(decision_ts),
        "tx_ts": (str(tx_ts) if tx_ts is not None else None),
        "rx_ts": (str(rx_ts) if rx_ts is not None else None),
    }


def decide_execution(
    *,
    prediction: Mapping[str, Any],
    exec_state: ExecutionState,
    mode: str,
    decision_ts: Optional[str] = None,
) -> dict[str, Any]:
    side = str(prediction.get("side") or "FLAT").upper()
    size_reco = int(prediction.get("size_reco") or 0)
    stop_abs = prediction.get("stop_abs")
    target_abs = prediction.get("target_abs")
    reason_detail: dict[str, Any] = {}

    decision = "send"
    reason_code = "ok"
    size_final = size_reco
    stop_sent = float(stop_abs) if stop_abs is not None else None
    target_sent = float(target_abs) if target_abs is not None else None

    if exec_state.lockout_active:
        decision = "flatten"
        reason_code = str(exec_state.lockout_reason or "lockout_active")
        size_final = 0
        stop_sent = None
        target_sent = None
        reason_detail = {"lockout_active": True}
    elif exec_state.position_qty != 0 and side in {"LONG", "SHORT"}:
        decision = "block"
        reason_code = "position_conflict"
        size_final = 0
        stop_sent = None
        target_sent = None
        reason_detail = {"position_qty": exec_state.position_qty, "position_side": exec_state.position_side}
    elif side == "FLAT":
        decision = "block"
        reason_code = "model_flat"
        size_final = 0
        stop_sent = None
        target_sent = None

    if decision_ts is None:
        decision_ts = datetime.now(timezone.utc).isoformat()

    return build_execution_decision(
        prediction_id=str(prediction.get("prediction_id") or ""),
        decision=decision,
        reason_code=reason_code,
        reason_detail=reason_detail,
        nt_state={
            "connected": exec_state.nt_connected,
            "handshake_ok": exec_state.handshake_ok,
            "nt_ready": exec_state.nt_ready,
            "snapshot_age_sec": exec_state.snapshot_age_sec,
            "snapshot_ts": exec_state.last_snapshot_ts,
        },
        lockout_state={"hard_lockout": exec_state.lockout_active, "reason": exec_state.lockout_reason},
        size_final=size_final,
        stop_sent=stop_sent,
        target_sent=target_sent,
        mode=str(mode),
        decision_ts=str(decision_ts),
        tx_ts=None,
        rx_ts=None,
    )
