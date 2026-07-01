from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


GuardRailAction = str


@dataclass(frozen=True)
class GuardRailReason:
    code: str
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = dict(self.detail)
        return payload


@dataclass(frozen=True)
class GuardRailConfig:
    price_age_warn_sec: float = 0.0
    price_age_max_sec: float = 10.0
    snapshot_age_max_sec: float = 60.0
    bar_age_max_sec: float = 90.0
    bar_age_max_sec_configured: float = 90.0
    bar_interval_sec: float = 0.0
    effective_bar_age_max_sec: float = 90.0
    protection_timeout_sec: float = 5.0
    close_emit_timeout_sec: float = 30.0
    reconciliation_grace_bars: int = 3
    cancel_loop_max_updates: int = 3
    max_exit_pending_sec: float = 15.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "price_age_warn_sec": float(self.price_age_warn_sec),
            "price_age_max_sec": float(self.price_age_max_sec),
            "snapshot_age_max_sec": float(self.snapshot_age_max_sec),
            "bar_age_max_sec": float(self.bar_age_max_sec),
            "bar_age_max_sec_configured": float(self.bar_age_max_sec_configured),
            "bar_interval_sec": float(self.bar_interval_sec),
            "effective_bar_age_max_sec": float(self.effective_bar_age_max_sec),
            "protection_timeout_sec": float(self.protection_timeout_sec),
            "close_emit_timeout_sec": float(self.close_emit_timeout_sec),
            "reconciliation_grace_bars": int(self.reconciliation_grace_bars),
            "cancel_loop_max_updates": int(self.cancel_loop_max_updates),
            "max_exit_pending_sec": float(self.max_exit_pending_sec),
        }


@dataclass(frozen=True)
class GuardRailState:
    allowed_to_arm: bool
    allowed_to_emit_entries: bool
    required_action: GuardRailAction
    reasons: Sequence[GuardRailReason] = field(default_factory=list)
    lockout_code: Optional[str] = None
    preflight_ok: bool = True

    def reason_dicts(self) -> List[Dict[str, Any]]:
        return [reason.as_dict() for reason in self.reasons]

    def summary(self) -> Dict[str, Any]:
        return {
            "allowed_to_arm": bool(self.allowed_to_arm),
            "allowed_to_emit_entries": bool(self.allowed_to_emit_entries),
            "required_action": self.required_action,
            "lockout_code": self.lockout_code,
            "preflight_ok": bool(self.preflight_ok),
            "reasons": self.reason_dicts(),
        }


def _reason(code: str, message: str, **detail: Any) -> GuardRailReason:
    return GuardRailReason(code=code, message=message, detail={k: v for k, v in detail.items() if v is not None})


def _bool(ctx: Mapping[str, Any], key: str, default: bool = False) -> bool:
    try:
        return bool(ctx.get(key, default))
    except Exception:
        return default


def _float(ctx: Mapping[str, Any], key: str) -> Optional[float]:
    val = ctx.get(key)
    if val is None:
        return None
    try:
        num = float(val)
    except Exception:
        return None
    if num != num:  # NaN check without importing math
        return None
    return num


def compute_effective_bar_age_max_sec(bar_interval_sec: Optional[float]) -> Optional[float]:
    if bar_interval_sec is None:
        return None
    try:
        interval = float(bar_interval_sec)
    except Exception:
        return None
    if interval <= 0:
        return None
    # Require at least two bars plus a small grace, bounded by a sane floor.
    return max(2.0 * interval + 5.0, 30.0)


PROTECTION_WORKING_STATES = {
    "WORKING",
    "ACCEPTED",
    "SUBMITTED",
    "PARTFILLED",
    "PENDINGCHANGE",
    "EXITS_WORKING",
    "EXITS_SUBMITTED",
    "STOP_WORKING",
}


def _order_state(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    txt = str(val).strip().upper()
    return txt or None


def is_protection_working(stop_state: Optional[str], target_state: Optional[str]) -> bool:
    stop = _order_state(stop_state)
    target = _order_state(target_state)
    if stop is None or target is None:
        return False
    return stop in PROTECTION_WORKING_STATES and target in PROTECTION_WORKING_STATES


def is_stop_working(stop_state: Optional[str]) -> bool:
    stop = _order_state(stop_state)
    if stop is None:
        return False
    return stop in PROTECTION_WORKING_STATES


@dataclass
class CloseWatch:
    correlation_id: str
    intent_id: str
    bar_ts: Optional[str]
    started_ts: float
    last_progress_ts: float
    lifecycle_seen: bool = False
    resolved: bool = False
    blocked_reason: Optional[str] = None
    last_status: Optional[str] = None
    last_status_ts: Optional[float] = None


class CloseWatchdog:
    def __init__(self) -> None:
        self._pending: Dict[str, CloseWatch] = {}

    def _prune_resolved(self) -> None:
        for key, watch in list(self._pending.items()):
            if bool(getattr(watch, "resolved", False)):
                self._pending.pop(key, None)

    def register_intent(
        self,
        *,
        correlation_id: str,
        intent_id: str,
        bar_ts: Optional[str],
        decision: str,
        reason_code: str,
        now: float,
    ) -> Optional[Dict[str, Any]]:
        if decision in {"SKIPPED_NOOP", "SKIPPED_IDEMPOTENT"}:
            return {
                "correlation_id": correlation_id,
                "intent_id": intent_id,
                "decision": decision,
                "reason_code": reason_code,
                "bar_ts": bar_ts,
            }
        # Only SENT intents should be tracked as pending close watches.
        # Non-sent outcomes are terminal at registration time and should not
        # appear in pending_watches() where they can be misinterpreted later.
        if decision != "SENT":
            return {
                "correlation_id": str(correlation_id),
                "intent_id": str(intent_id),
                "decision": decision,
                "reason_code": reason_code,
                "bar_ts": bar_ts,
            }
        watch = CloseWatch(
            correlation_id=str(correlation_id),
            intent_id=str(intent_id),
            bar_ts=bar_ts,
            started_ts=float(now),
            last_progress_ts=float(now),
        )
        self._pending[watch.correlation_id] = watch
        return None

    def record_order_event(self, *, correlation_id: str, status: str, now: float) -> Dict[str, Any]:
        watch = self._pending.get(str(correlation_id))
        if watch is None:
            return {"known": False, "resolved": False}
        status_upper = _order_state(status) or ""
        progress_statuses = {
            "SUBMITTED",
            "ACCEPTED",
            "WORKING",
            "PARTFILLED",
            "EXITS_SUBMITTED",
            "EXITS_WORKING",
            "STOP_WORKING",
            "CANCELPENDING",
            "CANCELLED",
            "REPLACED",
            "UPDATED",
            "SENT",
        }
        blocked_statuses = {"BLOCKED", "REJECTED", "LOCKOUT", "ERROR"}
        terminal_statuses = {"FILLED", "FLATTENED", "CLOSED", "ALREADY_FLAT"}
        is_progress = status_upper in progress_statuses
        is_blocked = status_upper in blocked_statuses
        is_terminal = status_upper in terminal_statuses
        if is_progress or is_blocked or is_terminal:
            watch.lifecycle_seen = True
            watch.last_progress_ts = float(now)
        watch.last_status = status_upper
        watch.last_status_ts = float(now)
        if is_blocked or is_terminal:
            watch.resolved = True
            self._pending.pop(watch.correlation_id, None)
        return {
            "known": True,
            "resolved": bool(watch.resolved),
            "lifecycle_seen": bool(watch.lifecycle_seen),
            "last_status": watch.last_status,
            "last_status_ts": watch.last_status_ts,
        }

    def check_timeouts(
        self,
        *,
        now: float,
        close_emit_timeout_sec: float,
        max_exit_pending_sec: float,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        silent_drop: Optional[Dict[str, Any]] = None
        exit_pending: Optional[Dict[str, Any]] = None
        for key, watch in list(self._pending.items()):
            if watch.resolved:
                self._pending.pop(key, None)
                continue
            age = float(now) - float(watch.started_ts)
            progress_age = float(now) - float(watch.last_progress_ts)
            if not watch.lifecycle_seen and age > float(close_emit_timeout_sec):
                silent_drop = {
                    "correlation_id": watch.correlation_id,
                    "intent_id": watch.intent_id,
                    "age_sec": age,
                    "progress_age_sec": progress_age,
                    "close_emit_timeout_sec": float(close_emit_timeout_sec),
                    "bar_ts": watch.bar_ts,
                }
                watch.resolved = True
                self._pending.pop(key, None)
                break
            if watch.lifecycle_seen and progress_age > float(close_emit_timeout_sec):
                silent_drop = {
                    "correlation_id": watch.correlation_id,
                    "intent_id": watch.intent_id,
                    "age_sec": age,
                    "progress_age_sec": progress_age,
                    "close_emit_timeout_sec": float(close_emit_timeout_sec),
                    "last_status": watch.last_status,
                    "last_status_ts": watch.last_status_ts,
                    "bar_ts": watch.bar_ts,
                }
                watch.resolved = True
                self._pending.pop(key, None)
                break
            if watch.lifecycle_seen and age > float(max_exit_pending_sec):
                exit_pending = {
                    "correlation_id": watch.correlation_id,
                    "intent_id": watch.intent_id,
                    "age_sec": age,
                    "max_exit_pending_sec": float(max_exit_pending_sec),
                    "last_status": watch.last_status,
                    "last_status_ts": watch.last_status_ts,
                    "bar_ts": watch.bar_ts,
                }
                watch.resolved = True
                self._pending.pop(key, None)
                break
        return {"silent_drop": silent_drop, "exit_pending": exit_pending}

    def pending_watches(self) -> List[CloseWatch]:
        self._prune_resolved()
        return list(self._pending.values())


class CancelLoopDetector:
    def __init__(self) -> None:
        self._counts: Dict[str, Dict[str, Any]] = {}

    def record(
        self,
        *,
        correlation_id: str,
        status: str,
        now: float,
        is_exit_or_protection: bool,
        max_updates: int,
    ) -> Optional[Dict[str, Any]]:
        if not is_exit_or_protection:
            return None
        status_upper = str(status or "").upper()
        if status_upper not in {"CANCELPENDING", "CANCELLED"}:
            return None
        key = str(correlation_id)
        rec = self._counts.setdefault(key, {"count": 0, "last_status": None, "last_ts": None})
        rec["count"] = int(rec.get("count", 0) or 0) + 1
        rec["last_status"] = status_upper
        rec["last_ts"] = float(now)
        if int(rec["count"]) > int(max_updates):
            return {
                "correlation_id": key,
                "count": int(rec["count"]),
                "max_updates": int(max_updates),
                "last_status": status_upper,
                "last_ts": float(now),
            }
        return None


def evaluate_guardrails(ctx: Mapping[str, Any], cfg: GuardRailConfig, *, preflight: bool) -> GuardRailState:
    reasons: List[GuardRailReason] = []
    warn_codes = {"price_age_warn"}

    hard_lockout_active = _bool(ctx, "hard_lockout_active")
    lockout_code = ctx.get("hard_lockout_code")
    lockout_sticky = _bool(ctx, "lockout_sticky")
    if hard_lockout_active:
        reasons.append(
            _reason(
                "hard_lockout_active",
                "Hard lockout is active; explicit reset required.",
                hard_lockout_code=lockout_code,
                lockout_sticky=lockout_sticky,
            )
        )

    enforce_nt_ready = _bool(ctx, "enforce_nt_ready_checks", True)
    enforce_snapshot = _bool(ctx, "enforce_snapshot_checks", True)
    enforce_snapshot_price = _bool(ctx, "enforce_snapshot_price_checks", enforce_snapshot)
    enforce_bar = _bool(ctx, "enforce_bar_checks", True)
    enforce_blocking = _bool(ctx, "enforce_blocking_orders_checks", enforce_snapshot)
    bar_interval_sec = _float(ctx, "bar_interval_sec")
    if bar_interval_sec is None:
        bar_interval_sec = float(getattr(cfg, "bar_interval_sec", 0.0) or 0.0)
    bar_interval_known = _bool(ctx, "bar_interval_known", bar_interval_sec > 0)
    effective_bar_age_max_sec = _float(ctx, "effective_bar_age_max_sec")
    if effective_bar_age_max_sec is None:
        effective_bar_age_max_sec = float(
            getattr(cfg, "effective_bar_age_max_sec", None) or getattr(cfg, "bar_age_max_sec", 0.0) or 0.0
        )

    nt_connected = _bool(ctx, "nt_connected")
    handshake_ok = _bool(ctx, "handshake_ok")
    nt_ready = _bool(ctx, "nt_ready")
    if enforce_nt_ready:
        if not nt_connected:
            reasons.append(_reason("nt_not_connected", "NT bridge is not connected.", nt_connected=nt_connected))
        if nt_connected and not handshake_ok:
            reasons.append(_reason("nt_handshake_not_ok", "NT handshake is not OK.", handshake_ok=handshake_ok))
        if nt_connected and handshake_ok and not nt_ready:
            reasons.append(_reason("nt_not_ready", "NT is not ready.", nt_ready=nt_ready))

    snapshot_has_price = _bool(ctx, "snapshot_has_price")
    last_price = ctx.get("last_price")
    price_age_sec = _float(ctx, "price_age_sec")
    if enforce_snapshot and enforce_snapshot_price:
        if not snapshot_has_price or last_price is None:
            reasons.append(
                _reason(
                    "snapshot_price_missing",
                    "Snapshot price is missing.",
                    snapshot_has_price=snapshot_has_price,
                    last_price=last_price,
                )
            )
        if snapshot_has_price and price_age_sec is None:
            reasons.append(_reason("price_age_missing", "Price age is missing.", price_age_sec=price_age_sec))
    if price_age_sec is not None and float(getattr(cfg, "price_age_warn_sec", 0.0) or 0.0) > 0:
        warn_threshold = float(getattr(cfg, "price_age_warn_sec", 0.0) or 0.0)
        if price_age_sec > warn_threshold and price_age_sec <= float(cfg.price_age_max_sec):
            reasons.append(
                _reason(
                    "price_age_warn",
                    "Price age exceeds warning threshold.",
                    price_age_sec=price_age_sec,
                    price_age_warn_sec=warn_threshold,
                )
            )
    if price_age_sec is not None and price_age_sec > float(cfg.price_age_max_sec):
        reasons.append(
            _reason(
                "price_age_stale",
                "Price age exceeds threshold.",
                price_age_sec=price_age_sec,
                price_age_max_sec=cfg.price_age_max_sec,
            )
        )

    snapshot_age_sec = _float(ctx, "snapshot_age_sec")
    if enforce_snapshot:
        if snapshot_age_sec is None:
            reasons.append(_reason("snapshot_age_missing", "Snapshot age is missing.", snapshot_age_sec=snapshot_age_sec))
        elif snapshot_age_sec > float(cfg.snapshot_age_max_sec):
            reasons.append(
                _reason(
                    "snapshot_age_stale",
                    "Snapshot age exceeds threshold.",
                    snapshot_age_sec=snapshot_age_sec,
                    snapshot_age_max_sec=cfg.snapshot_age_max_sec,
                )
            )

    last_bar_ts = ctx.get("last_bar_ts")
    bar_age_sec = _float(ctx, "bar_age_sec")
    if enforce_bar:
        if not bar_interval_known or effective_bar_age_max_sec <= 0:
            reasons.append(
                _reason(
                    "no_bar_interval",
                    "Bar interval is unknown; preflight must fail fast.",
                    bar_interval_sec=bar_interval_sec,
                    effective_bar_age_max_sec=effective_bar_age_max_sec,
                )
            )
        elif last_bar_ts is None:
            if not preflight:
                reasons.append(_reason("bar_ts_missing", "No bar_ts available for freshness checks."))
        elif bar_age_sec is None:
            if not preflight:
                reasons.append(_reason("bar_age_missing", "Bar age is missing.", last_bar_ts=last_bar_ts))
        elif bar_age_sec > float(effective_bar_age_max_sec):
            reasons.append(
                _reason(
                    "bar_age_stale",
                    "Bar age exceeds threshold.",
                    last_bar_ts=last_bar_ts,
                    bar_age_sec=bar_age_sec,
                    bar_age_max_sec=effective_bar_age_max_sec,
                    bar_interval_sec=bar_interval_sec,
                )
            )

    blocking_orders = ctx.get("snapshot_blocking_orders_count")
    try:
        blocking_orders_int = int(blocking_orders or 0)
    except Exception:
        blocking_orders_int = 0
    if enforce_blocking and blocking_orders_int > 0:
        reasons.append(
            _reason(
                "blocking_orders_present",
                "Snapshot reports working orders that block arming.",
                snapshot_blocking_orders_count=blocking_orders_int,
                snapshot_blocking_orders_sample=ctx.get("snapshot_blocking_orders_sample"),
            )
        )

    cleanup_requested = _bool(ctx, "cleanup_requested")
    armed = _bool(ctx, "armed")
    position_state = str(ctx.get("position_state") or "UNKNOWN")
    if cleanup_requested and armed:
        reasons.append(
            _reason(
                "cleanup_while_armed",
                "Preflight cleanup attempted while armed.",
                armed=armed,
                position_state=position_state,
            )
        )
    if cleanup_requested and position_state not in {"FLAT", "UNKNOWN"}:
        reasons.append(
            _reason(
                "cleanup_with_position",
                "Preflight cleanup attempted with a non-flat position.",
                armed=armed,
                position_state=position_state,
            )
        )

    protection_timeout = ctx.get("protection_timeout_detail")
    if protection_timeout:
        reasons.append(
            _reason(
                "nt_protection_timeout",
                "Protection did not become working within the SLA.",
                protection_timeout_detail=protection_timeout,
            )
        )

    silent_drop = ctx.get("silent_drop_detail")
    if silent_drop:
        reasons.append(
            _reason(
                "silent_drop_detected",
                "Close/flatten intent produced no lifecycle or block record within SLA.",
                silent_drop_detail=silent_drop,
            )
        )

    cancel_loop = ctx.get("cancel_loop_detail")
    if cancel_loop:
        reasons.append(
            _reason(
                "cancel_loop",
                "Cancel loop detected on exit/protection orders.",
                cancel_loop_detail=cancel_loop,
            )
        )

    exit_stuck = ctx.get("exit_stuck_detail")
    if exit_stuck:
        reasons.append(
            _reason(
                "exit_stuck",
                "Position remained EXITING beyond max_exit_pending_sec.",
                exit_stuck_detail=exit_stuck,
            )
        )

    reporting_mismatch = ctx.get("reporting_mismatch_detail")
    reporting_mismatch_disarm = ctx.get("reporting_mismatch_disarm_detail")
    if reporting_mismatch and _bool(reporting_mismatch, "disarm_only"):
        reasons.append(
            _reason(
                "reporting_mismatch_disarm",
                "Reporting mismatch detected; halting entries until reconciled.",
                reporting_mismatch_detail=reporting_mismatch,
            )
        )
        reporting_mismatch = None
    elif reporting_mismatch:
        lockout_code = str(reporting_mismatch.get("lockout_code") or "reporting_mismatch")
        reasons.append(
            _reason(
                lockout_code,
                "Reporting mismatch detected between stream state and fills.",
                reporting_mismatch_detail=reporting_mismatch,
            )
        )
    elif reporting_mismatch_disarm:
        reasons.append(
            _reason(
                "reporting_mismatch_disarm",
                "Reporting mismatch detected; halting entries until reconciled.",
                reporting_mismatch_detail=reporting_mismatch_disarm,
            )
        )

    fatal_codes = {
        "hard_lockout_active",
        "nt_protection_timeout",
        "silent_drop_detected",
        "cancel_loop",
        "exit_stuck",
    }
    reporting_lockout_code = None
    if isinstance(reporting_mismatch, Mapping):
        reporting_lockout_code = reporting_mismatch.get("lockout_code")
    if reporting_lockout_code:
        fatal_codes.add(str(reporting_lockout_code))
    effective_reasons = [r for r in reasons if r.code not in warn_codes]
    has_fatal = any(r.code in fatal_codes for r in effective_reasons)

    # Live trading must fail fast on readiness/staleness issues even after arming.
    live_disarm_codes = {
        "nt_not_connected",
        "nt_handshake_not_ok",
        "nt_not_ready",
        "snapshot_price_missing",
        "price_age_missing",
        "price_age_stale",
        "snapshot_age_missing",
        "snapshot_age_stale",
        "bar_ts_missing",
        "bar_age_missing",
        "bar_age_stale",
        "no_bar_interval",
        "blocking_orders_present",
        "cleanup_with_position",
        "reporting_mismatch_disarm",
    }
    has_live_disarm = (not preflight) and any(r.code in live_disarm_codes for r in effective_reasons)

    if has_live_disarm and not has_fatal:
        return GuardRailState(
            allowed_to_arm=False,
            allowed_to_emit_entries=False,
            required_action="DISARM",
            reasons=reasons,
            lockout_code=None,
            preflight_ok=False,
        )

    if has_fatal:
        fatal_reason = next((r for r in reasons if r.code in fatal_codes), reasons[0])
        lockout_code = fatal_reason.code if fatal_reason.code != "hard_lockout_active" else str(lockout_code or "hard_lockout_active")
        return GuardRailState(
            allowed_to_arm=False,
            allowed_to_emit_entries=False,
            required_action="LOCKOUT",
            reasons=reasons,
            lockout_code=lockout_code,
            preflight_ok=False,
        )

    preflight_fail = bool(effective_reasons)
    if preflight_fail and preflight:
        return GuardRailState(
            allowed_to_arm=False,
            allowed_to_emit_entries=False,
            required_action="DISARM",
            reasons=reasons,
            lockout_code=None,
            preflight_ok=False,
        )

    allowed = not preflight_fail
    return GuardRailState(
        allowed_to_arm=allowed,
        allowed_to_emit_entries=allowed,
        required_action="NONE" if allowed else "DISARM",
        reasons=reasons,
        lockout_code=None,
        preflight_ok=allowed,
    )
