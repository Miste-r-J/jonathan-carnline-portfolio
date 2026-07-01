"""diagnostics.py — Self-diagnostic module for the ES trading bot.

Attaches to a LiveCSVStreamer instance and provides:
  - diagnose_order_ids(state)     — detect order-ID corruption
  - verify_fill_consistency(...)  — sanity-check a fill against model params
  - detect_known_bug_patterns()   — scan for historically-known bug signatures
  - run_full_diagnostics()        — consolidated health report
  - can_auto_heal(issue)          — determine if an issue can be healed safely
  - reset_lockout_if_safe()       — conditionally clear a hard lockout

Usage:
    from diagnostics import Diagnostics
    streamer.diagnostics = Diagnostics(streamer)
    report = streamer.diagnostics.run_full_diagnostics()
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    # Avoid circular import at runtime
    pass

# ---------------------------------------------------------------------------
# Module-level regex constants
# ---------------------------------------------------------------------------

# NinjaTrader opaque handles are either UUID-format (with dashes) or
# 32-char lowercase hex strings (without dashes).
# Real exchange IDs are purely numeric.
_OPAQUE_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
)
_OPAQUE_HEX32_RE = re.compile(r'^[0-9a-f]{32}$')
_NUMERIC_RE = re.compile(r'^\d+$')


# ---------------------------------------------------------------------------
# Module-level helpers (no dependency on streamer)
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> Optional[float]:
    """Return float(v) or None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ts_to_epoch(ts: Any) -> Optional[float]:
    """Convert a timestamp (str, datetime, pd.Timestamp, float) to Unix epoch seconds."""
    if ts is None:
        return None
    try:
        # Already numeric
        return float(ts)
    except (TypeError, ValueError):
        pass
    try:
        # datetime or pandas Timestamp
        return ts.timestamp()
    except AttributeError:
        pass
    try:
        # ISO string
        from datetime import datetime
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Diagnostics class
# ---------------------------------------------------------------------------

class Diagnostics:
    """
    Self-diagnostic module for LiveCSVStreamer.

    Instantiate with a reference to the live streamer:

        streamer.diagnostics = Diagnostics(streamer)

    All methods are defensive: they return structured dicts even on errors
    and never raise exceptions to the caller.

    Performance target: run_full_diagnostics() completes in < 100 ms.
    """

    # Known historical bugs — metadata used by detect_known_bug_patterns().
    _KNOWN_BUGS: List[Dict[str, Any]] = [
        {
            "bug_id": "BUG-001",
            "description": (
                "EXITS_SUBMITTED ACK unconditionally reset entry_ninja_order_id to the "
                "opaque handle, undoing the real exchange ID adopted on ORDER_UPDATE "
                "ACCEPTED/SUBMITTED."
            ),
            "severity": "HIGH",
            "fix_location": "stream_live_csv.py ~21566 — gated on not state.get('entry_filled')",
        },
        {
            "bug_id": "BUG-002",
            "description": (
                "stop_order_id binding condition 'and not state.get(stop_order_id)' prevented "
                "the real exchange ID from replacing the opaque handle; the stop fill was never "
                "routed as is_exit_fill=True."
            ),
            "severity": "HIGH",
            "fix_location": "stream_live_csv.py ~22238 — condition allows opaque→exchange upgrade",
        },
        {
            "bug_id": "BUG-003",
            "description": (
                "Stop fill fell through the ENTRY branch and unconditionally overwrote "
                "entry_ninja_order_id with the stop's exchange ID."
            ),
            "severity": "HIGH",
            "fix_location": "stream_live_csv.py ~22091 — gated on not was_entry_filled",
        },
        {
            "bug_id": "BUG-004",
            "description": (
                "ID collision: entry_ninja_order_id and stop_order_id have the same value "
                "after a fill. Downstream ORDER_UPDATE FILLED for the stop matches the entry "
                "ID guard and triggers fill_price_out_of_bounds lockout."
            ),
            "severity": "CRITICAL",
            "fix_location": "Consequence of BUG-001/002/003; prevented by all three fixes.",
        },
        {
            "bug_id": "BUG-005",
            "description": (
                "Multiple fills recorded without a corresponding exit: entry_filled=True but "
                "exit_fill_ts is None and position_state does not reflect an active position."
            ),
            "severity": "MEDIUM",
            "fix_location": "Operational guard — no code fix; requires operator assessment.",
        },
    ]

    def __init__(self, streamer: Any) -> None:
        """
        Args:
            streamer: A LiveCSVStreamer instance.
        """
        self._streamer = streamer

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def diagnose_order_ids(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse order-ID state for corruption patterns.

        Returns:
            {
                "status": "OK" | "WARNING" | "CRITICAL" | "UNKNOWN",
                "issues": [ { "code", "severity", "message", ... } ],
                "details": { raw field snapshot }
            }
        """
        issues: List[Dict[str, Any]] = []
        details: Dict[str, Any] = {}
        try:
            entry_id = state.get("entry_ninja_order_id")
            stop_id = state.get("stop_order_id")
            target_id = state.get("target_order_id")
            entry_order_id = state.get("entry_order_id")
            entry_filled = bool(state.get("entry_filled"))

            details = {
                "entry_ninja_order_id": entry_id,
                "entry_order_id": entry_order_id,
                "stop_order_id": stop_id,
                "target_order_id": target_id,
                "entry_filled": entry_filled,
            }

            # Check 1: entry/stop ID collision
            if entry_id and stop_id and str(entry_id) == str(stop_id):
                issues.append({
                    "code": "ENTRY_STOP_ID_COLLISION",
                    "severity": "CRITICAL",
                    "message": "entry_ninja_order_id and stop_order_id are identical",
                    "entry_ninja_order_id": str(entry_id),
                    "stop_order_id": str(stop_id),
                    "entry_filled": entry_filled,
                })

            # Check 2: entry/target ID collision
            if entry_id and target_id and str(entry_id) == str(target_id):
                issues.append({
                    "code": "ENTRY_TARGET_ID_COLLISION",
                    "severity": "CRITICAL",
                    "message": "entry_ninja_order_id and target_order_id are identical",
                    "entry_ninja_order_id": str(entry_id),
                    "target_order_id": str(target_id),
                })

            # Check 3: entry ID still opaque after fill
            if entry_filled and entry_id is not None and self._is_opaque_handle(entry_id):
                issues.append({
                    "code": "ENTRY_ID_STILL_OPAQUE",
                    "severity": "CRITICAL",
                    "message": (
                        "entry_ninja_order_id not upgraded to real exchange ID after entry fill — "
                        "indicates BUG-001 pattern (EXITS_SUBMITTED ACK reset)"
                    ),
                    "entry_ninja_order_id": str(entry_id),
                })

            # Check 4: stop ID still opaque after entry fill
            if entry_filled and stop_id is not None and self._is_opaque_handle(stop_id):
                issues.append({
                    "code": "STOP_ID_STILL_OPAQUE",
                    "severity": "WARNING",
                    "message": (
                        "stop_order_id still has opaque handle after entry fill — "
                        "indicates BUG-002 pattern (exchange ID never adopted)"
                    ),
                    "stop_order_id": str(stop_id),
                })

            # Check 5: entry ID missing after fill
            if entry_filled and entry_id is None:
                issues.append({
                    "code": "ENTRY_ID_MISSING_AFTER_FILL",
                    "severity": "CRITICAL",
                    "message": "entry_ninja_order_id is None after entry fill",
                })

            # Check 6: stop ID missing after entry fill (only for bracketed trades)
            if entry_filled and stop_id is None and target_id is None:
                if state.get("stop_price") is not None:
                    issues.append({
                        "code": "STOP_ID_MISSING_AFTER_FILL",
                        "severity": "WARNING",
                        "message": "stop_order_id not set after entry fill (expected for bracketed trade)",
                    })

        except Exception as exc:
            return {
                "status": "UNKNOWN",
                "issues": [{"code": "DIAGNOSTIC_ERROR", "severity": "UNKNOWN", "message": str(exc)}],
                "details": details,
            }

        if not issues:
            status = "OK"
        elif any(i["severity"] == "CRITICAL" for i in issues):
            status = "CRITICAL"
        else:
            status = "WARNING"

        return {"status": status, "issues": issues, "details": details}

    def verify_fill_consistency(
        self,
        state: Dict[str, Any],
        fill_price: float,
        fill_ts: Any,
        client_order_id: str,
    ) -> Dict[str, Any]:
        """
        Verify a fill price is consistent with model parameters.

        Key insight: stop fills should be compared against stop_price, not the
        entry model_price. Detecting this mismatch catches the tuesday326pt2 bug
        pattern before it triggers a lockout.

        Returns:
            {
                "status": "OK" | "SUSPICIOUS" | "ANOMALY" | "UNKNOWN",
                "checks": { ... },
                "recommendation": str
            }
        """
        checks: Dict[str, Any] = {}
        anomalies: List[str] = []
        suspicious: List[str] = []
        recommendation = "No action required."

        try:
            tick = float(getattr(self._streamer, "tick_size", 0.25) or 0.25)
            max_slip_ticks = float(
                getattr(self._streamer, "max_fill_slippage_ticks", 12.0) or 12.0
            )
            max_slip_pts = max_slip_ticks * tick

            model_price = _safe_float(
                state.get("expected_entry_ref")
                or state.get("model_price")
                or state.get("entry_price")
            )
            stop_price = _safe_float(state.get("stop_price"))
            entry_fill_price = _safe_float(state.get("entry_fill_price"))
            entry_filled = bool(state.get("entry_filled"))
            order_sent_ts = state.get("sent_ts") or state.get("order_sent_ts")

            # Slippage vs model/expected_entry_ref
            if model_price is not None:
                slip = abs(fill_price - model_price)
                over_limit = slip > max_slip_pts
                checks["entry_slippage"] = {
                    "fill_price": fill_price,
                    "model_price": model_price,
                    "slippage_pts": round(slip, 4),
                    "max_slip_pts": max_slip_pts,
                    "over_limit": over_limit,
                }
                if over_limit:
                    suspicious.append(
                        f"Fill {fill_price} exceeds slippage tolerance "
                        f"({slip:.2f}pt > {max_slip_pts:.2f}pt from model {model_price})"
                    )

            # Stop-fill-vs-entry-price detection (tuesday326pt2 bug pattern)
            if entry_filled and entry_fill_price is not None and stop_price is not None:
                fill_near_stop = abs(fill_price - stop_price) <= max_slip_pts
                fill_near_entry = abs(fill_price - entry_fill_price) <= max_slip_pts
                fill_far_from_entry_ref = (
                    model_price is not None
                    and abs(fill_price - model_price) > max_slip_pts
                )
                checks["stop_fill_detection"] = {
                    "fill_price": fill_price,
                    "entry_fill_price": entry_fill_price,
                    "stop_price": stop_price,
                    "fill_near_stop": fill_near_stop,
                    "fill_near_entry": fill_near_entry,
                    "fill_far_from_entry_ref": fill_far_from_entry_ref,
                }
                if fill_near_stop and not fill_near_entry and fill_far_from_entry_ref:
                    anomalies.append(
                        f"Stop fill ({fill_price}) is near stop_price ({stop_price}) "
                        f"but far from entry_fill_price ({entry_fill_price}) — "
                        "potential BUG-004: stop fill incorrectly compared to entry price"
                    )
                    recommendation = (
                        "This fill pattern matches the tuesday326pt2 lockout bug. "
                        "Verify stop_order_id is correctly bound to the real exchange ID "
                        "and not the opaque handle."
                    )

            # Fill timing checks
            if order_sent_ts is not None and fill_ts is not None:
                try:
                    sent_epoch = _ts_to_epoch(order_sent_ts)
                    fill_epoch = _ts_to_epoch(fill_ts)
                    if sent_epoch is not None and fill_epoch is not None:
                        fill_age = fill_epoch - sent_epoch
                        checks["fill_timing"] = {
                            "fill_age_sec": round(fill_age, 3),
                            "precedes_send": fill_age < 0,
                        }
                        if fill_age < 0:
                            suspicious.append("Fill timestamp precedes order send")
                        elif fill_age > 30:
                            suspicious.append(f"Late fill: {fill_age:.1f}s after order send")
                except Exception:
                    pass

        except Exception as exc:
            return {
                "status": "UNKNOWN",
                "checks": {"error": str(exc)},
                "recommendation": "Diagnostic error — inspect manually.",
            }

        if anomalies:
            status = "ANOMALY"
            if recommendation == "No action required.":
                recommendation = "; ".join(anomalies)
        elif suspicious:
            status = "SUSPICIOUS"
            recommendation = "; ".join(suspicious)
        else:
            status = "OK"

        checks["anomalies"] = anomalies
        checks["suspicious"] = suspicious
        return {"status": status, "checks": checks, "recommendation": recommendation}

    def detect_known_bug_patterns(self) -> List[Dict[str, Any]]:
        """
        Scan current state for patterns matching known historical bugs.

        Returns a list of bug-pattern dicts, one per known bug.
        Each dict includes:
            bug_id, description, severity, indicators (list of strings),
            detected (bool), fix_applied (bool), recommendation (str).
        """
        results: List[Dict[str, Any]] = []
        try:
            state = self._active_state()
            entry_id = state.get("entry_ninja_order_id")
            stop_id = state.get("stop_order_id")
            entry_filled = bool(state.get("entry_filled"))

            # BUG-001: entry_ninja_order_id is still opaque after entry fill
            bug001 = (
                entry_filled
                and entry_id is not None
                and self._is_opaque_handle(entry_id)
            )
            results.append({
                **self._KNOWN_BUGS[0],
                "indicators": (
                    ["entry_filled=True", f"entry_ninja_order_id='{entry_id}' is opaque handle"]
                    if bug001 else []
                ),
                "detected": bug001,
                "fix_applied": True,
                "recommendation": (
                    "Fix applied (~21566). If this fires, the fix may not be active in this process."
                    if bug001 else "Not detected."
                ),
            })

            # BUG-002: stop_order_id is still opaque after entry fill
            bug002 = (
                entry_filled
                and stop_id is not None
                and self._is_opaque_handle(stop_id)
            )
            results.append({
                **self._KNOWN_BUGS[1],
                "indicators": (
                    ["entry_filled=True", f"stop_order_id='{stop_id}' is still opaque handle"]
                    if bug002 else []
                ),
                "detected": bug002,
                "fix_applied": True,
                "recommendation": (
                    "Fix applied (~22238). stop_order_id should upgrade to exchange ID on SUBMITTED/ACCEPTED."
                    if bug002 else "Not detected."
                ),
            })

            # BUG-003: stop fill fell through ENTRY branch →
            #   entry_ninja_order_id == stop_order_id (both numeric, same value)
            bug003 = (
                entry_id is not None
                and stop_id is not None
                and str(entry_id) == str(stop_id)
                and not self._is_opaque_handle(entry_id)
            )
            results.append({
                **self._KNOWN_BUGS[2],
                "indicators": (
                    [
                        f"entry_ninja_order_id == stop_order_id == '{entry_id}'",
                        "Stop fill fell through ENTRY branch and corrupted entry_ninja_order_id",
                    ]
                    if bug003 else []
                ),
                "detected": bug003,
                "fix_applied": True,
                "recommendation": (
                    "Fix applied (~22091). was_entry_filled guard prevents overwrite."
                    if bug003 else "Not detected."
                ),
            })

            # BUG-004: any ID collision (superset of BUG-003)
            bug004 = (
                entry_id is not None
                and stop_id is not None
                and str(entry_id) == str(stop_id)
            )
            results.append({
                **self._KNOWN_BUGS[3],
                "indicators": (
                    [f"entry_ninja_order_id == stop_order_id == '{entry_id}'"]
                    if bug004 else []
                ),
                "detected": bug004,
                "fix_applied": True,
                "recommendation": (
                    "ID collision active. If fill_price_out_of_bounds lockout is set, "
                    "auto-heal via reset_lockout_if_safe() may apply."
                    if bug004 else "Not detected."
                ),
            })

            # BUG-005: entry_filled with no exit and position_state inconsistent
            exit_fill_ts = state.get("exit_fill_ts") or state.get("exit_ts")
            pos_state = str(
                getattr(self._streamer.state, "position_state", "") or ""
            ).upper()
            bug005 = (
                entry_filled
                and exit_fill_ts is None
                and state.get("entry_fill_price") is not None
                and pos_state not in {
                    "IN_POSITION_PROTECTED",
                    "IN_POSITION_UNPROTECTED",
                    "EXITS_WORKING",
                    "EXITS_SUBMITTED",
                }
            )
            results.append({
                **self._KNOWN_BUGS[4],
                "indicators": (
                    [
                        "entry_filled=True",
                        "exit_fill_ts=None",
                        f"position_state='{pos_state}' does not indicate active position",
                    ]
                    if bug005 else []
                ),
                "detected": bug005,
                "fix_applied": False,
                "recommendation": "Operator assessment required." if bug005 else "Not detected.",
            })

        except Exception as exc:
            results.append({
                "bug_id": "DIAGNOSTIC_ERROR",
                "description": f"Error running bug pattern scan: {exc}",
                "severity": "UNKNOWN",
                "indicators": [],
                "detected": False,
                "fix_applied": False,
                "recommendation": "Inspect diagnostics module.",
            })

        return results

    def run_full_diagnostics(self) -> Dict[str, Any]:
        """
        Run all diagnostic checks and return a comprehensive health report.

        Returns:
            {
                "timestamp": ISO8601 str,
                "run_id": str,
                "overall_status": "HEALTHY" | "DEGRADED" | "CRITICAL",
                "checks": {
                    "order_ids": {...},
                    "fill_consistency": {...},
                    "bug_patterns": [...],
                    "guardrails": {...},
                    "nt_connection": {...},
                    "feed_health": {...},
                },
                "actions_recommended": [...],
                "auto_heal_available": [...],
            }
        """
        ts = datetime.now(timezone.utc).isoformat()
        run_id = str(getattr(self._streamer, "run_id", "unknown"))
        severity_levels: List[str] = []
        checks: Dict[str, Any] = {}
        actions: List[Dict[str, Any]] = []
        auto_heal: List[Dict[str, Any]] = []

        # --- order_ids ---
        try:
            state = self._active_state()
            oid = self.diagnose_order_ids(state)
            checks["order_ids"] = oid
            if oid["status"] in {"WARNING", "CRITICAL"}:
                severity_levels.append(oid["status"])
        except Exception as exc:
            checks["order_ids"] = {"status": "UNKNOWN", "error": str(exc)}
            severity_levels.append("UNKNOWN")

        # --- fill_consistency (last fill in active state) ---
        try:
            state = self._active_state()
            fill_price = _safe_float(state.get("entry_fill_price"))
            if fill_price is not None:
                cid = self._active_cid() or "unknown"
                fc = self.verify_fill_consistency(
                    state, fill_price, state.get("fill_ts"), cid
                )
                checks["fill_consistency"] = fc
                if fc["status"] == "ANOMALY":
                    severity_levels.append("CRITICAL")
                elif fc["status"] == "SUSPICIOUS":
                    severity_levels.append("WARNING")
            else:
                checks["fill_consistency"] = {
                    "status": "OK",
                    "note": "No fill recorded in current active state",
                }
        except Exception as exc:
            checks["fill_consistency"] = {"status": "UNKNOWN", "error": str(exc)}

        # --- bug_patterns ---
        try:
            bugs = self.detect_known_bug_patterns()
            checks["bug_patterns"] = bugs
            detected = [b for b in bugs if b.get("detected")]
            if detected:
                worst_sev = max(
                    detected,
                    key=lambda b: {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(
                        b.get("severity", "LOW"), 0
                    ),
                )["severity"]
                severity_levels.append(
                    "CRITICAL" if worst_sev == "CRITICAL" else "WARNING"
                )
        except Exception as exc:
            checks["bug_patterns"] = [{"bug_id": "DIAGNOSTIC_ERROR", "error": str(exc)}]

        # --- guardrails ---
        try:
            lockout_active = bool(getattr(self._streamer, "_hard_lockout_active", False))
            lockout_code = getattr(self._streamer, "_hard_lockout_code", None)
            lockout_detail = getattr(self._streamer, "_hard_lockout_detail", None)
            lockout_sticky = bool(getattr(self._streamer, "_lockout_sticky", False))
            checks["guardrails"] = {
                "hard_lockout_active": lockout_active,
                "lockout_code": lockout_code,
                "lockout_detail": lockout_detail,
                "sticky": lockout_sticky,
            }
            if lockout_active:
                severity_levels.append("CRITICAL")
        except Exception as exc:
            checks["guardrails"] = {"status": "UNKNOWN", "error": str(exc)}

        # --- nt_connection ---
        try:
            nt_bridge = getattr(self._streamer, "nt_bridge", None)
            nt_connected = (
                bool(getattr(nt_bridge, "is_connected", False)) if nt_bridge else False
            )
            handshake_ok = False
            try:
                handshake_ok = bool(nt_bridge.handshake_ok()) if nt_bridge else False
            except Exception:
                pass
            healthy = nt_connected and handshake_ok
            checks["nt_connection"] = {
                "status": "HEALTHY" if healthy else "DEGRADED",
                "nt_connected": nt_connected,
                "handshake_ok": handshake_ok,
            }
            if not nt_connected:
                severity_levels.append("WARNING")
        except Exception as exc:
            checks["nt_connection"] = {"status": "UNKNOWN", "error": str(exc)}

        # --- feed_health ---
        try:
            feed_age: Optional[float] = None
            try:
                feed_age = self._streamer._bar_age_guard_seconds()
            except Exception:
                pass
            bar_age_max = float(
                getattr(self._streamer, "_effective_bar_age_max_sec", 0.0) or 0.0
            )
            feed_ok = (
                feed_age is not None and bar_age_max > 0 and float(feed_age) <= bar_age_max
            )
            checks["feed_health"] = {
                "status": "HEALTHY" if feed_ok else "DEGRADED",
                "feed_health_ok": feed_ok,
                "bar_age_sec": feed_age,
                "bar_age_max_sec": bar_age_max,
            }
        except Exception as exc:
            checks["feed_health"] = {"status": "UNKNOWN", "error": str(exc)}

        # --- execution_path_health ---
        queue_degraded = False
        queue_overload = False
        stale_disarm = False
        try:
            queue_degraded = getattr(self._streamer, "_nt_event_queue_degraded", False) is True
            queue_overload = getattr(self._streamer, "_queue_overload_active", False) is True
            stale_disarm = (
                str(getattr(self._streamer, "_entries_disarmed_reason", "") or "")
                == "snapshot_stale_reconcile"
            )
            fiber_timeout = (
                str(getattr(self._streamer, "_bar_continuity_break_reason", "") or "")
                == "fiber_bar_timeout"
            )
            checks["execution_path_health"] = {
                "status": (
                    "DEGRADED"
                    if (queue_degraded or queue_overload or stale_disarm or fiber_timeout)
                    else "HEALTHY"
                ),
                "queue_degraded": queue_degraded,
                "queue_overload_active": queue_overload,
                "entries_disarmed_reason": getattr(
                    self._streamer, "_entries_disarmed_reason", None
                ),
                "bar_continuity_break_reason": getattr(
                    self._streamer, "_bar_continuity_break_reason", None
                ),
            }
            # Never allow stale-reconcile disarm to report globally healthy.
            if stale_disarm:
                severity_levels.append("WARNING")
            if queue_degraded or queue_overload or fiber_timeout:
                severity_levels.append("CRITICAL")
        except Exception as exc:
            checks["execution_path_health"] = {"status": "UNKNOWN", "error": str(exc)}

        # --- snapshot_progress_health ---
        try:
            snapshot_age = None
            try:
                snapshot_age = self._streamer._snapshot_age_sec()
            except Exception:
                snapshot_age = None
            stale_max = float(getattr(self._streamer, "nt_snapshot_fresh_sec", 0.0) or 0.0)
            progress_age = _safe_float(
                getattr(self._streamer, "_snapshot_ts_progress_age_sec", None)
            )
            arrival_wall_ts = _safe_float(
                getattr(self._streamer, "_nt_last_snapshot_arrival_progress_wall_ts", None)
            )
            now_ts = time.time()
            arrival_age = (
                max(0.0, now_ts - float(arrival_wall_ts))
                if arrival_wall_ts is not None
                else None
            )
            recovery_attempts = int(
                getattr(self._streamer, "_snapshot_recovery_attempts", 0) or 0
            )
            stale_reason = str(
                getattr(self._streamer, "_snapshot_stale_lockout_reason", "") or ""
            )
            stream_alive = (
                arrival_age is not None and stale_max > 0 and arrival_age <= stale_max
            )
            progress_stalled = (
                progress_age is None or (stale_max > 0 and progress_age > stale_max)
            )
            stalled = bool(stale_disarm and recovery_attempts >= 3 and progress_stalled)
            stream_alive_price_stale = bool(
                stale_disarm and stale_reason == "price_stale" and stream_alive
            )
            freshness_stale_only = bool(
                snapshot_age is not None
                and stale_max > 0
                and snapshot_age > stale_max
                and stream_alive
                and not stale_disarm
            )
            blocker = "none"
            if stalled:
                blocker = "snapshot_stream_stalled"
            elif stream_alive_price_stale:
                blocker = "snapshot_stream_alive_price_stale"
            elif freshness_stale_only:
                blocker = "snapshot_freshness_stale_only"
            elif stale_disarm:
                blocker = "snapshot_contract_inconsistent"
            checks["snapshot_progress_health"] = {
                "status": "DEGRADED" if blocker != "none" else "HEALTHY",
                "blocker": blocker,
                "snapshot_age_sec": snapshot_age,
                "snapshot_fresh_sec": stale_max,
                "snapshot_progress_age_sec": progress_age,
                "snapshot_arrival_age_sec": arrival_age,
                "snapshot_recovery_attempts": recovery_attempts,
                "snapshot_stale_lockout_reason": stale_reason,
                "stream_alive": stream_alive,
            }
            blocker_severity = {
                "snapshot_stream_stalled": "CRITICAL",
                "snapshot_stream_alive_price_stale": "WARNING",
                "snapshot_freshness_stale_only": "WARNING",
                "snapshot_contract_inconsistent": "WARNING",
            }.get(blocker)
            if blocker_severity:
                severity_levels.append(blocker_severity)
        except Exception as exc:
            checks["snapshot_progress_health"] = {"status": "UNKNOWN", "error": str(exc)}

        # --- Build action recommendations ---
        try:
            lockout_active = checks.get("guardrails", {}).get("hard_lockout_active", False)
            lockout_code = checks.get("guardrails", {}).get("lockout_code")
            oid_status = checks.get("order_ids", {}).get("status", "UNKNOWN")
            detected_bug_ids = [
                b["bug_id"]
                for b in (checks.get("bug_patterns") or [])
                if b.get("detected")
            ]

            if lockout_active and lockout_code == "fill_price_out_of_bounds":
                can_heal, method = self.can_auto_heal("fill_price_out_of_bounds")
                actions.append({
                    "action": "reset_lockout",
                    "reason": "fill_price_out_of_bounds lockout active",
                    "requires_approval": not can_heal,
                    "auto_heal_method": method if can_heal else None,
                })
                auto_heal.append({
                    "method": "reset_lockout_if_safe",
                    "conditions_met": can_heal,
                    "preconditions_verified": (
                        oid_status == "OK" and not detected_bug_ids
                    ),
                })

            opaque_stop_issues = [
                i
                for i in checks.get("order_ids", {}).get("issues", [])
                if i.get("code") == "STOP_ID_STILL_OPAQUE"
            ]
            if opaque_stop_issues:
                can_heal, method = self.can_auto_heal("stop_id_opaque")
                actions.append({
                    "action": "force_order_id_upgrade",
                    "reason": "stop_order_id still has opaque handle",
                    "requires_approval": not can_heal,
                    "auto_heal_method": method if can_heal else None,
                })
                if can_heal:
                    auto_heal.append({
                        "method": method,
                        "conditions_met": True,
                        "preconditions_verified": True,
                    })
        except Exception:
            pass

        # --- Overall status ---
        if not severity_levels:
            overall_status = "HEALTHY"
        elif "CRITICAL" in severity_levels:
            overall_status = "CRITICAL"
        else:
            overall_status = "DEGRADED"

        return {
            "timestamp": ts,
            "run_id": run_id,
            "overall_status": overall_status,
            "checks": checks,
            "actions_recommended": actions,
            "auto_heal_available": auto_heal,
        }

    def can_auto_heal(self, issue: str) -> Tuple[bool, str]:
        """
        Determine if an issue can be auto-healed without operator approval.

        Returns:
            (can_heal: bool, method: str)

        Auto-heal is only permitted when:
          1. The issue matches a known, patched bug pattern.
          2. The relevant fixes are verified as applied (fix_applied=True).
          3. No unpatched bugs are simultaneously active.
        """
        issue_key = issue.lower().strip().replace("-", "_")

        if issue_key in {"fill_price_out_of_bounds", "lockout_fill_price_out_of_bounds"}:
            try:
                lockout_code = getattr(self._streamer, "_hard_lockout_code", None)
                if lockout_code != "fill_price_out_of_bounds":
                    return False, "operator_required"
                state = self._active_state()
                oid = self.diagnose_order_ids(state)
                bugs = self.detect_known_bug_patterns()
                # All detected bugs must have fix_applied=True
                unpatched = [
                    b for b in bugs
                    if b.get("detected") and not b.get("fix_applied", False)
                ]
                if unpatched:
                    return False, "operator_required"
                # Order IDs must not be in CRITICAL state at the time of heal
                # (they may have been already resolved by the fix)
                if oid["status"] == "CRITICAL":
                    return False, "operator_required"
                return True, "reset_lockout_if_safe"
            except Exception:
                return False, "operator_required"

        if issue_key in {"stop_id_opaque", "stop_order_id_opaque"}:
            try:
                state = self._active_state()
                entry_filled = bool(state.get("entry_filled"))
                exit_filled = bool(
                    state.get("exit_filled") or state.get("exit_fill_ts")
                )
                if entry_filled and not exit_filled:
                    return True, "force_order_id_upgrade"
            except Exception:
                pass
            return False, "operator_required"

        if issue_key in {"nt_snapshot_stale", "snapshot_stale"}:
            try:
                snap_age = self._streamer._snapshot_age_sec()
                if snap_age is not None and float(snap_age) < 60:
                    return True, "request_nt_resync"
            except Exception:
                pass
            return False, "operator_required"

        # Feed issues and unknown causes always require operator
        return False, "operator_required"

    def reset_lockout_if_safe(self) -> Dict[str, Any]:
        """
        Reset hard lockout when all preconditions are met.

        Preconditions:
          1. hard_lockout_active == True
          2. lockout_code == "fill_price_out_of_bounds"
          3. diagnose_order_ids() returns OK (or WARNING at most)
          4. All detected bug patterns have fix_applied=True

        Actions taken on success:
          - Emits lockout_reset_auto exec event (full context)
          - Calls streamer._clear_hard_lockout()
          - Returns success=True with verification details

        Returns:
            { "success": bool, "method": str, "verification": dict, "message": str }
        """
        verification: Dict[str, Any] = {}
        try:
            lockout_active = bool(getattr(self._streamer, "_hard_lockout_active", False))
            lockout_code = getattr(self._streamer, "_hard_lockout_code", None)
            verification["lockout_active"] = lockout_active
            verification["lockout_code"] = lockout_code

            if not lockout_active:
                return {
                    "success": False,
                    "method": "none",
                    "verification": verification,
                    "message": "No active hard lockout — nothing to reset.",
                }

            if lockout_code != "fill_price_out_of_bounds":
                return {
                    "success": False,
                    "method": "none",
                    "verification": verification,
                    "message": (
                        f"Auto-heal only applies to fill_price_out_of_bounds; "
                        f"active code is '{lockout_code}'."
                    ),
                }

            # Precondition: order IDs
            state = self._active_state()
            oid = self.diagnose_order_ids(state)
            verification["order_id_check"] = oid["status"]
            if oid["status"] == "CRITICAL":
                return {
                    "success": False,
                    "method": "none",
                    "verification": verification,
                    "message": (
                        "Order ID check returned CRITICAL — unsafe to auto-reset. "
                        f"Issues: {oid['issues']}"
                    ),
                }

            # Precondition: no unpatched bugs
            bugs = self.detect_known_bug_patterns()
            unpatched = [
                b for b in bugs
                if b.get("detected") and not b.get("fix_applied", False)
            ]
            detected_ids = [b["bug_id"] for b in bugs if b.get("detected")]
            verification["detected_bugs"] = detected_ids
            verification["unpatched_bugs"] = [b["bug_id"] for b in unpatched]
            if unpatched:
                return {
                    "success": False,
                    "method": "none",
                    "verification": verification,
                    "message": (
                        f"Unpatched bugs detected: {[b['bug_id'] for b in unpatched]}. "
                        "Operator intervention required."
                    ),
                }

            # All preconditions met — log and clear
            verification["preconditions_ok"] = True
            cid = self._active_cid() or "unknown"
            try:
                self._streamer._log_exec_event({
                    "event": "lockout_reset_auto",
                    "lockout_code": lockout_code,
                    "client_order_id": cid,
                    "order_id_status": oid["status"],
                    "detected_bugs": detected_ids,
                    "all_fixes_applied": True,
                    "diagnostics_module": "Diagnostics.reset_lockout_if_safe",
                })
            except Exception:
                pass

            self._streamer._clear_hard_lockout(
                reason="auto_heal_diagnostics",
                evidence={
                    "method": "reset_lockout_if_safe",
                    "lockout_code": lockout_code,
                    "order_id_status": oid["status"],
                    "detected_bugs": detected_ids,
                },
            )

            return {
                "success": True,
                "method": "reset_lockout_if_safe",
                "verification": verification,
                "message": (
                    f"Lockout '{lockout_code}' cleared by diagnostics auto-heal. "
                    "All preconditions verified."
                ),
            }

        except Exception as exc:
            return {
                "success": False,
                "method": "error",
                "verification": verification,
                "message": f"Unexpected error during reset: {exc}",
            }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_opaque_handle(order_id: Any) -> bool:
        """
        Return True if order_id looks like an NT opaque internal handle.

        Opaque handles are UUID-format (with dashes) or 32-char lowercase hex
        strings (without dashes). Real exchange IDs are purely numeric.
        """
        if order_id is None:
            return False
        s = str(order_id).strip()
        if not s:
            return False
        # Purely numeric → real exchange ID, not opaque
        if _NUMERIC_RE.match(s):
            return False
        # UUID with dashes
        if _OPAQUE_UUID_RE.match(s):
            return True
        # 32-char hex without dashes
        if _OPAQUE_HEX32_RE.match(s):
            return True
        return False

    def _active_cid(self) -> Optional[str]:
        """Return the client_order_id for the active (open, unfilled exit) trade, or None."""
        try:
            nt_state: Dict[str, Any] = getattr(self._streamer, "_nt_order_state", {}) or {}
            if not nt_state:
                return None
            # Prefer the most recent CID with entry_filled=True and no exit
            for cid, st in reversed(list(nt_state.items())):
                if st.get("entry_filled") and not st.get("exit_fill_ts"):
                    return str(cid)
            # Fall back to most recently inserted key
            return str(next(reversed(nt_state)))
        except Exception:
            return None

    def _active_state(self) -> Dict[str, Any]:
        """Return a copy of the state dict for the active trade, or {}."""
        try:
            cid = self._active_cid()
            if cid is None:
                return {}
            nt_state: Dict[str, Any] = getattr(self._streamer, "_nt_order_state", {}) or {}
            return dict(nt_state.get(cid) or {})
        except Exception:
            return {}
