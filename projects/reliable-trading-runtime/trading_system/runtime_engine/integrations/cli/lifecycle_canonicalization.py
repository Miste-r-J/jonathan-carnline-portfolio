"""
lifecycle_canonicalization.py - Phase 1: Unified lifecycle finalization for live_trading_runtime.py

This module consolidates all lifecycle decision logic (requested action -> resolved action -> execution intent)
into a single canonical source of truth. It eliminates duplication between _maybe_emit_signal() and
_maybe_send_nt_order() processing branches.

Key functions:
- _resolve_block_reason(): Unified block detection (gate/policy/phase blocks)
- _canonical_lifecycle_record(): Single finalization point for all actions
- _write_lifecycle_event(): Canonical ledger output
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from datetime import datetime as dt_type


def _resolve_block_reason(
    *,
    action: str,
    phase2_setup_pass: Any,
    override_applied: Any,
    allow_setup_fail_entries: bool = False,
    blocked_by: Sequence[str],
    current_phase: str,
    phase_allows_execution: bool,
    gates_detail: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, Optional[str], List[str], bool]:
    """
    Unified block detection across gate/policy/phase dimensions.

    Returns: (is_blocked, block_reason, blocked_by_list, phase_allows_execution)
    """
    action_upper = str(action or "").upper()
    blocked = [str(item) for item in list(blocked_by or []) if str(item)]

    # Non-entry actions always pass through
    if action_upper not in {"OPEN", "FLIP"}:
        return False, None, blocked, phase_allows_execution

    # Check phase gate
    if not phase_allows_execution and action_upper in {"OPEN", "CLOSE", "FLIP", "EXIT"}:
        if "phase" not in blocked:
            blocked.append("phase")
        return True, "phase_not_executable", blocked, False

    # Check setup gate (only for OPEN/FLIP)
    setup_pass = bool(phase2_setup_pass) if phase2_setup_pass is not None else True
    override_used = bool(override_applied)
    setup_fail_entries_allowed = bool(allow_setup_fail_entries)

    if action_upper in {"OPEN", "FLIP"} and not setup_pass and not override_used and not setup_fail_entries_allowed:
        if "setup" not in blocked:
            blocked.append("setup")
        return True, "setup_blocked", blocked, phase_allows_execution

    if action_upper in {"OPEN", "FLIP"} and setup_fail_entries_allowed and "setup" in blocked:
        blocked = [item for item in blocked if item != "setup"]

    # Check for any other gate blockers (vwap, ema, tod, risk, etc.)
    if blocked:
        return True, blocked[0] if blocked else "blocked", blocked, phase_allows_execution

    return False, None, blocked, phase_allows_execution


def _canonical_lifecycle_record(
    *,
    ev: Dict[str, Any],
    phase2_meta: Optional[Mapping[str, Any]] = None,
    gates_detail: Optional[Mapping[str, Any]] = None,
    current_phase: str = "LIVE",
    phase_allows_execution: bool = True,
    allow_setup_fail_entries: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Single finalization point for all lifecycle decisions.

    Produces a standardized record with:
    - requested_action: what was asked (OPEN/CLOSE/FLIP)
    - resolved_action: after all gates/policy/phase checks
    - execution_intent_action: what goes to NT execution
    - emit_allowed: whether signal can be rendered
    - blocked_by: list of block reasons
    - block_reason: first blocker
    - phase_allows_execution: boolean
    - flip_decomposed: if FLIP, track close/open legs

    Returns: canonical lifecycle record dict
    """
    action_upper = str(ev.get("type") or "").upper()
    blocked_by_list = list(ev.get("_signal_blocked_by") or [])

    phase2_meta = phase2_meta or {}
    gates_detail = gates_detail or {}

    # Determine phase2 setup pass
    phase2_setup_pass = None
    if phase2_meta.get("setup_pass") is not None:
        phase2_setup_pass = phase2_meta.get("setup_pass")
    elif gates_detail.get("gate_state", {}).get("setup") is not None:
        phase2_setup_pass = gates_detail.get("gate_state", {}).get("setup")
    else:
        phase2_setup_pass = True  # default: pass

    override_applied = gates_detail.get("override_applied", False)
    if allow_setup_fail_entries is None:
        policy_sources = (
            phase2_meta.get("allow_setup_fail_entries"),
            (phase2_meta.get("phase2_force_open_policy") or {}).get("allow_setup_fail_entries")
            if isinstance(phase2_meta.get("phase2_force_open_policy"), Mapping)
            else None,
            gates_detail.get("allow_setup_fail_entries"),
            (gates_detail.get("phase2_force_open_policy") or {}).get("allow_setup_fail_entries")
            if isinstance(gates_detail.get("phase2_force_open_policy"), Mapping)
            else None,
        )
        allow_setup_fail_entries = any(bool(value) for value in policy_sources if value is not None)

    # Resolve all blocking conditions
    is_blocked, block_reason, final_blocked_by, phase_allows = _resolve_block_reason(
        action=action_upper,
        phase2_setup_pass=phase2_setup_pass,
        override_applied=override_applied,
        allow_setup_fail_entries=bool(allow_setup_fail_entries),
        blocked_by=blocked_by_list,
        current_phase=current_phase,
        phase_allows_execution=phase_allows_execution,
        gates_detail=gates_detail,
    )

    # Determine resolved and execution intent actions
    requested_action = action_upper
    resolved_action = "NO_TRADE" if is_blocked else action_upper
    execution_intent_action = resolved_action
    emit_allowed = not is_blocked

    # Handle FLIP decomposition for execution
    flip_decomposed = None
    if action_upper == "FLIP" and not is_blocked:
        # FLIP decomposes to CLOSE + OPEN with shared transition_id
        transition_id = str(ev.get("transition_id") or "")
        flip_decomposed = {
            "close_step": {
                "action": "CLOSE",
                "transition_id": transition_id,
                "transition_step": "close",
            },
            "open_step": {
                "action": "OPEN",
                "transition_id": transition_id,
                "transition_step": "open",
            }
        }

    record = {
        "requested_action": requested_action,
        "resolved_action": resolved_action,
        "display_action": "HOLD" if not emit_allowed else resolved_action,
        "execution_intent_action": execution_intent_action,
        "side": str(ev.get("side") or "FLAT").upper(),
        "price": ev.get("price"),
        "prob": ev.get("prob"),
        "emit_allowed": emit_allowed,
        "publish_ready": emit_allowed,
        "blocked_reason": block_reason,
        "blocked_by": final_blocked_by,
        "transition_id": ev.get("transition_id"),
        "transition_step": None,
        "signal_id": ev.get("signal_id"),
        "client_order_id": ev.get("client_order_id"),
        "phase": current_phase,
        "phase_allows_execution": phase_allows,
        "flip_decomposed": flip_decomposed,
        "source": "model",  # or "override", "fallback"
        "feature_hash": ev.get("feature_hash") or ev.get("features_hash"),
        "model_version": ev.get("model_version"),
        "config_hash": ev.get("config_hash"),
        "raw_model_output": ev.get("raw_model_output"),
    }

    return record


def _write_lifecycle_event(
    *,
    record: Dict[str, Any],
    bar_ts: Optional[Any] = None,
    out_file: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Format a lifecycle record for writing to lifecycle_events.jsonl.

    Returns: dict suitable for JSON serialization to lifecycle_events.jsonl
    """
    import json
    from datetime import datetime as dt_type

    # Ensure we have a bar_ts
    if bar_ts is None:
        bar_ts = record.get("bar_ts")

    bar_ts_str = None
    if bar_ts is not None:
        if isinstance(bar_ts, str):
            bar_ts_str = bar_ts
        elif isinstance(bar_ts, dt_type):
            bar_ts_str = bar_ts.isoformat()
        else:
            try:
                bar_ts_str = str(bar_ts)
            except:
                pass

    # Build the canonical ledger entry
    lifecycle_entry = {
        "ts": dt_type.utcnow().isoformat(),
        "bar_ts": bar_ts_str,
        "phase": record.get("phase", "LIVE"),
        "requested_action": record.get("requested_action"),
        "resolved_action": record.get("resolved_action"),
        "execution_intent_action": record.get("execution_intent_action"),
        "display_action": record.get("display_action"),
        "side": record.get("side"),
        "price": record.get("price"),
        "prob": record.get("prob"),
        "emit_allowed": record.get("emit_allowed"),
        "publish_ready": record.get("publish_ready"),
        "blocked_reason": record.get("blocked_reason"),
        "blocked_by": record.get("blocked_by", []),
        "transition_id": record.get("transition_id"),
        "transition_step": record.get("transition_step"),
        "signal_id": record.get("signal_id"),
        "client_order_id": record.get("client_order_id"),
        "source": record.get("source", "model"),
        "dedupe_key": f"{bar_ts_str}|{record.get('signal_id')}|{record.get('requested_action')}",
    }

    # Remove None values for cleaner output
    lifecycle_entry = {k: v for k, v in lifecycle_entry.items() if v is not None or k in ["blocked_reason", "transition_step"]}

    return lifecycle_entry


# Schema definition for state.csv columns (new columns to add)
LIFECYCLE_COLUMNS_FOR_STATE_CSV = [
    "requested_action",
    "resolved_action",
    "execution_intent_action",
    "transition_id",
    "blocked_by",
]

# Schema definition for lifecycle_events.jsonl
LIFECYCLE_EVENT_SCHEMA = [
    "ts",
    "bar_ts",
    "phase",
    "requested_action",
    "resolved_action",
    "execution_intent_action",
    "display_action",
    "side",
    "price",
    "prob",
    "emit_allowed",
    "publish_ready",
    "blocked_reason",
    "blocked_by",
    "transition_id",
    "transition_step",
    "signal_id",
    "client_order_id",
    "source",
    "dedupe_key",
]
