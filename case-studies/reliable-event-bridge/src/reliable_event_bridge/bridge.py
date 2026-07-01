"""A small, domain-neutral demonstration of reliable command processing.

This module is an original reference implementation. It demonstrates engineering
patterns used in a larger private integration without containing production
source, strategy logic, credentials, endpoints, or proprietary configuration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class Command:
    """A command entering the bridge from an upstream system."""

    command_id: str
    generation: int
    action: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """The downstream adapter's correlated terminal response."""

    command_id: str
    status: str
    detail: str = ""


class ReliableEventBridge:
    """Fail-closed command bridge with an auditable event trail.

    The class intentionally models only reliability behavior. The downstream
    action is supplied by the caller and can represent any controlled system.
    """

    def __init__(self, *, max_queue: int = 100) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be at least 1")
        self.max_queue = max_queue
        self.generation = 0
        self.handshake_ready = False
        self.degraded = False
        self.kill_switch_enabled = False
        self._session_nonce: str | None = None
        self._queue: deque[Command] = deque()
        self._seen_command_ids: set[str] = set()
        self._terminal_command_ids: set[str] = set()
        self._ledger: list[dict[str, Any]] = []

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def ledger(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._ledger)

    def connect(self, *, session_nonce: str) -> int:
        """Start a new connection generation in a non-ready state."""

        if not session_nonce.strip():
            raise ValueError("session_nonce is required")
        while self._queue:
            abandoned = self._queue.popleft()
            self._record(
                "connection_reset_abandoned",
                command_id=abandoned.command_id,
                reason="connection_generation_changed",
            )
        self.generation += 1
        self.handshake_ready = False
        self.degraded = False
        self._session_nonce = session_nonce
        self._record("connection_started", generation=self.generation)
        return self.generation

    def acknowledge_handshake(self, *, generation: int, session_nonce: str) -> bool:
        """Accept only an acknowledgement for the current connection identity."""

        if generation != self.generation or session_nonce != self._session_nonce:
            self.handshake_ready = False
            self.degraded = True
            self._record(
                "handshake_rejected",
                generation=generation,
                reason="connection_identity_mismatch",
            )
            return False
        self.handshake_ready = True
        self._record("handshake_accepted", generation=generation)
        return True

    def set_kill_switch(self, enabled: bool, *, reason: str) -> None:
        """Enable or disable the explicit operational stop control."""

        self.kill_switch_enabled = enabled
        self._record(
            "kill_switch_changed",
            enabled=enabled,
            reason=reason or "operator_request",
        )

    def submit(self, command: Command) -> str:
        """Validate and enqueue a command, returning a stable disposition."""

        if not command.command_id.strip():
            return self._reject(command, "missing_command_id")
        if self.kill_switch_enabled:
            return self._reject(command, "kill_switch_enabled")
        if not self.handshake_ready:
            return self._reject(command, "handshake_not_ready")
        if command.generation != self.generation:
            return self._reject(command, "stale_connection_generation")
        if command.command_id in self._seen_command_ids:
            self._record(
                "duplicate_suppressed",
                command_id=command.command_id,
                generation=command.generation,
            )
            return "duplicate"
        if len(self._queue) >= self.max_queue:
            self.degraded = True
            return self._reject(command, "queue_capacity_exceeded")

        self._seen_command_ids.add(command.command_id)
        self._queue.append(command)
        self._record(
            "command_accepted",
            command_id=command.command_id,
            generation=command.generation,
            action=command.action,
            queue_depth=len(self._queue),
        )
        return "accepted"

    def process_next(
        self,
        executor: Callable[[Command], ExecutionReceipt],
    ) -> ExecutionReceipt | None:
        """Execute one command and require an exact receipt correlation."""

        if not self._queue:
            return None
        command = self._queue.popleft()
        try:
            receipt = executor(command)
        except Exception as exc:  # boundary intentionally converts exceptions to evidence
            receipt = ExecutionReceipt(command.command_id, "failed", type(exc).__name__)
            self._record(
                "execution_failed",
                command_id=command.command_id,
                reason=type(exc).__name__,
            )
            return receipt

        if receipt.command_id != command.command_id:
            self.degraded = True
            self._record(
                "correlation_rejected",
                command_id=command.command_id,
                received_command_id=receipt.command_id,
                reason="receipt_identity_mismatch",
            )
            return ExecutionReceipt(command.command_id, "failed", "receipt_identity_mismatch")

        normalized_status = receipt.status.strip().lower()
        if normalized_status not in {"completed", "failed"}:
            self.degraded = True
            self._record(
                "receipt_rejected",
                command_id=command.command_id,
                reason="non_terminal_status",
            )
            return ExecutionReceipt(command.command_id, "failed", "non_terminal_status")

        self._terminal_command_ids.add(command.command_id)
        self._record(
            "execution_terminal",
            command_id=command.command_id,
            status=normalized_status,
            detail=receipt.detail,
        )
        return ExecutionReceipt(command.command_id, normalized_status, receipt.detail)

    def reconcile(self, receipts: Iterable[ExecutionReceipt]) -> int:
        """Record recovered terminal state without re-executing commands."""

        recovered = 0
        for receipt in receipts:
            if not receipt.command_id.strip():
                self._record("reconcile_ignored", reason="missing_command_id")
                continue
            normalized_status = receipt.status.strip().lower()
            if normalized_status not in {"completed", "failed"}:
                self._record(
                    "reconcile_ignored",
                    command_id=receipt.command_id,
                    reason="non_terminal_status",
                )
                continue
            if receipt.command_id in self._terminal_command_ids:
                self._record("reconcile_duplicate", command_id=receipt.command_id)
                continue
            self._seen_command_ids.add(receipt.command_id)
            self._terminal_command_ids.add(receipt.command_id)
            self._record(
                "reconcile_recovered",
                command_id=receipt.command_id,
                status=normalized_status,
                execution_allowed=False,
            )
            recovered += 1
        return recovered

    def to_jsonl(self) -> str:
        """Serialize the evidence trail using stable, machine-readable records."""

        return "\n".join(json.dumps(event, sort_keys=True) for event in self._ledger)

    def _reject(self, command: Command, reason: str) -> str:
        self._record(
            "command_rejected",
            command_id=command.command_id or None,
            generation=command.generation,
            reason=reason,
        )
        return "rejected"

    def _record(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        self._ledger.append(record)


def command_as_dict(command: Command) -> dict[str, Any]:
    """Return a serializable command shape for adapters and diagnostics."""

    return asdict(command)
