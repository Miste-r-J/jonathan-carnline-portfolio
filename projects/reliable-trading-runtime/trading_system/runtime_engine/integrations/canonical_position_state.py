"""
canonical_position_state.py - Single Source of Truth for Position State

Implements immutable snapshot pattern with versioning, authority-based write locks,
and atomic snapshot adoption for Phase 1 (Dual-Write Mode) of position state fix.
"""

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

logger = logging.getLogger("canonical_pos_state")


class CanonicalPositionState:
    """
    Immutable snapshot of position state.
    All writes create new instances (copy-on-write semantics).
    """

    __slots__ = (
        "_CanonicalPositionState__version",
        "_CanonicalPositionState__timestamp",
        "_CanonicalPositionState__quantity",
        "_CanonicalPositionState__entry_price",
        "_CanonicalPositionState__entry_stop",
        "_CanonicalPositionState__entry_target",
        "_CanonicalPositionState__pnl",
        "_CanonicalPositionState__status_flags",
        "_CanonicalPositionState__metadata",
    )

    def __init__(
        self,
        quantity: float,
        entry_price: float,
        entry_stop: Optional[float] = None,
        entry_target: Optional[float] = None,
        pnl: float = 0.0,
        version: int = 0,
        status_flags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize immutable snapshot."""
        object.__setattr__(self, "_CanonicalPositionState__version", int(version))
        object.__setattr__(self, "_CanonicalPositionState__timestamp", time.time())
        object.__setattr__(self, "_CanonicalPositionState__quantity", float(quantity))
        object.__setattr__(self, "_CanonicalPositionState__entry_price", float(entry_price))
        object.__setattr__(self, "_CanonicalPositionState__entry_stop", entry_stop if entry_stop is None else float(entry_stop))
        object.__setattr__(self, "_CanonicalPositionState__entry_target", entry_target if entry_target is None else float(entry_target))
        object.__setattr__(self, "_CanonicalPositionState__pnl", float(pnl))
        object.__setattr__(self, "_CanonicalPositionState__status_flags", frozenset(status_flags or set()))
        object.__setattr__(self, "_CanonicalPositionState__metadata", types.MappingProxyType(metadata or {}) if hasattr(types, 'MappingProxyType') else (metadata or {}))

    def __setattr__(self, name: str, value: Any) -> None:
        """Enforce immutability - block all writes after init."""
        raise AttributeError(f"Cannot modify immutable snapshot: {name}")

    @property
    def version(self) -> int:
        """Current snapshot version number."""
        return self.__version

    @property
    def timestamp(self) -> float:
        """Snapshot creation timestamp."""
        return self.__timestamp

    @property
    def quantity(self) -> float:
        """Position size (read-only)."""
        return self.__quantity

    @property
    def entry_price(self) -> float:
        """Average entry price (read-only)."""
        return self.__entry_price

    @property
    def entry_stop(self) -> Optional[float]:
        """Stop-loss level (read-only)."""
        return self.__entry_stop

    @property
    def entry_target(self) -> Optional[float]:
        """Take-profit level (read-only)."""
        return self.__entry_target

    @property
    def pnl(self) -> float:
        """Unrealized P&L (read-only)."""
        return self.__pnl

    @property
    def status_flags(self) -> FrozenSet[str]:
        """Position status flags (read-only)."""
        return self.__status_flags

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Snapshot metadata (read-only)."""
        return self.__metadata

    def compare_with(self, other: "CanonicalPositionState") -> Dict[str, Tuple[Any, Any]]:
        """
        Compare two snapshots and return differences.

        Returns:
            dict: {"field_name": (old_value, new_value), ...}
        """
        if not isinstance(other, CanonicalPositionState):
            raise TypeError("Can only compare with CanonicalPositionState")

        diffs = {}
        for field in ("quantity", "entry_price", "entry_stop", "entry_target", "pnl"):
            old_val = getattr(self, field)
            new_val = getattr(other, field)
            if old_val != new_val:
                diffs[field] = (old_val, new_val)

        return diffs

    def __repr__(self) -> str:
        return (
            f"CanonicalPositionState(v{self.version}, qty={self.quantity}, "
            f"entry_price={self.entry_price}, stop={self.entry_stop})"
        )


class WriteAuthorityLock:
    """Token representing exclusive write authority."""

    def __init__(
        self,
        subsystem: str,
        fields: FrozenSet[str],
        timestamp: float,
        manager: "CanonicalPositionStateManager",
        lock_type: str = "standard",
    ):
        self.subsystem = subsystem
        self.fields = fields
        self.timestamp = timestamp
        self.manager = manager
        self.lock_type = lock_type

    def __repr__(self) -> str:
        return f"WriteAuthorityLock({self.subsystem}, fields={self.fields}, type={self.lock_type})"


class CanonicalPositionStateManager:
    """
    Manages CanonicalPositionState lifecycle and authority.
    Central hub for all position state access.
    """

    def __init__(self):
        """Initialize manager with empty state."""
        self._current_state: Optional[CanonicalPositionState] = None
        self._state_lock = threading.RLock()
        self._subscribers: Dict[str, List[Dict[str, Any]]] = {
            "on_snapshot": [],
            "on_field_change": [],
            "on_conflict": [],
        }
        self._snapshot_history: List[CanonicalPositionState] = []
        self._max_history_size = 100
        self._change_audit: List[Dict[str, Any]] = []
        self._snapshot_in_progress = False
        self._repair_in_progress = False
        self._snapshot_lock_timeout_ms = 500.0
        self._repair_validation_timeout_ms = 500.0
        self._atomic_adoption_window_ms = 50.0
        self._position_snapshot_dedup_window_ms = 50.0
        self._last_snapshot_event_time = 0.0
        self._pending_snapshot_event: Optional[Dict[str, Any]] = None
        self._event_batch_lock = threading.Lock()
        
        # Metrics
        self.canonical_pos_write_attempts = 0
        self.canonical_pos_drift_events = 0
        self.snapshot_repair_collision_waits = 0
        self.position_snapshot_dedup_count = 0
        self._drift_events_window_start = time.time()
        self._drift_events_in_window = 0

    def acquire_write_lock(
        self,
        subsystem: str,
        fields: Optional[Set[str]] = None,
        timeout_ms: float = 1000.0,
    ) -> Optional[WriteAuthorityLock]:
        """
        Acquire exclusive write lock for specific fields.

        Args:
            subsystem: Requesting subsystem identifier
            fields: Set of field names to acquire lock for
            timeout_ms: Maximum wait time in milliseconds

        Returns:
            WriteAuthorityLock token if acquired, None if timeout
        """
        if fields is None:
            fields = set()

        # Validate authority
        authorized_fields = self._get_authorized_fields(subsystem)
        if fields and not fields.issubset(authorized_fields):
            raise ValueError(
                f"Subsystem '{subsystem}' not authorized for fields: "
                f"{fields - authorized_fields}"
            )

        # Acquire lock with timeout
        timeout_sec = timeout_ms / 1000.0
        acquired = self._state_lock.acquire(timeout=timeout_sec)

        if not acquired:
            return None

        lock_token = WriteAuthorityLock(
            subsystem=subsystem,
            fields=frozenset(fields or authorized_fields),
            timestamp=time.time(),
            manager=self,
        )

        self._log_audit(
            event="lock_acquired",
            subsystem=subsystem,
            fields=list(fields) if fields else [],
            timestamp=time.time(),
        )

        return lock_token

    def acquire_write_lock_snapshot(
        self,
        subsystem: str = "snapshot_handler",
        timeout_ms: float = 500.0,
    ) -> Optional[WriteAuthorityLock]:
        """
        Acquire write lock FOR SNAPSHOT ADOPTION ONLY.
        Special semantics: short-lived exclusive lock (50ms).
        """
        if self._repair_in_progress:
            # Back off - let repair complete first
            time.sleep(0.01)
            return self.acquire_write_lock_snapshot(subsystem, timeout_ms)

        # Set flag BEFORE acquiring lock
        self._snapshot_in_progress = True

        try:
            timeout_sec = timeout_ms / 1000.0
            acquired = self._state_lock.acquire(timeout=timeout_sec)

            if not acquired:
                self._snapshot_in_progress = False
                return None

            return WriteAuthorityLock(
                subsystem=subsystem,
                fields=frozenset(
                    {
                        "quantity",
                        "entry_price",
                        "entry_stop",
                        "entry_target",
                        "status_flags",
                    }
                ),
                timestamp=time.time(),
                manager=self,
                lock_type="snapshot_exclusive",
            )

        except Exception as e:
            self._snapshot_in_progress = False
            raise

    def acquire_read_lock_repair(
        self,
        subsystem: str = "protection_repair",
        timeout_ms: float = 300.0,
    ) -> Optional["ReadAuthorityLock"]:
        """
        Acquire read lock for protection repair validation.
        Non-exclusive, shared read lock.
        """
        start_time = time.time()
        timeout_sec = timeout_ms / 1000.0
        collision_detected = False

        # Spin-wait while snapshot in progress
        while self._snapshot_in_progress:
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                raise TimeoutError(
                    f"Repair validation timeout: snapshot_in_progress "
                    f"for {elapsed*1000:.0f}ms"
                )
            collision_detected = True
            self.snapshot_repair_collision_waits += 1
            time.sleep(0.005)  # 5ms poll interval

        if collision_detected:
            logger.info(
                "[SNAPSHOT_REPAIR_COLLISION] snapshot_in_progress detected, "
                f"waited {(time.time() - start_time)*1000:.1f}ms"
            )

        # Set flag to signal snapshot handler to wait
        self._repair_in_progress = True

        # NO LOCK NEEDED - read is non-blocking
        return ReadAuthorityLock(
            subsystem=subsystem,
            fields=frozenset(
                {"quantity", "entry_price", "entry_stop", "entry_target", "status_flags"}
            ),
            timestamp=time.time(),
            manager=self,
        )

    def create_snapshot(
        self,
        lock: WriteAuthorityLock,
        **field_updates: Any,
    ) -> CanonicalPositionState:
        """
        Create new immutable snapshot with updated fields.

        Args:
            lock: WriteAuthorityLock from acquire_write_lock()
            **field_updates: Fields to update

        Returns:
            New CanonicalPositionState instance

        Raises:
            ValueError: If fields not authorized for lock holder
            RuntimeError: If no current state exists
        """
        if not isinstance(lock, WriteAuthorityLock):
            raise TypeError("lock must be WriteAuthorityLock")

        if self._current_state is None:
            raise RuntimeError("Cannot create snapshot: no current state")

        # Validate authorized fields
        for field in field_updates.keys():
            if field not in lock.fields:
                raise ValueError(
                    f"Field '{field}' not in authorized lock fields: {lock.fields}"
                )

        # Build new snapshot
        snapshot_data = {
            "quantity": field_updates.get("quantity", self._current_state.quantity),
            "entry_price": field_updates.get("entry_price", self._current_state.entry_price),
            "entry_stop": field_updates.get("entry_stop", self._current_state.entry_stop),
            "entry_target": field_updates.get("entry_target", self._current_state.entry_target),
            "pnl": field_updates.get("pnl", self._current_state.pnl),
            "version": self._current_state.version + 1,
            "status_flags": field_updates.get("status_flags", self._current_state.status_flags),
            "metadata": {
                **dict(self._current_state.metadata),
                "created_by": lock.subsystem,
                "parent_version": self._current_state.version,
            },
        }

        new_snapshot = CanonicalPositionState(**snapshot_data)

        # Log audit
        self._log_audit(
            event="snapshot_created",
            subsystem=lock.subsystem,
            version=new_snapshot.version,
            changes=field_updates,
            timestamp=time.time(),
        )

        return new_snapshot

    def adopt_snapshot_atomic(
        self,
        lock: WriteAuthorityLock,
        new_snapshot: CanonicalPositionState,
    ) -> bool:
        """
        Atomically adopt new snapshot as current state.
        Validates consistency before commitment.
        """
        start_time = time.time()

        try:
            # Validate snapshot consistency
            is_valid, errors = self._validate_snapshot(new_snapshot)
            if not is_valid:
                self._log_audit(
                    event="snapshot_validation_failed",
                    subsystem=lock.subsystem,
                    version=new_snapshot.version,
                    errors=errors,
                    timestamp=time.time(),
                )
                return False

            # Compute diffs for change notifications
            diffs = (
                self._current_state.compare_with(new_snapshot)
                if self._current_state
                else {}
            )

            # === ATOMIC WRITE POINT ===
            self._current_state = new_snapshot
            # ===========================

            # Maintain history (FIFO with max size)
            self._snapshot_history.append(new_snapshot)
            if len(self._snapshot_history) > self._max_history_size:
                self._snapshot_history.pop(0)

            # CRITICAL: Clear flag BEFORE releasing lock
            self._snapshot_in_progress = False

            # Release lock immediately after flag clear
            self.release_write_lock(lock)

            # Emit notifications (no longer under lock)
            self._emit_batched_snapshot_event(
                {
                    "version": new_snapshot.version,
                    "diffs": diffs,
                    "source": lock.subsystem,
                    "timestamp": time.time(),
                }
            )

            elapsed = (time.time() - start_time) * 1000
            if elapsed > self._atomic_adoption_window_ms:
                logger.warning(
                    f"Snapshot adoption took {elapsed:.1f}ms "
                    f"(target: {self._atomic_adoption_window_ms}ms)"
                )

            self._log_audit(
                event="snapshot_adopted",
                subsystem=lock.subsystem,
                version=new_snapshot.version,
                diffs=list(diffs.keys()),
                elapsed_ms=elapsed,
                timestamp=time.time(),
            )

            return True

        except Exception as e:
            self._snapshot_in_progress = False
            if lock:
                try:
                    self.release_write_lock(lock)
                except:
                    pass
            self._log_audit(
                event="snapshot_adoption_error",
                subsystem=lock.subsystem,
                error=str(e),
                timestamp=time.time(),
            )
            raise

    def release_write_lock(self, lock: WriteAuthorityLock) -> bool:
        """Release write lock and allow other subsystems to acquire."""
        if not isinstance(lock, WriteAuthorityLock):
            return False

        try:
            self._state_lock.release()

            self._log_audit(
                event="lock_released",
                subsystem=lock.subsystem,
                timestamp=time.time(),
            )

            return True
        except:
            return False

    def release_read_lock(self, lock: "ReadAuthorityLock") -> bool:
        """Release read lock and allow snapshot handler to proceed."""
        if not isinstance(lock, ReadAuthorityLock):
            return False

        try:
            self._repair_in_progress = False
            return True
        except:
            return False

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], None],
        fields: Optional[Set[str]] = None,
    ) -> str:
        """
        Subscribe to position state change notifications.

        Args:
            event_type: 'on_snapshot', 'on_field_change', 'on_conflict'
            callback: Callable(event_data: Dict) -> None
            fields: Optional set of field names to filter on

        Returns:
            subscription_id for later unsubscribe
        """
        if event_type not in self._subscribers:
            raise ValueError(f"Unknown event type: {event_type}")

        subscription_id = str(uuid.uuid4())

        # Wrap callback with field filtering
        if fields:
            original_callback = callback

            def filtered_callback(event_data: Dict[str, Any]) -> None:
                if event_data.get("diffs"):
                    filtered_diffs = {
                        k: v
                        for k, v in event_data["diffs"].items()
                        if k in fields
                    }
                    if filtered_diffs:
                        original_callback({**event_data, "diffs": filtered_diffs})

            callback = filtered_callback

        self._subscribers[event_type].append(
            {
                "id": subscription_id,
                "callback": callback,
            }
        )

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from notifications."""
        for event_list in self._subscribers.values():
            for i, sub in enumerate(event_list):
                if sub["id"] == subscription_id:
                    event_list.pop(i)
                    return True
        return False

    def get_current_state(self) -> Optional[CanonicalPositionState]:
        """Get current snapshot (thread-safe read)."""
        return self._current_state

    def record_drift(self, field: str, old_value: Any, new_value: Any, source_line: int, source_func: str) -> None:
        """Record a drift event between old and canonical state."""
        self.canonical_pos_drift_events += 1
        self._drift_events_in_window += 1

        # Check if drift exceeds threshold (>10 in 60 seconds)
        now = time.time()
        if now - self._drift_events_window_start > 60.0:
            self._drift_events_window_start = now
            self._drift_events_in_window = 1
        else:
            if self._drift_events_in_window > 10:
                logger.error(
                    f"[CANONICAL_DRIFT_ALERT] Drift events exceeded 10 in 60s window: "
                    f"{self._drift_events_in_window} events. "
                    f"Consider rollback: canonical_position_state_enabled = False"
                )

        logger.info(
            f"[CANONICAL_DRIFT] field={field} old={old_value} new={new_value} "
            f"at={source_line} source={source_func} total_drifts={self.canonical_pos_drift_events}"
        )

    def _validate_snapshot(
        self,
        snapshot: CanonicalPositionState,
    ) -> Tuple[bool, List[str]]:
        """
        Validate snapshot for consistency violations.

        Returns:
            (is_valid: bool, error_messages: List[str])
        """
        errors = []

        # Check for logical inconsistencies
        if snapshot.quantity < 0:
            errors.append(f"Negative quantity: {snapshot.quantity}")

        if snapshot.entry_price < 0:
            errors.append(f"Negative entry_price: {snapshot.entry_price}")

        # Stop must be different from entry and target
        if snapshot.entry_stop is not None:
            if snapshot.entry_stop == snapshot.entry_price:
                errors.append("entry_stop cannot equal entry_price")

        # Target must be different from entry and stop
        if snapshot.entry_target is not None:
            if snapshot.entry_target == snapshot.entry_price:
                errors.append("entry_target cannot equal entry_price")

        # Version must increment
        if self._current_state and snapshot.version <= self._current_state.version:
            errors.append(
                f"Version must increment: {self._current_state.version} -> "
                f"{snapshot.version}"
            )

        return (len(errors) == 0, errors)

    def _emit_batched_snapshot_event(self, event_data: Dict[str, Any]) -> None:
        """
        Emit SINGLE position_snapshot event per adoption cycle.
        Deduplicates events within window_ms.
        """
        with self._event_batch_lock:
            now = time.time()
            time_since_last_event = (now - self._last_snapshot_event_time) * 1000

            if time_since_last_event >= self._position_snapshot_dedup_window_ms:
                # Outside dedup window - emit immediately
                self._emit_notification("on_snapshot", event_data)
                self._last_snapshot_event_time = now
                self._pending_snapshot_event = None
            else:
                # Within dedup window - queue for later
                self.position_snapshot_dedup_count += 1
                self._pending_snapshot_event = event_data

                logger.debug(
                    f"[SNAPSHOT_DEDUP] Deduplicated event within {time_since_last_event:.1f}ms window"
                )

                # Schedule deferred emission
                def deferred_emit() -> None:
                    sleep_time = (
                        self._position_snapshot_dedup_window_ms - time_since_last_event
                    ) / 1000.0
                    time.sleep(sleep_time)
                    with self._event_batch_lock:
                        if self._pending_snapshot_event:
                            self._emit_notification(
                                "on_snapshot", self._pending_snapshot_event
                            )
                            self._last_snapshot_event_time = time.time()
                            self._pending_snapshot_event = None

                thread = threading.Thread(target=deferred_emit, daemon=True)
                thread.start()

    def _emit_notification(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Emit notification to all subscribers (async)."""
        callbacks = [sub["callback"] for sub in self._subscribers.get(event_type, [])]
        for callback in callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")

    def _log_audit(self, **event_data: Any) -> None:
        """Log audit entry for compliance and debugging."""
        entry = {
            "timestamp": event_data.get("timestamp", time.time()),
            **event_data,
        }
        self._change_audit.append(entry)

        # Trim audit log to reasonable size
        if len(self._change_audit) > 10000:
            self._change_audit = self._change_audit[-5000:]

    def _get_authorized_fields(self, subsystem: str) -> Set[str]:
        """Get set of fields authorized for write by subsystem."""
        authority_map = {
            "snapshot_handler": {
                "quantity",
                "entry_price",
                "entry_stop",
                "entry_target",
                "status_flags",
            },
            "protection_repair": {"entry_stop", "entry_target", "status_flags"},
            "order_executor": {"quantity", "entry_price"},
            "pnl_calculator": {"pnl"},
            "gate_validator": set(),  # Read-only
        }
        return authority_map.get(subsystem, set())


class ReadAuthorityLock:
    """Token representing read authority (non-exclusive)."""

    def __init__(
        self,
        subsystem: str,
        fields: FrozenSet[str],
        timestamp: float,
        manager: CanonicalPositionStateManager,
    ):
        self.subsystem = subsystem
        self.fields = fields
        self.timestamp = timestamp
        self.manager = manager

    def __repr__(self) -> str:
        return f"ReadAuthorityLock({self.subsystem}, fields={self.fields})"


# Python 3.9+ types module
try:
    import types
except ImportError:
    types = None
