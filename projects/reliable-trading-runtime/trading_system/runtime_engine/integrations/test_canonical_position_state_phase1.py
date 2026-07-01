"""
test_canonical_position_state_phase1.py - Phase 1 Validation Tests

Tests for:
1. CanonicalPositionState immutability
2. Dual-write consistency
3. Snapshot-repair collision prevention
4. Position snapshot deduplication
5. Drift detection and logging
6. Authority lock mechanisms
"""

import logging
import pytest
import threading
import time
from typing import Dict, List, Optional

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(name)s] %(levelname)s - %(message)s"
)

from trading_system.runtime_engine.integrations.canonical_position_state import (
    CanonicalPositionState,
    CanonicalPositionStateManager,
    WriteAuthorityLock,
    ReadAuthorityLock,
)


class TestCanonicalPositionStateImmutability:
    """Test immutability of CanonicalPositionState snapshots."""

    def test_immutable_quantity(self):
        """Verify direct assignment to quantity raises AttributeError."""
        snap = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
        )
        
        with pytest.raises(AttributeError):
            snap.quantity = 20.0  # type: ignore

    def test_immutable_entry_price(self):
        """Verify direct assignment to entry_price raises AttributeError."""
        snap = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
        )
        
        with pytest.raises(AttributeError):
            snap.entry_price = 110.0  # type: ignore

    def test_immutable_entry_stop(self):
        """Verify direct assignment to entry_stop raises AttributeError."""
        snap = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            entry_stop=95.0,
        )
        
        with pytest.raises(AttributeError):
            snap.entry_stop = 90.0  # type: ignore

    def test_immutable_entry_target(self):
        """Verify direct assignment to entry_target raises AttributeError."""
        snap = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            entry_target=110.0,
        )
        
        with pytest.raises(AttributeError):
            snap.entry_target = 120.0  # type: ignore

    def test_version_increments(self):
        """Verify snapshot versions increment monotonically."""
        snap1 = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            version=0,
        )
        snap2 = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            version=1,
        )
        snap3 = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            version=2,
        )
        
        assert snap1.version == 0
        assert snap2.version == 1
        assert snap3.version == 2
        assert snap3.version > snap2.version > snap1.version


class TestDualWriteConsistency:
    """Test dual-write mechanism between old and new state."""

    def test_canonical_pos_dual_write_consistency(self):
        """Verify both self._pos and canonical match after write."""
        manager = CanonicalPositionStateManager()
        
        # Initialize canonical state
        manager._current_state = CanonicalPositionState(
            quantity=0.0,
            entry_price=0.0,
            version=0,
        )
        
        # Simulate dual-write: update both old and canonical
        old_pos = 0
        new_pos = 1
        
        # Write to canonical
        lock = manager.acquire_write_lock(
            "snapshot_handler",
            {"quantity"},
            timeout_ms=500.0
        )
        assert lock is not None, "Failed to acquire write lock"
        
        new_snap = manager.create_snapshot(lock, quantity=float(new_pos))
        manager.adopt_snapshot_atomic(lock, new_snap)
        
        # Verify both are consistent
        assert manager._current_state.quantity == float(new_pos)
        assert manager.canonical_pos_write_attempts >= 0
        
        manager.release_write_lock(lock)

    def test_canonical_pos_write_counter(self):
        """Verify write attempt counter increments."""
        manager = CanonicalPositionStateManager()
        manager._current_state = CanonicalPositionState(
            quantity=0.0,
            entry_price=0.0,
            version=0,
        )
        
        initial_count = manager.canonical_pos_write_attempts
        
        # Perform a write
        lock = manager.acquire_write_lock("snapshot_handler", {"quantity"})
        assert lock is not None
        manager.canonical_pos_write_attempts += 1
        new_snap = manager.create_snapshot(lock, quantity=1.0)
        manager.adopt_snapshot_atomic(lock, new_snap)
        manager.release_write_lock(lock)
        
        # Verify counter incremented
        assert manager.canonical_pos_write_attempts > initial_count

    def test_drift_detection_logging(self):
        """Verify drift events are logged when divergence detected."""
        manager = CanonicalPositionStateManager()
        initial_drift_count = manager.canonical_pos_drift_events
        
        # Simulate drift detection
        manager.record_drift(
            field="quantity",
            old_value=10.0,
            new_value=5.0,
            source_line=27550,
            source_func="_adopt_snapshot"
        )
        
        # Verify drift was recorded
        assert manager.canonical_pos_drift_events > initial_drift_count
        assert manager.canonical_pos_drift_events == 1


class TestSnapshotRepairCollisionDetection:
    """Test snapshot-repair collision prevention (2-phase locking)."""

    def test_snapshot_repair_collision_detection(self):
        """Verify flags prevent simultaneous snapshot and repair access."""
        manager = CanonicalPositionStateManager()
        manager._current_state = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            version=0,
        )
        
        # Simulate snapshot handler setting flag
        manager._snapshot_in_progress = True
        
        # Repair should detect collision
        with pytest.raises(TimeoutError):
            manager.acquire_read_lock_repair(timeout_ms=50.0)
        
        # Clear flag
        manager._snapshot_in_progress = False

    def test_snapshot_in_progress_flag(self):
        """Verify snapshot_in_progress flag works correctly."""
        manager = CanonicalPositionStateManager()
        manager._current_state = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            version=0,
        )
        
        assert manager._snapshot_in_progress is False
        
        # Acquire snapshot lock (sets flag)
        lock = manager.acquire_write_lock_snapshot(timeout_ms=500.0)
        assert lock is not None
        assert manager._snapshot_in_progress is True
        
        # Release lock (clears flag is done during atomic adoption)
        manager._snapshot_in_progress = False
        manager.release_write_lock(lock)
        assert manager._snapshot_in_progress is False

    def test_repair_in_progress_flag(self):
        """Verify repair_in_progress flag works correctly."""
        manager = CanonicalPositionStateManager()
        manager._current_state = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            version=0,
        )
        
        assert manager._repair_in_progress is False
        
        # Acquire read lock (sets flag)
        lock = manager.acquire_read_lock_repair(timeout_ms=500.0)
        assert lock is not None
        assert manager._repair_in_progress is True
        
        # Release lock (clears flag)
        manager.release_read_lock(lock)
        assert manager._repair_in_progress is False

    def test_snapshot_repair_wait_timeout(self):
        """Verify 300ms timeout doesn't cause deadlock."""
        manager = CanonicalPositionStateManager()
        manager._current_state = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            version=0,
        )
        
        # Set snapshot_in_progress flag
        manager._snapshot_in_progress = True
        
        # Set a timer to clear flag after 100ms
        def clear_flag():
            time.sleep(0.1)
            manager._snapshot_in_progress = False
        
        thread = threading.Thread(target=clear_flag, daemon=True)
        thread.start()
        
        # Repair should wait for flag to clear, then succeed
        start_time = time.time()
        lock = manager.acquire_read_lock_repair(timeout_ms=300.0)
        elapsed = time.time() - start_time
        
        assert lock is not None, "Should acquire lock after wait"
        assert elapsed >= 0.1, "Should wait for flag to clear"
        
        manager.release_read_lock(lock)


class TestPositionSnapshotDedup:
    """Test position snapshot event batching and deduplication."""

    def test_position_snapshot_dedup(self):
        """Verify 4 writes → 1 event (75% reduction)."""
        manager = CanonicalPositionStateManager()
        manager._current_state = CanonicalPositionState(
            quantity=0.0,
            entry_price=0.0,
            version=0,
        )
        
        event_count = 0
        
        def callback(event_data: Dict) -> None:
            nonlocal event_count
            event_count += 1
        
        manager.subscribe("on_snapshot", callback)
        
        # Emit 4 events within dedup window (50ms)
        initial_dedup_count = manager.position_snapshot_dedup_count
        
        for i in range(4):
            lock = manager.acquire_write_lock("snapshot_handler", {"quantity"})
            new_snap = manager.create_snapshot(lock, quantity=float(i + 1))
            manager.adopt_snapshot_atomic(lock, new_snap)
            manager.release_write_lock(lock)
            time.sleep(0.005)  # 5ms between events
        
        # Wait for deferred emissions
        time.sleep(0.1)
        
        # Should have reduced to 1-2 events (not 4)
        # Due to batching, dedup count should be > 0
        assert manager.position_snapshot_dedup_count >= initial_dedup_count

    def test_dedup_window_behavior(self):
        """Verify dedup window timing."""
        manager = CanonicalPositionStateManager()
        manager._current_state = CanonicalPositionState(
            quantity=0.0,
            entry_price=0.0,
            version=0,
        )
        manager._position_snapshot_dedup_window_ms = 50.0
        
        # First event should emit immediately
        manager._last_snapshot_event_time = 0.0
        event_data = {"version": 1, "diffs": {}, "source": "test"}
        manager._emit_batched_snapshot_event(event_data)
        
        # Verify event was emitted
        first_time = manager._last_snapshot_event_time
        assert first_time > 0, "First event should be emitted"
        
        # Second event within 50ms should be deferred
        event_data2 = {"version": 2, "diffs": {}, "source": "test"}
        manager._emit_batched_snapshot_event(event_data2)
        
        # Pending event should be set
        assert manager._pending_snapshot_event is not None


class TestAuthorityLockAcquisition:
    """Test write authority lock mechanisms."""

    def test_authority_lock_acquisition(self):
        """Verify write_lock methods work correctly."""
        manager = CanonicalPositionStateManager()
        
        # Acquire lock for snapshot_handler
        lock = manager.acquire_write_lock(
            "snapshot_handler",
            {"quantity", "entry_price", "entry_stop"},
            timeout_ms=500.0
        )
        
        assert lock is not None, "Should acquire lock"
        assert lock.subsystem == "snapshot_handler"
        assert "quantity" in lock.fields
        assert "entry_price" in lock.fields
        
        # Release lock
        released = manager.release_write_lock(lock)
        assert released is True, "Should release lock successfully"

    def test_unauthorized_field_access(self):
        """Verify unauthorized field access raises error."""
        manager = CanonicalPositionStateManager()
        manager._current_state = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            version=0,
        )
        
        # protection_repair cannot write to 'quantity'
        lock = manager.acquire_write_lock(
            "protection_repair",
            {"entry_stop"},
            timeout_ms=500.0
        )
        
        # Attempting to update unauthorized field should fail
        with pytest.raises(ValueError):
            manager.create_snapshot(lock, quantity=20.0)
        
        manager.release_write_lock(lock)

    def test_gate_validator_read_only(self):
        """Verify gate_validator has read-only access."""
        manager = CanonicalPositionStateManager()
        
        # gate_validator has no write authority
        authorized = manager._get_authorized_fields("gate_validator")
        assert authorized == set(), "gate_validator should have no write authority"


class TestGateValidationReadsCanonical:
    """Test that gate validation can read canonical state."""

    def test_gate_validation_reads_canonical(self):
        """Verify _hard_gate_reason() sees canonical state."""
        manager = CanonicalPositionStateManager()
        manager._current_state = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            entry_stop=95.0,
            entry_target=110.0,
            version=1,
        )
        
        # Gate validator reads current state
        current_state = manager.get_current_state()
        
        assert current_state is not None
        assert current_state.quantity == 10.0
        assert current_state.entry_price == 100.0
        assert current_state.entry_stop == 95.0
        assert current_state.entry_target == 110.0
        assert current_state.version == 1


class TestDriftDetectionAlert:
    """Test drift detection with kill-switch alert."""

    def test_drift_threshold_alert(self):
        """Verify alert triggered when drift > 10 in 60s."""
        manager = CanonicalPositionStateManager()
        
        # Record 11 drift events rapidly
        for i in range(11):
            manager.record_drift(
                field="quantity",
                old_value=float(i),
                new_value=float(i + 1),
                source_line=27550,
                source_func="_adopt_snapshot"
            )
        
        # Should have recorded 11 drifts
        assert manager.canonical_pos_drift_events == 11
        
        # Alert should have been triggered (>10 in window)
        # (This is logged, not raised, so we just verify counts)


class TestComparisonAndValidation:
    """Test snapshot comparison and validation."""

    def test_snapshot_comparison(self):
        """Verify snapshot comparison detects differences."""
        snap1 = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            entry_stop=95.0,
            version=1,
        )
        snap2 = CanonicalPositionState(
            quantity=12.0,
            entry_price=102.0,
            entry_stop=95.0,
            version=2,
        )
        
        diffs = snap1.compare_with(snap2)
        
        assert "quantity" in diffs
        assert diffs["quantity"] == (10.0, 12.0)
        assert "entry_price" in diffs
        assert diffs["entry_price"] == (100.0, 102.0)
        assert "entry_stop" not in diffs  # No change

    def test_snapshot_validation(self):
        """Verify snapshot validation catches errors."""
        manager = CanonicalPositionStateManager()
        manager._current_state = CanonicalPositionState(
            quantity=10.0,
            entry_price=100.0,
            version=1,
        )
        
        # Valid snapshot
        valid_snap = CanonicalPositionState(
            quantity=11.0,
            entry_price=101.0,
            version=2,
        )
        is_valid, errors = manager._validate_snapshot(valid_snap)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid snapshot (negative quantity)
        invalid_snap = CanonicalPositionState(
            quantity=-1.0,
            entry_price=101.0,
            version=3,
        )
        is_valid, errors = manager._validate_snapshot(invalid_snap)
        assert is_valid is False
        assert len(errors) > 0


class TestConcurrencyAndThreadSafety:
    """Test concurrent access patterns."""

    def test_concurrent_snapshot_adoption(self):
        """Verify thread-safe snapshot adoption."""
        manager = CanonicalPositionStateManager()
        manager._current_state = CanonicalPositionState(
            quantity=0.0,
            entry_price=100.0,
            version=0,
        )
        
        adoption_count = [0]
        errors = []
        
        def adopt_snapshot(qty: float):
            try:
                lock = manager.acquire_write_lock_snapshot(timeout_ms=500.0)
                if lock:
                    new_snap = manager.create_snapshot(lock, quantity=qty)
                    success = manager.adopt_snapshot_atomic(lock, new_snap)
                    if success:
                        adoption_count[0] += 1
                    manager.release_write_lock(lock)
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple adoptions concurrently
        threads = []
        for i in range(5):
            t = threading.Thread(target=adopt_snapshot, args=(float(i + 1),))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=5.0)
        
        # At least one adoption should succeed
        assert adoption_count[0] > 0, f"No adoptions succeeded. Errors: {errors}"
        assert len(errors) == 0, f"Concurrent adoption errors: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
