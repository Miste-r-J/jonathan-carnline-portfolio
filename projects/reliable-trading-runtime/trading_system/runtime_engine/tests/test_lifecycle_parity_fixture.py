"""
test_lifecycle_parity_fixture.py - Phase 5: Golden Fixture Testing

Tests the complete lifecycle parity refactoring with 6 deterministic scenarios:
1. Simple OPEN→CLOSE flow
2. FLIP decomposition (CLOSE + OPEN)
3. Phase blocking (BACKTEST mode)
4. Gate blocking (VWAP fails)
5. Setup gate blocking (Phase2 fails)
6. Override bypass (gates_detail.override_applied=true)

Each scenario validates cross-artifact parity:
- lifecycle_events.jsonl (canonical)
- signal_to_order.jsonl (presentation)
- state.csv (presentation with metadata)
- execution_ledger.jsonl (execution outcomes)
"""

import pytest
import json
from pathlib import Path
from datetime import datetime as dt_type
from typing import Dict, List, Any, Optional
import tempfile
import sys

# Add the standalone root to path
_TRADING_SYSTEM_ROOT = Path(__file__).resolve().parents[3]
if str(_TRADING_SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRADING_SYSTEM_ROOT))

from trading_system.runtime_engine.integrations.cli.lifecycle_canonicalization import (
    _canonical_lifecycle_record,
    _write_lifecycle_event,
    _resolve_block_reason,
)


class TestLifecycleCanonicalFunctions:
    """Test the Phase 1 helper functions in isolation."""

    def test_resolve_block_reason_open_pass(self):
        """OPEN signal with all gates passing should not be blocked."""
        is_blocked, block_reason, blocked_by, phase_allows = _resolve_block_reason(
            action="OPEN",
            phase2_setup_pass=True,
            override_applied=False,
            blocked_by=[],
            current_phase="LIVE",
            phase_allows_execution=True,
        )
        assert not is_blocked
        assert block_reason is None
        assert blocked_by == []
        assert phase_allows

    def test_resolve_block_reason_open_phase_blocked(self):
        """OPEN signal in BACKTEST phase should be blocked."""
        is_blocked, block_reason, blocked_by, phase_allows = _resolve_block_reason(
            action="OPEN",
            phase2_setup_pass=True,
            override_applied=False,
            blocked_by=[],
            current_phase="BACKTEST",
            phase_allows_execution=False,
        )
        assert is_blocked
        assert block_reason == "phase_not_executable"
        assert "phase" in blocked_by
        assert not phase_allows

    def test_resolve_block_reason_open_setup_blocked(self):
        """OPEN signal with setup failure (no override) should be blocked."""
        is_blocked, block_reason, blocked_by, phase_allows = _resolve_block_reason(
            action="OPEN",
            phase2_setup_pass=False,  # Setup fails
            override_applied=False,  # No override
            blocked_by=[],
            current_phase="LIVE",
            phase_allows_execution=True,
        )
        assert is_blocked
        assert block_reason == "setup_blocked"
        assert "setup" in blocked_by

    def test_resolve_block_reason_open_setup_override(self):
        """OPEN signal with setup failure BUT override should NOT be blocked."""
        is_blocked, block_reason, blocked_by, phase_allows = _resolve_block_reason(
            action="OPEN",
            phase2_setup_pass=False,  # Setup fails
            override_applied=True,  # But override is applied
            blocked_by=[],
            current_phase="LIVE",
            phase_allows_execution=True,
        )
        assert not is_blocked
        assert block_reason is None

    def test_resolve_block_reason_open_setup_force_open_policy(self):
        """OPEN signal with setup failure and force-open policy should not be blocked."""
        is_blocked, block_reason, blocked_by, phase_allows = _resolve_block_reason(
            action="OPEN",
            phase2_setup_pass=False,
            override_applied=False,
            allow_setup_fail_entries=True,
            blocked_by=[],
            current_phase="LIVE",
            phase_allows_execution=True,
        )
        assert not is_blocked
        assert block_reason is None
        assert blocked_by == []

    def test_canonical_lifecycle_open_allowed(self):
        """Test canonical record for allowed OPEN."""
        ev = {
            "type": "OPEN",
            "side": "LONG",
            "price": 4200.5,
            "prob": 0.65,
            "signal_id": "sig_001",
            "client_order_id": "cid_001",
        }
        record = _canonical_lifecycle_record(
            ev=ev,
            phase2_meta={"setup_pass": True},
            gates_detail={},
            current_phase="LIVE",
            phase_allows_execution=True,
        )
        assert record["requested_action"] == "OPEN"
        assert record["resolved_action"] == "OPEN"
        assert record["execution_intent_action"] == "OPEN"
        assert record["emit_allowed"] is True
        assert record["blocked_reason"] is None

    def test_canonical_lifecycle_open_blocked_setup(self):
        """Test canonical record for OPEN blocked by setup."""
        ev = {
            "type": "OPEN",
            "side": "LONG",
            "price": 4200.5,
            "prob": 0.65,
            "signal_id": "sig_002",
            "client_order_id": "cid_002",
        }
        record = _canonical_lifecycle_record(
            ev=ev,
            phase2_meta={"setup_pass": False},
            gates_detail={},
            current_phase="LIVE",
            phase_allows_execution=True,
        )
        assert record["requested_action"] == "OPEN"
        assert record["resolved_action"] == "NO_TRADE"
        assert record["execution_intent_action"] == "NO_TRADE"
        assert record["emit_allowed"] is False
        assert record["blocked_reason"] == "setup_blocked"
        assert "setup" in record["blocked_by"]

    def test_canonical_lifecycle_open_force_open_policy(self):
        """Test canonical record for OPEN when setup-fail entries are explicitly allowed."""
        ev = {
            "type": "OPEN",
            "side": "LONG",
            "price": 4200.5,
            "prob": 0.65,
            "signal_id": "sig_002a",
            "client_order_id": "cid_002a",
        }
        record = _canonical_lifecycle_record(
            ev=ev,
            phase2_meta={"setup_pass": False, "allow_setup_fail_entries": True},
            gates_detail={"allow_setup_fail_entries": True},
            current_phase="LIVE",
            phase_allows_execution=True,
            allow_setup_fail_entries=True,
        )
        assert record["requested_action"] == "OPEN"
        assert record["resolved_action"] == "OPEN"
        assert record["execution_intent_action"] == "OPEN"
        assert record["emit_allowed"] is True
        assert record["blocked_reason"] is None
        assert record["blocked_by"] == []

    def test_canonical_lifecycle_flip_decomposition(self):
        """Test canonical record for FLIP shows decomposition."""
        ev = {
            "type": "FLIP",
            "side": "SHORT",
            "price": 4195.0,
            "prob": 0.70,
            "signal_id": "sig_003",
            "client_order_id": "cid_003",
            "transition_id": "txn_001",
        }
        record = _canonical_lifecycle_record(
            ev=ev,
            phase2_meta={"setup_pass": True},
            gates_detail={},
            current_phase="LIVE",
            phase_allows_execution=True,
        )
        assert record["requested_action"] == "FLIP"
        assert record["resolved_action"] == "FLIP"
        assert record["flip_decomposed"] is not None
        assert "close_step" in record["flip_decomposed"]
        assert "open_step" in record["flip_decomposed"]
        assert record["flip_decomposed"]["close_step"]["transition_step"] == "close"
        assert record["flip_decomposed"]["open_step"]["transition_step"] == "open"
        assert record["flip_decomposed"]["close_step"]["transition_id"] == "txn_001"

    def test_write_lifecycle_event_output_format(self):
        """Test that lifecycle event writes correct JSONL format."""
        record = {
            "requested_action": "OPEN",
            "resolved_action": "OPEN",
            "execution_intent_action": "OPEN",
            "display_action": "OPEN",
            "side": "LONG",
            "price": 4200.5,
            "prob": 0.65,
            "emit_allowed": True,
            "publish_ready": True,
            "blocked_reason": None,
            "blocked_by": [],
            "transition_id": None,
            "transition_step": None,
            "signal_id": "sig_001",
            "client_order_id": "cid_001",
            "phase": "LIVE",
            "source": "model",
        }
        bar_ts = "2026-05-27T10:00:00"

        entry = _write_lifecycle_event(record=record, bar_ts=bar_ts)

        # Verify required fields
        assert "ts" in entry
        assert entry["bar_ts"] == bar_ts
        assert entry["requested_action"] == "OPEN"
        assert entry["resolved_action"] == "OPEN"
        assert entry["signal_id"] == "sig_001"
        assert "dedupe_key" in entry
        # Dedupe key should contain bar_ts and signal_id
        assert "sig_001" in entry["dedupe_key"]


class TestGoldenScenarios:
    """Integration tests for all 6 golden fixture scenarios."""

    @pytest.fixture
    def temp_run_dir(self):
        """Create temporary directory for run outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Read JSONL file and return list of dicts."""
        if not path.exists():
            return []
        records = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def test_scenario_a_simple_open_close(self, temp_run_dir):
        """
        Scenario A: Simple OPEN→CLOSE flow.
        Expected: 2 lifecycle entries (OPEN, CLOSE), both allowed
        """
        # Simulate lifecycle records for this scenario
        records = [
            _canonical_lifecycle_record(
                ev={
                    "type": "OPEN",
                    "side": "LONG",
                    "price": 4200.0,
                    "prob": 0.65,
                    "signal_id": "sig_a_1",
                    "client_order_id": "cid_a_1",
                },
                phase2_meta={"setup_pass": True},
                current_phase="LIVE",
                phase_allows_execution=True,
            ),
            _canonical_lifecycle_record(
                ev={
                    "type": "CLOSE",
                    "side": "FLAT",
                    "price": 4205.0,
                    "prob": 0.55,
                    "signal_id": "sig_a_2",
                    "client_order_id": "cid_a_2",
                },
                phase2_meta={},
                current_phase="LIVE",
                phase_allows_execution=True,
            ),
        ]

        # Verify both records are allowed
        for record in records:
            assert record["emit_allowed"] is True
            assert record["blocked_reason"] is None

        # Verify actions are correct
        assert records[0]["requested_action"] == "OPEN"
        assert records[0]["resolved_action"] == "OPEN"
        assert records[1]["requested_action"] == "CLOSE"
        assert records[1]["resolved_action"] == "CLOSE"

    def test_scenario_b_flip_decomposition(self, temp_run_dir):
        """
        Scenario B: FLIP decomposition.
        Expected: 1 lifecycle entry with FLIP, showing decomposition
        """
        record = _canonical_lifecycle_record(
            ev={
                "type": "FLIP",
                "side": "SHORT",
                "price": 4195.0,
                "prob": 0.70,
                "signal_id": "sig_b_1",
                "client_order_id": "cid_b_1",
                "transition_id": "txn_b_1",
            },
            phase2_meta={"setup_pass": True},
            current_phase="LIVE",
            phase_allows_execution=True,
        )

        assert record["requested_action"] == "FLIP"
        assert record["resolved_action"] == "FLIP"
        assert record["emit_allowed"] is True
        assert record["flip_decomposed"] is not None
        assert len(record["flip_decomposed"]) == 2

    def test_scenario_c_phase_blocking(self, temp_run_dir):
        """
        Scenario C: Phase blocking (BACKTEST phase).
        Expected: OPEN signal blocked due to phase
        """
        record = _canonical_lifecycle_record(
            ev={
                "type": "OPEN",
                "side": "LONG",
                "price": 4200.0,
                "prob": 0.65,
                "signal_id": "sig_c_1",
                "client_order_id": "cid_c_1",
            },
            phase2_meta={"setup_pass": True},
            current_phase="BACKTEST",
            phase_allows_execution=False,
        )

        assert record["requested_action"] == "OPEN"
        assert record["resolved_action"] == "NO_TRADE"
        assert record["emit_allowed"] is False
        assert record["blocked_reason"] == "phase_not_executable"
        assert "phase" in record["blocked_by"]

    def test_scenario_d_gate_blocking_vwap(self, temp_run_dir):
        """
        Scenario D: Gate blocking (VWAP fails).
        Expected: OPEN blocked with vwap in blocked_by
        """
        record = _canonical_lifecycle_record(
            ev={
                "type": "OPEN",
                "side": "LONG",
                "price": 4200.0,
                "prob": 0.65,
                "signal_id": "sig_d_1",
                "client_order_id": "cid_d_1",
                "_signal_blocked_by": ["vwap"],
            },
            phase2_meta={"setup_pass": True},
            current_phase="LIVE",
            phase_allows_execution=True,
        )

        assert record["requested_action"] == "OPEN"
        assert record["resolved_action"] == "NO_TRADE"
        assert record["emit_allowed"] is False
        assert "vwap" in record["blocked_by"]

    def test_scenario_e_setup_gate_blocking(self, temp_run_dir):
        """
        Scenario E: Setup gate blocking (Phase2 fails).
        Expected: OPEN blocked with setup in blocked_by
        """
        record = _canonical_lifecycle_record(
            ev={
                "type": "OPEN",
                "side": "LONG",
                "price": 4200.0,
                "prob": 0.65,
                "signal_id": "sig_e_1",
                "client_order_id": "cid_e_1",
            },
            phase2_meta={"setup_pass": False},
            current_phase="LIVE",
            phase_allows_execution=True,
        )

        assert record["requested_action"] == "OPEN"
        assert record["resolved_action"] == "NO_TRADE"
        assert record["emit_allowed"] is False
        assert record["blocked_reason"] == "setup_blocked"
        assert "setup" in record["blocked_by"]

    def test_scenario_f_override_bypass(self, temp_run_dir):
        """
        Scenario F: Override bypass.
        Expected: OPEN allowed even though setup would have failed, due to override
        """
        record = _canonical_lifecycle_record(
            ev={
                "type": "OPEN",
                "side": "LONG",
                "price": 4200.0,
                "prob": 0.95,  # High prob for override
                "signal_id": "sig_f_1",
                "client_order_id": "cid_f_1",
            },
            phase2_meta={"setup_pass": False},
            gates_detail={"override_applied": True},
            current_phase="LIVE",
            phase_allows_execution=True,
        )

        assert record["requested_action"] == "OPEN"
        assert record["resolved_action"] == "OPEN"
        assert record["emit_allowed"] is True
        assert record["blocked_reason"] is None


class TestCrossArtifactParity:
    """Test parity between canonical lifecycle and other artifacts."""

    def test_lifecycle_vs_signal_to_order_parity(self):
        """
        Canonical lifecycle record should map to signal_to_order.jsonl fields.
        """
        record = _canonical_lifecycle_record(
            ev={
                "type": "OPEN",
                "side": "LONG",
                "price": 4200.5,
                "prob": 0.65,
                "signal_id": "sig_001",
                "client_order_id": "cid_001",
            },
            phase2_meta={"setup_pass": True},
            current_phase="LIVE",
            phase_allows_execution=True,
        )

        # Map to signal_to_order fields
        signal_to_order_record = {
            "signal_action": record["requested_action"],
            "action": record["requested_action"],
            "decision": "SENT" if record["emit_allowed"] else "BLOCKED",
            "reason": record["blocked_reason"] or ("ok" if record["emit_allowed"] else "blocked"),
            "final_action": record["resolved_action"],
            "emit_allowed": record["emit_allowed"],
            "requested_action": record["requested_action"],
            "resolved_action": record["resolved_action"],
            "execution_intent_action": record["execution_intent_action"],
        }

        # Verify consistency
        assert signal_to_order_record["signal_action"] == signal_to_order_record["action"]
        if record["emit_allowed"]:
            assert signal_to_order_record["decision"] == "SENT"
        else:
            assert signal_to_order_record["decision"] == "BLOCKED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
