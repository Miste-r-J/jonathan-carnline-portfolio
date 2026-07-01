"""
test_diagnostics.py — Unit and integration tests for the Diagnostics module.

Run with:
    python -m pytest trading_system/runtime_engine/integrations/cli/test_diagnostics.py -v

Integration test (requires tuesday326pt2 run data):
    python -m pytest trading_system/runtime_engine/integrations/cli/test_diagnostics.py -v -m integration
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing diagnostics without installing the package
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from diagnostics import Diagnostics, _safe_float, _ts_to_epoch  # noqa: E402

# Re-export the static method for easier unit testing
def _is_opaque(oid: Any) -> bool:
    return Diagnostics._is_opaque_handle(oid)


# ---------------------------------------------------------------------------
# Helpers — build a minimal mock streamer
# ---------------------------------------------------------------------------

TUESDAY_RUN_DIR = Path(
    r"C:\test-data\paper-run"
)
TUESDAY_RUN_AVAILABLE = TUESDAY_RUN_DIR.exists()

# Real order IDs from tuesday326pt2
REAL_ENTRY_OID = "406099040457"
REAL_STOP_OID = "406099040466"
OPAQUE_ENTRY_HANDLE = "392826f4a8fc49009643f2681d50b753"
OPAQUE_STOP_HANDLE = "41e37e1407594ff98effef8431cf15c1"
ENTRY_PRICE = 6745.5
STOP_PRICE = 6755.5


def _make_streamer(
    *,
    hard_lockout_active: bool = False,
    lockout_code: Optional[str] = None,
    lockout_detail: Optional[dict] = None,
    lockout_sticky: bool = False,
    nt_order_state: Optional[Dict[str, Any]] = None,
    tick_size: float = 0.25,
    max_fill_slippage_ticks: float = 4.0,
    position_state: str = "IN_POSITION_PROTECTED",
    nt_connected: bool = True,
    handshake_ok_ret: bool = True,
    feed_age: Optional[float] = 5.0,
    bar_age_max: float = 90.0,
) -> MagicMock:
    """Build a mock LiveCSVStreamer instance for testing."""
    streamer = MagicMock()
    streamer._hard_lockout_active = hard_lockout_active
    streamer._hard_lockout_code = lockout_code
    streamer._hard_lockout_detail = lockout_detail or {}
    streamer._lockout_sticky = lockout_sticky
    streamer._nt_order_state = nt_order_state or {}
    streamer.tick_size = tick_size
    streamer.max_fill_slippage_ticks = max_fill_slippage_ticks
    streamer.run_id = "test-run-id"
    streamer._effective_bar_age_max_sec = bar_age_max

    # state object
    state_obj = MagicMock()
    state_obj.position_state = position_state
    streamer.state = state_obj

    # NT bridge
    nt_bridge = MagicMock()
    nt_bridge.is_connected = nt_connected
    nt_bridge.handshake_ok.return_value = handshake_ok_ret
    streamer.nt_bridge = nt_bridge

    # Feed health
    streamer._bar_age_guard_seconds.return_value = feed_age

    # Clear lockout — real method stub
    streamer._clear_hard_lockout = MagicMock()
    streamer._log_exec_event = MagicMock()

    return streamer


def _make_open_trade_state(
    entry_ninja_order_id: Any = REAL_ENTRY_OID,
    stop_order_id: Any = REAL_STOP_OID,
    entry_filled: bool = True,
    entry_fill_price: float = ENTRY_PRICE,
    stop_price: float = STOP_PRICE,
    exit_fill_ts: Any = None,
) -> Dict[str, Any]:
    return {
        "entry_ninja_order_id": entry_ninja_order_id,
        "stop_order_id": stop_order_id,
        "entry_filled": entry_filled,
        "entry_fill_price": entry_fill_price,
        "stop_price": stop_price,
        "exit_fill_ts": exit_fill_ts,
        "expected_entry_ref": ENTRY_PRICE,
    }


# ===========================================================================
# Unit Tests: _is_opaque_handle
# ===========================================================================

class TestIsOpaqueHandle:
    def test_uuid_with_dashes_is_opaque(self):
        assert _is_opaque("41e37e14-0759-4ff9-8eff-ef8431cf15c1") is True

    def test_32char_hex_is_opaque(self):
        assert _is_opaque(OPAQUE_ENTRY_HANDLE) is True
        assert _is_opaque(OPAQUE_STOP_HANDLE) is True

    def test_numeric_exchange_id_is_not_opaque(self):
        assert _is_opaque(REAL_ENTRY_OID) is False
        assert _is_opaque(REAL_STOP_OID) is False
        assert _is_opaque("406099040466") is False

    def test_none_is_not_opaque(self):
        assert _is_opaque(None) is False

    def test_empty_string_is_not_opaque(self):
        assert _is_opaque("") is False

    def test_integer_is_not_opaque(self):
        assert _is_opaque(123456789) is False

    def test_mixed_short_string_is_not_opaque(self):
        # Not 32 chars, not UUID format
        assert _is_opaque("abc123") is False

    def test_uppercase_hex_is_not_opaque(self):
        # Our regex is lowercase-only — NT opaque handles are lowercase
        # This is intentional; real exchange IDs are never 32-char hex either
        assert _is_opaque("392826F4A8FC49009643F2681D50B753") is False


# ===========================================================================
# Unit Tests: diagnose_order_ids
# ===========================================================================

class TestDiagnoseOrderIds:
    def _diag(self, **kwargs) -> Diagnostics:
        return Diagnostics(_make_streamer(**kwargs))

    def test_healthy_state_returns_ok(self):
        d = self._diag()
        state = _make_open_trade_state()
        result = d.diagnose_order_ids(state)
        assert result["status"] == "OK"
        assert result["issues"] == []

    def test_entry_stop_collision_returns_critical(self):
        """The tuesday326pt2 bug: both IDs converge on stop's exchange ID."""
        d = self._diag()
        state = _make_open_trade_state(
            entry_ninja_order_id=REAL_STOP_OID,
            stop_order_id=REAL_STOP_OID,
        )
        result = d.diagnose_order_ids(state)
        assert result["status"] == "CRITICAL"
        codes = [i["code"] for i in result["issues"]]
        assert "ENTRY_STOP_ID_COLLISION" in codes

    def test_entry_id_still_opaque_after_fill(self):
        d = self._diag()
        state = _make_open_trade_state(entry_ninja_order_id=OPAQUE_ENTRY_HANDLE)
        result = d.diagnose_order_ids(state)
        assert result["status"] == "CRITICAL"
        codes = [i["code"] for i in result["issues"]]
        assert "ENTRY_ID_STILL_OPAQUE" in codes

    def test_stop_id_still_opaque_after_fill_is_warning(self):
        d = self._diag()
        state = _make_open_trade_state(stop_order_id=OPAQUE_STOP_HANDLE)
        result = d.diagnose_order_ids(state)
        assert result["status"] == "WARNING"
        codes = [i["code"] for i in result["issues"]]
        assert "STOP_ID_STILL_OPAQUE" in codes

    def test_entry_id_missing_after_fill(self):
        d = self._diag()
        state = _make_open_trade_state(entry_ninja_order_id=None)
        result = d.diagnose_order_ids(state)
        assert result["status"] == "CRITICAL"
        codes = [i["code"] for i in result["issues"]]
        assert "ENTRY_ID_MISSING_AFTER_FILL" in codes

    def test_stop_id_missing_with_stop_price_is_warning(self):
        d = self._diag()
        state = _make_open_trade_state(stop_order_id=None)
        result = d.diagnose_order_ids(state)
        assert result["status"] == "WARNING"
        codes = [i["code"] for i in result["issues"]]
        assert "STOP_ID_MISSING_AFTER_FILL" in codes

    def test_no_stop_id_no_stop_price_is_ok(self):
        """If no stop_price configured, missing stop_order_id is not a warning."""
        d = self._diag()
        state = {
            "entry_ninja_order_id": REAL_ENTRY_OID,
            "stop_order_id": None,
            "entry_filled": True,
            "entry_fill_price": ENTRY_PRICE,
            # no stop_price key
        }
        result = d.diagnose_order_ids(state)
        assert result["status"] == "OK"

    def test_empty_state_returns_ok(self):
        d = self._diag()
        result = d.diagnose_order_ids({})
        assert result["status"] == "OK"

    def test_collision_message_contains_collision_keyword(self):
        d = self._diag()
        state = _make_open_trade_state(
            entry_ninja_order_id=REAL_STOP_OID,
            stop_order_id=REAL_STOP_OID,
        )
        result = d.diagnose_order_ids(state)
        collision_issues = [
            i for i in result["issues"] if i["code"] == "ENTRY_STOP_ID_COLLISION"
        ]
        assert collision_issues
        assert "collision" in collision_issues[0]["message"].lower() or "identical" in collision_issues[0]["message"].lower()

    def test_details_populated(self):
        d = self._diag()
        state = _make_open_trade_state()
        result = d.diagnose_order_ids(state)
        assert "entry_ninja_order_id" in result["details"]
        assert "stop_order_id" in result["details"]


# ===========================================================================
# Unit Tests: verify_fill_consistency
# ===========================================================================

class TestVerifyFillConsistency:
    def _diag(self, max_fill_slippage_ticks: float = 4.0) -> Diagnostics:
        return Diagnostics(_make_streamer(max_fill_slippage_ticks=max_fill_slippage_ticks))

    def test_normal_entry_fill_is_ok(self):
        d = self._diag()
        state = _make_open_trade_state(entry_filled=False)
        result = d.verify_fill_consistency(state, ENTRY_PRICE, None, "test-cid")
        assert result["status"] == "OK"

    def test_fill_within_slippage_is_ok(self):
        d = self._diag(max_fill_slippage_ticks=4.0)  # 4 * 0.25 = 1.0 pt
        state = _make_open_trade_state(entry_filled=False)
        result = d.verify_fill_consistency(state, ENTRY_PRICE + 0.75, None, "test-cid")
        assert result["status"] == "OK"

    def test_fill_exceeds_slippage_is_suspicious(self):
        d = self._diag(max_fill_slippage_ticks=4.0)  # max = 1.0 pt
        state = _make_open_trade_state(entry_filled=False)
        result = d.verify_fill_consistency(state, ENTRY_PRICE + 2.0, None, "test-cid")
        assert result["status"] == "SUSPICIOUS"

    def test_stop_fill_vs_entry_price_is_anomaly(self):
        """
        Core test: tuesday326pt2 pattern.
        After entry at 6745.5, stop hits at 6755.5.
        A fill of 6755.5 should be detected as near stop_price,
        far from entry_fill_price → ANOMALY.
        """
        d = self._diag(max_fill_slippage_ticks=4.0)  # 4 * 0.25 = 1.0 pt
        state = {
            "entry_filled": True,
            "entry_fill_price": ENTRY_PRICE,   # 6745.5
            "stop_price": STOP_PRICE,          # 6755.5
            "expected_entry_ref": ENTRY_PRICE,
        }
        result = d.verify_fill_consistency(state, STOP_PRICE, None, "test-cid")
        assert result["status"] == "ANOMALY"
        assert any(
            "stop fill" in a.lower() or "bug-004" in a.lower()
            for a in result["checks"].get("anomalies", [])
        )

    def test_fill_timestamp_before_send_is_suspicious(self):
        d = self._diag()
        state = {
            "entry_filled": False,
            "sent_ts": 1000.0,
            "expected_entry_ref": ENTRY_PRICE,
        }
        result = d.verify_fill_consistency(state, ENTRY_PRICE, 999.0, "test-cid")
        assert result["status"] in {"SUSPICIOUS", "OK"}  # depends on slippage
        suspicious = result["checks"].get("suspicious", [])
        assert any("precedes" in s.lower() for s in suspicious)

    def test_no_model_price_skips_slippage_check(self):
        d = self._diag()
        state = {"entry_filled": False}  # no expected_entry_ref
        result = d.verify_fill_consistency(state, 9999.0, None, "test-cid")
        assert result["status"] == "OK"

    def test_returns_unknown_on_crash(self):
        """Verify defensive error handling."""
        streamer = MagicMock()
        streamer.tick_size = "NOT_A_FLOAT"  # triggers ValueError in float()
        d = Diagnostics(streamer)
        result = d.verify_fill_consistency({}, 100.0, None, "cid")
        assert result["status"] == "UNKNOWN"


# ===========================================================================
# Unit Tests: detect_known_bug_patterns
# ===========================================================================

class TestDetectKnownBugPatterns:
    def _make_diag(self, state: Dict[str, Any], position_state: str = "IN_POSITION_PROTECTED") -> Diagnostics:
        cid = "test-cid"
        streamer = _make_streamer(
            nt_order_state={cid: state},
            position_state=position_state,
        )
        return Diagnostics(streamer)

    def test_clean_state_detects_no_bugs(self):
        state = _make_open_trade_state()
        d = self._make_diag(state)
        results = d.detect_known_bug_patterns()
        detected = [r for r in results if r.get("detected")]
        assert detected == []

    def test_bug001_opaque_entry_id_detected(self):
        state = _make_open_trade_state(entry_ninja_order_id=OPAQUE_ENTRY_HANDLE)
        d = self._make_diag(state)
        results = d.detect_known_bug_patterns()
        bug001 = next(r for r in results if r["bug_id"] == "BUG-001")
        assert bug001["detected"] is True
        assert bug001["fix_applied"] is True

    def test_bug002_opaque_stop_id_detected(self):
        state = _make_open_trade_state(stop_order_id=OPAQUE_STOP_HANDLE)
        d = self._make_diag(state)
        results = d.detect_known_bug_patterns()
        bug002 = next(r for r in results if r["bug_id"] == "BUG-002")
        assert bug002["detected"] is True
        assert bug002["fix_applied"] is True

    def test_bug003_id_collision_numeric(self):
        """Stop fill corrupted entry_ninja_order_id to stop's exchange ID."""
        state = _make_open_trade_state(
            entry_ninja_order_id=REAL_STOP_OID,
            stop_order_id=REAL_STOP_OID,
        )
        d = self._make_diag(state)
        results = d.detect_known_bug_patterns()
        bug003 = next(r for r in results if r["bug_id"] == "BUG-003")
        bug004 = next(r for r in results if r["bug_id"] == "BUG-004")
        assert bug003["detected"] is True
        assert bug004["detected"] is True

    def test_bug004_any_collision(self):
        """BUG-004 fires on any collision (incl. opaque handles)."""
        state = _make_open_trade_state(
            entry_ninja_order_id=OPAQUE_STOP_HANDLE,
            stop_order_id=OPAQUE_STOP_HANDLE,
        )
        d = self._make_diag(state)
        results = d.detect_known_bug_patterns()
        bug004 = next(r for r in results if r["bug_id"] == "BUG-004")
        assert bug004["detected"] is True

    def test_all_bugs_have_fix_applied_except_005(self):
        results = self._make_diag({}).detect_known_bug_patterns()
        for r in results:
            if r["bug_id"] in {"BUG-001", "BUG-002", "BUG-003", "BUG-004"}:
                assert r["fix_applied"] is True, f"{r['bug_id']} should have fix_applied=True"
            elif r["bug_id"] == "BUG-005":
                assert r["fix_applied"] is False


# ===========================================================================
# Unit Tests: can_auto_heal
# ===========================================================================

class TestCanAutoHeal:
    def test_fill_price_oob_with_clean_state_can_heal(self):
        cid = "test-cid"
        state = _make_open_trade_state()
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={cid: state},
        )
        d = Diagnostics(streamer)
        can_heal, method = d.can_auto_heal("fill_price_out_of_bounds")
        assert can_heal is True
        assert method == "reset_lockout_if_safe"

    def test_fill_price_oob_wrong_lockout_code_cannot_heal(self):
        cid = "test-cid"
        state = _make_open_trade_state()
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="other_code",  # different code
            nt_order_state={cid: state},
        )
        d = Diagnostics(streamer)
        can_heal, method = d.can_auto_heal("fill_price_out_of_bounds")
        assert can_heal is False
        assert method == "operator_required"

    def test_fill_price_oob_with_collision_cannot_heal(self):
        """If ID collision is still active, CRITICAL → operator required."""
        cid = "test-cid"
        state = _make_open_trade_state(
            entry_ninja_order_id=REAL_STOP_OID,
            stop_order_id=REAL_STOP_OID,
        )
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={cid: state},
        )
        d = Diagnostics(streamer)
        can_heal, method = d.can_auto_heal("fill_price_out_of_bounds")
        assert can_heal is False
        assert method == "operator_required"

    def test_stop_id_opaque_entry_filled_no_exit_can_heal(self):
        cid = "test-cid"
        state = _make_open_trade_state(stop_order_id=OPAQUE_STOP_HANDLE)
        streamer = _make_streamer(nt_order_state={cid: state})
        d = Diagnostics(streamer)
        can_heal, method = d.can_auto_heal("stop_id_opaque")
        assert can_heal is True
        assert method == "force_order_id_upgrade"

    def test_unknown_issue_requires_operator(self):
        d = Diagnostics(_make_streamer())
        can_heal, method = d.can_auto_heal("some_random_issue")
        assert can_heal is False
        assert method == "operator_required"

    def test_feed_degraded_requires_operator(self):
        d = Diagnostics(_make_streamer())
        can_heal, _ = d.can_auto_heal("feed_health_degraded")
        assert can_heal is False


# ===========================================================================
# Unit Tests: reset_lockout_if_safe
# ===========================================================================

class TestResetLockoutIfSafe:
    def test_no_lockout_returns_failure(self):
        d = Diagnostics(_make_streamer(hard_lockout_active=False))
        result = d.reset_lockout_if_safe()
        assert result["success"] is False
        assert "No active" in result["message"]

    def test_wrong_lockout_code_returns_failure(self):
        d = Diagnostics(
            _make_streamer(hard_lockout_active=True, lockout_code="other_reason")
        )
        result = d.reset_lockout_if_safe()
        assert result["success"] is False
        assert "other_reason" in result["message"]

    def test_clean_state_allows_reset(self):
        cid = "test-cid"
        state = _make_open_trade_state()
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={cid: state},
        )
        d = Diagnostics(streamer)
        result = d.reset_lockout_if_safe()
        assert result["success"] is True
        assert result["method"] == "reset_lockout_if_safe"
        streamer._clear_hard_lockout.assert_called_once()
        streamer._log_exec_event.assert_called()

    def test_critical_order_id_state_blocks_reset(self):
        """ID collision still active → should NOT auto-reset."""
        cid = "test-cid"
        state = _make_open_trade_state(
            entry_ninja_order_id=REAL_STOP_OID,
            stop_order_id=REAL_STOP_OID,
        )
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={cid: state},
        )
        d = Diagnostics(streamer)
        result = d.reset_lockout_if_safe()
        assert result["success"] is False
        streamer._clear_hard_lockout.assert_not_called()

    def test_log_exec_event_called_with_correct_event(self):
        cid = "test-cid"
        state = _make_open_trade_state()
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={cid: state},
        )
        d = Diagnostics(streamer)
        d.reset_lockout_if_safe()
        call_args = streamer._log_exec_event.call_args_list
        assert any(
            c[0][0].get("event") == "lockout_reset_auto"
            for c in call_args
        )


# ===========================================================================
# Unit Tests: run_full_diagnostics
# ===========================================================================

class TestRunFullDiagnostics:
    def test_healthy_streamer_returns_healthy(self):
        cid = "test-cid"
        state = _make_open_trade_state()
        streamer = _make_streamer(nt_order_state={cid: state})
        d = Diagnostics(streamer)
        report = d.run_full_diagnostics()
        assert report["overall_status"] == "HEALTHY"
        assert report["checks"]["order_ids"]["status"] == "OK"
        assert report["checks"]["nt_connection"]["status"] == "HEALTHY"

    def test_lockout_active_returns_critical(self):
        cid = "test-cid"
        state = _make_open_trade_state()
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={cid: state},
        )
        d = Diagnostics(streamer)
        report = d.run_full_diagnostics()
        assert report["overall_status"] == "CRITICAL"
        assert report["checks"]["guardrails"]["hard_lockout_active"] is True

    def test_auto_heal_recommended_when_lockout_safe(self):
        cid = "test-cid"
        state = _make_open_trade_state()
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={cid: state},
        )
        d = Diagnostics(streamer)
        report = d.run_full_diagnostics()
        actions = report["actions_recommended"]
        assert any(a["action"] == "reset_lockout" for a in actions)
        auto_heal = report["auto_heal_available"]
        assert any(a["method"] == "reset_lockout_if_safe" for a in auto_heal)

    def test_nt_disconnected_returns_degraded(self):
        d = Diagnostics(_make_streamer(nt_connected=False, handshake_ok_ret=False))
        report = d.run_full_diagnostics()
        assert report["overall_status"] == "DEGRADED"
        assert report["checks"]["nt_connection"]["status"] == "DEGRADED"

    def test_report_has_required_top_level_keys(self):
        d = Diagnostics(_make_streamer())
        report = d.run_full_diagnostics()
        for key in ("timestamp", "run_id", "overall_status", "checks",
                    "actions_recommended", "auto_heal_available"):
            assert key in report, f"Missing key: {key}"

    def test_report_checks_has_required_sections(self):
        d = Diagnostics(_make_streamer())
        report = d.run_full_diagnostics()
        for section in ("order_ids", "fill_consistency", "bug_patterns",
                        "guardrails", "nt_connection", "feed_health"):
            assert section in report["checks"], f"Missing check section: {section}"

    def test_empty_order_state_returns_healthy(self):
        d = Diagnostics(_make_streamer(nt_order_state={}))
        report = d.run_full_diagnostics()
        # No active trade → no order ID issues
        assert report["checks"]["order_ids"]["status"] == "OK"

    def test_completes_fast(self):
        """run_full_diagnostics must complete in < 100ms."""
        import time
        cid = "test-cid"
        state = _make_open_trade_state()
        streamer = _make_streamer(nt_order_state={cid: state})
        d = Diagnostics(streamer)
        start = time.perf_counter()
        d.run_full_diagnostics()
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"Diagnostics too slow: {elapsed_ms:.1f}ms"


# ===========================================================================
# Integration Test: tuesday326pt2 run data
# ===========================================================================

@pytest.mark.integration
@pytest.mark.skipif(
    not TUESDAY_RUN_AVAILABLE,
    reason="tuesday326pt2 run data not found",
)
class TestTuesday326pt2Integration:
    """
    Verify diagnostics correctly identifies the tuesday326pt2 bug pattern
    from the actual run artefacts.
    """

    def _load_lockout(self) -> Dict[str, Any]:
        return json.loads((TUESDAY_RUN_DIR / "lockout.json").read_text())

    def _bug_state(self) -> Dict[str, Any]:
        """Reconstruct the corrupted state at the moment of lockout."""
        return {
            # After BUG-003: stop's exchange ID leaked into entry_ninja_order_id
            "entry_ninja_order_id": REAL_STOP_OID,   # 406099040466 (corrupted)
            "entry_order_id": OPAQUE_ENTRY_HANDLE,
            "stop_order_id": REAL_STOP_OID,           # 406099040466 (never upgraded from opaque → already overwritten)
            "entry_filled": True,
            "entry_fill_price": ENTRY_PRICE,
            "stop_price": STOP_PRICE,
            "expected_entry_ref": ENTRY_PRICE,
            "exit_fill_ts": None,
        }

    def test_lockout_file_has_correct_code(self):
        lockout = self._load_lockout()
        assert lockout["lockout_code"] == "fill_price_out_of_bounds"
        assert lockout["sticky"] is True

    def test_diagnose_order_ids_returns_critical(self):
        cid = "tuesday-cid"
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            lockout_sticky=True,
            nt_order_state={cid: self._bug_state()},
        )
        d = Diagnostics(streamer)
        result = d.diagnose_order_ids(self._bug_state())
        assert result["status"] == "CRITICAL", f"Expected CRITICAL, got: {result}"
        codes = [i["code"] for i in result["issues"]]
        assert "ENTRY_STOP_ID_COLLISION" in codes

    def test_verify_fill_detects_stop_fill_anomaly(self):
        cid = "tuesday-cid"
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={cid: self._bug_state()},
            max_fill_slippage_ticks=4.0,
        )
        d = Diagnostics(streamer)
        result = d.verify_fill_consistency(
            self._bug_state(),
            fill_price=STOP_PRICE,   # 6755.5 — the stop fill
            fill_ts=None,
            client_order_id=cid,
        )
        assert result["status"] == "ANOMALY"

    def test_detect_bug_patterns_finds_bug003_or_bug004(self):
        cid = "tuesday-cid"
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={cid: self._bug_state()},
        )
        d = Diagnostics(streamer)
        results = d.detect_known_bug_patterns()
        detected_ids = {r["bug_id"] for r in results if r.get("detected")}
        # BUG-003 and/or BUG-004 must be detected (collision)
        assert detected_ids & {"BUG-003", "BUG-004"}, (
            f"Expected BUG-003 or BUG-004 in detected, got: {detected_ids}"
        )

    def test_full_diagnostics_overall_critical(self):
        cid = "tuesday-cid"
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            lockout_sticky=True,
            nt_order_state={cid: self._bug_state()},
        )
        d = Diagnostics(streamer)
        report = d.run_full_diagnostics()
        assert report["overall_status"] == "CRITICAL"

    def test_full_diagnostics_after_fix_no_collision(self):
        """After the three fixes, the clean post-fix state should be HEALTHY."""
        cid = "fixed-cid"
        clean_state = _make_open_trade_state(
            entry_ninja_order_id=REAL_ENTRY_OID,   # 406099040457 (correct)
            stop_order_id=REAL_STOP_OID,           # 406099040466 (correct, different)
        )
        streamer = _make_streamer(
            hard_lockout_active=False,
            nt_order_state={cid: clean_state},
        )
        d = Diagnostics(streamer)
        report = d.run_full_diagnostics()
        # No collision → no bug pattern → HEALTHY (assuming NT connected, feed ok)
        assert report["checks"]["order_ids"]["status"] == "OK"
        bugs_detected = [
            r["bug_id"]
            for r in report["checks"].get("bug_patterns", [])
            if r.get("detected")
        ]
        assert bugs_detected == [], f"False positive bugs: {bugs_detected}"

    def test_auto_heal_available_method_is_reset_lockout(self):
        """
        Expected: auto_heal_available[0].method == 'reset_lockout_if_safe'
        but conditions_met may be False due to active ID collision.
        """
        cid = "tuesday-cid"
        streamer = _make_streamer(
            hard_lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={cid: self._bug_state()},
        )
        d = Diagnostics(streamer)
        report = d.run_full_diagnostics()
        auto_heals = report["auto_heal_available"]
        assert any(a["method"] == "reset_lockout_if_safe" for a in auto_heals)
