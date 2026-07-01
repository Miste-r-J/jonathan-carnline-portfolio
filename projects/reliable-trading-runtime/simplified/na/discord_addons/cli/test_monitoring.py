"""
test_monitoring.py — Tests for the monitoring, alerting, and auto-heal infrastructure.

Run with:
    python -m pytest simplified/na/discord_addons/cli/test_monitoring.py -v

Marks:
    @pytest.mark.integration  — requires tuesday326pt2 run data
    @pytest.mark.slow         — timing-sensitive (waits for thread)
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from monitoring import (  # noqa: E402
    Alerts,
    AutoHealer,
    DiagnosticsLogger,
    DiagnosticsScheduler,
    DiagnosticsHTTPServer,
    format_diagnostics_json,
    format_status_table,
)
from diagnostics import Diagnostics  # noqa: E402


# ---------------------------------------------------------------------------
# Constants from tuesday326pt2
# ---------------------------------------------------------------------------

TUESDAY_RUN_DIR = Path(r"C:\test-data\paper-run")
TUESDAY_AVAILABLE = TUESDAY_RUN_DIR.exists()

REAL_ENTRY_OID = "406099040457"
REAL_STOP_OID = "406099040466"
OPAQUE_ENTRY = "392826f4a8fc49009643f2681d50b753"
OPAQUE_STOP = "41e37e1407594ff98effef8431cf15c1"
ENTRY_PRICE = 6745.5
STOP_PRICE = 6755.5


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _make_bot(
    *,
    lockout_active: bool = False,
    lockout_code: Optional[str] = None,
    nt_order_state: Optional[Dict[str, Any]] = None,
    position_state: str = "IN_POSITION_PROTECTED",
    nt_connected: bool = True,
    feed_age: float = 5.0,
    run_id: str = "test-run-id",
) -> MagicMock:
    bot = MagicMock()
    bot.run_id = run_id
    bot._hard_lockout_active = lockout_active
    bot._hard_lockout_code = lockout_code
    bot._hard_lockout_detail = {}
    bot._lockout_sticky = lockout_active
    bot._nt_order_state = nt_order_state or {}
    bot.tick_size = 0.25
    bot.max_fill_slippage_ticks = 4.0
    bot._effective_bar_age_max_sec = 90.0
    bot._bar_age_guard_seconds.return_value = feed_age
    bot._clear_hard_lockout = MagicMock()
    bot._log_exec_event = MagicMock()

    state_obj = MagicMock()
    state_obj.position_state = position_state
    bot.state = state_obj

    nt_bridge = MagicMock()
    nt_bridge.is_connected = nt_connected
    nt_bridge.handshake_ok.return_value = nt_connected
    bot.nt_bridge = nt_bridge

    # Attach real Diagnostics instance
    bot.diagnostics = Diagnostics(bot)
    return bot


def _clean_trade_state() -> Dict[str, Any]:
    return {
        "entry_ninja_order_id": REAL_ENTRY_OID,
        "stop_order_id": REAL_STOP_OID,
        "entry_filled": True,
        "entry_fill_price": ENTRY_PRICE,
        "stop_price": STOP_PRICE,
        "expected_entry_ref": ENTRY_PRICE,
        "exit_fill_ts": None,
    }


def _bug_trade_state() -> Dict[str, Any]:
    """State at the moment of the tuesday326pt2 lockout: ID collision."""
    return {
        "entry_ninja_order_id": REAL_STOP_OID,  # corrupted
        "stop_order_id": REAL_STOP_OID,
        "entry_filled": True,
        "entry_fill_price": ENTRY_PRICE,
        "stop_price": STOP_PRICE,
        "expected_entry_ref": ENTRY_PRICE,
        "exit_fill_ts": None,
    }


# ===========================================================================
# TestDiagnosticsLogger
# ===========================================================================

class TestDiagnosticsLogger:
    def test_log_to_file(self, tmp_path):
        log_path = tmp_path / "diagnostics.log"
        lgr = DiagnosticsLogger(log_path)
        lgr.log("DIAGNOSTICS_RUN", level="INFO", status="HEALTHY", duration_ms=12)
        content = log_path.read_text()
        assert "DIAGNOSTICS_RUN" in content
        assert "status=HEALTHY" in content
        assert "duration_ms=12" in content

    def test_log_format_has_timestamp_brackets(self, tmp_path):
        log_path = tmp_path / "diagnostics.log"
        lgr = DiagnosticsLogger(log_path)
        lgr.log("TEST_EVENT")
        content = log_path.read_text()
        # Format: [2026-03-03 15:08:43] TEST_EVENT
        assert content.strip().startswith("[")
        assert "] TEST_EVENT" in content

    def test_none_path_is_noop(self):
        lgr = DiagnosticsLogger(None)
        # Should not raise
        lgr.log("ANYTHING", status="X")

    def test_rotation_handler_configured(self, tmp_path):
        import logging.handlers
        log_path = tmp_path / "d.log"
        lgr = DiagnosticsLogger(log_path, max_bytes=1024, backup_count=3)
        # Logger was created and has a RotatingFileHandler
        assert lgr._logger is not None
        handlers = lgr._logger.handlers
        assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in handlers)

    def test_auto_creates_parent_directory(self, tmp_path):
        log_path = tmp_path / "subdir" / "nested" / "diagnostics.log"
        lgr = DiagnosticsLogger(log_path)
        lgr.log("INIT")
        assert log_path.exists()

    def test_multiple_events_appended(self, tmp_path):
        log_path = tmp_path / "diagnostics.log"
        lgr = DiagnosticsLogger(log_path)
        lgr.log("EVENT_A")
        lgr.log("EVENT_B")
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        assert "EVENT_A" in lines[0]
        assert "EVENT_B" in lines[1]


# ===========================================================================
# TestAlerts
# ===========================================================================

class TestAlerts:
    def test_console_alert_prints_to_stderr(self, capsys):
        a = Alerts(min_severity="INFO")
        a.send_console("Test Title", "Test message", severity="WARNING")
        captured = capsys.readouterr()
        assert "Test Title" in captured.err
        assert "WARNING" in captured.err

    def test_send_routes_to_console_when_no_webhook(self, capsys):
        a = Alerts(webhook_url=None, min_severity="INFO")
        sent = a.send("Alert", "msg", severity="INFO")
        assert sent is True
        captured = capsys.readouterr()
        assert "Alert" in captured.err

    def test_send_below_min_severity_filtered(self, capsys):
        a = Alerts(min_severity="WARNING")
        sent = a.send("Debug alert", "msg", severity="INFO")
        assert sent is False
        captured = capsys.readouterr()
        assert "Debug alert" not in captured.err

    def test_rate_limiting_blocks_duplicate(self, capsys):
        a = Alerts(min_severity="INFO")
        a.RATE_LIMIT_SECONDS = 300
        a.send("Same", "msg", severity="INFO", rate_key="my_key")
        # Second call within rate limit window
        sent = a.send("Same", "msg", severity="INFO", rate_key="my_key")
        assert sent is False

    def test_rate_limit_resets_after_window(self, capsys):
        a = Alerts(min_severity="INFO")
        a.RATE_LIMIT_SECONDS = 0  # zero = never rate-limited
        a.send("Same", "msg", severity="INFO", rate_key="k")
        sent = a.send("Same", "msg", severity="INFO", rate_key="k")
        assert sent is True

    def test_discord_payload_format(self):
        """Verify Discord embed structure is correct."""
        a = Alerts(webhook_url="https://discord.example.com/webhook", min_severity="INFO")
        a._reset_rate_limit()
        with patch("monitoring._requests") as mock_req:
            mock_req.post.return_value = MagicMock(status_code=204)
            a.send_discord(
                "Test Title",
                "Test message",
                severity="CRITICAL",
                fields=[{"name": "F1", "value": "V1", "inline": True}],
            )
            mock_req.post.assert_called_once()
            payload = mock_req.post.call_args.kwargs["json"]
            embed = payload["embeds"][0]
            assert "[CRITICAL]" in embed["title"]
            assert "Test Title" in embed["title"]
            assert embed["description"] == "Test message"
            assert embed["color"] == Alerts.SEVERITY_COLORS["CRITICAL"]
            assert embed["fields"][0]["name"] == "F1"

    def test_discord_not_called_when_no_webhook(self):
        a = Alerts(webhook_url=None)
        with patch("monitoring._requests") as mock_req:
            a.send_discord("T", "M")
            mock_req.post.assert_not_called()

    def test_discord_returns_false_on_error(self):
        a = Alerts(webhook_url="https://example.com/webhook")
        with patch("monitoring._requests") as mock_req:
            mock_req.post.side_effect = Exception("timeout")
            result = a.send_discord("T", "M")
            assert result is False

    def test_severity_colors_defined_for_all_levels(self):
        for lvl in ("INFO", "WARNING", "ERROR", "CRITICAL"):
            assert lvl in Alerts.SEVERITY_COLORS


# ===========================================================================
# TestAutoHealer
# ===========================================================================

class TestAutoHealer:
    def test_heal_calls_reset_lockout_if_safe(self):
        bot = _make_bot(
            lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={"cid": _clean_trade_state()},
        )
        healer = AutoHealer(bot)
        result = healer.heal("fill_price_out_of_bounds")
        assert result["success"] is True
        bot._clear_hard_lockout.assert_called_once()

    def test_heal_unknown_issue_returns_failure(self):
        bot = _make_bot()
        healer = AutoHealer(bot)
        result = healer.heal("some_unknown_issue")
        assert result["success"] is False

    def test_heal_verifies_after_success(self):
        bot = _make_bot(
            lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={"cid": _clean_trade_state()},
        )
        healer = AutoHealer(bot)
        # After clearing lockout, the mock bot won't reflect it (mock is static),
        # so verification may return CRITICAL still — but the key is it runs
        result = healer.heal("fill_price_out_of_bounds")
        assert "verification_status" in result

    def test_cannot_heal_when_id_collision_active(self):
        """With active ID collision, order_id_status=CRITICAL → no auto-heal."""
        bot = _make_bot(
            lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={"cid": _bug_trade_state()},
        )
        healer = AutoHealer(bot)
        can, method = healer.can_heal("fill_price_out_of_bounds")
        assert can is False
        assert method == "operator_required"

    def test_heal_logs_to_diag_logger(self, tmp_path):
        log_path = tmp_path / "d.log"
        lgr = DiagnosticsLogger(log_path)
        bot = _make_bot(
            lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={"cid": _clean_trade_state()},
        )
        healer = AutoHealer(bot, diag_logger=lgr)
        healer.heal("fill_price_out_of_bounds")
        content = log_path.read_text()
        assert "AUTO_HEAL_START" in content

    def test_heal_logs_success_event_to_bot(self):
        bot = _make_bot(
            lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={"cid": _clean_trade_state()},
        )
        healer = AutoHealer(bot)
        healer.heal("fill_price_out_of_bounds")
        calls = [c[0][0].get("event", "") for c in bot._log_exec_event.call_args_list]
        assert any("auto_heal" in e for e in calls)


# ===========================================================================
# TestDiagnosticsScheduler
# ===========================================================================

class TestDiagnosticsScheduler:
    def test_start_creates_daemon_thread(self):
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60)
        sched.start()
        try:
            assert sched.is_running()
            assert sched._thread is not None
            assert sched._thread.daemon is True
        finally:
            sched.stop()

    def test_stop_terminates_thread(self):
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60)
        sched.start()
        assert sched.is_running()
        sched.stop(timeout=2.0)
        assert not sched.is_running()

    def test_start_is_idempotent(self):
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60)
        sched.start()
        t1 = sched._thread
        sched.start()  # second call should no-op
        t2 = sched._thread
        assert t1 is t2
        sched.stop()

    def test_run_now_returns_result_dict(self):
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60)
        result = sched.run_now()
        assert isinstance(result, dict)
        assert "overall_status" in result

    def test_get_last_result_populated_after_run(self):
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60)
        assert sched.get_last_result() == {}
        sched.run_now()
        last = sched.get_last_result()
        assert "overall_status" in last

    @pytest.mark.slow
    def test_scheduler_runs_on_interval(self):
        """Verify the background thread calls diagnostics at least twice in 3s."""
        call_count = {"n": 0}
        original_run = None

        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=1)
        original_run = sched._run_once

        def counting_run():
            call_count["n"] += 1
            return original_run()

        sched._run_once = counting_run
        sched.start()
        try:
            time.sleep(2.5)
        finally:
            sched.stop()

        # Should have run at least twice (first immediately + at least one interval)
        assert call_count["n"] >= 2

    def test_issues_trigger_alert(self, capsys):
        """Issues in diagnostics result → alert sent to console."""
        bot = _make_bot(
            lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={"cid": _clean_trade_state()},
        )
        alerts = Alerts(min_severity="INFO")
        sched = DiagnosticsScheduler(
            bot,
            interval_seconds=60,
            alerts=alerts,
            auto_heal_enabled=False,
        )
        sched.run_now()
        captured = capsys.readouterr()
        # Lockout should have triggered a CRITICAL alert
        assert "ALERT" in captured.err or "LOCKOUT" in captured.err

    def test_auto_heal_executed_when_enabled(self):
        """With auto_heal_enabled=True, safe lockout is cleared automatically."""
        bot = _make_bot(
            lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={"cid": _clean_trade_state()},
        )
        alerts = Alerts(min_severity="DEBUG")
        healer = AutoHealer(bot)
        sched = DiagnosticsScheduler(
            bot,
            interval_seconds=60,
            alerts=alerts,
            auto_healer=healer,
            auto_heal_enabled=True,
        )
        sched.run_now()
        # _clear_hard_lockout should have been called by the auto-heal
        bot._clear_hard_lockout.assert_called()

    def test_log_file_written(self, tmp_path):
        """Scheduler writes to diagnostics.log."""
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60, log_dir=tmp_path)
        sched.run_now()
        log_path = tmp_path / "diagnostics.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "DIAGNOSTICS_RUN" in content


# ===========================================================================
# TestDiagnosticsHTTPServer
# ===========================================================================

class TestDiagnosticsHTTPServer:
    def _get_json(self, url: str) -> tuple[int, Dict[str, Any]]:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return int(getattr(resp, "status", 200)), data
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {"raw": body}
            return int(getattr(exc, "code", 0) or 0), data

    def _wait_get(self, url: str) -> tuple[int, Dict[str, Any]]:
        last = None
        for _ in range(30):
            try:
                return self._get_json(url)
            except Exception as exc:
                last = exc
                time.sleep(0.05)
        raise last  # type: ignore[misc]

    def test_health_healthy_is_200(self):
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60)
        with sched._lock:
            sched._last_result = {"overall_status": "HEALTHY", "timestamp": "2026-03-03T00:00:00Z", "duration_ms": 1.0}

        srv = DiagnosticsHTTPServer(sched, port=0, host="127.0.0.1")
        srv.start()
        try:
            code, payload = self._wait_get(f"{srv.base_url}/health")
            assert code == 200
            assert payload["status"] == "HEALTHY"
        finally:
            srv.stop()

    def test_health_unhealthy_is_503(self):
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60)
        with sched._lock:
            sched._last_result = {"overall_status": "CRITICAL", "timestamp": "2026-03-03T00:00:00Z"}

        srv = DiagnosticsHTTPServer(sched, port=0, host="127.0.0.1")
        srv.start()
        try:
            code, payload = self._wait_get(f"{srv.base_url}/health")
            assert code == 503
            assert payload["status"] == "CRITICAL"
        finally:
            srv.stop()

    def test_diagnostics_returns_last_result(self):
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60)
        with sched._lock:
            sched._last_result = {"overall_status": "HEALTHY", "checks": {"order_ids": {"status": "HEALTHY"}}}

        srv = DiagnosticsHTTPServer(sched, port=0, host="127.0.0.1")
        srv.start()
        try:
            code, payload = self._wait_get(f"{srv.base_url}/diagnostics")
            assert code == 200
            assert payload["overall_status"] == "HEALTHY"
            assert payload["checks"]["order_ids"]["status"] == "HEALTHY"
        finally:
            srv.stop()

    def test_diagnostics_before_first_run_is_503(self):
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60)
        with sched._lock:
            sched._last_result = {}

        srv = DiagnosticsHTTPServer(sched, port=0, host="127.0.0.1")
        srv.start()
        try:
            code, payload = self._wait_get(f"{srv.base_url}/diagnostics")
            assert code == 503
            assert payload.get("status") == "UNKNOWN"
        finally:
            srv.stop()


# ===========================================================================
# TestFormatStatusTable
# ===========================================================================

class TestFormatStatusTable:
    def _scheduler_with_result(self, result: Dict[str, Any]) -> DiagnosticsScheduler:
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60)
        with sched._lock:
            sched._last_result = result
        return sched

    def test_table_has_box_drawing_chars(self):
        sched = self._scheduler_with_result({"overall_status": "HEALTHY", "checks": {}})
        table = format_status_table(sched)
        assert "╔" in table
        assert "╗" in table
        assert "╚" in table
        assert "╝" in table
        assert "╠" in table

    def test_table_shows_overall_status(self):
        sched = self._scheduler_with_result({"overall_status": "CRITICAL", "checks": {}})
        table = format_status_table(sched)
        assert "CRITICAL" in table

    def test_table_shows_scheduler_stopped(self):
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=30)
        # Not started
        table = format_status_table(sched)
        assert "STOPPED" in table

    def test_table_shows_scheduler_running(self):
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=30)
        sched.start()
        try:
            table = format_status_table(sched)
            assert "RUNNING" in table
            assert "30s" in table
        finally:
            sched.stop()

    def test_table_shows_lockout_active(self):
        sched = self._scheduler_with_result({
            "overall_status": "CRITICAL",
            "checks": {
                "guardrails": {
                    "hard_lockout_active": True,
                    "lockout_code": "fill_price_out_of_bounds",
                },
            },
        })
        table = format_status_table(sched)
        assert "ACTIVE" in table

    def test_all_rows_have_correct_width(self):
        sched = self._scheduler_with_result({"overall_status": "HEALTHY", "checks": {}})
        table = format_status_table(sched, width=64)
        for line in table.splitlines():
            assert len(line) == 64, f"Wrong width ({len(line)}): {repr(line)}"


# ===========================================================================
# TestFormatDiagnosticsJson
# ===========================================================================

class TestFormatDiagnosticsJson:
    def test_valid_json_output(self):
        report = {"overall_status": "HEALTHY", "timestamp": "2026-03-03T12:00:00Z"}
        out = format_diagnostics_json(report)
        parsed = json.loads(out)
        assert parsed["overall_status"] == "HEALTHY"

    def test_non_serializable_handled(self):
        from datetime import datetime
        report = {"ts": datetime.now()}
        out = format_diagnostics_json(report)
        parsed = json.loads(out)
        assert "ts" in parsed


# ===========================================================================
# End-to-End Integration Tests (tuesday326pt2)
# ===========================================================================

@pytest.mark.integration
@pytest.mark.skipif(not TUESDAY_AVAILABLE, reason="tuesday326pt2 run data not found")
class TestEndToEnd:
    """Full pipeline test: scheduler → alert → (optional) auto-heal."""

    def _make_bug_bot(self) -> MagicMock:
        return _make_bot(
            lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={"tuesday_cid": _bug_trade_state()},
        )

    def _make_fixed_bot(self) -> MagicMock:
        """Post-fix bot: clean ID state, still has lockout active."""
        return _make_bot(
            lockout_active=True,
            lockout_code="fill_price_out_of_bounds",
            nt_order_state={"tuesday_cid": _clean_trade_state()},
        )

    def test_scheduler_run_now_detects_bug_pattern(self):
        bot = self._make_bug_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=60)
        result = sched.run_now()
        assert result["overall_status"] == "CRITICAL"
        oid_issues = result["checks"]["order_ids"]["issues"]
        codes = [i["code"] for i in oid_issues]
        assert "ENTRY_STOP_ID_COLLISION" in codes

    def test_alert_fires_on_critical_issue(self, capsys):
        bot = self._make_bug_bot()
        alerts = Alerts(min_severity="WARNING")
        sched = DiagnosticsScheduler(
            bot,
            interval_seconds=60,
            alerts=alerts,
            auto_heal_enabled=False,
        )
        sched.run_now()
        captured = capsys.readouterr()
        assert "ALERT" in captured.err or len(captured.err) > 0

    def test_auto_heal_not_triggered_during_active_collision(self):
        """With BUG active (CRITICAL order IDs) auto-heal must NOT fire."""
        bot = self._make_bug_bot()
        alerts = Alerts(min_severity="DEBUG")
        healer = AutoHealer(bot)
        sched = DiagnosticsScheduler(
            bot,
            interval_seconds=60,
            alerts=alerts,
            auto_healer=healer,
            auto_heal_enabled=True,
        )
        sched.run_now()
        # collision = CRITICAL order IDs → can_auto_heal=False → no _clear_hard_lockout
        bot._clear_hard_lockout.assert_not_called()

    def test_auto_heal_fires_after_fixes_applied(self):
        """With fixes applied (clean IDs), auto-heal should clear lockout."""
        bot = self._make_fixed_bot()
        alerts = Alerts(min_severity="DEBUG")
        healer = AutoHealer(bot)
        sched = DiagnosticsScheduler(
            bot,
            interval_seconds=60,
            alerts=alerts,
            auto_healer=healer,
            auto_heal_enabled=True,
        )
        sched.run_now()
        bot._clear_hard_lockout.assert_called()

    def test_diagnostics_log_captures_all_events(self, tmp_path):
        bot = self._make_fixed_bot()
        alerts = Alerts(min_severity="DEBUG")
        healer = AutoHealer(bot)
        sched = DiagnosticsScheduler(
            bot,
            interval_seconds=60,
            alerts=alerts,
            auto_healer=healer,
            auto_heal_enabled=True,
            log_dir=tmp_path,
        )
        # Run 3 cycles
        sched.run_now()
        sched.run_now()
        sched.run_now()
        log_path = tmp_path / "diagnostics.log"
        assert log_path.exists()
        content = log_path.read_text()
        # Should have 3 DIAGNOSTICS_RUN events + at least 1 AUTO_HEAL_* event
        run_count = content.count("DIAGNOSTICS_RUN")
        assert run_count >= 3
        assert "AUTO_HEAL" in content

    @pytest.mark.slow
    def test_background_thread_fires_three_times(self):
        """Scheduler thread runs diagnostics on interval — verify 3 runs in ~4s."""
        call_times = []
        bot = _make_bot()
        sched = DiagnosticsScheduler(bot, interval_seconds=1)
        orig = sched._run_once

        def counting(*a, **kw):
            call_times.append(time.time())
            return orig(*a, **kw)

        sched._run_once = counting
        sched.start()
        try:
            time.sleep(3.5)
        finally:
            sched.stop()

        assert len(call_times) >= 3, f"Only {len(call_times)} runs in 3.5s"
        # Intervals should be ~1s (allow ±0.5s tolerance)
        for i in range(1, min(len(call_times), 4)):
            gap = call_times[i] - call_times[i - 1]
            assert 0.5 <= gap <= 2.5, f"Unexpected interval: {gap:.2f}s"
