"""monitoring.py — Monitoring, alerting, and auto-heal infrastructure.

Provides:
  - DiagnosticsLogger    : rotating file log for diagnostic events
  - Alerts               : Discord webhook + console alerts with rate limiting
  - AutoHealer           : executes safe auto-heal actions with verification
  - DiagnosticsScheduler : background-thread scheduler (daemon, never blocks bot)
  - DiagnosticsHTTPServer: lightweight HTTP server exposing /health and /diagnostics
  - format_status_table  : CLI box-drawing status display

All classes are defensive: no exception propagates to the calling bot thread.
"""

from __future__ import annotations

import http.server
import json
import logging
import logging.handlers
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional dependency: requests (needed for Discord webhook)
# ---------------------------------------------------------------------------
try:
    import requests as _requests

    _REQUESTS_AVAILABLE = True
except ImportError:
    _requests = None  # type: ignore
    _REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# DiagnosticsLogger
# ---------------------------------------------------------------------------

class DiagnosticsLogger:
    """
    Rotating file logger for diagnostic events.

    Log line format:
        [2026-03-03 15:08:43] DIAGNOSTICS_RUN status=HEALTHY duration_ms=12
    """

    def __init__(
        self,
        log_path: Optional[Path],
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        self._logger: Optional[logging.Logger] = None
        if log_path is None:
            return
        try:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Use a unique logger name so multiple instances don't share handlers
            name = f"diagnostics_{id(self)}"
            lgr = logging.getLogger(name)
            lgr.setLevel(logging.DEBUG)
            lgr.propagate = False
            handler = logging.handlers.RotatingFileHandler(
                str(log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            handler.setFormatter(
                logging.Formatter(
                    fmt="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            lgr.addHandler(handler)
            self._logger = lgr
        except Exception:
            pass

    def log(self, event: str, level: str = "INFO", **fields: Any) -> None:
        """Write one structured log line."""
        if self._logger is None:
            return
        try:
            parts = [event]
            for k, v in fields.items():
                parts.append(f"{k}={v!r}" if isinstance(v, (list, dict)) else f"{k}={v}")
            msg = " ".join(parts)
            lvl = getattr(logging, level.upper(), logging.INFO)
            self._logger.log(lvl, msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class Alerts:
    """
    Sends diagnostic alerts to Discord webhook and/or console.

    Rate-limiting: at most one alert per issue key per RATE_LIMIT_SECONDS.
    Severity filter: alerts below min_severity are silently dropped.
    """

    SEVERITY_COLORS: Dict[str, int] = {
        "INFO": 0x00CC44,     # green
        "WARNING": 0xFFCC00,  # yellow
        "ERROR": 0xFF8C00,    # orange
        "CRITICAL": 0xFF2222, # red
    }
    # Ordering for min_severity filter
    _SEVERITY_ORDER: Dict[str, int] = {
        "DEBUG": 0,
        "INFO": 1,
        "WARNING": 2,
        "ERROR": 3,
        "CRITICAL": 4,
    }
    RATE_LIMIT_SECONDS: int = 300  # 5 minutes

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        min_severity: str = "WARNING",
        bot_name: str = "Misterj Trades Bot",
    ) -> None:
        self._webhook_url = webhook_url or None
        self._min_severity_level = self._SEVERITY_ORDER.get(min_severity.upper(), 2)
        self._bot_name = bot_name
        self._alert_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    # --- Public API -------------------------------------------------------

    def send(
        self,
        title: str,
        message: str,
        severity: str = "INFO",
        fields: Optional[List[Dict[str, Any]]] = None,
        rate_key: Optional[str] = None,
    ) -> bool:
        """
        Send an alert via Discord (if configured) and console.

        Returns True if the alert was actually sent (not filtered/rate-limited).
        """
        # Min-severity filter
        sev_level = self._SEVERITY_ORDER.get(severity.upper(), 1)
        if sev_level < self._min_severity_level:
            return False

        # Rate limiting
        key = rate_key or f"{severity}:{title}"
        if self._is_rate_limited(key):
            return False
        self._mark_alerted(key)

        sent = False
        if self._webhook_url:
            sent = self.send_discord(title, message, severity, fields=fields)
        self.send_console(title, message, severity)
        return True

    def send_discord(
        self,
        title: str,
        message: str,
        severity: str = "INFO",
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """POST to Discord webhook. Returns True on success."""
        if not _REQUESTS_AVAILABLE or not self._webhook_url:
            return False
        try:
            color = self.SEVERITY_COLORS.get(severity.upper(), 0xAAAAAA)
            embed: Dict[str, Any] = {
                "title": f"[{severity.upper()}] {title}",
                "description": str(message)[:2000],
                "color": color,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if fields:
                embed["fields"] = [
                    {
                        "name": str(f.get("name", ""))[:256],
                        "value": str(f.get("value", ""))[:1024],
                        "inline": bool(f.get("inline", False)),
                    }
                    for f in fields[:25]
                ]
            payload = {
                "username": self._bot_name,
                "embeds": [embed],
            }
            resp = _requests.post(
                self._webhook_url,
                json=payload,
                timeout=10,
            )
            return resp.status_code in {200, 204}
        except Exception:
            return False

    def send_console(
        self,
        title: str,
        message: str,
        severity: str = "INFO",
    ) -> None:
        """Print alert to stderr."""
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            sev_upper = severity.upper()
            import sys
            print(
                f"[{ts}] ALERT/{sev_upper} {title} — {message}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass

    # --- Internal helpers -------------------------------------------------

    def _is_rate_limited(self, key: str) -> bool:
        with self._lock:
            last = self._alert_times.get(key)
        if last is None:
            return False
        return (time.time() - last) < self.RATE_LIMIT_SECONDS

    def _mark_alerted(self, key: str) -> None:
        with self._lock:
            self._alert_times[key] = time.time()

    def _reset_rate_limit(self, key: Optional[str] = None) -> None:
        """Test helper — clear rate limit state."""
        with self._lock:
            if key is None:
                self._alert_times.clear()
            else:
                self._alert_times.pop(key, None)


# ---------------------------------------------------------------------------
# AutoHealer
# ---------------------------------------------------------------------------

class AutoHealer:
    """
    Executes auto-heal actions for known diagnostic issues.

    Currently supported heal methods:
        reset_lockout_if_safe  — calls bot.diagnostics.reset_lockout_if_safe()
        force_order_id_upgrade — placeholder (no bot method yet; logs intent)
        request_nt_resync      — placeholder (no bot method yet; logs intent)
    """

    def __init__(
        self,
        bot: Any,
        diag_logger: Optional[DiagnosticsLogger] = None,
    ) -> None:
        self._bot = bot
        self._diag_logger = diag_logger

    def can_heal(self, issue_code: str) -> Tuple[bool, str]:
        """Delegate to bot.diagnostics.can_auto_heal()."""
        try:
            diag = getattr(self._bot, "diagnostics", None)
            if diag is None:
                return False, "diagnostics_not_available"
            return diag.can_auto_heal(issue_code)
        except Exception:
            return False, "operator_required"

    def heal(self, issue_code: str) -> Dict[str, Any]:
        """
        Execute heal for issue_code.  Runs diagnostics before and after.

        Returns:
            { success, method, message, verification_status, verification_passed }
        """
        can, method = self.can_heal(issue_code)
        if not can:
            result: Dict[str, Any] = {
                "success": False,
                "method": method,
                "message": f"Cannot auto-heal '{issue_code}': {method}",
                "verification_status": None,
                "verification_passed": False,
            }
            self._log_heal("AUTO_HEAL_SKIP", issue_code=issue_code, method=method)
            return result

        self._log_heal("AUTO_HEAL_START", issue_code=issue_code, method=method)

        try:
            result = self._execute(method)
        except Exception as exc:
            result = {
                "success": False,
                "method": method,
                "message": f"Heal raised: {exc}",
                "verification_status": None,
                "verification_passed": False,
            }

        # Verify outcome
        if result.get("success"):
            try:
                diag = getattr(self._bot, "diagnostics", None)
                verify_report = diag.run_full_diagnostics() if diag else {}
                vstatus = verify_report.get("overall_status", "UNKNOWN")
                result["verification_status"] = vstatus
                result["verification_passed"] = vstatus == "HEALTHY"
            except Exception:
                result["verification_status"] = "UNKNOWN"
                result["verification_passed"] = False

        event = "AUTO_HEAL_SUCCESS" if result.get("success") else "AUTO_HEAL_FAILED"
        self._log_heal(
            event,
            issue_code=issue_code,
            method=method,
            success=result.get("success"),
            verification_passed=result.get("verification_passed"),
        )
        return result

    def _execute(self, method: str) -> Dict[str, Any]:
        """Dispatch to the appropriate heal implementation."""
        if method == "reset_lockout_if_safe":
            diag = getattr(self._bot, "diagnostics", None)
            if diag is None:
                return {"success": False, "method": method, "message": "diagnostics not available"}
            return diag.reset_lockout_if_safe()

        if method == "force_order_id_upgrade":
            # Bot method not yet implemented; log intent only
            return {
                "success": False,
                "method": method,
                "message": "force_order_id_upgrade not implemented in bot — manual intervention required",
            }

        if method == "request_nt_resync":
            try:
                fn = getattr(self._bot, "_force_snapshot_resync", None)
                if fn is not None:
                    fn()
                    return {"success": True, "method": method, "message": "NT resync requested"}
            except Exception as exc:
                return {"success": False, "method": method, "message": str(exc)}
            return {"success": False, "method": method, "message": "resync not available"}

        return {"success": False, "method": method, "message": f"No executor for method '{method}'"}

    def _log_heal(self, event: str, **kwargs: Any) -> None:
        if self._diag_logger:
            self._diag_logger.log(event, level="INFO", **kwargs)
        try:
            fn = getattr(self._bot, "_log_exec_event", None)
            if fn:
                fn({"event": event.lower(), **kwargs})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# DiagnosticsScheduler
# ---------------------------------------------------------------------------

class DiagnosticsScheduler:
    """
    Background-thread scheduler that runs diagnostics every N seconds.

    Thread is daemonized — dies automatically when the main process exits.
    Defensive: any exception inside the loop is caught and logged; the bot
    trading thread is never affected.
    """

    def __init__(
        self,
        bot: Any,
        interval_seconds: int = 30,
        alerts: Optional[Alerts] = None,
        auto_healer: Optional[AutoHealer] = None,
        auto_heal_enabled: bool = False,
        diag_logger: Optional[DiagnosticsLogger] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        self._bot = bot
        self._interval = max(1, int(interval_seconds))
        self._alerts = alerts
        self._auto_healer = auto_healer
        self._auto_heal_enabled = auto_heal_enabled

        # Build logger from log_dir if not provided explicitly
        if diag_logger is None and log_dir is not None:
            diag_logger = DiagnosticsLogger(Path(log_dir) / "diagnostics.log")
        self._diag_logger = diag_logger

        # Wire the same logger into the healer so all auto-heal events share the log
        if self._auto_healer is not None and self._diag_logger is not None:
            if getattr(self._auto_healer, "_diag_logger", None) is None:
                self._auto_healer._diag_logger = self._diag_logger

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._run_now_event = threading.Event()
        self._last_result: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # --- Public API -------------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler thread (no-op if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="diag-scheduler",
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the scheduler to stop and wait for it to exit."""
        self._stop_event.set()
        self._run_now_event.set()  # unblock any pending wait
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def run_now(self) -> Dict[str, Any]:
        """
        Force an immediate diagnostic run and return the result.
        Thread-safe — can be called from any thread.
        """
        return self._run_once()

    def get_last_result(self) -> Dict[str, Any]:
        """Return the result of the most recent diagnostic run."""
        with self._lock:
            return dict(self._last_result)

    def is_running(self) -> bool:
        """True if the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # --- Internal ---------------------------------------------------------

    def _run_loop(self) -> None:
        """Main scheduler loop — runs in the background thread."""
        self._run_once()
        while not self._stop_event.is_set():
            triggered = self._run_now_event.wait(timeout=self._interval)
            if triggered:
                self._run_now_event.clear()
            if self._stop_event.is_set():
                break
            self._run_once()

    def _run_once(self) -> Dict[str, Any]:
        """Execute one diagnostic cycle: run → log → alert → heal."""
        start_ns = time.perf_counter_ns()
        result: Dict[str, Any] = {}
        try:
            diag = getattr(self._bot, "diagnostics", None)
            if diag is None:
                result = {"overall_status": "UNKNOWN", "error": "diagnostics not attached"}
            else:
                result = diag.run_full_diagnostics()
        except Exception as exc:
            result = {"overall_status": "UNKNOWN", "error": str(exc)}

        elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 1)
        result["duration_ms"] = elapsed_ms

        with self._lock:
            self._last_result = result

        # Log
        issues = self._extract_issue_codes(result)
        self._log(
            "DIAGNOSTICS_RUN",
            status=result.get("overall_status", "UNKNOWN"),
            duration_ms=elapsed_ms,
            issues=issues or "[]",
        )

        # Alert + maybe heal
        if result.get("overall_status") != "HEALTHY":
            self._process_result(result)

        return result

    def _process_result(self, result: Dict[str, Any]) -> None:
        """Send alerts and optionally execute auto-heal for each issue."""
        try:
            overall = result.get("overall_status", "UNKNOWN")
            severity = "CRITICAL" if overall == "CRITICAL" else "WARNING"
            run_id = result.get("run_id", "unknown")

            issues = self._extract_issues(result)
            for issue in issues:
                code = issue.get("code", "UNKNOWN")
                message = issue.get("message", "")
                alert_key = f"issue:{code}"

                # Check auto-heal first (skip alert if healing)
                healed = False
                auto_heals = result.get("auto_heal_available", [])
                heal_info = next(
                    (a for a in auto_heals if a.get("conditions_met")), None
                )
                if heal_info and self._auto_heal_enabled and self._auto_healer:
                    heal_result = self._auto_healer.heal(code)
                    if heal_result.get("success"):
                        healed = True
                        self._alert(
                            "Auto-Heal Executed",
                            f"Healed {code} via {heal_result['method']}",
                            severity="INFO",
                            fields=[
                                {"name": "Run ID", "value": run_id[:24], "inline": True},
                                {"name": "Method", "value": heal_result["method"], "inline": True},
                                {"name": "Verified", "value": str(heal_result.get("verification_passed", "?")), "inline": True},
                            ],
                            rate_key=f"healed:{code}",
                        )
                        self._log(
                            "ALERT_SENT",
                            severity="INFO",
                            message=f"Auto-Heal Executed: {code}",
                        )

                if not healed:
                    fields = [
                        {"name": "Run ID", "value": run_id[:24], "inline": True},
                        {"name": "Issue", "value": code, "inline": True},
                    ]
                    if heal_info:
                        fields.append({
                            "name": "Auto-Heal",
                            "value": f"Available: {heal_info.get('method', '?')} (disabled)",
                            "inline": True,
                        })
                    sent = self._alert(
                        f"Issue Detected: {code}",
                        message,
                        severity=severity,
                        fields=fields,
                        rate_key=alert_key,
                    )
                    if sent:
                        self._log("ALERT_SENT", severity=severity, code=code)

        except Exception:
            pass

    def _alert(self, title: str, message: str, severity: str, fields=None, rate_key=None) -> bool:
        if self._alerts is None:
            return False
        try:
            return self._alerts.send(
                title, message, severity, fields=fields, rate_key=rate_key
            )
        except Exception:
            return False

    def _log(self, event: str, level: str = "INFO", **fields: Any) -> None:
        if self._diag_logger:
            try:
                self._diag_logger.log(event, level=level, **fields)
            except Exception:
                pass

    def _extract_issues(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pull all issue dicts from the diagnostic result."""
        issues: List[Dict[str, Any]] = []
        try:
            checks = result.get("checks", {})
            # order_ids
            for i in (checks.get("order_ids") or {}).get("issues", []):
                issues.append(i)
            # bug_patterns
            for b in checks.get("bug_patterns") or []:
                if b.get("detected"):
                    issues.append({
                        "code": b.get("bug_id", "UNKNOWN_BUG"),
                        "message": b.get("description", ""),
                        "severity": b.get("severity", "HIGH"),
                    })
            # guardrails
            g = checks.get("guardrails") or {}
            if g.get("hard_lockout_active"):
                issues.append({
                    "code": f"LOCKOUT_{str(g.get('lockout_code', 'UNKNOWN')).upper()}",
                    "message": f"Hard lockout active: {g.get('lockout_code')}",
                    "severity": "CRITICAL",
                })
        except Exception:
            pass
        return issues

    def _extract_issue_codes(self, result: Dict[str, Any]) -> str:
        try:
            codes = [i.get("code", "?") for i in self._extract_issues(result)]
            return json.dumps(codes)
        except Exception:
            return "[]"


# ---------------------------------------------------------------------------
# DiagnosticsHTTPServer
# ---------------------------------------------------------------------------


def _make_handler_class(scheduler: "DiagnosticsScheduler") -> type:
    """Return a BaseHTTPRequestHandler class with scheduler bound via closure."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        _scheduler = scheduler

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/health":
                self._serve_health()
            elif path == "/diagnostics":
                self._serve_diagnostics()
            else:
                self._respond(
                    404,
                    {"error": "Not found", "endpoints": ["/health", "/diagnostics"]},
                )

        def _serve_health(self) -> None:
            result = self._scheduler.get_last_result()
            overall = result.get("overall_status", "UNKNOWN") if result else "UNKNOWN"
            code = 200 if overall == "HEALTHY" else 503
            self._respond(
                code,
                {
                    "status": overall,
                    "timestamp": result.get("timestamp", "") if result else "",
                    "duration_ms": result.get("duration_ms") if result else None,
                },
            )

        def _serve_diagnostics(self) -> None:
            result = self._scheduler.get_last_result()
            if result:
                self._respond(200, result)
            else:
                self._respond(
                    503,
                    {"status": "UNKNOWN", "error": "No diagnostic run completed yet"},
                )

        def _respond(self, code: int, data: Dict[str, Any]) -> None:
            try:
                body = json.dumps(data, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                pass

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # suppress default access log to stderr

    return _Handler


class DiagnosticsHTTPServer:
    """
    Minimal HTTP server exposing diagnostics status over localhost.

    Serves cached results from a DiagnosticsScheduler — no blocking diagnostics
    run on the request path.

    Endpoints:
        GET /health       — 200 {"status": "HEALTHY"} or 503 {"status": "..."}
        GET /diagnostics  — 200 full JSON report (last scheduler result)

    Tips:
        - Use port=0 to get a random ephemeral port (read back via .port).
        - Default host is 127.0.0.1 (localhost only); never exposed externally.
    """

    def __init__(
        self,
        scheduler: "DiagnosticsScheduler",
        port: int = 8765,
        host: str = "127.0.0.1",
    ) -> None:
        self._scheduler = scheduler
        self._host = host
        self._port = port
        self._server: Optional[http.server.HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        """Actual bound port — useful when port=0 was requested."""
        if self._server is not None:
            return self._server.server_address[1]
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self.port}"

    def start(self) -> None:
        """Start the HTTP server in a daemon thread. No-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            handler_cls = _make_handler_class(self._scheduler)
            self._server = http.server.HTTPServer((self._host, self._port), handler_cls)
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="diag-http",
            )
            self._thread.start()
        except Exception:
            self._server = None
            self._thread = None

    def stop(self, timeout: float = 5.0) -> None:
        """Shutdown the HTTP server and wait for thread to exit."""
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._server = None
        self._thread = None

    def is_running(self) -> bool:
        """True if the server thread is alive."""
        return self._thread is not None and self._thread.is_alive()


# ---------------------------------------------------------------------------
# CLI display helpers
# ---------------------------------------------------------------------------

def format_status_table(scheduler: DiagnosticsScheduler, *, width: int = 64) -> str:
    """
    Render a box-drawing diagnostics status table.

    Example:
        ╔══════════════════════════════════════════════════════════════╗
        ║                    DIAGNOSTICS STATUS                        ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  Overall:        HEALTHY                                     ║
        ...
        ╚══════════════════════════════════════════════════════════════╝
    """
    inner = width - 2  # chars between ║ and ║

    def top() -> str:
        return "╔" + "═" * inner + "╗"

    def bot_line() -> str:
        return "╚" + "═" * inner + "╝"

    def sep() -> str:
        return "╠" + "═" * inner + "╣"

    def row(label: str, value: str) -> str:
        content = f"  {label:<20}{value}"
        return "║" + content[:inner].ljust(inner) + "║"

    def hdr(text: str) -> str:
        return "║" + text.center(inner) + "║"

    # Gather data
    last = scheduler.get_last_result()
    checks = last.get("checks", {}) if last else {}

    # Overall status + timing
    overall = last.get("overall_status", "UNKNOWN") if last else "UNKNOWN"
    duration = last.get("duration_ms", "?") if last else "?"
    ts_str = last.get("timestamp", "") if last else ""
    last_run_display = "Never"
    if ts_str:
        try:
            last_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age_s = int(time.time() - last_dt.timestamp())
            last_run_display = f"{last_dt.strftime('%H:%M:%S')} ({age_s}s ago, {duration}ms)"
        except Exception:
            last_run_display = ts_str[:19]

    # Scheduler
    sched_status = (
        f"RUNNING (interval: {scheduler._interval}s)"
        if scheduler.is_running()
        else "STOPPED"
    )

    # Alerts config
    if scheduler._alerts:
        has_discord = bool(getattr(scheduler._alerts, "_webhook_url", None))
        alerts_cfg = f"Enabled (Discord: {'configured' if has_discord else 'not set'})"
    else:
        alerts_cfg = "Disabled"

    auto_heal_cfg = "Enabled" if scheduler._auto_heal_enabled else "Disabled"

    # Check-level statuses
    oid = (checks.get("order_ids") or {}).get("status", "?")

    fc_status = (checks.get("fill_consistency") or {}).get("status", "?")

    bugs = checks.get("bug_patterns") or []
    detected = [b.get("bug_id", "?") for b in bugs if b.get("detected")]
    bug_display = f"ISSUES: {', '.join(detected)}" if detected else "All fixes applied"

    g = checks.get("guardrails") or {}
    lockout_active = g.get("hard_lockout_active", False)
    lockout_display = f"ACTIVE ({g.get('lockout_code', '?')})" if lockout_active else "INACTIVE"

    nt = checks.get("nt_connection") or {}
    nt_display = "CONNECTED" if nt.get("nt_connected") else "DISCONNECTED"

    feed = checks.get("feed_health") or {}
    feed_age = feed.get("bar_age_sec")
    feed_display = (
        f"OK ({feed_age:.0f}s)" if feed.get("feed_health_ok") and feed_age is not None
        else "DEGRADED" if not feed.get("feed_health_ok") else "?"
    )

    lines = [
        top(),
        hdr("DIAGNOSTICS STATUS"),
        sep(),
        row("Overall:", f"{overall}"),
        row("Scheduler:", sched_status),
        row("Last Run:", last_run_display),
        row("Alerts:", alerts_cfg),
        row("Auto-Heal:", auto_heal_cfg),
        sep(),
        row("Order IDs:", oid),
        row("Fill Consistency:", fc_status),
        row("Bug Patterns:", bug_display),
        row("Lockout:", lockout_display),
        row("NT Connection:", nt_display),
        row("Feed Health:", feed_display),
        bot_line(),
    ]
    return "\n".join(lines)


def format_diagnostics_json(report: Dict[str, Any]) -> str:
    """Pretty-print a diagnostics report as JSON."""
    return json.dumps(report, indent=2, default=str)
