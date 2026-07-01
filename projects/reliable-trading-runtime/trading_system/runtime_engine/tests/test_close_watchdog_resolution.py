from __future__ import annotations

from trading_system.runtime_engine.integrations.guardrails import CloseWatchdog


def test_close_watchdog_ignores_non_sent_intent() -> None:
    watchdog = CloseWatchdog()

    detail = watchdog.register_intent(
        correlation_id="cid-1",
        intent_id="intent-1",
        bar_ts="2026-04-01T07:15:00-06:00",
        decision="BLOCKED_SAFETY",
        reason_code="guardrail_block",
        now=1.0,
    )

    assert detail is not None
    assert detail["decision"] == "BLOCKED_SAFETY"
    assert watchdog.pending_watches() == []


def test_close_watchdog_removes_terminal_watch_immediately() -> None:
    watchdog = CloseWatchdog()

    registered = watchdog.register_intent(
        correlation_id="cid-2",
        intent_id="intent-2",
        bar_ts="2026-04-01T07:20:00-06:00",
        decision="SENT",
        reason_code="sent",
        now=2.0,
    )
    assert registered is None
    assert len(watchdog.pending_watches()) == 1

    result = watchdog.record_order_event(
        correlation_id="cid-2",
        status="FLATTENED",
        now=3.0,
    )

    assert result["known"] is True
    assert result["resolved"] is True
    assert watchdog.pending_watches() == []
