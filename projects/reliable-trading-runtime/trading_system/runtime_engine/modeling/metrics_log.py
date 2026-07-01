"""
Console-friendly metrics emitters for the standalone streamer.

The production bot writes to structured telemetry topics; here we only need a
tiny wrapper so ``live_trading_runtime`` can keep calling the same helpers without
failing when the logging backend is absent.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

logger = logging.getLogger("trading_runtime.metrics_log")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] metrics %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _emit(event: str, payload: Mapping[str, Any]) -> None:
    record = {"event": event, **dict(payload)}
    try:
        logger.info(json.dumps(record))
    except Exception:
        logger.info("%s %s", event, record)


def log_trade_event(**payload: Any) -> None:
    _emit("trade_event", payload)


def log_online_update(**payload: Any) -> None:
    _emit("online_update", payload)


def log_risk_state(**payload: Any) -> None:
    _emit("risk_state", payload)


def log_run_config(**payload: Any) -> None:
    _emit("run_config", payload)


__all__ = ["log_trade_event", "log_online_update", "log_risk_state", "log_run_config"]
