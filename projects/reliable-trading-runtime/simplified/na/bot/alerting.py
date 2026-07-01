from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import logging
import requests

logger = logging.getLogger(__name__)


class AlertSink:
    def send(self, level: str, message: str, **context) -> None:  # pragma: no cover
        raise NotImplementedError


class StdoutAlertSink(AlertSink):
    def send(self, level: str, message: str, **context) -> None:
        rec = {"level": level, "message": message, "context": context}
        print(json.dumps(rec), file=sys.stderr if level.lower() in {"warning", "error"} else sys.stdout)


class FileAlertSink(AlertSink):
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, level: str, message: str, **context) -> None:
        rec = {"level": level, "message": message, "context": context}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


class WebhookAlertSink(AlertSink):
    def __init__(self, url: str, *, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> None:
        self.url = url
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = timeout

    def send(self, level: str, message: str, **context) -> None:
        payload = {"level": level, "message": message, "context": context}
        try:
            resp = requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure
            logger.warning("Webhook alert send failed: %s", exc)


class DiscordWebhookAlertSink(WebhookAlertSink):
    def send(self, level: str, message: str, **context) -> None:
        embed_fields = []
        for key, value in context.items():
            embed_fields.append(f"**{key}**: {value}")
        payload = {
            "content": f"[{level.upper()}] {message}",
        }
        if embed_fields:
            payload["embeds"] = [{
                "title": "Context",
                "description": "\n".join(embed_fields)[:1800],
            }]
        try:
            resp = requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure
            logger.warning("Discord webhook send failed: %s", exc)


def build_alert_sink_from_config(sink_cfg: Optional[str]) -> AlertSink:
    if not sink_cfg:
        return StdoutAlertSink()
    if sink_cfg == "stdout":
        return StdoutAlertSink()
    if sink_cfg.startswith("file:"):
        return FileAlertSink(sink_cfg.split("file:", 1)[1])
    if sink_cfg.startswith("webhook:"):
        return WebhookAlertSink(sink_cfg.split("webhook:", 1)[1].strip())
    if sink_cfg.startswith("discord:"):
        return DiscordWebhookAlertSink(sink_cfg.split("discord:", 1)[1].strip())
    return StdoutAlertSink()


__all__ = [
    "AlertSink",
    "StdoutAlertSink",
    "FileAlertSink",
    "WebhookAlertSink",
    "DiscordWebhookAlertSink",
    "build_alert_sink_from_config",
]
