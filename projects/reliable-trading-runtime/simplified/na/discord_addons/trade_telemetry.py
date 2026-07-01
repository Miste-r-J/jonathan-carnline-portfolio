from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeTelemetry:
    event_type: str
    ts: str
    run_id: str
    account: Optional[str]
    instrument: Optional[str]
    signal_id: Optional[str]
    client_order_id: Optional[str]
    side: Optional[str]
    qty: Optional[float]
    price: Optional[float]
    position_qty: Optional[float]
    realized_pnl: Optional[float]
    unrealized_pnl: Optional[float]
    bridge_status: Optional[str]
    source: str
    payload_version: int
    is_execution_truth: bool
    meta: Dict[str, Any]


class DiscordTelemetryReporter:
    def __init__(self, *, router: Any, out_dir: Path, mode: Optional[str] = None) -> None:
        self.router = router
        self.out_dir = Path(out_dir)
        self.mode = str(mode or os.environ.get("OPENCLAW_DISCORD_REPORT_MODE", "shadow")).strip().lower()
        if self.mode not in {"off", "shadow", "live"}:
            self.mode = "shadow"
        self.shadow_path = self.out_dir / "discord_reporter_shadow.jsonl"
        self._seen: set[str] = set()

    def handle(self, record: Dict[str, Any]) -> None:
        dedupe_key = self._dedupe_key(record)
        if dedupe_key in self._seen:
            return
        self._seen.add(dedupe_key)
        payload = self._build_payload(record)
        if payload is None:
            return
        if self.mode == "off":
            return
        if self.mode == "shadow":
            self._write_shadow({"dedupe_key": dedupe_key, **payload, "record": record})
            return
        self._publish_live(payload, dedupe_key)

    def _publish_live(self, payload: Dict[str, Any], dedupe_key: str) -> None:
        route = {
            "event_type": payload.get("event_type"),
            "instrument": payload.get("instrument"),
            "audience": payload.get("audience"),
            "channel_key": payload.get("channel_key"),
            "channel_keys": payload.get("channel_keys"),
            "include_recap": payload.get("include_recap"),
        }
        try:
            if payload.get("kind") == "embed" and hasattr(self.router, "publish_embed"):
                self.router.publish_embed(
                    title=payload["title"],
                    description=payload["description"],
                    fields=payload.get("fields"),
                    color=payload.get("color"),
                    timestamp=payload.get("timestamp"),
                    dedupe_key=dedupe_key,
                    route=route,
                )
            elif hasattr(self.router, "publish_text"):
                self.router.publish_text(payload["description"], dedupe_key=dedupe_key, route=route)
        except Exception:
            logger.debug("Failed to publish telemetry to Discord", exc_info=True)
            self._write_shadow({"dedupe_key": dedupe_key, **payload})

    def _write_shadow(self, payload: Dict[str, Any]) -> None:
        try:
            self.shadow_path.parent.mkdir(parents=True, exist_ok=True)
            with self.shadow_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        except Exception:
            logger.debug("Failed to write telemetry shadow payload", exc_info=True)

    @staticmethod
    def _dedupe_key(record: Dict[str, Any]) -> str:
        bits = [
            str(record.get("event_type") or ""),
            str(record.get("run_id") or ""),
            str(record.get("client_order_id") or ""),
            str(record.get("signal_id") or ""),
            str(record.get("meta", {}).get("status") or ""),
            str(record.get("meta", {}).get("reason") or ""),
        ]
        return "|".join(bits)

    @staticmethod
    def _build_payload(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        event_type = str(record.get("event_type") or "").upper()
        instrument = record.get("instrument") or "?"
        side = record.get("side") or "?"
        qty = record.get("qty")
        price = record.get("price")
        normalized_event = event_type.lower()
        channel_key = None
        audience = "ops"
        include_recap = False
        if event_type in {"SIGNAL_EMITTED", "ORDER_ACK"}:
            normalized_event = "signal"
            audience = "pro"
        elif event_type in {"FILL"}:
            normalized_event = "fill"
            audience = "pro"
            include_recap = True
        elif event_type in {"LOCKOUT"}:
            normalized_event = "lockout"
            audience = "ops"
        elif event_type in {"HEALTH_CHANGE"}:
            normalized_event = "health_update"
            audience = "ops"
            channel_key = "health"
        elif event_type in {"DAILY_SUMMARY"}:
            normalized_event = "performance_report"
            audience = "ops"
            channel_key = "performance"
            include_recap = True
        description = (
            f"{event_type} {instrument} {side}"
            f"{'' if qty is None else f' qty={qty}'}"
            f"{'' if price is None else f' @ {price}'}"
        ).strip()
        if event_type in {"SIGNAL_EMITTED", "ORDER_ACK", "FILL", "LOCKOUT", "HEALTH_CHANGE", "DAILY_SUMMARY"}:
            return {
                "kind": "embed",
                "event_type": normalized_event,
                "instrument": instrument,
                "audience": audience,
                "channel_key": channel_key,
                "include_recap": include_recap,
                "title": event_type.replace("_", " ").title(),
                "description": description,
                "fields": [
                    {"name": "Run", "value": str(record.get("run_id") or "?"), "inline": True},
                    {"name": "CID", "value": str(record.get("client_order_id") or "n/a"), "inline": True},
                    {"name": "Bridge", "value": str(record.get("bridge_status") or "n/a"), "inline": True},
                ],
                "color": 0x2D9CDB if event_type not in {"LOCKOUT", "HEALTH_CHANGE"} else 0xE67E22,
                "timestamp": record.get("ts"),
            }
        if event_type == "POSITION_SNAPSHOT":
            return {
                "kind": "text",
                "event_type": "health_update",
                "instrument": instrument,
                "audience": "ops",
                "channel_key": "health",
                "description": f"POSITION {instrument} qty={record.get('position_qty')} pnl={record.get('realized_pnl')}",
            }
        return None


class TradeTelemetryStore:
    def __init__(self, *, out_dir: Path, reporter: Optional[DiscordTelemetryReporter] = None) -> None:
        self.out_dir = Path(out_dir)
        self.reporter = reporter
        self.telemetry_path = self.out_dir / "trade_telemetry.jsonl"
        self.live_pnl_state_path = self.out_dir / "live_pnl_state.json"
        self.system_health_path = self.out_dir / "system_health.json"
        self._system_health: Dict[str, Any] = {
            "gateway": {"status": "unknown"},
            "discord_mcp": {"status": "unknown"},
            "memory": {"status": "unknown"},
            "telemetry": {"status": "up"},
            "nt_bridge": {"status": "unknown"},
            "last_smoke_test_ts": None,
            "as_of": None,
        }

    def emit(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = TradeTelemetry(
            event_type=str(event_type or "").upper(),
            ts=str(payload.get("ts") or payload.get("timestamp") or ""),
            run_id=str(payload.get("run_id") or ""),
            account=self._to_optional_str(payload.get("account")),
            instrument=self._to_optional_str(payload.get("instrument")),
            signal_id=self._to_optional_str(payload.get("signal_id")),
            client_order_id=self._to_optional_str(payload.get("client_order_id")),
            side=self._to_optional_str(payload.get("side")),
            qty=self._to_optional_float(payload.get("qty")),
            price=self._to_optional_float(payload.get("price")),
            position_qty=self._to_optional_float(payload.get("position_qty")),
            realized_pnl=self._to_optional_float(payload.get("realized_pnl")),
            unrealized_pnl=self._to_optional_float(payload.get("unrealized_pnl")),
            bridge_status=self._to_optional_str(payload.get("bridge_status")),
            source=str(payload.get("source") or "stream_live_csv"),
            payload_version=1,
            is_execution_truth=bool(payload.get("is_execution_truth", False)),
            meta=dict(payload.get("meta") or {}),
        )
        doc = asdict(record)
        self._write_jsonl(self.telemetry_path, doc)
        self.update_health("telemetry", status="up", detail={"last_event_type": doc["event_type"]})
        if record.position_qty is not None or record.realized_pnl is not None or record.unrealized_pnl is not None:
            self.update_live_pnl_state(doc)
        if self.reporter is not None:
            self.reporter.handle(doc)
        return doc

    def update_live_pnl_state(self, payload: Dict[str, Any]) -> None:
        pos_qty = self._to_optional_float(payload.get("position_qty"))
        side = self._to_optional_str(payload.get("side"))
        if pos_qty is not None and abs(float(pos_qty)) <= 1e-9:
            side = "FLAT"
        doc = {
            "run_id": payload.get("run_id"),
            "ts": payload.get("ts") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "account": payload.get("account"),
            "instrument": payload.get("instrument"),
            "position_qty": pos_qty,
            "side": side,
            "realized_pnl": payload.get("realized_pnl"),
            "unrealized_pnl": payload.get("unrealized_pnl"),
            "bridge_status": payload.get("bridge_status"),
            "raw": payload,
        }
        self._write_json(self.live_pnl_state_path, doc)

    def update_health(self, component: str, *, status: str, detail: Optional[Dict[str, Any]] = None) -> None:
        self._system_health[str(component)] = {"status": status, **(detail or {})}
        self._system_health["as_of"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._write_json(self.system_health_path, self._system_health)

    def mark_smoke_test(self) -> None:
        self._system_health["last_smoke_test_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._write_json(self.system_health_path, self._system_health)

    @staticmethod
    def _to_optional_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            number = float(value)
        except Exception:
            return None
        if not (number == number):
            return None
        return number

    @staticmethod
    def _to_optional_str(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        return str(value)

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        # Atomic write (temp + os.replace) with bounded retries: protects hot
        # snapshot files from external locks (antivirus / cloud sync) so a
        # blocked write can never leave a truncated file or wedge the writer.
        text = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)
        for _attempt in range(3):
            tmp = path.with_name(f".{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, path)
                return
            except Exception:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                time.sleep(0.05)
        logger.debug("Failed to write JSON payload %s after retries", path, exc_info=True)

    @staticmethod
    def _write_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        # Append with bounded retries; each attempt uses a fresh handle so a
        # transient external lock cannot permanently wedge this writer.
        text = json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
        for _attempt in range(3):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(text)
                return
            except Exception:
                time.sleep(0.05)
        logger.debug("Failed to append JSONL payload %s after retries", path, exc_info=True)
