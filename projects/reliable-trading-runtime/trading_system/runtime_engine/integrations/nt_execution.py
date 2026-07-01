from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ExecutionRecord:
    client_order_id: str
    status: str
    instrument: str
    side: str
    qty: int
    stop_price: Optional[float]
    target_price: Optional[float]
    oco_id: Optional[str]
    entry_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    target_order_id: Optional[str] = None
    ts: float = field(default_factory=time.time)
    error: Optional[str] = None


class ExecutionLedger:
    """
    Persistent JSONL ledger keyed by client_order_id to enforce idempotency and restart safety.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock = threading.Lock()
        self._cache: Dict[str, ExecutionRecord] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                cid = rec.get("client_order_id")
                if not cid:
                    continue
                self._cache[cid] = ExecutionRecord(**{**rec})
        except Exception:
            # Corruption should not crash startup; continue with empty ledger.
            self._cache = {}

    def _append(self, rec: ExecutionRecord) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec), ensure_ascii=True) + "\n")
                fh.flush()
                try:
                    import os
                    os.fsync(fh.fileno())
                except Exception:
                    pass

    def get(self, client_order_id: str) -> Optional[ExecutionRecord]:
        return self._cache.get(client_order_id)

    def mark(self, rec: ExecutionRecord) -> None:
        self._cache[rec.client_order_id] = rec
        self._append(rec)

    def mark_error(self, client_order_id: str, error: str) -> None:
        existing = self._cache.get(client_order_id)
        if existing:
            existing.status = "error"
            existing.error = error
            existing.ts = time.time()
            self._append(existing)
        else:
            self.mark(
                ExecutionRecord(
                    client_order_id=client_order_id,
                    status="error",
                    instrument="unknown",
                    side="",
                    qty=0,
                    stop_price=None,
                    target_price=None,
                    oco_id=None,
                    error=error,
                )
            )
