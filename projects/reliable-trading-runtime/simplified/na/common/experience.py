from __future__ import annotations

"""
Trade experience schema + helpers shared across live streaming and online learning.

The streamer appends a JSON line for every completed trade, which allows the online
learner/evolution stack to repeatedly build sliding-window datasets without touching
execution logs or Discord output.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Experience:
    """
    Canonical, risk-aware view of a single trade.

    Attributes:
        trade_id: Stable identifier (broker/fx id or synthetic fallback).
        instrument: Symbol or alias (ES, NQ, etc.).
        strategy: Strategy/preset identifier.
        preset: Preset name applied at runtime.
        model_run_id: Champion/model identifier used by the streamer.
        timestamp: Exit timestamp (UTC or local; stored in ISO-8601).
        side: "LONG"/"SHORT".
        qty: Contracts/shares traded.
        entry_price: Entry fill price.
        exit_price: Exit fill price.
        pnl: Net PnL in account currency.
        r_multiple: Realized R multiple (risk-normalized).
        base_r_multiple: Counterfactual R from base (pre-online) action.
        features: Feature vector captured at entry.
        risk_flags: Risk guard flags active for the trade (lockouts, news, etc.).
        metadata: Extra structured data (probabilities, calibration info, etc.).
    """

    trade_id: str
    instrument: str
    strategy: str
    preset: str
    model_run_id: Optional[str]
    timestamp: datetime
    side: str
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    r_multiple: float
    base_r_multiple: float
    features: Sequence[float] = field(default_factory=tuple)
    risk_flags: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        payload["features"] = list(self.features)
        payload["risk_flags"] = list(self.risk_flags)
        payload["metadata"] = dict(self.metadata)
        return payload

    @staticmethod
    def from_record(data: Mapping[str, Any]) -> "Experience":
        ts_raw = data.get("timestamp")
        if isinstance(ts_raw, str):
            timestamp = datetime.fromisoformat(ts_raw)
        elif isinstance(ts_raw, datetime):
            timestamp = ts_raw
        else:  # pragma: no cover - guardrail
            timestamp = datetime.utcnow()
        return Experience(
            trade_id=str(data.get("trade_id") or ""),
            instrument=str(data.get("instrument") or ""),
            strategy=str(data.get("strategy") or ""),
            preset=str(data.get("preset") or ""),
            model_run_id=str(data.get("model_run_id")) if data.get("model_run_id") is not None else None,
            timestamp=timestamp,
            side=str(data.get("side") or ""),
            qty=float(data.get("qty") or 0.0),
            entry_price=float(data.get("entry_price") or 0.0),
            exit_price=float(data.get("exit_price") or 0.0),
            pnl=float(data.get("pnl") or 0.0),
            r_multiple=float(data.get("r_multiple") or 0.0),
            base_r_multiple=float(data.get("base_r_multiple") or 0.0),
            features=tuple(float(x) for x in data.get("features") or ()),
            risk_flags=tuple(str(x) for x in data.get("risk_flags") or ()),
            metadata=dict(data.get("metadata") or {}),
        )


class ExperienceWriter:
    """Append-only JSONL writer for Experience records."""

    def __init__(self, directory: Path, symbol: str, pattern: str = "{symbol}_experience.jsonl") -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol.upper()
        self.pattern = pattern
        self.path = self.directory / pattern.format(symbol=self.symbol)

    def append(self, experience: Experience) -> None:
        record = experience.to_record()
        record.setdefault("symbol", self.symbol)
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


class ExperienceReader:
    """Utility to load recent experiences for online learning."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def iter_records(self) -> Iterator[Experience]:
        if not self.path.exists():
            return iter(())

        def _generator() -> Iterator[Experience]:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield Experience.from_record(payload)

        return _generator()

    def recent(
        self,
        *,
        window_days: Optional[int] = None,
        max_samples: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> List[Experience]:
        records: List[Experience] = []
        cutoff = None
        if window_days is not None:
            cutoff = (now or datetime.utcnow()) - timedelta(days=window_days)
        for exp in self.iter_records():
            if cutoff and exp.timestamp < cutoff:
                continue
            records.append(exp)
        if max_samples is not None and max_samples > 0 and len(records) > max_samples:
            records = records[-max_samples:]
        return records


def max_drawdown(pnl_series: Iterable[float]) -> float:
    """Return the max drawdown (negative number) for a cumulative PnL series."""
    peak = 0.0
    trough = 0.0
    cumulative = 0.0
    worst = 0.0
    for value in pnl_series:
        cumulative += float(value)
        if cumulative > peak:
            peak = cumulative
            trough = cumulative
        if cumulative < trough:
            trough = cumulative
            drawdown = trough - peak
            worst = min(worst, drawdown)
    return worst


__all__ = ["Experience", "ExperienceWriter", "ExperienceReader", "max_drawdown"]
