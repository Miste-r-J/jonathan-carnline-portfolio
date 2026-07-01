from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

try:  # Optional dependency; Prometheus support enabled when available.
    from prometheus_client import CollectorRegistry, Counter, generate_latest
except Exception:  # pragma: no cover - optional dependency absent
    CollectorRegistry = None  # type: ignore
    Counter = None  # type: ignore
    generate_latest = None  # type: ignore


class MetricsSink(Protocol):
    def emit(self, record: Dict[str, Any]) -> None:  # pragma: no cover - interface
        ...


class StructuredLogMetricsSink:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("trading_runtime.metrics")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s metrics %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def emit(self, record: Dict[str, Any]) -> None:
        self.logger.info(json.dumps(record))


class InMemoryMetricsSink:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def emit(self, record: Dict[str, Any]) -> None:
        self.records.append(record)


class CompositeMetricsSink:
    def __init__(self, sinks: Sequence[MetricsSink]) -> None:
        self._sinks = list(sinks)

    def emit(self, record: Dict[str, Any]) -> None:
        for sink in self._sinks:
            sink.emit(record)


class PrometheusMetricsSink:
    """
    Simple bridge that materializes Counters per metric+label signature.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        if CollectorRegistry is None or Counter is None:  # pragma: no cover - handled by dependency
            raise RuntimeError("prometheus_client is not installed")
        self.registry = registry or CollectorRegistry()
        self._counters: Dict[Tuple[str, Tuple[str, ...]], Any] = {}

    def _normalize_metric_name(self, name: str) -> str:
        safe = []
        for ch in name:
            if ch.isalnum() or ch == "_":
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe)

    def emit(self, record: Dict[str, Any]) -> None:
        metric_name = self._normalize_metric_name(f"{record['namespace']}_{record['metric']}")
        label_keys = tuple(sorted(record.get("labels") or ()))
        key = (metric_name, label_keys)
        counter = self._counters.get(key)
        if counter is None:
            counter = Counter(metric_name, f"{metric_name} counter", labelnames=label_keys, registry=self.registry)
            self._counters[key] = counter
        labels = {k: str(record["labels"].get(k, "")) for k in label_keys}
        counter.labels(**labels).inc(float(record.get("value", 0.0) or 0.0))


@dataclass
class MetricsCollector:
    namespace: str
    sink: MetricsSink = StructuredLogMetricsSink()

    def incr(self, metric: str, value: float = 1.0, **labels: Any) -> None:
        payload = {
            "namespace": self.namespace,
            "metric": metric,
            "value": float(value),
            "labels": labels,
            "timestamp": time.time(),
        }
        self.sink.emit(payload)

    def event(self, metric: str, **labels: Any) -> None:
        self.incr(metric, value=1.0, **labels)


def render_prometheus(registry: CollectorRegistry) -> bytes:
    if generate_latest is None:  # pragma: no cover - dependency guard
        raise RuntimeError("prometheus_client is not installed")
    return generate_latest(registry)


__all__ = [
    "MetricsCollector",
    "StructuredLogMetricsSink",
    "InMemoryMetricsSink",
    "CompositeMetricsSink",
    "PrometheusMetricsSink",
    "MetricsSink",
    "render_prometheus",
]
