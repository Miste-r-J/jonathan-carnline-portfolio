from __future__ import annotations

import json
import logging
import random
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiscordDeliveryJob:
    job_id: str
    channel_id: str
    kind: str
    dedupe_key: Optional[str]
    payload: Dict[str, Any]
    attempt_count: int
    next_attempt_ts: float
    status: str


class DiscordDeliveryError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retry_after: Optional[float] = None,
        retryable: bool = True,
        status_code: Optional[int] = None,
        body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.retryable = retryable
        self.status_code = status_code
        self.body = body


class _TokenBucket:
    def __init__(self, *, rate_per_sec: float, burst: int) -> None:
        self.rate_per_sec = max(0.01, float(rate_per_sec))
        self.capacity = max(1.0, float(burst))
        self.tokens = self.capacity
        self.updated_at = time.monotonic()

    def wait_time(self) -> float:
        now = time.monotonic()
        elapsed = max(0.0, now - self.updated_at)
        self.updated_at = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0
        missing = 1.0 - self.tokens
        self.tokens = 0.0
        return missing / self.rate_per_sec


class DiscordDeliveryQueue:
    def __init__(
        self,
        *,
        send_callable: Callable[[Dict[str, Any]], None],
        dead_letter_path: Path,
        max_attempts: int = 5,
        max_total_wait_sec: float = 45.0,
        global_rate_per_sec: float = 4.0,
        global_burst: int = 8,
        channel_rate_per_sec: float = 1.0,
        channel_burst: int = 3,
        dedupe_window: int = 2048,
    ) -> None:
        self._send_callable = send_callable
        self._dead_letter_path = Path(dead_letter_path)
        self._max_attempts = max(1, int(max_attempts))
        self._max_total_wait_sec = max(0.0, float(max_total_wait_sec))
        self._global_bucket = _TokenBucket(rate_per_sec=global_rate_per_sec, burst=global_burst)
        self._channel_buckets: Dict[str, _TokenBucket] = {}
        self._channel_rate_per_sec = channel_rate_per_sec
        self._channel_burst = channel_burst
        self._seen_keys: Deque[str] = deque(maxlen=max(32, int(dedupe_window)))

    def _channel_bucket(self, channel_id: str) -> _TokenBucket:
        bucket = self._channel_buckets.get(channel_id)
        if bucket is None:
            bucket = _TokenBucket(rate_per_sec=self._channel_rate_per_sec, burst=self._channel_burst)
            self._channel_buckets[channel_id] = bucket
        return bucket

    def submit(
        self,
        *,
        channel_id: str,
        kind: str,
        payload: Dict[str, Any],
        dedupe_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if dedupe_key and dedupe_key in self._seen_keys:
            return {"status": "deduped", "dedupe_key": dedupe_key}

        job = DiscordDeliveryJob(
            job_id=str(uuid.uuid4()),
            channel_id=str(channel_id or "default"),
            kind=str(kind or "message"),
            dedupe_key=dedupe_key,
            payload=dict(payload),
            attempt_count=0,
            next_attempt_ts=time.time(),
            status="queued",
        )
        started = time.monotonic()
        last_error: Optional[DiscordDeliveryError] = None

        while job.attempt_count < self._max_attempts:
            bucket_wait = max(
                self._global_bucket.wait_time(),
                self._channel_bucket(job.channel_id).wait_time(),
            )
            if bucket_wait > 0:
                time.sleep(bucket_wait)

            if self._max_total_wait_sec and (time.monotonic() - started) > self._max_total_wait_sec:
                break

            job.attempt_count += 1
            job.status = "in_progress"
            try:
                self._send_callable(job.payload)
                job.status = "sent"
                if dedupe_key:
                    self._seen_keys.append(dedupe_key)
                return asdict(job)
            except DiscordDeliveryError as exc:
                last_error = exc
                retry_after = exc.retry_after
                if not exc.retryable:
                    break
                delay = retry_after if retry_after is not None else min(8.0, 0.5 * (2 ** (job.attempt_count - 1)))
                delay = max(0.05, float(delay)) + random.uniform(0.0, min(1.0, delay * 0.25))
                job.next_attempt_ts = time.time() + delay
                job.status = "retrying"
                time.sleep(delay)
            except Exception as exc:  # pragma: no cover - defensive
                last_error = DiscordDeliveryError(str(exc), retryable=True)
                delay = min(8.0, 0.5 * (2 ** (job.attempt_count - 1))) + random.uniform(0.0, 0.5)
                job.next_attempt_ts = time.time() + delay
                job.status = "retrying"
                time.sleep(delay)

        job.status = "dead_letter"
        self._write_dead_letter(job, last_error)
        return asdict(job)

    def _write_dead_letter(self, job: DiscordDeliveryJob, error: Optional[DiscordDeliveryError]) -> None:
        payload = asdict(job)
        payload["dead_lettered_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if error is not None:
            payload["error"] = {
                "message": str(error),
                "retry_after": error.retry_after,
                "retryable": error.retryable,
                "status_code": error.status_code,
                "body": error.body,
            }
        try:
            self._dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
            with self._dead_letter_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        except Exception:
            logger.warning("Failed to write Discord dead-letter record", exc_info=True)
