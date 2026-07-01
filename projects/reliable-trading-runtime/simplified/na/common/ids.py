from __future__ import annotations

from datetime import datetime
from typing import Optional

_DEFAULT_TS_FORMAT = "%Y%m%d%H%M%S"
_INVALID_FS_CHARS = {c: "_" for c in '<>:"/\\|?*'}


def _sanitize(text: str) -> str:
    return text.translate(_INVALID_FS_CHARS).replace(":", "_").replace(" ", "_")


def _timestamp_fragment(ts: Optional[datetime] = None) -> str:
    stamp = ts or datetime.utcnow()
    return stamp.strftime(_DEFAULT_TS_FORMAT)


def generate_model_id(
    instrument: str,
    label_domain: str,
    horizon: int,
    *,
    timestamp: Optional[datetime] = None,
    suffix: Optional[str] = None,
) -> str:
    base = f"{instrument.upper()}:{label_domain}:{int(horizon)}:{_timestamp_fragment(timestamp)}"
    base = _sanitize(base)
    if suffix:
        return _sanitize(f"{base}-{suffix}")
    return base


def generate_preset_id(instrument: str, preset_name: str, *, timestamp: Optional[datetime] = None) -> str:
    ident = f"{instrument.upper()}:{_sanitize(preset_name).lower()}:{_timestamp_fragment(timestamp)}"
    return _sanitize(ident)


__all__ = ["generate_model_id", "generate_preset_id"]
