from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time, timedelta
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from pytz import (
    AmbiguousTimeError as PytzAmbiguousTimeError,
    NonExistentTimeError as PytzNonExistentTimeError,
    timezone as pytz_timezone,
)
from zoneinfo import ZoneInfo

__all__ = [
    "SessionDefinition",
    "ensure_dataframe_index",
    "local_index",
    "localize_with_dst",
]


def _parse_time(value: str, fallback: str) -> time:
    raw = str(value or fallback)
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time string: {value!r}")
    return time(hour=int(parts[0]), minute=int(parts[1]))


def _minutes_since_midnight(val: time) -> int:
    return val.hour * 60 + val.minute


def localize_with_dst(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """
    Localize a tz-naive DatetimeIndex to the requested timezone while handling DST gaps.
    """
    di = pd.DatetimeIndex(index)
    if di.tz is not None:
        return di.tz_convert(tz)
    zone = pytz_timezone(tz)
    localized = []
    for ts in di.to_pydatetime():
        naive = ts.replace(tzinfo=None)
        try:
            localized.append(zone.localize(naive, is_dst=None))
        except PytzNonExistentTimeError:
            adjusted = naive + timedelta(minutes=1)
            localized.append(zone.localize(adjusted, is_dst=True))
        except PytzAmbiguousTimeError:
            localized.append(zone.localize(naive, is_dst=True))
    return pd.DatetimeIndex(localized)


def ensure_dataframe_index(df: pd.DataFrame, tz: str, *, naive_is_utc: bool = True) -> pd.DataFrame:
    """
    Ensure a tz-aware DatetimeIndex for the given dataframe in the requested session tz.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "Datetime" in out.columns:
            out = out.set_index("Datetime")
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'Datetime' column.")
    try:
        out.index = pd.to_datetime(out.index, errors="coerce")
    except ValueError:
        # Some inputs include explicit offsets that vary across DST (e.g. -07:00 vs -06:00),
        # which pandas treats as "mixed timezones" unless coerced to UTC during parsing.
        out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
    if not isinstance(out.index, pd.DatetimeIndex):
        # Mixed tz offsets can yield an object Index; force utc-aware parsing.
        out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is None:
        if naive_is_utc:
            out.index = out.index.tz_localize("UTC").tz_convert(tz)
        else:
            out.index = localize_with_dst(out.index, tz)
    else:
        out.index = out.index.tz_convert(tz)
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out


def local_index(idx: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """
    Return a DatetimeIndex converted to the requested session timezone.
    """
    di = pd.DatetimeIndex(idx)
    if di.tz is None:
        return localize_with_dst(di, tz)
    return di.tz_convert(tz)


def _window_mask(minutes: np.ndarray, start_min: int, end_min: int) -> np.ndarray:
    if start_min == end_min:
        return np.zeros_like(minutes, dtype=bool)
    if start_min < end_min:
        return (minutes >= start_min) & (minutes < end_min)
    return (minutes >= start_min) | (minutes < end_min)


@dataclass(frozen=True)
class SessionDefinition:
    tz: str = "America/Denver"
    rth_start: time = field(default_factory=lambda: time(7, 30))
    rth_end: time = field(default_factory=lambda: time(14, 0))
    orb_minutes: int = 15
    overnight_start: Optional[time] = None
    overnight_end: Optional[time] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_zone", ZoneInfo(self.tz))
        object.__setattr__(self, "_rth_start_min", _minutes_since_midnight(self.rth_start))
        object.__setattr__(self, "_rth_end_min", _minutes_since_midnight(self.rth_end))
        ovn_start = self.overnight_start or self.rth_end
        ovn_end = self.overnight_end or self.rth_start
        object.__setattr__(self, "overnight_start", ovn_start)
        object.__setattr__(self, "overnight_end", ovn_end)
        object.__setattr__(self, "_overnight_start_min", _minutes_since_midnight(ovn_start))
        object.__setattr__(self, "_overnight_end_min", _minutes_since_midnight(ovn_end))

    @classmethod
    def from_strings(
        cls,
        *,
        tz: str = "America/Denver",
        rth_start: str = "07:30",
        rth_end: str = "14:00",
        orb_minutes: int = 15,
        overnight_start: Optional[str] = None,
        overnight_end: Optional[str] = None,
    ) -> "SessionDefinition":
        return cls(
            tz=tz,
            rth_start=_parse_time(rth_start, "07:30"),
            rth_end=_parse_time(rth_end, "14:00"),
            orb_minutes=int(orb_minutes),
            overnight_start=_parse_time(overnight_start, rth_end) if overnight_start else None,
            overnight_end=_parse_time(overnight_end, rth_start) if overnight_end else None,
        )

    def align_index(self, idx: pd.DatetimeIndex, *, naive_is_utc: bool = False) -> pd.DatetimeIndex:
        """
        Convert an index to the session timezone.
        """
        di = pd.DatetimeIndex(idx)
        if di.tz is None:
            if naive_is_utc:
                di = di.tz_localize("UTC")
            else:
                di = localize_with_dst(di, self.tz)
        return di.tz_convert(self.tz)

    def _minutes_array(self, idx_local: pd.DatetimeIndex) -> np.ndarray:
        return idx_local.hour * 60 + idx_local.minute

    def _rollover_mask(self, minutes: np.ndarray) -> np.ndarray:
        start = self._rth_start_min
        end = self._rth_end_min
        if start < end:
            return minutes >= end
        return (minutes >= end) & (minutes < start)

    def session_index(self, idx_local: pd.DatetimeIndex, *, name: str = "session_date") -> pd.Series:
        """
        Return per-bar session dates with rollover applied at the RTH end.
        """
        idx_local = self.align_index(idx_local)
        minutes = self._minutes_array(idx_local)
        base = pd.Series(idx_local.normalize(), index=idx_local, name=name)
        rollover_mask = self._rollover_mask(minutes)
        if rollover_mask.any():
            base.loc[rollover_mask] = base.loc[rollover_mask] + pd.Timedelta(days=1)
        return base

    def session_open_datetimes(self, session_index: pd.Series) -> pd.DatetimeIndex:
        base = pd.DatetimeIndex(session_index.to_numpy())
        offset = pd.to_timedelta(self._rth_start_min, unit="m")
        return base + offset

    def minutes_since_open(
        self,
        idx_local: pd.DatetimeIndex,
        session_index: Optional[pd.Series] = None,
    ) -> pd.Series:
        idx_local = self.align_index(idx_local)
        if session_index is None:
            session_index = self.session_index(idx_local)
        opens = self.session_open_datetimes(session_index)
        delta = (pd.Series(idx_local, index=idx_local) - opens).dt.total_seconds() / 60.0
        return delta

    def in_rth_mask(self, idx_local: pd.DatetimeIndex) -> pd.Series:
        idx_local = self.align_index(idx_local)
        mins = self._minutes_array(idx_local)
        mask = _window_mask(mins, self._rth_start_min, self._rth_end_min)
        return pd.Series(mask, index=idx_local)

    def in_overnight_mask(self, idx_local: pd.DatetimeIndex) -> pd.Series:
        idx_local = self.align_index(idx_local)
        mins = self._minutes_array(idx_local)
        mask = _window_mask(mins, self._overnight_start_min, self._overnight_end_min)
        return pd.Series(mask, index=idx_local)

    def orb_mask(self, idx_local: pd.DatetimeIndex, session_index: Optional[pd.Series] = None) -> pd.Series:
        session_index = session_index or self.session_index(idx_local)
        mins_from_open = self.minutes_since_open(idx_local, session_index=session_index)
        mask = (mins_from_open >= 0) & (mins_from_open < self.orb_minutes)
        return pd.Series(mask.values, index=idx_local)

    def metadata(self) -> dict:
        return {
            "tz": self.tz,
            "rth_start": self.rth_start.strftime("%H:%M"),
            "rth_end": self.rth_end.strftime("%H:%M"),
            "orb_minutes": int(self.orb_minutes),
            "overnight_start": (self.overnight_start or self.rth_end).strftime("%H:%M"),
            "overnight_end": (self.overnight_end or self.rth_start).strftime("%H:%M"),
        }
