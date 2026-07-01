from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Sequence, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

from na.bot.config import INSTRUMENTS

Direction = Literal["bullish", "bearish"]
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close")

__all__ = [
    "BOSConfig",
    "StructureDetector",
    "StructureEvent",
    "StructureReport",
]


@dataclass(frozen=True)
class BOSConfig:
    swing_lookback: int = 3
    min_swing_separation: int = 2
    analysis_hours: int = 48
    retest_window_bars: int = 60
    retest_tolerance_ticks: float = 2.0
    min_break_displacement_ticks: float = 1.0
    require_close_through_level: bool = True
    tick_size: Optional[float] = None

    def __post_init__(self) -> None:
        if self.swing_lookback < 1:
            raise ValueError("swing_lookback must be >= 1")
        if self.min_swing_separation < 1:
            raise ValueError("min_swing_separation must be >= 1")
        if self.analysis_hours < 1:
            raise ValueError("analysis_hours must be >= 1")
        if self.retest_window_bars < 1:
            raise ValueError("retest_window_bars must be >= 1")
        if self.retest_tolerance_ticks < 0:
            raise ValueError("retest_tolerance_ticks must be >= 0")
        if self.min_break_displacement_ticks < 0:
            raise ValueError("min_break_displacement_ticks must be >= 0")


@dataclass(frozen=True)
class StructureEvent:
    direction: Direction
    structure_origin_time: datetime
    structure_origin_price: float
    break_time: datetime
    break_level: float
    break_price: float
    displacement: float
    retest_hit: bool
    retest_time: Optional[datetime] = None
    retest_price: Optional[float] = None
    retest_distance: Optional[float] = None

    def describe(self, tz: Optional[ZoneInfo] = None) -> str:
        label = "Bullish BOS" if self.direction == "bullish" else "Bearish BOS"
        break_dt = self.break_time if tz is None else self.break_time.astimezone(tz)
        break_label = break_dt.strftime("%m/%d %H:%M")
        line = (
            f"- {label} at {self.break_level:.2f} ({break_label}) "
            f"cleared by {self.displacement:+.2f}"
        )
        if self.retest_hit and self.retest_time:
            retest_dt = self.retest_time if tz is None else self.retest_time.astimezone(tz)
            retest_label = retest_dt.strftime("%m/%d %H:%M")
            delta = 0.0 if self.retest_distance is None else self.retest_distance
            line = f"{line}; retest {retest_label} (delta {delta:+.2f})"
        else:
            line = f"{line}; retest pending"
        return line


@dataclass(frozen=True)
class StructureReport:
    instrument: str
    window_hours: int
    events: Tuple[StructureEvent, ...]
    notes: Tuple[str, ...]
    last_bar_time: Optional[datetime]

    def to_markdown(self, tz: Optional[ZoneInfo] = None) -> str:
        lines: List[str] = []
        if self.events:
            lines.extend(
                event.describe(tz)
                for event in sorted(self.events, key=lambda e: e.break_time)
            )
        else:
            lines.append(f"- No break of structure detected in the last {self.window_hours}h.")
        if self.notes:
            lines.append("- Notes: " + "; ".join(self.notes))
        return "\n".join(lines)


@dataclass(frozen=True)
class SwingPoint:
    kind: Literal["high", "low"]
    index: int
    price: float
    timestamp: datetime


class StructureDetector:
    """
    Identifies break-of-structure events and optional retests for the premarket planner.
    """

    def __init__(self, instrument: str = "ES", config: Optional[BOSConfig] = None) -> None:
        self.instrument = instrument.upper()
        self.config = config or BOSConfig()
        self._tick_size = self._resolve_tick_size()

    def analyze(self, frame: pd.DataFrame) -> StructureReport:
        prepared = self._prepare_frame(frame)
        last_bar = prepared.index.max() if not prepared.empty else None
        notes: List[str] = []

        min_bars = self.config.swing_lookback * 2 + self.config.min_swing_separation
        if len(prepared) < min_bars:
            notes.append(f"Need at least {min_bars} bars but only have {len(prepared)}.")
            return StructureReport(
                instrument=self.instrument,
                window_hours=self.config.analysis_hours,
                events=tuple(),
                notes=tuple(notes),
                last_bar_time=last_bar,
            )

        swings = self._detect_swings(prepared)
        if not swings:
            notes.append("No qualified swing highs or lows detected.")
            return StructureReport(
                instrument=self.instrument,
                window_hours=self.config.analysis_hours,
                events=tuple(),
                notes=tuple(notes),
                last_bar_time=last_bar,
            )

        events = self._detect_events(prepared, swings)
        if not events:
            notes.append("No confirmed break of structure in the analysis window.")

        return StructureReport(
            instrument=self.instrument,
            window_hours=self.config.analysis_hours,
            events=tuple(events),
            notes=tuple(notes),
            last_bar_time=last_bar,
        )

    def _resolve_tick_size(self) -> float:
        if self.config.tick_size and self.config.tick_size > 0:
            return float(self.config.tick_size)
        spec = INSTRUMENTS.get(self.instrument)
        tick = getattr(spec, "tick_size", None)
        if isinstance(tick, (int, float)) and tick > 0:
            return float(tick)
        return 0.25

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("analyze expects a pandas DataFrame")
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Frame missing required columns: {', '.join(missing)}")

        frame = df.copy()
        if "Datetime" in frame.columns:
            frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="coerce")
            frame = frame.dropna(subset=["Datetime"])
            frame = frame.set_index("Datetime")
        elif not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must use a DatetimeIndex or contain a Datetime column")

        frame = frame.sort_index()
        numeric_cols = {col: "float64" for col in REQUIRED_COLUMNS}
        frame = frame.astype(numeric_cols, errors="ignore")

        if frame.empty:
            return frame

        cutoff = frame.index.max() - timedelta(hours=self.config.analysis_hours)
        frame = frame[frame.index >= cutoff]

        return frame

    def _detect_swings(self, frame: pd.DataFrame) -> List[SwingPoint]:
        highs = frame["High"].astype(float).to_numpy()
        lows = frame["Low"].astype(float).to_numpy()
        timestamps = frame.index.to_list()

        swings: List[SwingPoint] = []
        last_index = {"high": -10_000, "low": -10_000}
        lookback = self.config.swing_lookback
        min_sep = self.config.min_swing_separation

        for idx in range(lookback, len(frame) - lookback):
            window_slice = slice(idx - lookback, idx + lookback + 1)

            candidate_high = highs[idx]
            if candidate_high >= highs[window_slice].max() and (idx - last_index["high"]) >= min_sep:
                swings.append(SwingPoint("high", idx, candidate_high, timestamps[idx]))
                last_index["high"] = idx
                continue

            candidate_low = lows[idx]
            if candidate_low <= lows[window_slice].min() and (idx - last_index["low"]) >= min_sep:
                swings.append(SwingPoint("low", idx, candidate_low, timestamps[idx]))
                last_index["low"] = idx

        return swings

    def _detect_events(self, frame: pd.DataFrame, swings: Sequence[SwingPoint]) -> List[StructureEvent]:
        closes = frame["Close"].astype(float).to_numpy()
        highs = frame["High"].astype(float).to_numpy()
        lows = frame["Low"].astype(float).to_numpy()
        timestamps = frame.index.to_list()

        events: List[StructureEvent] = []
        seen_direction: set[Direction] = set()

        for swing in reversed(swings):
            direction: Direction = "bullish" if swing.kind == "high" else "bearish"
            if direction in seen_direction:
                continue
            event = self._detect_break_from_swing(
                swing=swing,
                closes=closes,
                highs=highs,
                lows=lows,
                timestamps=timestamps,
            )
            if event:
                events.append(event)
                seen_direction.add(direction)

        return events

    def _detect_break_from_swing(
        self,
        swing: SwingPoint,
        closes: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        timestamps: Sequence[datetime],
    ) -> Optional[StructureEvent]:
        target = float(swing.price)
        displacement = self.config.min_break_displacement_ticks * self._tick_size
        tolerance = self.config.retest_tolerance_ticks * self._tick_size

        start = swing.index + 1
        if start >= len(closes):
            return None

        break_idx = None
        for idx in range(start, len(closes)):
            close = closes[idx]
            high = highs[idx]
            low = lows[idx]
            if swing.kind == "high":
                threshold = target + displacement
                crossed = close >= threshold if self.config.require_close_through_level else high >= threshold
            else:
                threshold = target - displacement
                crossed = close <= threshold if self.config.require_close_through_level else low <= threshold
            if crossed:
                break_idx = idx
                break

        if break_idx is None:
            return None

        break_price = float(closes[break_idx])
        direction: Direction = "bullish" if swing.kind == "high" else "bearish"

        retest_idx = None
        band_low = target - tolerance
        band_high = target + tolerance
        end = min(len(closes), break_idx + 1 + self.config.retest_window_bars)
        for idx in range(break_idx + 1, end):
            high = highs[idx]
            low = lows[idx]
            touched = low <= band_high and high >= band_low
            if touched:
                retest_idx = idx
                break

        displacement_value = break_price - target if direction == "bullish" else target - break_price

        if retest_idx is not None:
            retest_price = float(closes[retest_idx])
            retest_distance = retest_price - target
            retest_time = timestamps[retest_idx]
            retest_hit = True
        else:
            retest_price = None
            retest_distance = None
            retest_time = None
            retest_hit = False

        return StructureEvent(
            direction=direction,
            structure_origin_time=timestamps[swing.index],
            structure_origin_price=target,
            break_time=timestamps[break_idx],
            break_level=target,
            break_price=break_price,
            displacement=displacement_value,
            retest_hit=retest_hit,
            retest_time=retest_time,
            retest_price=retest_price,
            retest_distance=retest_distance,
        )
