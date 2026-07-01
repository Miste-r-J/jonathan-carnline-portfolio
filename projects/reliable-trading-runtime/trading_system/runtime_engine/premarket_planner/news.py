from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from zoneinfo import ZoneInfo

from .config import PlannerConfig


@dataclass(frozen=True)
class NewsEvent:
    scheduled_at: datetime
    impact: str
    title: str
    currency: str = ""


def _resolve_time_column(columns: Iterable[str]) -> Optional[str]:
    col_map = {str(column).lower(): str(column) for column in columns}
    for candidate in ("timestamp", "datetime", "time", "event_time", "scheduled", "time_local"):
        if candidate in col_map:
            return col_map[candidate]
    return None


def load_news_events(config: PlannerConfig) -> list[NewsEvent]:
    path = config.news_csv_path
    if path is None or not Path(path).exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []

    time_col = _resolve_time_column(df.columns)
    if time_col is None:
        return []

    impact_col = next((column for column in df.columns if str(column).lower() in {"impact", "importance", "color", "level"}), None)
    title_col = next((column for column in df.columns if str(column).lower() in {"title", "event", "headline", "description"}), None)
    currency_col = next((column for column in df.columns if str(column).lower() in {"currency", "ccy", "symbol"}), None)

    tz = ZoneInfo(config.news_source_tz or config.session_tz)
    events: list[NewsEvent] = []
    parsed = pd.to_datetime(df[time_col], errors="coerce")
    for idx, timestamp in parsed.items():
        if pd.isna(timestamp):
            continue
        dt = timestamp.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        else:
            dt = dt.astimezone(tz)
        impact = str(df.iloc[idx][impact_col]).strip() if impact_col else ""
        title = str(df.iloc[idx][title_col]).strip() if title_col else "Scheduled event"
        currency = str(df.iloc[idx][currency_col]).strip() if currency_col else ""
        events.append(NewsEvent(scheduled_at=dt, impact=impact, title=title, currency=currency))
    events.sort(key=lambda item: item.scheduled_at)
    return events


def build_news_summary(config: PlannerConfig, *, now: Optional[datetime] = None, limit: int = 5) -> str:
    if config.news_csv_path is None:
        return "No news source configured. Set `news_csv_path` or `AUTOMATION_NEWS_CSV`."
    if not config.news_csv_path.exists():
        return f"News source not found: {config.news_csv_path}"

    events = load_news_events(config)
    if not events:
        return "No scheduled news events found in the configured news source."

    tz = ZoneInfo(config.news_source_tz or config.session_tz)
    current = now.astimezone(tz) if now and now.tzinfo else (now.replace(tzinfo=tz) if now else datetime.now(tz))
    before = timedelta(minutes=config.news_blackout_before_min)
    after = timedelta(minutes=config.news_blackout_after_min)
    active = [
        event
        for event in events
        if (event.scheduled_at - before) <= current <= (event.scheduled_at + after)
    ]
    upcoming = [event for event in events if event.scheduled_at >= current][:limit]

    lines = ["News risk summary"]
    if active:
        lines.append(f"- Active blackout: {active[0].title} at {active[0].scheduled_at.strftime('%Y-%m-%d %H:%M %Z')}")
    else:
        lines.append("- Active blackout: none")

    if not upcoming:
        lines.append("- Upcoming events: none")
        return "\n".join(lines)

    lines.append("- Upcoming events:")
    for event in upcoming:
        details = [event.scheduled_at.strftime("%Y-%m-%d %H:%M %Z")]
        if event.impact:
            details.append(event.impact.upper())
        if event.currency:
            details.append(event.currency.upper())
        details.append(event.title)
        lines.append(f"- {' | '.join(details)}")
    return "\n".join(lines)
