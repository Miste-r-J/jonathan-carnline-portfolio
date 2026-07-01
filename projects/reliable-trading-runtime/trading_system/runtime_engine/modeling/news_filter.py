"""News risk filter for blocking entries during high-impact events."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Protocol
import requests

logger = logging.getLogger(__name__)


class NewsProvider(Protocol):
    """Protocol for news data providers."""

    def get_high_impact_events(self, instrument: str, since: datetime) -> List[dict]:
        """Get high-impact news events for instrument since given time."""
        ...


class ForexFactoryProvider:
    """ForexFactory news provider (placeholder implementation)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://www.forexfactory.com"  # Placeholder

    def get_high_impact_events(self, instrument: str, since: datetime) -> List[dict]:
        """Fetch high-impact news events."""
        # Placeholder implementation - in practice would call ForexFactory API
        # Return mock data for now
        return [
            {
                "timestamp": datetime.now(),
                "impact": "high",
                "currency": instrument.replace("F", ""),  # ESF -> ES, etc.
                "title": "Mock High Impact Event"
            }
        ]


class NewsFilter:
    """Filters trading activity based on news events."""

    def __init__(self,
                 provider: NewsProvider,
                 impact_levels: List[str],
                 blackout_before_min: int = 0,
                 blackout_after_min: int = 0):
        self.provider = provider
        self.impact_levels = set(impact_levels)
        self.blackout_before = timedelta(minutes=blackout_before_min)
        self.blackout_after = timedelta(minutes=blackout_after_min)
        self._last_check = datetime.min
        self._events_cache: List[dict] = []

    def should_block_entry(self, instrument: str, current_time: datetime) -> tuple[bool, str]:
        """Check if entry should be blocked due to news events."""
        # Refresh cache if needed (every 5 minutes)
        if (current_time - self._last_check).total_seconds() > 300:
            try:
                since = current_time - timedelta(hours=24)  # Look back 24 hours
                self._events_cache = self.provider.get_high_impact_events(instrument, since)
                self._last_check = current_time
            except Exception as e:
                logger.warning(f"Failed to fetch news events: {e}")
                return False, ""

        # Check if current time falls within any blackout window
        for event in self._events_cache:
            if event.get("impact", "").lower() not in self.impact_levels:
                continue

            event_time = event.get("timestamp")
            if not event_time:
                continue

            event_time = event_time if isinstance(event_time, datetime) else datetime.fromisoformat(event_time)

            blackout_start = event_time - self.blackout_before
            blackout_end = event_time + self.blackout_after

            if blackout_start <= current_time <= blackout_end:
                return True, f"News blackout: {event.get('title', 'High impact event')}"

        return False, ""

    def should_force_flat(self, instrument: str, current_time: datetime) -> tuple[bool, str]:
        """Check if positions should be forced flat due to news."""
        # For now, same logic as blocking entries
        # Could be extended to have different thresholds
        return self.should_block_entry(instrument, current_time)

    def update_last_event(self, event_time: datetime):
        """Update the last known news event time."""
        # Could be used to manually set events from external sources
        pass
