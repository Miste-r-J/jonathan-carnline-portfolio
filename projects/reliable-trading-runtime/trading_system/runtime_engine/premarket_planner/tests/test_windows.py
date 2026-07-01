from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo

from trading_system.runtime_engine.premarket_planner.config import (
    BackfillConfig,
    DataConfig,
    MetricsConfig,
    OutputConfig,
    PlannerConfig,
    WindowsConfig,
)
from trading_system.runtime_engine.premarket_planner.core import _compute_overnight_window


def _make_config() -> PlannerConfig:
    return PlannerConfig(
        enabled=True,
        instrument="ES",
        session_tz="America/Denver",
        rth_start=time(hour=6, minute=30),
        rth_end=time(hour=12, minute=59),
        emit_time_local=time(hour=6, minute=30),
        data=DataConfig(csv_path=Path("/tmp/es.csv")),
        windows=WindowsConfig(
            eth_start_hour=13,
            eth_end_hour=7,
            eth_end_minute=29,
            min_eth_bars=30,
            last_hours_fallback=8,
        ),
        metrics=MetricsConfig(atr_len=14, atr_timeframe="5m", compute_rth_vwap=True),
        output=OutputConfig(discord_webhook="", round_decimals=2),
        backfill=BackfillConfig(enabled=True, hours=24),
    )


def _make_frame(
    start: datetime,
    end: datetime,
    freq: str = "5min",
) -> pd.DataFrame:
    index = pd.date_range(start=start, end=end, freq=freq)
    if index.empty or index[-1] != end:
        extra = pd.DatetimeIndex([end])
        index = index.union(extra)
    return pd.DataFrame(
        {
            "Open": range(len(index)),
            "High": range(len(index)),
            "Low": range(len(index)),
            "Close": range(len(index)),
            "Volume": [100] * len(index),
        },
        index=index,
    )


def test_eth_window_without_fallback():
    tz = ZoneInfo("America/Denver")
    config = _make_config()
    window_end = datetime(2024, 3, 2, 7, 29, tzinfo=tz)
    window_start = datetime(2024, 3, 1, 13, 0, tzinfo=tz)
    df = _make_frame(window_start, window_end)

    window = _compute_overnight_window(df, config, window_end, _noop_logger())

    assert not window.used_fallback
    assert window.initial_bars == window.bars == len(df)
    assert window.start == window_start
    assert window.end == window_end
    assert window.frame.index.max() == window_end


def test_eth_window_with_fallback_when_insufficient():
    tz = ZoneInfo("America/Denver")
    config = _make_config()
    window_end = datetime(2024, 3, 2, 7, 29, tzinfo=tz)
    fallback_start = window_end - timedelta(hours=2)
    df = _make_frame(fallback_start, window_end)

    window = _compute_overnight_window(df, config, window_end, _noop_logger())

    assert window.used_fallback is True
    assert window.initial_bars < config.windows.min_eth_bars
    assert window.bars == len(df)
    assert window.frame.index.min() == fallback_start
    assert window.fallback_reason is not None


def test_eth_window_reports_empty_when_no_data():
    tz = ZoneInfo("America/Denver")
    config = _make_config()
    window_end = datetime(2024, 3, 2, 7, 29, tzinfo=tz)
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df.index = pd.DatetimeIndex([], tz=tz)

    window = _compute_overnight_window(df, config, window_end, _noop_logger())

    assert window.used_fallback is True
    assert window.bars == 0
    assert window.frame.empty


def _noop_logger():
    class _Logger:
        def debug(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    return _Logger()
