import pandas as pd
import pytest

from simplified.na.bot.labels_triple_barrier import (
    TripleBarrierParams,
    make_triple_barrier_labels,
)


def _build_df(ohlc, start="2024-01-02 07:30"):
    idx = pd.date_range(start=start, periods=len(ohlc), freq="5min", tz="America/Denver")
    frame = pd.DataFrame(ohlc, columns=["Open", "High", "Low", "Close"])
    frame["Datetime"] = idx
    return frame[["Datetime", "Open", "High", "Low", "Close"]]


BASE_PARAMS = TripleBarrierParams(stop_ticks=8, target_ticks=12, max_hold_bars=3)


def test_long_target_hit_returns_positive_r():
    df = _build_df(
        [
            (100.0, 100.0, 99.5, 100.0),
            (100.0, 104.0, 99.0, 103.0),  # entry bar
            (103.0, 103.5, 101.0, 102.0),
        ]
    )
    out = make_triple_barrier_labels(
        df,
        tz="America/Denver",
        rth_start="07:30",
        rth_end="14:00",
        params=BASE_PARAMS,
    )
    assert out["y_dir"].iloc[0] == 1
    assert out["y_r"].iloc[0] == pytest.approx(12 / 8)  # +1.5R
    assert out["long"]["exit"].iloc[0] == "target"


def test_short_direction_selected_when_long_stops():
    df = _build_df(
        [
            (100.0, 100.0, 99.5, 100.0),
            (100.0, 101.0, 96.5, 97.0),  # drop hard -> great short
            (97.0, 98.0, 94.0, 95.0),
        ]
    )
    out = make_triple_barrier_labels(
        df,
        tz="America/Denver",
        rth_start="07:30",
        rth_end="14:00",
        params=BASE_PARAMS,
    )
    assert out["y_dir"].iloc[0] == -1
    assert out["y_r"].iloc[0] == pytest.approx(12 / 8)
    assert out["short"]["exit"].iloc[0] == "target"


def test_tie_break_prefers_stop_when_both_hit_same_bar():
    params = TripleBarrierParams(stop_ticks=8, target_ticks=12, max_hold_bars=2, tie_break="stop_first")
    df = _build_df(
        [
            (100.0, 100.0, 99.5, 100.0),
            (100.0, 104.0, 97.5, 101.0),  # both stop/target reachable
            (101.0, 102.0, 99.0, 100.5),
        ]
    )
    out = make_triple_barrier_labels(
        df,
        tz="America/Denver",
        rth_start="07:30",
        rth_end="14:00",
        params=params,
    )
    assert out["long"]["exit"].iloc[0] == "stop"
    assert pd.isna(out["y_dir"].iloc[0]) or out["y_dir"].iloc[0] == 0


def test_timeout_uses_close_and_can_still_label_positive():
    params = TripleBarrierParams(stop_ticks=4, target_ticks=8, max_hold_bars=2)
    df = _build_df(
        [
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 101.0, 99.6, 100.5),
            (100.5, 101.5, 100.0, 101.0),  # timeout exit with gain
            (101.0, 101.5, 100.5, 100.8),
        ]
    )
    out = make_triple_barrier_labels(
        df,
        tz="America/Denver",
        rth_start="07:30",
        rth_end="14:00",
        params=params,
    )
    assert out["long"]["exit"].iloc[0] == "timeout"
    assert out["y_dir"].iloc[0] == 1
    assert out["y_r"].iloc[0] > 0  # partial R from timeout close


def test_session_boundary_limits_hold_to_rth_end():
    params = TripleBarrierParams(stop_ticks=8, target_ticks=12, max_hold_bars=10)
    df = _build_df(
        [
            (100.0, 100.0, 99.5, 100.0),  # 13:45
            (100.0, 101.0, 99.5, 100.7),  # 13:50 entry
            (100.7, 101.2, 100.0, 101.0),  # 13:55, RTH end
        ],
        start="2024-01-02 13:45",
    )
    out = make_triple_barrier_labels(
        df,
        tz="America/Denver",
        rth_start="07:30",
        rth_end="14:00",
        params=params,
    )
    assert out["long"]["exit"].iloc[0] == "timeout"
    # exit occurs at last RTH bar (index 2) so tte = 2 (exit idx 2 - event idx 0)
    assert out["long"]["tte"].iloc[0] == pytest.approx(2)
