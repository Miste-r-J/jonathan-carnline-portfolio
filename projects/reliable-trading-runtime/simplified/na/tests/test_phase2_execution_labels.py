from __future__ import annotations

import pandas as pd
import pytest

import na.bot.train_phase2 as train_phase2
from na.bot.train_phase2 import _build_exec_label_frame, _direction_thresholds


def _exec_frame(highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    size = len(highs)
    return pd.DataFrame(
        {
            "Datetime": pd.date_range("2026-01-01", periods=size, freq="5min", tz="America/Denver"),
            "Open": [100.0] * size,
            "High": highs,
            "Low": lows,
            "Close": closes,
        }
    )


def test_same_bar_up_and_down_target_hit_is_labeled_flat() -> None:
    frame = _exec_frame(
        highs=[100.0, 102.0, 100.0, 100.0],
        lows=[100.0, 98.0, 100.0, 100.0],
        closes=[100.0, 101.0, 100.0, 100.0],
    )

    labels = _build_exec_label_frame(
        frame,
        threshold=0.01,
        instrument="ES",
        max_hold_bars=1,
        commission_per_contract=0.0,
        slippage_ticks=0.0,
    )

    assert int(labels.iloc[0]["target"]) == 0


def test_p_short_is_returned_as_short_class_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[list[float]] = []

    def fake_precision_threshold(probs, labels, **kwargs):
        observed.append(list(probs))
        return 0.67 if len(observed) == 1 else 0.71

    monkeypatch.setattr(train_phase2, "_precision_threshold", fake_precision_threshold)
    p_long, p_short = _direction_thresholds(
        probs=[0.9, 0.2],
        labels=[1, 0],
    )

    assert p_long == pytest.approx(0.67)
    assert p_short == pytest.approx(0.71)
    assert observed[1] == pytest.approx([0.1, 0.8])
