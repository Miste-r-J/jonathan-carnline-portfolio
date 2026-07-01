from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.runtime_engine.modeling.train_phase2 import (
    Phase2SplitIndices,
    _apply_purge_embargo,
    _drift_features_to_drop,
    _split_calibration_subset,
    _split_indices_by_time,
)


def test_ratio_style_splits_purge_left_and_embargo_right() -> None:
    splits = Phase2SplitIndices(
        train_index=pd.Index(range(0, 60)),
        val_index=pd.Index(range(60, 80)),
        test_index=pd.Index(range(80, 100)),
    )

    safe = _apply_purge_embargo(splits, purge_bars=4, embargo_bars=2)

    assert safe.train_index.tolist() == list(range(0, 56))
    assert safe.val_index.tolist() == list(range(62, 76))
    assert safe.test_index.tolist() == list(range(82, 100))


def test_explicit_time_splits_receive_the_same_purge_embargo_policy() -> None:
    frame = pd.DataFrame(
        {"Datetime": pd.date_range("2026-01-01", periods=30, freq="h", tz="America/Denver")}
    )
    raw = _split_indices_by_time(
        frame,
        tz="America/Denver",
        train_start="2026-01-01 00:00",
        train_end="2026-01-01 09:00",
        val_start="2026-01-01 10:00",
        val_end="2026-01-01 19:00",
        test_start="2026-01-01 20:00",
        test_end="2026-01-02 05:00",
    )

    safe = _apply_purge_embargo(raw, purge_bars=2, embargo_bars=1)

    assert safe.train_index.tolist() == list(range(0, 8))
    assert safe.val_index.tolist() == list(range(11, 18))
    assert safe.test_index.tolist() == list(range(21, 30))


def test_validation_early_stop_and_calibration_are_also_separated() -> None:
    X = pd.DataFrame({"feature": np.arange(40.0)})
    y = pd.Series([0, 1] * 20)

    X_early, X_cal, y_early, y_cal = _split_calibration_subset(
        X,
        y,
        purge_bars=2,
        embargo_bars=1,
    )

    assert X_early.index.tolist() == list(range(0, 18))
    assert X_cal.index.tolist() == list(range(21, 40))
    assert y_early.index.equals(X_early.index)
    assert y_cal.index.equals(X_cal.index)


def test_drift_deletion_uses_validation_not_test_data() -> None:
    train = pd.DataFrame(
        {
            "stable": np.arange(100, dtype=float),
            "shifted": np.arange(100, dtype=float),
        }
    )
    validation = pd.DataFrame(
        {
            "stable": np.arange(100, dtype=float),
            "shifted": np.arange(100, dtype=float) + 1000.0,
        }
    )

    dropped, diagnostics = _drift_features_to_drop(train, validation, ks_limit=0.2)

    assert dropped == ["shifted"]
    assert [row["feature"] for row in diagnostics] == ["shifted"]
