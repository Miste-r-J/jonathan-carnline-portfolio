from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from na.tools.quarantine_rebuild_csv import scan_csv


def test_scan_csv_detects_symbol_mismatch(tmp_path: Path):
    src = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "Datetime": ["2026-05-12T10:00:00-06:00", "2026-05-12T10:05:00-06:00"],
            "Open": [7000.0, 7001.0],
            "High": [7001.0, 7002.0],
            "Low": [6999.0, 7000.0],
            "Close": [7000.5, 7001.5],
            "Volume": [100, 101],
            "Instrument": ["NQ", "NQ"],
            "IntervalMin": [5, 5],
            "SourceTz": ["America/Denver", "America/Denver"],
            "SessionId": ["abc", "abc"],
        }
    ).to_csv(src, index=False)

    result = scan_csv(src, "ES", allow_v1=True)
    assert result.ok is False
    assert result.reason == "symbol_mismatch"
