from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / "tools" / "build_phase2_training_dataset.py"
    spec = importlib.util.spec_from_file_location("build_phase2_training_dataset_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_builder_appends_only_new_rows_and_applies_rollover(tmp_path: Path) -> None:
    mod = _load_module()
    historical = tmp_path / "ES6.csv"
    extension = tmp_path / "ES.csv"
    exporters = tmp_path / "exporters"
    exporters.mkdir()
    base_rows = [
        {"Datetime": "2026-04-27T20:50:00Z", "Open": 1, "High": 2, "Low": 0, "Close": 1, "Volume": 10},
        {"Datetime": "2026-04-27T20:55:00Z", "Open": 2, "High": 3, "Low": 1, "Close": 2, "Volume": 11},
    ]
    _write(historical, base_rows)
    _write(
        extension,
        [
            base_rows[-1],
            {"Datetime": "2026-05-01T20:55:00Z", "Open": 3, "High": 4, "Low": 2, "Close": 3, "Volume": 12},
        ],
    )
    _write(
        exporters / "ES_06-26_5m_20260611.csv",
        [
            {"Datetime": "2026-06-11T23:55:00Z", "Open": 4, "High": 5, "Low": 3, "Close": 4, "Volume": 13},
            {"Datetime": "2026-06-12T00:00:00Z", "Open": 5, "High": 6, "Low": 4, "Close": 5, "Volume": 14},
        ],
    )
    _write(
        exporters / "ES_09-26_5m_20260612.csv",
        [
            {"Datetime": "2026-06-11T23:55:00Z", "Open": 14, "High": 15, "Low": 13, "Close": 14, "Volume": 15},
            {"Datetime": "2026-06-12T00:00:00Z", "Open": 15, "High": 16, "Low": 14, "Close": 15, "Volume": 16},
        ],
    )

    frame, provenance = mod.build_dataset(
        historical_csv=historical,
        extension_csvs=[extension],
        exporter_dir=exporters,
        rollover_utc=pd.Timestamp("2026-06-12T00:00:00Z"),
    )

    assert frame["Datetime"].tolist() == [
        "2026-04-27T20:50:00Z",
        "2026-04-27T20:55:00Z",
        "2026-05-01T20:55:00Z",
        "2026-06-11T23:55:00Z",
        "2026-06-12T00:00:00Z",
    ]
    assert frame["Close"].tolist()[-2:] == [4, 15]
    assert provenance["output"]["duplicate_timestamps"] == 0


def test_builder_uses_extension_to_fill_internal_historical_gap(tmp_path: Path) -> None:
    mod = _load_module()
    historical = tmp_path / "ES6.csv"
    extension = tmp_path / "ES.csv"
    exporters = tmp_path / "exporters"
    exporters.mkdir()
    _write(
        historical,
        [
            {"Datetime": "2026-01-01T00:00:00Z", "Open": 1, "High": 2, "Low": 0, "Close": 1, "Volume": 10},
            {"Datetime": "2026-04-01T00:00:00Z", "Open": 4, "High": 5, "Low": 3, "Close": 4, "Volume": 10},
        ],
    )
    _write(
        extension,
        [
            {"Datetime": "2026-02-01T00:00:00Z", "Open": 2, "High": 3, "Low": 1, "Close": 2, "Volume": 10},
            {"Datetime": "2026-03-01T00:00:00Z", "Open": 3, "High": 4, "Low": 2, "Close": 3, "Volume": 10},
        ],
    )

    frame, _ = mod.build_dataset(
        historical_csv=historical,
        extension_csvs=[extension],
        exporter_dir=exporters,
        rollover_utc=pd.Timestamp("2026-06-12T00:00:00Z"),
    )

    assert frame["Datetime"].tolist() == [
        "2026-01-01T00:00:00Z",
        "2026-02-01T00:00:00Z",
        "2026-03-01T00:00:00Z",
        "2026-04-01T00:00:00Z",
    ]


def test_builder_drops_non_es_price_contamination_and_records_it(tmp_path: Path) -> None:
    mod = _load_module()
    historical = tmp_path / "ES6.csv"
    extension = tmp_path / "ES.csv"
    exporters = tmp_path / "exporters"
    exporters.mkdir()
    _write(
        historical,
        [{"Datetime": "2026-05-08T16:40:00Z", "Open": 7400, "High": 7402, "Low": 7399, "Close": 7401, "Volume": 10}],
    )
    _write(
        extension,
        [
            {"Datetime": "2026-05-08T16:45:00Z", "Open": 80100, "High": 80200, "Low": 80000, "Close": 80150, "Volume": 100},
            {"Datetime": "2026-05-10T22:05:00Z", "Open": 7420, "High": 7422, "Low": 7419, "Close": 7421, "Volume": 20},
        ],
    )

    frame, provenance = mod.build_dataset(
        historical_csv=historical,
        extension_csvs=[extension],
        exporter_dir=exporters,
        rollover_utc=pd.Timestamp("2026-06-12T00:00:00Z"),
    )

    assert frame["Close"].tolist() == [7401, 7421]
    assert provenance["sources"]["extensions"][0]["invalid_es_rows_dropped"] == 1
