from __future__ import annotations

from pathlib import Path

from na.premarket_planner.analysis import _load_prices
from na.premarket_planner.core import _infer_timezone_from_csv


def test_load_prices_skips_malformed_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "ES.csv"
    csv_path.write_text(
        "\n".join(
            [
                "datetime,open,high,low,close,volume",
                "2026-04-22 06:25:00,5300,5302,5298,5301,100",
                "2026-04-22 06:30:00,5301,5305,5300,5304,110,unexpected,columns",
                "2026-04-22 06:35:00,5304,5306,5302,5303,120",
            ]
        ),
        encoding="utf-8",
    )

    df = _load_prices(str(csv_path))

    assert len(df) == 2
    assert list(df.columns) == ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    assert df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2026-04-22 06:25:00",
        "2026-04-22 06:35:00",
    ]


def test_infer_timezone_ignores_malformed_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "ES.csv"
    csv_path.write_text(
        "\n".join(
            [
                "datetime,open,high,low,close,volume",
                "2026-04-22T12:25:00Z,5300,5302,5298,5301,100",
                "2026-04-22T12:30:00Z,5301,5305,5300,5304,110,unexpected,columns",
                "2026-04-22T12:35:00Z,5304,5306,5302,5303,120",
            ]
        ),
        encoding="utf-8",
    )

    inferred = _infer_timezone_from_csv(csv_path)

    assert inferred == "UTC"


def test_load_prices_normalizes_mixed_timezone_offsets(tmp_path: Path) -> None:
    csv_path = tmp_path / "ES.csv"
    csv_path.write_text(
        "\n".join(
            [
                "datetime,open,high,low,close,volume",
                "2026-03-07T23:55:00-07:00,5300,5302,5298,5301,100",
                "2026-03-08T00:00:00-06:00,5301,5305,5300,5304,110",
                "2026-03-08T00:05:00-06:00,5304,5306,5302,5303,120",
            ]
        ),
        encoding="utf-8",
    )

    df = _load_prices(str(csv_path))

    assert len(df) == 3
    assert str(df["Datetime"].dtype).endswith(", UTC]")
