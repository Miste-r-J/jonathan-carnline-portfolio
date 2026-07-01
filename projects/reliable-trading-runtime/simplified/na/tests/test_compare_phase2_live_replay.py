from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.compare_phase2_live_replay import compare_live_old_new, load_trade_output


FIELDS = [
    "trade_key",
    "entry_ts",
    "exit_ts",
    "side",
    "qty",
    "actual_entry_price",
    "actual_exit_price",
    "pnl_usd",
    "mfe_points",
    "mae_points",
]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _trade(key: str, pnl: float, mfe: float, mae: float) -> dict:
    return {
        "trade_key": key,
        "entry_ts": "2026-06-18T17:15:00-06:00",
        "exit_ts": "2026-06-18T17:20:00-06:00",
        "side": "LONG",
        "qty": 1,
        "actual_entry_price": 100,
        "actual_exit_price": 101,
        "pnl_usd": pnl,
        "mfe_points": mfe,
        "mae_points": mae,
    }


def test_compare_old_new_reports_pnl_mfe_mae_deltas(tmp_path: Path) -> None:
    old_path = tmp_path / "old.csv"
    new_path = tmp_path / "new.csv"
    old_only = _trade("old-only", -25, 1, 3)
    old_only["entry_ts"] = "2026-06-18T17:20:00-06:00"
    new_only = _trade("new-only", 25, 2, 2)
    new_only["entry_ts"] = "2026-06-18T17:25:00-06:00"
    _write(old_path, [_trade("same", 50, 2, 1), old_only])
    _write(new_path, [_trade("same", 100, 3.5, 0.5), new_only])

    report = compare_live_old_new(
        live_baseline=None,
        old_replay=old_path,
        new_replay=new_path,
    )

    comparison = report["comparisons"][0]
    assert comparison["matched_trade_count"] == 1
    assert comparison["aggregate_delta"]["pnl_usd_sum"] == pytest.approx(100.0)
    assert comparison["trade_deltas"][0]["pnl_usd_delta"] == pytest.approx(50.0)
    assert comparison["trade_deltas"][0]["mfe_points_delta"] == pytest.approx(1.5)
    assert comparison["trade_deltas"][0]["mae_points_delta"] == pytest.approx(-0.5)
    assert comparison["left_only_keys"] == ["old-only"]
    assert comparison["right_only_keys"] == ["new-only"]


def test_compare_includes_live_to_each_replay(tmp_path: Path) -> None:
    live = tmp_path / "live.csv"
    old = tmp_path / "old.csv"
    new = tmp_path / "new.csv"
    _write(live, [_trade("same", 50, 2, 1)])
    _write(old, [_trade("same", 0, 1, 2)])
    _write(new, [_trade("same", 75, 2.5, 0.75)])

    report = compare_live_old_new(
        live_baseline=live,
        old_replay=old,
        new_replay=new,
    )

    assert [item["label"] for item in report["comparisons"]] == [
        "old_replay_vs_new_replay",
        "live_vs_old_replay",
        "live_vs_new_replay",
    ]
    assert report["comparisons"][2]["aggregate_delta"]["pnl_usd_sum"] == pytest.approx(25.0)


def test_loader_derives_pnl_from_actual_fill_prices(tmp_path: Path) -> None:
    path = tmp_path / "trades.csv"
    row = _trade("", 0, 2, 1)
    row["pnl_usd"] = ""
    row["side"] = "SHORT"
    row["actual_entry_price"] = 100
    row["actual_exit_price"] = 98
    _write(path, [row])

    loaded = load_trade_output(path)

    assert loaded["trades"][0]["pnl_usd"] == pytest.approx(100.0)
    assert loaded["sha256"]


def test_unique_contract_bar_side_fallback_matches_live_to_replay(tmp_path: Path) -> None:
    live = tmp_path / "live.csv"
    replay = tmp_path / "replay.csv"
    live_row = _trade("live-key", 50, 2, 1)
    live_row["entry_ts"] = "2026-06-18T17:15:41-06:00"
    replay_row = _trade("replay-key", 75, 3, 0.5)
    replay_row["entry_ts"] = "2026-06-18T17:15:00-06:00"
    _write(live, [live_row])
    _write(replay, [replay_row])

    report = compare_live_old_new(
        live_baseline=live,
        old_replay=replay,
        new_replay=replay,
    )

    comparison = report["comparisons"][1]
    assert comparison["matched_trade_count"] == 1
    assert comparison["trade_deltas"][0]["match_method"] == "contract_5m_bar_side"
