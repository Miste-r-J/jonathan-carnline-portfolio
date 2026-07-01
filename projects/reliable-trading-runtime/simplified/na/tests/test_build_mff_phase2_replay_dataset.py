from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.build_mff_phase2_replay_dataset import (
    build_dataset,
    canonicalize_snapshots,
    enumerate_mff_runs,
    normalize_contract,
)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_contract_normalization_covers_june_and_september() -> None:
    assert normalize_contract("ES JUN26") == "ES 06-26"
    assert normalize_contract("ES_06-26_5m_20260611") == "ES 06-26"
    assert normalize_contract("ES SEP26") == "ES 09-26"
    assert normalize_contract("ES_09-26_5m_20260621") == "ES 09-26"


def test_canonical_snapshot_latest_row_wins(tmp_path: Path) -> None:
    fields = ["Datetime", "Open", "High", "Low", "Close", "Volume", "Instrument", "IntervalMin"]
    older = tmp_path / "ES_JUN26_5m_20260607.csv"
    newer = tmp_path / "ES_06-26_5m_20260609.csv"
    _write_csv(
        older,
        fields,
        [
            {
                "Datetime": "2026-06-07T10:00:00-06:00",
                "Open": 100,
                "High": 102,
                "Low": 99,
                "Close": 101,
                "Volume": 10,
                "Instrument": "ES JUN26",
                "IntervalMin": 5,
            }
        ],
    )
    _write_csv(
        newer,
        fields,
        [
            {
                "Datetime": "2026-06-07T10:00:00-06:00",
                "Open": 100,
                "High": 104,
                "Low": 98,
                "Close": 103,
                "Volume": 20,
                "Instrument": "ES 06-26",
                "IntervalMin": 5,
            }
        ],
    )

    rows, metadata = canonicalize_snapshots([older, newer])

    assert len(rows) == 1
    assert rows[0]["Contract"] == "ES 06-26"
    assert rows[0]["High"] == 104.0
    assert rows[0]["SourceFile"] == newer.name
    assert metadata["duplicate_keys_replaced_or_ignored"] == 1


def test_build_dataset_finds_only_mff_runs_and_derives_fill_metrics(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    mff_run = runs_root / "mff_run"
    demo_run = runs_root / "demo_run"
    mff_run.mkdir(parents=True)
    demo_run.mkdir(parents=True)
    (mff_run / "status.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "chosen_account": "MFFUEVRPD447934003",
                "exec_instrument": "ES JUN26",
                "preset": "phase2",
            }
        ),
        encoding="utf-8",
    )
    (demo_run / "status.json").write_text(
        json.dumps({"run_id": "run-2", "chosen_account": "DEMO123", "exec_instrument": "ES JUN26"}),
        encoding="utf-8",
    )
    trade_fields = [
        "entry_ts",
        "exit_ts",
        "side",
        "qty",
        "actual_entry_price",
        "actual_exit_price",
        "filled_qty",
        "exit_reason",
        "client_order_id",
        "mfe_points",
        "mae_points",
    ]
    _write_csv(
        mff_run / "trades.csv",
        trade_fields,
        [
            {
                "entry_ts": "2026-06-07T10:00:01-06:00",
                "exit_ts": "2026-06-07T10:05:01-06:00",
                "side": "LONG",
                "qty": 1,
                "actual_entry_price": 100,
                "actual_exit_price": 103,
                "filled_qty": 1,
                "exit_reason": "target_hit",
                "client_order_id": "run-1|phase2|ES JUN26|bar|OPEN|LONG|x",
            }
        ],
    )
    snapshot_root = tmp_path / "snapshots"
    _write_csv(
        snapshot_root / "ES_JUN26_5m_20260607.csv",
        ["Datetime", "Open", "High", "Low", "Close", "Volume", "Instrument", "IntervalMin"],
        [
            {
                "Datetime": "2026-06-07T10:05:00-06:00",
                "Open": 100,
                "High": 105,
                "Low": 98,
                "Close": 103,
                "Volume": 10,
                "Instrument": "ES JUN26",
                "IntervalMin": 5,
            }
        ],
    )

    manifest = build_dataset(
        runs_root=runs_root,
        snapshot_root=snapshot_root,
        output_dir=tmp_path / "out",
    )

    assert enumerate_mff_runs(runs_root)[0]["run_name"] == "mff_run"
    assert manifest["run_inventory"]["mff_run_count"] == 1
    assert manifest["fill_inventory"]["actual_fill_rows"] == 1
    assert manifest["fill_inventory"]["pnl_usd"] == pytest.approx(150.0)
    assert manifest["fill_inventory"]["mfe_available_rows"] == 1
    baseline_path = Path(manifest["artifacts"]["actual_fill_baseline"]["path"])
    row = next(csv.DictReader(baseline_path.open(encoding="utf-8")))
    assert float(row["mfe_points"]) == pytest.approx(5.0)
    assert float(row["mae_points"]) == pytest.approx(2.0)
    assert Path(manifest["manifest_path"]).name.startswith("mff_phase2_replay_dataset_")


def test_immutable_artifact_refuses_hash_named_collision(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    snapshot_root = tmp_path / "snapshots"
    runs_root.mkdir()
    snapshot_root.mkdir()
    first = build_dataset(runs_root=runs_root, snapshot_root=snapshot_root, output_dir=tmp_path / "out")
    bars_path = Path(first["artifacts"]["canonical_bars"]["path"])
    bars_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable artifact collision"):
        build_dataset(runs_root=runs_root, snapshot_root=snapshot_root, output_dir=tmp_path / "out")
