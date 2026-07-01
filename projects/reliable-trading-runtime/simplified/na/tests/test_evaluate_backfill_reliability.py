import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.evaluate_backfill_reliability import evaluate_backfill_reliability


def _write_state(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "datetime",
        "action",
        "side",
        "price",
        "prob",
        "entry_conf",
        "hold_conf",
        "size",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {k: row.get(k) for k in fieldnames}
            writer.writerow(payload)


def _write_gating(path: Path, rows: list[dict]) -> None:
    header = {"type": "header", "run_id": "RUN"}
    data = [header] + rows
    path.write_text("\n".join(json.dumps(x) for x in data) + "\n", encoding="utf-8")


def _base_gating(bar_ts: str, action: str, blocked_by: list[str] | None = None, reason_detail: str = "") -> dict:
    return {
        "bar_ts": bar_ts,
        "phase": "BACKFILL",
        "action": action,
        "blocked_by": blocked_by or [],
        "reason_detail": reason_detail,
        "pred_p_long": 0.6,
        "phase2": {"setup_prob": 0.8, "short_prob": 0.2, "direction_signal": 1},
    }


def test_reconstruction_enforces_single_position_state_machine(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_state(
        run_dir / "state.csv",
        [
            {"datetime": "2026-04-21T00:00:00-06:00", "action": "OPEN", "side": "LONG", "price": 100, "size": 1},
            {"datetime": "2026-04-21T00:05:00-06:00", "action": "OPEN", "side": "LONG", "price": 101, "size": 1},
            {"datetime": "2026-04-21T00:10:00-06:00", "action": "FLIP", "side": "SHORT", "price": 99, "size": 1},
            {"datetime": "2026-04-21T00:15:00-06:00", "action": "CLOSE", "side": "SHORT", "price": 98, "size": 1},
        ],
    )
    _write_gating(
        run_dir / "gating_events.jsonl",
        [
            _base_gating("2026-04-21T00:00:00-06:00", "OPEN"),
            _base_gating("2026-04-21T00:05:00-06:00", "OPEN"),
            _base_gating("2026-04-21T00:10:00-06:00", "FLIP"),
            _base_gating("2026-04-21T00:15:00-06:00", "CLOSE"),
        ],
    )

    report = evaluate_backfill_reliability(
        run_dir,
        phase="BACKFILL",
        target_min=400,
        target_max=1200,
        window_days=20,
        cost_profile="current_default",
        setup_thresholds=None,
        direction_long_thresholds=None,
        direction_short_thresholds=None,
    )

    recon = report["reconstructed_trades"]
    assert recon["trade_count"] == 2
    assert recon["non_overlapping_position_logic"]["ignored_open_while_in_position"] == 1


def test_cost_model_applies_commission_and_slippage_exactly(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_state(
        run_dir / "state.csv",
        [
            {"datetime": "2026-04-21T00:00:00-06:00", "action": "OPEN", "side": "LONG", "price": 100, "size": 1},
            {"datetime": "2026-04-21T00:05:00-06:00", "action": "CLOSE", "side": "LONG", "price": 101, "size": 1},
        ],
    )
    _write_gating(
        run_dir / "gating_events.jsonl",
        [
            _base_gating("2026-04-21T00:00:00-06:00", "OPEN"),
            _base_gating("2026-04-21T00:05:00-06:00", "CLOSE"),
        ],
    )

    report = evaluate_backfill_reliability(
        run_dir,
        phase="BACKFILL",
        target_min=400,
        target_max=1200,
        window_days=20,
        cost_profile="current_default",
        setup_thresholds=None,
        direction_long_thresholds=None,
        direction_short_thresholds=None,
    )

    recon = report["reconstructed_trades"]
    assert recon["gross_pnl_usd"] == pytest.approx(50.0)
    # ES defaults => commission 2.0 + slippage 1 tick (12.5) per side = 14.5; roundtrip cost = 29.0
    assert recon["cost_usd"] == pytest.approx(29.0)
    assert recon["net_pnl_usd"] == pytest.approx(21.0)


def test_funnel_and_blocker_attribution_split_strategy_vs_execution(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_state(
        run_dir / "state.csv",
        [
            {"datetime": "2026-04-21T00:00:00-06:00", "action": "NO_TRADE", "side": "FLAT", "price": 100, "size": 1},
            {"datetime": "2026-04-21T00:05:00-06:00", "action": "NO_TRADE", "side": "FLAT", "price": 100, "size": 1},
            {"datetime": "2026-04-21T00:10:00-06:00", "action": "NO_TRADE", "side": "FLAT", "price": 100, "size": 1},
            {"datetime": "2026-04-21T00:15:00-06:00", "action": "NO_TRADE", "side": "FLAT", "price": 100, "size": 1},
        ],
    )
    _write_gating(
        run_dir / "gating_events.jsonl",
        [
            _base_gating("2026-04-21T00:00:00-06:00", "NO_TRADE", blocked_by=["startup_resync"], reason_detail="not_armed reason=startup_resync"),
            _base_gating("2026-04-21T00:05:00-06:00", "NO_TRADE", blocked_by=["prob"], reason_detail="prob=0.55"),
            _base_gating("2026-04-21T00:10:00-06:00", "NO_TRADE", blocked_by=[], reason_detail=""),
            _base_gating("2026-04-21T00:15:00-06:00", "NO_TRADE", blocked_by=["blocked_stale_bar", "prob"], reason_detail="stale+prob"),
        ],
    )

    report = evaluate_backfill_reliability(
        run_dir,
        phase="BACKFILL",
        target_min=400,
        target_max=1200,
        window_days=20,
        cost_profile="current_default",
        setup_thresholds=None,
        direction_long_thresholds=None,
        direction_short_thresholds=None,
    )

    funnel = report["signal_funnel"]
    assert funnel["candidate_signals"] == 4
    assert funnel["execution_eligible"] == 1
    # startup-only row is strategy-eligible; unblocked row is strategy-eligible
    assert funnel["strategy_eligible"] == 2

    buckets = report["blocker_attribution"]["bucket_counts"]
    assert buckets["prob"]["count"] >= 2
    assert buckets["startup_resync"]["count"] >= 1
    assert buckets["stale"]["count"] >= 1


def test_phase_parity_reports_matched_dates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_state(
        run_dir / "state.csv",
        [
            {"datetime": "2026-04-21T00:00:00-06:00", "action": "NO_TRADE", "side": "FLAT", "price": 100, "size": 1},
            {"datetime": "2026-04-21T00:05:00-06:00", "action": "NO_TRADE", "side": "FLAT", "price": 100, "size": 1},
        ],
    )
    gating_rows = [
        {
            **_base_gating("2026-04-21T00:00:00-06:00", "NO_TRADE", blocked_by=["prob"], reason_detail="prob"),
            "phase": "BACKFILL",
        },
        {
            **_base_gating("2026-04-21T00:05:00-06:00", "NO_TRADE", blocked_by=["startup_resync"], reason_detail="startup_resync"),
            "phase": "LIVE",
        },
    ]
    _write_gating(run_dir / "gating_events.jsonl", gating_rows)

    report = evaluate_backfill_reliability(
        run_dir,
        phase="BACKFILL",
        target_min=400,
        target_max=1200,
        window_days=20,
        cost_profile="current_default",
        setup_thresholds=None,
        direction_long_thresholds=None,
        direction_short_thresholds=None,
    )

    parity = report["phase_parity"]
    assert parity["matched_date_count"] == 1
    assert "BACKFILL" in parity["overall_distribution"]
    assert "LIVE" in parity["overall_distribution"]


def test_reliability_pipeline_emits_stage_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_state(
        run_dir / "state.csv",
        [
            {"datetime": "2026-04-21T00:00:00-06:00", "action": "NO_TRADE", "side": "FLAT", "price": 100, "size": 1},
            {"datetime": "2026-04-21T00:05:00-06:00", "action": "NO_TRADE", "side": "FLAT", "price": 101, "size": 1},
            {"datetime": "2026-04-21T00:10:00-06:00", "action": "NO_TRADE", "side": "FLAT", "price": 102, "size": 1},
        ],
    )
    _write_gating(
        run_dir / "gating_events.jsonl",
        [
            {
                **_base_gating("2026-04-21T00:00:00-06:00", "NO_TRADE", blocked_by=[]),
                "phase": "BACKFILL",
                "pred_p_long": 0.62,
                "phase2": {"setup_prob": 0.8, "short_prob": 0.38, "direction_signal": 1},
            },
            {
                **_base_gating("2026-04-21T00:05:00-06:00", "NO_TRADE", blocked_by=[]),
                "phase": "BACKFILL",
                "pred_p_long": 0.61,
                "phase2": {"setup_prob": 0.81, "short_prob": 0.39, "direction_signal": 1},
            },
            {
                **_base_gating("2026-04-21T00:10:00-06:00", "NO_TRADE", blocked_by=[]),
                "phase": "LIVE",
                "pred_p_long": 0.60,
                "phase2": {"setup_prob": 0.82, "short_prob": 0.40, "direction_signal": 1},
            },
        ],
    )

    out_path = tmp_path / "promotion.json"
    script = Path(__file__).resolve().parents[2] / "tools" / "run_backfill_live_reliability_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dirs",
            str(run_dir),
            "--target-min",
            "0",
            "--target-max",
            "100000",
            "--window-days",
            "20",
            "--output",
            str(out_path),
        ],
        check=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "stage_a_backfill" in payload
    assert "stage_b_live_shadow" in payload
    assert "stage_c_promotion" in payload
