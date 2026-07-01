import json
from pathlib import Path

from tools.backfilltest_tournament_audit import audit_backfilltest


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_state_csv(path: Path, rows: list[dict[str, object]]) -> None:
    header = "datetime,action,resolved_action,side,price\n"
    body = "".join(
        f"{row['datetime']},{row['action']},{row.get('resolved_action','')},{row['side']},{row['price']}\n"
        for row in rows
    )
    path.write_text(header + body, encoding="utf-8")


def test_audit_ranks_competitive_runs_and_flags_tod_drift(tmp_path: Path) -> None:
    root = tmp_path / "backfilltest"
    root.mkdir()

    winner = root / "modelrun20k_candidate_v1_c1_v2"
    winner.mkdir()
    _write_json(
        winner / "resolved_config.json",
        {
            "preset": "es_maxpack_10_full_send_prop_safe_pnl",
            "resolved_thresholds": {"p_setup_required": 0.38, "p_long_required": 0.62, "p_short_required": 0.62},
            "threshold_sources": {"p_setup_required": "preset p_setup"},
            "threshold_resolution": {"preset_manifest_divergence": {"p_setup": {"preset": 0.38, "manifest": 0.35}}},
            "trade_window_metadata": {
                "configured_start": "08:10",
                "configured_end": "14:00",
                "effective_start": "00:00",
                "effective_end": "23:59",
                "effective_reason": "gate_tod_disabled_auto_24h",
            },
            "entry_gates": {"gate_vwap": True, "gate_ema": False, "gate_tod": False, "vwap_gate_mode": "any", "gate_mode": "None"},
            "trade_management_overlay": {"pnl_overlay_enabled": True, "pnl_runner_enabled": True, "pnl_runner_arm_r": 2.25, "pnl_runner_giveback_r": 0.5, "pnl_runner_suppress_target": True},
            "phase2_force_open_policy": {"enabled": True, "min_setup": 0.35, "min_entry_conf": 0.55, "live_only": False},
            "regime_policy": {"allow_countertrend_in_unresolved": True, "allow_countertrend_fade_in_trend": None},
            "pnl_shelf_enabled": False,
        },
    )
    _write_json(
        winner / "run_health_summary.json",
        {"verdict": "clean_stopped", "trade_evidence": {"executable_fill_rows": 12, "total_rows": 12}, "executor_stats_all_phases": {"executor_sent_total": 12, "nt_order_entry_total": 12}},
    )
    _write_json(
        winner / "backfill_diagnostics_report.json",
        {"expectancy_source": "state_csv_reconstruction"},
    )
    _write_json(
        winner / "backfill_slice_expectancy_report.json",
        {
            "expectancy_by_slice": [
                {"hour": "08", "side": "LONG", "setup": "unknown", "trades": 10, "win_rate": 0.7, "expectancy_points": 4.0, "avg_win_points": 8.0, "avg_loss_points": -4.0, "points_sum": 40.0},
                {"hour": "10", "side": "SHORT", "setup": "unknown", "trades": 5, "win_rate": 0.6, "expectancy_points": 2.0, "avg_win_points": 6.0, "avg_loss_points": -4.0, "points_sum": 10.0},
            ]
        },
    )
    _write_json(
        winner / "removed_trade_impact_report.json",
        {"removed_trade_impact": {"by_reason": {"stop_already_breached": 1}}},
    )
    _write_state_csv(
        winner / "state.csv",
        [
            {"datetime": "2026-05-01T08:10:00-06:00", "action": "OPEN", "side": "LONG", "price": 7000},
            {"datetime": "2026-05-01T08:15:00-06:00", "action": "CLOSE", "side": "LONG", "price": 7004},
        ],
    )

    fixture = root / "deterministic_fixture_run1"
    fixture.mkdir()
    _write_json(
        fixture / "resolved_config.json",
        {
            "preset": "fixture_preset",
            "resolved_thresholds": {"p_setup_required": 0.5, "p_long_required": 0.5, "p_short_required": 0.5},
            "threshold_sources": {"p_setup_required": "preset p_setup"},
            "trade_window_metadata": {"effective_start": "00:00", "effective_end": "23:59", "effective_reason": "fixture"},
            "entry_gates": {"gate_vwap": False, "gate_ema": False, "gate_tod": False},
            "trade_management_overlay": {},
            "phase2_force_open_policy": {},
            "regime_policy": {},
        },
    )
    _write_json(fixture / "run_health_summary.json", {"verdict": "clean_stopped", "trade_evidence": {"executable_fill_rows": 0, "total_rows": 0}, "executor_stats_all_phases": {"executor_sent_total": 0, "nt_order_entry_total": 0}})
    _write_json(fixture / "backfill_diagnostics_report.json", {"expectancy_source": "trades_csv_fallback"})
    _write_json(
        fixture / "backfill_slice_expectancy_report.json",
        {"expectancy_by_slice": [{"hour": "08", "side": "LONG", "setup": "unknown", "trades": 5, "win_rate": 0.0, "expectancy_points": 0.0, "avg_win_points": 0.0, "avg_loss_points": 0.0, "points_sum": 0.0}]},
    )
    _write_json(fixture / "removed_trade_impact_report.json", {"removed_trade_impact": {"by_reason": {"stop_already_breached": 2}}})
    _write_state_csv(
        fixture / "state.csv",
        [
            {"datetime": "2026-05-01T08:10:00-06:00", "action": "OPEN", "side": "LONG", "price": 100},
            {"datetime": "2026-05-01T08:15:00-06:00", "action": "CLOSE", "side": "LONG", "price": 101},
        ],
    )

    livetest = root / "modelrunlivetest68"
    livetest.mkdir()
    _write_json(
        livetest / "resolved_config.json",
        {
            "preset": "es_elite_v1",
            "resolved_thresholds": {"p_setup_required": 0.27, "p_long_required": 0.57, "p_short_required": 0.57},
            "threshold_sources": {"p_setup_required": "phase2_tag"},
            "trade_window_metadata": {"configured_start": "00:00", "configured_end": "23:59", "effective_start": "00:00", "effective_end": "23:59", "effective_reason": "gate_tod_disabled_auto_24h"},
            "entry_gates": {"gate_vwap": False, "gate_ema": False, "gate_tod": False},
            "trade_management_overlay": {"pnl_overlay_enabled": True, "pnl_runner_enabled": True, "pnl_runner_arm_r": 0.5, "pnl_runner_giveback_r": 0.15, "pnl_runner_suppress_target": True},
            "phase2_force_open_policy": {"enabled": True, "min_setup": 0.27, "min_entry_conf": 0.1, "live_only": True},
            "regime_policy": {"allow_countertrend_in_unresolved": True, "allow_countertrend_fade_in_trend": True},
            "pnl_shelf_enabled": True,
        },
    )
    _write_json(livetest / "run_health_summary.json", {"verdict": "unsafe", "trade_evidence": {"executable_fill_rows": 0, "total_rows": 0}, "executor_stats_all_phases": {"executor_sent_total": 0, "nt_order_entry_total": 0}})
    _write_state_csv(
        livetest / "state.csv",
        [
            {"datetime": "2026-05-01T08:10:00-06:00", "action": "OPEN", "resolved_action": "OPEN", "side": "LONG", "price": 200},
            {"datetime": "2026-05-01T08:15:00-06:00", "action": "FLIP", "resolved_action": "FLIP", "side": "SHORT", "price": 205},
            {"datetime": "2026-05-01T08:20:00-06:00", "action": "CLOSE", "resolved_action": "CLOSE", "side": "SHORT", "price": 202},
        ],
    )

    payload = audit_backfilltest(root)

    assert payload["summary"]["classification_counts"]["candidate_family"] == 1
    assert payload["summary"]["classification_counts"]["fixture"] == 1
    assert payload["summary"]["classification_counts"]["livetest_family"] == 1

    winner_record = payload["competitive_rankings"][0]
    assert winner_record["run_name"] == "modelrun20k_candidate_v1_c1_v2"
    assert winner_record["distortions"]["tod_disabled_24h_drift"] is True
    assert winner_record["metric_source"] == "diagnostics_reports"

    fixture_record = next(run for run in payload["full_corpus_rankings"] if run["run_name"] == "deterministic_fixture_run1")
    assert fixture_record["competitive"] is False
    assert fixture_record["metric_source"] == "state_csv_fallback"
    assert fixture_record["placeholder_reason"] == "all_zero_expectancy_rows"

    livetest_record = next(run for run in payload["competitive_rankings"] if run["run_name"] == "modelrunlivetest68")
    assert livetest_record["metric_source"] == "state_csv_fallback"
    assert livetest_record["trade_count"] == 2
    assert payload["winner"]["winning_run"] == "modelrun20k_candidate_v1_c1_v2"
