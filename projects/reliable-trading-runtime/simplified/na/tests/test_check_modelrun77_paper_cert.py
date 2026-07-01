from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.check_modelrun77_paper_cert import build_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _paper_run(root: Path, name: str, fills: int) -> None:
    run = root / name
    _write_json(run / "status.json", {"phase2_tag": "modelrun77_prop_v3_20260622", "preset": "es_modelrun77_prop_v3_paper", "nt_exec_policy": "paper", "hard_lockout_active": False})
    _write_json(run / "run_health_summary.json", {"verdict": "running_healthy", "process_alive": False})
    _write_json(
        run / "resolved_config.json",
        {
            "preset": "es_modelrun77_prop_v3_paper",
            "phase2_tag": "modelrun77_prop_v3_20260622",
            "phase2_manifest_thresholds_used": True,
            "resolved_thresholds": {"p_setup_required": 0.20, "p_long_required": 0.66, "p_short_required": 0.75},
            "phase2_force_open_policy": {"enabled": False, "allow_setup_fail_entries": False},
            "risk_limits": {"max_daily_loss_usd": 500, "max_risk_per_trade_usd": 400, "max_trades_per_day": 6, "max_losses_per_day": 3},
        },
    )
    with (run / "trades.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["actual_entry_price", "actual_exit_price"])
        writer.writeheader()
        for _ in range(fills):
            writer.writerow({"actual_entry_price": 100, "actual_exit_price": 101})


def test_certifies_two_clean_sessions_with_thirty_fills(tmp_path: Path) -> None:
    _paper_run(tmp_path, "modelrun77_final_paper_cert_20260622_100000", 15)
    _paper_run(tmp_path, "modelrun77_final_paper_cert_20260622_110000", 15)
    report = build_report(tmp_path)
    assert report["certified"] is True


def test_rejects_force_open_session(tmp_path: Path) -> None:
    _paper_run(tmp_path, "modelrun77_final_paper_cert_20260622_100000", 30)
    config_path = tmp_path / "modelrun77_final_paper_cert_20260622_100000" / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["phase2_force_open_policy"]["enabled"] = True
    _write_json(config_path, config)
    report = build_report(tmp_path, min_sessions=1)
    assert report["certified"] is False
