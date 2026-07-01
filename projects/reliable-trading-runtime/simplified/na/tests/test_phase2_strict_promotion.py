from __future__ import annotations

import json
from pathlib import Path

from tools import train_retrain_v6_strict as strict


class _Logger:
    def log(self, _message: str) -> None:
        pass


def _bench(new: dict, baseline: dict) -> dict:
    return {
        "new": {"slippage": {"slip_1": new}},
        "baseline": {"slippage": {"slip_1": baseline}},
    }


def test_run_benchmark_passes_requested_date_bounds(monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "benchmark"
    out_dir.mkdir()
    (out_dir / "phase2_benchmark.json").write_text(
        json.dumps({"rows": [{"tag": "candidate"}, {"tag": "baseline"}]}),
        encoding="utf-8",
    )
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", fake_run)
    strict._run_benchmark(
        tmp_path,
        tmp_path / "artifacts",
        "baseline",
        "candidate",
        out_dir,
        _Logger(),
        start_date="2026-02-01",
        end_date="2026-02-28",
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("--test-start") + 1] == "2026-02-01"
    assert cmd[cmd.index("--test-end") + 1] == "2026-02-28"


def test_promotion_requires_non_regression_against_baseline() -> None:
    baseline = {
        "total_pnl_usd": 150000.0,
        "sharpe": 20.0,
        "trade_count": 600,
        "win_rate": 0.50,
        "avg_trade_r": 1.2,
        "max_drawdown": -10000.0,
        "profit_factor": 1.5,
    }
    regressed = dict(baseline, total_pnl_usd=149999.0)

    passed, checks = strict._promotion_gate_results(
        _bench(regressed, baseline),
        {"setup": 0.01, "direction": 0.01, "close": 0.0},
    )

    assert not passed
    pnl_check = next(row for row in checks if row["name"] == "baseline_non_regression_pnl_slip1")
    assert not pnl_check["pass"]


def test_benchmark_manifest_enriches_existing_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "trainer_metadata": {"seed": 42},
                "thresholds": {"legacy_threshold": 0.25},
                "close": {"feature_count": 211},
                "config": {"custom_flag": "keep"},
            }
        ),
        encoding="utf-8",
    )

    strict._write_benchmark_ready_manifest(
        path,
        tag="candidate",
        csv_path=tmp_path / "ES.csv",
        thresholds={"p_setup": 0.3, "p_long": 0.6, "p_short": 0.6, "close_threshold": 0.8},
        close_enabled=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["trainer_metadata"] == {"seed": 42}
    assert payload["thresholds"]["legacy_threshold"] == 0.25
    assert payload["close"]["feature_count"] == 211
    assert payload["close"]["model_path"] is None
    assert payload["config"]["custom_flag"] == "keep"


def test_strict_manifest_enriches_existing_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "trainer_metadata": {"feature_schema": "v6"},
                "benchmark": {"existing_note": "keep"},
                "close": {"feature_count": 199},
            }
        ),
        encoding="utf-8",
    )
    bench_row = {
        "slippage": {
            "slip_1": {
                "total_pnl_usd": 200000.0,
                "sharpe": 22.0,
                "trade_count": 700,
                "win_rate": 0.55,
                "avg_trade_r": 1.4,
                "max_drawdown": -9000.0,
                "profit_factor": 1.8,
            }
        }
    }

    strict._write_strict_manifest(
        path,
        "candidate",
        tmp_path / "ES.csv",
        12.5,
        {"p_setup": 0.3, "p_long": 0.6, "p_short": 0.6, "close_threshold": 0.8},
        bench_row,
        False,
        True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["trainer_metadata"] == {"feature_schema": "v6"}
    assert payload["benchmark"]["existing_note"] == "keep"
    assert payload["benchmark"]["pnl_slip1"] == 200000.0
    assert payload["close"]["feature_count"] == 199
    assert payload["close"]["model_path"] is None
