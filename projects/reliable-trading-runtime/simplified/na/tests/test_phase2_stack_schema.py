from __future__ import annotations

import importlib.util
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


BOT_ROOT = Path(__file__).resolve().parents[3]
SIMPLIFIED_ROOT = BOT_ROOT / "simplified"
TOOLS_ROOT = SIMPLIFIED_ROOT / "tools"
REPO_TOOLS_ROOT = BOT_ROOT / "tools"


def _load_module(name: str, path: Path):
    if str(SIMPLIFIED_ROOT) not in sys.path:
        sys.path.insert(0, str(SIMPLIFIED_ROOT))
    if str(TOOLS_ROOT) not in sys.path:
        sys.path.insert(0, str(TOOLS_ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_train_phase2_grid_accepts_stack_setup_prob_as_direction_only_feature():
    mod = _load_module("train_phase2_grid_for_stack_test", TOOLS_ROOT / "train_phase2_grid.py")

    mod._validate_setup_direction_schema(["a", "b"], ["a", "b", "stack_setup_prob"], stack_setup_prob=True)

    with pytest.raises(RuntimeError, match="schemas differ"):
        mod._validate_setup_direction_schema(["a", "b"], ["a", "b", "stack_setup_prob"], stack_setup_prob=False)

    with pytest.raises(RuntimeError, match="unexpected_direction_features"):
        mod._validate_setup_direction_schema(["a", "b"], ["a", "b", "bad_extra"], stack_setup_prob=True)

    with pytest.raises(RuntimeError, match="missing_setup_features"):
        mod._validate_setup_direction_schema(["a", "b"], ["a", "stack_setup_prob"], stack_setup_prob=True)

    with pytest.raises(RuntimeError, match="trailing"):
        mod._validate_setup_direction_schema(["a", "b"], ["a", "stack_setup_prob", "b"], stack_setup_prob=True)


def test_validate_features_uses_setup_hash_for_stacked_live_base_schema(tmp_path):
    validator = _load_module("validate_features_for_stack_test", REPO_TOOLS_ROOT / "validate_features.py")
    from na.bot.feature_hash import compute_feature_hash

    (tmp_path / "setup.features.json").write_text(json.dumps({"features": ["a", "b"]}), encoding="utf-8")
    (tmp_path / "dir.features.json").write_text(
        json.dumps({"features": ["a", "b", "stack_setup_prob"]}),
        encoding="utf-8",
    )
    manifest = {
        "setup_model_path": "setup.joblib",
        "dir_model_path": "dir.joblib",
        "feature_hash": compute_feature_hash(["a", "b", "stack_setup_prob"]),
        "feature_hashes": {
            "setup": compute_feature_hash(["a", "b"]),
            "direction": compute_feature_hash(["a", "b", "stack_setup_prob"]),
        },
        "config": {"stack_setup_prob": True},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validator._expected_live_feature_hash(manifest) == compute_feature_hash(["a", "b"])
    ok, err = validator._validate_stacked_sidecars(manifest_path, manifest, compute_feature_hash)
    assert ok is True
    assert err is None


def test_generational_rejected_candidate_row_is_not_promotion_candidate():
    mod = _load_module("train_phase2_generational_for_stack_test", TOOLS_ROOT / "train_phase2_generational.py")
    manifest = {
        "tag": "retrain_v7_gen01_cand01",
        "_manifest_path": "C:/tmp/manifest.json",
        "rejected": True,
        "rejected_reason": "No threshold combo met constraints",
        "metrics": {"setup": {"label_info": {"class_counts": {"trade": 1, "flat": 9}}}},
    }

    row = mod._rejected_candidate_row(manifest, generation=1, parent_tag="retrain_v4", cfg={"stack_setup_prob": True})

    assert row["rejected"] is True
    assert row["promotion_pass"] is False
    assert row["risk_score"] == float("-inf")
    assert "trainer rejected candidate" in row["promotion_reasons"][0]


def test_train_phase2_helpers_parse_mtf_and_shadow_gate(tmp_path):
    if str(SIMPLIFIED_ROOT) not in sys.path:
        sys.path.insert(0, str(SIMPLIFIED_ROOT))
    mod = importlib.import_module("na.bot.train_phase2")

    assert mod._parse_mtf_timeframes("15m,60min, 240") == ["15min", "60min", "240min"]

    summary_path = tmp_path / "run_health_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "feed_health_ok": False,
                "verdict": "unsafe",
                "unresolved_warnings": ["feed_health_not_ok"],
                "position_state": "IN_POSITION_PROTECTED",
            }
        ),
        encoding="utf-8",
    )
    gate = mod._load_live_shadow_gate(str(summary_path))

    assert gate["exists"] is True
    assert gate["passed"] is False
    assert "feed_health_not_ok" in gate["reasons"]
    assert "verdict_unsafe" in gate["reasons"]


def test_prepare_multi_tf_source_handles_mixed_dst_offsets():
    if str(SIMPLIFIED_ROOT) not in sys.path:
        sys.path.insert(0, str(SIMPLIFIED_ROOT))
    mod = importlib.import_module("na.bot.train_phase2")
    import pandas as pd

    df = pd.DataFrame(
        {
            "Datetime": [
                "2021-03-12T16:05:00-07:00",
                "2021-03-15T16:05:00-06:00",
            ],
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.5, 1.5],
            "Close": [1.2, 2.2],
            "Volume": [100, 200],
        }
    )

    out = mod._prepare_multi_tf_source(df, tz="America/Denver", csv_naive_is_utc=False)

    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 2
    assert str(out.index.tz) == "America/Denver"


def test_generational_gates_block_unsafe_live_shadow():
    mod = _load_module("train_phase2_generational_for_shadow_gate_test", TOOLS_ROOT / "train_phase2_generational.py")
    args = SimpleNamespace(
        require_slippage_pass=True,
        require_safe_shadow_pass=True,
        max_bad_fade_rate=0.18,
        min_trades_floor=20,
        drawdown_tolerance=1.05,
        loss_streak_tolerance=0,
    )
    champion = {
        "slippage": {"slip_1": {"trade_count": 30, "total_pnl_usd": 1000.0, "profit_factor": 1.2, "sharpe": 1.0, "max_drawdown": -1000.0}},
        "behavior_audit_test": {"countertrend_rate": 0.10},
        "max_consecutive_losses": 3,
    }
    candidate = {
        "slippage": {"slip_1": {"trade_count": 30, "total_pnl_usd": 1200.0, "profit_factor": 1.3, "sharpe": 1.1, "max_drawdown": -900.0}},
        "behavior_audit_test": {"countertrend_rate": 0.08},
        "max_consecutive_losses": 3,
        "slippage_pass": True,
        "live_shadow_pass": False,
        "promotion_blocked": True,
        "promotion_blocked_reason": "live_shadow_gate_failed",
    }

    ok, reasons = mod._candidate_passes_gates(args, champion, candidate)

    assert ok is False
    assert "live_shadow_gate_failed" in reasons
    assert "live shadow gate failed" in reasons


def test_reliability_matrix_builds_expected_experiments():
    mod = _load_module("run_es_reliability_matrix_test", TOOLS_ROOT / "run_es_reliability_matrix.py")
    args = SimpleNamespace(tag_prefix="es_reliability")

    specs = mod.build_experiment_specs(args)

    assert [spec["name"] for spec in specs] == [
        "baseline_5m",
        "mtf_15m_context",
        "mtf_15m_60m_context",
        "cleaner_trade_h12",
    ]
    assert specs[-1]["horizon"] == 12
    assert specs[-1]["mtf_timeframes"] == "15m,60m"
