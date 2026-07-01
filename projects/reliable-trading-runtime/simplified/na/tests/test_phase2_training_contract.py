from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from na.bot.feature_hash import compute_feature_hash
from na.bot.feature_constants import MANDATORY_MODEL_FEATURES
from na.bot.train_phase2 import _feature_contract_payload, _parse_args


ROOT = Path(__file__).resolve().parents[2]


def _load_strict_module():
    path = ROOT / "tools" / "train_retrain_v6_strict.py"
    spec = importlib.util.spec_from_file_location("train_retrain_v6_strict_contract_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_feature_contract_uses_exact_fitted_order() -> None:
    payload = _feature_contract_payload(["selected_b", "selected_a"], model_role="setup")

    assert payload["features"] == ["selected_b", "selected_a"]
    assert payload["feature_count"] == 2
    assert payload["feature_hash"] == compute_feature_hash(["selected_b", "selected_a"])


def test_feature_contract_rejects_empty_feature_set() -> None:
    with pytest.raises(ValueError, match="no fitted features"):
        _feature_contract_payload([], model_role="direction")


def test_sweep_selection_ignores_failed_rows() -> None:
    strict = _load_strict_module()
    selected = strict._select_successful_sweep_row(
        [
            {"trial": "failed", "direction_logloss_val": 0.01, "error": "feature mismatch"},
            {"trial": "valid_b", "direction_logloss_val": 0.42},
            {"trial": "valid_a", "direction_logloss_val": 0.31},
        ],
        metric="direction_logloss_val",
        phase_name="test sweep",
    )

    assert selected["trial"] == "valid_a"


def test_sweep_selection_fails_closed_when_every_trial_errors() -> None:
    strict = _load_strict_module()
    with pytest.raises(strict.StrictRunError, match="zero successful trials"):
        strict._select_successful_sweep_row(
            [
                {"direction_logloss_val": 9.99, "error": "feature mismatch"},
                {"direction_logloss_val": 9.99, "error": "bad calibration"},
            ],
            metric="direction_logloss_val",
            phase_name="test sweep",
        )


def test_direct_trainer_cli_exposes_execution_label_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_phase2",
            "--csv",
            "bars.csv",
            "--label-mode",
            "exec",
            "--label-max-hold-bars",
            "18",
            "--label-commission-per-contract",
            "2.5",
            "--label-slippage-ticks",
            "1.5",
        ],
    )
    args = _parse_args()

    assert args.label_mode == "exec"
    assert args.label_max_hold_bars == 18
    assert args.label_commission_per_contract == 2.5
    assert args.label_slippage_ticks == 1.5


def test_runtime_feature_hash_is_distinct_from_selected_model_hash() -> None:
    runtime_hash = compute_feature_hash(MANDATORY_MODEL_FEATURES)
    selected_hash = compute_feature_hash(["selected_b", "selected_a"])

    assert runtime_hash != selected_hash
