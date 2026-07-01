"""Config integrity: same-level duplicate YAML keys must fail fast, and the
cleaned master.yaml must keep resolving the expected es_elite_v1 values."""
from pathlib import Path

import pytest

from trading_system.runtime_engine.runtime_config.registry import MASTER_PATH, _load_yaml, get_registry


def test_master_yaml_loads_without_duplicates():
    # The real, cleaned master.yaml must load cleanly through the strict loader.
    data = _load_yaml(MASTER_PATH)
    assert "risk_presets" in data


def test_same_level_duplicate_key_raises(tmp_path: Path):
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "risk_presets:\n"
        "  demo:\n"
        "    max_daily_loss: 400\n"
        "    max_daily_loss: 500\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        _load_yaml(bad)
    assert "max_daily_loss" in str(exc.value)


def test_cross_level_risk_block_split_is_allowed(tmp_path: Path):
    # A key under `risk:` AND at the preset top-level is NOT a same-level
    # duplicate (different parents) and must remain legal.
    ok = tmp_path / "split.yaml"
    ok.write_text(
        "risk_presets:\n"
        "  demo:\n"
        "    max_daily_loss: 500\n"
        "    risk:\n"
        "      max_daily_loss: 400\n",
        encoding="utf-8",
    )
    data = _load_yaml(ok)  # must not raise
    assert data["risk_presets"]["demo"]["max_daily_loss"] == 500


def test_es_elite_v1_resolved_values_unchanged():
    get_registry.cache_clear()
    reg = get_registry()
    params = reg.risk_presets["es_elite_v1"].as_cli_overrides()
    assert params["min_execute_grade"] == "A+"
    assert params["p_long"] == pytest.approx(0.70)
    assert params["p_short"] == pytest.approx(0.68)
    assert params["max_daily_loss"] == 500
    assert params["max_position_contracts"] == 2
    assert params["weekend_flatten"] is True
    assert params["max_trades_per_day"] == 10
