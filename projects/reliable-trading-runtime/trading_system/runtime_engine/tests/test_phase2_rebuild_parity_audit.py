from __future__ import annotations

import json
from pathlib import Path

from trading_system.development_tools.phase2_rebuild_parity_audit import audit_phase2_rebuild_parity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _master_yaml(tag: str, close_path: str) -> str:
    return f"""
risk_presets:
  es_elite_v1:
    phase2: true
    phase2_tag: {tag}
    phase2_close_enabled: true
    close_model_path: {close_path}
    phase2_close_threshold: 0.90
    phase2_use_manifest_thresholds: true
    gate_tod: true
    trade_window_start: "07:00"
    trade_window_end: "14:00"
  es_elite_v1_live_ready:
    inherits: es_elite_v1
    phase2_tag: {tag}
    close_model_path: {close_path}
"""


def _runtime_stub() -> str:
    return """
manifest_path = root / "artifacts" / "phase2" / "candidates" / tag / "manifest.json"
source_map["p_setup_required"] = "phase2_tag"
source_map["p_long_required"] = "phase2_tag"
source_map["p_short_required"] = "phase2_tag"
phase2_manifest_thresholds_used = True
close_model_path = "close.joblib"
LIVE_APPROVED_PRESETS = ("es_elite_v1",)
--phase2_tag
"""


def _candidate_manifest(close_path: str, *, promotion_result: str = "manual") -> dict:
    return {
        "tag": "retrain_v6_fixed_v2_hyper_016",
        "thresholds": {"p_setup": 0.06, "p_long": 0.57, "p_short": 0.57},
        "close_model_path": close_path,
        "close": {"enabled": True, "model_path": close_path},
        "promotion_result": promotion_result,
        "rejected": False,
    }


def _resolved_config(close_path: str) -> dict:
    return {
        "preset": "es_elite_v1",
        "phase2_tag": "retrain_v6_fixed_v2_hyper_016",
        "phase2_manifest_thresholds_used": True,
        "threshold_sources": {
            "p_setup_required": "phase2_tag",
            "p_long_required": "phase2_tag",
            "p_short_required": "phase2_tag",
        },
        "resolved_thresholds": {
            "p_setup_required": 0.06,
            "p_long_required": 0.57,
            "p_short_required": 0.57,
        },
        "phase2_close": {
            "enabled": True,
            "model_path": close_path,
        },
        "trade_window_metadata": {
            "effective_reason": "configured_window",
        },
    }


def _candidate_files(candidate_dir: Path) -> None:
    for name in (
        "setup.joblib",
        "setup.features.json",
        "setup.meta.json",
        "setup.registry.json",
        "dir.joblib",
        "dir.features.json",
        "dir.meta.json",
        "dir.registry.json",
    ):
        _write_text(candidate_dir / name, "{}")


def test_phase2_rebuild_parity_no_retrain_needed(tmp_path: Path) -> None:
    repo_root = tmp_path
    close_model = repo_root / "artifacts" / "phase2" / "candidates" / "retrain_v6_fixed_v2_hyper_016" / "close.joblib"
    _write_text(close_model, "close")

    master_config = repo_root / "runtime_engine" / "runtime_config" / "master.yaml"
    _write_text(master_config, _master_yaml("retrain_v6_fixed_v2_hyper_016", str(close_model)))

    runtime_file = repo_root / "runtime_engine" / "integrations" / "cli" / "live_trading_runtime.py"
    _write_text(runtime_file, _runtime_stub())

    candidate_dir = repo_root / "artifacts" / "phase2" / "candidates" / "retrain_v6_fixed_v2_hyper_016"
    _candidate_files(candidate_dir)
    _write_json(candidate_dir / "manifest.json", _candidate_manifest("close.joblib"))
    _write_json(repo_root / "artifacts" / "phase2" / "candidates" / "current_champion.json", {"tag": "retrain_v6_pass2_grid_02"})

    resolved_config = repo_root / "run" / "resolved_config.json"
    _write_json(resolved_config, _resolved_config(str(close_model)))

    report = audit_phase2_rebuild_parity(
        preset="es_elite_v1",
        master_config=master_config,
        runtime_file=runtime_file,
        candidate_root=repo_root / "artifacts" / "phase2" / "candidates",
        resolved_config_path=resolved_config,
    )

    assert report.decision == "NO_RETRAIN_NEEDED"
    assert report.manifest_check["deployable_check_passed"] is True
    assert report.resolved_config_check["manifest_thresholds_used"] is True
    assert report.runtime_check["loads_manifest_from_candidate_tag"] is True
    assert report.runtime_check["preset_is_live_approved"] is True


def test_phase2_rebuild_parity_requires_retrain_on_close_model_break(tmp_path: Path) -> None:
    repo_root = tmp_path
    close_model = repo_root / "missing" / "close.joblib"

    master_config = repo_root / "runtime_engine" / "runtime_config" / "master.yaml"
    _write_text(master_config, _master_yaml("retrain_v6_fixed_v2_hyper_016", str(close_model)))

    runtime_file = repo_root / "runtime_engine" / "integrations" / "cli" / "live_trading_runtime.py"
    _write_text(runtime_file, _runtime_stub())

    candidate_dir = repo_root / "artifacts" / "phase2" / "candidates" / "retrain_v6_fixed_v2_hyper_016"
    _candidate_files(candidate_dir)
    _write_json(candidate_dir / "manifest.json", _candidate_manifest(str(close_model)))
    _write_json(repo_root / "artifacts" / "phase2" / "candidates" / "current_champion.json", {"tag": "retrain_v6_pass2_grid_02"})

    resolved_config = repo_root / "run" / "resolved_config.json"
    _write_json(resolved_config, _resolved_config(str(close_model)))

    report = audit_phase2_rebuild_parity(
        preset="es_elite_v1",
        master_config=master_config,
        runtime_file=runtime_file,
        candidate_root=repo_root / "artifacts" / "phase2" / "candidates",
        resolved_config_path=resolved_config,
    )

    assert report.decision == "RETRAIN_REQUIRED"
    assert report.manifest_check["close_model_exists"] is False


def test_phase2_rebuild_parity_requires_explicit_tod_window(tmp_path: Path) -> None:
    repo_root = tmp_path
    close_model = repo_root / "artifacts" / "phase2" / "candidates" / "retrain_v6_fixed_v2_hyper_016" / "close.joblib"
    _write_text(close_model, "close")

    master_config = repo_root / "runtime_engine" / "runtime_config" / "master.yaml"
    _write_text(master_config, _master_yaml("retrain_v6_fixed_v2_hyper_016", str(close_model)))

    runtime_file = repo_root / "runtime_engine" / "integrations" / "cli" / "live_trading_runtime.py"
    _write_text(runtime_file, _runtime_stub())

    candidate_dir = repo_root / "artifacts" / "phase2" / "candidates" / "retrain_v6_fixed_v2_hyper_016"
    _candidate_files(candidate_dir)
    _write_json(candidate_dir / "manifest.json", _candidate_manifest("close.joblib"))
    _write_json(repo_root / "artifacts" / "phase2" / "candidates" / "current_champion.json", {"tag": "retrain_v6_pass2_grid_02"})

    resolved_config = repo_root / "run" / "resolved_config.json"
    payload = _resolved_config(str(close_model))
    payload["trade_window_metadata"]["effective_reason"] = "gate_tod_disabled_auto_24h"
    _write_json(resolved_config, payload)

    report = audit_phase2_rebuild_parity(
        preset="es_elite_v1",
        master_config=master_config,
        runtime_file=runtime_file,
        candidate_root=repo_root / "artifacts" / "phase2" / "candidates",
        resolved_config_path=resolved_config,
    )

    assert report.decision == "RETRAIN_REQUIRED"
    assert report.resolved_config_check["explicit_tod_window_enforced"] is False


def test_phase2_rebuild_parity_resolves_inherited_preset(tmp_path: Path) -> None:
    repo_root = tmp_path
    close_model = repo_root / "artifacts" / "phase2" / "candidates" / "retrain_v6_fixed_v2_hyper_016" / "close.joblib"
    _write_text(close_model, "close")

    master_config = repo_root / "runtime_engine" / "runtime_config" / "master.yaml"
    _write_text(master_config, _master_yaml("retrain_v6_fixed_v2_hyper_016", str(close_model)))

    runtime_file = repo_root / "runtime_engine" / "integrations" / "cli" / "live_trading_runtime.py"
    _write_text(runtime_file, """
manifest_path = root / "artifacts" / "phase2" / "candidates" / tag / "manifest.json"
source_map["p_setup_required"] = "phase2_tag"
source_map["p_long_required"] = "phase2_tag"
source_map["p_short_required"] = "phase2_tag"
phase2_manifest_thresholds_used = True
close_model_path = "close.joblib"
LIVE_APPROVED_PRESETS = ("es_elite_v1_live_ready",)
--phase2_tag
""")

    candidate_dir = repo_root / "artifacts" / "phase2" / "candidates" / "retrain_v6_fixed_v2_hyper_016"
    _candidate_files(candidate_dir)
    _write_json(candidate_dir / "manifest.json", _candidate_manifest("close.joblib"))
    _write_json(repo_root / "artifacts" / "phase2" / "candidates" / "current_champion.json", {"tag": "retrain_v6_pass2_grid_02"})

    resolved_config = repo_root / "run" / "resolved_config.json"
    payload = _resolved_config(str(close_model))
    payload["preset"] = "es_elite_v1_live_ready"
    _write_json(resolved_config, payload)

    report = audit_phase2_rebuild_parity(
        preset="es_elite_v1_live_ready",
        master_config=master_config,
        runtime_file=runtime_file,
        candidate_root=repo_root / "artifacts" / "phase2" / "candidates",
        resolved_config_path=resolved_config,
    )

    assert report.decision == "NO_RETRAIN_NEEDED"
    assert report.preset_check["phase2_enabled"] is True
