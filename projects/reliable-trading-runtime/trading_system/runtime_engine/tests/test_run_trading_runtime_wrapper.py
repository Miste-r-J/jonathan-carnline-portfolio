from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_wrapper():
    path = Path(__file__).resolve().parents[2] / "run_trading_runtime.py"
    spec = importlib.util.spec_from_file_location("run_live_trading_runtime_wrapper", path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _aggressive_args(*extra: str) -> list[str]:
    return [
        "--phase2",
        "--phase2_tag",
        "retrain_v4",
        "--preset",
        "es_maxpack_10_full_send_prop_safe_pnl",
        *extra,
    ]


def _activity_rebalance_args(*extra: str) -> list[str]:
    return [
        "--phase2",
        "--phase2_tag",
        "retrain_v6_pass2_grid_02",
        "--preset",
        "es_maxpack_10_full_send_prop_challenge_community",
        *extra,
    ]


def test_wrapper_does_not_append_hidden_setup_override() -> None:
    mod = _load_wrapper()

    argv = mod._with_aggressive_setup_override(_aggressive_args("--max_losses_per_day", "2"))

    assert "--p_setup" not in argv
    assert argv[argv.index("--max_losses_per_day") + 1] == "2"


def test_aggressive_wrapper_warning_is_disabled(capsys) -> None:
    mod = _load_wrapper()
    argv = _aggressive_args("--p_setup", "0.027", "--max_losses_per_day", "2")

    mod._warn_aggressive_loss_cap(argv)

    assert capsys.readouterr().err == ""


def test_aggressive_wrapper_accepts_four_loss_cap_without_warning(capsys) -> None:
    mod = _load_wrapper()
    argv = _aggressive_args("--p_setup", "0.027", "--max_losses_per_day", "4")

    mod._warn_aggressive_loss_cap(argv)

    assert capsys.readouterr().err == ""


def test_activity_rebalance_wrapper_adds_cli_threshold_overrides() -> None:
    mod = _load_wrapper()

    argv = mod._with_aggressive_setup_override(
        _activity_rebalance_args("--max_losses_per_day", "3", "--force_cli_thresholds")
    )

    assert argv[argv.index("--p_setup") + 1] == "0.30"
    assert argv[argv.index("--p_long") + 1] == "0.55"
    assert argv[argv.index("--p_short") + 1] == "0.55"


def test_activity_rebalance_wrapper_does_not_inject_thresholds_without_force_flag() -> None:
    mod = _load_wrapper()

    argv = mod._with_aggressive_setup_override(_activity_rebalance_args("--max_losses_per_day", "3"))

    assert "--p_setup" not in argv
    assert "--p_long" not in argv
    assert "--p_short" not in argv


def test_activity_rebalance_wrapper_preserves_explicit_cli_thresholds() -> None:
    mod = _load_wrapper()

    argv = mod._with_aggressive_setup_override(
        _activity_rebalance_args("--p_setup", "0.31", "--p_long", "0.56", "--p_short", "0.54")
    )

    assert argv[argv.index("--p_setup") + 1] == "0.31"
    assert argv[argv.index("--p_long") + 1] == "0.56"
    assert argv[argv.index("--p_short") + 1] == "0.54"


def test_run70_parity_profile_injects_expected_flags() -> None:
    mod = _load_wrapper()

    argv = mod._with_run_profile(
        [
            "--run_profile",
            "run70_parity_live",
            "--run_mode",
            "live",
        ]
    )

    assert "--run_profile" not in argv
    assert "--run70_parity_live_profile" in argv
    assert "--replay_execution_intended" in argv
    assert "--allow_replay_exec_to_nt" in argv


def test_live_ready_wrapper_strips_manifest_bypass_flags() -> None:
    mod = _load_wrapper()

    argv = mod._strip_live_ready_threshold_bypass(
        [
            "--phase2",
            "--preset",
            "es_elite_v1_live_ready_24h_test",
            "--no_phase2_use_manifest_thresholds",
            "--allow_preset_manifest_threshold_mismatch",
            "--execution_mode",
            "model_master",
        ]
    )

    assert "--no_phase2_use_manifest_thresholds" not in argv
    assert "--allow_preset_manifest_threshold_mismatch" not in argv
    assert "--phase2_use_manifest_thresholds" in argv
