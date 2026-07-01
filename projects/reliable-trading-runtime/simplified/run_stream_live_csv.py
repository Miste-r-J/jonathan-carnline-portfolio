from __future__ import annotations

import importlib
import sys
from pathlib import Path

AGGRESSIVE_PHASE2_TAG = "retrain_v4"
AGGRESSIVE_PRESET = "es_maxpack_10_full_send_prop_safe_pnl"
AGGRESSIVE_MIN_LOSSES_PER_DAY = 4
ACTIVITY_REBALANCE_PRESET = "es_maxpack_10_full_send_prop_challenge_community"
ACTIVITY_REBALANCE_THRESHOLDS = {
    "--p_setup": "0.30",
    "--p_long": "0.55",
    "--p_short": "0.55",
}
LIVE_READY_PRESETS = {
    "es_elite_v1_live_ready",
    "es_elite_v1_live_ready_24h_test",
}
RUN70_PARITY_PROFILE = "run70_parity_live"


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _flag_value(argv: list[str], flag: str) -> str | None:
    for idx, arg in enumerate(argv):
        if arg == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        prefix = f"{flag}="
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _with_aggressive_setup_override(argv: list[str]) -> list[str]:
    preset = _flag_value(argv, "--preset")
    if preset != ACTIVITY_REBALANCE_PRESET:
        return argv
    if not _has_flag(argv, "--force_cli_thresholds"):
        return argv
    updated = list(argv)
    for flag, value in ACTIVITY_REBALANCE_THRESHOLDS.items():
        if not _has_flag(updated, flag):
            updated.extend([flag, value])
    return updated


def _warn_aggressive_loss_cap(argv: list[str]) -> None:
    return


def _with_run_profile(argv: list[str]) -> list[str]:
    profile = _flag_value(argv, "--run_profile")
    if profile != RUN70_PARITY_PROFILE:
        return argv
    updated: list[str] = []
    skip_next = False
    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--run_profile":
            if idx + 1 < len(argv):
                skip_next = True
            continue
        if arg.startswith("--run_profile="):
            continue
        updated.append(arg)
    if not _has_flag(updated, "--run70_parity_live_profile"):
        updated.append("--run70_parity_live_profile")
    if not _has_flag(updated, "--replay_execution_intended"):
        updated.append("--replay_execution_intended")
    if not _has_flag(updated, "--allow_replay_exec_to_nt"):
        updated.append("--allow_replay_exec_to_nt")
    return updated


def _strip_live_ready_threshold_bypass(argv: list[str]) -> list[str]:
    preset = _flag_value(argv, "--preset")
    if preset not in LIVE_READY_PRESETS:
        return argv
    updated: list[str] = []
    skip_next = False
    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in {
            "--no_phase2_use_manifest_thresholds",
            "--allow_preset_manifest_threshold_mismatch",
        }:
            continue
        if arg.startswith("--no_phase2_use_manifest_thresholds="):
            continue
        if arg.startswith("--allow_preset_manifest_threshold_mismatch="):
            continue
        if arg in {"--phase2_use_manifest_thresholds"}:
            continue
        if arg == "--phase2_use_manifest_thresholds" and idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
            skip_next = True
            continue
        updated.append(arg)
    if not _has_flag(updated, "--phase2_use_manifest_thresholds"):
        updated.append("--phase2_use_manifest_thresholds")
    return updated


def main(argv: list[str] | None = None) -> None:
    argv = _with_aggressive_setup_override(list(argv if argv is not None else sys.argv[1:]))
    argv = _with_run_profile(argv)
    argv = _strip_live_ready_threshold_bypass(argv)
    _warn_aggressive_loss_cap(argv)
    simplified_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(simplified_root))

    for key in list(sys.modules):
        if key == "na" or key.startswith("na."):
            sys.modules.pop(key, None)

    mod = importlib.import_module("na.discord_addons.cli.stream_live_csv")
    print(f"[simplified] using `na` from: {importlib.import_module('na').__file__}")
    print(f"[simplified] using stream_live_csv: {mod.__file__}")

    run = getattr(mod, "main", None)
    if not callable(run):
        raise SystemExit("na.discord_addons.cli.stream_live_csv.main not found")
    run(argv)


if __name__ == "__main__":
    main()
