from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment contract
    raise SystemExit("PyYAML is required for phase2_rebuild_parity_audit.py") from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_deployable_phase2_candidate import _check_manifest, _load_manifest


DEFAULT_MASTER_CONFIG = ROOT / "na" / "config" / "master.yaml"
DEFAULT_RUNTIME_FILE = ROOT / "na" / "discord_addons" / "cli" / "stream_live_csv.py"
DEFAULT_CANDIDATE_ROOT = ROOT / "artifacts" / "phase2" / "candidates"
TOD_24H_REASON = "gate_tod_disabled_auto_24h"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_optional_path(raw: Any, *, manifest_dir: Path) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    candidate = (manifest_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (ROOT / path).resolve()


@dataclass
class CheckResult:
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase2ParityReport:
    preset: str
    tag: str
    decision: str
    summary: str
    preset_check: dict[str, Any]
    manifest_check: dict[str, Any]
    runtime_check: dict[str, Any]
    resolved_config_check: dict[str, Any]
    packaging_check: dict[str, Any]
    current_champion_note: dict[str, Any]
    next_actions: list[str]


def _load_master_preset(master_config: Path, preset: str) -> dict[str, Any]:
    payload = yaml.safe_load(master_config.read_text(encoding="utf-8"))
    risk_presets = payload.get("risk_presets") if isinstance(payload, dict) else None
    if not isinstance(risk_presets, dict):
        raise SystemExit(f"risk_presets missing in {master_config}")
    block = risk_presets.get(preset)
    if not isinstance(block, dict):
        raise SystemExit(f"preset {preset!r} not found in {master_config}")

    def _resolve(name: str, seen: set[str]) -> dict[str, Any]:
        if name in seen:
            raise SystemExit(f"cyclic inherits detected for preset {name!r}")
        current = risk_presets.get(name)
        if not isinstance(current, dict):
            raise SystemExit(f"preset {name!r} not found in {master_config}")
        parent_name = current.get("inherits")
        if not parent_name:
            return dict(current)
        parent = _resolve(str(parent_name), seen | {name})
        merged = dict(parent)
        merged.update(current)
        return merged

    return _resolve(preset, set())


def _check_preset(master_config: Path, preset: str) -> CheckResult:
    block = _load_master_preset(master_config, preset)
    phase2_tag = str(block.get("phase2_tag") or "").strip()
    close_model_path = str(block.get("close_model_path") or "").strip()
    close_threshold = block.get("phase2_close_threshold")
    phase2_enabled = bool(block.get("phase2"))
    gate_tod = bool(block.get("gate_tod"))
    trade_window_start = block.get("trade_window_start")
    trade_window_end = block.get("trade_window_end")
    ok = bool(phase2_enabled and phase2_tag)
    return CheckResult(
        ok=ok,
        details={
            "master_config": str(master_config),
            "phase2_enabled": phase2_enabled,
            "phase2_tag": phase2_tag,
            "phase2_close_enabled": bool(block.get("phase2_close_enabled")),
            "close_model_path": close_model_path or None,
            "phase2_close_threshold": close_threshold,
            "phase2_use_manifest_thresholds": bool(block.get("phase2_use_manifest_thresholds")),
            "gate_tod": gate_tod,
            "trade_window_start": trade_window_start,
            "trade_window_end": trade_window_end,
        },
    )


def _check_manifest_bundle(candidate_root: Path, tag: str) -> CheckResult:
    manifest_path = candidate_root / tag / "manifest.json"
    manifest = dict(_load_manifest(manifest_path))
    deployable_rc = _check_manifest(manifest_path, manifest)
    manifest_dir = manifest_path.parent
    close_path = _resolve_optional_path(manifest.get("close_model_path"), manifest_dir=manifest_dir)
    required_files = [
        "manifest.json",
        "setup.joblib",
        "setup.features.json",
        "setup.meta.json",
        "setup.registry.json",
        "dir.joblib",
        "dir.features.json",
        "dir.meta.json",
        "dir.registry.json",
    ]
    missing_files = [name for name in required_files if not (manifest_dir / name).exists()]
    thresholds = manifest.get("thresholds") if isinstance(manifest.get("thresholds"), dict) else {}
    threshold_values = {
        "p_setup": thresholds.get("p_setup"),
        "p_long": thresholds.get("p_long"),
        "p_short": thresholds.get("p_short"),
    }
    close_enabled = bool(((manifest.get("close") or {}) if isinstance(manifest.get("close"), dict) else {}).get("enabled"))
    close_exists = bool(close_path and close_path.exists())
    close_inside_candidate_dir = bool(close_path and close_path.resolve().parent == manifest_dir.resolve())
    ok = (
        deployable_rc == 0
        and not missing_files
        and all(v is not None for v in threshold_values.values())
        and (not close_enabled or close_exists)
        and (not close_enabled or close_inside_candidate_dir)
    )
    return CheckResult(
        ok=ok,
        details={
            "manifest_path": str(manifest_path),
            "deployable_check_passed": deployable_rc == 0,
            "promotion_result": manifest.get("promotion_result"),
            "thresholds": threshold_values,
            "close_enabled": close_enabled,
            "close_model_path": str(close_path) if close_path else None,
            "close_model_exists": close_exists,
            "close_model_inside_candidate_dir": close_inside_candidate_dir,
            "missing_candidate_files": missing_files,
        },
    )


def _check_runtime_contract(runtime_file: Path, *, preset: str) -> CheckResult:
    text = runtime_file.read_text(encoding="utf-8", errors="replace")
    checks = {
        "loads_manifest_from_candidate_tag": 'artifacts" / "phase2" / "candidates" / tag / "manifest.json"' in text,
        "exposes_phase2_tag_flag": '--phase2_tag' in text,
        "records_phase2_manifest_thresholds_used": "phase2_manifest_thresholds_used" in text,
        "resolves_setup_source_to_phase2_tag": 'source_map["p_setup_required"] = "phase2_tag"' in text,
        "resolves_long_source_to_phase2_tag": 'source_map["p_long_required"] = "phase2_tag"' in text,
        "resolves_short_source_to_phase2_tag": 'source_map["p_short_required"] = "phase2_tag"' in text,
        "references_close_model_path": "close_model_path" in text,
        "preset_is_live_approved": f'"{preset}"' in text and "LIVE_APPROVED_PRESETS" in text,
    }
    ok = all(checks.values())
    return CheckResult(
        ok=ok,
        details={
            "runtime_file": str(runtime_file),
            **checks,
        },
    )


def _check_resolved_config(
    resolved_config_path: Path | None,
    *,
    preset: str,
    tag: str,
    expected_thresholds: dict[str, Any],
    candidate_root: Path,
) -> CheckResult:
    if resolved_config_path is None:
        return CheckResult(
            ok=False,
            details={
                "resolved_config_path": None,
                "reason": "representative_resolved_config_not_provided",
            },
        )
    payload = _read_json(resolved_config_path)
    threshold_sources = payload.get("threshold_sources") if isinstance(payload.get("threshold_sources"), dict) else {}
    resolved_thresholds = payload.get("resolved_thresholds") if isinstance(payload.get("resolved_thresholds"), dict) else {}
    phase2_close = payload.get("phase2_close") if isinstance(payload.get("phase2_close"), dict) else {}
    trade_window_metadata = payload.get("trade_window_metadata") if isinstance(payload.get("trade_window_metadata"), dict) else {}
    checks = {
        "preset_matches": str(payload.get("preset") or "") == preset,
        "phase2_tag_matches": str(payload.get("phase2_tag") or "") == tag,
        "manifest_thresholds_used": bool(payload.get("phase2_manifest_thresholds_used")),
        "setup_source_phase2_tag": str(threshold_sources.get("p_setup_required") or "") == "phase2_tag",
        "long_source_phase2_tag": str(threshold_sources.get("p_long_required") or "") == "phase2_tag",
        "short_source_phase2_tag": str(threshold_sources.get("p_short_required") or "") == "phase2_tag",
        "resolved_setup_matches_manifest": resolved_thresholds.get("p_setup_required") == expected_thresholds.get("p_setup"),
        "resolved_long_matches_manifest": resolved_thresholds.get("p_long_required") == expected_thresholds.get("p_long"),
        "resolved_short_matches_manifest": resolved_thresholds.get("p_short_required") == expected_thresholds.get("p_short"),
        "close_runtime_enabled": bool(phase2_close.get("enabled")),
    }
    close_model_path = phase2_close.get("model_path")
    close_model_resolved = Path(str(close_model_path)).resolve() if close_model_path else None
    candidate_dir = (candidate_root / tag).resolve() if tag else None
    close_model_exists = bool(close_model_resolved and close_model_resolved.exists())
    checks["close_model_path_exists"] = close_model_exists
    checks["close_model_inside_candidate_dir"] = bool(
        close_model_resolved and candidate_dir and close_model_resolved.parent == candidate_dir
    )
    checks["explicit_tod_window_enforced"] = str(trade_window_metadata.get("effective_reason") or "") != TOD_24H_REASON
    ok = all(checks.values())
    return CheckResult(
        ok=ok,
        details={
            "resolved_config_path": str(resolved_config_path),
            "threshold_sources": threshold_sources,
            "resolved_thresholds": resolved_thresholds,
            "phase2_close": phase2_close,
            "trade_window_metadata": trade_window_metadata,
            **checks,
        },
    )


def _check_current_champion(candidate_root: Path, tag: str) -> dict[str, Any]:
    champion_path = candidate_root / "current_champion.json"
    payload = _read_json(champion_path)
    champion_tag = str(payload.get("tag") or "") if payload else ""
    return {
        "current_champion_path": str(champion_path),
        "current_champion_tag": champion_tag or None,
        "matches_active_preset_tag": champion_tag == tag if champion_tag else False,
        "note": "informational_only_preset_tag_is_runtime_truth",
    }


def _decision_from_checks(*, preset_check: CheckResult, manifest_check: CheckResult, runtime_check: CheckResult, resolved_config_check: CheckResult) -> tuple[str, str, list[str]]:
    resolved_false_checks = [
        key for key, value in resolved_config_check.details.items()
        if key.endswith(("_matches", "_used", "_tag", "_exists")) and value is False
    ]
    if (
        preset_check.ok
        and manifest_check.ok
        and runtime_check.ok
        and not resolved_config_check.ok
        and resolved_false_checks == ["close_model_path_exists"]
        and bool(manifest_check.details.get("close_model_exists"))
    ):
        return (
            "NO_RETRAIN_NEEDED",
            "Current candidate is deployable, but the representative runtime resolved a missing local close model path; treat this as a candidate reconstitution/repackage gap before considering retraining.",
            [
                "Reconstitute the current candidate bundle so the runtime resolves a real close model path for `retrain_v6_fixed_v2_hyper_016`.",
                "Rerun the parity audit after repairing the candidate close-model wiring; do not retrain unless parity still fails afterward.",
                "Keep `es_elite_v1` on the current tag until the repaired bundle produces a clean `resolved_config.json`.",
            ],
        )
    critical_failures = []
    if not preset_check.ok:
        critical_failures.append("preset_phase2_tag_missing_or_phase2_disabled")
    if not manifest_check.ok:
        critical_failures.append("candidate_manifest_or_bundle_not_deployable")
    if not runtime_check.ok:
        critical_failures.append("runtime_manifest_contract_drifted")
    if not resolved_config_check.ok:
        critical_failures.append("representative_runtime_resolution_not_in_parity")
    if not critical_failures:
        return (
            "NO_RETRAIN_NEEDED",
            "Current candidate bundle and runtime contract are intact; rebuild can stay on the packaging/revalidation path.",
            [
                "Keep `es_elite_v1` pointed at `retrain_v6_fixed_v2_hyper_016`.",
                "Revalidate the candidate bundle and live launch contract; do not retrain unless a future parity check fails.",
                "If you need a backup copy, reconstitute the candidate directory with the same manifest fields and close-model wiring.",
            ],
        )
    return (
        "RETRAIN_REQUIRED",
        "One or more critical parity checks failed; rebuild must branch into a new packaged candidate before any preset swap.",
        [
            "Train a new candidate tag under `artifacts/phase2/candidates/<new_tag>` only after fixing the failing parity checks.",
            "Package the new candidate with explicit manifest thresholds, valid registry files, and a real close-model path.",
            "Smoke the new tag explicitly with `--phase2_tag <new_tag>` and verify `resolved_config.json` before repointing `es_elite_v1`.",
        ]
        + [f"Fix parity failure: {item}" for item in critical_failures],
    )


def audit_phase2_rebuild_parity(
    *,
    preset: str,
    master_config: Path,
    runtime_file: Path,
    candidate_root: Path,
    resolved_config_path: Path | None,
) -> Phase2ParityReport:
    preset_check = _check_preset(master_config, preset)
    tag = str(preset_check.details.get("phase2_tag") or "")
    manifest_check = _check_manifest_bundle(candidate_root, tag) if tag else CheckResult(ok=False, details={"reason": "missing_tag"})
    runtime_check = _check_runtime_contract(runtime_file, preset=preset)
    expected_thresholds = dict((manifest_check.details.get("thresholds") or {})) if manifest_check.details else {}
    resolved_config_check = _check_resolved_config(
        resolved_config_path,
        preset=preset,
        tag=tag,
        expected_thresholds=expected_thresholds,
        candidate_root=candidate_root,
    )
    decision, summary, next_actions = _decision_from_checks(
        preset_check=preset_check,
        manifest_check=manifest_check,
        runtime_check=runtime_check,
        resolved_config_check=resolved_config_check,
    )
    return Phase2ParityReport(
        preset=preset,
        tag=tag,
        decision=decision,
        summary=summary,
        preset_check=preset_check.details,
        manifest_check=manifest_check.details,
        runtime_check=runtime_check.details,
        resolved_config_check=resolved_config_check.details,
        packaging_check={
            "training_entrypoint": str(ROOT / "tools" / "train_v2.py"),
            "packaging_tool": str(ROOT / "tools" / "package_train_v2_candidate.py"),
            "deployability_tool": str(ROOT / "tools" / "check_deployable_phase2_candidate.py"),
            "evaluation_tool": str(ROOT / "tools" / "run_sharp.py"),
            "runtime_consumer": str(runtime_file),
        },
        current_champion_note=_check_current_champion(candidate_root, tag),
        next_actions=next_actions,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether the current Phase-2 live chain can be rebuilt without retraining.")
    parser.add_argument("--preset", default="es_elite_v1")
    parser.add_argument("--master-config", default=str(DEFAULT_MASTER_CONFIG))
    parser.add_argument("--runtime-file", default=str(DEFAULT_RUNTIME_FILE))
    parser.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument("--resolved-config", default=None, help="Representative resolved_config.json from a real run.")
    parser.add_argument("--json-out", default=None, help="Optional output path for the report JSON.")
    args = parser.parse_args()

    report = audit_phase2_rebuild_parity(
        preset=str(args.preset),
        master_config=Path(args.master_config).expanduser().resolve(),
        runtime_file=Path(args.runtime_file).expanduser().resolve(),
        candidate_root=Path(args.candidate_root).expanduser().resolve(),
        resolved_config_path=Path(args.resolved_config).expanduser().resolve() if args.resolved_config else None,
    )
    payload = asdict(report)
    text = json.dumps(payload, ensure_ascii=True, indent=2)
    if args.json_out:
        Path(args.json_out).expanduser().resolve().write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
