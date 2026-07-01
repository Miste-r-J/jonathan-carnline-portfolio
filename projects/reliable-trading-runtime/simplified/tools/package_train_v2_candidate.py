from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from na.bot.feature_constants import MANDATORY_MODEL_FEATURES
from na.bot.feature_hash import compute_feature_hash

ARTIFACT_ROOT = ROOT / "artifacts" / "phase2" / "candidates"
CORE_MODEL_ARTIFACTS = (
    "setup.joblib",
    "setup.features.json",
    "setup.label_schema.json",
    "setup.feature_bounds.json",
    "setup.meta.json",
    "setup.metrics.json",
    "setup.registry.json",
    "setup_raw.joblib",
    "dir.joblib",
    "dir.features.json",
    "dir.label_schema.json",
    "dir.feature_bounds.json",
    "dir.meta.json",
    "dir.metrics.json",
    "dir.registry.json",
    "dir_raw.joblib",
)
CLOSE_MODEL_ARTIFACTS = (
    "close.joblib",
    "close.features.json",
    "close.label_schema.json",
    "close.meta.json",
    "close.metrics.json",
    "close.registry.json",
    "close_raw.joblib",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        _copy(src, dst)


def _resolve_v2_meta(path: Path) -> Path:
    if path.suffix == ".json":
        return path
    return path.with_suffix(".meta.json")


def _label_schema() -> dict[str, Any]:
    return {
        "domain": "directional_binary",
        "positive_label": 1,
        "negative_label": 0,
        "params": {
            "source": "train_v2_bundle",
            "mapping": {
                "0": "SHORT",
                "1": "LONG",
            },
        },
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _manifest_dir_for_tag(tag: str) -> Path:
    return ARTIFACT_ROOT / str(tag)


def _resolve_manifest_artifact_dir(manifest: dict[str, Any], *, manifest_dir: Path) -> Path:
    raw = manifest.get("artifact_dir")
    if not raw or str(raw).strip() == ".":
        return manifest_dir
    artifact_dir = Path(str(raw)).expanduser()
    if artifact_dir.is_absolute():
        return artifact_dir
    return (manifest_dir / artifact_dir).resolve()


def _resolve_optional_artifact_path(raw: Any, *, manifest_dir: Path, artifact_dir: Path) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    local_candidate = (manifest_dir / path).resolve()
    if local_candidate.exists():
        return local_candidate
    artifact_candidate = (artifact_dir / path).resolve()
    if artifact_candidate.exists():
        return artifact_candidate
    return local_candidate


def _normalized_close_manifest(close_payload: dict[str, Any]) -> dict[str, Any]:
    close_cfg = dict(close_payload or {})
    close_cfg["enabled"] = bool(close_cfg.get("enabled", True))
    close_cfg["model_path"] = "close.joblib"
    return close_cfg


def _feature_list(path: Path) -> list[str]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        return [str(item) for item in (payload.get("features") or [])]
    return []


def _rewrite_registry_model_path(out_dir: Path, role: str) -> None:
    registry_path = out_dir / f"{role}.registry.json"
    payload: dict[str, Any] = {}
    if registry_path.exists():
        try:
            payload = _load_json(registry_path)
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload["model_path"] = str((out_dir / f"{role}.joblib").resolve())
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _apply_close_overrides(close_cfg: dict[str, Any], args: argparse.Namespace, *, default_model_path: str) -> dict[str, Any]:
    cfg = dict(close_cfg or {})
    if getattr(args, "close_enabled", None) is not None:
        cfg["enabled"] = bool(args.close_enabled)
    else:
        cfg["enabled"] = bool(cfg.get("enabled", True))
    if getattr(args, "close_threshold", None) is not None:
        cfg["threshold"] = float(args.close_threshold)
    elif cfg.get("threshold") is not None:
        cfg["threshold"] = float(cfg["threshold"])
    model_path = str(getattr(args, "close_model_path", None) or cfg.get("model_path") or default_model_path)
    cfg["model_path"] = model_path
    return cfg


def build_candidate(args: argparse.Namespace) -> Path:
    v2_meta_path = _resolve_v2_meta(Path(args.v2_meta).expanduser().resolve())
    v2_meta = _load_json(v2_meta_path)
    v2_model_path = Path(v2_meta["model_path"])
    if not v2_model_path.is_absolute():
        v2_model_path = (ROOT / v2_model_path).resolve()
    baseline_dir = ARTIFACT_ROOT / str(args.baseline_tag)
    baseline_manifest_path = baseline_dir / "manifest.json"
    baseline_manifest = _load_json(baseline_manifest_path)
    tag = str(args.tag or f"retrain_v2_bundle_{v2_meta['run_id']}")
    out_dir = ARTIFACT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in CORE_MODEL_ARTIFACTS[:8] + CLOSE_MODEL_ARTIFACTS:
        src = baseline_dir / name
        if src.exists():
            _copy(src, out_dir / name)

    _copy(v2_model_path, out_dir / "dir.joblib")
    _copy(v2_meta_path, out_dir / "dir.meta.json")
    bundle = joblib.load(out_dir / "dir.joblib")
    dir_features = [str(col) for col in list(bundle.get("feature_names") or [])]
    (out_dir / "dir.features.json").write_text(
        json.dumps({"features": dir_features}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dir.label_schema.json").write_text(json.dumps(_label_schema(), indent=2), encoding="utf-8")
    (out_dir / "dir.metrics.json").write_text(
        json.dumps(v2_meta.get("metrics") or {}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dir.registry.json").write_text(
        json.dumps({"model_path": f"simplified\\artifacts\\phase2\\candidates\\{tag}\\dir.joblib"}, indent=2),
        encoding="utf-8",
    )

    baseline_thresholds = baseline_manifest.get("thresholds") or {}
    config = dict(baseline_manifest.get("config") or {})
    config["direction_model_family"] = "train_v2_bundle"
    config["direction_bundle_run_id"] = v2_meta.get("run_id")
    config["direction_feature_sources"] = v2_meta.get("feature_sources") or ["5m", "15m", "daily"]
    manifest = {
        "tag": tag,
        "generated_at": v2_meta.get("generated_at"),
        "csv": "data\\intraday\\es\\ES6.csv",
        "instrument": baseline_manifest.get("instrument", "ES"),
        "timeframe": "5m",
        "artifact_dir": ".",
        "setup_model_path": "setup.joblib",
        "dir_model_path": "dir.joblib",
        "close_model_path": "close.joblib",
        "thresholds": {
            "p_setup": float(args.p_setup if args.p_setup is not None else baseline_thresholds.get("p_setup", 0.35)),
            "p_long": float(args.p_long if args.p_long is not None else 0.72),
            "p_short": float(args.p_short if args.p_short is not None else 0.72),
        },
        "close": _apply_close_overrides(
            _normalized_close_manifest(dict(baseline_manifest.get("close") or {})),
            args,
            default_model_path="close.joblib",
        ),
        "config": config,
        "metrics": {
            "setup": (baseline_manifest.get("metrics") or {}).get("setup"),
            "direction": v2_meta.get("metrics"),
            "close": (baseline_manifest.get("metrics") or {}).get("close"),
        },
        "direction_diagnostics": {
            "mean": ((v2_meta.get("metrics") or {}).get("test") or {}).get("prob_mean"),
            "std": ((v2_meta.get("metrics") or {}).get("test") or {}).get("prob_std"),
            "extreme_frac": ((v2_meta.get("metrics") or {}).get("test") or {}).get("extreme_frac"),
        },
        "rejected": False,
        "rejected_reason": None,
        "promotion_result": "paper_candidate",
        "parent_tag": str(args.baseline_tag),
        "source_model_path": str(v2_model_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def build_rebundled_candidate(args: argparse.Namespace) -> Path:
    source_tag = str(args.source_tag or "").strip()
    if not source_tag:
        raise SystemExit("--source-tag is required when --v2-meta is not provided")
    source_dir = _manifest_dir_for_tag(source_tag)
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = _load_json(source_manifest_path)
    source_artifact_dir = _resolve_manifest_artifact_dir(source_manifest, manifest_dir=source_dir)

    tag = str(args.tag or f"{source_tag}_livebundle_v1")
    out_dir = ARTIFACT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in CORE_MODEL_ARTIFACTS:
        _copy_if_exists(source_artifact_dir / name, out_dir / name)
        _copy_if_exists(source_dir / name, out_dir / name)

    close_source_override = str(args.close_source_path or "").strip()
    close_model_source = (
        Path(close_source_override).expanduser().resolve()
        if close_source_override
        else _resolve_optional_artifact_path(
            source_manifest.get("close_model_path"),
            manifest_dir=source_dir,
            artifact_dir=source_artifact_dir,
        )
    )
    if close_model_source is None or not close_model_source.exists():
        raise SystemExit(f"close model source missing for tag={source_tag}")
    close_source_dir = close_model_source.parent
    for name in CLOSE_MODEL_ARTIFACTS:
        src = close_source_dir / name
        if name == "close.joblib" and not src.exists():
            src = close_model_source
        if not src.exists():
            if name == "close_raw.joblib":
                continue
            raise SystemExit(f"required close artifact missing: {src}")
        _copy(src, out_dir / name)

    for role in ("setup", "dir", "close"):
        _rewrite_registry_model_path(out_dir, role)

    setup_features = _feature_list(out_dir / "setup.features.json")
    dir_features = _feature_list(out_dir / "dir.features.json")
    close_features = _feature_list(out_dir / "close.features.json")
    runtime_feature_hash = compute_feature_hash(MANDATORY_MODEL_FEATURES)
    setup_feature_hash = compute_feature_hash(setup_features) if setup_features else None
    direction_feature_hash = compute_feature_hash(dir_features) if dir_features else None
    close_feature_hash = compute_feature_hash(close_features) if close_features else None

    thresholds = dict(source_manifest.get("thresholds") or {})
    close_cfg = _apply_close_overrides(
        _normalized_close_manifest(dict(source_manifest.get("close") or {})),
        args,
        default_model_path="close.joblib",
    )
    close_cfg.update(
        {
            "feature_hash": close_feature_hash,
            "feature_count": len(close_features),
            "label_schema_file": "close.label_schema.json" if close_features else None,
        }
    )
    close_model_path = str(getattr(args, "close_model_path", None) or "close.joblib")
    manifest = {
        "tag": tag,
        "generated_at": _utc_now_iso(),
        "csv": source_manifest.get("csv", "data\\intraday\\es\\ES.csv"),
        "instrument": source_manifest.get("instrument", "ES"),
        "timeframe": source_manifest.get("timeframe", "5m"),
        "artifact_dir": ".",
        "setup_model_path": "setup.joblib",
        "dir_model_path": "dir.joblib",
        "close_model_path": close_model_path,
        "thresholds": {
            "p_setup": float(args.p_setup if args.p_setup is not None else thresholds.get("p_setup", 0.35)),
            "p_long": float(args.p_long if args.p_long is not None else thresholds.get("p_long", 0.57)),
            "p_short": float(args.p_short if args.p_short is not None else thresholds.get("p_short", 0.57)),
        },
        "feature_hash": runtime_feature_hash,
        "runtime_feature_hash": runtime_feature_hash,
        "feature_hashes": {
            "setup": setup_feature_hash,
            "direction": direction_feature_hash,
            "close": close_feature_hash,
        },
        "close": close_cfg,
        "config": dict(source_manifest.get("config") or {}),
        "metrics": dict(source_manifest.get("metrics") or {}),
        "direction_diagnostics": dict(source_manifest.get("direction_diagnostics") or {}),
        "rejected": bool(source_manifest.get("rejected", False)),
        "rejected_reason": source_manifest.get("rejected_reason"),
        "promotion_result": str(args.promotion_result or source_manifest.get("promotion_result") or "paper_candidate"),
        "parent_tag": source_tag,
        "close_parent_path": str(close_model_source),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a train_v2 direction artifact as a Phase-2 candidate.")
    parser.add_argument("--v2-meta", default=None, help="Path to direction_v2_*.meta.json or matching .joblib.")
    parser.add_argument("--source-tag", default=None, help="Existing candidate tag to rebundle into a self-contained live bundle.")
    parser.add_argument("--close-source-path", default=None, help="Optional explicit close.joblib source path for rebundle mode.")
    parser.add_argument("--baseline-tag", default="retrain_v6_pass2_grid_02")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--p-setup", dest="p_setup", type=float, default=None)
    parser.add_argument("--p-long", dest="p_long", type=float, default=None)
    parser.add_argument("--p-short", dest="p_short", type=float, default=None)
    parser.add_argument("--promotion-result", default="paper_candidate")
    parser.add_argument("--close-enabled", dest="close_enabled", action="store_true", default=None)
    parser.add_argument("--no-close-enabled", dest="close_enabled", action="store_false")
    parser.add_argument("--close-threshold", dest="close_threshold", type=float, default=None)
    parser.add_argument("--close-model-path", dest="close_model_path", default=None)
    args = parser.parse_args()
    if bool(args.v2_meta) == bool(args.source_tag):
        parser.error("provide exactly one of --v2-meta or --source-tag")
    out_dir = build_candidate(args) if args.v2_meta else build_rebundled_candidate(args)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
