from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_THRESHOLDS = ("p_setup", "p_long", "p_short")


def _load_manifest(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise SystemExit(f"FAIL incomplete_candidate manifest_missing path={path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"FAIL invalid_manifest path={path} error={exc}") from exc
    if not isinstance(payload, Mapping):
        raise SystemExit(f"FAIL invalid_manifest path={path} error=top_level_not_object")
    return payload


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest:
        return Path(args.manifest).expanduser().resolve()
    root = Path(args.artifact_root).expanduser().resolve()
    return root / str(args.tag) / "manifest.json"


def _check_manifest(path: Path, payload: Mapping[str, Any]) -> int:
    tag = str(payload.get("tag") or path.parent.name)
    thresholds = payload.get("thresholds") or {}
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    missing = [key for key in REQUIRED_THRESHOLDS if thresholds.get(key) is None]
    rejected = bool(payload.get("rejected"))
    promotion_result = str(payload.get("promotion_result") or "")
    errors = []
    warnings = []
    if rejected:
        errors.append(f"rejected={payload.get('rejected_reason') or 'true'}")
    if missing:
        errors.append("missing_thresholds=" + ",".join(missing))
    if promotion_result == "pending":
        errors.append("promotion_result=pending")
    elif not promotion_result:
        warnings.append("promotion_result=missing")
    if errors:
        print(f"FAIL tag={tag} path={path}")
        for item in errors:
            print(f"  {item}")
        for item in warnings:
            print(f"  WARN {item}")
        return 1
    print(
        "PASS "
        f"tag={tag} "
        f"p_setup={thresholds.get('p_setup')} "
        f"p_long={thresholds.get('p_long')} "
        f"p_short={thresholds.get('p_short')} "
        f"promotion_result={promotion_result or 'missing'}"
    )
    for item in warnings:
        print(f"  WARN {item}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a Phase-2 candidate manifest is deployable.")
    parser.add_argument("--manifest", default=None, help="Path to candidate manifest.json.")
    parser.add_argument("--tag", default=None, help="Candidate tag under --artifact-root.")
    parser.add_argument(
        "--artifact-root",
        default=str(Path(__file__).resolve().parents[1] / "artifacts" / "phase2" / "candidates"),
        help="Phase-2 candidates root used with --tag.",
    )
    args = parser.parse_args()
    if not args.manifest and not args.tag:
        parser.error("provide --manifest or --tag")
    path = _manifest_path(args)
    return _check_manifest(path, _load_manifest(path))


if __name__ == "__main__":
    raise SystemExit(main())
