from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_POINTER_PATH = Path(
    os.getenv(
        "STATUS_CHAMPION_POINTER_PATH",
        _REPO_ROOT / "runs" / "stream_sim_tests" / "champion_pointer.json",
    )
).resolve()


@dataclass(frozen=True)
class ChampionPresetPointer:
    preset_name: str
    bundle_path: Path
    created_at: datetime
    context: Optional[str] = None
    model_sha: Optional[str] = None
    symbol: Optional[str] = None


@dataclass(frozen=True)
class ChampionArtifacts:
    pointer: ChampionPresetPointer
    preset_payload: dict[str, Any]
    model_path: Optional[Path]


def _serialize_pointer(pointer: ChampionPresetPointer) -> dict[str, Any]:
    return {
        "preset_name": pointer.preset_name,
        "bundle_path": str(pointer.bundle_path),
        "created_at": pointer.created_at.astimezone(timezone.utc).isoformat(),
        "context": pointer.context,
        "model_sha": pointer.model_sha,
        "symbol": pointer.symbol,
    }


def _deserialize_pointer(payload: dict[str, Any]) -> ChampionPresetPointer:
    created_raw = str(payload.get("created_at") or datetime.now(timezone.utc).isoformat())
    created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return ChampionPresetPointer(
        preset_name=str(payload.get("preset_name") or "unknown"),
        bundle_path=Path(str(payload.get("bundle_path") or "")).expanduser(),
        created_at=created_at.astimezone(timezone.utc),
        context=payload.get("context"),
        model_sha=payload.get("model_sha"),
        symbol=(str(payload.get("symbol")).upper() if payload.get("symbol") else None),
    )


def _read_preset_payload(bundle_path: Path) -> dict[str, Any]:
    if not bundle_path.exists():
        return {}
    suffix = bundle_path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            with bundle_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
                return loaded if isinstance(loaded, dict) else {}
        if suffix == ".json":
            with bundle_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}
    return {}


def _infer_model_path(bundle_path: Path, payload: dict[str, Any]) -> Optional[Path]:
    candidates = [
        payload.get("model_path"),
        payload.get("model"),
        payload.get("artifact_path"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate)).expanduser()
        if not path.is_absolute():
            path = (bundle_path.parent / path).resolve()
        return path
    return None


def write_champion_pointer(pointer: ChampionPresetPointer) -> None:
    _POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    _POINTER_PATH.write_text(json.dumps(_serialize_pointer(pointer), indent=2), encoding="utf-8")


def load_champion_preset() -> ChampionArtifacts:
    if _POINTER_PATH.exists():
        payload = json.loads(_POINTER_PATH.read_text(encoding="utf-8"))
        pointer = _deserialize_pointer(payload if isinstance(payload, dict) else {})
    else:
        pointer = ChampionPresetPointer(
            preset_name="unconfigured",
            bundle_path=Path(""),
            created_at=datetime.now(timezone.utc),
            context="default_fallback",
            model_sha=None,
            symbol=None,
        )
    preset_payload = _read_preset_payload(pointer.bundle_path) if str(pointer.bundle_path) else {}
    model_path = _infer_model_path(pointer.bundle_path, preset_payload) if str(pointer.bundle_path) else None
    return ChampionArtifacts(pointer=pointer, preset_payload=preset_payload, model_path=model_path)

