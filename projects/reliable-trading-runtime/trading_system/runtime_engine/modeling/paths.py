from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    # `trading_system/runtime_engine/modeling/paths.py` -> project root
    return Path(__file__).resolve().parents[3]


REPO_ROOT: Path = _repo_root()
TRADING_SYSTEM_DIR: Path = REPO_ROOT / "trading_system"
ARTIFACTS_DIR: Path = TRADING_SYSTEM_DIR / "artifacts"
PHASE2_CANDIDATES_DIR: Path = ARTIFACTS_DIR / "phase2" / "candidates"
