from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    # `simplified/na/bot/paths.py` -> `simplified/na/bot` -> `simplified/na` -> `simplified` -> repo root
    return Path(__file__).resolve().parents[3]


REPO_ROOT: Path = _repo_root()
SIMPLIFIED_DIR: Path = REPO_ROOT / "simplified"
ARTIFACTS_DIR: Path = SIMPLIFIED_DIR / "artifacts"
PHASE2_CANDIDATES_DIR: Path = ARTIFACTS_DIR / "phase2" / "candidates"

