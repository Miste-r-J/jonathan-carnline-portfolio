from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def compare_prediction_bundles(path_a: Path, path_b: Path) -> List[Tuple[int, str, str]]:
    """Return list of (idx, hash_a, hash_b) mismatches."""
    a = _read_jsonl(path_a)
    b = _read_jsonl(path_b)
    mismatches: List[Tuple[int, str, str]] = []
    limit = min(len(a), len(b))
    for i in range(limit):
        ha = str(a[i].get("deterministic_hash") or "")
        hb = str(b[i].get("deterministic_hash") or "")
        if ha != hb:
            mismatches.append((i, ha, hb))
        ka = str(a[i].get("prediction_key") or "")
        kb = str(b[i].get("prediction_key") or "")
        if ka != kb:
            mismatches.append((i, ka, kb))
    if len(a) != len(b):
        mismatches.append((limit, f"len={len(a)}", f"len={len(b)}"))
    return mismatches
