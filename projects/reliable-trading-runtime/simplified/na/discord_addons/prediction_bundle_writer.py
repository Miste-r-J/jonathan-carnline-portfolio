from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class PredictionBundleWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, bundle: Mapping[str, Any]) -> None:
        line = json.dumps(bundle, ensure_ascii=True, sort_keys=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
