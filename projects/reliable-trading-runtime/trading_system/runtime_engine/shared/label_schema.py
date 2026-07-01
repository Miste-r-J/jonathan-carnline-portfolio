from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LabelSchema:
    domain: str
    horizon_bars: int
    trend_ma_window: int
    trend_slope_window: int
    drop_flats: bool
    positive_label: int
    negative_label: int
    params: dict | None = None

    def model_dump(self) -> dict:
        return asdict(self)

    def json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.model_dump(), indent=indent)

    @classmethod
    def model_validate_json(cls, payload: str) -> "LabelSchema":
        import json

        data = json.loads(payload)
        return cls(**data)


def load_label_schema(path: str | Path) -> Optional[LabelSchema]:
    resolved = Path(path)
    if not resolved.exists():
        return None
    try:
        return LabelSchema.model_validate_json(resolved.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_label_schema_for_model(model_path: str | Path) -> Optional[LabelSchema]:
    candidate = Path(model_path).with_suffix(".label_schema.json")
    return load_label_schema(candidate)


__all__ = ["LabelSchema", "load_label_schema", "load_label_schema_for_model"]
