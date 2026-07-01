from __future__ import annotations

import json
from pathlib import Path

import pytest

from na.discord_addons.cli.stream_live_csv import _validate_phase2_label_schema


def _write_schema(tmp_path: Path, name: str, mapping: dict[str, str]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"params": {"mapping": mapping}}), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("role", "mapping"),
    [
        ("setup", {"0": "FLAT", "1": "TRADE"}),
        ("direction", {"0": "SHORT", "1": "LONG"}),
        ("close", {"0": "HOLD", "1": "CLOSE"}),
    ],
)
def test_validate_phase2_label_schema_accepts_expected_mappings(
    tmp_path: Path, role: str, mapping: dict[str, str]
) -> None:
    schema_path = _write_schema(tmp_path, f"{role}.label_schema.json", mapping)

    _validate_phase2_label_schema(schema_path, role=role)


@pytest.mark.parametrize(
    "mapping",
    [
        {"0": "CLOSE", "1": "HOLD"},
        {"0": "HOLD"},
    ],
)
def test_validate_phase2_label_schema_rejects_invalid_close_mapping(
    tmp_path: Path, mapping: dict[str, str]
) -> None:
    schema_path = _write_schema(tmp_path, "close.label_schema.json", mapping)

    with pytest.raises(RuntimeError, match=r"Phase-2 close label schema"):
        _validate_phase2_label_schema(schema_path, role="close")
