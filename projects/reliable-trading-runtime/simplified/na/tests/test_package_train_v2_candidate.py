from __future__ import annotations

import json
from pathlib import Path

from tools.package_train_v2_candidate import build_rebundled_candidate


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_rebundled_candidate_copies_local_close_bundle(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    monkeypatch.setattr("tools.package_train_v2_candidate.ROOT", repo_root)
    monkeypatch.setattr("tools.package_train_v2_candidate.ARTIFACT_ROOT", repo_root / "artifacts" / "phase2" / "candidates")

    source_dir = repo_root / "artifacts" / "phase2" / "candidates" / "source_tag"
    close_dir = repo_root / "artifacts" / "phase2" / "candidates" / "close_source"
    for name in (
        "setup.joblib",
        "setup.features.json",
        "setup.label_schema.json",
        "setup.feature_bounds.json",
        "setup.meta.json",
        "setup.metrics.json",
        "setup.registry.json",
        "dir.joblib",
        "dir.features.json",
        "dir.label_schema.json",
        "dir.feature_bounds.json",
        "dir.meta.json",
        "dir.metrics.json",
        "dir.registry.json",
    ):
        _write_text(source_dir / name, name)
    _write_json(source_dir / "setup.features.json", {"features": ["atr_14", "ret_1"]})
    _write_json(source_dir / "dir.features.json", {"features": ["ret_1", "rsi_14"]})
    for name in (
        "close.joblib",
        "close.features.json",
        "close.label_schema.json",
        "close.meta.json",
        "close.metrics.json",
        "close.registry.json",
    ):
        _write_text(close_dir / name, name)
    _write_json(close_dir / "close.features.json", {"features": ["ret_1"]})
    _write_json(
        source_dir / "manifest.json",
        {
            "tag": "source_tag",
            "csv": "data\\intraday\\es\\ES.csv",
            "instrument": "ES",
            "timeframe": "5m",
            "artifact_dir": ".",
            "setup_model_path": "setup.joblib",
            "dir_model_path": "dir.joblib",
            "close_model_path": str(close_dir / "close.joblib"),
            "thresholds": {"p_setup": 0.06, "p_long": 0.57, "p_short": 0.57},
            "close": {"enabled": True, "threshold": 0.9, "model_path": str(close_dir / "close.joblib")},
            "config": {"tz": "America/Denver"},
            "metrics": {"setup": {"ok": True}},
            "promotion_result": "manual",
            "rejected": False,
        },
    )

    class Args:
        source_tag = "source_tag"
        close_source_path = None
        tag = "source_tag_livebundle_v1"
        p_setup = None
        p_long = None
        p_short = None
        promotion_result = "paper_candidate"

    out_dir = build_rebundled_candidate(Args())

    assert (out_dir / "close.joblib").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["close_model_path"] == "close.joblib"
    assert manifest["close"]["model_path"] == "close.joblib"
    assert manifest["thresholds"] == {"p_setup": 0.06, "p_long": 0.57, "p_short": 0.57}
    assert manifest["feature_hash"] == manifest["runtime_feature_hash"]
    assert set(manifest["feature_hashes"]) == {"setup", "direction", "close"}
    assert manifest["close"]["feature_count"] == 1
    for role in ("setup", "dir", "close"):
        registry = json.loads((out_dir / f"{role}.registry.json").read_text(encoding="utf-8"))
        assert Path(registry["model_path"]).parent == out_dir
