from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from tools.run_sharp import _load_model


class _ModelWithFeatureContract:
    feature_names_in_ = np.asarray(["selected_a", "selected_b"])


def test_load_model_prefers_fitted_feature_contract(tmp_path: Path) -> None:
    model_path = tmp_path / "setup.joblib"
    joblib.dump(_ModelWithFeatureContract(), model_path)
    model_path.with_suffix(".features.json").write_text(
        json.dumps({"features": ["broad_a", "broad_b", "broad_c"]}),
        encoding="utf-8",
    )

    _, features, _ = _load_model(model_path)

    assert features == ["selected_a", "selected_b"]
