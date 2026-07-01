from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pandas.errors import PerformanceWarning
except Exception:  # pragma: no cover - pandas version dependent
    PerformanceWarning = RuntimeWarning


def _synthetic_ohlcv(rows: int = 500, *, seed: int = 0) -> pd.DataFrame:
    from trading_system.runtime_engine.modeling.config import CLOSE_COL, HIGH_COL, LOW_COL, OPEN_COL, VOLUME_COL

    rng = np.random.default_rng(seed)
    # RTH-local timestamps (America/Denver). Tests call build_features with csv_naive_is_utc=False.
    idx = pd.date_range("2025-01-02 07:30", periods=rows, freq="1min")
    return pd.DataFrame(
        {
            OPEN_COL: 5900 + rng.standard_normal(len(idx)),
            HIGH_COL: 5901 + np.abs(rng.standard_normal(len(idx))),
            LOW_COL: 5899 - np.abs(rng.standard_normal(len(idx))),
            CLOSE_COL: 5900 + rng.standard_normal(len(idx)),
            VOLUME_COL: rng.integers(100, 1000, len(idx)).astype(float),
        },
        index=idx,
    )


def test_no_performance_warning() -> None:
    """build_features() must not emit pandas PerformanceWarning."""
    import warnings

    from trading_system.runtime_engine.modeling.features import build_features

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_features(_synthetic_ohlcv(), strategy_config={}, csv_naive_is_utc=False)

    perf_warnings = [
        w
        for w in caught
        if issubclass(w.category, (PerformanceWarning,))
        or "highly fragmented" in str(w.message).lower()
        or "DataFrame is highly fragmented" in str(w.message)
    ]
    assert perf_warnings == [], (
        f"build_features() emitted {len(perf_warnings)} PerformanceWarning(s):\n"
        + "\n".join(str(w.message) for w in perf_warnings)
    )


def test_build_features_latency() -> None:
    """build_features() on 500 bars must complete in under 5 seconds."""
    import time

    from trading_system.runtime_engine.modeling.features import build_features

    df = _synthetic_ohlcv(rows=500)
    start = time.perf_counter()
    build_features(df, strategy_config={}, csv_naive_is_utc=False)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, (
        f"build_features() took {elapsed:.2f}s on 500 bars — "
        "exceeds 5s budget. Check for fragmentation or O(n²) loops."
    )


def test_mandatory_feature_list_is_201() -> None:
    from trading_system.runtime_engine.modeling.feature_constants import MANDATORY_MODEL_FEATURES

    assert len(MANDATORY_MODEL_FEATURES) == 201
    assert len(set(MANDATORY_MODEL_FEATURES)) == 201


def test_build_features_produces_full_schema_no_duplicates() -> None:
    from trading_system.runtime_engine.modeling.feature_constants import MANDATORY_MODEL_FEATURES
    from trading_system.runtime_engine.modeling.features import build_features

    out = build_features(_synthetic_ohlcv(), strategy_config={}, csv_naive_is_utc=False)
    dupes = out.columns[out.columns.duplicated()].tolist()
    assert not dupes, f"Duplicate columns: {dupes}"
    missing = [f for f in MANDATORY_MODEL_FEATURES if f not in out.columns]
    assert not missing, f"Missing {len(missing)} features: {missing}"


def test_known_alias_and_sign_invariants() -> None:
    from trading_system.runtime_engine.modeling.features import build_features

    out = build_features(_synthetic_ohlcv(), strategy_config={}, csv_naive_is_utc=False)

    sample = out[
        [
            "Datetime",
            "is_inside_orb",
            "inside_orb_flag",
            "dist_to_orb_low_atr",
            "dist_orb_low_atr",
        ]
    ].head(10)
    print("\n[alias/sign sample]\n", sample.to_string(index=False))

    assert (out["inside_orb_flag"] == out["is_inside_orb"]).all()
    assert (out["dist_orb_high_atr"] == out["dist_to_orb_high_atr"]).all()
    assert (out["dist_orb_low_atr"] == -out["dist_to_orb_low_atr"]).all()


def test_feature_hash_matches_manifest() -> None:
    from trading_system.runtime_engine.modeling.feature_hash import compute_feature_hash
    from trading_system.runtime_engine.modeling.feature_constants import MANDATORY_MODEL_FEATURES

    manifest_path = Path("trading_system/artifacts/phase2/candidates/retrain_v2_full/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["feature_hash"]
    assert compute_feature_hash(MANDATORY_MODEL_FEATURES) == expected


def test_known_training_config_issues_documented() -> None:
    """Ensures VAL_SPLIT_MISMATCH is documented and not silently dropped."""
    from trading_system.runtime_engine.modeling.feature_constants import KNOWN_TRAINING_CONFIG_ISSUES

    ids = [i["id"] for i in KNOWN_TRAINING_CONFIG_ISSUES]
    assert "VAL_SPLIT_MISMATCH" in ids, (
        "VAL_SPLIT_MISMATCH must remain in KNOWN_TRAINING_CONFIG_ISSUES until "
        "a retrain with aligned val splits is completed and verified."
    )
    issue = next(i for i in KNOWN_TRAINING_CONFIG_ISSUES if i["id"] == "VAL_SPLIT_MISMATCH")
    assert issue["severity"] == "high"
    assert "direction" in issue["affected_roles"]


def test_no_duplicate_columns() -> None:
    from trading_system.runtime_engine.modeling.features import build_features

    out = build_features(_synthetic_ohlcv(), strategy_config={}, csv_naive_is_utc=False)
    dupes = out.columns[out.columns.duplicated()].tolist()
    assert dupes == [], f"Duplicate columns: {dupes}"


def test_mandatory_features_all_present() -> None:
    from trading_system.runtime_engine.modeling.feature_constants import MANDATORY_MODEL_FEATURES
    from trading_system.runtime_engine.modeling.features import build_features

    out = build_features(_synthetic_ohlcv(), strategy_config={}, csv_naive_is_utc=False)
    missing = [f for f in MANDATORY_MODEL_FEATURES if f not in out.columns]
    assert missing == [], f"Missing {len(missing)} features: {missing}"


def test_feature_count_exactly_201() -> None:
    from trading_system.runtime_engine.modeling.feature_constants import MANDATORY_MODEL_FEATURES
    from trading_system.runtime_engine.modeling.features import build_features

    out = build_features(_synthetic_ohlcv(), strategy_config={}, csv_naive_is_utc=False)
    required = set(MANDATORY_MODEL_FEATURES)
    model_cols = [c for c in out.columns if c in required]
    assert len(model_cols) == 201, f"Expected 201 model features, got {len(model_cols)}"


def test_ret_1_not_overwritten_by_flow() -> None:
    from trading_system.runtime_engine.modeling.config import CLOSE_COL
    from trading_system.runtime_engine.modeling.features import build_features
    from trading_system.runtime_engine.modeling.features_flow_ohlcv import add_flow_ohlcv_features

    out = build_features(_synthetic_ohlcv(), strategy_config={}, csv_naive_is_utc=False)
    before = out["ret_1"].copy()
    after = add_flow_ohlcv_features(out.copy())["ret_1"]
    corr = before.corr(after)
    assert corr > 0.9999, f"ret_1 changed after flow feature computation: corr={corr}"
    direct = pd.to_numeric(out[CLOSE_COL], errors="coerce").pct_change()
    corr_direct = before.corr(direct)
    assert corr_direct > 0.9999, f"ret_1 deviates from close pct_change(1): corr={corr_direct}"


def _synthetic_ohlcv_below_orb_low(rows: int = 220) -> pd.DataFrame:
    from trading_system.runtime_engine.modeling.config import CLOSE_COL, HIGH_COL, LOW_COL, OPEN_COL, VOLUME_COL

    idx = pd.date_range("2025-01-02 07:30", periods=rows, freq="1min")
    base = pd.DataFrame(index=idx, columns=[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL], dtype=float)
    # ORB window (first 15 bars): stable band.
    base.iloc[:15, base.columns.get_loc(OPEN_COL)] = 100.0
    base.iloc[:15, base.columns.get_loc(HIGH_COL)] = 101.0
    base.iloc[:15, base.columns.get_loc(LOW_COL)] = 99.0
    base.iloc[:15, base.columns.get_loc(CLOSE_COL)] = 100.0
    base.iloc[:15, base.columns.get_loc(VOLUME_COL)] = 500.0
    # After ORB: mostly flat above orb low.
    base.iloc[15:120, base.columns.get_loc(OPEN_COL)] = 100.0
    base.iloc[15:120, base.columns.get_loc(HIGH_COL)] = 100.5
    base.iloc[15:120, base.columns.get_loc(LOW_COL)] = 99.5
    base.iloc[15:120, base.columns.get_loc(CLOSE_COL)] = 100.0
    base.iloc[15:120, base.columns.get_loc(VOLUME_COL)] = 500.0
    # Inject a block clearly below ORB low (orb_low ~= 99.0).
    base.iloc[120:150, base.columns.get_loc(OPEN_COL)] = 98.0
    base.iloc[120:150, base.columns.get_loc(HIGH_COL)] = 98.5
    base.iloc[120:150, base.columns.get_loc(LOW_COL)] = 97.5
    base.iloc[120:150, base.columns.get_loc(CLOSE_COL)] = 98.0
    base.iloc[120:150, base.columns.get_loc(VOLUME_COL)] = 600.0
    # Fill remaining bars.
    base.iloc[150:, base.columns.get_loc(OPEN_COL)] = 100.0
    base.iloc[150:, base.columns.get_loc(HIGH_COL)] = 100.5
    base.iloc[150:, base.columns.get_loc(LOW_COL)] = 99.5
    base.iloc[150:, base.columns.get_loc(CLOSE_COL)] = 100.0
    base.iloc[150:, base.columns.get_loc(VOLUME_COL)] = 500.0
    return base


def test_dist_orb_low_atr_sign() -> None:
    """When close < orb_low, dist_to_orb_low_atr must be negative."""
    from trading_system.runtime_engine.modeling.config import CLOSE_COL
    from trading_system.runtime_engine.modeling.features import build_features

    out = build_features(_synthetic_ohlcv_below_orb_low(), strategy_config={}, csv_naive_is_utc=False)
    below_mask = pd.to_numeric(out[CLOSE_COL], errors="coerce") < pd.to_numeric(out["orb15_low"], errors="coerce")
    if below_mask.any():
        vals = pd.to_numeric(out.loc[below_mask, "dist_to_orb_low_atr"], errors="coerce")
        print("\n[dist_to_orb_low_atr below ORB low]\n", vals.head(10).to_string(index=False))
        assert (vals <= 0).all(), f"dist_to_orb_low_atr positive when below ORB low: {vals.head(10).tolist()}"


def test_config_validation_raises_on_flow_disabled() -> None:
    from trading_system.runtime_engine.modeling.exceptions import ConfigurationError
    from trading_system.runtime_engine.modeling.feature_constants import validate_runtime_config_vs_model
    import pytest

    with pytest.raises(ConfigurationError):
        validate_runtime_config_vs_model({"flow_ohlcv": {"enabled": False}})


def test_manifest_paths_are_relative() -> None:
    import json
    from pathlib import Path

    manifest = json.loads(
        Path("trading_system/artifacts/phase2/candidates/retrain_v2_full/manifest.json").read_text(encoding="utf-8")
    )
    for key in ("setup_model_path", "dir_model_path", "artifact_dir", "csv"):
        val = str(manifest.get(key, "") or "")
        assert not val.startswith("C:\\"), f"{key} is still an absolute path: {val}"
        assert not val.startswith("/"), f"{key} is an absolute path: {val}"


def test_inside_orb_alias_consistent() -> None:
    from trading_system.runtime_engine.modeling.features import build_features

    out = build_features(_synthetic_ohlcv(), strategy_config={}, csv_naive_is_utc=False)
    assert out["inside_orb_flag"].equals(out["is_inside_orb"]), "inside_orb_flag and is_inside_orb have diverged"


def test_orb_range_units_match() -> None:
    """On RTH rows, orb_* must equal orb15_*, and orb_range/orb15_rng units must match."""
    from trading_system.runtime_engine.modeling.features import build_features

    out = build_features(_synthetic_ohlcv(), strategy_config={}, csv_naive_is_utc=False)
    rth = pd.to_numeric(out["is_rth"], errors="coerce").fillna(0).astype(bool)
    if rth.any():
        for a, b in [
            ("orb_high", "orb15_high"),
            ("orb_low", "orb15_low"),
            ("orb_mid", "orb15_mid"),
        ]:
            diff = (pd.to_numeric(out.loc[rth, a], errors="coerce") - pd.to_numeric(out.loc[rth, b], errors="coerce")).abs().max()
            diff = float(diff) if diff is not None else 0.0
            assert diff < 1e-6, f"{a} vs {b} diverge on RTH rows: max_diff={diff}"
        hi = pd.to_numeric(out.loc[rth, "orb_high"], errors="coerce")
        lo = pd.to_numeric(out.loc[rth, "orb_low"], errors="coerce")
        mid = pd.to_numeric(out.loc[rth, "orb_mid"], errors="coerce")
        orb_range_raw = hi - lo
        declared = pd.to_numeric(out.loc[rth, "orb_range"], errors="coerce")
        diff_raw = float((orb_range_raw - declared).abs().max())
        assert diff_raw < 1e-6, f"orb_range (raw) vs (orb_high-orb_low) diverge on RTH rows: max_diff={diff_raw}"
        derived = orb_range_raw / (mid + 1e-12)
        rng = pd.to_numeric(out.loc[rth, "orb15_rng"], errors="coerce")
        diff = float((derived - rng).abs().max())
        assert diff < 1e-6, f"orb_range/mid vs orb15_rng diverge on RTH rows: max_diff={diff}"
