from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from na.bot.phase2_v2_runtime import (
    build_train_v2_feature_frame,
    classify_train_v2_warmup_issue,
    infer_train_v2_history_requirements,
    is_train_v2_bundle,
)
from na.discord_addons.cli.stream_live_csv import (
    _apply_phase2_manifest_overrides,
    _load_phase2_manifest,
    _predict_proba_safely,
    load_model_features_and_calibrator,
)


ROOT = Path(__file__).resolve().parents[2]
V2_MODEL = ROOT / "models" / "v2_candidate" / "direction_v2_5029b7200c16.joblib"
V2_TAG = "retrain_v2_bundle_5029b7200c16"


def _synthetic_ohlcv(rows: int = 500, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05 06:30", periods=rows, freq="5min")
    close = 5900 + np.cumsum(rng.normal(0, 0.8, size=len(idx)))
    open_ = close + rng.normal(0, 0.2, size=len(idx))
    high = np.maximum(open_, close) + np.abs(rng.normal(0.4, 0.15, size=len(idx)))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.4, 0.15, size=len(idx)))
    volume = rng.integers(50, 400, size=len(idx))
    return pd.DataFrame(
        {
            "Datetime": idx,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def test_v2_bundle_loader_and_predict_path() -> None:
    model, features, calibrator, meta = load_model_features_and_calibrator(str(V2_MODEL))

    assert is_train_v2_bundle(model, meta)
    assert calibrator is None
    assert meta is not None
    assert meta.get("model") == "direction_v2_ensemble"
    assert features is not None
    assert len(features) == 88

    X = pd.DataFrame(0.0, index=range(4), columns=features)
    proba = _predict_proba_safely(model, X, meta=meta, model_path=V2_MODEL)

    assert proba.shape == (4,)
    assert np.all(np.isfinite(proba))
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_build_train_v2_feature_frame_exposes_mtf_columns() -> None:
    feats = build_train_v2_feature_frame(_synthetic_ohlcv(), tz="America/Denver")

    assert "Datetime" in feats.columns
    assert "ret_1" in feats.columns
    assert "ret_15m_3_tf15" in feats.columns
    assert "sma_day_5_day" in feats.columns
    assert len(feats) > 100


def test_train_v2_truncated_history_classifies_daily_nan_as_warmup() -> None:
    feature_path = (
        ROOT / "artifacts" / "phase2" / "candidates" / "retrain_v2_776a77a63611" / "dir.features.json"
    )
    expected = json.loads(feature_path.read_text(encoding="utf-8"))["features"]
    raw = _synthetic_ohlcv(rows=3876)
    feats = build_train_v2_feature_frame(raw, tz="America/Denver")
    last = feats.tail(1)

    bad = []
    for name in expected:
        val = pd.to_numeric(last[name], errors="coerce").iloc[0]
        if not np.isfinite(val):
            bad.append(name)

    assert bad == ["sma_day_20_day", "dist_sma_day_20_day"]
    requirements = infer_train_v2_history_requirements(expected)
    assert requirements["required_unique_days"] == 20
    warmup = classify_train_v2_warmup_issue(
        bad,
        available_bars=len(raw),
        unique_days=int(pd.to_datetime(raw["Datetime"]).dt.normalize().nunique()),
        feature_names=expected,
    )
    assert warmup is not None
    assert warmup["warmup_features"] == bad


def test_train_v2_sufficient_history_keeps_latest_direction_row_finite() -> None:
    feature_path = (
        ROOT / "artifacts" / "phase2" / "candidates" / "retrain_v2_776a77a63611" / "dir.features.json"
    )
    expected = json.loads(feature_path.read_text(encoding="utf-8"))["features"]
    raw = _synthetic_ohlcv(rows=6000)
    feats = build_train_v2_feature_frame(raw, tz="America/Denver")
    last = feats.tail(1)

    bad = []
    for name in expected:
        val = pd.to_numeric(last[name], errors="coerce").iloc[0]
        if not np.isfinite(val):
            bad.append(name)

    assert bad == []


def test_manifest_override_for_v2_candidate_tag() -> None:
    manifest_path, manifest = _load_phase2_manifest(V2_TAG)
    args = argparse.Namespace(
        disable_safety_gates=False,
        phase2=False,
        phase2_tag=V2_TAG,
        phase2_use_manifest_thresholds=True,
        _preset_fields=set(),
        p_setup=None,
        p_long=None,
        p_short=None,
        setup_model_path=None,
        dir_model_path=None,
        close_model_path=None,
        phase2_close_enabled=None,
        phase2_close_threshold=None,
        model=None,
        entry_trend_filter=None,
        min_signal_persistence_bars=None,
        cooldown_bars_after_flip=None,
        session_tz=None,
        trade_window_start=None,
        trade_window_end=None,
        rth_start=None,
        rth_end=None,
    )

    args = _apply_phase2_manifest_overrides(args, manifest_path, manifest, argv_tokens=[])

    assert args.phase2 is True
    assert str(Path(args.dir_model_path).name) == "dir.joblib"
    assert args.p_long == 0.72
    assert args.p_short == 0.72
    assert args.p_setup == 0.35
