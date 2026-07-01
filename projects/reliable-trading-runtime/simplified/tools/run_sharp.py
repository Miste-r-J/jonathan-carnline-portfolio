from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import time
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from na.bot.features import build_features
from na.bot.config import instrument_by_alias
from na.bot.phase2_sim import Phase2DecisionPolicy, Phase2SimConfig, phase2_decisions, simulate_trades
from na.bot.phase2_v2_runtime import (
    augment_feature_frame_with_train_v2,
    is_train_v2_bundle,
    predict_train_v2_bundle_proba,
)


def _load_manifest(tag: Optional[str], manifest_path: Optional[str]) -> Dict[str, Any]:
    if manifest_path:
        path = Path(manifest_path)
    elif tag:
        path = ROOT / "artifacts" / "phase2" / "candidates" / tag / "manifest.json"
    else:
        raise SystemExit("Provide either --tag or --manifest")
    if not path.exists():
        raise SystemExit(f"Manifest not found at {path}")
    data = json.loads(path.read_text())
    data.setdefault("tag", tag or data.get("tag", path.parent.name))
    data["_manifest_dir"] = str(path.parent.resolve())
    return data


def _manifest_path(manifest: Dict[str, Any], key: str) -> Path:
    raw = manifest.get(key)
    if not raw:
        raise SystemExit(f"Manifest is missing {key}")
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    manifest_dir = Path(str(manifest.get("_manifest_dir") or "."))
    candidate = (manifest_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (ROOT / path).resolve()


def _manifest_optional_path(manifest: Dict[str, Any], key: str) -> Optional[Path]:
    raw = manifest.get(key)
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    manifest_dir = Path(str(manifest.get("_manifest_dir") or "."))
    candidate = (manifest_dir / path).resolve()
    if candidate.exists():
        return candidate
    candidate = (ROOT.parent / path).resolve()
    if candidate.exists():
        return candidate
    candidate = (ROOT / path).resolve()
    if candidate.exists():
        return candidate
    return candidate


def _load_model(path: Path) -> tuple[Any, List[str], Dict[str, Any]]:
    model = joblib.load(path)
    feat_path = path.with_suffix(".features.json")
    features: List[str] = []
    if feat_path.exists():
        try:
            blob = json.loads(feat_path.read_text())
            if isinstance(blob, dict):
                features = list(blob.get("features") or [])
            elif isinstance(blob, list):
                features = [str(item) for item in blob]
        except Exception:
            features = []
    meta_path = path.with_suffix(".meta.json")
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        model_features = [str(item) for item in model_features]
        if model_features:
            # Calibrated/filtered estimators can retain a broad training
            # sidecar while the fitted estimator accepts only the selected
            # feature subset. The fitted contract is authoritative.
            features = model_features
    return model, features, meta


def _select_long_prob(model: Any, raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        return np.clip(arr, 0.0, 1.0)
    classes = getattr(model, "classes_", None)
    if classes is not None:
        try:
            idx = list(classes).index(1)
            return np.clip(arr[:, idx], 0.0, 1.0)
        except Exception:
            pass
        try:
            idx = list(classes).index("LONG")
            return np.clip(arr[:, idx], 0.0, 1.0)
        except Exception:
            pass
    return np.clip(arr[:, -1], 0.0, 1.0)


def _predict_proba(model: Any, frame: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    X = frame.select_dtypes(include=[np.number, "bool"]).astype(float)
    if feature_names:
        for feat in feature_names:
            if feat not in X.columns:
                X[feat] = 0.0
        X = X[feature_names]
    if is_train_v2_bundle(model):
        return predict_train_v2_bundle_proba(model, X)
    raw = model.predict_proba(X)
    return _select_long_prob(model, raw)


def _parse_time(value: str) -> time:
    hh, mm = value.split(":")
    return time(hour=int(hh), minute=int(mm))


def _filter_time_window(df: pd.DataFrame, *, start_at: Any, end_at: Any, tz: str) -> pd.DataFrame:
    if start_at is None and end_at is None:
        return df
    dt = pd.to_datetime(df["Datetime"], errors="coerce", utc=True).dt.tz_convert(tz)
    mask = pd.Series(True, index=df.index)
    if start_at is not None:
        start = pd.to_datetime(start_at, errors="coerce")
        if start.tzinfo is None:
            start = start.tz_localize(tz)
        else:
            start = start.tz_convert(tz)
        mask &= dt >= start
    if end_at is not None:
        end = pd.to_datetime(end_at, errors="coerce")
        if end.tzinfo is None:
            end = end.tz_localize(tz)
            if not any(ch.isdigit() and idx > 9 for idx, ch in enumerate(str(end_at))):
                end = end + pd.Timedelta(days=1)
                mask &= dt < end
                return df.loc[mask].reset_index(drop=True)
        else:
            end = end.tz_convert(tz)
        mask &= dt <= end
    return df.loc[mask].reset_index(drop=True)


def _summarize_phase2(df: pd.DataFrame) -> Dict[str, Any]:
    setup_probs = df["phase2_setup_prob"]
    raw = df["dir_prob_raw"]
    effective = df["dir_prob_effective"]
    suppressed = np.logical_and(np.isclose(effective, 0.5), ~np.isclose(raw, 0.5))
    reasons = (
        df["phase2_reason"]
        .fillna("")
        .value_counts()
        .head(10)
        .to_dict()
    )
    return {
        "setup_prob_mean": float(np.nanmean(setup_probs)) if len(setup_probs) else float("nan"),
        "setup_prob_std": float(np.nanstd(setup_probs)) if len(setup_probs) else float("nan"),
        "suppressed_fraction": float(suppressed.mean()) if len(suppressed) else 0.0,
        "reason_counts": reasons,
    }


def run_candidate(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = _load_manifest(args.tag, args.manifest)
    csv_path = args.csv or manifest.get("csv")
    if not csv_path:
        raise SystemExit("CSV path must be provided via --csv or manifest field.")
    if not args.csv:
        resolved_csv = _manifest_optional_path(manifest, "csv")
        if resolved_csv is not None:
            csv_path = str(resolved_csv)

    cfg = manifest.get("config") or {}
    tz = cfg.get("tz", "America/Denver")
    rth_start = cfg.get("rth_start", "07:30")
    rth_end = cfg.get("rth_end", "14:00")
    orb = cfg.get("orb_minutes", 15)

    df_raw = pd.read_csv(csv_path)
    feats = build_features(
        df_raw,
        tz=tz,
        rth_start=rth_start,
        rth_end=rth_end,
        orb_minutes=orb,
        csv_naive_is_utc=cfg.get("csv_naive_is_utc", False),
    )
    feats = feats.reset_index(drop=True)
    feats = _filter_time_window(
        feats,
        start_at=getattr(args, "start_at", None),
        end_at=getattr(args, "end_at", None),
        tz=tz,
    )
    setup_path = _manifest_path(manifest, "setup_model_path")
    dir_path = _manifest_path(manifest, "dir_model_path")
    setup_model, setup_feats, setup_meta = _load_model(setup_path)
    dir_model, dir_feats, dir_meta = _load_model(dir_path)
    if is_train_v2_bundle(dir_model, dir_meta):
        feats = augment_feature_frame_with_train_v2(feats, df_raw, tz=tz)
    if feats.empty:
        raise SystemExit("Filtered evaluation frame is empty; check --start-at/--end-at.")
    setup_probs = _predict_proba(setup_model, feats, setup_feats)
    requires_stack = bool(dir_meta.get("stack_setup_prob"))
    feats_for_dir = feats.copy()
    if requires_stack or ("stack_setup_prob" in dir_feats):
        feats_for_dir["stack_setup_prob"] = setup_probs
    dir_probs = _predict_proba(dir_model, feats_for_dir, dir_feats)

    thresholds = dict(manifest.get("thresholds") or {})
    if getattr(args, "p_setup", None) is not None:
        thresholds["p_setup"] = float(args.p_setup)
    if getattr(args, "p_long", None) is not None:
        thresholds["p_long"] = float(args.p_long)
    if getattr(args, "p_short", None) is not None:
        thresholds["p_short"] = float(args.p_short)

    close_path = None
    close_cfg = manifest.get("close") or {}
    close_rel = manifest.get("close_model_path") or (close_cfg.get("model_path") if isinstance(close_cfg, dict) else None)
    if close_rel:
        close_path = _manifest_path({**manifest, "close_model_path": close_rel}, "close_model_path")
    print(f"[INFO] Loaded setup_model={setup_path}, dir_model={dir_path}, close_model={close_path or 'n/a'}")
    print(f"[INFO] Using thresholds p_setup={thresholds.get('p_setup')}, p_long={thresholds.get('p_long')}, p_short={thresholds.get('p_short')}")

    decision_policy = Phase2DecisionPolicy.from_mapping((cfg.get("decision_policy") or {}) if isinstance(cfg, dict) else {})
    df_phase2 = phase2_decisions(feats.copy(), setup_probs, dir_probs, thresholds, policy=decision_policy)

    instrument = instrument_by_alias(args.instrument)
    point_value = instrument.point_value
    tick_value = instrument.tick_value
    commission = args.commission_per_contract if getattr(args, "commission_per_contract", None) is not None else 2.0
    slippage_ticks = args.slippage_ticks if getattr(args, "slippage_ticks", None) is not None else 1.0
    sim_cfg = Phase2SimConfig(
        tz=tz,
        trade_window_start=args.trade_window_start or cfg.get("trade_window_start", "07:30"),
        trade_window_end=args.trade_window_end or cfg.get("trade_window_end", "12:00"),
        point_value=point_value,
        tick_value=tick_value,
        contracts=args.contracts,
        max_hold_bars=args.max_hold_bars,
        commission_per_contract=commission,
        slippage_ticks=slippage_ticks,
    )
    sim = simulate_trades(df_phase2, thresholds, cfg=sim_cfg)
    phase2_summary = _summarize_phase2(df_phase2)

    out_dir = Path(args.out_dir).expanduser() / manifest["tag"]
    if not getattr(args, "skip_store", False):
        out_dir.mkdir(parents=True, exist_ok=True)
        df_phase2.to_csv(out_dir / "phase2_eval.csv", index=False)
        (out_dir / "sharp_summary.json").write_text(json.dumps({"sim": sim, "phase2": phase2_summary}, indent=2))
    return {
        "tag": manifest["tag"],
        "setup_model": str(setup_path),
        "dir_model": str(dir_path),
        "close_model": str(close_path) if close_path is not None else None,
        "thresholds": thresholds,
        "sim": sim,
        "phase2": phase2_summary,
        "out_dir": str(out_dir),
    }


def _parse_main_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run lightweight SHARP evaluation for a Phase-2 candidate.")
    ap.add_argument("--tag", help="Candidate tag to load under artifacts/phase2/candidates/<tag>.")
    ap.add_argument("--manifest", help="Explicit manifest path (overrides --tag).")
    ap.add_argument("--csv", help="CSV path (defaults to manifest csv field).")
    ap.add_argument("--instrument", default="MES", help="Instrument alias for sizing (default MES).")
    ap.add_argument("--contracts", type=int, default=1, help="Contracts for simulation (default 1).")
    ap.add_argument("--trade-window-start", default=None, help="Override trade window start (HH:MM).")
    ap.add_argument("--trade-window-end", default=None, help="Override trade window end (HH:MM).")
    ap.add_argument("--max-hold-bars", type=int, default=24, help="Max bars to hold a position before flattening.")
    ap.add_argument("--out-dir", default="runs/sharp", help="Base directory for SHARP outputs.")
    ap.add_argument("--p_setup", type=float, default=None, help="Override Phase-2 setup threshold.")
    ap.add_argument("--p_long", type=float, default=None, help="Override direction long threshold.")
    ap.add_argument("--p_short", type=float, default=None, help="Override direction short threshold.")
    ap.add_argument("--start-at", dest="start_at", default=None, help="Evaluate at/after this session timestamp/date.")
    ap.add_argument("--end-at", dest="end_at", default=None, help="Evaluate at/before this session timestamp/date.")
    ap.add_argument("--commission_per_contract", type=float, default=None, help="Commission USD per contract per side.")
    ap.add_argument("--slippage_ticks", type=float, default=None, help="Slippage in ticks per side.")
    ap.add_argument("--skip_store", action="store_true", help="Skip writing eval CSV/summary to disk.")
    return ap.parse_args()


def main() -> None:
    args = _parse_main_args()
    result = run_candidate(args)
    summary = {
        "tag": result["tag"],
        "pnl_usd": result["sim"]["total_pnl_usd"],
        "max_dd": result["sim"]["max_drawdown"],
        "sharpe": result["sim"]["sharpe"],
        "trades": len(result["sim"]["trades"]),
        "win_rate": result["sim"]["win_rate"],
        "flip_rate_per_day": result["sim"]["flip_rate_per_day"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
