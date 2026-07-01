from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from na.bot.features import build_features
from na.bot.models import compute_sha256, load_model, model_features
from na.common.label_schema import load_label_schema_for_model
from na.explain.shap_xgb import (
    ShapExplainerCache,
    align_frame_for_model,
    summarize_mean_abs_shap,
    write_json_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger("na.bot.explain_shap")


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "timestamp": "Datetime",
        "time": "Datetime",
        "datetime": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    columns = {col: rename_map.get(str(col).strip().lower(), col) for col in df.columns}
    df = df.rename(columns=columns)
    if "Datetime" not in df.columns:
        raise KeyError("CSV must contain a Datetime column (or timestamp/time).")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise KeyError(f"CSV missing required OHLCV column '{col}'.")
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    return df


def _prepare_feature_frame(
    csv_path: Path,
    *,
    session_tz: str,
    rth_start: str,
    rth_end: str,
    orb_min: int,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _normalize_ohlcv_columns(df)
    feats = build_features(
        df,
        tz=session_tz,
        rth_start=rth_start,
        rth_end=rth_end,
        orb_minutes=orb_min,
        csv_naive_is_utc=False,
    )
    if feats.empty:
        raise RuntimeError("build_features returned an empty frame; check CSV inputs.")
    if "Datetime" not in feats.columns:
        feats = feats.reset_index().rename(columns={feats.index.name or "index": "Datetime"})
    feats = feats.sort_values("Datetime").reset_index(drop=True)
    return feats


def _limit_rows(df: pd.DataFrame, limit: Optional[int]) -> pd.DataFrame:
    if not limit:
        return df
    if limit <= 0:
        return df
    return df.tail(limit).reset_index(drop=True)


def _sample_background(X: pd.DataFrame, size: int) -> Optional[np.ndarray]:
    if size <= 0 or X.empty:
        return None
    size = min(size, len(X))
    return X.sample(size, random_state=42).to_numpy(copy=True)


def _plot_summary(shap_module: Any, values: np.ndarray, X: pd.DataFrame, path: Path, *, plot_type: str, max_display: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    shap_module.summary_plot(
        values,
        X,
        show=False,
        plot_type=plot_type,
        max_display=max_display,
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_dependence(shap_module: Any, values: np.ndarray, X: pd.DataFrame, feature: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    shap_module.dependence_plot(feature, values, X, show=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def _timestamped_outdir(base: Path, instrument: Optional[str]) -> Path:
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    if instrument:
        base = base / instrument.upper()
    return base / ts


def _load_model_and_features(model_path: Path) -> Tuple[Any, Optional[Sequence[str]]]:
    model = load_model(str(model_path))
    feats = model_features(str(model_path))
    return model, feats


def _schema_class_mapping(model_path: Path) -> Tuple[Optional[Any], Dict[int, str]]:
    schema = load_label_schema_for_model(model_path)
    mapping: Dict[int, str] = {}
    if schema and getattr(schema, "params", None):
        raw_map = schema.params.get("mapping") if isinstance(schema.params, dict) else None
        if isinstance(raw_map, dict):
            for key, value in raw_map.items():
                try:
                    mapping[int(key)] = str(value)
                except Exception:
                    continue
    return schema, mapping


def _resolve_class_labels(
    num_classes: int,
    model_classes: Sequence[Any],
    schema_map: Dict[int, str],
) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    if model_classes:
        for idx in range(min(num_classes, len(model_classes))):
            raw = model_classes[idx]
            name = None
            try:
                key = int(raw)
                name = schema_map.get(key)
            except Exception:
                name = None
            if name is None:
                name = str(raw)
            labels[idx] = name
    if not labels and schema_map:
        for idx, (_key, val) in enumerate(sorted(schema_map.items(), key=lambda item: item[0])):
            labels[idx] = str(val)
            if len(labels) >= num_classes:
                break
    for idx in range(num_classes):
        labels.setdefault(idx, f"class_{idx}")
    return labels


def _positive_class_index(model_classes: Sequence[Any], schema: Optional[Any]) -> Optional[int]:
    if schema is None or not model_classes:
        return None
    try:
        return list(model_classes).index(schema.positive_label)
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP explainability reports for XGBoost models.")
    parser.add_argument("--model", required=True, help="Path to trained model artifact (joblib/pickle).")
    parser.add_argument("--data", required=True, help="CSV containing OHLCV data used for training/scoring.")
    parser.add_argument("--outdir", required=True, help="Base directory to write SHAP reports.")
    parser.add_argument("--instrument", help="Optional instrument code for organizing output directories.")
    parser.add_argument("--n-rows", type=int, default=20000, help="Max rows from the CSV to use for SHAP computation.")
    parser.add_argument("--background", type=int, default=2000, help="Rows to sample for TreeExplainer background.")
    parser.add_argument("--top-n", type=int, default=30, help="Max features to display in summary plots.")
    parser.add_argument("--dependence-k", type=int, default=8, help="Number of top features for dependence plots.")
    parser.add_argument("--session-tz", default="America/Denver", help="Session timezone for feature builder.")
    parser.add_argument("--rth-start", default="07:30", help="RTH start (HH:MM).")
    parser.add_argument("--rth-end", default="14:00", help="RTH end (HH:MM).")
    parser.add_argument("--orb-minutes", type=int, default=30, help="Opening range breakout minutes for features.")
    parser.add_argument("--class-index", type=int, default=None, help="Specific class index to explain.")
    parser.add_argument("--all-classes", action="store_true", help="Generate SHAP outputs for every class.")
    return parser.parse_args()


def main() -> int:
    try:
        import shap  # type: ignore
    except Exception as exc:
        LOGGER.error("SHAP is not installed: %s", exc)
        return 1

    args = _parse_args()
    model_path = Path(args.model).expanduser()
    data_path = Path(args.data).expanduser()
    outdir = Path(args.outdir).expanduser()
    if not model_path.exists():
        LOGGER.error("Model path %s does not exist.", model_path)
        return 1
    if not data_path.exists():
        LOGGER.error("Data path %s does not exist.", data_path)
        return 1

    feats = _prepare_feature_frame(
        data_path,
        session_tz=args.session_tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_min=int(args.orb_minutes),
    )
    feats = _limit_rows(feats, args.n_rows)
    model, feature_names = _load_model_and_features(model_path)
    X_df = feats.set_index("Datetime")
    X_matrix, used_features = align_frame_for_model(X_df, feature_names)
    if not len(used_features):
        LOGGER.error("Unable to align features for the provided model; check the sidecar JSON.")
        return 1
    X_frame = pd.DataFrame(X_matrix, columns=used_features)
    X_frame = X_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    background = _sample_background(X_frame, int(args.background or 0))
    cache = ShapExplainerCache(model, used_features, background=background)
    class_result = cache.explain_matrix_classes(X_frame.to_numpy())
    if class_result is None:
        LOGGER.error("Failed to compute SHAP values. Ensure the model is tree-based and SHAP supports it.")
        return 1
    shap_matrices, base_values = class_result
    num_classes = len(shap_matrices)
    if num_classes == 0:
        LOGGER.error("SHAP returned no class matrices; aborting.")
        return 1
    model_classes = list(getattr(model, "classes_", []))
    schema, schema_map = _schema_class_mapping(model_path)
    class_labels = _resolve_class_labels(num_classes, model_classes, schema_map)
    positive_idx = _positive_class_index(model_classes, schema)
    if args.all_classes:
        target_classes = list(range(num_classes))
    elif args.class_index is not None:
        idx = max(0, min(int(args.class_index), num_classes - 1))
        target_classes = [idx]
    elif num_classes == 2:
        target_classes = [0, 1]
    else:
        target_classes = [0]
    target_classes = sorted(set(target_classes))
    if not target_classes:
        target_classes = [0]

    report_dir = _timestamped_outdir(outdir, args.instrument)
    report_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing SHAP report to %s", report_dir)

    class_reports = []
    for class_index in target_classes:
        shap_values = np.asarray(shap_matrices[class_index])
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(len(X_frame), -1)
        elif shap_values.shape[0] != len(X_frame):
            shap_values = shap_values.reshape(len(X_frame), -1)
        class_label = class_labels.get(class_index, f"class_{class_index}")
        safe_label = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(class_label))
        suffix = f"class{class_index}_{safe_label}"
        beeswarm_path = report_dir / f"summary_beeswarm_{suffix}.png"
        bar_path = report_dir / f"summary_bar_{suffix}.png"
        _plot_summary(shap, shap_values, X_frame, beeswarm_path, plot_type="dot", max_display=args.top_n)
        _plot_summary(shap, shap_values, X_frame, bar_path, plot_type="bar", max_display=args.top_n)

        top_features = summarize_mean_abs_shap(shap_values, used_features)
        dep_limit = min(len(top_features), max(int(args.dependence_k), 0))
        dependence_paths = []
        for name, _ in top_features[:dep_limit]:
            safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
            dep_path = report_dir / f"dependence_{suffix}_{safe_name}.png"
            _plot_dependence(shap, shap_values, X_frame, name, dep_path)
            dependence_paths.append(dep_path.name)

        class_reports.append(
            {
                "class_index": class_index,
                "class_label": class_label,
                "base_value": float(base_values[class_index]) if class_index < len(base_values) and base_values[class_index] is not None else None,
                "top_features": [
                    {"name": name, "mean_abs_shap": value}
                    for name, value in top_features[: args.top_n]
                ],
                "plots": {
                    "summary_beeswarm": beeswarm_path.name,
                    "summary_bar": bar_path.name,
                    "dependence": dependence_paths,
                },
            }
        )

    shap_direction = None
    if positive_idx is not None:
        positive_label = class_labels.get(positive_idx, f"class_{positive_idx}")
        shap_direction = f"Positive SHAP values push the model margin toward {positive_label} (class {positive_idx})."

    summary = {
        "model_path": str(model_path),
        "model_sha": compute_sha256(model_path),
        "instrument": args.instrument,
        "data_path": str(data_path),
        "rows_used": int(len(X_frame)),
        "background_rows": int(background.shape[0]) if background is not None else 0,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "class_index": target_classes[0],
        "explained_classes": target_classes,
        "class_mapping": class_labels,
        "model_classes": [str(val) for val in model_classes],
        "classes": class_reports,
        "schema": schema.model_dump() if schema else None,
        "shap_direction": shap_direction,
    }
    write_json_summary(report_dir / "summary.json", summary)
    LOGGER.info("SHAP report complete. Files written to %s", report_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
