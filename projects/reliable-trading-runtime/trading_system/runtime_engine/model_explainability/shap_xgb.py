from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from trading_system.runtime_engine.modeling.models import load_model, model_features

try:  # Optional import; keep module import-time light.
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CalibratedClassifierCV = None  # type: ignore

LOGGER = logging.getLogger("trading_system.runtime_engine.model_explainability.shap")

_SHAP_MODULE: Optional[Any] = None
_SHAP_WARNED = False


def _import_shap() -> Optional[Any]:
    global _SHAP_MODULE, _SHAP_WARNED
    if _SHAP_MODULE is not None:
        return _SHAP_MODULE
    try:
        import shap  # type: ignore

        _SHAP_MODULE = shap
    except Exception:  # pragma: no cover - optional dependency
        if not _SHAP_WARNED:
            LOGGER.warning("SHAP is not installed. Install with `pip install shap` to enable explainability.")
            _SHAP_WARNED = True
        _SHAP_MODULE = None
    return _SHAP_MODULE


def shap_available() -> bool:
    return _import_shap() is not None


def load_xgb_model(path: str | Path) -> Any:
    """
    Convenience wrapper around trading_system.runtime_engine.modeling.models.load_model with clear logging.
    """
    expanded = Path(path).expanduser()
    if not expanded.exists():
        raise FileNotFoundError(f"Model path {expanded} does not exist")
    LOGGER.info("Loading model from %s", expanded)
    return load_model(str(expanded))


def resolve_feature_names(path: str | Path) -> Optional[List[str]]:
    feats = model_features(str(path))
    if feats is None:
        return None
    return [str(f) for f in feats]


def _drop_leak_prone_columns(frame: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
    drop_exact = {"Open", "High", "Low", "Close", "Volume"}
    suspicious = ("follow", "future", "lead", "ahead", "fwd")
    safe = {"hi_break_follow", "lo_break_follow"}
    keep_idx: List[int] = []
    keep_names: List[str] = []
    for idx, name in enumerate(names):
        if name in drop_exact:
            continue
        lower = name.lower()
        if name not in safe and any(tok in lower for tok in suspicious):
            continue
        keep_idx.append(idx)
        keep_names.append(name)
    if len(keep_idx) == len(names):
        return frame, names
    return frame[:, keep_idx], keep_names


def align_frame_for_model(df: Any, feature_names: Optional[Sequence[str]]) -> Tuple[np.ndarray, List[str]]:
    """
    Return numpy matrix aligned with the model's expected feature order.
    """
    import pandas as pd  # local import to avoid mandatory dependency at module load

    if not isinstance(df, pd.DataFrame):
        raise TypeError("align_frame_for_model expects a pandas DataFrame")
    numeric = df.select_dtypes(include=[np.number, "bool"]).astype(float)
    if feature_names is None:
        matrix = numeric.to_numpy()
        cols = list(numeric.columns)
        matrix, cols = _drop_leak_prone_columns(matrix, cols)
        return matrix, cols
    cols: List[str] = []
    for name in feature_names:
        cols.append(name)
        if name not in numeric.columns:
            numeric[name] = 0.0
    matrix = numeric[cols].to_numpy()
    return matrix, cols


@dataclass(frozen=True)
class ShapRowSummary:
    base_value: Optional[float]
    class_index: Optional[int]
    shap_sum: Optional[float]
    top_positive: List[Dict[str, Any]]
    top_negative: List[Dict[str, Any]]
    probability: Optional[float] = None
    margin: Optional[float] = None
    contributions: Optional[List[Dict[str, Any]]] = None


def _unwrap_tree_estimator(model: Any) -> Any:
    """
    SHAP TreeExplainer does not support CalibratedClassifierCV directly.
    If we detect a calibrator, return its underlying tree estimator.
    """
    if CalibratedClassifierCV is not None and isinstance(model, CalibratedClassifierCV):
        base = getattr(model, "base_estimator", None)
        if base is not None:
            LOGGER.debug("Using base_estimator %s for SHAP explainability.", type(base).__name__)
            return base
        calibrated = getattr(model, "calibrated_classifiers_", None)
        if calibrated:
            first = calibrated[0]
            classifier = getattr(first, "classifier", None)
            if classifier is not None:
                LOGGER.debug("Using classifier from calibrated_classifiers_[0] (%s) for SHAP explainability.", type(classifier).__name__)
                return classifier
        estimator_attr = getattr(model, "estimator", None)
        if estimator_attr is not None:
            LOGGER.debug("Using estimator attribute %s for SHAP explainability.", type(estimator_attr).__name__)
            return estimator_attr
        try:
            estimator_method = model._get_estimator()  # type: ignore[attr-defined]
            if estimator_method is not None:
                LOGGER.debug("Using _get_estimator() result %s for SHAP explainability.", type(estimator_method).__name__)
                return estimator_method
        except Exception:
            pass
        LOGGER.debug("Unable to unwrap CalibratedClassifierCV; falling back to calibrator wrapper.")
        return model
    return model


class ShapExplainerCache:
    """
    Lazily build and reuse a TreeExplainer for an XGBoost/GBM style model.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        *,
        background: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self._tree_model = _unwrap_tree_estimator(model)
        self.feature_names = list(feature_names)
        self._background = background
        self._explainer: Optional[Any] = None
        self._shap_missing_warned = False

    def set_background(self, background: Optional[np.ndarray]) -> None:
        self._background = background
        self._explainer = None

    def _ensure_explainer(self) -> Optional[Any]:
        if self._explainer is not None:
            return self._explainer
        shap = _import_shap()
        if shap is None:
            self._shap_missing_warned = True
            return None
        params = {"feature_names": self.feature_names}
        if self._background is not None and len(self._background):
            params["data"] = self._background
        try:
            self._explainer = shap.TreeExplainer(self._tree_model, **params)
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Failed to initialize SHAP TreeExplainer: %s", exc)
            self._explainer = None
        return self._explainer

    def explain_matrix_classes(
        self,
        X: np.ndarray,
    ) -> Optional[Tuple[List[np.ndarray], List[Optional[float]]]]:
        explainer = self._ensure_explainer()
        if explainer is None:
            return None
        try:
            raw = explainer.shap_values(X, check_additivity=False)
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("SHAP explain_matrix_classes failed: %s", exc, exc_info=True)
            return None
        matrices: List[np.ndarray] = []
        if isinstance(raw, list):
            matrices = [np.asarray(v) for v in raw]
        else:
            arr = np.asarray(raw)
            if arr.ndim == 3:
                # SHAP has multiple conventions for multi-class:
                # - (n_classes, n_samples, n_features)
                # - (n_samples, n_features, n_classes)
                n_samples = int(getattr(X, "shape", (0,))[0] or 0)
                if n_samples > 0 and arr.shape[0] == n_samples:
                    matrices = [arr[:, :, k] for k in range(arr.shape[2])]
                else:
                    matrices = [arr[i] for i in range(arr.shape[0])]
            elif arr.ndim == 2:
                matrices = [arr]
            else:
                matrices = [arr.reshape(arr.shape[-1])]
        expected = getattr(explainer, "expected_value", None)
        expected_list: List[Optional[float]]
        if isinstance(expected, list):
            expected_list = [float(v) if v is not None else None for v in expected]
        elif isinstance(expected, np.ndarray) and expected.ndim > 0:
            expected_list = [float(expected[i]) for i in range(expected.shape[0])]
        else:
            expected_list = [float(expected) if expected is not None else None]
        if len(expected_list) < len(matrices):
            last = expected_list[-1] if expected_list else None
            expected_list.extend([last] * (len(matrices) - len(expected_list)))
        return matrices, expected_list

    def explain_matrix(
        self,
        X: np.ndarray,
        *,
        class_index: Optional[int] = None,
        predicted_index: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, Any, int]]:
        explained = self.explain_matrix_classes(X)
        if explained is None:
            return None
        matrices, expected_list = explained
        if not matrices:
            return None
        if len(matrices) == 1 and class_index is None and predicted_index is None:
            base_value = expected_list[0] if expected_list else None
            return matrices[0], base_value, 0
        idx = class_index
        if idx is None:
            idx = predicted_index
        if idx is None:
            idx = 0
        idx = max(0, min(int(idx), len(matrices) - 1))

        exp_val = expected_list[idx] if idx < len(expected_list) else expected_list[0] if expected_list else None
        return matrices[idx], exp_val, idx

    def explain_row(
        self,
        vector: Sequence[float],
        *,
        feature_values: Optional[Mapping[str, float]] = None,
        top_n: int = 10,
        include_full: bool = False,
        probability: Optional[float] = None,
        class_index: Optional[int] = None,
        predicted_index: Optional[int] = None,
    ) -> Optional[ShapRowSummary]:
        arr = np.asarray(vector, dtype=float).reshape(1, -1)
        result = self.explain_matrix(arr, class_index=class_index, predicted_index=predicted_index)
        if result is None:
            return None
        shap_values, base_value, used_class = result
        shap_row = np.asarray(shap_values).reshape(-1)
        if feature_values is None:
            feature_values = {
                name: float(val)
                for name, val in zip(self.feature_names, arr.reshape(-1)[: len(self.feature_names)])
            }
        contributions: List[Dict[str, Any]] = []
        for idx, name in enumerate(self.feature_names[: len(shap_row)]):
            shap_val = float(shap_row[idx])
            contributions.append(
                {
                    "name": name,
                    "value": float(feature_values.get(name, float("nan"))),
                    "shap": shap_val,
                }
            )
        positives = sorted(
            [item for item in contributions if item["shap"] >= 0],
            key=lambda item: item["shap"],
            reverse=True,
        )[:top_n]
        negatives = sorted(
            [item for item in contributions if item["shap"] < 0],
            key=lambda item: item["shap"],
        )[:top_n]
        shap_sum = float(np.sum(shap_row)) if shap_row.size else None
        margin = None
        if shap_sum is not None and base_value is not None:
            margin = float(base_value + shap_sum)
        payload = ShapRowSummary(
            base_value=float(base_value) if base_value is not None else None,
            class_index=int(used_class) if used_class is not None else None,
            shap_sum=shap_sum,
            top_positive=positives,
            top_negative=negatives,
            probability=float(probability) if probability is not None else None,
            margin=margin,
            contributions=contributions if include_full else None,
        )
        return payload


def summarize_mean_abs_shap(shap_matrix: np.ndarray, feature_names: Sequence[str]) -> List[Tuple[str, float]]:
    if shap_matrix.ndim == 3 and shap_matrix.shape[0] > 1:
        # collapse to single class (mean over classes)
        shap_matrix = np.mean(np.abs(shap_matrix), axis=0)
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    order = np.argsort(mean_abs)[::-1]
    summary: List[Tuple[str, float]] = []
    for idx in order:
        summary.append((feature_names[idx], float(mean_abs[idx])))
    return summary


def write_json_summary(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
