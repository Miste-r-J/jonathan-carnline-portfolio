from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Any, Sequence, Mapping

import numpy as np

_LOGGER = logging.getLogger(__name__)


def safe_clip(values: Iterable[float], *, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    """
    Clip probability-like values into [0, 1] without any row-wise renormalisation.
    Emits a single warning if any element falls outside the permissible bounds.
    """
    array = np.asarray(values, dtype=float)
    if np.any((array < lower) | (array > upper)):
        _LOGGER.warning("Probability values outside [%s, %s] detected; clipping", lower, upper)
    return np.clip(array, lower, upper, out=array.copy())


# ---------------------------------------------------------------------------
# Legacy helpers kept for model interoperability.
# ---------------------------------------------------------------------------

def _load_meta(meta: Any, model_path: str | Path | None) -> dict[str, Any]:
    if isinstance(meta, Mapping):  # type: ignore[arg-type]
        return dict(meta)
    if model_path:
        try:
            path = Path(model_path)
            if path.suffix:
                candidate = path.with_suffix(".meta.json")
            else:
                candidate = Path(str(path) + ".meta.json")
            if candidate.exists():
                return json.loads(candidate.read_text())
        except Exception:
            pass
    return {}


def _infer_long_index_from_meta(meta: dict[str, Any], classes: Sequence[Any]) -> int | None:
    if not classes:
        return None
    if "proba_code" in meta:
        try:
            code = meta["proba_code"]
            if code in classes:
                return list(classes).index(code)
        except Exception:
            pass
    if "class_names" in meta:
        try:
            names = [str(x).upper() for x in meta["class_names"]]
            for candidate in ("LONG", "BUY"):
                if candidate in names:
                    return names.index(candidate)
        except Exception:
            pass
    if "proba_class" in meta:
        tag = str(meta["proba_class"]).upper()
        if tag == "LONG" and classes:
            return 1 if len(classes) == 2 else len(classes) - 1
        if tag == "SHORT" and len(classes) >= 2:
            # mark that probabilities represent SHORT so caller can invert
            return None
    return None


def _should_invert_prob(meta: dict[str, Any]) -> bool:
    flag = str(meta.get("proba_class", "")).strip().upper()
    return flag == "SHORT"


def _coerce_classes(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        return raw.tolist()
    if isinstance(raw, (list, tuple)):
        return list(raw)
    try:
        return list(raw)
    except TypeError:
        return [raw]


def ensure_long_index(model: Any, *, meta: Any = None, model_path: str | Path | None = None) -> int:
    """
    Determine the column index representing LONG probabilities for a fitted model.
    Result is cached on the model via ``_na_long_index`` for reuse.
    """
    cached = getattr(model, "_na_long_index", None)
    if cached is not None:
        return int(cached)

    classes = _coerce_classes(getattr(model, "classes_", None))
    meta_dict = _load_meta(meta, model_path)

    idx = _infer_long_index_from_meta(meta_dict, classes)
    if idx is None:
        if classes:
            # Prefer class code 1 if present, otherwise final column.
            if 1 in classes:
                idx = int(list(classes).index(1))
            else:
                idx = max(len(classes) - 1, 0)
        else:
            idx = 1  # fallback for binary classifiers without explicit classes

    setattr(model, "_na_long_index", int(idx))
    setattr(model, "_na_invert_prob", bool(_should_invert_prob(meta_dict)))
    return int(idx)


def select_long_proba(
    proba: Iterable[float] | np.ndarray,
    model: Any,
    *,
    meta: Any = None,
    model_path: str | Path | None = None,
) -> np.ndarray:
    """
    Project raw ``predict_proba`` output to the LONG probability channel.
    """
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        out = arr[:, 0]
    elif arr.ndim >= 2:
        idx = ensure_long_index(model, meta=meta, model_path=model_path)
        idx = max(0, min(arr.shape[1] - 1, idx))
        out = arr[..., idx]
    else:
        out = arr

    invert = getattr(model, "_na_invert_prob", False)
    if invert:
        return 1.0 - out
    meta_dict = _load_meta(meta, model_path)
    if _should_invert_prob(meta_dict):
        return 1.0 - out
    return out


__all__ = ["ensure_long_index", "select_long_proba", "safe_clip"]
