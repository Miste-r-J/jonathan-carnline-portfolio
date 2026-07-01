from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import joblib


_REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    str(_REPO_ROOT / "artifacts" / "really_good_model.joblib"),
)
SYMBOL_MODEL_DIR = os.environ.get(
    "SYMBOL_MODEL_DIR",
    str(_REPO_ROOT / "artifacts" / "symbol_models"),
)

_model_cache: Dict[str, Any] = {}


def get_model_path_for_symbol(symbol: str) -> str:
    os.makedirs(SYMBOL_MODEL_DIR, exist_ok=True)
    return os.path.join(SYMBOL_MODEL_DIR, f"{symbol}_model.joblib")


def load_model_for_symbol(symbol: str):
    """
    Load or initialize the model for a given symbol.
    If no symbol-specific model exists, copy BASE_MODEL_PATH.
    Cache models in memory for reuse during a run.
    """
    if symbol in _model_cache:
        return _model_cache[symbol]
    target_path = get_model_path_for_symbol(symbol)
    if os.path.exists(target_path):
        model = joblib.load(target_path)
    else:
        model = joblib.load(BASE_MODEL_PATH)
        joblib.dump(model, target_path)
    _model_cache[symbol] = model
    return model


def save_model_for_symbol(symbol: str, model) -> None:
    path = get_model_path_for_symbol(symbol)
    joblib.dump(model, path)
    _model_cache[symbol] = model
