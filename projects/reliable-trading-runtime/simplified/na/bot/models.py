from __future__ import annotations
"""
Model registry & loader (production-ready, prop-firm bot compatible, patched v3)

Features
--------
- Discovers models from one or more artifact directories (env configurable)
- Optional manual registry entries with rich metadata & default presets
- Safe loading with joblib → cloudpickle → pickle fallback
- Robust feature list discovery (sidecar JSON or model.feature_names_in_)
- Calibrator inference + loader (filename/sidecar metadata; *.calib.joblib next to model)
- JSON-friendly listings, latest model selection, SHA-256 helper, and CLI
- Strict path expansion & clear error messages; small validation hooks

Enhancements
------------
✅ Thread-safe registry cache mutation (Lock-protected)
✅ Auto SHA-256 computation during discovery (can be disabled via env)
✅ Registry snapshot export to artifacts/registry.json (atomic write)
✅ Alias conflict handling with warnings; prefers newest mtime
✅ Added safe `load_model()` and `model_features()` helpers
✅ `compute_sha256()` updates live cache + snapshot
✅ Improved logs and CLI (`--refresh`, `features` command)

Env Vars
--------
BOT_ARTIFACTS_DIR         : base directory of models (default: "artifacts")
BOT_EXTRA_MODELS_DIRS     : extra model dirs (":" or ";" separated)
BOT_EXTRA_MODELS_DIR      : (legacy) single extra dir
BOT_MODELS_GLOB           : comma-separated patterns (default: "*.joblib,*.pkl")
BOT_REGISTRY_TTL_SEC      : cache TTL for discovery (default: 10)
BOT_DISABLE_SHA           : set "1" to skip SHA-256 during discovery (faster)
"""

import os
import re
import sys
import json
import time
import hashlib
import warnings
import pickle
import importlib
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Iterable

import numpy as np

# --- Optional deps (graceful fallbacks) ---
# NOTE: keep these lazy to avoid heavy imports during CLI startup and smoke tests.
joblib_load = None  # type: ignore
cloudpickle = None  # type: ignore


def _lazy_import_joblib_load():
    global joblib_load
    if joblib_load is not None:
        return joblib_load
    try:
        from joblib import load as _joblib_load  # type: ignore
    except Exception:  # pragma: no cover
        joblib_load = None  # type: ignore
        return None
    joblib_load = _joblib_load  # type: ignore
    return joblib_load


def _lazy_import_cloudpickle():
    global cloudpickle
    if cloudpickle is not None:
        return cloudpickle
    try:
        import cloudpickle as _cloudpickle  # type: ignore
    except Exception:  # pragma: no cover
        cloudpickle = None  # type: ignore
        return None
    cloudpickle = _cloudpickle  # type: ignore
    return cloudpickle


def _ensure_posixpath_compat() -> None:
    """
    On Windows, make unpickling Linux-trained artifacts that reference
    pathlib.PosixPath work by aliasing it to a Windows-capable Path.

    Many sklearn/joblib models pickle pathlib.PosixPath instances when
    trained on Linux. On Windows, the stdlib's PosixPath refuses to
    instantiate (UnsupportedOperation). We patch the pathlib module so
    that any references to pathlib.PosixPath during unpickling resolve
    to a WindowsPath-compatible implementation instead.
    """
    if os.name != "nt":
        return
    try:
        import pathlib as _pl  # local import to avoid cycles
        posix_cls = getattr(_pl, "PosixPath", None)
        windows_cls = getattr(_pl, "WindowsPath", None)
        if not isinstance(posix_cls, type) or not isinstance(windows_cls, type):
            return

        # If we've already patched, do nothing.
        if getattr(_pl.PosixPath, "__name__", "") == "PatchedPosixPath":  # type: ignore[attr-defined]
            return

        class PatchedPosixPath(windows_cls):  # type: ignore[misc]
            """Compat shim: treat PosixPath like WindowsPath on Windows."""

            pass

        _pl.PosixPath = PatchedPosixPath  # type: ignore[assignment]
    except Exception:
        # Best-effort only; if this fails we fall back to the normal behaviour.
        return

# --- Optional config linkage (for preset suggestions/validation) ---
try:
    from .config import PRESETS as _CONFIG_PRESETS  # type: ignore
except Exception:  # pragma: no cover
    try:
        from config import PRESETS as _CONFIG_PRESETS  # type: ignore
    except Exception:
        _CONFIG_PRESETS = None


# =============================
# Model Registry
# =============================

@dataclass(frozen=True)
class ModelSpec:
    alias: str
    path: Path
    features_sidecar: Optional[Path] = None
    default_preset: Optional[str] = None
    description: str = ""
    calibrator: Optional[str] = None
    created_ts: float = 0.0
    modified_ts: float = 0.0
    size_bytes: int = 0
    sha256: Optional[str] = None


# ---------------------- Helpers ----------------------

def _expand(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()


def _split_dirs(var: str | None) -> List[Path]:
    if not var:
        return []
    parts = [s.strip() for s in var.replace(";", ":").split(":") if s.strip()]
    return [_expand(s) for s in parts]


def _parse_globs(var: str | None) -> List[str]:
    raw = var or "*.joblib,*.pkl"
    return [g.strip() for g in raw.split(",") if g.strip()]



ARTIFACTS_DIR = _expand(os.environ.get("BOT_ARTIFACTS_DIR", "artifacts"))
EXTRA_DIRS = _split_dirs(os.environ.get("BOT_EXTRA_MODELS_DIRS"))
if not EXTRA_DIRS and os.environ.get("BOT_EXTRA_MODELS_DIR"):
    EXTRA_DIRS = [_expand(os.environ["BOT_EXTRA_MODELS_DIR"])]

MODEL_PATTERNS = _parse_globs(os.environ.get("BOT_MODELS_GLOB"))
REG_TTL = int(os.environ.get("BOT_REGISTRY_TTL_SEC", "10"))
DISABLE_SHA = os.environ.get("BOT_DISABLE_SHA", "0") == "1"


def _file_sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _infer_calibrator(model_path: Path) -> Optional[str]:
    # Sidecar wins over filename hints
    sidecar = model_path.with_suffix(".features.json")
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text())
            meth = None
            if isinstance(meta, dict):
                meth = meta.get("calibration") or meta.get("calib")
            if isinstance(meth, str):
                return meth
        except Exception:
            pass
    name = model_path.stem.lower()
    if "isotonic" in name:
        return "isotonic"
    if "sigmoid" in name or "platt" in name:
        return "sigmoid"
    return None


def compute_position_size(signal_grade: str, base_position: int = 1):
    grade_multiplier = {"A+": 2.0, "B+": 1.5}
    return int(base_position * grade_multiplier.get(signal_grade, 1.0))


def adaptive_bias(market_context):
    recent_breakouts = market_context.get("breakout_outcomes", [])[-10:]
    if not recent_breakouts:
        return 0.5
    p_buy = sum(recent_breakouts) / len(recent_breakouts)
    return max(0.4, min(0.6, p_buy))
# ---------------------- Discovery ----------------------

def _discover_under(root: Path, patterns: Iterable[str]) -> Dict[str, ModelSpec]:
    reg: Dict[str, ModelSpec] = {}
    if not root.exists():
        return reg
    for pat in patterns:
        for fp in root.rglob(pat):
            if not fp.is_file() or fp.name.startswith("._"):
                continue
            try:
                st = fp.stat()
                if st.st_size == 0:
                    continue
                alias = fp.stem
                sidecar = fp.with_suffix(".features.json")
                digest = None if DISABLE_SHA else _file_sha256(fp)
                reg[alias] = ModelSpec(
                    alias=alias,
                    path=fp,
                    features_sidecar=sidecar if sidecar.exists() else None,
                    default_preset=None,
                    description="",
                    calibrator=_infer_calibrator(fp),
                    created_ts=st.st_ctime,
                    modified_ts=st.st_mtime,
                    size_bytes=st.st_size,
                    sha256=digest,
                )
            except Exception:
                # Continue scanning even if a single file misbehaves
                continue
    return reg


def _merge_registry(reg: Dict[str, ModelSpec], new: Dict[str, ModelSpec]) -> Dict[str, ModelSpec]:
    # Prefer newest (mtime) on alias collision; warn in both cases
    for alias, spec in new.items():
        if alias in reg:
            old = reg[alias]
            if spec.modified_ts > old.modified_ts:
                warnings.warn(f"[Registry] Alias '{alias}' conflict; taking newer {spec.path} over {old.path}")
                reg[alias] = spec
            else:
                warnings.warn(f"[Registry] Alias '{alias}' conflict; keeping {old.path}, ignoring {spec.path}")
        else:
            reg[alias] = spec
    return reg


# ---------------------- Cache & Snapshot ----------------------

_REG_CACHE_TS: float = 0.0
_REG_CACHE: Dict[str, ModelSpec] = {}
_REG_LOCK = threading.Lock()
_SNAPSHOT_PATH = ARTIFACTS_DIR / "registry.json"


def _export_snapshot(reg: Dict[str, ModelSpec]) -> None:
    try:
        from .paths import REPO_ROOT

        def _rel(p: Optional[Path]) -> Optional[str]:
            if p is None:
                return None
            try:
                return str(Path(p).resolve().relative_to(REPO_ROOT))
            except Exception:
                # Fallback to plain string if outside repo root.
                return str(p)

        payload = {}
        for k, v in reg.items():
            d = asdict(v)
            d["path"] = _rel(v.path)
            d["features_sidecar"] = _rel(v.features_sidecar) if v.features_sidecar else None
            payload[k] = d
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _SNAPSHOT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(_SNAPSHOT_PATH)  # atomic on most OSes
    except Exception as e:
        warnings.warn(f"[Registry] Snapshot export failed: {e}")


def _build_registry(force: bool = False) -> Dict[str, ModelSpec]:
    global _REG_CACHE, _REG_CACHE_TS
    now = time.time()
    with _REG_LOCK:
        if not force and _REG_CACHE and (now - _REG_CACHE_TS) < REG_TTL:
            return _REG_CACHE
        reg: Dict[str, ModelSpec] = {}
        reg = _merge_registry(reg, _discover_under(ARTIFACTS_DIR, MODEL_PATTERNS))
        for d in EXTRA_DIRS:
            reg = _merge_registry(reg, _discover_under(d, MODEL_PATTERNS))
        _REG_CACHE = reg
        _REG_CACHE_TS = now
        _export_snapshot(reg)
        print(f"[Registry] Refreshed {len(reg)} models from {ARTIFACTS_DIR} + {len(EXTRA_DIRS)} extra dir(s).")
        return reg


# Initialize on import (fresh)
_build_registry(force=True)


# ---------------------- Public API ----------------------

def refresh_registry(force: bool = False) -> Dict[str, ModelSpec]:
    return _build_registry(force=force)


def list_models() -> Dict[str, dict]:
    reg = refresh_registry(False)
    # Sorted newest first
    sorted_items = sorted(reg.items(), key=lambda kv: kv[1].modified_ts, reverse=True)
    return {
        alias: {
            "path": str(spec.path),
            "features_sidecar": str(spec.features_sidecar) if spec.features_sidecar else None,
            "default_preset": spec.default_preset,
            "description": spec.description,
            "calibrator": spec.calibrator,
            "created": spec.created_ts,
            "modified": spec.modified_ts,
            "size_bytes": spec.size_bytes,
            "sha256": spec.sha256,
        }
        for alias, spec in sorted_items
    }


def resolve_model(spec_or_path: str) -> ModelSpec:
    # Accept direct file path or alias
    reg = refresh_registry(False)
    raw = str(spec_or_path or "").strip()
    p = _expand(raw)

    candidate_paths: List[Path] = [p]
    # When launched from the repo root, relative specs like `artifacts\foo.joblib`
    # may resolve to `<repo>\artifacts` instead of `<repo>\simplified\artifacts`.
    # Add deterministic fallbacks rooted at the simplified package tree.
    try:
        simplified_root = Path(__file__).resolve().parents[2]
        candidate_paths.append((simplified_root / raw).resolve())
        if "\\" in raw or "/" in raw:
            candidate_paths.append((simplified_root / "artifacts" / Path(raw).name).resolve())
    except Exception:
        pass

    seen: set[str] = set()
    resolved_path: Optional[Path] = None
    for cand in candidate_paths:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            resolved_path = cand
            break

    if resolved_path is not None:
        # If a JSON evaluation/metadata file is passed (e.g. artifacts/eval/*.json),
        # interpret its stem as a registry alias and prefer the discovered model
        # artifact over trying to unpickle the JSON itself.
        if resolved_path.suffix.lower() == ".json":
            alias = resolved_path.stem
            if alias in reg:
                return reg[alias]
        st = resolved_path.stat()
        digest = None if DISABLE_SHA else _file_sha256(resolved_path)
        sidecar = resolved_path.with_suffix(".features.json")
        return ModelSpec(
            alias=resolved_path.stem,
            path=resolved_path,
            features_sidecar=sidecar if sidecar.exists() else None,
            created_ts=st.st_ctime,
            modified_ts=st.st_mtime,
            size_bytes=st.st_size,
            sha256=digest,
            calibrator=_infer_calibrator(resolved_path),
        )
    if raw in reg:
        return reg[raw]
    raise FileNotFoundError(f"No model found for '{spec_or_path}'.")


def model_path(spec_or_path: str) -> str:
    return str(resolve_model(spec_or_path).path)


def compute_sha256(spec_or_path: str, *, force: bool = False) -> str:
    spec = resolve_model(spec_or_path)
    if spec.sha256 and not force:
        return spec.sha256
    digest = _file_sha256(spec.path)
    with _REG_LOCK:
        # Update the live cache entry (if present) with new digest
        reg = refresh_registry(False)
        current = reg.get(spec.alias)
        updated = ModelSpec(**{**asdict(current or spec), "sha256": digest})
        _REG_CACHE[spec.alias] = updated
        _export_snapshot(_REG_CACHE)
    return digest


# --------- Safe loading & feature helpers ---------

def _renorm_proba(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.clip(arr, eps, 1.0)
    denom = arr.sum(axis=1, keepdims=True)
    denom = np.where(denom <= eps, 1.0, denom)
    return arr / denom


class _OvRCalibratedFuserShim:
    """Lightweight replica of the training-time OvRCalibratedFuser for inference."""

    def __init__(self, model_long, model_short, order=("SHORT", "FLAT", "LONG")):
        self.model_long = model_long
        self.model_short = model_short
        self.order = order
        self.feature_names_in_ = getattr(model_long, "feature_names_in_", None)

    def predict_proba(self, X):
        p_long = np.asarray(self.model_long.predict_proba(X))[:, 1]
        p_short = np.asarray(self.model_short.predict_proba(X))[:, 1]
        p_flat = np.clip(1.0 - p_long - p_short, 1e-9, 1.0)
        stacked = np.stack([p_short, p_flat, p_long], axis=1)
        return _renorm_proba(stacked)

    def fit(self, *args, **kwargs):  # pragma: no cover - sklearn API compat
        return self

    def get_params(self, deep: bool = False):  # pragma: no cover
        return {}

    def set_params(self, **kwargs):  # pragma: no cover
        return self


def _handle_missing_fuser(exc: Exception) -> bool:
    """Attempt to register OvRCalibratedFuser on the module mentioned in the error."""
    if not isinstance(exc, AttributeError):
        return False
    msg = str(exc)
    if "OvRCalibratedFuser" not in msg:
        return False
    match = re.search(r"module '([^']+)'", msg)
    module_name = match.group(1) if match else "__main__"
    try:
        trainer_mod = importlib.import_module("na.bot.train_hgb_multi")
        fuser_cls = getattr(trainer_mod, "OvRCalibratedFuser", None)
    except Exception:
        fuser_cls = None

    if fuser_cls is None:
        fuser_cls = globals().get("OvRCalibratedFuser", None)
    if fuser_cls is None:
        fuser_cls = _OvRCalibratedFuserShim
    if fuser_cls is None:
        return False

    target_modules = []
    mod = sys.modules.get(module_name)
    if mod is None:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            mod = None
    if mod is not None:
        target_modules.append(mod)
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and main_mod not in target_modules:
        target_modules.append(main_mod)
    registered = False
    for target in target_modules:
        if target is None:
            continue
        if getattr(target, "OvRCalibratedFuser", None) is fuser_cls:
            registered = True
            continue
        try:
            setattr(target, "OvRCalibratedFuser", fuser_cls)
            registered = True
        except Exception:
            continue
    return registered


def _attempt_load(callable_loader):
    try:
        return callable_loader()
    except AttributeError as exc:
        if _handle_missing_fuser(exc):
            return callable_loader()
        raise


def load_model(spec_or_path: str):
    """Load a model with joblib → cloudpickle → pickle fallback.

    WARNING: Loading serialized models executes arbitrary code. Only load trusted artifacts.
    """
    spec = resolve_model(spec_or_path)

    # Ensure pathlib.PosixPath from Linux-trained artifacts can be unpickled on Windows.
    _ensure_posixpath_compat()

    # joblib first
    jl = _lazy_import_joblib_load()
    if jl is not None:
        try:
            return _attempt_load(lambda: jl(spec.path))
        except Exception:
            pass

    # cloudpickle next
    cp = _lazy_import_cloudpickle()
    if cp is not None:
        try:
            def _cloudpickle_loader():
                with spec.path.open("rb") as f:
                    return cp.load(f)
            return _attempt_load(_cloudpickle_loader)
        except Exception:
            pass

    # stdlib pickle last
    def _pickle_loader():
        with spec.path.open("rb") as f:
            return pickle.load(f)
    return _attempt_load(_pickle_loader)


def model_features(spec_or_path: str) -> Optional[List[str]]:
    """Return feature list from sidecar or model.feature_names_in_, if available."""
    spec = resolve_model(spec_or_path)
    if spec.features_sidecar and spec.features_sidecar.exists():
        try:
            meta = json.loads(spec.features_sidecar.read_text())
            feats = meta.get("features") if isinstance(meta, dict) else meta
            if isinstance(feats, list):
                return [str(x) for x in feats]
        except Exception:
            pass
    try:
        m = load_model(spec_or_path)
        feats = getattr(m, "feature_names_in_", None)
        return list(feats) if feats is not None else None
    except Exception:
        return None


# ---------------------- CLI ----------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Model registry CLI (patched)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="List discovered models (JSON)")
    p_ls.add_argument("--refresh", action="store_true", help="Force refresh registry cache")

    p_path = sub.add_parser("path", help="Print model path")
    p_path.add_argument("arg")

    p_hash = sub.add_parser("hash", help="Compute SHA-256 of model file")
    p_hash.add_argument("arg")
    p_hash.add_argument("--force", action="store_true")

    p_feats = sub.add_parser("features", help="Print feature list (if available)")
    p_feats.add_argument("arg")

    args = ap.parse_args()

    if args.cmd == "ls":
        refresh_registry(force=args.refresh)
        print(json.dumps(list_models(), indent=2))
    elif args.cmd == "path":
        print(model_path(args.arg))
    elif args.cmd == "hash":
        print(compute_sha256(args.arg, force=args.force))
    elif args.cmd == "features":
        print(json.dumps(model_features(args.arg), indent=2))
