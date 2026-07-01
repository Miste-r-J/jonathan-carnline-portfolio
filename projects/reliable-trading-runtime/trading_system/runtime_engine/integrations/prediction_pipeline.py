from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional


def _stable_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_features_hash(features: Mapping[str, Any]) -> str:
    items = []
    for key in sorted(features.keys()):
        val = features.get(key)
        if val is None:
            norm = None
        else:
            try:
                fval = float(val)
            except Exception:
                norm = str(val)
            else:
                if fval != fval:
                    norm = None
                else:
                    norm = float(f"{fval:.10f}")
        items.append([key, norm])
    return _sha256(_stable_dumps(items))


_FORBIDDEN_META_SUBSTRINGS = ("fill", "order", "nt_", "lockout", "snapshot", "position", "realized", "pnl", "exec")


def _sanitize_meta(
    payload: Optional[Mapping[str, Any]],
    *,
    strict: bool,
    field_name: str,
) -> tuple[dict[str, Any], bool]:
    meta = dict(payload or {})
    blocked_keys = []
    for key in list(meta.keys()):
        key_lower = str(key).lower()
        if any(sub in key_lower for sub in _FORBIDDEN_META_SUBSTRINGS):
            blocked_keys.append(str(key))
    if blocked_keys:
        if strict:
            raise RuntimeError(f"Prediction meta '{field_name}' contains execution-derived keys: {blocked_keys}")
        for key in blocked_keys:
            meta.pop(key, None)
        return meta, True
    return meta, False


def build_prediction_bundle(
    *,
    run_id: str,
    bar_ts: str,
    instrument: str,
    features_hash: str,
    model_version: str,
    proba: float,
    side: str,
    entry_ref: float,
    stop_abs: float,
    target_abs: Optional[float],
    size_reco: int,
    risk_meta: Optional[Mapping[str, Any]] = None,
    policy_meta_model: Optional[Mapping[str, Any]] = None,
    sanitize_strict: bool = True,
) -> dict[str, Any]:
    prediction_key_src = {
        "bar_ts": bar_ts,
        "instrument": instrument,
        "features_hash": features_hash,
        "model_version": model_version,
        "proba": float(proba),
        "side": side,
        "entry_ref": entry_ref,
        "stop_abs": stop_abs,
        "target_abs": target_abs,
        "size_reco": size_reco,
    }
    prediction_key = _sha256(_stable_dumps(prediction_key_src))
    risk_meta_sanitized, risk_sanitized = _sanitize_meta(
        risk_meta, strict=sanitize_strict, field_name="risk_meta"
    )
    policy_meta_sanitized, policy_sanitized = _sanitize_meta(
        policy_meta_model, strict=sanitize_strict, field_name="policy_meta_model"
    )
    if not sanitize_strict and (risk_sanitized or policy_sanitized):
        policy_meta_sanitized["sanitized"] = True
    bundle = {
        "prediction_id": prediction_key,
        "prediction_key": prediction_key,
        "run_id": run_id,
        "bar_ts": bar_ts,
        "instrument": instrument,
        "features_hash": features_hash,
        "model_version": model_version,
        "proba": float(proba),
        "side": str(side),
        "entry_ref": float(entry_ref),
        "stop_abs": float(stop_abs),
        "target_abs": (float(target_abs) if target_abs is not None else None),
        "size_reco": int(size_reco),
        "risk_meta": risk_meta_sanitized,
        "policy_meta_model": policy_meta_sanitized,
    }
    hash_payload = {k: v for k, v in bundle.items() if k not in {"run_id", "deterministic_hash"}}
    deterministic_hash = _sha256(_stable_dumps(hash_payload))
    bundle["deterministic_hash"] = deterministic_hash
    return bundle
