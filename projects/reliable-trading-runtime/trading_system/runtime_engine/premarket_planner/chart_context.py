from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from trading_system.runtime_engine.modeling.config import INSTRUMENTS

from .analysis import (
    PremarketSnapshot,
    _determine_bias,
    _format_confidence,
    _load_latest_signal,
    _signal_age_hours,
)
from .config import PlannerConfig, WORKSPACE_ROOT
from .core import PlanComputation
from .structure import StructureDetector

DEFAULT_PLAN_CONTEXT_PATH = WORKSPACE_ROOT / 'runs' / 'live' / 'common' / 'premarket_plan.json'

# Signals revert to structure guidance after this many hours (overridable via env)
MAX_SIGNAL_AGE_HOURS = float(os.getenv("PREMARKET_SIGNAL_MAX_AGE_HOURS", "4"))


def serialize_plan_context(computation: PlanComputation, config: PlannerConfig) -> Dict[str, Any]:
    snapshot = computation.snapshot
    current_price = _safe_float(snapshot.current_price)
    areas = _collect_area_entries(snapshot, computation, config)
    supports, resistances = _split_support_resistance(areas)
    structure = _build_structure_summary(snapshot)
    swings, structure_notes = _build_structure_events(computation, config)
    structure["swings"] = swings
    structure["notes"] = structure_notes
    volume_profile = _build_volume_profile_entries(computation, config, current_price)
    history = _build_history_bundle(snapshot, computation, config)
    trend = _build_trend_bundle(computation)
    bias = _build_bias_bundle(snapshot, computation, config, trend)
    trade_prediction, trade_readiness = _build_trade_prediction(
        bias,
        supports,
        resistances,
        structure,
        computation,
        snapshot,
        config,
    )
    timezone_label, chart_zone = _resolve_chart_timezone(config, computation)
    generated_at = datetime.now(chart_zone).isoformat()
    bar_timestamp = _convert_timestamp(snapshot.last_update, chart_zone, config.session_zone).isoformat()
    payload = {
        "areas": areas,
        "supports": supports,
        "resistances": resistances,
        "structure": structure,
        "history": history,
        "volume_profile": volume_profile,
        "bias": bias,
        "trend": trend,
        "planner_timestamp": generated_at,
        "planner_generated_at": generated_at,
        "planner_bar_timestamp": bar_timestamp,
        "current_price": current_price,
        "session_timezone": timezone_label,
        "trade_prediction": trade_prediction,
        "trade_readiness": trade_readiness,
    }
    return payload


def dump_plan_context(path: Path, context: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "context": context,
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_plan_context(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and "context" in payload:
        context = payload["context"]
    else:
        context = payload
    if not isinstance(context, dict):
        return None
    return context


def _resolve_chart_timezone(config: PlannerConfig, computation: PlanComputation) -> Tuple[str, ZoneInfo]:
    candidates = [
        config.display_timezone,
        getattr(getattr(config, "data", None), "csv_timezone", None),
        getattr(computation, "source_timezone", None),
        config.session_tz,
    ]
    for name in candidates:
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            return name.strip(), ZoneInfo(name.strip())
        except Exception:
            continue
    return config.session_tz, config.session_zone


def _convert_timestamp(value: Any, target_zone: ZoneInfo, fallback_zone: ZoneInfo) -> datetime:
    ts: Optional[datetime] = None
    if isinstance(value, datetime):
        ts = value
    elif hasattr(value, "to_pydatetime"):
        try:
            ts = value.to_pydatetime()
        except Exception:
            ts = None
    if ts is None:
        return datetime.now(target_zone)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=fallback_zone)
    return ts.astimezone(target_zone)


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _collect_area_entries(
    snapshot: "PremarketSnapshot",
    computation: PlanComputation,
    config: PlannerConfig,
    max_items: int = 8,
) -> List[Dict[str, Any]]:
    current_price = _safe_float(snapshot.current_price)
    entries: Dict[str, Dict[str, Any]] = {}

    def register(label: str, raw_value: Any, source: str) -> None:
        price = _safe_float(raw_value)
        if price is None:
            return
        delta = None if current_price is None else price - current_price
        distance = abs(delta) if delta is not None else None
        identifier = f"{source}-{_slugify(label)}-{int(round(price * 100))}"
        entries[identifier] = {
            "id": identifier,
            "label": label,
            "price": price,
            "delta": delta,
            "distance": distance,
            "type": _area_type(delta),
            "source": source,
        }

    register("Prior RTH High", snapshot.prev_high, "rth")
    register("Prior RTH Low", snapshot.prev_low, "rth")
    register("Prior Close", snapshot.prev_close, "rth")
    register("Prior RTH VWAP", snapshot.prev_session_vwap, "rth")
    register("Overnight High", snapshot.overnight_high, "overnight")
    register("Overnight Low", snapshot.overnight_low, "overnight")
    register("Current Price", snapshot.current_price, "last")

    if snapshot.overnight_high and snapshot.overnight_low:
        mid = (float(snapshot.overnight_high) + float(snapshot.overnight_low)) / 2.0
        register("Overnight Mid", mid, "overnight")

    overnight_vwap = _compute_overnight_vwap(computation)
    register("Overnight VWAP", overnight_vwap, "overnight")

    twenty_four_hour = _compute_recent_extremes(computation, hours=24)
    if twenty_four_hour:
        register("24h High", twenty_four_hour["high"], "recent")
        register("24h Low", twenty_four_hour["low"], "recent")

    items = list(entries.values())
    items.sort(key=lambda item: (math.inf if item["distance"] is None else item["distance"], item["price"]))
    return items[:max_items]


def _split_support_resistance(
    areas: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    supports = [area for area in areas if area.get("type") == "support"]
    resistances = [area for area in areas if area.get("type") == "resistance"]
    return supports[:3], resistances[:3]


def _build_trend_bundle(computation: PlanComputation) -> Dict[str, Optional[str]]:
    label, detail = _detect_trend(computation)
    state = _normalize_trend_state(label)
    return {"state": state, "label": label, "detail": detail}


def _normalize_trend_state(label: Optional[str]) -> str:
    mapping = {
        "Uptrend": "up",
        "Downtrend": "down",
        "Sideways": "range",
    }
    return mapping.get(label or "", "unknown")


def _detect_trend(computation: PlanComputation) -> Tuple[str, Optional[str]]:
    df_local = getattr(computation, "dataframe", None)
    if df_local is None or "Close" not in df_local:
        return "Unknown", None

    try:
        closes = df_local["Close"].astype(float).dropna()
    except Exception:
        return "Unknown", None

    closes = closes.tail(120)
    if len(closes) < 20:
        return "Unknown", None

    short_window = closes.tail(20)
    long_window = closes.tail(60) if len(closes) >= 60 else closes

    short_mean = float(short_window.mean())
    long_mean = float(long_window.mean())
    latest = float(closes.iloc[-1])
    first = float(closes.iloc[0])

    slope = latest - first
    pct_change = None
    if first != 0:
        pct_change = (latest / first - 1.0) * 100.0

    ratio = short_mean / long_mean if long_mean else 1.0
    tolerance = 0.001

    if ratio > 1.0 + tolerance and slope > 0:
        label = "Uptrend"
    elif ratio < 1.0 - tolerance and slope < 0:
        label = "Downtrend"
    else:
        label = "Sideways"

    detail_parts: List[str] = [
        f"20-bar SMA {short_mean:.2f}",
        f"60-bar SMA {long_mean:.2f}",
    ]
    if pct_change is not None and math.isfinite(pct_change):
        detail_parts.append(f"{pct_change:+.2f}% over ~{len(closes) * 5 // 60}h")

    detail = ", ".join(detail_parts)
    return label, detail


def _build_structure_summary(snapshot: "PremarketSnapshot") -> Dict[str, Any]:
    high_summary = _classify_extreme(snapshot.overnight_high, snapshot.prev_high, "high")
    low_summary = _classify_extreme(snapshot.overnight_low, snapshot.prev_low, "low")
    return {"highs": high_summary, "lows": low_summary}


def _structure_bias_hint(structure: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    highs = structure.get("highs") or {}
    lows = structure.get("lows") or {}
    high_label = str(highs.get("label") or "").lower()
    low_label = str(lows.get("label") or "").lower()
    if not high_label or not low_label:
        return None

    def _pick_value(source: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            val = _safe_float(source.get(key))
            if val is not None:
                return val
        return None

    if "higher" in high_label and "higher" in low_label:
        return {
            "direction": "LONG",
            "probabilityPct": 62,
            "confidence": "Structure",
            "market_state": "Structure favors upside",
            "entryLabel": lows.get("label") or "Structure Support",
            "targetLabel": highs.get("label") or "Structure High",
            "invalidateLabel": "Prior Swing Low",
            "entryPrice": _pick_value(lows, "current", "reference"),
            "targetPrice": _pick_value(highs, "reference", "current"),
            "invalidatePrice": _pick_value(lows, "reference"),
            "timestampLabel": "Structure favors upside",
        }
    if "lower" in high_label and "lower" in low_label:
        return {
            "direction": "SHORT",
            "probabilityPct": 38,
            "confidence": "Structure",
            "market_state": "Structure favors downside",
            "entryLabel": highs.get("label") or "Structure High",
            "targetLabel": lows.get("label") or "Structure Support",
            "invalidateLabel": "Prior Swing High",
            "entryPrice": _pick_value(highs, "current", "reference"),
            "targetPrice": _pick_value(lows, "reference", "current"),
            "invalidatePrice": _pick_value(highs, "reference"),
            "timestampLabel": "Structure favors downside",
        }
    return None


def _classify_extreme(current: Any, reference: Any, axis: str) -> Optional[Dict[str, Any]]:
    current_value = _safe_float(current)
    reference_value = _safe_float(reference)
    if current_value is None or reference_value is None:
        return None
    if axis == "high":
        if current_value > reference_value:
            label = "Higher High"
        elif current_value < reference_value:
            label = "Lower High"
        else:
            label = "Equal High"
    else:
        if current_value > reference_value:
            label = "Higher Low"
        elif current_value < reference_value:
            label = "Lower Low"
        else:
            label = "Equal Low"
    return {
        "label": label,
        "current": current_value,
        "reference": reference_value,
    }


def _build_structure_events(
    computation: PlanComputation,
    config: PlannerConfig,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    try:
        detector = StructureDetector(instrument=config.instrument)
        report = detector.analyze(computation.dataframe)
    except Exception as exc:  # pragma: no cover - optional dependency noise
        return [], [f"Structure analysis unavailable: {exc}"]

    events: List[Dict[str, Any]] = []
    for event in report.events:
        break_time = _datetime_to_unix(event.break_time)
        retest_time = _datetime_to_unix(event.retest_time) if event.retest_time else None
        events.append(
            {
                "id": f"{event.direction}-{break_time}",
                "direction": event.direction,
                "label": "Buy Trigger" if event.direction == "bullish" else "Sell Trigger",
                "break_level": float(event.break_level),
                "break_time": break_time,
                "break_price": float(event.break_price),
                "displacement": float(event.displacement),
                "retest_hit": bool(event.retest_hit),
                "retest_time": retest_time,
                "retest_price": _safe_float(event.retest_price),
                "retest_distance": _safe_float(event.retest_distance),
            }
        )
    notes = list(report.notes or ())
    return events, notes


def _build_volume_profile_entries(
    computation: PlanComputation,
    config: PlannerConfig,
    current_price: Optional[float],
    max_levels: int = 3,
) -> List[Dict[str, Any]]:
    frame = getattr(computation.window, "frame", None)
    if frame is None or getattr(frame, "empty", True):
        return []
    if "Volume" not in frame.columns:
        return []

    price_cols = [col for col in ("Open", "High", "Low", "Close") if col in frame.columns]
    if not price_cols:
        return []

    try:
        prices = frame[price_cols].astype(float).mean(axis=1)
        volumes = frame["Volume"].astype(float)
    except Exception:
        return []

    mask = volumes > 0
    prices = prices[mask]
    volumes = volumes[mask]
    if prices.empty:
        return []

    tick_size = 0.25
    spec = INSTRUMENTS.get(config.instrument.upper()) if INSTRUMENTS else None
    tick_candidate = getattr(spec, "tick_size", None)
    if isinstance(tick_candidate, (int, float)) and tick_candidate > 0:
        tick_size = float(tick_candidate)

    bucket_indices = (prices / tick_size).round().astype(int)
    buckets: Dict[float, float] = {}
    for idx, volume in zip(bucket_indices, volumes):
        price = idx * tick_size
        buckets[price] = buckets.get(price, 0.0) + float(volume)

    if not buckets:
        return []

    total_volume = sum(buckets.values())
    sorted_levels = sorted(buckets.items(), key=lambda item: item[1], reverse=True)[:max_levels]

    entries: List[Dict[str, Any]] = []
    for price, volume in sorted_levels:
        pct = (volume / total_volume * 100.0) if total_volume else 0.0
        delta = None if current_price is None else price - current_price
        entries.append(
            {
                "price": price,
                "volume": volume,
                "share": pct,
                "delta": delta,
            }
        )
    return entries


def _build_history_bundle(
    snapshot: "PremarketSnapshot",
    computation: PlanComputation,
    config: PlannerConfig,
    session_limit: int = 4,
) -> Dict[str, Any]:
    sessions = _collect_session_history(computation, config, limit=session_limit)
    overnight = {
        "high": _safe_float(snapshot.overnight_high),
        "low": _safe_float(snapshot.overnight_low),
        "range": _safe_float(snapshot.overnight_range),
        "volume": _safe_float(snapshot.premarket_volume),
    }
    return {
        "sessions": sessions,
        "overnight": overnight,
        "analysis_date": snapshot.analysis_date.isoformat(),
        "gap": _safe_float(snapshot.gap_points),
        "gap_pct": _safe_float(snapshot.gap_pct),
    }


def _collect_session_history(
    computation: PlanComputation,
    config: PlannerConfig,
    limit: int,
) -> List[Dict[str, Any]]:
    frame = computation.dataframe
    if frame is None or getattr(frame, "empty", True):
        return []

    try:
        rth = frame.between_time(config.rth_start_str, config.rth_end_str, inclusive="both")
    except TypeError:
        rth = frame.between_time(config.rth_start_str, config.rth_end_str)

    if rth.empty:
        return []

    session_dates = sorted(set(rth.index.date))
    entries: List[Dict[str, Any]] = []
    for session_date in reversed(session_dates):
        session = rth[rth.index.date == session_date]
        if session.empty:
            continue
        high = float(session["High"].max())
        low = float(session["Low"].min())
        close = float(session["Close"].iloc[-1])
        open_price = float(session["Open"].iloc[0])
        volume = float(session["Volume"].sum())
        change = close - open_price
        direction = "Up" if change > 0 else "Down" if change < 0 else "Flat"
        entries.append(
            {
                "date": session_date.isoformat(),
                "high": high,
                "low": low,
                "close": close,
                "range": high - low,
                "volume": volume,
                "change": change,
                "direction": direction,
            }
        )
        if len(entries) >= limit:
            break
    return list(reversed(entries))


def _build_bias_bundle(
    snapshot: "PremarketSnapshot",
    computation: PlanComputation,
    config: PlannerConfig,
    trend: Optional[Dict[str, Optional[str]]],
) -> Optional[Dict[str, Any]]:

    blockers: List[str] = []
    atr_value = getattr(snapshot, "atr14", None)
    if atr_value is None or (isinstance(atr_value, float) and math.isnan(atr_value)):
        blockers.append("ATR(14) unavailable")
    if getattr(snapshot, "missing_overnight", False):
        blockers.append("Overnight session missing (insufficient data)")

    latest_signal = None
    signals_path = str(config.signals_path) if config.signals_path else None
    try:
        latest_signal = _load_latest_signal(signals_path, snapshot.session_tz)
    except Exception:
        latest_signal = None

    age_hours: Optional[float] = None
    if latest_signal:
        try:
            now_dt = datetime.now(config.session_zone)
            age_hours = _signal_age_hours(latest_signal, now_dt, config.session_zone)
        except Exception:
            age_hours = None

    stale_signal = age_hours is not None and age_hours > MAX_SIGNAL_AGE_HOURS

    structure_hint: Optional[Dict[str, str]] = None
    try:
        structure_hint = _structure_bias_hint(_build_structure_summary(snapshot))
    except Exception:
        structure_hint = None

    direction_source = "signal"
    direction = "FLAT"
    if latest_signal and not blockers and not stale_signal:
        direction = latest_signal.side.upper()
    elif structure_hint:
        direction = structure_hint["direction"]
        direction_source = "structure"
    else:
        direction_source = "flat"

    confidence_text = "N/A"
    if direction_source == "structure" and structure_hint:
        confidence_text = structure_hint.get("confidence", "Structure")
    elif latest_signal:
        try:
            confidence_text = _format_confidence(latest_signal)
        except Exception:
            confidence_text = f"{latest_signal.confidence:.2f}"

    market_state = "Monitoring levels"
    if getattr(snapshot, "missing_overnight", False):
        market_state = "Not enough recent data"
    elif stale_signal:
        if structure_hint and direction_source == "structure":
            market_state = structure_hint.get("market_state", "Structure bias active")
        else:
            market_state = "Signals stale"
    elif blockers:
        market_state = "Blockers active"
    elif direction_source == "structure" and structure_hint:
        market_state = structure_hint.get("market_state", "Structure bias active")

    warnings = list(computation.warnings or ())
    if blockers:
        warnings.extend(blockers)

    trend_label = (trend or {}).get("label") or (trend or {}).get("state") or "Neutral"
    trend_detail = (trend or {}).get("detail")

    bias_bundle: Dict[str, Any] = {
        "direction": direction,
        "confidence": confidence_text or "—",
        "market_state": market_state,
        "warnings": warnings,
        "trend": trend_label,
        "trend_detail": trend_detail,
        "direction_source": direction_source,
    }

    if latest_signal:
        bias_bundle["signal"] = {
            "side": latest_signal.side,
            "grade": latest_signal.grade,
            "probability": latest_signal.probability,
            "confidence": latest_signal.confidence,
            "timestamp": latest_signal.timestamp.isoformat(),
            "age_hours": age_hours,
            "stale": stale_signal,
        }
    else:
        bias_bundle["signal"] = None

    return bias_bundle


def _format_price_label(label: Optional[str], price: Optional[float], decimals: int) -> Optional[str]:
    if price is None and not label:
        return None
    if price is None:
        return label
    fmt = f"{{:.{decimals}f}}"
    if not label:
        return fmt.format(price)
    return f"{label} · {fmt.format(price)}"


def _normalize_timestamp_label(raw: Optional[str], tz: timezone) -> Optional[str]:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(tz).strftime("Signal %I:%M %p").lstrip("0").replace(" 0", " ")


def _select_level(levels: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    if index < 0 or index >= len(levels):
        return None
    return levels[index]


def _derive_flow_bias(computation: PlanComputation, horizon_minutes: int = 240) -> Optional[Dict[str, Any]]:
    frame = computation.dataframe
    if frame is None or getattr(frame, "empty", True):
        return None
    latest = frame.index.max()
    if latest is None:
        return None
    window_start = latest - timedelta(minutes=horizon_minutes)
    window = frame[frame.index >= window_start]
    if window.empty or "Close" not in window:
        return None
    try:
        closes = window["Close"].astype(float).dropna()
    except Exception:
        return None
    if len(closes) < 12:
        return None
    first = float(closes.iloc[0])
    last = float(closes.iloc[-1])
    if first == 0 or not math.isfinite(first) or not math.isfinite(last):
        return None
    pct_change = (last / first - 1.0) * 100.0
    magnitude = abs(pct_change)
    if magnitude < 0.1:
        return None
    direction = "LONG" if pct_change > 0 else "SHORT"
    probability_pct = min(80.0, max(55.0, 55.0 + magnitude * 4.0))
    return {
        "direction": direction,
        "probabilityPct": probability_pct,
        "confidence": f"Flow ({pct_change:+.2f}% over 4h)",
        "market_state": f"Flow bias {pct_change:+.2f}% over 4h",
        "timestampLabel": f"Flow {pct_change:+.2f}% over 4h",
    }


def _build_trade_prediction(
    bias: Optional[Dict[str, Any]],
    supports: List[Dict[str, Any]],
    resistances: List[Dict[str, Any]],
    structure: Dict[str, Any],
    computation: PlanComputation,
    snapshot: "PremarketSnapshot",
    config: PlannerConfig,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    structure_hint = _structure_bias_hint(structure) or {}
    flow_hint = _derive_flow_bias(computation)
    direction = (bias or {}).get("direction")
    direction_source = (bias or {}).get("direction_source") or ("structure" if structure_hint else None)
    if not direction or direction.upper() not in {"LONG", "SHORT"}:
        direction = structure_hint.get("direction") or "Neutral"
        direction_source = direction_source or ("structure" if structure_hint else "none")

    signal = (bias or {}).get("signal") or {}
    signal_stale = signal.get("stale")
    if flow_hint and (direction_source != "signal" or signal_stale):
        direction = flow_hint.get("direction") or direction
        direction_source = "flow"

    direction_norm = direction.upper()
    if direction_norm not in {"LONG", "SHORT"}:
        direction_norm = "NEUTRAL"
    direction_class = (
        "long" if direction_norm == "LONG" else "short" if direction_norm == "SHORT" else "neutral"
    )

    probability_pct: Optional[float] = None
    probability_label = "—"
    signal_prob = signal.get("probability")
    if signal_prob is not None and not signal_stale:
        probability_pct = float(signal_prob) * 100.0
    elif direction_source == "flow" and flow_hint and flow_hint.get("probabilityPct") is not None:
        probability_pct = float(flow_hint["probabilityPct"])
    elif structure_hint.get("probabilityPct") is not None:
        probability_pct = float(structure_hint["probabilityPct"])
    elif direction_class == "long":
        probability_pct = 60.0
    elif direction_class == "short":
        probability_pct = 40.0
    else:
        probability_pct = 50.0
    probability_label = f"{probability_pct:.0f}%"

    confidence_label = (
        (bias or {}).get("confidence")
        or (flow_hint or {}).get("confidence")
        or structure_hint.get("confidence")
        or "—"
    )
    decimals = getattr(getattr(config, "output", None), "round_decimals", 2) or 2

    def _fallback_price(kind: str) -> Optional[float]:
        block = structure_hint if structure_hint else {}
        key = f"{kind}Price"
        return _safe_float(block.get(key))

    entry_area = None
    target_area = None
    invalidate_area = None
    if direction_class == "long":
        entry_area = _select_level(supports, 0)
        target_area = _select_level(resistances, 0) or _select_level(resistances, 1)
        invalidate_area = _select_level(supports, 1) or _select_level(resistances, 0)
    elif direction_class == "short":
        entry_area = _select_level(resistances, 0)
        target_area = _select_level(supports, 0) or _select_level(supports, 1)
        invalidate_area = _select_level(resistances, 1) or _select_level(supports, 0)

    entry_price = _safe_float((entry_area or {}).get("price")) or _fallback_price("entry")
    target_price = _safe_float((target_area or {}).get("price")) or _fallback_price("target")
    invalidate_price = _safe_float((invalidate_area or {}).get("price")) or _fallback_price("invalidate")

    entry_zone = _format_price_label(
        (entry_area or {}).get("label") or structure_hint.get("entryLabel"),
        entry_price,
        decimals,
    )
    target_zone = _format_price_label(
        (target_area or {}).get("label") or structure_hint.get("targetLabel"),
        target_price,
        decimals,
    )
    invalidate_zone = _format_price_label(
        (invalidate_area or {}).get("label") or structure_hint.get("invalidateLabel"),
        invalidate_price,
        decimals,
    )

    if direction_class == "long":
        headline = "Strength favored if buyers defend support"
        action = "Plan to buy dips back into support once structure confirms demand."
        steps = [
            f"Fade pullbacks into {entry_zone or 'support'}",
            f"Scale out near {target_zone or 'overhead resistance'}",
        ]
    elif direction_class == "short":
        headline = "Selling pressure expected into lower guardrails"
        action = "Favor short setups into resistance with stops above invalidation."
        steps = [
            f"Sell bounces into {entry_zone or 'resistance'}",
            f"Cover toward {target_zone or 'next support'}",
        ]
    else:
        headline = "Balanced tape — stay nimble"
        action = "Stay patient until structure or signals confirm a fresh bias."
        steps = ["Wait for a break of structure highs/lows before deploying risk."]

    signal_timestamp = (
        _normalize_timestamp_label(signal.get("timestamp"), config.timezone)
        if signal and not signal_stale
        else None
    )
    timestamp_label = (
        signal_timestamp
        or (flow_hint or {}).get("timestampLabel")
        or structure_hint.get("timestampLabel")
        or "Structure snapshot"
    )

    base_readiness = probability_pct
    market_state = (
        (bias or {}).get("market_state")
        or (flow_hint or {}).get("market_state")
        or structure_hint.get("market_state")
        or "Monitoring levels"
    )
    penalty = 0.0
    state_lower = market_state.lower()
    if "blocker" in state_lower:
        penalty += 25.0
    elif "stale" in state_lower:
        penalty += 15.0
    if (bias or {}).get("warnings"):
        penalty += 10.0
    readiness_pct = max(0.0, min(100.0, base_readiness - penalty))
    if signal_stale and direction_source in {"structure", "flow"}:
        readiness_pct = min(readiness_pct, base_readiness - 5.0)

    if readiness_pct >= 70:
        readiness_color, readiness_emoji = ("green", "🟢")
    elif readiness_pct >= 55:
        readiness_color, readiness_emoji = ("yellow", "🟡")
    else:
        readiness_color, readiness_emoji = ("red", "🔴")

    readiness = {
        "percent": readiness_pct,
        "color": readiness_color,
        "emoji": readiness_emoji,
        "label": f"{direction_norm.title()} Bias" if direction_class != "neutral" else "Neutral Bias",
        "detail": market_state,
        "source": direction_source,
    }

    prediction = {
        "direction": direction_norm.title(),
        "directionClass": direction_class,
        "directionSource": direction_source,
        "probabilityPct": probability_pct,
        "probabilityLabel": probability_label,
        "confidenceLabel": confidence_label,
        "headline": headline,
        "action": action,
        "entryZone": entry_zone,
        "target": target_zone,
        "invalidate": invalidate_zone,
        "entryPrice": entry_price,
        "targetPrice": target_price,
        "invalidatePrice": invalidate_price,
        "steps": steps,
        "timestampLabel": timestamp_label,
    }

    return prediction, readiness


def _compute_overnight_vwap(computation: PlanComputation) -> Optional[float]:
    frame = getattr(computation.window, "frame", None)
    if frame is None or getattr(frame, "empty", True):
        return None
    if "Volume" not in frame.columns or "Close" not in frame.columns:
        return None
    try:
        volumes = frame["Volume"].astype(float)
        if volumes.sum() == 0:
            return None
        closes = frame["Close"].astype(float)
        return float((closes * volumes).sum() / volumes.sum())
    except Exception:
        return None


def _compute_recent_extremes(computation: PlanComputation, hours: int = 24) -> Optional[Dict[str, float]]:
    frame = computation.dataframe
    if frame is None or getattr(frame, "empty", True):
        return None
    latest = frame.index.max()
    window_start = latest - timedelta(hours=hours)
    recent = frame[frame.index >= window_start]
    if recent.empty:
        recent = frame
    try:
        high = float(recent["High"].max())
        low = float(recent["Low"].min())
    except Exception:
        return None
    return {"high": high, "low": low}


def _slugify(value: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value))
    token = "-".join(segment for segment in token.split("-") if segment)
    return token or "level"


def _area_type(delta: Optional[float]) -> str:
    if delta is None:
        return "neutral"
    if delta > 0:
        return "resistance"
    if delta < 0:
        return "support"
    return "neutral"


def _datetime_to_unix(value: Optional[datetime]) -> Optional[int]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())
