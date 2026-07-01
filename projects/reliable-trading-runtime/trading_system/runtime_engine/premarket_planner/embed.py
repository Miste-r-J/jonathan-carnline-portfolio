from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple

from zoneinfo import ZoneInfo

from .analysis import (
    PremarketSnapshot,
    _determine_bias,
    _format_confidence,
    _format_pct,
    _format_price,
    _format_volume,
    _load_latest_signal,
    _signal_age_hours,
)
from .structure import StructureDetector

from trading_system.runtime_engine.modeling.config import INSTRUMENTS
from .config import PlannerConfig
from .core import PlanComputation, WindowStatus


@dataclass(frozen=True)
class EmbedField:
    name: str
    value: str
    inline: bool = False


@dataclass(frozen=True)
class PlanEmbed:
    title: str
    description: str
    color: int
    fields: Tuple[EmbedField, ...]
    footer: str
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "color": self.color,
            "fields": [
                {"name": f.name, "value": f.value, "inline": f.inline} for f in self.fields
            ],
            "footer": {"text": self.footer},
            "timestamp": self.timestamp.isoformat(),
        }

    def to_discord_embed(self):
        try:
            import discord  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("discord.py is required to build Discord embeds") from exc

        embed = discord.Embed(
            title=self.title,
            description=self.description,
            color=self.color,
            timestamp=self.timestamp,
        )
        for field in self.fields:
            embed.add_field(name=field.name, value=field.value, inline=field.inline)
        embed.set_footer(text=self.footer)
        return embed


@dataclass(frozen=True)
class PlanPayload:
    content: str
    embed: PlanEmbed
    warnings: Tuple[str, ...]


def build_plan_payload(
    config: PlannerConfig,
    computation: PlanComputation,
) -> PlanPayload:
    snapshot = computation.snapshot
    session_tz = ZoneInfo(snapshot.session_tz)
    display_tz = _resolve_display_zone(config, computation)
    last_update = snapshot.last_update.astimezone(display_tz)
    window = computation.window

    blockers: List[str] = []
    if snapshot.atr14 is None:
        blockers.append("ATR(14) unavailable")
    if snapshot.missing_overnight:
        blockers.append("Overnight session missing (insufficient data)")

    signal_path = str(config.signals_path) if config.signals_path else None
    signal_snapshot = _load_latest_signal(signal_path, snapshot.session_tz)
    now_dt = datetime.now(session_tz)
    signal_age_hours = _signal_age_hours(signal_snapshot, now_dt, session_tz)
    stale_signal = signal_age_hours is not None and signal_age_hours > 18.0

    bias = _determine_bias(signal_snapshot, blockers)
    if stale_signal:
        bias = "FLAT"
    confidence = _format_confidence(signal_snapshot)

    if snapshot.missing_overnight:
        market_state = "Not enough recent data"
    elif stale_signal:
        market_state = "Signals stale"
    else:
        market_state = "Monitoring levels"

    atr_value = "N/A" if snapshot.atr14 is None else f"{snapshot.atr14:.2f}"

    trend_label, trend_detail = _determine_trend(computation)

    description = _build_description(
        snapshot,
        session_tz,
        config.metrics.atr_len,
        bias=bias,
        confidence=confidence,
        market_state=market_state,
        atr_value=atr_value,
        trend=trend_label,
        trend_detail=trend_detail,
    )
    fields = _build_fields(snapshot, window, computation, display_tz, blockers, config)

    warnings_list = list(computation.warnings)
    if stale_signal and signal_snapshot is not None:
        warnings_list.append("Latest signal is stale (>18h); treating bias as FLAT.")
    warnings = tuple(warnings_list)
    if warnings:
        warning_text = "\n".join(f"• {item}" for item in warnings)
        fields += (EmbedField(name="Warnings", value=warning_text, inline=False),)

    footer = f"Updated {last_update.strftime('%m/%d/%y • %I:%M %p %Z')}"
    title = f"📋 Premarket Plan — {config.instrument.upper()}"

    payload = PlanEmbed(
        title=title,
        description=description,
        color=_resolve_color(computation),
        fields=fields,
        footer=footer,
        timestamp=last_update,
    )

    content = f"{config.instrument.upper()} premarket snapshot for {snapshot.analysis_date:%Y-%m-%d}"
    return PlanPayload(content=content, embed=payload, warnings=warnings)


def _build_description(
    snapshot: PremarketSnapshot,
    tz: ZoneInfo,
    atr_len: int,
    *,
    bias: str,
    confidence: str,
    market_state: str,
    atr_value: str,
    trend: str,
    trend_detail: Optional[str],
) -> str:
    gap_points = _format_price(snapshot.gap_points)
    gap_pct = _format_pct(snapshot.gap_pct)
    current_price = _format_price(snapshot.current_price)
    analysis_date = snapshot.analysis_date.strftime("%A, %b %d")
    trend_line = f"**Trend:** {trend}"
    if trend_detail:
        trend_line = f"{trend_line} — {trend_detail}"
    return "\n".join(
        [
            f"**Bias:** {bias}",
            f"**Confidence:** {confidence}",
            f"**Market State:** {market_state}",
            f"**ATR({atr_len}):** {atr_value}",
            trend_line,
            "",
            f"**Session:** {analysis_date} ({tz.key})",
            f"**Current:** {current_price}",
            f"**Gap:** {gap_points} ({gap_pct})",
        ]
    )


def _build_fields(
    snapshot: PremarketSnapshot,
    window: WindowStatus,
    computation: PlanComputation,
    display_tz: ZoneInfo,
    blockers: List[str],
    config: PlannerConfig,
) -> Tuple[EmbedField, ...]:
    prev_fields = "\n".join(
        [
            f"• Close: {_format_price(snapshot.prev_close)}",
            f"• High: {_format_price(snapshot.prev_high)}",
            f"• Low: {_format_price(snapshot.prev_low)}",
            f"• Range: {_format_price(snapshot.prev_range)}",
            f"• VWAP: {_format_price(snapshot.prev_session_vwap)}",
        ]
    )
    overnight_fields = "\n".join(
        [
            f"• High: {_format_price(snapshot.overnight_high)}",
            f"• Low: {_format_price(snapshot.overnight_low)}",
            f"• Range: {_format_price(snapshot.overnight_range)}",
            f"• Volume: {_format_volume(snapshot.premarket_volume)}",
        ]
    )
    display_label = getattr(display_tz, "key", str(display_tz))
    window_lines = [
        f"• Bars: {window.bars} (initial {window.initial_bars})",
        f"• Window ({display_label}): "
        f"{window.start.astimezone(display_tz).strftime('%m/%d %H:%M')} → "
        f"{window.end.astimezone(display_tz).strftime('%m/%d %H:%M')}",
        f"• Source TZ: {computation.source_timezone}",
        f"• VWAP: {computation.vwap_source}",
    ]
    if window.used_fallback and window.fallback_reason:
        window_lines.append(f"• Fallback: {window.fallback_reason}")

    fields: Tuple[EmbedField, ...] = (
        EmbedField(name="Previous Session", value=prev_fields, inline=True),
        EmbedField(name="Overnight Stats", value=overnight_fields, inline=True),
        EmbedField(name="Data Quality", value="\n".join(window_lines), inline=False),
    )

    structure_text = _build_structure_summary(snapshot)
    if structure_text:
        fields += (EmbedField(name="Market Structure", value=structure_text, inline=False),)

    areas_text = _build_areas_of_interest(snapshot, computation)
    if areas_text:
        fields += (EmbedField(name="Areas of Interest", value=areas_text, inline=False),)

    profile_text = _build_volume_profile(snapshot, computation, config)
    if profile_text:
        fields += (EmbedField(name="Volume Profile", value=profile_text, inline=False),)

    bos_text = _build_break_of_structure(computation, config, display_tz)
    if bos_text:
        fields += (EmbedField(name="Break & Retest", value=bos_text, inline=False),)

    if blockers:
        blockers_text = "\n".join(f"• {item}" for item in blockers)
        fields += (EmbedField(name="Blockers", value=blockers_text, inline=False),)

    return fields


def _resolve_color(computation: PlanComputation) -> int:
    return 0x2ecc71 if not computation.warnings else 0xf39c12


def _determine_trend(computation: PlanComputation) -> Tuple[str, Optional[str]]:
    df_local = computation.dataframe
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


def _build_structure_summary(snapshot: PremarketSnapshot) -> str:
    prev_high_ok = _is_valid_number(snapshot.prev_high)
    prev_low_ok = _is_valid_number(snapshot.prev_low)
    overnight_high_ok = _is_valid_number(snapshot.overnight_high)
    overnight_low_ok = _is_valid_number(snapshot.overnight_low)

    lines: List[str] = []
    if prev_high_ok and overnight_high_ok:
        lines.append(
            f"• Highs: {_classify_high(snapshot.overnight_high, snapshot.prev_high)} "
            f"({_format_price(snapshot.overnight_high)} vs {_format_price(snapshot.prev_high)})"
        )
    if prev_low_ok and overnight_low_ok:
        lines.append(
            f"• Lows: {_classify_low(snapshot.overnight_low, snapshot.prev_low)} "
            f"({_format_price(snapshot.overnight_low)} vs {_format_price(snapshot.prev_low)})"
        )

    if not lines:
        return ""
    return "\n".join(lines)


def _classify_high(current_high: float, reference_high: float) -> str:
    if current_high > reference_high:
        return "Higher high"
    if current_high < reference_high:
        return "Lower high"
    return "Equal high"


def _classify_low(current_low: float, reference_low: float) -> str:
    if current_low > reference_low:
        return "Higher low"
    if current_low < reference_low:
        return "Lower low"
    return "Equal low"


def _is_valid_number(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _build_areas_of_interest(
    snapshot: PremarketSnapshot,
    computation: PlanComputation,
) -> str:
    current_price = float(snapshot.current_price) if _is_valid_number(snapshot.current_price) else None

    entries: List[Tuple[float, str, float, Optional[float]]] = []

    def register(label: str, raw_value: Optional[float]) -> None:
        if not _is_valid_number(raw_value):
            return
        value = float(raw_value)  # type: ignore[arg-type]
        delta = None if current_price is None else value - current_price
        distance = 0.0 if delta is None else abs(delta)
        entries.append((distance, label, value, delta))

    register("Prior RTH High", snapshot.prev_high)
    register("Prior RTH Low", snapshot.prev_low)
    register("Prior Close (Gap Fill)", snapshot.prev_close)
    register("Prior RTH VWAP", snapshot.prev_session_vwap)
    register("Overnight High", snapshot.overnight_high)
    register("Overnight Low", snapshot.overnight_low)

    if _is_valid_number(snapshot.overnight_high) and _is_valid_number(snapshot.overnight_low):
        overnight_mid = (float(snapshot.overnight_high) + float(snapshot.overnight_low)) / 2.0
        register("Overnight Mid", overnight_mid)

    window_frame = computation.window.frame
    if hasattr(window_frame, "empty") and not window_frame.empty:
        try:
            volume_sum = float(window_frame["Volume"].sum())
            if volume_sum > 0:
                overnight_vwap = float((window_frame["Close"] * window_frame["Volume"]).sum() / volume_sum)
                register("Overnight VWAP", overnight_vwap)
        except Exception:
            pass

    local_df = computation.dataframe
    if hasattr(local_df, "empty") and not local_df.empty:
        try:
            latest = local_df.index.max()
            window_start = latest - timedelta(hours=24)
            recent = local_df[local_df.index >= window_start]
            if not recent.empty:
                register("24h High", float(recent["High"].max()))
                register("24h Low", float(recent["Low"].min()))
        except Exception:
            pass

    if not entries:
        return ""

    entries.sort(key=lambda item: (item[0], item[2]))
    top = entries[:4]

    lines: List[str] = []
    for _, label, value, delta in top:
        if delta is None:
            lines.append(f"• {label}: {_format_price(value)}")
        else:
            lines.append(f"• {label}: {_format_price(value)} (Δ {delta:+.2f})")
    return "\n".join(lines)


def _build_volume_profile(
    snapshot: PremarketSnapshot,
    computation: PlanComputation,
    config: PlannerConfig,
    max_levels: int = 3,
) -> str:
    frame = computation.window.frame
    if frame is None or getattr(frame, "empty", True):
        return ""

    if "Volume" not in frame.columns:
        return ""

    price_cols = [col for col in ("Open", "High", "Low", "Close") if col in frame.columns]
    if not price_cols:
        return ""

    try:
        prices = frame[price_cols].astype(float).mean(axis=1)
        volumes = frame["Volume"].astype(float)
    except Exception:
        return ""

    mask = volumes > 0
    prices = prices[mask]
    volumes = volumes[mask]
    if prices.empty:
        return ""

    instrument = config.instrument.upper()
    tick_size = 0.25
    spec = INSTRUMENTS.get(instrument)
    if spec is not None and getattr(spec, "tick_size", None):
        tick_size = float(spec.tick_size) or tick_size

    if tick_size <= 0:
        tick_size = 0.25

    bucket_indices = (prices / tick_size).round().astype(int)
    buckets: dict[float, float] = {}
    for price_idx, volume in zip(bucket_indices, volumes):
        bucket_price = price_idx * tick_size
        buckets[bucket_price] = buckets.get(bucket_price, 0.0) + float(volume)

    if not buckets:
        return ""

    total_volume = sum(buckets.values())
    sorted_levels = sorted(buckets.items(), key=lambda item: item[1], reverse=True)[:max_levels]

    current_price = float(snapshot.current_price) if _is_valid_number(snapshot.current_price) else None

    lines: List[str] = []
    for price, volume in sorted_levels:
        pct = (volume / total_volume * 100.0) if total_volume else 0.0
        line = f"• {_format_price(price)}: {volume:,.0f} ({pct:.1f}%)"
        if current_price is not None:
            line = f"{line} (Δ {price - current_price:+.2f})"
        lines.append(line)

    return "\n".join(lines)


def _build_break_of_structure(
    computation: PlanComputation,
    config: PlannerConfig,
    display_tz: ZoneInfo,
) -> str:
    frame = computation.dataframe
    if frame is None or getattr(frame, "empty", True):
        return ""

    try:
        detector = StructureDetector(instrument=config.instrument)
        report = detector.analyze(frame)
    except Exception as exc:  # pragma: no cover - defensive
        return f"• Unable to evaluate structure: {exc}"

    return report.to_markdown(display_tz)


def _resolve_display_zone(
    config: PlannerConfig,
    computation: PlanComputation,
) -> ZoneInfo:
    candidates: List[Optional[str]] = [
        config.display_timezone,
        computation.source_timezone,
        config.session_tz,
    ]
    for name in candidates:
        if not name:
            continue
        try:
            return ZoneInfo(name)
        except Exception:
            continue
    return config.session_zone
