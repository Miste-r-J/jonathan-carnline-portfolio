from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


NARRATION_DICT: Dict[str, Any] = {
    "header": (
        "{ts_local} | {instrument} {tf} | "
        "Last {price:.2f} ({bar_dir_word} {bar_change_ticks:+.0f}t) | "
        "Range {bar_range_ticks:.0f}t | Vol {rel_vol_word} | Volatility {vol_state_word}"
    ),
    "regime": {
        "trend_up": "Regime: **Bull trend** - higher highs/lows, pullbacks being bought.",
        "trend_down": "Regime: **Bear trend** - lower highs/lows, rallies being sold.",
        "range": "Regime: **Range/auction** - two-way trade, edges matter more than direction.",
        "chop": "Regime: **Chop** - overlapping bars, signal quality degraded; favor patience.",
    },
    "structure": {
        "above_vwap_strong": "Location: **Above VWAP** and holding - buyers in control; VWAP is support.",
        "above_vwap_weak": "Location: Above VWAP but **not clean** - watch for a VWAP retest.",
        "below_vwap_strong": "Location: **Below VWAP** and holding - sellers in control; VWAP is resistance.",
        "below_vwap_weak": "Location: Below VWAP but **not clean** - watch for a VWAP reclaim attempt.",
        "at_vwap": "Location: **At VWAP** - balanced; wait for acceptance or rejection.",
    },
    "ema": {
        "bull_stack": "EMA structure: **Bull stack** (fast > mid > slow) - dips tend to be buyable until broken.",
        "bear_stack": "EMA structure: **Bear stack** (fast < mid < slow) - pops tend to be sellable until reclaimed.",
        "tangled": "EMA structure: **Tangled** - mean-reversion environment; trend trades are lower confidence.",
    },
    "momentum": {
        "impulse_up": "Momentum: **Impulse up** - expanding bodies, follow-through strength.",
        "impulse_down": "Momentum: **Impulse down** - expanding bodies, downside continuation risk.",
        "stall": "Momentum: **Stalling** - energy fading; expect rotation or pullback.",
        "neutral": "Momentum: Neutral - no clear push; let levels decide.",
    },
    "range_behavior": {
        "expanding": "Range behavior: **Expanding** - volatility rising; widen stops / reduce size.",
        "contracting": "Range behavior: **Contracting** - compression; breakout risk building.",
        "normal": "Range behavior: Normal - conditions stable.",
    },
    "acceptance": {
        "accept_high": "Auction: **Acceptance above value** - price is holding higher; buyers comfortable.",
        "reject_high": "Auction: **Rejection from highs** - supply hit; watch a move back into value.",
        "accept_low": "Auction: **Acceptance below value** - sellers comfortable; downside continuation risk.",
        "reject_low": "Auction: **Rejection from lows** - demand stepped in; watch a move back into value.",
        "none": "Auction: No clear acceptance/rejection signal yet.",
    },
    "key_levels": {
        "vwap": "Key level: VWAP {vwap:.2f} ({vwap_delta_ticks:+.0f}t away).",
        "value": "Value: VAH {vah:.2f} / VAL {val:.2f}.",
        "pdh_pdl": "Prior day: PDH {pdh:.2f} / PDL {pdl:.2f}.",
        "or": "Opening range: ORH {orh:.2f} / ORL {orl:.2f}.",
        "session_high_low": "Session: High {sess_high:.2f} / Low {sess_low:.2f}.",
    },
    "risk": {
        "high": "Risk note: **High** (fast tape / big ranges). Be selective; avoid chasing.",
        "medium": "Risk note: Medium - good trades exist, but demand clean setups.",
        "low": "Risk note: Low - orderly; structure tends to respect levels.",
    },
    "what_to_watch": {
        "bull": "Watch next: buyers want to **hold above VWAP/EMAs** and push toward {next_up_level_name}.",
        "bear": "Watch next: sellers want to **hold below VWAP/EMAs** and press toward {next_dn_level_name}.",
        "range": "Watch next: wait for **edge tests** (VAH/VAL) or a **VWAP acceptance** to define direction.",
    },
    "model": {
        "prob_line": "Model: P(long)={p_long:.2f} | P(short)={p_short:.2f} | edge={edge_word} | signal={signal_word}",
        "edge_strong_long": "Model bias: **Strong long** - only invalidate on structure break / VWAP loss.",
        "edge_weak_long": "Model bias: Mild long - prefer pullback entries; avoid breakouts into resistance.",
        "edge_neutral": "Model bias: Neutral - let the market confirm; reduce frequency.",
        "edge_weak_short": "Model bias: Mild short - prefer failed pops; avoid selling into major demand.",
        "edge_strong_short": "Model bias: **Strong short** - only invalidate on reclaim of key levels.",
    },
    "footer": (
        "Summary: {regime_word}, {location_word}, {ema_word}, {momentum_word}. "
        "{actionable_sentence}"
    ),
}


@dataclass
class MarketSnapshot:
    ts_local: str
    instrument: str
    tf: str
    price: float
    bar_open: float
    bar_high: float
    bar_low: float
    bar_close: float
    tick_size: float
    vwap: Optional[float] = None
    vah: Optional[float] = None
    val: Optional[float] = None
    pdh: Optional[float] = None
    pdl: Optional[float] = None
    orh: Optional[float] = None
    orl: Optional[float] = None
    sess_high: Optional[float] = None
    sess_low: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_mid: Optional[float] = None
    ema_slow: Optional[float] = None
    atr_ticks: Optional[float] = None
    bar_range_ticks: Optional[float] = None
    rel_vol: Optional[float] = None
    vol_of_vol: Optional[float] = None
    regime: Optional[str] = None
    momentum: Optional[str] = None
    range_behavior: Optional[str] = None
    acceptance: Optional[str] = None
    p_long: Optional[float] = None
    p_short: Optional[float] = None
    signal: Optional[int] = None


def _ticks(delta: float, tick_size: float) -> float:
    if tick_size <= 0:
        return delta
    return delta / tick_size


def render_market_narration(s: MarketSnapshot) -> str:
    bar_change_ticks = _ticks(s.bar_close - s.bar_open, s.tick_size)
    bar_dir_word = "up" if bar_change_ticks > 0 else ("down" if bar_change_ticks < 0 else "flat")

    bar_range_ticks = s.bar_range_ticks
    if bar_range_ticks is None:
        bar_range_ticks = _ticks(s.bar_high - s.bar_low, s.tick_size)

    rel_vol_word = "unknown"
    if s.rel_vol is not None:
        if s.rel_vol >= 1.5:
            rel_vol_word = "high"
        elif s.rel_vol <= 0.7:
            rel_vol_word = "low"
        else:
            rel_vol_word = "normal"

    vol_state_word = "unknown"
    if s.atr_ticks is not None:
        if s.atr_ticks >= 18:
            vol_state_word = "hot"
        elif s.atr_ticks <= 10:
            vol_state_word = "calm"
        else:
            vol_state_word = "balanced"

    location_key = "at_vwap"
    vwap_delta_ticks = 0.0
    if s.vwap is not None:
        vwap_delta_ticks = _ticks(s.price - s.vwap, s.tick_size)
        if abs(vwap_delta_ticks) <= 1.0:
            location_key = "at_vwap"
        elif vwap_delta_ticks > 1.0:
            location_key = "above_vwap_strong" if vwap_delta_ticks >= 6 else "above_vwap_weak"
        else:
            location_key = "below_vwap_strong" if vwap_delta_ticks <= -6 else "below_vwap_weak"

    ema_key = "tangled"
    if s.ema_fast is not None and s.ema_mid is not None and s.ema_slow is not None:
        if s.ema_fast > s.ema_mid > s.ema_slow:
            ema_key = "bull_stack"
        elif s.ema_fast < s.ema_mid < s.ema_slow:
            ema_key = "bear_stack"

    regime_key = s.regime or "range"
    regime_line = NARRATION_DICT["regime"].get(regime_key, NARRATION_DICT["regime"]["range"])

    mom_key = s.momentum or "neutral"
    mom_line = NARRATION_DICT["momentum"].get(mom_key, NARRATION_DICT["momentum"]["neutral"])

    rb_key = s.range_behavior or "normal"
    rb_line = NARRATION_DICT["range_behavior"].get(rb_key, NARRATION_DICT["range_behavior"]["normal"])

    acc_key = s.acceptance or "none"
    acc_line = NARRATION_DICT["acceptance"].get(acc_key, NARRATION_DICT["acceptance"]["none"])

    risk_key = "medium"
    if (s.atr_ticks is not None and s.atr_ticks >= 18) or (bar_range_ticks is not None and bar_range_ticks >= 20):
        risk_key = "high"
    elif (s.atr_ticks is not None and s.atr_ticks <= 10) and (bar_range_ticks is not None and bar_range_ticks <= 12):
        risk_key = "low"
    risk_line = NARRATION_DICT["risk"][risk_key]

    level_lines: List[str] = []
    if s.vwap is not None:
        level_lines.append(
            NARRATION_DICT["key_levels"]["vwap"].format(vwap=s.vwap, vwap_delta_ticks=vwap_delta_ticks)
        )
    if s.vah is not None and s.val is not None:
        level_lines.append(NARRATION_DICT["key_levels"]["value"].format(vah=s.vah, val=s.val))
    if s.pdh is not None and s.pdl is not None:
        level_lines.append(NARRATION_DICT["key_levels"]["pdh_pdl"].format(pdh=s.pdh, pdl=s.pdl))
    if s.orh is not None and s.orl is not None:
        level_lines.append(NARRATION_DICT["key_levels"]["or"].format(orh=s.orh, orl=s.orl))
    if s.sess_high is not None and s.sess_low is not None:
        level_lines.append(
            NARRATION_DICT["key_levels"]["session_high_low"].format(sess_high=s.sess_high, sess_low=s.sess_low)
        )

    model_lines: List[str] = []
    edge_word = "n/a"
    signal_word = "none"
    if s.signal is not None:
        signal_word = "LONG" if s.signal > 0 else ("SHORT" if s.signal < 0 else "FLAT")
    if s.p_long is not None and s.p_short is not None:
        edge = s.p_long - s.p_short
        if edge >= 0.20:
            edge_word = "strong_long"
        elif edge >= 0.08:
            edge_word = "weak_long"
        elif edge <= -0.20:
            edge_word = "strong_short"
        elif edge <= -0.08:
            edge_word = "weak_short"
        else:
            edge_word = "neutral"
        model_lines.append(
            NARRATION_DICT["model"]["prob_line"].format(
                p_long=s.p_long, p_short=s.p_short, edge_word=edge_word, signal_word=signal_word
            )
        )
        model_lines.append(NARRATION_DICT["model"].get(f"edge_{edge_word}", NARRATION_DICT["model"]["edge_neutral"]))

    next_up_level_name = "next resistance"
    next_dn_level_name = "next support"
    if s.vah is not None:
        next_up_level_name = f"VAH {s.vah:.2f}"
    if s.val is not None:
        next_dn_level_name = f"VAL {s.val:.2f}"

    if regime_key in ("trend_up",) or location_key.startswith("above_vwap"):
        watch_line = NARRATION_DICT["what_to_watch"]["bull"].format(next_up_level_name=next_up_level_name)
    elif regime_key in ("trend_down",) or location_key.startswith("below_vwap"):
        watch_line = NARRATION_DICT["what_to_watch"]["bear"].format(next_dn_level_name=next_dn_level_name)
    else:
        watch_line = NARRATION_DICT["what_to_watch"]["range"]

    header = NARRATION_DICT["header"].format(
        ts_local=s.ts_local,
        instrument=s.instrument,
        tf=s.tf,
        price=s.price,
        bar_dir_word=bar_dir_word,
        bar_change_ticks=bar_change_ticks,
        bar_range_ticks=bar_range_ticks,
        rel_vol_word=rel_vol_word,
        vol_state_word=vol_state_word,
    )

    location_line = NARRATION_DICT["structure"][location_key]
    ema_line = NARRATION_DICT["ema"][ema_key]

    footer = NARRATION_DICT["footer"].format(
        regime_word=regime_key,
        location_word=location_key,
        ema_word=ema_key,
        momentum_word=mom_key,
        actionable_sentence=watch_line,
    )

    parts = [
        header,
        regime_line,
        location_line,
        ema_line,
        mom_line,
        rb_line,
        acc_line,
        *level_lines,
        *model_lines,
        risk_line,
        footer,
    ]

    return "\n".join(parts)
