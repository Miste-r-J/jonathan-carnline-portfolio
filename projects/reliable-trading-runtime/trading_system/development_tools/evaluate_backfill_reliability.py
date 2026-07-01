import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


READINESS_ARTIFACT_BLOCKERS = {
    "startup_resync",
    "blocked_not_armed",
    "blocked_stale_bar",
    "blocked_past_bar_emit",
    "not_armed",
}


@dataclass
class CostModel:
    commission_per_contract_per_side: float
    slippage_ticks_per_side: float
    tick_size: float
    tick_value: float


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null", "nan", "nat", "--"}:
            return None
        return float(text)
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan", "nat", "undefined"} else text


def _phase_label(value: Any) -> str:
    text = _norm_text(value).upper()
    return text or "UNKNOWN"


def _instrument_from_run(run_dir: Path) -> str:
    status_path = run_dir / "status.json"
    if status_path.exists():
        try:
            obj = json.loads(status_path.read_text(encoding="utf-8", errors="replace"))
            for key in ("instrument", "instrument_alias", "exec_instrument_key"):
                val = _norm_text(obj.get(key))
                if val:
                    return val
        except Exception:
            pass
    return "ES"


def _point_value_for_instrument(instrument: str) -> float:
    upper = _norm_text(instrument).upper()
    if "MES" in upper:
        return 5.0
    if "MNQ" in upper:
        return 2.0
    if "MYM" in upper:
        return 0.5
    if "M2K" in upper:
        return 5.0
    if "NQ" in upper:
        return 20.0
    if "YM" in upper:
        return 5.0
    if "RTY" in upper:
        return 50.0
    return 50.0


def _tick_size_for_instrument(instrument: str) -> float:
    upper = _norm_text(instrument).upper()
    if "YM" in upper or "MYM" in upper:
        return 1.0
    return 0.25


def _tick_value(point_value: float, tick_size: float) -> float:
    if tick_size <= 0:
        return 0.0
    return float(point_value) * float(tick_size)


def _build_cost_model(run_dir: Path, profile: str) -> CostModel:
    instrument = _instrument_from_run(run_dir)
    point_value = _point_value_for_instrument(instrument)
    tick_size = _tick_size_for_instrument(instrument)
    tick_value = _tick_value(point_value, tick_size)
    if profile != "current_default":
        raise ValueError(f"Unsupported cost profile: {profile}")
    return CostModel(
        commission_per_contract_per_side=2.0,
        slippage_ticks_per_side=1.0,
        tick_size=tick_size,
        tick_value=tick_value,
    )


def _load_state_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=2):
            dt = _norm_text(row.get("datetime"))
            if not dt:
                continue
            rows.append(
                {
                    "line": idx,
                    "datetime": dt,
                    "action": _norm_text(row.get("action")).upper(),
                    "side": _norm_text(row.get("side")).upper(),
                    "price": _safe_float(row.get("price")),
                    "size": _safe_float(row.get("size")) or 1.0,
                    "prob": _safe_float(row.get("prob")),
                    "entry_conf": _safe_float(row.get("entry_conf")),
                    "hold_conf": _safe_float(row.get("hold_conf")),
                }
            )
    return rows


def _load_gating_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for idx, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "header":
            continue
        blocked = obj.get("blocked_by") if isinstance(obj.get("blocked_by"), list) else []
        phase2 = obj.get("phase2") if isinstance(obj.get("phase2"), dict) else {}
        out.append(
            {
                "line": idx,
                "bar_ts": _norm_text(obj.get("bar_ts") or obj.get("datetime") or obj.get("ts")),
                "phase": _phase_label(obj.get("phase")),
                "action": _norm_text(obj.get("action")).upper(),
                "side": _norm_text(obj.get("side")).upper(),
                "blocked_by": [str(x) for x in blocked if _norm_text(x)],
                "reason_detail": _norm_text(obj.get("reason_detail")),
                "pred_p_long": _safe_float(obj.get("pred_p_long") if obj.get("pred_p_long") is not None else obj.get("prob")),
                "pred_p_short": _safe_float(obj.get("pred_p_short")),
                "setup_prob": _safe_float(phase2.get("setup_prob")),
                "short_prob": _safe_float(phase2.get("short_prob")),
                "direction_signal": _safe_int(phase2.get("direction_signal"), default=0),
            }
        )
    return out


def _gating_by_bar_ts(gating_rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in gating_rows:
        ts = _norm_text(row.get("bar_ts"))
        if not ts:
            continue
        merged[ts] = row
    return merged


def _join_state_with_gating(
    state_rows: Iterable[Dict[str, Any]],
    gating_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_ts = _gating_by_bar_ts(gating_rows)
    out: List[Dict[str, Any]] = []
    for row in state_rows:
        g = by_ts.get(_norm_text(row.get("datetime")))
        out.append(
            {
                **row,
                "phase": _phase_label((g or {}).get("phase")),
                "blocked_by": list((g or {}).get("blocked_by") or []),
                "reason_detail": _norm_text((g or {}).get("reason_detail")),
                "setup_prob": _safe_float((g or {}).get("setup_prob")),
                "pred_p_long": _safe_float((g or {}).get("pred_p_long")),
                "short_prob": _safe_float((g or {}).get("short_prob")),
            }
        )
    return out


def _blocker_bucket(reason: str) -> str:
    r = _norm_text(reason).lower()
    if not r:
        return "unblocked"
    if "prob" in r:
        return "prob"
    if "startup_resync" in r:
        return "startup_resync"
    if "stale" in r:
        return "stale"
    if "regime" in r or "countertrend" in r:
        return "regime"
    if "setup" in r:
        return "setup"
    if "stop" in r:
        return "stop"
    return "other"


def _blocked_reason_set(row: Mapping[str, Any]) -> List[str]:
    blocked = row.get("blocked_by") if isinstance(row.get("blocked_by"), list) else []
    reasons = [str(x) for x in blocked if _norm_text(x)]
    rd = _norm_text(row.get("reason_detail"))
    if rd:
        reasons.append(rd)
    return reasons


def _is_strategy_eligible(row: Mapping[str, Any]) -> bool:
    reasons = _blocked_reason_set(row)
    for reason in reasons:
        reason_l = _norm_text(reason).lower()
        if (
            reason_l in READINESS_ARTIFACT_BLOCKERS
            or "startup_resync" in reason_l
            or "blocked_not_armed" in reason_l
            or "blocked_stale_bar" in reason_l
            or "not_armed" in reason_l
            or "blocked_past_bar_emit" in reason_l
        ):
            continue
        return False
    return True


def _is_execution_eligible(row: Mapping[str, Any]) -> bool:
    blocked = row.get("blocked_by") if isinstance(row.get("blocked_by"), list) else []
    return len([x for x in blocked if _norm_text(x)]) == 0


def _trade_cost_usd(cost: CostModel, qty: float) -> float:
    per_side = float(cost.commission_per_contract_per_side) + float(cost.slippage_ticks_per_side) * float(cost.tick_value)
    return per_side * 2.0 * float(qty)


def _gross_pnl_usd(side: str, entry_price: float, exit_price: float, qty: float, point_value: float) -> float:
    if side == "LONG":
        points = float(exit_price) - float(entry_price)
    elif side == "SHORT":
        points = float(entry_price) - float(exit_price)
    else:
        points = 0.0
    return points * float(point_value) * float(qty)


def _reconstruct_trades_from_actions(
    rows: Iterable[Dict[str, Any]],
    *,
    phase: str,
    point_value: float,
    cost: CostModel,
) -> Dict[str, Any]:
    target_phase = _phase_label(phase)
    seq = [r for r in rows if _phase_label(r.get("phase")) == target_phase]
    seq = sorted(seq, key=lambda r: _norm_text(r.get("datetime")))

    pos: Optional[Dict[str, Any]] = None
    trades: List[Dict[str, Any]] = []
    ignored_open_while_in_position = 0
    close_without_position = 0

    def _close_position(close_row: Mapping[str, Any], reason: str) -> None:
        nonlocal pos
        if pos is None:
            return
        exit_price = _safe_float(close_row.get("price"))
        if exit_price is None:
            return
        qty = float(pos.get("qty") or 1.0)
        gross = _gross_pnl_usd(str(pos.get("side") or ""), float(pos.get("entry_price") or 0.0), float(exit_price), qty, point_value)
        cost_usd = _trade_cost_usd(cost, qty)
        net = gross - cost_usd
        trades.append(
            {
                "entry_ts": pos.get("entry_ts"),
                "exit_ts": close_row.get("datetime"),
                "side": pos.get("side"),
                "qty": qty,
                "entry_price": pos.get("entry_price"),
                "exit_price": exit_price,
                "gross_pnl_usd": gross,
                "cost_usd": cost_usd,
                "net_pnl_usd": net,
                "exit_reason": reason,
            }
        )
        pos = None

    for row in seq:
        action = _norm_text(row.get("action")).upper()
        side = _norm_text(row.get("side")).upper()
        price = _safe_float(row.get("price"))
        qty = float(_safe_float(row.get("size")) or 1.0)
        if action in {"OPEN", "FLIP"} and side in {"LONG", "SHORT"} and price is not None:
            if pos is None:
                pos = {
                    "entry_ts": row.get("datetime"),
                    "side": side,
                    "entry_price": float(price),
                    "qty": qty,
                }
                continue
            if action == "FLIP" and side != _norm_text(pos.get("side")).upper():
                _close_position(row, reason="flip")
                pos = {
                    "entry_ts": row.get("datetime"),
                    "side": side,
                    "entry_price": float(price),
                    "qty": qty,
                }
            else:
                ignored_open_while_in_position += 1
        elif action in {"CLOSE", "FLAT", "FLATTEN"}:
            if pos is None:
                close_without_position += 1
            else:
                _close_position(row, reason="close")

    if pos is not None:
        last = seq[-1] if seq else None
        if last is not None and _safe_float(last.get("price")) is not None:
            _close_position(last, reason="forced_end_of_phase")

    daily_net: Dict[str, float] = defaultdict(float)
    for trade in trades:
        exit_ts = _norm_text(trade.get("exit_ts"))
        day = exit_ts.split("T", 1)[0] if "T" in exit_ts else exit_ts[:10]
        if day:
            daily_net[day] += float(trade.get("net_pnl_usd") or 0.0)

    return {
        "trades": trades,
        "trade_count": len(trades),
        "net_pnl_usd": float(sum(float(t.get("net_pnl_usd") or 0.0) for t in trades)),
        "gross_pnl_usd": float(sum(float(t.get("gross_pnl_usd") or 0.0) for t in trades)),
        "cost_usd": float(sum(float(t.get("cost_usd") or 0.0) for t in trades)),
        "daily_net": dict(sorted(daily_net.items())),
        "ignored_open_while_in_position": int(ignored_open_while_in_position),
        "close_without_position": int(close_without_position),
    }


def _daily_pass_rate(
    daily_net: Mapping[str, float],
    *,
    target_min: float,
    target_max: float,
    window_days: int,
) -> Dict[str, Any]:
    ordered = sorted((k, float(v)) for k, v in daily_net.items())
    if window_days > 0:
        ordered = ordered[-int(window_days) :]
    count = len(ordered)
    if count == 0:
        return {
            "window_days": int(window_days),
            "days_considered": 0,
            "pass_days": 0,
            "pass_rate": None,
            "target_min": float(target_min),
            "target_max": float(target_max),
        }
    pass_days = sum(1 for _, pnl in ordered if float(target_min) <= pnl <= float(target_max))
    return {
        "window_days": int(window_days),
        "days_considered": int(count),
        "pass_days": int(pass_days),
        "pass_rate": float(pass_days) / float(count),
        "target_min": float(target_min),
        "target_max": float(target_max),
    }


def _signal_funnel(gating_rows: Iterable[Dict[str, Any]], phase: str, realized_trades: int) -> Dict[str, Any]:
    target_phase = _phase_label(phase)
    rows = [r for r in gating_rows if _phase_label(r.get("phase")) == target_phase]
    candidates = len(rows)
    strategy_eligible = sum(1 for r in rows if _is_strategy_eligible(r))
    execution_eligible = sum(1 for r in rows if _is_execution_eligible(r))
    return {
        "phase": target_phase,
        "candidate_signals": int(candidates),
        "strategy_eligible": int(strategy_eligible),
        "execution_eligible": int(execution_eligible),
        "realized_trades": int(realized_trades),
        "strategy_eligible_rate": (float(strategy_eligible) / float(candidates)) if candidates > 0 else None,
        "execution_eligible_rate": (float(execution_eligible) / float(candidates)) if candidates > 0 else None,
        "realized_trade_rate": (float(realized_trades) / float(candidates)) if candidates > 0 else None,
    }


def _blocker_attribution(gating_rows: Iterable[Dict[str, Any]], phase: str) -> Dict[str, Any]:
    target_phase = _phase_label(phase)
    rows = [r for r in gating_rows if _phase_label(r.get("phase")) == target_phase]
    bucket_counter: Counter[str] = Counter()
    raw_counter: Counter[str] = Counter()
    for row in rows:
        reasons = _blocked_reason_set(row)
        if not reasons:
            continue
        for reason in reasons:
            raw_counter[str(reason)] += 1
            bucket_counter[_blocker_bucket(reason)] += 1
    total = sum(bucket_counter.values())
    bucket_payload = {}
    for key, value in bucket_counter.items():
        bucket_payload[key] = {
            "count": int(value),
            "share": (float(value) / float(total)) if total > 0 else None,
        }
    return {
        "phase": target_phase,
        "total_block_reasons": int(total),
        "bucket_counts": bucket_payload,
        "top_raw_reasons": [{"reason": reason, "count": int(count)} for reason, count in raw_counter.most_common(10)],
    }


def _phase_gate_distribution(gating_rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    by_phase: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in gating_rows:
        phase = _phase_label(row.get("phase"))
        reasons = _blocked_reason_set(row)
        if not reasons:
            by_phase[phase]["unblocked"] += 1
        else:
            for reason in reasons:
                by_phase[phase][_blocker_bucket(reason)] += 1
    for phase, counts in by_phase.items():
        out[phase] = {k: int(v) for k, v in counts.items()}
    return out


def _phase_parity(gating_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(gating_rows)
    overall = _phase_gate_distribution(rows)

    per_phase_dates: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        phase = _phase_label(row.get("phase"))
        bar_ts = _norm_text(row.get("bar_ts"))
        day = bar_ts.split("T", 1)[0] if "T" in bar_ts else ""
        if day:
            per_phase_dates[phase].add(day)

    matched = sorted(per_phase_dates.get("BACKFILL", set()) & per_phase_dates.get("LIVE", set()))
    matched_rows = [r for r in rows if (_norm_text(r.get("bar_ts")).split("T", 1)[0] in matched)] if matched else []
    matched_dist = _phase_gate_distribution(matched_rows)

    return {
        "overall_distribution": overall,
        "matched_dates": matched,
        "matched_distribution": matched_dist,
        "matched_date_count": len(matched),
    }


def _simulate_threshold_policy(
    joined_rows: Iterable[Dict[str, Any]],
    *,
    phase: str,
    setup_threshold: float,
    direction_long_threshold: float,
    direction_short_threshold: float,
    max_hold_bars: int,
    point_value: float,
    cost: CostModel,
) -> Dict[str, Any]:
    target_phase = _phase_label(phase)
    rows = sorted((r for r in joined_rows if _phase_label(r.get("phase")) == target_phase), key=lambda r: _norm_text(r.get("datetime")))

    pos: Optional[Dict[str, Any]] = None
    trades: List[Dict[str, Any]] = []

    def _close(row: Mapping[str, Any], reason: str) -> None:
        nonlocal pos
        if pos is None:
            return
        px = _safe_float(row.get("price"))
        if px is None:
            return
        qty = float(pos.get("qty") or 1.0)
        gross = _gross_pnl_usd(str(pos.get("side") or ""), float(pos.get("entry_price") or 0.0), float(px), qty, point_value)
        cost_usd = _trade_cost_usd(cost, qty)
        trades.append(
            {
                "entry_ts": pos.get("entry_ts"),
                "exit_ts": row.get("datetime"),
                "side": pos.get("side"),
                "qty": qty,
                "entry_price": pos.get("entry_price"),
                "exit_price": px,
                "gross_pnl_usd": gross,
                "cost_usd": cost_usd,
                "net_pnl_usd": gross - cost_usd,
                "exit_reason": reason,
            }
        )
        pos = None

    for row in rows:
        px = _safe_float(row.get("price"))
        if px is None:
            continue

        desired_side: Optional[str] = None
        setup_prob = _safe_float(row.get("setup_prob"))
        p_long = _safe_float(row.get("pred_p_long"))
        p_short = _safe_float(row.get("short_prob"))

        if setup_prob is not None and setup_prob >= float(setup_threshold):
            if p_long is not None and p_long >= float(direction_long_threshold):
                desired_side = "LONG"
            elif p_short is not None and p_short >= float(direction_short_threshold):
                desired_side = "SHORT"

        if pos is not None:
            pos["bars_in_trade"] = int(pos.get("bars_in_trade", 0) or 0) + 1
            if int(pos.get("bars_in_trade", 0) or 0) >= int(max_hold_bars):
                _close(row, "max_hold_bars")

        if desired_side is None:
            continue

        if pos is None:
            pos = {
                "entry_ts": row.get("datetime"),
                "entry_price": float(px),
                "side": desired_side,
                "qty": float(_safe_float(row.get("size")) or 1.0),
                "bars_in_trade": 0,
            }
            continue

        current_side = _norm_text(pos.get("side")).upper()
        if desired_side != current_side:
            _close(row, "signal_flip")
            pos = {
                "entry_ts": row.get("datetime"),
                "entry_price": float(px),
                "side": desired_side,
                "qty": float(_safe_float(row.get("size")) or 1.0),
                "bars_in_trade": 0,
            }

    if pos is not None and rows:
        _close(rows[-1], "forced_end_of_phase")

    daily_net: Dict[str, float] = defaultdict(float)
    for trade in trades:
        exit_ts = _norm_text(trade.get("exit_ts"))
        day = exit_ts.split("T", 1)[0] if "T" in exit_ts else exit_ts[:10]
        if day:
            daily_net[day] += float(trade.get("net_pnl_usd") or 0.0)

    return {
        "trade_count": len(trades),
        "net_pnl_usd": float(sum(float(t.get("net_pnl_usd") or 0.0) for t in trades)),
        "gross_pnl_usd": float(sum(float(t.get("gross_pnl_usd") or 0.0) for t in trades)),
        "cost_usd": float(sum(float(t.get("cost_usd") or 0.0) for t in trades)),
        "daily_net": dict(sorted(daily_net.items())),
    }


def _compute_drawdown(daily_net: Mapping[str, float]) -> float:
    curve = 0.0
    peak = 0.0
    max_dd = 0.0
    for _, pnl in sorted(daily_net.items()):
        curve += float(pnl)
        peak = max(peak, curve)
        max_dd = max(max_dd, peak - curve)
    return float(max_dd)


def _threshold_sweep(
    joined_rows: Iterable[Dict[str, Any]],
    *,
    phase: str,
    setup_thresholds: Iterable[float],
    direction_long_thresholds: Iterable[float],
    direction_short_thresholds: Iterable[float],
    max_hold_bars: int,
    point_value: float,
    cost: CostModel,
    target_min: float,
    target_max: float,
    window_days: int,
    max_daily_loss_cap: float,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for st, dl, ds in itertools.product(setup_thresholds, direction_long_thresholds, direction_short_thresholds):
        sim = _simulate_threshold_policy(
            joined_rows,
            phase=phase,
            setup_threshold=float(st),
            direction_long_threshold=float(dl),
            direction_short_threshold=float(ds),
            max_hold_bars=int(max_hold_bars),
            point_value=float(point_value),
            cost=cost,
        )
        pass_stats = _daily_pass_rate(
            sim.get("daily_net") or {},
            target_min=float(target_min),
            target_max=float(target_max),
            window_days=int(window_days),
        )
        dd = _compute_drawdown(sim.get("daily_net") or {})
        worst_day = min((float(v) for v in (sim.get("daily_net") or {}).values()), default=0.0)
        row = {
            "setup_threshold": float(st),
            "direction_long_threshold": float(dl),
            "direction_short_threshold": float(ds),
            "trade_count": int(sim.get("trade_count") or 0),
            "net_pnl_usd": float(sim.get("net_pnl_usd") or 0.0),
            "gross_pnl_usd": float(sim.get("gross_pnl_usd") or 0.0),
            "cost_usd": float(sim.get("cost_usd") or 0.0),
            "max_drawdown_usd": float(dd),
            "worst_day_usd": float(worst_day),
            "pass_rate": pass_stats.get("pass_rate"),
            "pass_days": pass_stats.get("pass_days"),
            "days_considered": pass_stats.get("days_considered"),
            "daily_net": sim.get("daily_net"),
            "conservative_cap_pass": bool(float(worst_day) >= -abs(float(max_daily_loss_cap))),
        }
        candidates.append(row)

    scored = sorted(
        candidates,
        key=lambda r: (
            -float(r.get("pass_rate") or -1.0),
            -float(r.get("net_pnl_usd") or 0.0),
            float(r.get("max_drawdown_usd") or 0.0),
        ),
    )

    best = scored[0] if scored else None
    return {
        "candidate_count": len(candidates),
        "best": best,
        "top10": scored[:10],
        "all_candidates": scored,
    }


def evaluate_backfill_reliability(
    run_dir: Path,
    *,
    phase: str,
    target_min: float,
    target_max: float,
    window_days: int,
    cost_profile: str,
    setup_thresholds: Optional[List[float]] = None,
    direction_long_thresholds: Optional[List[float]] = None,
    direction_short_thresholds: Optional[List[float]] = None,
    max_hold_bars: int = 8,
    max_daily_loss_cap: float = 400.0,
) -> Dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    state_rows = _load_state_rows(run_dir / "state.csv")
    gating_rows = _load_gating_rows(run_dir / "gating_events.jsonl")
    joined = _join_state_with_gating(state_rows, gating_rows)
    cost = _build_cost_model(run_dir, profile=cost_profile)
    point_value = _point_value_for_instrument(_instrument_from_run(run_dir))

    reconstructed = _reconstruct_trades_from_actions(
        joined,
        phase=phase,
        point_value=point_value,
        cost=cost,
    )
    funnel = _signal_funnel(gating_rows, phase=phase, realized_trades=int(reconstructed.get("trade_count") or 0))
    attribution = _blocker_attribution(gating_rows, phase=phase)
    pass_stats = _daily_pass_rate(
        reconstructed.get("daily_net") or {},
        target_min=target_min,
        target_max=target_max,
        window_days=window_days,
    )
    parity = _phase_parity(gating_rows)

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "phase": _phase_label(phase),
        "cost_profile": cost_profile,
        "cost_model": {
            "commission_per_contract_per_side": float(cost.commission_per_contract_per_side),
            "slippage_ticks_per_side": float(cost.slippage_ticks_per_side),
            "tick_size": float(cost.tick_size),
            "tick_value": float(cost.tick_value),
            "point_value": float(point_value),
        },
        "signal_funnel": funnel,
        "reconstructed_trades": {
            "trade_count": int(reconstructed.get("trade_count") or 0),
            "gross_pnl_usd": float(reconstructed.get("gross_pnl_usd") or 0.0),
            "cost_usd": float(reconstructed.get("cost_usd") or 0.0),
            "net_pnl_usd": float(reconstructed.get("net_pnl_usd") or 0.0),
            "daily_net": reconstructed.get("daily_net") or {},
            "non_overlapping_position_logic": {
                "ignored_open_while_in_position": int(reconstructed.get("ignored_open_while_in_position") or 0),
                "close_without_position": int(reconstructed.get("close_without_position") or 0),
            },
            "sample_trades": (reconstructed.get("trades") or [])[:10],
        },
        "blocker_attribution": attribution,
        "daily_target_pass_rate": pass_stats,
        "phase_parity": parity,
    }

    if setup_thresholds and direction_long_thresholds and direction_short_thresholds:
        report["threshold_sweep"] = _threshold_sweep(
            joined,
            phase=phase,
            setup_thresholds=setup_thresholds,
            direction_long_thresholds=direction_long_thresholds,
            direction_short_thresholds=direction_short_thresholds,
            max_hold_bars=max_hold_bars,
            point_value=point_value,
            cost=cost,
            target_min=target_min,
            target_max=target_max,
            window_days=window_days,
            max_daily_loss_cap=max_daily_loss_cap,
        )

    return report


def _parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for item in str(raw or "").split(","):
        txt = item.strip()
        if not txt:
            continue
        out.append(float(txt))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate backfill reliability with execution-like reconstruction.")
    p.add_argument("--run-dir", required=True, help="Run directory containing state.csv and gating_events.jsonl")
    p.add_argument("--phase", default="BACKFILL", choices=["BACKFILL", "CATCHUP", "LIVE"], help="Phase to evaluate")
    p.add_argument("--target-min", type=float, default=400.0)
    p.add_argument("--target-max", type=float, default=1200.0)
    p.add_argument("--window-days", type=int, default=20)
    p.add_argument("--cost-profile", default="current_default", choices=["current_default"])
    p.add_argument("--output", default=None, help="Optional explicit output JSON path")

    p.add_argument("--setup-thresholds", default="0.30,0.35,0.40")
    p.add_argument("--direction-long-thresholds", default="0.56,0.58,0.60,0.62")
    p.add_argument("--direction-short-thresholds", default="0.56,0.58,0.60,0.62")
    p.add_argument("--max-hold-bars", type=int, default=8)
    p.add_argument("--max-daily-loss-cap", type=float, default=400.0)
    p.add_argument("--disable-threshold-sweep", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_thresholds = None if args.disable_threshold_sweep else _parse_float_list(args.setup_thresholds)
    direction_long_thresholds = None if args.disable_threshold_sweep else _parse_float_list(args.direction_long_thresholds)
    direction_short_thresholds = None if args.disable_threshold_sweep else _parse_float_list(args.direction_short_thresholds)

    report = evaluate_backfill_reliability(
        Path(args.run_dir),
        phase=args.phase,
        target_min=float(args.target_min),
        target_max=float(args.target_max),
        window_days=int(args.window_days),
        cost_profile=str(args.cost_profile),
        setup_thresholds=setup_thresholds,
        direction_long_thresholds=direction_long_thresholds,
        direction_short_thresholds=direction_short_thresholds,
        max_hold_bars=int(args.max_hold_bars),
        max_daily_loss_cap=float(args.max_daily_loss_cap),
    )

    out_path = Path(args.output).expanduser().resolve() if args.output else (Path(args.run_dir).expanduser().resolve() / "backfill_reliability_report.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(out_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
