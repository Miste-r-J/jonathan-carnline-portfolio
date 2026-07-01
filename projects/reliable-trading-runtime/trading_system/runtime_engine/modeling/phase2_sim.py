from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Phase2SimConfig:
    """Runtime configuration for Phase-2 trade simulation."""

    tz: str
    trade_window_start: str
    trade_window_end: str
    point_value: float
    tick_value: float
    contracts: int = 1
    max_hold_bars: int = 24
    flatten_gap_minutes: int = 180
    commission_per_contract: float = 2.0
    slippage_ticks: float = 1.0


@dataclass(frozen=True)
class Phase2DecisionPolicy:
    """Live/replay policy for suppressing unsafe Phase-2 entry signals."""

    entry_trend_filter: str = "none"
    block_short_above_trend_score: Optional[float] = None
    countertrend_short_max_trend_score: Optional[float] = None
    countertrend_short_min_setup_when_strong_trend: Optional[float] = None
    short_flip_cooldown_bars_after_long_lineage: int = 0
    min_signal_persistence_bars: int = 1
    cooldown_bars_after_flip: int = 0

    def __post_init__(self) -> None:
        entry_trend_filter = str(self.entry_trend_filter or "none").strip().lower()
        if entry_trend_filter not in {"none", "vwap_ema"}:
            raise ValueError(f"Unsupported Phase-2 entry_trend_filter: {self.entry_trend_filter!r}")
        object.__setattr__(self, "entry_trend_filter", entry_trend_filter)
        short_block = self.block_short_above_trend_score
        if short_block is not None:
            short_block = float(short_block)
        object.__setattr__(self, "block_short_above_trend_score", short_block)
        strong_countertrend_score = self.countertrend_short_max_trend_score
        if strong_countertrend_score is not None:
            strong_countertrend_score = float(strong_countertrend_score)
        object.__setattr__(self, "countertrend_short_max_trend_score", strong_countertrend_score)
        strong_countertrend_setup = self.countertrend_short_min_setup_when_strong_trend
        if strong_countertrend_setup is not None:
            strong_countertrend_setup = float(strong_countertrend_setup)
        object.__setattr__(
            self,
            "countertrend_short_min_setup_when_strong_trend",
            strong_countertrend_setup,
        )
        object.__setattr__(
            self,
            "short_flip_cooldown_bars_after_long_lineage",
            max(0, int(self.short_flip_cooldown_bars_after_long_lineage or 0)),
        )
        object.__setattr__(self, "min_signal_persistence_bars", max(1, int(self.min_signal_persistence_bars or 1)))
        object.__setattr__(self, "cooldown_bars_after_flip", max(0, int(self.cooldown_bars_after_flip or 0)))

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "Phase2DecisionPolicy":
        payload = payload or {}
        return cls(
            entry_trend_filter=str(payload.get("entry_trend_filter") or "none"),
            block_short_above_trend_score=payload.get("block_short_above_trend_score"),
            countertrend_short_max_trend_score=payload.get("countertrend_short_max_trend_score"),
            countertrend_short_min_setup_when_strong_trend=payload.get(
                "countertrend_short_min_setup_when_strong_trend"
            ),
            short_flip_cooldown_bars_after_long_lineage=int(
                payload.get("short_flip_cooldown_bars_after_long_lineage") or 0
            ),
            min_signal_persistence_bars=int(payload.get("min_signal_persistence_bars") or 1),
            cooldown_bars_after_flip=int(payload.get("cooldown_bars_after_flip") or 0),
        )

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "entry_trend_filter": self.entry_trend_filter,
            "block_short_above_trend_score": self.block_short_above_trend_score,
            "countertrend_short_max_trend_score": self.countertrend_short_max_trend_score,
            "countertrend_short_min_setup_when_strong_trend": self.countertrend_short_min_setup_when_strong_trend,
            "short_flip_cooldown_bars_after_long_lineage": int(self.short_flip_cooldown_bars_after_long_lineage),
            "min_signal_persistence_bars": int(self.min_signal_persistence_bars),
            "cooldown_bars_after_flip": int(self.cooldown_bars_after_flip),
        }


def _raw_phase2_signal(
    setup_probs: np.ndarray,
    dir_probs: np.ndarray,
    thresholds: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p_setup = float(thresholds.get("p_setup", 0.6))
    p_long = float(thresholds.get("p_long", 0.6))
    p_short_required = float(thresholds.get("p_short", 0.6))
    p_short_cut = 1.0 - p_long

    setup_arr = np.asarray(setup_probs, dtype=float).reshape(-1)
    dir_arr = np.asarray(dir_probs, dtype=float).reshape(-1)
    setup_pass = np.isfinite(setup_arr) & (setup_arr >= p_setup)
    effective = np.array(dir_arr, copy=True)
    effective[~setup_pass] = 0.5
    direction = np.zeros(len(effective), dtype=int)
    direction[(effective >= p_long) & setup_pass] = 1
    short_prob = 1.0 - effective
    direction[(effective <= p_short_cut) & (short_prob >= p_short_required) & setup_pass] = -1
    return setup_arr, effective, setup_pass, direction


def _trend_filter_masks(feats: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    required = {"Close", "vwap_sess", "ema_20", "ema_50"}
    if not required.issubset(set(feats.columns)):
        empty = np.zeros(len(feats), dtype=bool)
        return empty, empty
    close = pd.to_numeric(feats["Close"], errors="coerce")
    vwap = pd.to_numeric(feats["vwap_sess"], errors="coerce")
    ema20 = pd.to_numeric(feats["ema_20"], errors="coerce")
    ema50 = pd.to_numeric(feats["ema_50"], errors="coerce")
    strong_up = ((close > vwap) & (ema20 > ema50)).fillna(False).to_numpy(dtype=bool)
    strong_down = ((close < vwap) & (ema20 < ema50)).fillna(False).to_numpy(dtype=bool)
    return strong_up, strong_down


def _trend_score_short_block_mask(feats: pd.DataFrame, threshold: Optional[float]) -> np.ndarray:
    if threshold is None:
        return np.zeros(len(feats), dtype=bool)
    for col in ("trend_score", "trend_avg"):
        if col in feats.columns:
            vals = pd.to_numeric(feats[col], errors="coerce")
            return (vals > float(threshold)).fillna(False).to_numpy(dtype=bool)
    return np.zeros(len(feats), dtype=bool)


def _countertrend_short_setup_block_mask(
    feats: pd.DataFrame,
    raw_direction: np.ndarray,
    setup_probs: Optional[np.ndarray],
    *,
    max_trend_score: Optional[float],
    min_setup_when_strong_trend: Optional[float],
) -> np.ndarray:
    if max_trend_score is None or min_setup_when_strong_trend is None:
        return np.zeros(len(feats), dtype=bool)
    strong_up, _ = _trend_filter_masks(feats)
    if not np.any(strong_up):
        return np.zeros(len(feats), dtype=bool)
    setup_arr = None
    if setup_probs is not None:
        setup_arr = np.asarray(setup_probs, dtype=float).reshape(-1)
    elif "phase2_setup_prob" in feats.columns:
        setup_arr = pd.to_numeric(feats["phase2_setup_prob"], errors="coerce").to_numpy(dtype=float)
    if setup_arr is None:
        return np.zeros(len(feats), dtype=bool)
    trend_block = _trend_score_short_block_mask(feats, max_trend_score)
    weak_setup = np.isfinite(setup_arr) & (setup_arr < float(min_setup_when_strong_trend))
    return (np.asarray(raw_direction, dtype=int).reshape(-1) < 0) & strong_up & trend_block & weak_setup


def apply_phase2_decision_policy(
    feats: pd.DataFrame,
    raw_direction: np.ndarray,
    *,
    setup_probs: Optional[np.ndarray] = None,
    policy: Optional[Phase2DecisionPolicy | Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Apply live-safe signal suppression to raw Phase-2 directions."""
    decision_policy = (
        policy
        if isinstance(policy, Phase2DecisionPolicy)
        else Phase2DecisionPolicy.from_mapping(policy if isinstance(policy, Mapping) else None)
    )
    raw = np.asarray(raw_direction, dtype=int).reshape(-1)
    final = np.array(raw, copy=True)
    reasons = ["setup_pass" if int(value) != 0 else "direction_uncertain" for value in raw]

    if decision_policy.entry_trend_filter == "vwap_ema":
        strong_up, strong_down = _trend_filter_masks(feats)
        bad_short = (final < 0) & strong_up
        bad_long = (final > 0) & strong_down
        blocked = bad_short | bad_long
        final[blocked] = 0
        for idx in np.flatnonzero(blocked):
            reasons[int(idx)] = "trend_filter_countertrend"

    blocked_shorts = (final < 0) & _trend_score_short_block_mask(
        feats,
        decision_policy.block_short_above_trend_score,
    )
    final[blocked_shorts] = 0
    for idx in np.flatnonzero(blocked_shorts):
        reasons[int(idx)] = "trend_score_short_block"

    blocked_countertrend_shorts = _countertrend_short_setup_block_mask(
        feats,
        final,
        setup_probs,
        max_trend_score=decision_policy.countertrend_short_max_trend_score,
        min_setup_when_strong_trend=decision_policy.countertrend_short_min_setup_when_strong_trend,
    )
    final[blocked_countertrend_shorts] = 0
    for idx in np.flatnonzero(blocked_countertrend_shorts):
        reasons[int(idx)] = "countertrend_short_setup_filter"

    short_flip_cooldown = int(decision_policy.short_flip_cooldown_bars_after_long_lineage)
    if short_flip_cooldown > 0:
        long_lineage_cooldown = 0
        for idx, value in enumerate(final.copy()):
            side = int(value)
            if side > 0:
                long_lineage_cooldown = short_flip_cooldown
                continue
            if side < 0 and long_lineage_cooldown > 0:
                final[idx] = 0
                reasons[idx] = "short_flip_cooldown"
                long_lineage_cooldown -= 1
                continue
            if long_lineage_cooldown > 0:
                long_lineage_cooldown -= 1

    min_persist = int(decision_policy.min_signal_persistence_bars)
    if min_persist > 1:
        seen_side = 0
        run_len = 0
        for idx, value in enumerate(final.copy()):
            side = int(value)
            if side == 0:
                seen_side = 0
                run_len = 0
                continue
            if side == seen_side:
                run_len += 1
            else:
                seen_side = side
                run_len = 1
            if run_len < min_persist:
                final[idx] = 0
                reasons[idx] = "signal_persistence"

    cooldown = int(decision_policy.cooldown_bars_after_flip)
    if cooldown > 0:
        active_side = 0
        cooldown_remaining = 0
        for idx, value in enumerate(final.copy()):
            side = int(value)
            if side == 0:
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                continue
            if cooldown_remaining > 0 and active_side != 0 and side != active_side:
                final[idx] = 0
                reasons[idx] = "flip_cooldown"
                cooldown_remaining -= 1
                continue
            if active_side != 0 and side != active_side:
                active_side = side
                cooldown_remaining = cooldown
            elif active_side == 0:
                active_side = side
            elif cooldown_remaining > 0:
                cooldown_remaining -= 1

    return final, reasons


def phase2_decisions(
    feats: pd.DataFrame,
    setup_probs: np.ndarray,
    dir_probs: np.ndarray,
    thresholds: Dict[str, float],
    *,
    policy: Optional[Phase2DecisionPolicy | Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Attach Phase-2 setup+direction decisions to a feature frame."""
    df = feats.copy()
    setup_arr, effective, setup_pass, raw_direction = _raw_phase2_signal(setup_probs, dir_probs, thresholds)
    raw_dir_probs = np.asarray(dir_probs, dtype=float).reshape(-1)
    bridge_direction = np.zeros(len(raw_dir_probs), dtype=int)
    short_prob_arr = 1.0 - raw_dir_probs
    bridge_direction[np.isfinite(raw_dir_probs) & (raw_dir_probs >= float(thresholds.get("p_long", 0.6)))] = 1
    bridge_direction[
        np.isfinite(raw_dir_probs)
        & (raw_dir_probs <= 1.0 - float(thresholds.get("p_long", 0.6)))
        & (short_prob_arr >= float(thresholds.get("p_short", 0.6)))
    ] = -1
    final_direction, policy_reasons = apply_phase2_decision_policy(
        df,
        raw_direction,
        setup_probs=setup_arr,
        policy=policy,
    )
    force_open_direction, force_open_policy_reasons = apply_phase2_decision_policy(
        df,
        bridge_direction,
        setup_probs=setup_arr,
        policy=policy,
    )
    df["phase2_setup_prob"] = setup_arr
    df["phase2_setup_pass"] = setup_pass
    df["phase2_reason"] = np.where(setup_pass, "direction_uncertain", "phase2_setup_gate")
    df["dir_prob_raw"] = dir_probs
    df["dir_prob_effective"] = effective
    df["phase2_direction_signal_raw"] = raw_direction
    df["phase2_policy_reason"] = policy_reasons
    df["phase2_policy_suppressed"] = (raw_direction != 0) & (final_direction == 0)
    df["phase2_force_open_direction_signal"] = force_open_direction
    df["phase2_force_open_policy_reason"] = force_open_policy_reasons
    df["phase2_force_open_policy_suppressed"] = (bridge_direction != 0) & (force_open_direction == 0)
    df.loc[df["phase2_policy_suppressed"], "phase2_reason"] = df.loc[
        df["phase2_policy_suppressed"], "phase2_policy_reason"
    ]
    df.loc[final_direction > 0, "phase2_reason"] = "dir_long"
    df.loc[final_direction < 0, "phase2_reason"] = "dir_short"
    df["phase2_direction_signal"] = final_direction
    return df


def _within_trade_window(local_times: pd.Series, start: str, end: str) -> np.ndarray:
    start_t = pd.to_datetime(start).time()
    end_t = pd.to_datetime(end).time()
    return (local_times.dt.time >= start_t) & (local_times.dt.time <= end_t)


def _cost_per_side(cfg: Phase2SimConfig) -> float:
    return cfg.commission_per_contract + cfg.slippage_ticks * cfg.tick_value


def _compute_profit_factor(trades: Sequence[Dict[str, float]]) -> float:
    gross_win = sum(max(float(t["pnl_usd"]), 0.0) for t in trades)
    gross_loss = sum(min(float(t["pnl_usd"]), 0.0) for t in trades)
    if gross_loss == 0.0:
        return float("inf") if gross_win > 0 else 0.0
    return abs(gross_win / gross_loss)


def _side_metrics(trades: Sequence[Dict[str, object]], side: int) -> Dict[str, float | int]:
    side_trades = [trade for trade in trades if int(trade["side"]) == side]
    gross_pnl = sum(float(trade["gross_pnl_usd"]) for trade in side_trades)
    costs = sum(float(trade["costs_usd"]) for trade in side_trades)
    realized_pnl = sum(float(trade["pnl_usd"]) for trade in side_trades)
    pnl_points = sum(float(trade["pnl_points"]) for trade in side_trades)
    wins = sum(1 for trade in side_trades if float(trade["pnl_usd"]) > 0.0)
    return {
        "trade_count": len(side_trades),
        "gross_pnl_usd": gross_pnl,
        "costs_usd": costs,
        "total_costs_usd": costs,
        "realized_pnl_usd": realized_pnl,
        "net_pnl_usd": realized_pnl,
        "total_pnl_usd": realized_pnl,
        "pnl_points": pnl_points,
        "win_rate": wins / len(side_trades) if side_trades else 0.0,
        "profit_factor": _compute_profit_factor(side_trades),
    }


def simulate_trades(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    *,
    cfg: Phase2SimConfig,
) -> Dict[str, object]:
    """Simulate trades for a Phase-2 decision frame."""
    dt_series = pd.to_datetime(df["Datetime"], errors="coerce")
    if getattr(dt_series.dt, "tz", None) is None:
        dt_series = dt_series.dt.tz_localize(cfg.tz)
    local = dt_series.dt.tz_convert(cfg.tz)
    within_window = _within_trade_window(local, cfg.trade_window_start, cfg.trade_window_end)

    if "phase2_direction_signal" in df.columns:
        signals = pd.to_numeric(df["phase2_direction_signal"], errors="coerce").fillna(0.0).to_numpy(dtype=int)
        signals = np.where(within_window, signals, 0)
    else:
        p_long = thresholds.get("p_long", 0.6)
        p_short_required = thresholds.get("p_short", 0.6)
        short_cut = 1.0 - p_long
        probs = df["dir_prob_effective"].to_numpy()
        signals = np.zeros(len(df), dtype=int)
        signals[(probs >= p_long) & within_window] = 1
        signals[(probs <= short_cut) & ((1.0 - probs) >= p_short_required) & within_window] = -1

    close = pd.to_numeric(df["Close"], errors="coerce").ffill()
    trades: List[Dict[str, object]] = []
    equity: List[float] = [0.0]
    position = 0
    entry_price = 0.0
    entry_idx = None
    entry_ts = None
    last_ts = None
    last_price = None
    hold_bars = 0
    realized_gross_pnl = 0.0
    total_costs = 0.0
    per_side_cost = _cost_per_side(cfg) * cfg.contracts

    for idx in range(len(df)):
        ts = dt_series.iloc[idx]
        if ts.tzinfo is None:
            ts = ts.tz_localize(cfg.tz)
        price = float(close.iloc[idx])
        target = int(signals[idx])

        gap_detected = last_ts is not None and ts - last_ts >= pd.Timedelta(minutes=cfg.flatten_gap_minutes)
        if gap_detected and position != 0 and entry_idx is not None and entry_ts is not None and last_price is not None:
            pnl_points = (float(last_price) - entry_price) * position
            gross_pnl_usd = pnl_points * cfg.point_value * cfg.contracts
            trade_costs_usd = 2.0 * per_side_cost
            net_pnl_usd = gross_pnl_usd - trade_costs_usd
            trades.append(
                {
                    "side": position,
                    "entry_ts": entry_ts.isoformat() if isinstance(entry_ts, pd.Timestamp) else str(entry_ts),
                    "exit_ts": last_ts.isoformat(),
                    "entry_price": entry_price,
                    "exit_price": float(last_price),
                    "pnl_points": pnl_points,
                    "gross_pnl_usd": gross_pnl_usd,
                    "costs_usd": trade_costs_usd,
                    "pnl_usd": net_pnl_usd,
                    "net_pnl_usd": net_pnl_usd,
                    "bars": hold_bars,
                    "reason": "gap_flatten",
                }
            )
            realized_gross_pnl += gross_pnl_usd
            total_costs += per_side_cost
            position = 0
            entry_idx = None
            entry_ts = None
            hold_bars = 0
        if gap_detected:
            target = 0
        last_ts = ts
        last_price = price

        if position != 0:
            hold_bars += 1
            if hold_bars >= cfg.max_hold_bars:
                target = 0

        if target != position:
            if position != 0 and entry_idx is not None and entry_ts is not None:
                pnl_points = (price - entry_price) * position
                gross_pnl_usd = pnl_points * cfg.point_value * cfg.contracts
                trade_costs_usd = 2.0 * per_side_cost
                net_pnl_usd = gross_pnl_usd - trade_costs_usd
                trades.append(
                    {
                        "side": position,
                        "entry_ts": entry_ts.isoformat() if isinstance(entry_ts, pd.Timestamp) else str(entry_ts),
                        "exit_ts": ts.isoformat(),
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl_points": pnl_points,
                        "gross_pnl_usd": gross_pnl_usd,
                        "costs_usd": trade_costs_usd,
                        "pnl_usd": net_pnl_usd,
                        "net_pnl_usd": net_pnl_usd,
                        "bars": hold_bars,
                        "reason": "signal_flip" if target != 0 else "flat",
                    }
                )
                realized_gross_pnl += gross_pnl_usd
                total_costs += per_side_cost
            position = target
            entry_idx = idx if target != 0 else None
            entry_ts = ts if target != 0 else None
            hold_bars = 0
            if target != 0:
                entry_price = price
                total_costs += per_side_cost

        unrealized_pnl = 0.0
        if position != 0:
            unrealized_pnl = (price - entry_price) * position * cfg.point_value * cfg.contracts
        equity.append(realized_gross_pnl - total_costs + unrealized_pnl)

    if position != 0 and entry_idx is not None and entry_ts is not None:
        price = float(close.iloc[-1])
        ts = dt_series.iloc[-1]
        pnl_points = (price - entry_price) * position
        gross_pnl_usd = pnl_points * cfg.point_value * cfg.contracts
        trade_costs_usd = 2.0 * per_side_cost
        net_pnl_usd = gross_pnl_usd - trade_costs_usd
        trades.append(
            {
                "side": position,
                "entry_ts": entry_ts.isoformat() if isinstance(entry_ts, pd.Timestamp) else str(entry_ts),
                "exit_ts": ts.isoformat(),
                "entry_price": entry_price,
                "exit_price": price,
                "pnl_points": pnl_points,
                "gross_pnl_usd": gross_pnl_usd,
                "costs_usd": trade_costs_usd,
                "pnl_usd": net_pnl_usd,
                "net_pnl_usd": net_pnl_usd,
                "bars": hold_bars,
                "reason": "forced_exit",
            }
        )
        realized_gross_pnl += gross_pnl_usd
        total_costs += per_side_cost
        equity.append(realized_gross_pnl - total_costs)

    equity_series = pd.Series(equity)
    running_max = equity_series.cummax()
    drawdowns = equity_series - running_max
    max_dd = abs(drawdowns.min())

    days = local.dt.date.unique()
    trades_per_day = len(trades) / max(1, len(days))
    flips = sum(1 for t in trades if t["reason"] == "signal_flip" and t["side"] != 0)
    flip_rate = flips / max(1, len(days))

    holds = [t["bars"] for t in trades]
    hold_stats = {
        "mean": float(np.mean(holds)) if holds else 0.0,
        "median": float(np.median(holds)) if holds else 0.0,
        "q75": float(np.quantile(holds, 0.75)) if holds else 0.0,
        "q90": float(np.quantile(holds, 0.90)) if holds else 0.0,
        "q95": float(np.quantile(holds, 0.95)) if holds else 0.0,
    }

    wins = [t for t in trades if t["pnl_usd"] > 0]
    total_pnl = sum(float(t["pnl_usd"]) for t in trades)
    gross_pnl = sum(float(t["gross_pnl_usd"]) for t in trades)
    total_costs_from_trades = sum(float(t["costs_usd"]) for t in trades)
    total_points = sum(float(t["pnl_points"]) for t in trades)
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_trade = total_pnl / len(trades) if trades else 0.0
    median_trade = float(np.median([t["pnl_usd"] for t in trades])) if trades else 0.0

    account_size = 50000.0
    trade_returns = [t["pnl_usd"] / account_size for t in trades if account_size > 0]
    sharpe = float("nan")
    if len(trade_returns) > 1:
        ret_series = np.array(trade_returns)
        std = ret_series.std()
        if std > 0:
            trades_per_year = trades_per_day * 252.0
            sharpe = (ret_series.mean() / std) * np.sqrt(max(trades_per_year, 1.0))

    long_trades = sum(1 for t in trades if t["side"] > 0)
    short_trades = sum(1 for t in trades if t["side"] < 0)
    total_side = long_trades + short_trades
    direction_bias = {
        "long_fraction": long_trades / total_side if total_side else 0.0,
        "short_fraction": short_trades / total_side if total_side else 0.0,
    }

    profit_factor = _compute_profit_factor(trades)
    side_metrics = {
        "long": _side_metrics(trades, 1),
        "short": _side_metrics(trades, -1),
    }

    return {
        "equity": equity,
        "max_drawdown": max_dd,
        "total_pnl_usd": total_pnl,
        "realized_pnl_usd": total_pnl,
        "net_pnl_usd": total_pnl,
        "gross_pnl_usd": gross_pnl,
        "total_costs_usd": total_costs_from_trades,
        "total_pnl_points": total_points,
        "trades": trades,
        "trade_count": len(trades),
        "win_rate": win_rate,
        "avg_trade_usd": avg_trade,
        "median_trade_usd": median_trade,
        "trades_per_day": trades_per_day,
        "flip_rate_per_day": flip_rate,
        "hold_bars": hold_stats,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "direction_bias": direction_bias,
        "side_metrics": side_metrics,
        "long_trade_count": side_metrics["long"]["trade_count"],
        "long_gross_pnl_usd": side_metrics["long"]["gross_pnl_usd"],
        "long_costs_usd": side_metrics["long"]["costs_usd"],
        "long_realized_pnl_usd": side_metrics["long"]["realized_pnl_usd"],
        "long_net_pnl_usd": side_metrics["long"]["realized_pnl_usd"],
        "long_pnl_usd": side_metrics["long"]["realized_pnl_usd"],
        "short_trade_count": side_metrics["short"]["trade_count"],
        "short_gross_pnl_usd": side_metrics["short"]["gross_pnl_usd"],
        "short_costs_usd": side_metrics["short"]["costs_usd"],
        "short_realized_pnl_usd": side_metrics["short"]["realized_pnl_usd"],
        "short_net_pnl_usd": side_metrics["short"]["realized_pnl_usd"],
        "short_pnl_usd": side_metrics["short"]["realized_pnl_usd"],
    }
