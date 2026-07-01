#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pnl_calc_enhanced.py — Local P&L + stats from OPEN/CLOSE/FLIP trade logs

Backwards compatible with the user's format:
  2025-10-10T08:20:00,OPEN,SHORT,6792.25,0.2792,A+,6799.0,6785.5
  2025-10-10T08:32:00,CLOSE,FLAT,6793.50,0.3266,A+,6799.0,6785.5

Columns (with or without header):
  timestamp, action(OPEN/CLOSE/FLIP), side(LONG/SHORT/FLAT), price, prob, grade, stop, target
Optional extra columns (auto-detected if present):
  contracts|size (int), symbol, note

Key upgrades vs. baseline:
- Handles FLIP: closes current at given price w/ slippage, then opens new side immediately.
- Optional per-trade size column that overrides global --contracts.
- Robust CSV ingest (header/no-header/extra cols), with row-level validation & skip reasons.
- Adds equity curve, max drawdown, streaks (W/L), payoff ratio, R-metrics, daily & monthly stats.
- Optional outputs: per-trade CSV (--out), stats JSON (--json), equity CSV (--equity).
- Deterministic timezone handling; pretty console table.

Usage examples:
  python pnl_calc_enhanced.py trades.csv --instrument ES --contracts 1
  cat trades.csv | python pnl_calc_enhanced.py - --instrument MES --contracts 2 \
    --commission 2.0 --slip-ticks 1 --equity equity.csv --json stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - best effort import
    from na.bot.config import instrument_by_alias as _instrument_lookup  # type: ignore
    from na.bot.config import INSTRUMENTS as _INSTRUMENTS  # type: ignore
except Exception:  # pragma: no cover - standalone usage
    _instrument_lookup = None
    _INSTRUMENTS = {}

# --------------------- Instrument metadata ---------------------
POINT_VALUE_BY_INSTRUMENT = {
    "ES": 50.0, "MES": 5.0,
    "NQ": 20.0, "MNQ": 2.0,
    "YM": 5.0,  "MYM": 0.5,
    "GC": 100.0, "MGC": 10.0,
    "CL": 1000.0,
    "6E": 125000.0,
}

TICK_SIZE_BY_INSTRUMENT = {
    "ES": 0.25, "MES": 0.25,
    "NQ": 0.25, "MNQ": 0.25,
    "YM": 1.0,  "MYM": 1.0,
    "GC": 0.1,  "MGC": 0.1,
    "CL": 0.01,
    "6E": 0.00005,
}

for _alias, _spec in getattr(_INSTRUMENTS, "items", lambda: [])():  # type: ignore
    POINT_VALUE_BY_INSTRUMENT.setdefault(_alias.upper(), _spec.point_value)
    TICK_SIZE_BY_INSTRUMENT.setdefault(_alias.upper(), _spec.tick_size)

EXPECTED_COLS = ["timestamp","action","side","price","prob","grade","stop","target"]
OPTIONAL_SIZE_COLS = ("contracts", "size")

# --------------------- Helpers ---------------------

def _empty_trade_frame() -> pd.DataFrame:
    """Return placeholder DataFrame when the source CSV has no rows."""
    return pd.DataFrame(columns=EXPECTED_COLS)


def _read_any(csv_path: str) -> pd.DataFrame:
    """Read CSV with or without headers. If header missing, assign EXPECTED_COLS.
    Extra columns are kept and lower-cased.
    """
    if csv_path == '-' or csv_path == '/dev/stdin':
        data = sys.stdin.read()
        from io import StringIO
        buf = StringIO(data)
        try:
            df = pd.read_csv(buf)
        except pd.errors.EmptyDataError:
            return _empty_trade_frame()
        except Exception:
            buf.seek(0)
            try:
                df = pd.read_csv(buf, header=None)
            except pd.errors.EmptyDataError:
                return _empty_trade_frame()
    else:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"No such file: {csv_path}")
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return _empty_trade_frame()
        except Exception:
            try:
                df = pd.read_csv(p, header=None)
            except pd.errors.EmptyDataError:
                return _empty_trade_frame()

    # Normalize columns to lowercase strings
    df.columns = [str(c).strip().lower() for c in df.columns]

    normalized_cols = {"entry_time", "exit_time", "entry_price", "exit_price", "pnl"}
    if normalized_cols.issubset(set(df.columns)):
        return df
    # If we detect any of the expected names, map them; else treat as no-header
    if set(df.columns) & set(EXPECTED_COLS):
        # Ensure all expected cols exist (fill by position if missing)
        rename = {}
        for col in list(df.columns):
            if col in EXPECTED_COLS:
                continue
            # attempt common aliases
            if col in ("time", "datetime"):
                rename[col] = "timestamp"
        if rename:
            df = df.rename(columns=rename)
        # If missing expected columns, try to backfill from left-to-right
        for i, name in enumerate(EXPECTED_COLS):
            if name not in df.columns and i < len(df.columns):
                df[name] = df.iloc[:, i]
    else:
        # Assume no header, map by position then append any extra cols
        base = df.copy()
        df = pd.DataFrame()
        for i, name in enumerate(EXPECTED_COLS):
            if i < base.shape[1]:
                df[name] = base.iloc[:, i]
            else:
                df[name] = np.nan
        # Append remaining unnamed columns with generic names c9, c10, ...
        for j in range(len(EXPECTED_COLS), base.shape[1]):
            df[f"c{j+1}"] = base.iloc[:, j]

    return df


def _to_datetime(s: pd.Series, tz: Optional[str]) -> pd.Series:
    # Parse with UTC to tolerate mixed/offset-aware timestamps without future warnings.
    dt = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d %H:%M:%S", utc=True)
    if dt.isna().all():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            dt = pd.to_datetime(s, errors="coerce", utc=True)
    elif dt.isna().any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            fallback = pd.to_datetime(s, errors="coerce", utc=True)
        dt = dt.fillna(fallback)
    if tz:
        try:
            dt = dt.dt.tz_convert(tz)
        except Exception:
            # If conversion fails, drop tz info to avoid crashing stats
            dt = dt.dt.tz_localize(None)
    return dt


def _adverse_slip(price: float, side: str, slip_pts: float, is_entry: bool) -> float:
    """Apply adverse slippage in POINTS (not ticks)."""
    if side.upper() == 'LONG':
        return price + slip_pts if is_entry else price - slip_pts
    elif side.upper() == 'SHORT':
        return price - slip_pts if is_entry else price + slip_pts
    return price


def _r_points(entry: float, stop: Optional[float], side: str) -> Optional[float]:
    if stop is None or (isinstance(stop, float) and math.isnan(stop)):
        return None
    if side.upper() == 'LONG':
        return max(entry - stop, 0.0) or None
    if side.upper() == 'SHORT':
        return max(stop - entry, 0.0) or None
    return None


def _streaks(bools: pd.Series) -> Tuple[int, int]:
    """Return (max_win_streak, max_loss_streak) for a boolean Series of wins."""
    max_w = max_l = cur_w = cur_l = 0
    for v in bools.astype(bool):
        if v:
            cur_w += 1; max_w = max(max_w, cur_w)
            cur_l = 0
        else:
            cur_l += 1; max_l = max(max_l, cur_l)
            cur_w = 0
    return max_w, max_l


def _max_drawdown(equity: pd.Series) -> Tuple[float, float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    roll_max = equity.cummax()
    dd = equity - roll_max
    trough_idx = dd.idxmin() if not dd.empty else None
    peak_value = roll_max.loc[:trough_idx].max() if trough_idx is not None else np.nan
    trough_value = equity.loc[trough_idx] if trough_idx is not None else np.nan
    mdd = float(peak_value - trough_value) if (not np.isnan(peak_value) and not np.isnan(trough_value)) else 0.0
    # find peak time
    peak_idx = roll_max.loc[:trough_idx].idxmax() if trough_idx is not None else None
    return mdd, float(trough_value) if not np.isnan(trough_value) else 0.0, peak_idx, trough_idx


# --------------------- Core computation ---------------------

def compute_pnl(
    df: pd.DataFrame,
    instrument: str,
    default_contracts: int,
    commission: float,
    slip_ticks: float,
    tick_size: Optional[float],
    tz: Optional[str],
    *,
    ignore_contracts_column: bool = False,
) -> (pd.DataFrame, Dict[str, Any], pd.DataFrame):
    ins = instrument.upper()
    point_value = POINT_VALUE_BY_INSTRUMENT.get(ins)
    tick_sz = float(tick_size) if tick_size is not None else TICK_SIZE_BY_INSTRUMENT.get(ins)

    if point_value is None or tick_sz is None:
        spec = None
        if _instrument_lookup is not None:
            try:
                spec = _instrument_lookup(ins)
            except Exception:
                spec = None
        if spec:
            point_value = point_value or spec.point_value
            if tick_size is None:
                tick_sz = spec.tick_size

    if point_value is None:
        raise ValueError(
            f"Unknown instrument '{instrument}'. Known: {sorted(POINT_VALUE_BY_INSTRUMENT.keys())}"
        )
    if tick_sz is None:
        raise ValueError(f"Unknown tick size for '{instrument}'. Use --tick-size to set explicitly.")

    df = df.copy()
    df['timestamp'] = _to_datetime(df['timestamp'], tz)
    df = df.sort_values('timestamp', kind='mergesort').reset_index(drop=True)

    # Numeric coercions
    for col in ['price','prob','stop','target']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalize strings
    for col in ['action','side','grade']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()
        else:
            df[col] = np.nan

    # Optional per-row contracts
    row_contracts = None
    if not ignore_contracts_column:
        for cname in OPTIONAL_SIZE_COLS:
            if cname in df.columns:
                row_contracts = pd.to_numeric(df[cname], errors='coerce').fillna(default_contracts).astype(int)
                break
    if row_contracts is None:
        row_contracts = pd.Series(default_contracts, index=df.index, dtype=int)

    # Iterate rows, pair OPEN->CLOSE, and support FLIP
    trades: List[Dict[str, Any]] = []
    open_trade: Optional[Dict[str, Any]] = None
    slip_points = slip_ticks * tick_sz  # ticks -> points

    def _close_trade(close_ts, exit_raw_price, exit_prob):
        nonlocal open_trade
        if open_trade is None or exit_raw_price is None or np.isnan(exit_raw_price):
            return None
        ot = open_trade
        exit_price = _adverse_slip(exit_raw_price, ot['side'], slip_points, is_entry=False)
        pts = (exit_price - ot['entry_price']) if ot['side'] == 'LONG' else (ot['entry_price'] - exit_price)
        dollars = pts * ot['point_value'] * ot['contracts']
        rt_cost = ot['commission_per_side'] * 2.0 * ot['contracts']
        dollars_net = dollars - rt_cost
        r_pts = _r_points(ot['entry_price_raw'], ot['stop'], ot['side'])
        r_mult = (pts / r_pts) if r_pts and r_pts > 0 else None
        rec = {
            **ot,
            'close_time': close_ts,
            'exit_price': exit_price,
            'exit_price_raw': exit_raw_price,
            'exit_prob': exit_prob,
            'pts': pts,
            'dollars_gross': dollars,
            'dollars_net': dollars_net,
            'r_mult': r_mult,
        }
        trades.append(rec)
        open_trade = None
        return rec

    for idx, row in df.iterrows():
        ts = row['timestamp']
        action = row['action']
        side = row['side']
        price = float(row['price']) if not pd.isna(row['price']) else None
        prob = None if pd.isna(row['prob']) else float(row['prob'])
        grade = None if (isinstance(row['grade'], float) and math.isnan(row['grade'])) else str(row['grade']) if row['grade'] is not None else None
        stop = None if pd.isna(row['stop']) else float(row['stop'])
        target = None if pd.isna(row['target']) else float(row['target'])
        contracts = int(row_contracts.loc[idx])

        if action == 'OPEN':
            # Force-close existing at this new OPEN price
            if open_trade is not None and price is not None:
                _close_trade(ts, price, prob)
            if price is not None and side in ('LONG','SHORT'):
                adj_entry = _adverse_slip(price, side, slip_points, is_entry=True)
                open_trade = {
                    'instrument': ins,
                    'contracts': contracts,
                    'commission_per_side': commission,
                    'point_value': point_value,
                    'tick_size': tick_sz,
                    'open_time': ts,
                    'side': side,
                    'entry_price': adj_entry,
                    'entry_price_raw': price,
                    'entry_prob': prob,
                    'grade': grade,
                    'stop': stop,
                    'target': target,
                }

        elif action == 'CLOSE':
            _close_trade(ts, price, prob)

        elif action == 'FLIP':
            # Close existing, then open new (side must be LONG/SHORT)
            _closed = _close_trade(ts, price, prob)
            if price is not None and side in ('LONG','SHORT'):
                adj_entry = _adverse_slip(price, side, slip_points, is_entry=True)
                open_trade = {
                    'instrument': ins,
                    'contracts': contracts,
                    'commission_per_side': commission,
                    'point_value': point_value,
                    'tick_size': tick_sz,
                    'open_time': ts,
                    'side': side,
                    'entry_price': adj_entry,
                    'entry_price_raw': price,
                    'entry_prob': prob,
                    'grade': grade,
                    'stop': stop,
                    'target': target,
                }
        else:
            # ignore
            continue

    trade_df = pd.DataFrame(trades)
    if trade_df.empty:
        stats = {
            'instrument': ins,
            'contracts_default': default_contracts,
            'commission_per_side': commission,
            'slip_ticks': slip_ticks,
            'tick_size': tick_sz,
            'point_value': point_value,
            'trades': 0, 'wins': 0, 'losses': 0,
            'win_rate_pct': 0.0,
            'gross_points': 0.0, 'net_points': 0.0,
            'gross_dollars': 0.0, 'net_dollars': 0.0,
        }
        return trade_df, stats, pd.DataFrame()

    # Per-trade metrics
    trade_df['day'] = trade_df['close_time'].dt.tz_convert(None) if getattr(trade_df['close_time'].dt, 'tz', None) is not None else trade_df['close_time']
    trade_df['day'] = trade_df['day'].dt.date
    trade_df['win'] = trade_df['pts'] > 0
    trade_df['loss'] = trade_df['pts'] < 0

    # Equity curve (net dollars cumulative)
    trade_df = trade_df.sort_values('close_time').reset_index(drop=True)
    equity = trade_df['dollars_net'].cumsum()
    trade_df['equity'] = equity

    # Summaries
    wins = int(trade_df['win'].sum())
    losses = int(trade_df['loss'].sum())
    trades_n = len(trade_df)
    win_rate = (wins / trades_n * 100.0) if trades_n else 0.0

    gross_pts = float(trade_df['pts'].sum())
    gross_usd = float(trade_df['dollars_gross'].sum())
    net_usd   = float(trade_df['dollars_net'].sum())
    net_pts = net_usd / (point_value * trade_df['contracts'].mean()) if point_value else np.nan

    avg_win_pts = float(trade_df.loc[trade_df['win'], 'pts'].mean()) if wins else np.nan
    avg_loss_pts = float(trade_df.loc[trade_df['loss'], 'pts'].mean()) if losses else np.nan
    avg_win_usd = float(trade_df.loc[trade_df['win'], 'dollars_net'].mean()) if wins else np.nan
    avg_loss_usd = float(trade_df.loc[trade_df['loss'], 'dollars_net'].mean()) if losses else np.nan

    pf = (trade_df.loc[trade_df['win'],'dollars_net'].sum() /
          abs(trade_df.loc[trade_df['loss'],'dollars_net'].sum())) if losses else float('inf')
    payoff = (abs(avg_win_usd) / abs(avg_loss_usd)) if (not math.isnan(avg_win_usd) and not math.isnan(avg_loss_usd) and avg_loss_usd != 0) else np.nan
    expectancy = net_usd / trades_n if trades_n else 0.0

    # R-metrics
    r_mults = trade_df['r_mult'].dropna()
    avg_r = float(r_mults.mean()) if not r_mults.empty else np.nan
    med_r = float(r_mults.median()) if not r_mults.empty else np.nan

    # Streaks & drawdown
    max_w_streak, max_l_streak = _streaks(trade_df['win'])
    mdd_abs, trough_val, peak_ts, trough_ts = _max_drawdown(trade_df['equity'])

    # Daily / Monthly breakdowns
    daily = (trade_df.groupby('day')
             .agg(trades=('pts','count'),
                  wins=('win','sum'),
                  losses=('loss','sum'),
                  net_pts=('pts','sum'),
                  net_usd=('dollars_net','sum'))
             .reset_index())

    close_times = pd.to_datetime(trade_df['close_time'], errors='coerce')
    if getattr(close_times.dt, 'tz', None) is not None:
        close_times = close_times.dt.tz_convert(None)
    trade_df['month'] = close_times.dt.to_period('M').astype(str)
    monthly = (trade_df.groupby('month')
               .agg(trades=('pts','count'),
                    wins=('win','sum'),
                    losses=('loss','sum'),
                    net_usd=('dollars_net','sum'))
               .reset_index())

    stats = {
        'instrument': ins,
        'contracts_default': default_contracts,
        'commission_per_side': commission,
        'slip_ticks': slip_ticks,
        'tick_size': tick_sz,
        'point_value': point_value,
        'trades': trades_n,
        'wins': wins,
        'losses': losses,
        'win_rate_pct': round(win_rate, 2),
        'gross_points': round(gross_pts, 4),
        'net_points': round(float(net_pts), 4) if not np.isnan(net_pts) else None,
        'gross_dollars': round(gross_usd, 2),
        'net_dollars': round(net_usd, 2),
        'avg_win_pts': round(avg_win_pts, 4) if not math.isnan(avg_win_pts) else None,
        'avg_loss_pts': round(avg_loss_pts, 4) if not math.isnan(avg_loss_pts) else None,
        'avg_win_usd': round(avg_win_usd, 2) if not math.isnan(avg_win_usd) else None,
        'avg_loss_usd': round(avg_loss_usd, 2) if not math.isnan(avg_loss_usd) else None,
        'profit_factor': round(pf, 4) if math.isfinite(pf) else None,
        'payoff_ratio': round(payoff, 4) if not np.isnan(payoff) else None,
        'expectancy_usd_per_trade': round(expectancy, 2),
        'avg_r_multiple': round(avg_r, 4) if not math.isnan(avg_r) else None,
        'median_r_multiple': round(med_r, 4) if not math.isnan(med_r) else None,
        'max_win_streak': int(max_w_streak),
        'max_loss_streak': int(max_l_streak),
        'max_drawdown_abs_usd': round(mdd_abs, 2),
        'equity_trough_usd': round(trough_val, 2),
        'equity_peak_time': str(peak_ts) if peak_ts is not None else None,
        'equity_trough_time': str(trough_ts) if trough_ts is not None else None,
    }

    # Attach breakdowns for optional output by caller
    stats['_daily'] = daily
    stats['_monthly'] = monthly

    return trade_df, stats, daily


# --------------------- Compatibility helpers ---------------------

def load_trades_dataframe(
    trades_csv_path: str,
    instrument: str,
    contracts: int = 1,
    tz: str = "America/Denver",
    commission_per_round_turn: float = 0.0,
    slip_ticks_per_side: float = 0.0,
    ignore_contracts_column: bool = False,
) -> pd.DataFrame:
    """
    Convert a trade log CSV into a normalized trade-level DataFrame.

    Accepts either the raw OPEN/CLOSE style logs consumed by compute_pnl or the simpler
    `entry_time/exit_time` CSVs produced by stream_live_csv/preset_eval.
    """
    raw = _read_any(trades_csv_path)
    normalized_cols = {"entry_time", "exit_time", "entry_price", "exit_price", "pnl"}
    lower_cols = set(raw.columns)
    if normalized_cols.issubset(lower_cols):
        simple = raw.copy()
        symbol_col = simple["symbol"] if "symbol" in simple.columns else instrument
        simple["symbol"] = symbol_col if isinstance(symbol_col, pd.Series) else instrument
        size_series = None
        for cand in ("size", "contracts", "qty"):
            if cand in simple.columns:
                size_series = simple[cand]
                break
        if size_series is None:
            size_series = 1.0
        simple["size"] = size_series
        for col in ("entry_time", "exit_time"):
            simple[col] = pd.to_datetime(simple[col], errors="coerce")
        for col in ("entry_price", "exit_price", "pnl", "size"):
            simple[col] = pd.to_numeric(simple[col], errors="coerce")
        simple["symbol"] = simple["symbol"].fillna(instrument).astype(str).str.upper()
        return simple[["symbol", "entry_time", "exit_time", "entry_price", "exit_price", "size", "pnl"]]

    entry_exit_cols = {"entry_ts", "exit_ts", "entry_price", "exit_price"}
    if entry_exit_cols.issubset(lower_cols):
        simple = raw.copy()
        if "symbol" not in simple.columns:
            simple["symbol"] = instrument
        size_series = None
        for cand in ("qty", "size", "contracts"):
            if cand in simple.columns:
                size_series = pd.to_numeric(simple[cand], errors="coerce")
                break
        if size_series is None:
            size_series = pd.Series(contracts, index=simple.index, dtype=float)
        simple["size"] = size_series.fillna(contracts).astype(float)
        simple["entry_time"] = pd.to_datetime(simple.get("entry_ts"), errors="coerce")
        simple["exit_time"] = pd.to_datetime(simple.get("exit_ts"), errors="coerce", utc=True)
        simple["entry_price"] = pd.to_numeric(simple.get("entry_price"), errors="coerce")
        simple["exit_price"] = pd.to_numeric(simple.get("exit_price"), errors="coerce")
        simple["symbol"] = simple["symbol"].fillna(instrument).astype(str).str.upper()
        return simple

    per_side_commission = float(commission_per_round_turn or 0.0) / 2.0
    trade_df, _stats, _daily = compute_pnl(
        raw,
        instrument=instrument,
        default_contracts=contracts,
        commission=per_side_commission,
        slip_ticks=slip_ticks_per_side,
        tick_size=None,
        tz=tz,
        ignore_contracts_column=ignore_contracts_column,
    )
    if trade_df.empty:
        return pd.DataFrame(
            columns=["symbol", "entry_time", "exit_time", "entry_price", "exit_price", "size", "pnl"]
        )
    entry_prices = trade_df["entry_price_raw"] if "entry_price_raw" in trade_df.columns else trade_df["entry_price"]
    exit_prices = trade_df["exit_price_raw"] if "exit_price_raw" in trade_df.columns else trade_df["exit_price"]
    simple = pd.DataFrame(
        {
            "symbol": trade_df["instrument"],
            "entry_time": trade_df["open_time"],
            "exit_time": trade_df["close_time"],
            "entry_price": entry_prices,
            "exit_price": exit_prices,
            "size": trade_df["contracts"],
            "pnl": trade_df["dollars_net"],
        }
    )
    return simple


def _resolve_instrument_meta(symbol: str) -> tuple[float, float]:
    """
    Return (point_value, tick_size) for an instrument symbol with safe fallbacks.
    """
    alias = str(symbol).upper()
    point_value = POINT_VALUE_BY_INSTRUMENT.get(alias)
    tick_size = TICK_SIZE_BY_INSTRUMENT.get(alias)
    if (point_value is None or tick_size is None) and _instrument_lookup is not None:
        try:
            spec = _instrument_lookup(alias)
            point_value = point_value if point_value is not None else getattr(spec, "point_value", None)
            tick_size = tick_size if tick_size is not None else getattr(spec, "tick_size", None)
        except Exception:
            spec = None
    # Preserve historical defaults if the instrument is still unknown
    return float(point_value or 50.0), float(tick_size or 0.25)


def _compute_trade_pnl_usd(trade_df: pd.DataFrame) -> pd.Series:
    """
    Compute per-trade PnL in USD using consistent tick-rounded math.
    """
    df = trade_df.copy()
    # Normalize column names that we expect downstream
    rename_map = {}
    if "entry_ts" in df.columns and "entry_time" not in df.columns:
        rename_map["entry_ts"] = "entry_time"
    if "exit_ts" in df.columns and "exit_time" not in df.columns:
        rename_map["exit_ts"] = "exit_time"
    if "qty" in df.columns and "size" not in df.columns:
        rename_map["qty"] = "size"
    if rename_map:
        df = df.rename(columns=rename_map)

    symbol_series = None
    for cand in ("symbol", "instrument"):
        if cand in df.columns:
            symbol_series = df[cand]
            break
    if symbol_series is None:
        symbol_series = pd.Series("ES", index=df.index)
    if not isinstance(symbol_series, pd.Series):
        symbol_series = pd.Series(symbol_series, index=df.index)
    symbol_series = symbol_series.ffill().bfill().fillna("ES").astype(str).str.upper()

    side = df.get("side", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
    entry_price_col = "entry_price_raw" if "entry_price_raw" in df.columns else "entry_price"
    exit_price_col = "exit_price_raw" if "exit_price_raw" in df.columns else "exit_price"
    entry_price = pd.to_numeric(df.get(entry_price_col), errors="coerce")
    exit_price = pd.to_numeric(df.get(exit_price_col), errors="coerce")

    size_series = None
    for cand in ("size", "contracts"):
        if cand in df.columns:
            size_series = pd.to_numeric(df[cand], errors="coerce")
            break
    if size_series is None:
        size_series = pd.Series(1.0, index=df.index, dtype=float)
    size_series = size_series.fillna(1.0).astype(float)

    meta_cache: Dict[str, tuple[float, float]] = {}

    def meta_for(alias: str) -> tuple[float, float]:
        if alias not in meta_cache:
            meta_cache[alias] = _resolve_instrument_meta(alias)
        return meta_cache[alias]

    point_values = symbol_series.map(lambda a: meta_for(a)[0])
    tick_sizes = symbol_series.map(lambda a: meta_for(a)[1])

    direction = side.map({"LONG": 1.0, "SHORT": -1.0})
    points = (exit_price - entry_price) * direction
    valid_ticks = tick_sizes > 0
    ticks = np.where(valid_ticks & points.notna(), np.rint(points / tick_sizes), np.nan)
    snapped_points = ticks * tick_sizes
    pnl_usd = snapped_points * point_values * size_series
    return pd.Series(pnl_usd, index=df.index)


def compute_pnl_stats_from_trades(
    trade_df: pd.DataFrame, *, day_tz: str = "America/Denver", last_n: Optional[int] = None
) -> Dict[str, Any]:
    """
    Simplified stats summary used by the bot orchestration layer.
    """
    if trade_df.empty:
        return {
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "num_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_daily_loss": 0.0,
        }

    df = trade_df.copy()
    rename_map = {}
    if "entry_ts" in df.columns and "entry_time" not in df.columns:
        rename_map["entry_ts"] = "entry_time"
    if "exit_ts" in df.columns and "exit_time" not in df.columns:
        rename_map["exit_ts"] = "exit_time"
    if "qty" in df.columns and "size" not in df.columns:
        rename_map["qty"] = "size"
    if rename_map:
        df = df.rename(columns=rename_map)

    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
    bad_ts_mask = df["exit_time"].isna()
    if bad_ts_mask.any():
        # Trades without an exit timestamp are still open/incomplete. For the
        # shutdown summary we should skip them rather than raising and masking
        # the rest of the completed trade stats.
        df = df.loc[~bad_ts_mask].copy()
        if df.empty:
            return {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "num_trades": 0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_daily_loss": 0.0,
                "profit_factor": 0.0,
                "max_loss_streak": 0,
            }
    exit_times = df["exit_time"]
    exit_local = exit_times.dt.tz_convert(day_tz)
    df["day"] = exit_local.dt.date

    pnl_usd = _compute_trade_pnl_usd(df)
    if pnl_usd.isna().all() and "pnl" in df.columns:
        pnl_usd = pd.to_numeric(df["pnl"], errors="coerce")
    df["pnl_usd"] = pnl_usd
    df = df.dropna(subset=["pnl_usd"])
    df = df.sort_values("exit_time")
    if last_n is not None and last_n > 0:
        df = df.tail(int(last_n))

    df["cum_pnl"] = df["pnl_usd"].cumsum()
    max_equity = df["cum_pnl"].cummax()
    drawdown = df["cum_pnl"] - max_equity
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    wins = df[df["pnl_usd"] > 0]["pnl_usd"]
    losses = df[df["pnl_usd"] < 0]["pnl_usd"]
    num_trades = len(df)
    win_rate = float(len(wins) / num_trades) if num_trades else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    total_pnl = float(df["pnl_usd"].sum())
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    daily_pnl = df.groupby("day")["pnl_usd"].sum()
    max_daily_loss = float(daily_pnl.min()) if len(daily_pnl) else 0.0
    _, max_loss_streak = _streaks(df["pnl_usd"] > 0)

    return {
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "num_trades": num_trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_daily_loss": max_daily_loss,
        "profit_factor": profit_factor,
        "max_loss_streak": int(max_loss_streak),
    }


def compute_pnl_stats(
    trades_csv_path: str,
    instrument: str,
    contracts: int = 1,
    tz: str = "America/Denver",
    commission_per_round_turn: float = 0.0,
    slip_ticks_per_side: float = 0.0,
    ignore_contracts_column: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    """
    Convenience wrapper returning a lightweight stats dict for a trades CSV path.
    """
    trades_df = load_trades_dataframe(
        trades_csv_path,
        instrument=instrument,
        contracts=contracts,
        tz=tz,
        commission_per_round_turn=commission_per_round_turn,
        slip_ticks_per_side=slip_ticks_per_side,
        ignore_contracts_column=ignore_contracts_column,
    )
    return compute_pnl_stats_from_trades(trades_df)


def compare_runs(stats_before: Dict[str, Any], stats_after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare baseline vs post-learning stats.
    """
    wr_before = float(stats_before.get("win_rate") or 0.0)
    wr_after = float(stats_after.get("win_rate") or 0.0)
    dd_before = float(stats_before.get("max_drawdown") or 0.0)
    dd_after = float(stats_after.get("max_drawdown") or 0.0)

    wr_delta = wr_after - wr_before
    dd_ok = dd_after >= dd_before - 500.0
    learning_effective = (wr_delta >= -0.02) and dd_ok

    return {
        "win_rate_before": wr_before,
        "win_rate_after": wr_after,
        "win_rate_delta": wr_delta,
        "max_dd_before": dd_before,
        "max_dd_after": dd_after,
        "learning_effective": learning_effective,
    }


# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Local P&L calculator for OPEN/CLOSE/FLIP trade logs with equity/stats.")
    ap.add_argument('csv', help="Path to CSV (or '-' for stdin)")
    ap.add_argument('--instrument', default='ES', help="Instrument code (ES, MES, NQ, MNQ, YM, GC, CL)")
    ap.add_argument('--contracts', type=int, default=1, help="Default contracts per trade (can be overridden by CSV 'contracts'/'size' col)")
    ap.add_argument('--commission', type=float, default=0.0, help="Commission per side per contract in USD (default 0)")
    ap.add_argument('--slip-ticks', type=float, default=0.0, help="Adverse slippage in ticks per entry/exit (default 0)")
    ap.add_argument('--tick-size', type=float, default=None, help="Override tick size (points per tick). e.g., ES=0.25")
    ap.add_argument('--tz', default='America/Denver', help="Timezone for timestamps (default America/Denver)")
    ap.add_argument('--out', default=None, help="Write per-trade results to CSV path")
    ap.add_argument('--json', dest='json_out', default=None, help="Write summary stats JSON to this path")
    ap.add_argument('--equity', dest='equity_out', default=None, help="Write equity curve CSV (close_time,equity)")
    ap.add_argument('--price-dp', type=int, default=2, help="Round printed dollar amounts to N decimals for console output")
    args = ap.parse_args()

    try:
        df_raw = _read_any(args.csv)
        trades_df, stats, daily = compute_pnl(
            df_raw,
            instrument=args.instrument,
            default_contracts=args.contracts,
            commission=args.commission,
            slip_ticks=args.slip_ticks,
            tick_size=args.tick_size,
            tz=args.tz,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Console pretty print
    pd.set_option('display.width', 140)
    pd.set_option('display.max_columns', 20)

    print("\n===== P&L SUMMARY =====")
    head = [
        ('instrument', stats['instrument']),
        ('contracts_default', stats['contracts_default']),
        ('commission_per_side', stats['commission_per_side']),
        ('slip_ticks', stats['slip_ticks']),
        ('tick_size', stats['tick_size']),
        ('point_value', stats['point_value']),
    ]
    for k, v in head:
        print(f"{k}: {v}")
    print("-----------------------")
    print(f"trades: {stats['trades']} | wins: {stats['wins']} | losses: {stats['losses']} | win_rate: {stats['win_rate_pct']}%")
    print(f"gross_points: {stats['gross_points']} | net_points: {stats['net_points']}")
    print(f"gross_dollars: ${stats['gross_dollars']:,} | net_dollars: ${stats['net_dollars']:,}")
    print(f"avg_win_pts: {stats['avg_win_pts']} | avg_loss_pts: {stats['avg_loss_pts']}")
    print(f"avg_win_usd: {stats['avg_win_usd']} | avg_loss_usd: {stats['avg_loss_usd']}")
    pf = stats['profit_factor'] if stats['profit_factor'] is not None else 'N/A'
    payoff = stats['payoff_ratio'] if stats['payoff_ratio'] is not None else 'N/A'
    exp = stats['expectancy_usd_per_trade']
    print(f"profit_factor: {pf} | payoff_ratio: {payoff} | expectancy_usd_per_trade: ${exp}")
    print(f"max_win_streak: {stats['max_win_streak']} | max_loss_streak: {stats['max_loss_streak']}")
    print(f"max_drawdown_abs_usd: ${stats['max_drawdown_abs_usd']:,} | equity_trough_usd: ${stats['equity_trough_usd']:,}")

    if not trades_df.empty:
        # Daily
        daily = stats.get('_daily')
        if isinstance(daily, pd.DataFrame) and not daily.empty:
            print("\n===== DAILY =====")
            for _, r in daily.iterrows():
                print(f"{r['day']} | trades {int(r['trades'])} | W {int(r['wins'])} L {int(r['losses'])} | net_pts {r['net_pts']:.2f} | net_usd ${r['net_usd']:.2f}")

        # Monthly
        monthly = stats.get('_monthly')
        if isinstance(monthly, pd.DataFrame) and not monthly.empty:
            print("\n===== MONTHLY =====")
            for _, r in monthly.iterrows():
                print(f"{r['month']} | trades {int(r['trades'])} | W {int(r['wins'])} L {int(r['losses'])} | net_usd ${r['net_usd']:.2f}")

    # Optional outputs
    if args.out:
        outp = Path(args.out)
        trades_df.to_csv(outp, index=False)
        print(f"\n[WROTE] per-trade CSV -> {outp.resolve()}")

    if args.json_out:
        # Convert non-serializable parts
        stats_copy = {k: (v if not isinstance(v, (pd.DataFrame, pd.Series)) else None) for k, v in stats.items()}
        jp = Path(args.json_out)
        with open(jp, 'w', encoding='utf-8') as f:
            json.dump(stats_copy, f, indent=2)
        print(f"[WROTE] stats JSON -> {jp.resolve()}")

    if args.equity_out:
        ep = Path(args.equity_out)
        eq = trades_df[['close_time','equity']].copy()
        eq.to_csv(ep, index=False)
        print(f"[WROTE] equity CSV -> {ep.resolve()}")


if __name__ == '__main__':
    main()
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually",
    category=UserWarning,
)
