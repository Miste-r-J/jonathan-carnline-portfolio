from __future__ import annotations
"""
market_replay_models.py — Historical market replay with execution (PRODUCTION vNext)

Key improvements vs. your previous version:
- FIX: Proper parquet writing for NewsRiskFilter.
- FIX: Defensive cleaning of NaNs in desired position before sim.
- FIX: Prop breach flatten uses the same speed-aware slippage logic as normal fills.
- FIX: Pandas deprecation (.view -> .astype) for timestamp math.
- ADD: Optional clipping of TR-per-second to avoid pathological slippage explosions.
- ADD: Start/resume controls: --start_at, --end_at, --resume_from_stream.
- ADD: FOLLOW mode: --follow makes the process wait for new bars and keep emitting until you exit.
- CLARITY: Reconcile CSV includes exec commissions & net series; headers clarified.
- TZ: Keep timestamps UTC-aware consistently through the pipeline.
- LOGGING: Per-bar log prints realized cumulative PnL only (wins/losses remain round-trip in summary).
- REPRO: Execution realism parameters and RNG seed recorded in summary.
- STREAM: Reliable L1 emitter for L2 bridge (--emit_stream_csv): absolute path, atomic header, per-row flush+fsync.
"""

import argparse
import json
import time
import os
from pathlib import Path, PurePath
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

_hydrate = None
try:
    from .backtest_models import _hydrate_cost_risk_engine as _hydrate  # type: ignore
except Exception:
    try:
        from .backtest_model import _hydrate_cost_risk_engine as _hydrate  # type: ignore
    except Exception:
        _hydrate = None

_run_backtest = None
try:
    from .backtest_model import run_backtest as _run_backtest  # type: ignore
except Exception:
    try:
        from .backtest_models import run_backtest as _run_backtest  # type: ignore
    except Exception:
        _run_backtest = None

try:
    from .config import PRESETS, CONTRACT_COST, ENGINE, instrument_by_alias  # type: ignore
except Exception:
    from config import PRESETS, CONTRACT_COST, ENGINE, instrument_by_alias  # type: ignore

try:
    from .models import model_path as resolve_model_path  # type: ignore
except Exception:
    from models import model_path as resolve_model_path  # type: ignore

try:
    from .news_filter import NewsRiskFilter  # type: ignore
except Exception:
    NewsRiskFilter = None  # type: ignore


# -----------------------
# Helpers (data prep)
# -----------------------

def _flatten_multiindex_columns(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        if getattr(df.columns, "duplicated", None) is not None and df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="last")]
        return df

    def _slice(level_idx: int, key: str) -> Optional[pd.DataFrame]:
        try:
            level_vals = list(df.columns.get_level_values(level_idx))
            if key in level_vals:
                out = df.xs(key, axis=1, level=level_idx, drop_level=True)
                return out
        except Exception:
            return None
        return None

    if symbol:
        out = _slice(1, symbol) or _slice(0, symbol)
        if out is not None:
            if isinstance(out.columns, pd.MultiIndex):
                return _flatten_multiindex_columns(out, symbol=None)
            if getattr(out.columns, "duplicated", None) is not None and out.columns.duplicated().any():
                out = out.loc[:, ~out.columns.duplicated(keep="last")]
            return out

    try:
        lv0 = pd.Index(df.columns.get_level_values(0)).unique()
        lv1 = pd.Index(df.columns.get_level_values(1)).unique()
        if len(lv1) == 1:
            return _flatten_multiindex_columns(df.droplevel(1, axis=1), symbol=None)
        if len(lv0) == 1:
            return _flatten_multiindex_columns(df.droplevel(0, axis=1), symbol=None)
    except Exception:
        pass

    out = df.copy()
    out.columns = ["_".join([str(p) for p in tup if p is not None]) for tup in out.columns.to_list()]
    if getattr(out.columns, "duplicated", None) is not None and out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated(keep="last")]
    return out


def _ensure_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ren = {}
    for c in list(df.columns):
        lc = str(c).lower()
        if lc in ("datetime", "date", "time", "timestamp"):
            ren[c] = "Datetime"
        elif lc == "open" or lc.endswith("_open"):
            ren[c] = "Open"
        elif lc == "high" or lc.endswith("_high"):
            ren[c] = "High"
        elif lc == "low" or lc.endswith("_low"):
            ren[c] = "Low"
        elif lc == "close" or lc.endswith("_close"):
            ren[c] = "Close"
        elif lc == "volume" or lc.endswith("_volume"):
            ren[c] = "Volume"
    if ren:
        df = df.rename(columns=ren)

    def _pick_series(name: str) -> pd.Series:
        if name not in df.columns:
            candidates = [c for c in df.columns if str(c).lower().endswith(name.lower())]
            if not candidates:
                raise ValueError(f"Input missing required column: {name}")
            s = df[candidates[0]]
        else:
            s = df[name]
        if isinstance(s, pd.DataFrame):
            for col in s.columns[::-1]:
                if not s[col].isna().all():
                    return s[col]
            return s.iloc[:, -1]
        return s

    # Keep UTC-aware timestamps throughout
    dt = pd.to_datetime(_pick_series("Datetime"), errors="coerce", utc=True)

    out = pd.DataFrame({
        "Datetime": dt,
        "Open": pd.to_numeric(_pick_series("Open"), errors="coerce"),
        "High": pd.to_numeric(_pick_series("High"), errors="coerce"),
        "Low": pd.to_numeric(_pick_series("Low"), errors="coerce"),
        "Close": pd.to_numeric(_pick_series("Close"), errors="coerce"),
    })

    try:
        vol_series = pd.to_numeric(_pick_series("Volume"), errors="coerce")
    except Exception:
        vol_series = pd.Series(0.0, index=out.index)
    out["Volume"] = vol_series.fillna(0.0)

    out = out.dropna(subset=["Datetime", "Open", "High", "Low", "Close"]).sort_values("Datetime").reset_index(drop=True)
    return out[["Datetime", "Open", "High", "Low", "Close", "Volume"]]


def fetch_yf(symbol: str, *, yf_period: Optional[str] = None,
             start_date: Optional[str] = None, end_date: Optional[str] = None,
             interval: str = "5m") -> pd.DataFrame:
    if yf is None:
        raise SystemExit("Please `pip install yfinance` to use yfinance source.")
    kw = dict(interval=interval, auto_adjust=False, prepost=False, threads=False, progress=False)
    if yf_period:
        raw = yf.download(tickers=symbol, period=yf_period, **kw)
    else:
        raw = yf.download(tickers=symbol, start=start_date, end=end_date, **kw)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned empty data; check symbol/period/market hours.")

    df = _flatten_multiindex_columns(raw, symbol=symbol).reset_index()  # add 'Datetime'

    # Ensure UTC-aware, keep tz for consistency (no tz_localize(None))
    if getattr(df["Datetime"].dtype, "tz", None) is not None:
        df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_convert("UTC")
    else:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)

    ren = {}
    for want in ["Open", "High", "Low", "Close", "Volume"]:
        if want not in df.columns:
            for c in df.columns:
                if str(c).lower() == want.lower():
                    ren[c] = want
                    break
    if ren:
        df = df.rename(columns=ren)

    if any(w not in df.columns for w in ["Open", "High", "Low", "Close"]):
        for want in ["Open", "High", "Low", "Close", "Volume"]:
            if want not in df.columns:
                matches = [c for c in df.columns if str(c).lower().endswith(want.lower())]
                if matches:
                    df[want] = df[matches[0]]

    return _ensure_ohlcv_cols(df)


def _apply_preset_to_namespace(preset_name: Optional[str], ns: argparse.Namespace) -> argparse.Namespace:
    if not preset_name:
        return ns
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset '{preset_name}'. Available: {', '.join(sorted(PRESETS))}")
    keymap = {
        "max_trades": "max_trades_per_day",
        "per_stop_pct": "per_trade_stop_pct",
        "per_stop_usd": "per_trade_stop_usd",
        "daily_stop_pct": "daily_loss_stop_pct",
        "trail_dd_pct": "trailing_drawdown_pct",
        "daily_stop_usd": "daily_loss_stop_usd",
        "trail_dd_usd": "trail_dd_usd",  # retained for compatibility
        "prop_trail_dd_usd": "prop_trail_dd_usd",
        "stop_after_win": "stop_after_win",
        "trail_intrabar": "trail_intrabar",
        "instrument": "instrument",
        "p_buy": "p_buy",
        "p_sell": "p_sell",
        "allowed_grades": "allow_grades",
        "trade_window_start": "trade_window_start",
        "trade_window_end": "trade_window_end",
        "session_tz": "session_tz",
        "commission": "commission",
        "slip_ticks": "slip_ticks",
        "pos_cap": "pos_cap",
        "max_position": "max_position",
        "profit_lock_usd": "profit_lock_usd",
        "near_breach_buffer_usd": "near_breach_buffer_usd",
        "target_r": "target_r",
        "policy_margin": "policy_margin",
    }
    preset = PRESETS[preset_name]
    for k, v in preset.items():
        dest = keymap.get(k, k)
        if hasattr(ns, dest) and getattr(ns, dest) is None and dest not in ("vol_target",):
            setattr(ns, dest, v)
    if "vol_target" in preset and not getattr(ns, "vol_target", False):
        ns.vol_target = bool(preset["vol_target"])
    if "stop_after_win" in preset and not getattr(ns, "stop_after_win", False):
        ns.stop_after_win = bool(preset["stop_after_win"])
    if "trail_intrabar" in preset and not getattr(ns, "trail_intrabar", False):
        ns.trail_intrabar = bool(preset["trail_intrabar"])
    return ns


# -----------------------
# Execution simulator
# -----------------------

def simulate_exec(
    data: pd.DataFrame,
    instrument,
    commission_per_contract: float,
    slippage_ticks_per_side_base: float,
    *,
    order_type: str = "market",
    latency_ms: int = 120,
    jitter_ms: int = 40,
    participation: float = 0.05,
    base_spread_ticks: float | None = None,
    k_tr: float = 0.15,
    impact_k: float = 0.50,
    flat_at_end: bool = True,
    fill_mode: str = "align",               # "align" or "catch_up"
    enforce_breach_halt: bool = True,
    reconcile_path: str | None = None,
    seed: int = 7,
    tr_per_sec_clip: Optional[float] = None,  # optional: max ticks/sec clip (None = no clip)
) -> tuple[pd.DataFrame, int, int, float, int]:
    """
    fill_mode="align": target position(data['position'][i]) at bar i (matches ledger accrual on [i,i+1)).
    Writes reconciliation CSV (if reconcile_path provided):
      Columns include pnl_exec_mtm (gross), exec_commissions per bar, cum_exec_net, etc.
    """
    assert fill_mode in ("align", "catch_up")
    rng = np.random.default_rng(seed)

    for col in ("Datetime", "Open", "High", "Low", "Close", "Volume", "position"):
        if col not in data.columns:
            raise ValueError(f"simulate_exec requires '{col}' in backtest results")

    # Defensive clean for positions
    data = data.copy()
    data["position"] = pd.to_numeric(data["position"], errors="coerce").fillna(0.0)

    pos_desired = np.asarray(data["position"].values, dtype=float)

    if enforce_breach_halt and ("prop_breached" in data.columns):
        breached = data["prop_breached"].astype(bool).values
        if breached.any():
            first_breach = int(np.argmax(breached))
            pos_desired[first_breach:] = 0.0

    n = len(data)
    if n < 2:
        return pd.DataFrame(), 0, 0, 0.0, int(pos_desired[-1] if n else 0)

    ts = pd.to_datetime(data["Datetime"], utc=True)
    ts_ns = ts.astype("int64")
    dt_sec = np.diff(ts_ns).astype(float) / 1e9  # ns → s
    if len(dt_sec) == 0 or not np.isfinite(dt_sec).all():
        dt_sec = np.array([60.0] * (n - 1), dtype=float)
    dt_sec = np.append(dt_sec, dt_sec[-1])  # last bar duration proxy

    open_px = data["Open"].astype(float).values
    high = data["High"].astype(float).values
    low  = data["Low"].astype(float).values
    close = data["Close"].astype(float).values
    vol  = np.maximum(1.0, data["Volume"].astype(float).values)

    tick_size = float(instrument.tick_size)
    point_val = float(instrument.point_value)

    tr = np.maximum(high - low, 1e-9)
    tr_ticks = tr / max(tick_size, 1e-12)
    tr_ticks_per_sec = tr_ticks / np.maximum(dt_sec, 1e-6)

    # Optional clip to avoid pathological spikes (e.g., bad timestamps)
    if tr_per_sec_clip is not None:
        tr_ticks_per_sec = np.clip(tr_ticks_per_sec, 0.0, float(tr_per_sec_clip))

    lots: List[Dict[str, float]] = []
    wins = losses = 0
    rows: List[Dict[str, Any]] = []
    exec_pos = 0.0
    exec_pos_series: List[float] = []

    if base_spread_ticks is None:
        base_spread_ticks = float(slippage_ticks_per_side_base)

    def _fifo_close(q: float, price: float, side_to_close: int, slip_val: float, i: int):
        nonlocal wins, losses
        j = 0
        while q > 1e-9 and j < len(lots):
            lot = lots[j]
            if (side_to_close > 0 and lot["qty"] > 1e-9) or (side_to_close < 0 and lot["qty"] < -1e-9):
                take = min(abs(lot["qty"]), q)
                if side_to_close > 0:  # selling longs
                    fill_price = price - slip_val
                    gross = (fill_price - lot["entry"]) * point_val * take
                    rows.append({"i": i, "action": "SELL", "qty": take, "price": fill_price,
                                 "pnl": gross - commission_per_contract * take})
                    lot["qty"] -= take
                else:  # covering shorts
                    fill_price = price + slip_val
                    gross = (lot["entry"] - fill_price) * point_val * take
                    rows.append({"i": i, "action": "COVER", "qty": take, "price": fill_price,
                                 "pnl": gross - commission_per_contract * take})
                    lot["qty"] += take

                if abs(lot["qty"]) <= 1e-9:
                    net_round = gross - 2 * commission_per_contract * take
                    wins += int(net_round > 0); losses += int(net_round < 0)
                    lots.pop(j); continue
                q -= take
            j += 1
        return q

    def _slip_ticks_for_bar(i: int, qty_for_impact: float = 1.0) -> float:
        spread_ticks = base_spread_ticks + k_tr * tr_ticks[i]
        half_spread_ticks = 0.5 * spread_ticks
        impact_ticks = impact_k * (abs(qty_for_impact) / max(1.0, participation * vol[i]))
        latency_s = max(0.0, latency_ms) / 1000.0
        jitter_s = max(0.0, jitter_ms) / 1000.0
        latency_ticks = latency_s * tr_ticks_per_sec[i]
        jitter_ticks = abs(rng.normal(0.0, jitter_s)) * tr_ticks_per_sec[i] * 0.5
        return max(0.0, half_spread_ticks + impact_ticks + latency_ticks + jitter_ticks)

    for i in range(n):
        target = pos_desired[i] if fill_mode == "align" else (pos_desired[i-1] if i > 0 else pos_desired[0])
        delta_total = target - exec_pos
        if abs(delta_total) > 1e-9:
            max_fill = max(1.0, participation * vol[i])
            qty_to_fill = float(np.sign(delta_total) * min(abs(delta_total), max_fill))

            px = float(open_px[i])

            slip_ticks = _slip_ticks_for_bar(i, qty_for_impact=qty_to_fill)
            slip_val = slip_ticks * tick_size

            if qty_to_fill > 0:  # net buy
                rem = _fifo_close(qty_to_fill, px, side_to_close=-1, slip_val=slip_val, i=i)
                if rem > 1e-9:
                    buy_price = px + slip_val
                    lots.append({"entry": buy_price, "qty": rem, "side": 1})
                    rows.append({"i": i, "action": "BUY", "qty": rem, "price": buy_price,
                                 "pnl": -commission_per_contract * rem})
            else:               # net sell
                need = -qty_to_fill
                rem = _fifo_close(need, px, side_to_close=+1, slip_val=slip_val, i=i)
                if rem > 1e-9:
                    sell_price = px - slip_val
                    lots.append({"entry": sell_price, "qty": -rem, "side": -1})
                    rows.append({"i": i, "action": "SHORT", "qty": rem, "price": sell_price,
                                 "pnl": -commission_per_contract * rem})

            exec_pos += qty_to_fill

        # Prop breach flatten (speed-aware slippage)
        if enforce_breach_halt and ("prop_breached" in data.columns) and bool(data["prop_breached"].iloc[i]):
            if abs(exec_pos) > 1e-9:
                px = float(open_px[i])
                slip_val = _slip_ticks_for_bar(i, qty_for_impact=exec_pos) * tick_size
                if exec_pos > 0:
                    _ = _fifo_close(exec_pos, px, side_to_close=+1, slip_val=slip_val, i=i)
                else:
                    _ = _fifo_close(abs(exec_pos), px, side_to_close=-1, slip_val=slip_val, i=i)
                exec_pos = 0.0

        exec_pos_series.append(exec_pos)

    if flat_at_end and abs(exec_pos) > 1e-9 and n >= 1:
        i = n - 1
        px = float(close[-1])
        slip_val = _slip_ticks_for_bar(i, qty_for_impact=exec_pos) * tick_size
        if exec_pos > 0:
            _ = _fifo_close(exec_pos, px, side_to_close=+1, slip_val=slip_val, i=i)
        else:
            _ = _fifo_close(abs(exec_pos), px, side_to_close=-1, slip_val=slip_val, i=i)
        exec_pos = 0.0
        exec_pos_series[-1] = 0.0

    trades = pd.DataFrame(rows)
    realized = float(trades["pnl"].sum()) if not trades.empty else 0.0

    # Reconciliation CSV
    if reconcile_path:
        exec_pos_arr = np.asarray(exec_pos_series, dtype=float)
        pnl_exec_mtm = np.zeros(n)
        if n > 1:
            pnl_exec_mtm[:-1] = exec_pos_arr[:-1] * (close[1:] - close[:-1]) * point_val
        pnl_bt_gross = np.asarray(data.get("pnl_gross", np.zeros(n)), dtype=float)

        # Commissions per bar (sum of negative trade PnL rows is a proxy)
        exec_commissions = np.zeros(n)
        if not trades.empty:
            exec_commissions = trades.groupby("i")["pnl"].apply(
                lambda s: float(s[s < 0.0].sum())
            ).reindex(range(n), fill_value=0.0).values

        out = pd.DataFrame({
            "Datetime": pd.to_datetime(data["Datetime"], utc=True),
            "pos_bt": data["position"].astype(float).values,
            "pos_exec": exec_pos_arr,
            "pnl_gross_bt": pnl_bt_gross,
            "pnl_exec_mtm": pnl_exec_mtm,
            "exec_commissions": exec_commissions,
        })
        out["cum_bt"] = out["pnl_gross_bt"].cumsum()
        out["cum_exec"] = out["pnl_exec_mtm"].cumsum()
        out["cum_exec_net"] = (out["pnl_exec_mtm"] + out["exec_commissions"]).cumsum()
        out["cum_diff_gross"] = out["cum_exec"] - out["cum_bt"]

        Path(reconcile_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(reconcile_path, index=False)

    return trades, wins, losses, realized, int(round(exec_pos))


# -----------------------
# Hydration (fallback)
# -----------------------

def _local_hydrate(ns: argparse.Namespace):
    from copy import deepcopy

    from .config import (  # type: ignore
        COST,
        CONTRACT_COST,
        RISK_DEFAULTS,
        CostConfig,
        ContractCostConfig,
        PROB_BANDS,
        ENGINE,
        build_feature_context,
        PRESETS as _CFG_PRESETS,
    )
    from trading_system.runtime_engine.l3.bot.risk_engine import RiskConfig  # type: ignore

    active_preset = None
    preset_name = getattr(ns, "preset", None)
    if preset_name and preset_name in _CFG_PRESETS:
        try:
            active_preset = deepcopy(_CFG_PRESETS[preset_name])
        except Exception:
            active_preset = dict(_CFG_PRESETS[preset_name])

    if (ns.commission is not None) or (ns.slip_ticks is not None):
        cost = ContractCostConfig(
            commission_per_contract=(ns.commission if ns.commission is not None else CONTRACT_COST.commission_per_contract),
            slippage_ticks_per_side=(ns.slip_ticks if ns.slip_ticks is not None else CONTRACT_COST.slippage_ticks_per_side),
        )
    else:
        cost = CONTRACT_COST

    rc = dict(RISK_DEFAULTS)
    if ns.daily_loss_stop_usd is not None:
        rc["daily_loss_stop_usd"] = float(ns.daily_loss_stop_usd)
    if ns.prop_trail_dd_usd is not None:
        rc["prop_trailing_dd_usd"] = float(ns.prop_trail_dd_usd); rc["prop_enabled"] = True
    if ns.stop_after_win:
        rc["stop_after_first_win"] = True
    if ns.trail_intrabar:
        rc["trail_use_intrabar_extremes"] = True

    risk = RiskConfig(**rc)

    engine_kwargs = dict(
        session_tz=(ns.session_tz or ENGINE.session_tz),
        trade_window_start=(ns.trade_window_start or ENGINE.trade_window_start),
        trade_window_end=(ns.trade_window_end or ENGINE.trade_window_end),
        max_trades_per_day=(ns.max_trades_per_day or ENGINE.max_trades_per_day),
        account_scale_usd=float(ns.account_scale_usd if ns.account_scale_usd is not None else ENGINE.account_scale_usd),
        allowed_grades=tuple((ns.allow_grades or ",".join(ENGINE.allowed_grades)).split(",")),
        prob_bands=PROB_BANDS,
        enable_dd_circuit=ENGINE.enable_dd_circuit,
        dd_limit=ENGINE.dd_limit,
        dd_resume_hysteresis=ENGINE.dd_resume_hysteresis,
        dd_disable_from_next_bar=ENGINE.dd_disable_from_next_bar,
        enable_vol_target=False,
        target_vol=None,
        vol_ema_span=ENGINE.vol_ema_span,
        vol_annualize_k=ENGINE.vol_annualize_k,
        pos_cap=ENGINE.pos_cap,
        use_llm=False,
        llm_review_all=False,
        llm_max_risk_bps=ENGINE.llm_max_risk_bps,
        llm_cooldown_min=ENGINE.llm_cooldown_min,
        symbol=ENGINE.symbol,
        instrument=(instrument_by_alias(ns.instrument) if ns.instrument else instrument_by_alias(ENGINE.instrument_alias)),
        enable_vwap_gate=True,
        enable_ema_gate=True,
        profit_lock_usd=(ns.profit_lock_usd if ns.profit_lock_usd is not None else None),
        near_breach_buffer_usd=float(ns.near_breach_buffer_usd or 100.0),
        target_r=ns.target_r,
        policy_margin=float(ns.policy_margin or 0.0),
    )
    feature_ctx = build_feature_context(preset=active_preset, name=preset_name)
    engine_kwargs["feature_context"] = feature_ctx
    engine_kwargs["preset_config"] = active_preset
    return cost, risk, engine_kwargs, feature_ctx, active_preset


def _sleep_speed(prev_ts: pd.Timestamp, next_ts: pd.Timestamp, speed: float):
    if speed <= 0:
        return
    dt = (pd.to_datetime(next_ts) - pd.to_datetime(prev_ts)).total_seconds()
    if dt < 0:
        print(f"[WARN] Non-monotonic timestamps: {prev_ts} → {next_ts} (Δ={dt}s). Skipping sleep.")
        return
    time.sleep(max(0.0, dt / max(1e-9, speed)))


# -----------------------
# Start/resume helpers & streaming
# -----------------------

def _parse_start_at(s: str, session_tz: str | None) -> pd.Timestamp:
    ts = pd.to_datetime(s, errors="raise")
    if ts.tzinfo is None:
        tz = session_tz or "UTC"
        ts = ts.tz_localize(tz).tz_convert("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _last_stream_timestamp(stream_fp: Path) -> Optional[pd.Timestamp]:
    try:
        if not stream_fp.exists() or stream_fp.stat().st_size == 0:
            return None
        with open(stream_fp, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = 8192
            buffer = b""
            while size > 0:
                read_size = min(chunk, size)
                size -= read_size
                f.seek(size)
                buffer = f.read(read_size) + buffer
                if buffer.count(b"\n") > 2:
                    break
        lines = buffer.decode("utf-8", errors="ignore").strip().splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            if line.startswith("Datetime"):
                continue
            first = line.split(",", 1)[0]
            ts = pd.to_datetime(first, utc=True, errors="coerce")
            if pd.notna(ts):
                return ts
    except Exception:
        return None
    return None


def _emit_rows_to_stream(stream_fp: Optional[Path], rows: pd.DataFrame):
    if stream_fp is None or rows.empty:
        return 0
    count = 0
    for _, row in rows.iterrows():
        dt_iso = pd.to_datetime(row["Datetime"], utc=True).isoformat()
        o = float(row["Open"]); h = float(row["High"]); l = float(row["Low"]); c = float(row["Close"])
        v = float(row.get("Volume", 0.0))
        p = row.get("proba", np.nan)
        p_str = "" if pd.isna(p) else f"{float(p):.6f}"
        payload = [dt_iso, o, h, l, c, v, p_str]
        with open(stream_fp, "a") as f:
            f.write(",".join(map(str, payload)) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        count += 1
    return count


# -----------------------
# CLI
# -----------------------

def main():

    ap = argparse.ArgumentParser(description="Historical market replay with ultra-realistic execution (prop-firm ready)")
    ap.add_argument("--model", required=True, help="Alias (models.py) or path to model artifact")
    ap.add_argument("--csv", help="Path to OHLCV CSV (Datetime,Open,High,Low,Close,Volume)")
    ap.add_argument("--symbol", default="ES=F")
    ap.add_argument("--yf_period", help="e.g., 60d, 90d, 730d")
    ap.add_argument("--start_date"); ap.add_argument("--end_date")
    ap.add_argument("--interval", default="5m", help="yfinance interval (default 5m)")
    ap.add_argument("--out_dir", default="runs/replay")
    ap.add_argument("--preset", choices=list(PRESETS.keys()))
    ap.add_argument("--fill_mode", choices=["align","catch_up"], default="align")
    ap.add_argument("--reconcile", action="store_true")
    # aliases
    ap.add_argument("--tz", dest="session_tz", help="Alias for --session_tz")
    ap.add_argument("--data", dest="csv", help="Alias for --csv")
    # harmless no-ops so they don’t crash if passed
    ap.add_argument("--rth_start", help=argparse.SUPPRESS)
    ap.add_argument("--rth_end", help=argparse.SUPPRESS)
    ap.add_argument("--orb_min", help=argparse.SUPPRESS)
    ap.add_argument("--out_csv", help=argparse.SUPPRESS)
    ap.add_argument("--out_trades", help=argparse.SUPPRESS)
    ap.add_argument("--out_plot", help=argparse.SUPPRESS)

    ap.add_argument("--no_enforce_breach_halt", action="store_true")

    # Exec realism
    ap.add_argument("--order_type", default="market", choices=["market"])
    ap.add_argument("--latency_ms", type=int, default=120)
    ap.add_argument("--jitter_ms", type=int, default=40)
    ap.add_argument("--participation", type=float, default=0.05)
    ap.add_argument("--base_spread_ticks", type=float, default=None)
    ap.add_argument("--k_tr", type=float, default=0.15)
    ap.add_argument("--impact_k", type=float, default=0.50)
    ap.add_argument("--speed", type=float, default=0.0, help="Replay speed multiplier; 0 disables sleeping")
    ap.add_argument("--flat_at_end", action="store_true", help="Force flatten at last bar (realize PnL)")
    ap.add_argument("--tr_per_sec_clip", type=float, default=None, help="Optional cap on ticks/second for slippage calc")

    # Optional news risk integration
    ap.add_argument("--news_csv", help="CSV of scheduled events for NewsRiskFilter (time_local,currency,impact,title)")
    ap.add_argument("--news_dampen_floor", type=float, default=None,
                    help="If set, probabilities during event windows are forced to this floor (e.g., 0.50)")

    # Minimal overrides (hydrator fills the rest)
    ap.add_argument("--p_buy", type=float); ap.add_argument("--p_sell", type=float); ap.add_argument("--allow_grades")
    ap.add_argument("--instrument"); ap.add_argument("--commission", type=float); ap.add_argument("--slip_ticks", type=float)
    ap.add_argument("--account_scale_usd", type=float)
    ap.add_argument("--session_tz"); ap.add_argument("--start", dest="trade_window_start"); ap.add_argument("--end", dest="trade_window_end")
    ap.add_argument("--max_trades", dest="max_trades_per_day", type=int)
    ap.add_argument("--daily_stop_usd", type=float); ap.add_argument("--daily_loss_stop_usd", type=float)
    ap.add_argument("--prop_trail_dd_usd", type=float)
    ap.add_argument("--stop_after_win", action="store_true"); ap.add_argument("--trail_intrabar", action="store_true")
    ap.add_argument("--profit_lock_usd", type=float); ap.add_argument("--near_breach_buffer_usd", type=float, default=None)
    ap.add_argument("--target_r", type=float, default=None)
    ap.add_argument("--policy_margin", type=float, default=0.0)

    # L1 stream for L2 (Discord bridge)
    ap.add_argument("--emit_stream_csv", default=None,
                    help="If set, append OHLCV+proba per bar to this CSV during replay (for L2)")

    # Start/resume window
    ap.add_argument("--start_at", help="Start replay at/after this timestamp (e.g. '2025-09-29 07:30', session tz or ISO UTC)")
    ap.add_argument("--end_at", help="Stop replay at/before this timestamp")
    ap.add_argument("--resume_from_stream", action="store_true",
                    help="If set, begin after the last Datetime present in --emit_stream_csv")

    # Follow mode
    ap.add_argument("--follow", action="store_true",
                    help="Keep the process alive; poll for new bars and emit as they appear")
    ap.add_argument("--poll_sec", type=float, default=5.0,
                    help="Polling interval seconds for --follow mode")

    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args.model)

    ns = argparse.Namespace(
        commission=args.commission, slip_ticks=args.slip_ticks, fee_bps=None, slippage_bps=None,
        instrument=args.instrument, account_scale_usd=args.account_scale_usd,
        session_tz=args.session_tz, trade_window_start=args.trade_window_start, trade_window_end=args.trade_window_end,
        p_buy=args.p_buy, p_sell=args.p_sell, allow_grades=args.allow_grades,
        max_trades_per_day=args.max_trades_per_day,
        per_trade_stop_pct=None, per_trade_stop_usd=None,
        daily_loss_stop_pct=None, daily_loss_stop_usd=(args.daily_loss_stop_usd if args.daily_loss_stop_usd is not None else args.daily_stop_usd),
        trailing_drawdown_pct=None, trail_dd_usd=None,
        prop_trail_dd_usd=args.prop_trail_dd_usd,
        prop_trail_on_unrealized=None, prop_breach_buffer_usd=args.near_breach_buffer_usd,
        stop_after_win=(args.stop_after_win or None), no_stop_after_win=False, trail_intrabar=(args.trail_intrabar or None),
        max_position=None, proba_cut_bad=None, trail_profit_pct=None, trail_profit_activate_pct=None,
        flip_exit_buf=None, atr_len=None, atr_k=None,
        breakeven_activate_rr=None, be_activate_pct=None, be_buffer_pct=None,
        max_bars_in_trade=None, cooldown_bars_after_stop=None,
        vol_target=None, target_vol=None, vol_ema_span=None, vol_annualize_k=None, pos_cap=None,
        use_llm=False, llm_review_all=None, llm_risk_bps=None, llm_cooldown_min=None, symbol=None,
        profit_lock_usd=args.profit_lock_usd,
        near_breach_buffer_usd=args.near_breach_buffer_usd,
        target_r=args.target_r, policy_margin=args.policy_margin,
        preset=args.preset,
    )

    if args.preset:
        ns = _apply_preset_to_namespace(args.preset, ns)
        if ns.commission is not None or ns.slip_ticks is not None:
            ns.fee_bps = None; ns.slippage_bps = None

    if _hydrate is not None:
        cost, risk, engine_kwargs, feature_ctx, active_preset = _hydrate(ns)
    else:
        cost, risk, engine_kwargs, feature_ctx, active_preset = _local_hydrate(ns)

    # -----------------------
    # Load source data
    # -----------------------
    if args.csv:
        raw = pd.read_csv(args.csv)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = _flatten_multiindex_columns(raw)
        df = _ensure_ohlcv_cols(raw)
        df = df.drop_duplicates(subset=["Datetime"], keep="last").sort_values("Datetime").reset_index(drop=True)
    else:
        df = fetch_yf(args.symbol, yf_period=args.yf_period, start_date=args.start_date, end_date=args.end_date, interval=args.interval)
        df = df.drop_duplicates(subset=["Datetime"], keep="last").sort_values("Datetime").reset_index(drop=True)

    src_csv = out_dir / "replay_source.csv"
    df.to_csv(src_csv, index=False)
    print(f"[INFO] Source data saved → {src_csv}")

    # Optional news filter parquet
    if args.news_csv and NewsRiskFilter is not None:
        try:
            nf = NewsRiskFilter(preset=active_preset)
            nf.refresh(csv_path=args.news_csv)
            nf.df_events.to_parquet(out_dir / "news_windows.parquet", index=False)
        except Exception as e:
            print(f"[WARN] NewsRiskFilter failed: {e}")

    # Output paths
    out_csv = out_dir / "backtest_results.csv"
    out_trades = out_dir / "backtest_trades.csv"
    out_plot = out_dir / "equity.png"
    charts_dir = out_dir / "charts"; charts_dir.mkdir(parents=True, exist_ok=True)

    if _run_backtest is None:
        raise RuntimeError("run_backtest not found. Ensure backtest_model(s).py is available in your package.")

    # --------------- Run backtest (initial) ---------------
    data, summary = _run_backtest(
        csv_path=str(src_csv),
        model_path=model_path,
        cost=cost,
        risk=risk,
        out_csv=str(out_csv),
        out_plot=str(out_plot),
        out_trades=str(out_trades),
        charts_dir=str(charts_dir),
        **engine_kwargs,
    )
    data = data.copy().sort_values("Datetime").reset_index(drop=True)

    # Ensure probability column
    if "proba" in data.columns:
        data["proba"] = pd.to_numeric(data["proba"], errors="coerce")
    elif "prob" in data.columns:
        data["proba"] = pd.to_numeric(data["prob"], errors="coerce")
    else:
        data["proba"] = np.nan

    # Stream setup
    stream_fp = Path(args.emit_stream_csv).expanduser() if args.emit_stream_csv else None
    if stream_fp and not stream_fp.is_absolute():
        stream_fp = (Path.cwd() / stream_fp).resolve()
    if stream_fp:
        stream_fp.parent.mkdir(parents=True, exist_ok=True)
        if not stream_fp.exists() or stream_fp.stat().st_size == 0:
            with open(stream_fp, "w") as f:
                f.write("Datetime,Open,High,Low,Close,Volume,proba\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        print(f"[INFO] L1 stream → {stream_fp}")
    else:
        print("[INFO] L1 stream disabled (no --emit_stream_csv)")

    # Defensive
    data["position"] = pd.to_numeric(data["position"], errors="coerce").fillna(0.0)

    # --------------- Determine initial replay window ---------------
    ts_all = pd.to_datetime(data["Datetime"], utc=True)
    ts_ns = ts_all.astype("int64")

    replay_start_i = 0
    replay_end_i = len(data)

    # --start_at
    if args.start_at:
        start_utc = _parse_start_at(args.start_at, engine_kwargs.get("session_tz"))
        replay_start_i = int(np.searchsorted(ts_ns.values, int(start_utc.value)))

    # --resume_from_stream
    elif args.resume_from_stream and stream_fp:
        last_dt = _last_stream_timestamp(stream_fp)
        if last_dt is not None:
            replay_start_i = int(np.searchsorted(ts_ns.values, int(last_dt.value + 1)))

    # --end_at
    if args.end_at:
        end_utc = _parse_start_at(args.end_at, engine_kwargs.get("session_tz"))
        replay_end_i = int(np.searchsorted(ts_ns.values, int(end_utc.value), side="right"))

    replay_start_i = max(0, min(replay_start_i, len(data)))
    replay_end_i   = max(replay_start_i, min(replay_end_i, len(data)))

    # --------------- Exec simulation (initial; full data) ---------------
    trades, wins, losses, realized, final_pos = simulate_exec(
        data,
        engine_kwargs["instrument"],
        commission_per_contract=float(getattr(cost, "commission_per_contract", 0.0)),
        slippage_ticks_per_side_base=float(getattr(cost, "slippage_ticks_per_side", 0.0)),
        order_type=args.order_type,
        latency_ms=args.latency_ms,
        jitter_ms=args.jitter_ms,
        participation=args.participation,
        base_spread_ticks=args.base_spread_ticks,
        k_tr=args.k_tr,
        impact_k=args.impact_k,
        flat_at_end=bool(args.flat_at_end),
        fill_mode=args.fill_mode,
        enforce_breach_halt=not args.no_enforce_breach_halt,
        reconcile_path=str(out_dir / "reconcile.csv") if args.reconcile else None,
        seed=7,
        tr_per_sec_clip=args.tr_per_sec_clip,
    )

    trades_path = out_dir / "exec_trades.csv"
    trades.to_csv(trades_path, index=False)

    # --------------- Emit initial window ---------------
    log_path = out_dir / "replay_log.txt"
    emitted_until_ts_utc = pd.to_datetime(data["Datetime"].iloc[replay_start_i - 1], utc=True) if replay_start_i > 0 else None

    with log_path.open("w") as log_f:
        realized_c = 0.0
        by_i = trades.groupby("i") if not trades.empty else None

        # Seed realized with prior bars if starting mid-stream
        if by_i is not None and replay_start_i > 0:
            realized_c = float(trades.loc[trades["i"] < replay_start_i, "pnl"].sum())

        # Emit initial historical window
        for i in range(replay_start_i, replay_end_i):
            tdf = (by_i.get_group(i) if (by_i is not None and i in by_i.groups) else pd.DataFrame())
            if not tdf.empty:
                realized_c += float(tdf["pnl"].astype(float).sum())

            ts_i = pd.to_datetime(data["Datetime"].iloc[i], utc=True)
            try:
                ts_local = ts_i.tz_convert(engine_kwargs["session_tz"])
            except Exception:
                ts_local = ts_i
            pos_now = int(round(data["position"].iloc[i]))
            line = f"{ts_local.strftime('%m/%d/%Y %H:%M')} / Pos: {pos_now}  realized {realized_c:.2f}"
            print(line); log_f.write(line + "\n")

            # Emit to L1 stream
            _emit_rows_to_stream(stream_fp, data.iloc[[i]])

            emitted_until_ts_utc = ts_i

            if i < replay_end_i - 1:
                _sleep_speed(data["Datetime"].iloc[i], data["Datetime"].iloc[i+1], args.speed)

        # --------------- FOLLOW MODE: keep process alive & emit new bars ---------------
        if args.follow:
            print(f"[FOLLOW] Waiting for new bars… polling every {args.poll_sec}s. CTRL+C to exit.")
            try:
                while True:
                    time.sleep(max(0.1, float(args.poll_sec)))

                    # Reload/refresh source
                    if args.csv:
                        try:
                            raw_new = pd.read_csv(args.csv)
                            if isinstance(raw_new.columns, pd.MultiIndex):
                                raw_new = _flatten_multiindex_columns(raw_new)
                            df_new = _ensure_ohlcv_cols(raw_new)
                        except Exception as e:
                            print(f"[FOLLOW] Could not reload source CSV: {e}")
                            continue
                    else:
                        # yfinance poll: extend data up to now
                        try:
                            df_new = fetch_yf(args.symbol, yf_period=args.yf_period,
                                              start_date=args.start_date, end_date=args.end_date, interval=args.interval)
                        except Exception as e:
                            print(f"[FOLLOW] yfinance fetch failed: {e}")
                            continue

                    # Append-only detection
                    last_known = pd.to_datetime(data["Datetime"].iloc[-1], utc=True) if len(data) else None
                    df_new = df_new.sort_values("Datetime").reset_index(drop=True)
                    if last_known is not None:
                        mask_new = pd.to_datetime(df_new["Datetime"], utc=True) > last_known
                        df_app = df_new.loc[mask_new].copy()
                    else:
                        df_app = df_new.copy()

                    if df_app.empty:
                        # nothing new
                        continue

                    # Persist/extend src_csv for reproducibility
                    with open(src_csv, "a") as f:
                        df_app.to_csv(f, header=False, index=False)

                    # Rerun backtest on the extended source (simple & robust)
                    data_ext, summary_ext = _run_backtest(
                        csv_path=str(src_csv),
                        model_path=model_path,
                        cost=cost,
                        risk=risk,
                        out_csv=str(out_csv),
                        out_plot=str(out_plot),
                        out_trades=str(out_trades),
                        charts_dir=str(charts_dir),
                        **engine_kwargs,
                    )
                    data_ext = data_ext.sort_values("Datetime").reset_index(drop=True)
                    if "proba" in data_ext.columns:
                        data_ext["proba"] = pd.to_numeric(data_ext["proba"], errors="coerce")
                    elif "prob" in data_ext.columns:
                        data_ext["proba"] = pd.to_numeric(data_ext["prob"], errors="coerce")
                    else:
                        data_ext["proba"] = np.nan
                    data_ext["position"] = pd.to_numeric(data_ext["position"], errors="coerce").fillna(0.0)

                    # Recompute exec over full set; we’ll emit only new tail rows
                    trades_ext, wins_ext, losses_ext, realized_ext, final_pos_ext = simulate_exec(
                        data_ext,
                        engine_kwargs["instrument"],
                        commission_per_contract=float(getattr(cost, "commission_per_contract", 0.0)),
                        slippage_ticks_per_side_base=float(getattr(cost, "slippage_ticks_per_side", 0.0)),
                        order_type=args.order_type,
                        latency_ms=args.latency_ms,
                        jitter_ms=args.jitter_ms,
                        participation=args.participation,
                        base_spread_ticks=args.base_spread_ticks,
                        k_tr=args.k_tr,
                        impact_k=args.impact_k,
                        flat_at_end=False,  # don't force flatten on rolling stream
                        fill_mode=args.fill_mode,
                        enforce_breach_halt=not args.no_enforce_breach_halt,
                        reconcile_path=None,
                        seed=7,
                        tr_per_sec_clip=args.tr_per_sec_clip,
                    )

                    # Determine which rows are new to emit
                    ts_ext = pd.to_datetime(data_ext["Datetime"], utc=True)
                    if emitted_until_ts_utc is None:
                        start_emit_i = 0
                    else:
                        start_emit_i = int(np.searchsorted(ts_ext.astype("int64").values, int(emitted_until_ts_utc.value + 1)))
                    if start_emit_i >= len(data_ext):
                        continue

                    # Update realized_c baseline using new trades
                    if not trades_ext.empty:
                        realized_c = float(trades_ext.loc[trades_ext["i"] < start_emit_i, "pnl"].sum())
                    else:
                        realized_c = 0.0

                    by_i_ext = trades_ext.groupby("i") if not trades_ext.empty else None

                    # Emit tail rows one by one at replay speed
                    for i in range(start_emit_i, len(data_ext)):
                        tdf = (by_i_ext.get_group(i) if (by_i_ext is not None and i in by_i_ext.groups) else pd.DataFrame())
                        if not tdf.empty:
                            realized_c += float(tdf["pnl"].astype(float).sum())

                        ts_i = pd.to_datetime(data_ext["Datetime"].iloc[i], utc=True)
                        try:
                            ts_local = ts_i.tz_convert(engine_kwargs["session_tz"])
                        except Exception:
                            ts_local = ts_i
                        pos_now = int(round(data_ext["position"].iloc[i]))
                        line = f"{ts_local.strftime('%m/%d/%Y %H:%M')} / Pos: {pos_now}  realized {realized_c:.2f}"
                        print(line); log_f.write(line + "\n")

                        # Emit to stream
                        _emit_rows_to_stream(stream_fp, data_ext.iloc[[i]])
                        emitted_until_ts_utc = ts_i

                        # Sleep in proportion to bar spacing (using requested speed)
                        if i < len(data_ext) - 1:
                            _sleep_speed(data_ext["Datetime"].iloc[i], data_ext["Datetime"].iloc[i+1], args.speed)

                    # Roll forward our in-memory references
                    data = data_ext
                    trades = trades_ext
                    summary = summary_ext

            except KeyboardInterrupt:
                print("\n[FOLLOW] Exit requested by user. Shutting down.")

    # Summary (include realism params & seed for reproducibility)
    summary_exec = dict(
        wins=wins, losses=losses, realized=round(realized, 2),
        rows=len(trades), final_pos=final_pos,
        backtest_final_equity=summary.get("final_equity"),
        backtest_trades=summary.get("trades"),
        instrument=getattr(engine_kwargs.get("instrument"), "alias", ""),
        commission_per_contract=float(getattr(cost, "commission_per_contract", 0.0)),
        slip_ticks=float(getattr(cost, "slippage_ticks_per_side", 0.0)),
        order_type=args.order_type,
        latency_ms=args.latency_ms,
        jitter_ms=args.jitter_ms,
        participation=args.participation,
        base_spread_ticks=args.base_spread_ticks if args.base_spread_ticks is not None else float(getattr(cost, "slippage_ticks_per_side", 0.0)),
        k_tr=args.k_tr,
        impact_k=args.impact_k,
        tr_per_sec_clip=args.tr_per_sec_clip,
        rng_seed=7,
    )
    (out_dir / "summary_exec.txt").write_text("\n".join(f"{k}={v}" for k, v in summary_exec.items()))
    Path("runs").mkdir(exist_ok=True)
    Path(PurePath("runs") / "last_replay_summary.json").write_text(json.dumps(summary_exec, indent=2, default=str))

    print("\n=== REPLAY SUMMARY ===")
    print(json.dumps(summary_exec, indent=2))


if __name__ == "__main__":
    main()
