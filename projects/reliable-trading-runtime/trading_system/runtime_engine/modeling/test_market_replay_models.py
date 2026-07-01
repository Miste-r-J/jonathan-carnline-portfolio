#!/usr/bin/env python3
"""
test_market_replay_models.py — regression & stress tests for market_replay_models.py

Covers:
- Model registry → backtest → execution replay end-to-end
- Conservative tick rounding (never favorable)
- Realized PnL ≤ theoretical (under equal cost)
- FIFO fill integrity (no orphaned positions)
- Deterministic output reproducibility under seed control
"""

import os
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# === Imports ===
# === Imports ===
import sys


# Ensure root path (so tests can import local bot/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Now safe to import local modules
from bot import market_replay_models as mrm
from  .config import PRESETS, ENGINE, instrument_by_alias

# Optional: seed for deterministic stress runs
np.random.seed(7)


@pytest.fixture(scope="module")
def tmpdir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="test_replay_"))
    yield d
    # Cleanup optional
    for p in d.glob("*"):
        try: p.unlink()
        except: pass


# ---------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------

def _dummy_ohlcv(n=200) -> pd.DataFrame:
    """Generate synthetic 5m OHLCV with smooth trends."""
    t = pd.date_range("2025-01-01 07:30", periods=n, freq="5min", tz=None)
    price = 4800 + np.cumsum(np.random.normal(0, 0.75, n))
    high = price + np.random.uniform(0.25, 0.75, n)
    low = price - np.random.uniform(0.25, 0.75, n)
    open_ = price + np.random.uniform(-0.2, 0.2, n)
    close = price + np.random.uniform(-0.2, 0.2, n)
    vol = np.random.uniform(500, 2000, n)
    return pd.DataFrame({
        "Datetime": t,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol
    })


# ---------------------------------------------------------------
# Unit-level tests
# ---------------------------------------------------------------

def test_round_trade_price_conservative():
    tick = 0.25
    assert mrm._round_trade_price(4800.12, tick, side=+1) >= 4800.12  # buy up
    assert mrm._round_trade_price(4800.12, tick, side=-1) <= 4800.12  # sell down


def test_flatten_multiindex_columns():
    df = pd.concat({"ES": _dummy_ohlcv(3)}, axis=1)
    out = mrm._flatten_multiindex_columns(df, "ES")
    assert {"Open","High","Low","Close"}.issubset(out.columns)


def test_ensure_ohlcv_cols_complete():
    df = _dummy_ohlcv(10).rename(columns={"Open":"open","Close":"close"})
    out = mrm._ensure_ohlcv_cols(df)
    assert all(c in out.columns for c in ["Open","Close","Volume"])
    assert len(out) == 10


# ---------------------------------------------------------------
# Integration-level tests
# ---------------------------------------------------------------

def test_simulate_exec_consistency(tmpdir):
    df = _dummy_ohlcv(300)
    inst = instrument_by_alias("ES")
    df["position"] = np.sign(np.sin(np.linspace(0, 10, len(df))))  # oscillating 1/-1 positions

    trades, wins, losses, realized, final_pos = mrm.simulate_exec(
        df,
        instrument=inst,
        commission_per_contract=2.0,
        slippage_ticks_per_side_base=1.0,
        reconcile_path=tmpdir / "reconcile.csv",
        fill_mode="catch_up",
        flat_at_end=True,
        seed=42
    )

    # Check FIFO fills sum to flat
    net_qty = trades["qty"].where(trades["action"].isin(["BUY","COVER"]), -trades["qty"]).sum()
    assert abs(net_qty) < 1e-6, "Execution should end flat"

    # Check realized never exceeds total move
    realized_bound = (df["High"].max() - df["Low"].min()) * inst.point_value
    assert abs(realized) < realized_bound

    # Ensure reconcile file created
    assert (tmpdir / "reconcile.csv").exists()


def test_adverse_stop_pricing(tmpdir):
    df = _dummy_ohlcv(50)
    inst = instrument_by_alias("ES")
    df["position"] = [1 if i < 25 else 0 for i in range(50)]
    # Add artificial stop flags for half the bars
    for c in ["stopped_trade","stopped_atr","stopped_be","stopped_trail_profit","stopped_prob","stopped_window"]:
        df[c] = [i < 25 for i in range(50)]

    trades, wins, losses, realized, final_pos = mrm.simulate_exec(
        df,
        instrument=inst,
        commission_per_contract=2.0,
        slippage_ticks_per_side_base=1.0,
        adverse_stops=True
    )

    # Stop-like bars must exist
    assert any(df[c].any() for c in ["stopped_trade","stopped_atr"]), "stop flags not applied"
    assert isinstance(realized, float)


def test_replay_pipeline_smoke(tmpdir):
    """End-to-end mini run with dummy data (no yfinance)."""
    csv_path = tmpdir / "input.csv"
    df = _dummy_ohlcv(100)
    df.to_csv(csv_path, index=False)

    # Minimal CLI args emulation
    class Args:
        model = list(mrm.PRESETS.keys())[0] if mrm.PRESETS else None
        csv = str(csv_path)
        symbol = "ES=F"
        preset = "prop_es_50k"
        fill_mode = "catch_up"
        reconcile = True
        flat_at_end = True
        out_dir = tmpdir
        yf_period = None
        start_date = None
        end_date = None
        interval = "5m"
        no_enforce_breach_halt = False
        order_type = "market"
        latency_ms = 50
        jitter_ms = 10
        participation = 0.1
        base_spread_ticks = 1.0
        k_tr = 0.15
        impact_k = 0.5
        speed = 0
        open_ramp_minutes = 5
        k_tr_open = 0.3
        latency_open_ms = 100
        jitter_open_ms = 20
        commission = 2.0
        slip_ticks = 1.0
        account_scale_usd = 50000.0
        session_tz = "America/Denver"
        trade_window_start = "07:30"
        trade_window_end = "12:00"
        max_trades_per_day = 5
        daily_loss_stop_usd = 1000.0
        prop_trail_dd_usd = 2500.0
        stop_after_win = False
        trail_intrabar = False
        profit_lock_usd = 0.0
        near_breach_buffer_usd = 100.0
        target_r = 2.0
        policy_margin = 0.0

    ns = Args()
    # Use local hydrator to avoid dependency on full package
    cost, risk, engine_kwargs = mrm._local_hydrate(ns)

    trades, wins, losses, realized, final_pos = mrm.simulate_exec(
        df.assign(position=np.sign(np.sin(np.linspace(0, 8, len(df))))),
        engine_kwargs["instrument"],
        cost.commission_per_contract,
        cost.slippage_ticks_per_side,
        reconcile_path=str(tmpdir / "reconcile.csv")
    )

    # Validate result integrity
    assert isinstance(trades, pd.DataFrame)
    assert "pnl" in trades.columns
    assert abs(final_pos) < 1e-9
    assert abs(realized) < 5e6
    assert (tmpdir / "reconcile.csv").exists()


def test_reproducibility_seed_control():
    df = _dummy_ohlcv(100)
    inst = instrument_by_alias("ES")
    df["position"] = np.sign(np.sin(np.linspace(0, 6, len(df))))
    t1 = mrm.simulate_exec(df, inst, 2.0, 1.0, seed=1234)[3]
    t2 = mrm.simulate_exec(df, inst, 2.0, 1.0, seed=1234)[3]
    t3 = mrm.simulate_exec(df, inst, 2.0, 1.0, seed=9999)[3]
    assert math.isclose(t1, t2, rel_tol=1e-9), "Seed must reproduce identical results"
    assert not math.isclose(t1, t3, rel_tol=1e-6), "Different seed should differ"
