#backtest_models.py

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List, Dict, Any
import inspect

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Logging ---
logger = logging.getLogger("backtest_model")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# --- Local imports (no circulars) ---
try:
    from .features import build_features  # your project feature builder
except Exception:
    from bot.features import build_features  # type: ignore

try:
    from .config import (
        COST as DEFAULT_COST,
        CONTRACT_COST,
        ENGINE as DEFAULTS,
        PROB_BANDS,
        RISK_DEFAULTS,
        PRESETS,
        CostConfig,
        ContractCostConfig,
        instrument_by_alias,
        build_feature_context,
    )
except Exception:
    from config import (  # type: ignore
        COST as DEFAULT_COST,
        CONTRACT_COST,
        ENGINE as DEFAULTS,
        PROB_BANDS,
        RISK_DEFAULTS,
        PRESETS,
        CostConfig,
        ContractCostConfig,
        instrument_by_alias,
        build_feature_context,
    )

try:
    from .backtest_threshold import backtest_threshold_futures, RiskConfig
except Exception:
    from backtest_threshold import backtest_threshold_futures, RiskConfig  # type: ignore

# Model registry & calibration helpers
try:
    from .models import (
        features_sidecar_path,
        model_path as resolve_model_path,
        load_model_features_and_calibrator,
        calibrate_proba,
        compute_sha256,
    )
except Exception:  # type: ignore
    def features_sidecar_path(p: str) -> str:  # pragma: no cover
        return str(Path(p).with_suffix(".features.json"))
    def resolve_model_path(p: str) -> str:  # pragma: no cover
        pth = Path(os.path.expanduser(p)).resolve()
        if pth.is_file():
            return str(pth)
        raise FileNotFoundError(f"Model not found for '{p}'. Provide a valid path or set up models.py alias.")
    def load_model_features_and_calibrator(p: str):  # pragma: no cover
        m = joblib.load(p); return m, None, None
    def calibrate_proba(p, c):  # pragma: no cover
        return np.asarray(p, dtype=float).clip(0, 1)
    def compute_sha256(p: str):  # pragma: no cover
        import hashlib
        h = hashlib.sha256(Path(p).read_bytes()).hexdigest(); return h

# Optional News filter (scheduled events)
try:
    from .news_filter import NewsRiskFilter  # production module you built earlier
except Exception:  # pragma: no cover
    NewsRiskFilter = None  # type: ignore


# ---------- constants ----------
BASE_COLS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
ENGINE_COLS = [
    "ret", "signal", "position", "equity", "peak_equity",
    "pnl_gross", "pnl_net", "day_pnl"
]




# ---------------------- Utilities ----------------------
def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # De-duplicate any repeated column names (yfinance/joins can cause this)
    if getattr(df.columns, "duplicated", None) is not None and df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    rename: Dict[Any, str] = {}
    for c in df.columns:
        lc = str(c).lower()
        if "open" in lc and "interest" not in lc:   rename[c] = "Open"
        if "high" in lc and "interest" not in lc:   rename[c] = "High"
        if "low" in lc and "interest" not in lc:    rename[c] = "Low"
        if "close" in lc and "adj" not in lc:       rename[c] = "Close"
        if "volume" in lc:                          rename[c] = "Volume"
        if lc in ("datetime","date","time","timestamp"): rename[c] = "Datetime"
    if rename:
        df = df.rename(columns=rename)

    if "Datetime" not in df.columns:
        raise ValueError("No Datetime column found.")
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if keep:
        df = df.dropna(subset=keep).reset_index(drop=True)
    return df



def _align_features(X: pd.DataFrame, expected: Sequence[str] | None) -> Tuple[pd.DataFrame, List[str]]:
    X = X.select_dtypes(include=[np.number, "bool"]).astype(float)
    # exclude obvious leak columns
    drop_exact = {"Open", "High", "Low", "Close", "Volume"}
    suspicious = ("follow", "future", "lead", "ahead", "fwd")
    safe_suspicious = {"hi_break_follow", "lo_break_follow"}
    cols = []
    for c in X.columns:
        if c in drop_exact: continue
        lc = str(c).lower()
        if c not in safe_suspicious and any(s in lc for s in suspicious): continue
        cols.append(c)
    X = X[cols]

    if expected is None:
        return X, []
    extra = [c for c in X.columns if c not in expected]
    missing = [c for c in expected if c not in X.columns]
    if extra:
        logger.warning(f"Dropping {len(extra)} extra features not seen at train time. Examples: {extra[:10]}")
        X = X.drop(columns=extra, errors="ignore")
    for m in missing:
        X[m] = 0.0
    X = X.loc[:, list(expected)]
    return X, missing


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(up) in [0,1] per row. Tries predict_proba, falls back to decision_function/predict."""
    try:
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba(X), dtype=float)
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1]
            if p.ndim == 1:
                return p.clip(0.0, 1.0)
        if hasattr(model, "decision_function"):
            s = np.asarray(model.decision_function(X), dtype=float)
            return 1.0 / (1.0 + np.exp(-s))
        y = np.asarray(model.predict(X), dtype=float)
        if set(np.unique(y)) <= {-1.0, 1.0}:
            y = (y + 1.0) / 2.0
        return y.clip(0.0, 1.0)
    except Exception:
        X2 = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba(X2), dtype=float)
            return p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.clip(0.0, 1.0)
        if hasattr(model, "decision_function"):
            s = np.asarray(model.decision_function(X2), dtype=float)
            return 1.0 / (1.0 + np.exp(-s))
        y = np.asarray(model.predict(X2), dtype=float)
        if set(np.unique(y)) <= {-1.0, 1.0}:
            y = (y + 1.0) / 2.0
        return y.clip(0.0, 1.0)


def _make_charts(res: pd.DataFrame, out_plot: str, charts_dir: str):
    Path(charts_dir).mkdir(exist_ok=True, parents=True)

    # Equity curve
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(pd.to_datetime(res["Datetime"]), res["equity"])  # no custom styles/colors
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    # Underwater (drawdown)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    dd = res["equity"].astype(float).values - res["peak_equity"].astype(float).values
    ax.plot(pd.to_datetime(res["Datetime"]), dd)
    ax.set_title("Underwater (Drawdown)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown (USD)")
    fig.tight_layout()
    fig.savefig(str(Path(charts_dir) / "underwater.png"), dpi=150)
    plt.close(fig)

    # Daily return histogram
    df = res.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    eq = df.set_index("Datetime")["equity"].astype(float)
    ret = eq.pct_change().fillna(0.0)
    daily = (1.0 + ret).resample("D").prod() - 1.0
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.hist(daily.dropna().values, bins=50)
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily Return")
    fig.tight_layout()
    fig.savefig(str(Path(charts_dir) / "daily_pnl_hist.png"), dpi=150)
    plt.close(fig)


def _extract_trades_from_res(res: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-bar results into a simple trades log (entry/exit, side, pnl, hold time).
    Uses equity deltas, so it aligns with contract-mode costs & RiskEngine exits.
    """
    df = res.copy()
    t = pd.to_datetime(df["Datetime"])
    pos = df["position"].astype(float).values
    eq = df["equity"].astype(float).values

    entries: List[Dict[str, Any]] = []
    entry_i: Optional[int] = None
    entry_side: int = 0
    entry_eq: Optional[float] = None

    for i in range(1, len(df)):
        prev_s, cur_s = int(np.sign(pos[i-1])), int(np.sign(pos[i]))
        # Entry or reversal closes previous trade
        if (prev_s == 0 and cur_s != 0) or (prev_s != 0 and cur_s != 0 and prev_s != cur_s):
            if entry_i is not None:
                pnl_usd = float(eq[i] - float(entry_eq))
                hold_min = float((t.iloc[i] - t.iloc[entry_i]).total_seconds() / 60.0)
                entries.append(dict(
                    entry_time=t.iloc[entry_i], exit_time=t.iloc[i],
                    side=("LONG" if entry_side>0 else "SHORT"),
                    entry_equity=float(entry_eq), exit_equity=float(eq[i]),
                    pnl_usd=pnl_usd, pnl_pct=pnl_usd / max(1e-9, float(entry_eq)),
                    hold_min=hold_min,
                ))
            entry_i = i
            entry_side = cur_s
            entry_eq = float(eq[i-1])
        elif prev_s != 0 and cur_s == 0 and entry_i is not None:
            pnl_usd = float(eq[i] - float(entry_eq))
            hold_min = float((t.iloc[i] - t.iloc[entry_i]).total_seconds() / 60.0)
            entries.append(dict(
                entry_time=t.iloc[entry_i], exit_time=t.iloc[i],
                side=("LONG" if entry_side>0 else "SHORT"),
                entry_equity=float(entry_eq), exit_equity=float(eq[i]),
                pnl_usd=pnl_usd, pnl_pct=pnl_usd / max(1e-9, float(entry_eq)),
                hold_min=hold_min,
            ))
            entry_i = None; entry_eq = None; entry_side = 0

    trades = pd.DataFrame(entries)
    if not trades.empty:
        wins = int((trades["pnl_usd"] > 0).sum())
        wr = wins / max(1, len(trades))
        logger.info(f"Trades: {len(trades)} | Wins: {wins} | Win rate: {wr:.2%}")
    return trades


def _build_news_mask_if_any(args: argparse.Namespace, feats: pd.DataFrame, session_tz: str) -> Optional[pd.Series]:
    if not getattr(args, "news_csv", None):
        return None
    if NewsRiskFilter is None:
        raise RuntimeError("news_csv provided but NewsRiskFilter module not available.")

    impacts = [s.strip().upper() for s in str(getattr(args, "news_impacts", "HIGH,MEDIUM")).split(',') if s.strip()]
    currencies = [s.strip().upper() for s in str(getattr(args, "news_currencies", "USD,EUR,GBP,JPY,CNY,AUD,NZD,CAD,CHF")).split(',') if s.strip()]

    nf = NewsRiskFilter(
        tz=session_tz,
        impacts=impacts,
        currencies=currencies,
        ttl_sec=999999,
        treat_upcoming_as_block=bool(int(getattr(args, 'news_upcoming_is_block', 0))),
        upcoming_lookahead_min=int(getattr(args, 'news_upcoming_min', 1)),
        preset=getattr(args, 'preset_config', None),
    )
    nf.refresh(csv_path=getattr(args, 'news_csv'), source_tz=getattr(args, 'news_source_tz', None))

    idx = pd.DatetimeIndex(pd.to_datetime(feats["Datetime"]))
    return nf.mask_for_index(idx)


def _save_manifest(path_like: str | Path, payload: Dict[str, Any]) -> None:
    try:
        p = Path(path_like)
        p.write_text(json.dumps(payload, indent=2, default=str))
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to write manifest {path_like}: {e}")


# ---------------------- Core API ----------------------
def run_backtest(
    csv_path: str | Path,
    model_path: str | Path,
    cost: Union[CostConfig, ContractCostConfig, None] = None,
    risk: RiskConfig | None = None,
    out_csv: str = "backtest_results.csv",
    out_plot: str = "equity_curve.png",
    out_trades: str = "trades.csv",
    charts_dir: str = "charts",
    # Engine kwargs (session, window, grades, instrument, account_scale_usd, EV gates, etc.)
    **engine_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """
    High-level pipeline:
      CSV -> features -> model -> proba (calibrated) -> EV + context gates -> RiskEngine -> results/charts/summary.
    """
    logger.info(f"Loading market data: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    df_raw = _normalize_ohlcv_columns(df_raw)

    # Build features using optional session params from engine_kwargs
    tz = engine_kwargs.pop("tz", DEFAULTS.session_tz)
    rth_start = engine_kwargs.pop("rth_start", getattr(DEFAULTS, 'rth_start', '07:30'))
    rth_end = engine_kwargs.pop("rth_end", getattr(DEFAULTS, 'rth_end', '14:00'))
    orb_min = int(engine_kwargs.pop("orb_min", getattr(DEFAULTS, 'orb_min', 15)))

    logger.info("Building features...")
    feats = build_features(df_raw.copy(), tz=tz, rth_start=rth_start, rth_end=rth_end, orb_minutes=orb_min)
    if not isinstance(feats, pd.DataFrame):
        raise TypeError("build_features(...) must return a pandas.DataFrame")

    # Ensure Datetime column
    if "Datetime" not in feats.columns:
        if isinstance(feats.index, pd.DatetimeIndex):
            idx = feats.index
            feats["Datetime"] = (idx.tz_convert("UTC").tz_localize(None) if getattr(idx, "tz", None) is not None else idx)
            feats = feats.reset_index(drop=True)
        else:
            raise ValueError("Features must include a 'Datetime' column or have a DatetimeIndex.")

    # Keep base OHLCV if missing
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in feats.columns and c in df_raw.columns:
            feats[c] = df_raw[c].values

    # Resolve & load model (+ features list + optional calibrator)
    model_path = str(model_path)
    logger.info(f"Loading model & metadata: {model_path}")
    try:
        model, expected, calibrator = load_model_features_and_calibrator(model_path)
    except Exception as e:
        logger.warning(f"Registry-aware loader failed ({e}); falling back to joblib...")
        model = joblib.load(model_path)
        expected = None
        calibrator = None

    # Feature order
    if expected:
        logger.info(f"Using {len(expected)} training features (feature list available)")
    else:
        # try sidecar (legacy)
        sidecar = features_sidecar_path(model_path)
        if sidecar and Path(sidecar).is_file():
            try:
                spec = json.loads(Path(sidecar).read_text())
                if isinstance(spec, dict) and "features" in spec:
                    expected = list(spec["features"])
                elif isinstance(spec, list):
                    expected = list(spec)
                if expected:
                    logger.info(f"Using {len(expected)} features (from sidecar)")
            except Exception:
                pass

    # Build X
    exclude = set(BASE_COLS + ENGINE_COLS)
    X = feats.drop(columns=[c for c in feats.columns if c in exclude], errors="ignore")
    X, missing = _align_features(X, expected)
    if missing:
        logger.warning(f"{len(missing)} expected features missing at inference (filled with 0.0). Examples: {missing[:10]}")
    # Clean X only (preserve row alignment)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    logger.info("Generating predictions...")
    proba_raw = _predict_proba(model, X)
    if len(proba_raw) != len(feats):
        raise ValueError(f"Prediction length mismatch: proba={len(proba_raw)} vs feats={len(feats)}")

    # Apply optional probability calibration
    proba = calibrate_proba(proba_raw, calibrator)

    # Defaults for cost/risk
    if cost is None:
        # If commission/slip provided in engine_kwargs, honor contract mode
        if ("commission" in engine_kwargs and engine_kwargs["commission"] is not None) or ("slip_ticks" in engine_kwargs and engine_kwargs["slip_ticks"] is not None):
            cost = ContractCostConfig(
                commission_per_contract=float(engine_kwargs.pop("commission", CONTRACT_COST.commission_per_contract)),
                slippage_ticks_per_side=float(engine_kwargs.pop("slip_ticks", CONTRACT_COST.slippage_ticks_per_side)),
            )
        else:
            cost = DEFAULT_COST

    risk = risk or RiskConfig(**RISK_DEFAULTS)

    # Instrument & account scale (contract ledger)
    instrument = engine_kwargs.pop("instrument", None)
    instrument_alias = engine_kwargs.pop("instrument_alias", None)
    if instrument is None and instrument_alias is not None:
        instrument = instrument_by_alias(instrument_alias)
    # Fallback to ENGINE default alias if still None
    if instrument is None and getattr(DEFAULTS, "instrument_alias", None):
        instrument = instrument_by_alias(DEFAULTS.instrument_alias)

    # Profit lock options
    profit_lock_usd = engine_kwargs.pop("profit_lock_usd", None)
    near_breach_buffer_usd = engine_kwargs.pop("near_breach_buffer_usd", 100.0)

    # Optional News mask: only pass if the backtest supports it
    news_mask = None
    try:
        # Build mask using the CLI-ish fields if present in engine_kwargs
        dummy_args = argparse.Namespace(
            news_csv=engine_kwargs.pop('news_csv', None),
            news_source_tz=engine_kwargs.pop('news_source_tz', None),
            news_upcoming_is_block=engine_kwargs.pop('news_upcoming_is_block', 0),
            news_upcoming_min=engine_kwargs.pop('news_upcoming_min', 1),
            news_impacts=engine_kwargs.pop('news_impacts', 'HIGH,MEDIUM'),
            news_currencies=engine_kwargs.pop('news_currencies', 'USD,EUR,GBP,JPY,CNY,AUD,NZD,CAD,CHF'),
        )
        if getattr(dummy_args, 'news_csv', None):
            news_mask = _build_news_mask_if_any(dummy_args, feats, session_tz=engine_kwargs.get('session_tz', DEFAULTS.session_tz))
    except Exception as e:
        logger.warning(f"Failed to build news mask: {e}")
        news_mask = None

    # Prepare call kwargs
    call_kwargs = dict(
        df=feats,
        proba=proba,
        cost=cost,
        risk=risk,
        close_col="Close",
        datetime_col="Datetime",
        instrument=instrument,
        profit_lock_usd=profit_lock_usd,
        near_breach_buffer_usd=near_breach_buffer_usd,
        **engine_kwargs,
    )

    # Add news mask only if function accepts it
    bt_sig = inspect.signature(backtest_threshold_futures)
    if 'news_block_mask' in bt_sig.parameters and news_mask is not None:
        call_kwargs['news_block_mask'] = news_mask

    # Backtest
    logger.info("Running backtest (prop-firm risk integrated)...")
    res, summary = backtest_threshold_futures(**call_kwargs)
    res_df = pd.DataFrame(res)

    # Save outputs
    Path(out_csv).write_text(res_df.to_csv(index=False))
    trades = _extract_trades_from_res(res_df.copy())
    Path(out_trades).write_text(trades.to_csv(index=False))

    _make_charts(res_df, out_plot, charts_dir)

    # Pretty print summary
    logger.info(f"Results saved to {Path(out_csv).resolve()}")
    logger.info(f"Trades log saved to {Path(out_trades).resolve()}")
    logger.info(f"Equity curve plot saved to {Path(out_plot).resolve()}")
    logger.info(f"Charts saved to {Path(charts_dir).resolve()}")
    print("[OK] Backtest Summary:")
    show_keys = [
        "instrument","initial_equity","final_equity","total_pnl_usd","trades","max_drawdown",
        "ev_thresh_long","ev_thresh_short","target_r","policy_margin","prob_cut_exits",
        "trail_profit_exits","stopped_be_exits","stopped_atr_exits","prop_breaches"
    ]
    for k in show_keys:
        if k in summary:
            print(f"  {k}: {summary[k]}")

    # Write a manifest for reproducibility (next to out_csv)
    manifest_path = Path(out_csv).with_name("run_manifest.json")
    try:
        model_sha = compute_sha256(model_path)
    except Exception:
        model_sha = None
    _save_manifest(
        manifest_path,
        {
            "inputs": {
                "csv_path": str(Path(csv_path).resolve()),
                "model_path": str(Path(model_path).resolve()),
                "model_sha256": model_sha,
            },
            "engine_kwargs": {k: (getattr(v, "alias", v) if k=="instrument" else v) for k, v in call_kwargs.items() if k not in {"df","proba","cost","risk","feature_context","preset_config"}},
            "cost": getattr(cost, "__dict__", str(cost)),
            "risk": getattr(risk, "__dict__", str(risk)),
            "summary": summary,
            "n_rows": int(len(res_df)),
        },
    )

    return res_df, summary


# ---------------------- CLI ----------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest a saved model with prop-firm risk + EV gating (contract mode ready)")
    p.add_argument("csv_path")
    p.add_argument("model_path_or_alias")

    # Preset (optional)
    if isinstance(PRESETS, dict):
        p.add_argument("--preset", choices=sorted(PRESETS.keys()))
    else:
        p.add_argument("--preset")

    # Feature/session params
    p.add_argument("--tz", default=DEFAULTS.session_tz)
    p.add_argument("--rth_start", default=getattr(DEFAULTS, "rth_start", "07:30"))
    p.add_argument("--rth_end", default=getattr(DEFAULTS, "rth_end", "14:00"))
    p.add_argument("--orb_min", type=int, default=getattr(DEFAULTS, "orb_min", 15))

    # EV thresholds
    p.add_argument("--target_r", type=float, default=None)
    p.add_argument("--policy_margin", type=float, default=0.0)

    # Fallback thresholds & grading
    p.add_argument("--p_buy", type=float, default=None)
    p.add_argument("--p_sell", type=float, default=None)
    p.add_argument("--allow_grades", type=str, default=None)

    # Context gates
    p.add_argument("--no_vwap_gate", action="store_true")
    p.add_argument("--no_ema_gate", action="store_true")

    # Instrument & contract costs
    p.add_argument("--instrument", type=str, default=None)
    p.add_argument("--commission", type=float, default=None)
    p.add_argument("--slip_ticks", type=float, default=None)
    p.add_argument("--account_scale_usd", type=float, default=None)

    # Equity-bps (legacy)
    p.add_argument("--fee_bps", type=float, default=None)
    p.add_argument("--slippage_bps", type=float, default=None)

    # Session / window / limits
    p.add_argument("--session_tz", type=str, default=None)
    p.add_argument("--start", dest="trade_window_start", type=str, default=None)
    p.add_argument("--end", dest="trade_window_end", type=str, default=None)
    p.add_argument("--max_trades", dest="max_trades_per_day", type=int, default=None)

    # Risk basics (+ a few prop toggles)
    p.add_argument("--per_stop_pct", dest="per_trade_stop_pct", type=float, default=None)
    p.add_argument("--per_stop_usd", dest="per_trade_stop_usd", type=float, default=None)
    p.add_argument("--daily_stop_pct", dest="daily_loss_stop_pct", type=float, default=None)
    p.add_argument("--daily_stop_usd", type=float, default=None)
    p.add_argument("--trail_dd_pct", dest="trailing_drawdown_pct", type=float, default=None)
    p.add_argument("--trail_dd_usd", type=float, default=None)
    p.add_argument("--max_position", type=int, default=None)
    p.add_argument("--proba_cut_bad", type=float, default=None)
    p.add_argument("--trail_profit_pct", type=float, default=None)
    p.add_argument("--trail_profit_activate_pct", type=float, default=None)
    p.add_argument("--flip_buf", dest="flip_exit_buf", type=float, default=None)
    p.add_argument("--atr_len", type=int, default=None)
    p.add_argument("--atr_k", type=float, default=None)
    p.add_argument("--be_rr", dest="breakeven_activate_rr", type=float, default=None)
    p.add_argument("--be_pct", dest="be_activate_pct", type=float, default=None)
    p.add_argument("--be_buffer_pct", type=float, default=None)
    p.add_argument("--max_bars", dest="max_bars_in_trade", type=int, default=None)
    p.add_argument("--cooldown_bars", dest="cooldown_bars_after_stop", type=int, default=None)

    # Vol targeting
    p.add_argument("--vol_target", action="store_true")
    p.add_argument("--target_vol", type=float, default=None)
    p.add_argument("--vol_ema_span", type=int, default=None)
    p.add_argument("--vol_annualize_k", type=float, default=None)
    p.add_argument("--pos_cap", type=float, default=None)

    # LLM telemetry hooks
    p.add_argument("--use-llm", dest="use_llm", action="store_true")
    p.add_argument("--llm-risk-bps", dest="llm_risk_bps", type=int, default=None)
    p.add_argument("--llm-cooldown-min", dest="llm_cooldown_min", type=int, default=None)
    p.add_argument("--symbol", type=str, default=None)

    # Prop-firm extras
    p.add_argument("--prop_trail_dd_usd", type=float, default=None)
    p.add_argument("--prop_trail_on_unrealized", action="store_true")
    p.add_argument("--prop_breach_buffer_usd", type=float, default=None)
    p.add_argument("--stop_after_win", action="store_true")
    p.add_argument("--no_stop_after_win", action="store_true")
    p.add_argument("--trail_intrabar", action="store_true")

    # EOD trailing profit lock
    p.add_argument("--profit_lock_usd", type=float, default=None)
    p.add_argument("--near_breach_buffer_usd", type=float, default=100.0)

    # News gating (optional)
    p.add_argument("--news_csv", type=str, default=None, help="CSV of scheduled events (time_local,currency,impact,title)")
    p.add_argument("--news_source_tz", type=str, default=None, help="Timezone of naive event times, e.g., America/New_York")
    p.add_argument("--news_impacts", type=str, default="HIGH,MEDIUM")
    p.add_argument("--news_currencies", type=str, default="USD,EUR,GBP,JPY,CNY,AUD,NZD,CAD,CHF")
    p.add_argument("--news_upcoming_is_block", type=int, default=0)
    p.add_argument("--news_upcoming_min", type=int, default=1)

    # Outputs
    p.add_argument("--out_csv", default="backtest_results.csv")
    p.add_argument("--out_trades", default="trades.csv")
    p.add_argument("--out_plot", default="equity_curve.png")
    p.add_argument("--charts_dir", default="charts")
    return p


def _apply_preset(ns: argparse.Namespace) -> None:
    if not getattr(ns, 'preset', None):
        return
    if not isinstance(PRESETS, dict) or ns.preset not in PRESETS:
        raise ValueError(f"Unknown preset: {ns.preset}")
    preset = PRESETS[ns.preset]
    # Map preset keys -> CLI field names where they differ
    keymap = {
        "max_trades": "max_trades_per_day",
        "per_stop_pct": "per_trade_stop_pct",
        "per_stop_usd": "per_trade_stop_usd",
        "daily_stop_pct": "daily_loss_stop_pct",
        "trail_dd_pct": "trailing_drawdown_pct",
        "daily_stop_usd": "daily_loss_stop_usd",
        "trail_dd_usd": "trail_dd_usd",
        "prop_trail_dd_usd": "prop_trail_dd_usd",
        "profit_lock_usd": "profit_lock_usd",
        "near_breach_buffer_usd": "near_breach_buffer_usd",
        "commission": "commission",
        "slip_ticks": "slip_ticks",
        "session_tz": "session_tz",
        "trade_window_start": "trade_window_start",
        "trade_window_end": "trade_window_end",
        "allowed_grades": "allow_grades",
        "p_buy": "p_buy",
        "p_sell": "p_sell",
        "pos_cap": "pos_cap",
        "max_position": "max_position",
        "target_r": "target_r",
        "policy_margin": "policy_margin",
    }
    # set values only if not set by CLI
    for k, v in preset.items():
        dest = keymap.get(k, k)
        if hasattr(ns, dest) and getattr(ns, dest) is None and dest not in ("vol_target",):
            setattr(ns, dest, v)
    # bool flags
    if "vol_target" in preset and not getattr(ns, 'vol_target', False):
        ns.vol_target = bool(preset["vol_target"])
    if "stop_after_win" in preset and not getattr(ns, 'stop_after_win', False):
        ns.stop_after_win = bool(preset["stop_after_win"])
    if "trail_intrabar" in preset and not getattr(ns, 'trail_intrabar', False):
        ns.trail_intrabar = bool(preset["trail_intrabar"])


def main():
    ap = _build_arg_parser()
    ns = ap.parse_args()

    # Apply preset first (allows CLI to override)
    _apply_preset(ns)

    active_preset = None
    if getattr(ns, "preset", None):
        try:
            active_preset = deepcopy(PRESETS[ns.preset])
        except Exception:
            active_preset = dict(PRESETS[ns.preset])
    feature_ctx = build_feature_context(preset=active_preset, name=getattr(ns, "preset", None))

    # Prefer contract-mode if commission/slip provided
    if ns.commission is not None or ns.slip_ticks is not None:
        ns.fee_bps = None; ns.slippage_bps = None

    # Hydrate cost
    if ns.commission is not None or ns.slip_ticks is not None:
        cost: Union[CostConfig, ContractCostConfig] = ContractCostConfig(
            commission_per_contract=(ns.commission if ns.commission is not None else CONTRACT_COST.commission_per_contract),
            slippage_ticks_per_side=(ns.slip_ticks if ns.slip_ticks is not None else CONTRACT_COST.slippage_ticks_per_side),
        )
    elif ns.fee_bps is None and ns.slippage_bps is None:
        cost = DEFAULT_COST
    else:
        cost = CostConfig(
            fee_bps=ns.fee_bps if ns.fee_bps is not None else DEFAULT_COST.fee_bps,
            slippage_bps=ns.slippage_bps if ns.slippage_bps is not None else DEFAULT_COST.slippage_bps,
        )

    # RiskConfig overrides
    rc = dict(RISK_DEFAULTS)
    keymap = [
        "per_trade_stop_pct","per_trade_stop_usd","daily_loss_stop_pct","trailing_drawdown_pct","max_position",
        "proba_cut_bad","trail_profit_pct","trail_profit_activate_pct","flip_exit_buf","atr_len","atr_k",
        "breakeven_activate_rr","be_activate_pct","be_buffer_pct","max_bars_in_trade","cooldown_bars_after_stop",
    ]
    for k in keymap:
        v = getattr(ns, k, None)
        if v is not None:
            rc[k] = v
    if getattr(ns, 'daily_stop_usd', None) is not None:
        rc["daily_loss_stop_usd"] = float(ns.daily_stop_usd)
    if getattr(ns, 'prop_trail_dd_usd', None) is not None:
        rc["prop_trailing_dd_usd"] = float(ns.prop_trail_dd_usd); rc["prop_enabled"] = True
    if getattr(ns, 'trail_dd_usd', None) is not None and rc.get("prop_trailing_dd_usd") is None:
        rc["prop_trailing_dd_usd"] = float(ns.trail_dd_usd); rc["prop_enabled"] = True
    if getattr(ns, 'stop_after_win', False) and not getattr(ns, 'no_stop_after_win', False):
        rc["stop_after_first_win"] = True
    if getattr(ns, 'no_stop_after_win', False):
        rc["stop_after_first_win"] = False
    if getattr(ns, 'trail_intrabar', False):
        rc["trail_use_intrabar_extremes"] = True
    if getattr(ns, 'prop_trail_on_unrealized', False):
        rc["prop_trail_on_unrealized"] = True
    if getattr(ns, 'prop_breach_buffer_usd', None) is not None:
        rc["prop_breach_buffer_usd"] = float(ns.prop_breach_buffer_usd)
    risk = RiskConfig(**rc)

    # Engine kwargs
    engine_kwargs = dict(
        # Feature/session
        tz=ns.tz, rth_start=ns.rth_start, rth_end=ns.rth_end, orb_min=int(ns.orb_min),
        # EV thresholds & fallback
        target_r=(ns.target_r if ns.target_r is not None else None),
        policy_margin=float(ns.policy_margin or 0.0),
        p_buy=(ns.p_buy if ns.p_buy is not None else DEFAULTS.p_buy),
        p_sell=(ns.p_sell if ns.p_sell is not None else DEFAULTS.p_sell),
        allowed_grades=tuple(g.strip() for g in (ns.allow_grades or ",".join(DEFAULTS.allowed_grades)).split(",")),
        prob_bands=PROB_BANDS,
        # Session/window/limits
        session_tz=(ns.session_tz or DEFAULTS.session_tz),
        trade_window_start=(ns.trade_window_start or DEFAULTS.trade_window_start),
        trade_window_end=(ns.trade_window_end or DEFAULTS.trade_window_end),
        max_trades_per_day=(ns.max_trades_per_day or DEFAULTS.max_trades_per_day),
        # Contract ledger
        account_scale_usd=float(ns.account_scale_usd if ns.account_scale_usd is not None else DEFAULTS.account_scale_usd),
        # DD circuit
        enable_dd_circuit=DEFAULTS.enable_dd_circuit,
        dd_limit=DEFAULTS.dd_limit,
        dd_resume_hysteresis=DEFAULTS.dd_resume_hysteresis,
        dd_disable_from_next_bar=DEFAULTS.dd_disable_from_next_bar,
        # Vol targeting
        enable_vol_target=(ns.vol_target or DEFAULTS.enable_vol_target),
        target_vol=(ns.target_vol if ns.target_vol is not None else DEFAULTS.target_vol),
        vol_ema_span=(ns.vol_ema_span if ns.vol_ema_span is not None else DEFAULTS.vol_ema_span),
        vol_annualize_k=(ns.vol_annualize_k if ns.vol_annualize_k is not None else DEFAULTS.vol_annualize_k),
        pos_cap=(ns.pos_cap if ns.pos_cap is not None else DEFAULTS.pos_cap),
        # LLM telemetry
        use_llm=(ns.use_llm or DEFAULTS.use_llm),
        llm_review_all=DEFAULTS.llm_review_all,
        llm_max_risk_bps=(ns.llm_risk_bps if ns.llm_risk_bps is not None else DEFAULTS.llm_max_risk_bps),
        llm_cooldown_min=(ns.llm_cooldown_min if ns.llm_cooldown_min is not None else DEFAULTS.llm_cooldown_min),
        symbol=(ns.symbol or DEFAULTS.symbol),
        # Instrument spec (resolved inside run_backtest too)
        instrument=(instrument_by_alias(ns.instrument) if ns.instrument else None),
        # Context gates
        enable_vwap_gate=not bool(ns.no_vwap_gate),
        enable_ema_gate=not bool(ns.no_ema_gate),
        # EOD profit lock
        profit_lock_usd=(ns.profit_lock_usd if ns.profit_lock_usd is not None else None),
        near_breach_buffer_usd=float(ns.near_breach_buffer_usd or 100.0),
        # Pass-through contract costs to cost builder (handled earlier if cost is None)
        commission=ns.commission, slip_ticks=ns.slip_ticks,
        # News gating pass-through (runner will build mask if present)
        news_csv=getattr(ns, 'news_csv', None),
        news_source_tz=getattr(ns, 'news_source_tz', None),
        news_impacts=getattr(ns, 'news_impacts', 'HIGH,MEDIUM'),
        news_currencies=getattr(ns, 'news_currencies', 'USD,EUR,GBP,JPY,CNY,AUD,NZD,CAD,CHF'),
        news_upcoming_is_block=getattr(ns, 'news_upcoming_is_block', 0),
        news_upcoming_min=getattr(ns, 'news_upcoming_min', 1),
        feature_context=feature_ctx,
        preset_config=active_preset,
    )

    # Threshold sanity
    if (engine_kwargs["p_buy"] is not None) and (engine_kwargs["p_sell"] is not None):
        if engine_kwargs["p_buy"] <= engine_kwargs["p_sell"]:
            logger.warning(f"p_buy ({engine_kwargs['p_buy']}) <= p_sell ({engine_kwargs['p_sell']}) → may stay flat.")

    model_path = resolve_model_path(ns.model_path_or_alias)

    run_backtest(
        csv_path=ns.csv_path,
        model_path=model_path,
        cost=cost,
        risk=risk,
        out_csv=ns.out_csv,
        out_trades=ns.out_trades,
        out_plot=ns.out_plot,
        charts_dir=ns.charts_dir,
        **engine_kwargs,
    )


if __name__ == "__main__":
    main()

__all__ = ["run_backtest"]
