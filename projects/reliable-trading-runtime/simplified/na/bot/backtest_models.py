#backtest_models.py
from __future__ import annotations
import argparse, json, math, inspect
from copy import deepcopy
from pathlib import Path, PurePath
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from joblib import load as joblib_load

try:  # pragma: no cover - dual import for script vs package usage
    from common.proba import ensure_long_index, select_long_proba  # type: ignore
except ImportError:  # pragma: no cover
    from na.common.proba import ensure_long_index, select_long_proba  # type: ignore

# Local imports
from .features import build_features
from .backtest_threshold import backtest_threshold_futures
from .models import model_path as resolve_model_path

from .config import (
    ENGINE,
    COST,
    CONTRACT_COST,
    RISK_DEFAULTS,
    PROB_BANDS,
    PRESETS,
    CostConfig,
    ContractCostConfig,
    instrument_by_alias,
    build_feature_context,
)
from na.l3.bot.risk_engine import RiskConfig

# Optional news risk filter
try:
    from .news_filter import NewsRiskFilter
except Exception:  # pragma: no cover
    NewsRiskFilter = None  # type: ignore


# -------------------- Helpers --------------------

def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Safer OHLCV renaming: avoids collisions with e.g. open_interest."""
    rename = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("datetime", "date", "time", "timestamp"):
            rename[c] = "Datetime"
        elif lc == "open" or lc.endswith("_open"):
            rename[c] = "Open"
        elif lc == "high" or lc.endswith("_high"):
            rename[c] = "High"
        elif lc == "low" or lc.endswith("_low"):
            rename[c] = "Low"
        elif lc == "close" or lc.endswith("_close"):
            rename[c] = "Close"
        elif lc == "volume" or lc.endswith("_volume"):
            rename[c] = "Volume"
    if rename:
        df = df.rename(columns=rename)

    if "Datetime" not in df.columns:
        raise ValueError("No Datetime column found.")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if keep:
        df = df.dropna(subset=keep).reset_index(drop=True)
    return df


def _drop_leak_prone_columns(X: pd.DataFrame) -> pd.DataFrame:
    drop_exact = {"Open", "High", "Low", "Close", "Volume"}
    suspicious = ("follow", "future", "lead", "ahead", "fwd")
    safe_suspicious = {"hi_break_follow", "lo_break_follow"}
    cols: List[str] = []
    for c in X.columns:
        if c in drop_exact:
            continue
        lc = str(c).lower()
        if c not in safe_suspicious and any(s in lc for s in suspicious):
            continue
        cols.append(c)
    return X[cols]


def _load_model_and_feature_list(path: Path):
    model = joblib_load(path)

    meta: Optional[dict] = None
    try:
        meta_path = Path(path).with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
    except Exception:
        meta = None

    if meta is not None:
        try:
            setattr(model, "_na_meta", meta)
        except Exception:
            pass
    ensure_long_index(model, meta=meta, model_path=path)

    feats: Optional[List[str]] = None
    # sidecar features.json if available
    try:
        sidecar = Path(path).with_suffix(".features.json")
        if sidecar.exists():
            js = json.loads(sidecar.read_text())
            if isinstance(js, dict):
                feats = js.get("features")
            elif isinstance(js, list):
                feats = js
    except Exception:
        feats = None
    if feats:
        return model, list(feats)
    # fallback to sklearn feature_names_in_
    try:
        feat_names = list(getattr(model, "feature_names_in_"))
    except Exception:
        feat_names = None
    return model, feat_names


def _align_X_for_model(feats: pd.DataFrame, feature_names: List[str] | None) -> Tuple[pd.DataFrame, List[str]]:
    X = feats.select_dtypes(include=[np.number, "bool"]).astype(float)
    if feature_names is None:
        X = _drop_leak_prone_columns(X)
    if feature_names is None:
        return X, list(X.columns)
    for f in feature_names:
        if f not in X.columns:
            X[f] = 0.0
    X = X[feature_names]
    return X, feature_names


def _predict_proba_safely(model, X: pd.DataFrame) -> np.ndarray:
    def _sigmoid(z):
        z = np.asarray(z, dtype=float)
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def _proba_from_model(M, D):
        meta = getattr(M, "_na_meta", None)
        if hasattr(M, "predict_proba"):
            p = M.predict_proba(D)
            return select_long_proba(p, M, meta=meta, model_path=None)
        if hasattr(M, "decision_function"):
            scores = np.asarray(M.decision_function(D), dtype=float)
            scores = np.clip(scores, -50, 50)
            return 1.0 / (1.0 + np.exp(-scores))
        pred = M.predict(D)
        arr = np.asarray(pred, dtype=float)
        classes = getattr(M, "classes_", None)
        if classes is not None:
            long_idx = ensure_long_index(M, meta=meta)
            try:
                classes_list = list(classes)
                long_code = classes_list[int(long_idx)]
                return (arr == long_code).astype(float)
            except Exception:
                pass
        if np.nanmax(arr) > 1.2 or np.nanmin(arr) < -0.2:
            arr = _sigmoid(arr)
        return arr

    try:
        raw = _proba_from_model(model, X)
    except Exception:
        X2 = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = _proba_from_model(model, X2)
    return np.clip(select_long_proba(raw, model, meta=getattr(model, "_na_meta", None)), 0.0, 1.0)

def _make_trades_csv(df_bt: pd.DataFrame, close_col: str) -> pd.DataFrame:
    """Build a simple trades log from backtest output (full round-trips)."""
    pos = df_bt["position"].values.astype(float)
    close = df_bt[close_col].values.astype(float)
    eq = df_bt["equity"].values.astype(float)
    dt = pd.to_datetime(df_bt["Datetime"]) if "Datetime" in df_bt.columns else pd.to_datetime(df_bt.index)

    trades: List[Dict[str, Any]] = []
    in_pos = False
    side = 0.0
    size = 0.0
    entry_idx = None
    entry_price = None
    entry_equity = None

    for i in range(1, len(df_bt)):
        prev = pos[i-1]
        now = pos[i]
        if not in_pos and now != 0:
            in_pos = True
            side = np.sign(now)
            size = abs(now)
            entry_idx = i
            entry_price = close[i]
            entry_equity = eq[i-1]
        elif in_pos and (now == 0 or np.sign(now) != side):
            exit_idx = i
            exit_price = close[i]
            pnl_usd = float(eq[exit_idx] - float(entry_equity))
            trades.append(dict(
                entry_time=str(dt[entry_idx]), exit_time=str(dt[exit_idx]),
                side=("LONG" if side>0 else "SHORT"), size=float(size),
                entry_price=float(entry_price), exit_price=float(exit_price),
                pnl_usd=float(pnl_usd),
            ))
            in_pos = False

    return pd.DataFrame(trades)


def _get(ns: argparse.Namespace, *names: str, default=None):
    """Return first non-None attribute among names; safe if attribute missing."""
    for name in names:
        if hasattr(ns, name):
            val = getattr(ns, name)
            if val is not None:
                return val
    return default


# -------------------- CLI & Runner --------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest a saved model with prop-firm risk integration (contract mode)")

    # Required
    p.add_argument("csv_path")
    p.add_argument("model")  # alias or path
    p.add_argument("--preset", choices=list(PRESETS.keys()))

    # Features pipeline
    p.add_argument("--tz", default="America/Denver")
    p.add_argument("--rth_start", default="07:30")
    p.add_argument("--rth_end", default="14:00")
    p.add_argument("--orb_min", type=int, default=15)

    # EV thresholds (optional; only passed if supported downstream)
    p.add_argument("--target_r", type=float, default=None)
    p.add_argument("--policy_margin", type=float, default=0.0)

    # Core thresholds & grading (fallback)
    p.add_argument("--p_buy", type=float, default=None)
    p.add_argument("--p_sell", type=float, default=None)
    p.add_argument("--allow_grades", type=str, default=None)  # "A+,B+"

    # Context gates (only forwarded if accepted downstream)
    p.add_argument("--no_vwap_gate", action="store_true")
    p.add_argument("--no_ema_gate", action="store_true")

    # === Contract mode: instrument & costs ===
    p.add_argument("--instrument", type=str, default=None, help="Instrument alias (ES, NQ, MES, MNQ)")
    p.add_argument("--commission", type=float, default=None, help="USD per contract per side")
    p.add_argument("--slip_ticks", type=float, default=None, help="Slippage ticks per side")
    p.add_argument("--account_scale_usd", type=float, default=None, help="Baseline USD ledger")

    # Costs (legacy bps)
    p.add_argument("--fee_bps", type=float, default=None)
    p.add_argument("--slippage_bps", type=float, default=None)

    # Session / window / limits
    p.add_argument("--session_tz", type=str, default=None)
    p.add_argument("--start", dest="trade_window_start", type=str, default=None)
    p.add_argument("--end", dest="trade_window_end", type=str, default=None)
    p.add_argument("--max_trades", dest="max_trades_per_day", type=int, default=None)

    # Risk exits → RiskConfig
    p.add_argument("--per_stop_pct", dest="per_trade_stop_pct", type=float, default=None)
    p.add_argument("--per_stop_usd", dest="per_trade_stop_usd", type=float, default=None)
    p.add_argument("--daily_stop_pct", dest="daily_loss_stop_pct", type=float, default=None)
    p.add_argument("--daily_stop_usd", type=float, default=None)
    p.add_argument("--daily_loss_stop_usd", type=float, default=None, help="Alias for --daily_stop_usd")
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

    # LLM (telemetry only)
    p.add_argument("--use-llm", dest="use_llm", action="store_true")
    p.add_argument("--llm-risk-bps", dest="llm_risk_bps", type=int, default=None)
    p.add_argument("--llm-cooldown-min", dest="llm_cooldown_min", type=int, default=None)
    p.add_argument("--symbol", type=str, default=None)

    # Prop-firm toggles
    p.add_argument("--prop_trail_dd_usd", type=float, default=None)
    p.add_argument("--prop_trail_on_unrealized", action="store_true")
    p.add_argument("--prop_breach_buffer_usd", type=float, default=None)
    p.add_argument("--stop_after_win", action="store_true")
    p.add_argument("--no_stop_after_win", action="store_true")
    p.add_argument("--trail_intrabar", action="store_true")

    # EOD trailing profit lock (optional)
    p.add_argument("--profit_lock_usd", type=float, default=None)
    p.add_argument("--near_breach_buffer_usd", type=float, default=100.0)

    # News risk (optional)
    p.add_argument("--news_csv", type=str, default=None, help="CSV of scheduled events (time_local,currency,impact,title)")
    p.add_argument("--news_source_tz", type=str, default=None, help="Timezone of naive event times (e.g., America/New_York)")
    p.add_argument("--news_impacts", type=str, default="HIGH,MEDIUM")
    p.add_argument("--news_currencies", type=str, default="USD,EUR,GBP,JPY,CNY,AUD,NZD,CAD,CHF")
    
    # Outputs
    p.add_argument("--out_csv", default="backtest_results.csv")
    p.add_argument("--out_trades", default="trades.csv")
    p.add_argument("--out_plot", default="equity_curve.png")
    p.add_argument("--charts_dir", default="charts")
    p.add_argument("--news_upcoming_is_block", type=int, default=0)
    
    p.add_argument("--news_upcoming_min", type=int, default=1)

    return p


def _hydrate_cost_risk_engine(args: argparse.Namespace):
    active_preset = None
    preset_name = getattr(args, "preset", None)
    if preset_name and preset_name in PRESETS:
        try:
            active_preset = deepcopy(PRESETS[preset_name])
        except Exception:
            active_preset = dict(PRESETS[preset_name])
    # Costs
    if (args.commission is not None) or (args.slip_ticks is not None):
        if (args.fee_bps is not None) or (args.slippage_bps is not None):
            print("[INFO] commission/slip_ticks provided → using contract mode, ignoring equity-bps.")
        cost = ContractCostConfig(
            commission_per_contract=(args.commission if args.commission is not None else CONTRACT_COST.commission_per_contract),
            slippage_ticks_per_side=(args.slip_ticks if args.slip_ticks is not None else CONTRACT_COST.slippage_ticks_per_side),
        )
    elif args.fee_bps is None and args.slippage_bps is None:
        cost = ContractCostConfig(
            commission_per_contract=CONTRACT_COST.commission_per_contract,
            slippage_ticks_per_side=CONTRACT_COST.slippage_ticks_per_side,
        )
    else:
        cost = CostConfig(
            fee_bps=(args.fee_bps if args.fee_bps is not None else COST.fee_bps),
            slippage_bps=(args.slippage_bps if args.slippage_bps is not None else COST.slippage_bps),
        )

    # Choose instrument early (also used to wire tick_size)
    inst = instrument_by_alias(args.instrument or ENGINE.instrument_alias)

    # RiskConfig
    rc = dict(RISK_DEFAULTS)
    map_keys = [
        "per_trade_stop_pct","per_trade_stop_usd","daily_loss_stop_pct","trailing_drawdown_pct","max_position",
        "proba_cut_bad","trail_profit_pct","trail_profit_activate_pct","flip_exit_buf","atr_len","atr_k",
        "breakeven_activate_rr","be_activate_pct","be_buffer_pct","max_bars_in_trade","cooldown_bars_after_stop",
    ]
    for k in map_keys:
        v = getattr(args, k, None)
        if v is not None:
            rc[k] = v

    # Daily USD (accept both names)
    daily_usd = _get(args, "daily_loss_stop_usd", "daily_stop_usd")
    if daily_usd is not None:
        rc["daily_loss_stop_usd"] = float(daily_usd)

    # Prop trailing DD
    prop_val = getattr(args, 'prop_trail_dd_usd', None)
    if prop_val is not None:
        rc["prop_trailing_dd_usd"] = float(args.prop_trail_dd_usd)
        rc["prop_enabled"] = True
    trail_dd = getattr(args, "trail_dd_usd", None)
    if trail_dd is not None and rc.get("prop_trailing_dd_usd") is None:
        rc["prop_trailing_dd_usd"] = float(args.trail_dd_usd)
        rc["prop_enabled"] = True

    # Behavior toggles
    if args.stop_after_win and not args.no_stop_after_win:
        rc["stop_after_first_win"] = True
    if args.no_stop_after_win:
        rc["stop_after_first_win"] = False
    if args.trail_intrabar:
        rc["trail_use_intrabar_extremes"] = True
    if args.prop_trail_on_unrealized:
        rc["prop_trail_on_unrealized"] = True
    if args.prop_breach_buffer_usd is not None:
        rc["prop_breach_buffer_usd"] = float(args.prop_breach_buffer_usd)

    # NEW: wire tick_size for stop rounding
    rc["tick_size"] = float(inst.tick_size)

    risk = RiskConfig(**rc)

    # allowed_grades normalization (drop empties/whitespace)
    allow_str = args.allow_grades if args.allow_grades is not None else ",".join(ENGINE.allowed_grades)
    allow_str = allow_str.strip()
    allow_str = ",".join([g for g in allow_str.split(",") if g.strip()])
    allowed_grades = tuple(g.strip() for g in allow_str.split(",")) if allow_str else tuple(ENGINE.allowed_grades)

    engine_kwargs = dict(
        # EV thresholding (only forwarded if supported downstream)
        target_r=(args.target_r if getattr(args, "target_r", None) is not None else None),
        policy_margin=float(args.policy_margin or 0.0),

        # Fallback classic thresholds
        p_buy=(args.p_buy if args.p_buy is not None else ENGINE.p_buy),
        p_sell=(args.p_sell if args.p_sell is not None else ENGINE.p_sell),
        allowed_grades=allowed_grades,
        prob_bands=PROB_BANDS,

        # Session/window/limits
        session_tz=(args.session_tz or ENGINE.session_tz),
        trade_window_start=(args.trade_window_start or ENGINE.trade_window_start),
        trade_window_end=(args.trade_window_end or ENGINE.trade_window_end),
        max_trades_per_day=(args.max_trades_per_day if args.max_trades_per_day is not None else ENGINE.max_trades_per_day),

        # Contract ledger
        initial_equity=float(args.account_scale_usd if args.account_scale_usd is not None else ENGINE.account_scale_usd),
        account_scale_usd=float(args.account_scale_usd if args.account_scale_usd is not None else ENGINE.account_scale_usd),

        # DD circuit
        enable_dd_circuit=ENGINE.enable_dd_circuit,
        dd_limit=ENGINE.dd_limit,
        dd_resume_hysteresis=ENGINE.dd_resume_hysteresis,
        dd_disable_from_next_bar=ENGINE.dd_disable_from_next_bar,

        # Vol targeting
        enable_vol_target=(args.vol_target or ENGINE.enable_vol_target),
        target_vol=(args.target_vol if args.target_vol is not None else ENGINE.target_vol),
        vol_ema_span=(args.vol_ema_span if args.vol_ema_span is not None else ENGINE.vol_ema_span),
        vol_annualize_k=(args.vol_annualize_k if args.vol_annualize_k is not None else ENGINE.vol_annualize_k),
        pos_cap=(args.pos_cap if args.pos_cap is not None else ENGINE.pos_cap),

        # LLM telemetry
        use_llm=(args.use_llm or ENGINE.use_llm),
        llm_review_all=ENGINE.llm_review_all,
        llm_max_risk_bps=(args.llm_risk_bps if args.llm_risk_bps is not None else ENGINE.llm_max_risk_bps),
        llm_cooldown_min=(args.llm_cooldown_min if args.llm_cooldown_min is not None else ENGINE.llm_cooldown_min),
        symbol=(args.symbol or ENGINE.symbol),

        # Instrument spec
        instrument=inst,

        # Context gates
        enable_vwap_gate=not bool(getattr(args, "no_vwap_gate", False)),
        enable_ema_gate=not bool(getattr(args, "no_ema_gate", False)),

        # EOD profit lock
        profit_lock_usd=(args.profit_lock_usd if args.profit_lock_usd is not None else None),
        near_breach_buffer_usd=float(args.near_breach_buffer_usd or 100.0),
    )

    # Optional sanity warning when user overrides thresholds oddly
    if args.p_buy is not None and args.p_sell is not None and args.p_buy <= args.p_sell:
        print(f"[WARN] p_buy ({args.p_buy}) <= p_sell ({args.p_sell}); strategy may be mostly flat.")

    feature_ctx = build_feature_context(preset=active_preset, name=preset_name)
    engine_kwargs["feature_context"] = feature_ctx
    engine_kwargs["preset_config"] = active_preset
    return cost, risk, engine_kwargs, feature_ctx, active_preset


def _plot_equity_curve(df_bt: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,4))
        plt.plot(pd.to_datetime(df_bt["Datetime"]), df_bt["equity"])  # no explicit styles/colors
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Equity (USD)")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed to save equity curve plot: {e}")


def _build_news_mask_if_any(args: argparse.Namespace, feats: pd.DataFrame, session_tz: str) -> pd.Series | None:
    if args.news_csv is None:
        return None
    if NewsRiskFilter is None:
        print("[WARN] news_csv provided but NewsRiskFilter not available; continuing.")
        return None

    impacts = [s.strip().upper() for s in str(args.news_impacts).split(",") if s.strip()]
    currencies = [s.strip().upper() for s in str(args.news_currencies).split(",") if s.strip()]
    nf = NewsRiskFilter(
        tz=session_tz,
        impacts=impacts,
        currencies=currencies,
        ttl_sec=999999,
        treat_upcoming_as_block=bool(int(args.news_upcoming_is_block)),
        upcoming_lookahead_min=int(args.news_upcoming_min),
        preset=active_preset,
    )
    try:
        nf.refresh(csv_path=args.news_csv, source_tz=args.news_source_tz)
    except Exception as e:
        print(f"[WARN] NewsRiskFilter refresh failed: {e}; continuing without news mask.")
        return None

    # Align to features timeline (feats.Datetime is naive UTC in our stack)
    idx = pd.DatetimeIndex(pd.to_datetime(feats["Datetime"]))
    mask = nf.mask_for_index(idx)
    return mask


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    # Apply preset BEFORE hydration
    if args.preset:
        keymap = {
            "max_trades": "max_trades_per_day",
            "per_stop_pct": "per_trade_stop_pct",
            "per_stop_usd": "per_trade_stop_usd",
            "daily_stop_pct": "daily_loss_stop_pct",
            "trail_dd_pct": "trailing_drawdown_pct",
            # USD aliases
            "daily_stop_usd": "daily_loss_stop_usd",
            "trail_dd_usd": "trail_dd_usd",
            "prop_trail_dd_usd": "prop_trail_dd_usd",
            "stop_after_win": "stop_after_win",
            "trail_intrabar": "trail_intrabar",
            # Profit lock
            "profit_lock_usd": "profit_lock_usd",
            "near_breach_buffer_usd": "near_breach_buffer_usd",
            # Costs / session / grades / thresholds
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
            # EV
            "target_r": "target_r",
            "policy_margin": "policy_margin",
        }
        preset = PRESETS[args.preset]
        for k, v in preset.items():
            dest = keymap.get(k, k)
            if hasattr(args, dest) and getattr(args, dest) is None and dest not in ("vol_target",):
                setattr(args, dest, v)
        # bool flags
        if "vol_target" in preset and not args.vol_target:
            args.vol_target = bool(preset["vol_target"])
        if "stop_after_win" in preset and not args.stop_after_win:
            args.stop_after_win = bool(preset["stop_after_win"])
        if "trail_intrabar" in preset and not args.trail_intrabar:
            args.trail_intrabar = bool(preset["trail_intrabar"])

    # Prefer contract mode if commission/slip provided
    if args.commission is not None or args.slip_ticks is not None:
        args.fee_bps = None
        args.slippage_bps = None

    model_path = resolve_model_path(args.model)
    cost, risk, engine_kwargs, feature_ctx, active_preset = _hydrate_cost_risk_engine(args)

    # Load data & build features for inference
    df_raw = pd.read_csv(Path(args.csv_path))
    df_raw = _normalize_ohlcv_columns(df_raw)

    feats = build_features(
        df_raw,
        tz=args.tz,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        orb_minutes=int(args.orb_min),
    )
    if "Datetime" not in feats.columns:
        raise ValueError("build_features must include a 'Datetime' column")
    feats["Datetime"] = pd.to_datetime(feats["Datetime"])  # ensure datetime type
    feats = feats.sort_values("Datetime").reset_index(drop=True)

    # Load model and align features
    model, feature_names = _load_model_and_feature_list(Path(model_path))
    X_all, used_feats = _align_X_for_model(feats.set_index("Datetime"), feature_names)

    # Predict probabilities aligned to feats index
    p = _predict_proba_safely(model, X_all)
    if len(p) != len(feats):
        raise RuntimeError(f"Predicted proba length {len(p)} != features length {len(feats)}")

    # Optional news risk mask (soft-fail)
    news_mask = _build_news_mask_if_any(args, feats, session_tz=engine_kwargs["session_tz"]) if hasattr(args, "news_csv") else None

    # Build call kwargs accepting only supported params
    base_kwargs = dict(
        df=feats,
        proba=p,
        cost=cost,
        risk=risk,
        feature_context=feature_ctx,
        preset_config=engine_kwargs.get("preset_config"),
        close_col="Close",
        datetime_col="Datetime",

        # Session/window/limits
        session_tz=engine_kwargs["session_tz"],
        trade_window_start=engine_kwargs["trade_window_start"],
        trade_window_end=engine_kwargs["trade_window_end"],
        max_trades_per_day=engine_kwargs["max_trades_per_day"],

        # Grading & probs
        allowed_grades=engine_kwargs["allowed_grades"],
        prob_bands=engine_kwargs["prob_bands"],

        # Drawdown circuit
        enable_dd_circuit=engine_kwargs["enable_dd_circuit"],
        dd_limit=engine_kwargs["dd_limit"],
        dd_resume_hysteresis=engine_kwargs["dd_resume_hysteresis"],
        dd_disable_from_next_bar=engine_kwargs["dd_disable_from_next_bar"],

        # Vol targeting
        enable_vol_target=engine_kwargs["enable_vol_target"],
        target_vol=engine_kwargs["target_vol"],
        vol_ema_span=engine_kwargs["vol_ema_span"],
        vol_annualize_k=engine_kwargs["vol_annualize_k"],
        pos_cap=engine_kwargs["pos_cap"],

        # LLM telemetry
        use_llm=engine_kwargs["use_llm"],
        llm_review_all=engine_kwargs["llm_review_all"],
        llm_max_risk_bps=engine_kwargs["llm_max_risk_bps"],
        llm_cooldown_min=engine_kwargs["llm_cooldown_min"],
        symbol=engine_kwargs["symbol"],

        # Instrument & account scale
        instrument=engine_kwargs["instrument"],
        account_scale_usd=engine_kwargs["account_scale_usd"],

        # Prop / EOD trailing extras
        profit_lock_usd=engine_kwargs.get("profit_lock_usd"),
        near_breach_buffer_usd=engine_kwargs.get("near_breach_buffer_usd", 100.0),

        # EV thresholds & context gates
        target_r=engine_kwargs.get("target_r"),
        policy_margin=engine_kwargs.get("policy_margin", 0.0),
        enable_vwap_gate=engine_kwargs.get("enable_vwap_gate", True),
        enable_ema_gate=engine_kwargs.get("enable_ema_gate", True),

        # News risk
        news_block_mask=news_mask,
    )
    sig = inspect.signature(backtest_threshold_futures)
    accepted = set(sig.parameters.keys())
    call_kwargs = {k: v for k, v in base_kwargs.items() if k in accepted}

    df_bt, summary = backtest_threshold_futures(**call_kwargs)

    # Outputs
    out_csv = Path(args.out_csv)
    out_trades = Path(args.out_trades)
    out_plot = Path(args.out_plot)
    charts_dir = Path(args.charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    df_bt.to_csv(out_csv, index=False)

    # Trades CSV
    trades_df = _make_trades_csv(df_bt, close_col="Close")
    trades_df.to_csv(out_trades, index=False)

    # Equity curve plot
    _plot_equity_curve(df_bt, out_plot)

    # Persist summary
    Path("runs").mkdir(exist_ok=True)
    Path(PurePath("runs") / "last_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Print concise summary to stdout
    keys = [
        "instrument","initial_equity","final_equity","total_pnl_usd","trades","max_drawdown",
        "ev_thresh_long","ev_thresh_short","target_r","policy_margin","prob_cut_exits",
        "trail_profit_exits","stopped_be_exits","stopped_atr_exits"
    ]
    print(json.dumps({k: summary[k] for k in keys if k in summary}, indent=2))


if __name__ == "__main__":
    main()
