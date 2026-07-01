from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.runtime_engine.modeling.prop_scoring import PropScoringConfig, evaluate_prop_candidate
from run_sharp import run_candidate  # type: ignore


SLIPPAGE_LEVELS = (0.0, 1.0, 2.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a Phase-2 candidate with aggressive prop-style gates.")
    p.add_argument("--tag", help="Candidate tag under artifacts/phase2/candidates.")
    p.add_argument("--manifest", help="Explicit manifest path.")
    p.add_argument("--csv", default=str(REPO_ROOT / "data" / "intraday" / "es" / "ES6.csv"))
    p.add_argument("--instrument", default="ES")
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--artifact-root", default=str(ROOT / "artifacts" / "phase2" / "candidates"))
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    p.add_argument("--test-start", default="2025-11-01")
    p.add_argument("--test-end", default="2026-01-16")
    p.add_argument("--trade-window-start", default="00:00")
    p.add_argument("--trade-window-end", default="23:59")
    p.add_argument("--max-hold-bars", type=int, default=24)
    p.add_argument("--commission-per-contract", type=float, default=2.0)
    p.add_argument("--p_setup", type=float, default=None, help="Override manifest setup threshold.")
    p.add_argument("--p_long", type=float, default=None, help="Override manifest long threshold.")
    p.add_argument("--p_short", type=float, default=None, help="Override manifest short threshold.")
    p.add_argument("--max-daily-loss", type=float, default=900.0)
    p.add_argument("--max-drawdown", type=float, default=5000.0)
    p.add_argument("--max-trades-per-day", type=int, default=10)
    p.add_argument("--min-trades-per-day", type=float, default=4.0)
    p.add_argument("--max-consecutive-losses", type=int, default=3)
    p.add_argument("--max-flip-rate", type=float, default=0.25)
    p.add_argument("--preferred-flip-rate", type=float, default=0.20)
    p.add_argument("--max-bad-fade-rate", type=float, default=0.18)
    p.add_argument("--min-profit-factor", type=float, default=2.0)
    p.add_argument("--target-profit-factor", type=float, default=3.0)
    p.add_argument("--min-trade-count", type=int, default=50)
    return p.parse_args()


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest:
        return Path(args.manifest).expanduser().resolve()
    if not args.tag:
        raise SystemExit("Provide --tag or --manifest.")
    return Path(args.artifact_root).expanduser().resolve() / args.tag / "manifest.json"


def _read_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Manifest is not a JSON object: {path}")
    return payload


def _sharp_args(args: argparse.Namespace, manifest_path: Path, slippage: float) -> SimpleNamespace:
    return SimpleNamespace(
        tag=None,
        manifest=str(manifest_path),
        csv=args.csv,
        instrument=args.instrument,
        contracts=args.contracts,
        trade_window_start=args.trade_window_start,
        trade_window_end=args.trade_window_end,
        max_hold_bars=args.max_hold_bars,
        out_dir=str(ROOT / "runs" / "prop_scoring" / "_sharp"),
        p_setup=args.p_setup,
        p_long=args.p_long,
        p_short=args.p_short,
        start_at=args.test_start,
        end_at=args.test_end,
        commission_per_contract=args.commission_per_contract,
        slippage_ticks=slippage,
        skip_store=True,
    )


def _slippage_summary(sim: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "total_pnl_usd": float(sim.get("total_pnl_usd") or 0.0),
        "profit_factor": float(sim.get("profit_factor") or 0.0),
        "max_drawdown": float(sim.get("max_drawdown") or 0.0),
        "sharpe": float(sim.get("sharpe") or 0.0),
        "trade_count": int(sim.get("trade_count") or len(sim.get("trades") or [])),
        "win_rate": float(sim.get("win_rate") or 0.0),
        "flip_rate_per_day": float(sim.get("flip_rate_per_day") or 0.0),
        "trades_per_day": float(sim.get("trades_per_day") or 0.0),
        "avg_trade_usd": float(sim.get("avg_trade_usd") or 0.0),
    }


def main() -> int:
    args = _parse_args()
    manifest_path = _manifest_path(args)
    manifest = _read_manifest(manifest_path)
    slippage: Dict[str, Dict[str, Any]] = {}
    trades = []
    for level in SLIPPAGE_LEVELS:
        result = run_candidate(_sharp_args(args, manifest_path, level))
        sim = result.get("sim") or {}
        slippage[f"slip_{level:g}"] = _slippage_summary(sim)
        if abs(level - 1.0) < 1e-9:
            trades = list(sim.get("trades") or [])

    behavior = manifest.get("behavior_audit_test") or {}
    bad_fade_rate = float(behavior.get("countertrend_rate") or 0.0) if isinstance(behavior, Mapping) else 0.0
    config = PropScoringConfig(
        max_daily_loss=args.max_daily_loss,
        max_drawdown=args.max_drawdown,
        max_trades_per_day=args.max_trades_per_day,
        min_trades_per_day=args.min_trades_per_day,
        max_consecutive_losses=args.max_consecutive_losses,
        max_flip_rate_per_day=args.max_flip_rate,
        preferred_flip_rate_per_day=args.preferred_flip_rate,
        max_bad_fade_rate=args.max_bad_fade_rate,
        min_profit_factor=args.min_profit_factor,
        target_profit_factor=args.target_profit_factor,
        min_trade_count=args.min_trade_count,
    )
    prop = evaluate_prop_candidate(
        slip1=slippage.get("slip_1") or {},
        slippage=slippage,
        trades=trades,
        bad_fade_rate=bad_fade_rate,
        config=config,
    )
    payload = {
        "tag": manifest.get("tag") or args.tag or manifest_path.parent.name,
        "manifest": str(manifest_path),
        "rejected": bool(manifest.get("rejected")),
        "thresholds": manifest.get("thresholds") or {},
        "threshold_overrides": {
            "p_setup": args.p_setup,
            "p_long": args.p_long,
            "p_short": args.p_short,
        },
        "bad_fade_rate": bad_fade_rate,
        "slippage": slippage,
        "prop_evaluation": prop,
    }
    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
