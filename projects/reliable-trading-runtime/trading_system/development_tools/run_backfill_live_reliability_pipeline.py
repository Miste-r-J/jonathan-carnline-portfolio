import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_backfill_reliability import evaluate_backfill_reliability


def _parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for item in str(raw or "").split(","):
        txt = item.strip()
        if not txt:
            continue
        out.append(float(txt))
    return out


def _parse_run_dirs(raw: str) -> List[Path]:
    text = str(raw or "").strip()
    if not text:
        return []
    if any(ch in text for ch in ["*", "?"]):
        return sorted(Path(".").resolve().glob(text))
    return [Path(chunk.strip()).expanduser().resolve() for chunk in text.split(",") if chunk.strip()]


def _threshold_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(row.get("setup_threshold") or 0.0),
        float(row.get("direction_long_threshold") or 0.0),
        float(row.get("direction_short_threshold") or 0.0),
    )


def _aggregate_candidates(
    reports: Iterable[Dict[str, Any]],
    *,
    min_pass_rate: float,
    max_daily_loss_cap: float,
) -> Dict[str, Any]:
    by_key: Dict[Tuple[float, float, float], List[Dict[str, Any]]] = defaultdict(list)
    for report in reports:
        sweep = report.get("threshold_sweep") or {}
        for row in list(sweep.get("all_candidates") or []):
            by_key[_threshold_key(row)].append(row)

    aggregated: List[Dict[str, Any]] = []
    for key, rows in by_key.items():
        pass_rates = [float(r.get("pass_rate") or 0.0) for r in rows]
        net = [float(r.get("net_pnl_usd") or 0.0) for r in rows]
        drawdowns = [float(r.get("max_drawdown_usd") or 0.0) for r in rows]
        worst_days = [float(r.get("worst_day_usd") or 0.0) for r in rows]
        avg_pass_rate = sum(pass_rates) / float(len(pass_rates)) if pass_rates else 0.0
        avg_net = sum(net) / float(len(net)) if net else 0.0
        worst_day = min(worst_days) if worst_days else 0.0
        max_dd = max(drawdowns) if drawdowns else 0.0
        cap_pass = worst_day >= -abs(float(max_daily_loss_cap))
        reliability_pass = avg_pass_rate >= float(min_pass_rate)
        aggregated.append(
            {
                "setup_threshold": key[0],
                "direction_long_threshold": key[1],
                "direction_short_threshold": key[2],
                "samples": len(rows),
                "avg_pass_rate": avg_pass_rate,
                "avg_net_pnl_usd": avg_net,
                "max_drawdown_usd": max_dd,
                "worst_day_usd": worst_day,
                "conservative_cap_pass": bool(cap_pass),
                "reliability_pass": bool(reliability_pass),
            }
        )

    ranked = sorted(
        aggregated,
        key=lambda r: (
            -float(r.get("avg_pass_rate") or 0.0),
            -float(r.get("avg_net_pnl_usd") or 0.0),
            float(r.get("max_drawdown_usd") or 0.0),
        ),
    )
    return {
        "candidate_count": len(ranked),
        "ranked": ranked,
    }


def _make_stage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(rows),
        "top10": rows[:10],
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage A/B/C backfill-live reliability promotion pipeline.")
    p.add_argument("--run-dirs", required=True, help="Comma-separated run directories (or glob pattern).")
    p.add_argument("--target-min", type=float, default=400.0)
    p.add_argument("--target-max", type=float, default=1200.0)
    p.add_argument("--window-days", type=int, default=20)
    p.add_argument("--cost-profile", default="current_default", choices=["current_default"])
    p.add_argument("--setup-thresholds", default="0.30,0.35,0.40")
    p.add_argument("--direction-long-thresholds", default="0.56,0.58,0.60,0.62")
    p.add_argument("--direction-short-thresholds", default="0.56,0.58,0.60,0.62")
    p.add_argument("--max-hold-bars", type=int, default=8)
    p.add_argument("--max-daily-loss-cap", type=float, default=400.0)
    p.add_argument("--min-backfill-pass-rate", type=float, default=0.70)
    p.add_argument("--min-live-pass-rate", type=float, default=0.70)
    p.add_argument("--output", default=None, help="Optional output JSON path.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_dirs = _parse_run_dirs(args.run_dirs)
    if not run_dirs:
        raise SystemExit("No run directories resolved.")

    setup_thresholds = _parse_float_list(args.setup_thresholds)
    long_thresholds = _parse_float_list(args.direction_long_thresholds)
    short_thresholds = _parse_float_list(args.direction_short_thresholds)

    backfill_reports: List[Dict[str, Any]] = []
    live_reports: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        backfill_reports.append(
            evaluate_backfill_reliability(
                run_dir,
                phase="BACKFILL",
                target_min=float(args.target_min),
                target_max=float(args.target_max),
                window_days=int(args.window_days),
                cost_profile=str(args.cost_profile),
                setup_thresholds=setup_thresholds,
                direction_long_thresholds=long_thresholds,
                direction_short_thresholds=short_thresholds,
                max_hold_bars=int(args.max_hold_bars),
                max_daily_loss_cap=float(args.max_daily_loss_cap),
            )
        )
        live_reports.append(
            evaluate_backfill_reliability(
                run_dir,
                phase="LIVE",
                target_min=float(args.target_min),
                target_max=float(args.target_max),
                window_days=int(args.window_days),
                cost_profile=str(args.cost_profile),
                setup_thresholds=setup_thresholds,
                direction_long_thresholds=long_thresholds,
                direction_short_thresholds=short_thresholds,
                max_hold_bars=int(args.max_hold_bars),
                max_daily_loss_cap=float(args.max_daily_loss_cap),
            )
        )

    backfill_agg = _aggregate_candidates(
        backfill_reports,
        min_pass_rate=float(args.min_backfill_pass_rate),
        max_daily_loss_cap=float(args.max_daily_loss_cap),
    )
    live_agg = _aggregate_candidates(
        live_reports,
        min_pass_rate=float(args.min_live_pass_rate),
        max_daily_loss_cap=float(args.max_daily_loss_cap),
    )

    live_map = {
        (
            float(row.get("setup_threshold") or 0.0),
            float(row.get("direction_long_threshold") or 0.0),
            float(row.get("direction_short_threshold") or 0.0),
        ): row
        for row in list(live_agg.get("ranked") or [])
    }

    stage_a = [
        row
        for row in list(backfill_agg.get("ranked") or [])
        if bool(row.get("reliability_pass")) and bool(row.get("conservative_cap_pass"))
    ]

    stage_b: List[Dict[str, Any]] = []
    for row in stage_a:
        key = (
            float(row.get("setup_threshold") or 0.0),
            float(row.get("direction_long_threshold") or 0.0),
            float(row.get("direction_short_threshold") or 0.0),
        )
        live_row = live_map.get(key)
        if not live_row:
            continue
        if not (bool(live_row.get("reliability_pass")) and bool(live_row.get("conservative_cap_pass"))):
            continue
        stage_b.append(
            {
                **row,
                "live_avg_pass_rate": float(live_row.get("avg_pass_rate") or 0.0),
                "live_avg_net_pnl_usd": float(live_row.get("avg_net_pnl_usd") or 0.0),
                "live_max_drawdown_usd": float(live_row.get("max_drawdown_usd") or 0.0),
                "live_worst_day_usd": float(live_row.get("worst_day_usd") or 0.0),
            }
        )

    stage_c_ranked = sorted(
        stage_b,
        key=lambda r: (
            -(float(r.get("avg_pass_rate") or 0.0) + float(r.get("live_avg_pass_rate") or 0.0)) / 2.0,
            -(float(r.get("avg_net_pnl_usd") or 0.0) + float(r.get("live_avg_net_pnl_usd") or 0.0)) / 2.0,
            max(float(r.get("max_drawdown_usd") or 0.0), float(r.get("live_max_drawdown_usd") or 0.0)),
        ),
    )

    chosen = stage_c_ranked[0] if stage_c_ranked else None

    payload = {
        "input_runs": [str(p) for p in run_dirs],
        "params": {
            "target_min": float(args.target_min),
            "target_max": float(args.target_max),
            "window_days": int(args.window_days),
            "cost_profile": str(args.cost_profile),
            "max_daily_loss_cap": float(args.max_daily_loss_cap),
            "min_backfill_pass_rate": float(args.min_backfill_pass_rate),
            "min_live_pass_rate": float(args.min_live_pass_rate),
            "setup_thresholds": setup_thresholds,
            "direction_long_thresholds": long_thresholds,
            "direction_short_thresholds": short_thresholds,
            "max_hold_bars": int(args.max_hold_bars),
        },
        "stage_a_backfill": _make_stage(stage_a),
        "stage_b_live_shadow": _make_stage(stage_b),
        "stage_c_promotion": {
            "count": len(stage_c_ranked),
            "top10": stage_c_ranked[:10],
            "chosen": chosen,
        },
        "backfill_aggregate": {"candidate_count": backfill_agg.get("candidate_count"), "top10": list(backfill_agg.get("ranked") or [])[:10]},
        "live_aggregate": {"candidate_count": live_agg.get("candidate_count"), "top10": list(live_agg.get("ranked") or [])[:10]},
    }

    output = Path(args.output).expanduser().resolve() if args.output else (Path.cwd() / "reliability_promotion_report.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
