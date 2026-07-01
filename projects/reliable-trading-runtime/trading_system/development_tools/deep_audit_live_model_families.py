from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def count_trades(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual = sum(
        bool((r.get("actual_entry_price") or r.get("entry_fill_price")) and (r.get("actual_exit_price") or r.get("exit_fill_price")))
        for r in rows
    )
    return len(rows), actual


def inventory(runs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for run in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        status = load_json(run / "status.json")
        config = load_json(run / "resolved_config.json")
        manifest = load_json(run / "run_manifest.json")
        health = load_json(run / "run_health_summary.json")
        stats = health.get("executor_stats_all_phases") or status.get("executor_stats_all_phases") or {}
        trade_rows, actual_rows = count_trades(run / "trades.csv")
        rows.append(
            {
                "run": run.name,
                "tag": config.get("phase2_tag") or "<missing>",
                "preset": manifest.get("preset") or status.get("preset") or config.get("preset") or "<missing>",
                "config_hash": manifest.get("config_hash") or status.get("config_hash") or "<missing>",
                "account": status.get("chosen_account") or status.get("detected_account") or status.get("snapshot_account") or "<missing>",
                "nt_exec_policy": manifest.get("nt_exec_policy") or status.get("nt_exec_policy") or "<missing>",
                "force_open": (config.get("phase2_force_open_policy") or {}).get("enabled"),
                "setup_fail_entries": (config.get("phase2_force_open_policy") or {}).get("allow_setup_fail_entries"),
                "trade_rows": trade_rows,
                "actual_closed_rows": actual_rows,
                "candidate_signals": int(stats.get("candidate_signals_total") or 0),
                "model_emits": int(stats.get("model_emits_total") or 0),
                "nt_entries": int(stats.get("nt_order_entry_total") or 0),
                "health_verdict": health.get("verdict") or "<missing>",
            }
        )
    families = []
    for tag in sorted({row["tag"] for row in rows}):
        group = [row for row in rows if row["tag"] == tag]
        families.append(
            {
                "tag": tag,
                "runs": len(group),
                "presets": dict(Counter(row["preset"] for row in group)),
                "accounts": dict(Counter(row["account"] for row in group)),
                "policies": dict(Counter(row["nt_exec_policy"] for row in group)),
                "config_hashes": dict(Counter(row["config_hash"] for row in group)),
                "force_open": dict(Counter(str(row["force_open"]) for row in group)),
                "setup_fail_entries": dict(Counter(str(row["setup_fail_entries"]) for row in group)),
                "trade_rows": sum(row["trade_rows"] for row in group),
                "actual_closed_rows": sum(row["actual_closed_rows"] for row in group),
                "candidate_signals": sum(row["candidate_signals"] for row in group),
                "model_emits": sum(row["model_emits"] for row in group),
                "nt_entries": sum(row["nt_entries"] for row in group),
            }
        )
    return rows, families


def replay_trades(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("exit_price"):
                    continue
                entry, exit_ = float(row["entry_price"]), float(row["exit_price"])
                gross = (exit_ - entry if row.get("side") == "LONG" else entry - exit_) * 50.0 * float(row.get("qty") or 1)
                result.append({"entry_ts": row.get("entry_ts"), "side": row.get("side"), "gross": gross})
    return sorted(result, key=lambda row: str(row["entry_ts"]))


def metrics(values: list[float]) -> dict[str, Any]:
    equity = peak = drawdown = 0.0
    wins = losses = 0.0
    streak = max_streak = 0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
        if value > 0:
            wins += value
            streak = 0
        else:
            losses -= value
            streak += 1
            max_streak = max(max_streak, streak)
    return {
        "trades": len(values), "net_pnl_usd": equity, "profit_factor": wins / losses if losses else None,
        "max_drawdown_usd": drawdown, "max_loss_streak": max_streak,
    }


def stress(trades: list[dict[str, Any]]) -> dict[str, Any]:
    friction = {str(cost): metrics([row["gross"] - cost for row in trades]) for cost in (4, 29, 54, 79, 104)}
    base = [row["gross"] - 29 for row in trades]
    rng = random.Random(20260622)
    nets, drawdowns = [], []
    for _ in range(10_000):
        sample = [rng.choice(base) for _ in base]
        result = metrics(sample)
        nets.append(result["net_pnl_usd"])
        drawdowns.append(result["max_drawdown_usd"])
    nets.sort(); drawdowns.sort()
    return {
        "friction_per_round_trip_usd": friction,
        "bootstrap_10000": {
            "net_p05_usd": nets[499], "net_median_usd": nets[4999], "net_p95_usd": nets[9499],
            "probability_net_nonpositive": sum(value <= 0 for value in nets) / len(nets),
            "drawdown_p95_usd": drawdowns[9499],
            "probability_drawdown_at_least_2000": sum(value >= 2000 for value in drawdowns) / len(drawdowns),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--mff-fills", type=Path, required=True)
    parser.add_argument("--replay-trades", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    runs, families = inventory(args.runs_root)
    with args.mff_fills.open("r", encoding="utf-8-sig", newline="") as handle:
        fills = [row for row in csv.DictReader(handle) if str(row.get("closed")).lower() == "true"]
    gross = [float(row["pnl_usd"]) for row in fills]
    actual = metrics([value - 4.0 for value in gross])
    candidate = replay_trades(args.replay_trades)
    payload = {
        "schema_version": 1,
        "runs_root": str(args.runs_root.resolve()),
        "run_count": len(runs),
        "model_family_count": len(families),
        "families": families,
        "actual_mff_content_addressed": {**actual, "gross_pnl_usd": sum(gross), "cost_assumption_usd_per_round_trip": 4.0},
        "final_candidate_same_live_sessions": {**metrics([row["gross"] - 29 for row in candidate]), "gross_pnl_usd": sum(row["gross"] for row in candidate), "cost_assumption_usd_per_round_trip": 29.0},
        "final_candidate_stress": stress(candidate),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "deep_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out_dir / "run_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader(); writer.writerows(runs)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
