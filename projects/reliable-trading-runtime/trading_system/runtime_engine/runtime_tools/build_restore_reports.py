from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _count_actions(state_csv: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not state_csv.exists():
        return counts
    with state_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            action = str(row.get("action") or row.get("type") or "").upper()
            counts[action] = counts.get(action, 0) + 1
    return counts


def _trade_rows(trades_csv: Path) -> int:
    if not trades_csv.exists():
        return 0
    with trades_csv.open("r", encoding="utf-8", newline="") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def build_gap_analysis(old_run: Path, new_run: Path, out_md: Path) -> None:
    old_cfg = _read_json(old_run / "resolved_config.json")
    new_cfg = _read_json(new_run / "resolved_config.json")
    old_counts = _count_actions(old_run / "state.csv")
    new_counts = _count_actions(new_run / "state.csv")
    old_health = _read_json(old_run / "run_health_summary.json")
    new_health = _read_json(new_run / "run_health_summary.json")

    lines = [
        "# restore_gap_analysis",
        "",
        "## Baseline",
        f"- old_run: `{old_run}`",
        f"- new_run: `{new_run}`",
        "",
        "## What old behavior is missing",
        f"- OPEN/FLIP emission delta: old OPEN={old_counts.get('OPEN',0)} FLIP={old_counts.get('FLIP',0)} vs new OPEN={new_counts.get('OPEN',0)} FLIP={new_counts.get('FLIP',0)}.",
        f"- Backfill gating-open emits: old={((old_health.get('executor_stats_all_phases') or {}).get('gating_open_emits_total'))} vs new={((new_health.get('executor_stats_all_phases') or {}).get('gating_open_emits_total'))}.",
        f"- New run has additional hard-safety blocks (`stop_not_finite`/`target_not_finite`) in BACKFILL phase ({((new_health.get('executor_stats_by_phase') or {}).get('BACKFILL') or {}).get('blocked_reasons')}).",
        "",
        "## What code/config changed it",
        f"- `entry_gates.gate_mode`: old=`{((old_cfg.get('entry_gates') or {}).get('gate_mode'))}` vs new=`{((new_cfg.get('entry_gates') or {}).get('gate_mode'))}`.",
        f"- runner/shelf policy drift: old runner suppress target=`{((old_cfg.get('trade_management_overlay') or {}).get('pnl_runner_suppress_target'))}`, new=`{((new_cfg.get('trade_management_overlay') or {}).get('pnl_runner_suppress_target'))}`; old shelf arm mode=`{((old_cfg.get('trade_management_overlay') or {}).get('pnl_shelf_arm_mode'))}`, new=`{((new_cfg.get('trade_management_overlay') or {}).get('pnl_shelf_arm_mode'))}`.",
        f"- guardrail drift: reconciliation_grace_bars old=`{((old_cfg.get('guardrails') or {}).get('reconciliation_grace_bars'))}` new=`{((new_cfg.get('guardrails') or {}).get('reconciliation_grace_bars'))}`, cancel_loop_max_updates old=`{((old_cfg.get('guardrails') or {}).get('cancel_loop_max_updates'))}` new=`{((new_cfg.get('guardrails') or {}).get('cancel_loop_max_updates'))}`.",
        "- directional bridge policy was strict-setup in both artifacts; restored preset introduces explicit `directional_bridge_restored` with full geometry/final-send-guard constraints.",
        "",
        "## Safety classification",
        "- safe_to_restore: legacy directional-emission personality, setup-fail strong-direction bridge (with modern geometry + final-send-guard + lifecycle required).",
        "- restore_with_guardrails: FLIP frequency/behavior only when close->reverse lifecycle confirms; no optimistic state mutation.",
        "- do_not_restore: planned-only execution truth, shelf force-exit, any bypass that mutates position without fill/snapshot truth.",
        "",
        "## Safe restoration actions",
        "- Add `es_elite_restored_v1` and explicit `directional_bridge_restored` policy.",
        "- Emit `RESTORED_DIRECTIONAL_BRIDGE_ENTRY/BLOCKED` and `RESTORED_FLIP_*` audit events for traceability.",
        "- Persist execution labeling fields (`execution_type`, `is_planned_only`, `is_executed`, `fill_confirmed`, `broker_protected_confirmed`) so simulated vs executed truth is auditable.",
        "- Enforce validator failures when planned-only is counted executed or forbidden shelf/stair force-exit markers appear.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parity_report(old_run: Path, restored_run: Path, out_md: Path) -> None:
    old_counts = _count_actions(old_run / "state.csv")
    new_counts = _count_actions(restored_run / "state.csv")
    old_trade_rows = _trade_rows(old_run / "trades.csv")
    new_trade_rows = _trade_rows(restored_run / "trades.csv")

    table = [
        "Metric | old stronger | restored_v1 | difference | explanation",
        "---|---:|---:|---:|---",
        f"OPEN count | {old_counts.get('OPEN',0)} | {new_counts.get('OPEN',0)} | {new_counts.get('OPEN',0)-old_counts.get('OPEN',0)} | state.csv action counts",
        f"CLOSE count | {old_counts.get('CLOSE',0)} | {new_counts.get('CLOSE',0)} | {new_counts.get('CLOSE',0)-old_counts.get('CLOSE',0)} | state.csv action counts",
        f"FLIP count | {old_counts.get('FLIP',0)} | {new_counts.get('FLIP',0)} | {new_counts.get('FLIP',0)-old_counts.get('FLIP',0)} | state.csv action counts",
        f"trade count | {old_trade_rows} | {new_trade_rows} | {new_trade_rows-old_trade_rows} | trades.csv populated rows",
        "net PnL | n/a from artifact | n/a from artifact | n/a | this artifact pair lacks closed-row PnL fields",
        "gross profit | n/a | n/a | n/a | not derivable from current trades.csv payload",
        "gross loss | n/a | n/a | n/a | not derivable from current trades.csv payload",
        "profit factor | n/a | n/a | n/a | not derivable from current trades.csv payload",
        "win rate | n/a | n/a | n/a | not derivable from current trades.csv payload",
        "max drawdown | n/a | n/a | n/a | not derivable from current trades.csv payload",
        "long PnL | n/a | n/a | n/a | not derivable from current trades.csv payload",
        "short PnL | n/a | n/a | n/a | not derivable from current trades.csv payload",
        "FLIP_CLOSE PnL | n/a | n/a | n/a | not derivable from current trades.csv payload",
        "setup-fail restored entries | n/a | n/a | n/a | requires RESTORED_DIRECTIONAL_BRIDGE_ENTRY events from restored run",
        "planned_only count | n/a | n/a | n/a | requires restored execution_ledger lifecycle rows",
        "executed count | n/a | n/a | n/a | requires restored execution_ledger lifecycle rows",
    ]
    out_md.write_text("# old_vs_restored_parity_report\n\n" + "\n".join(table) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build restoration gap/parity markdown reports")
    ap.add_argument("old_run", type=Path)
    ap.add_argument("new_run", type=Path)
    ap.add_argument("--gap-out", type=Path, required=True)
    ap.add_argument("--parity-out", type=Path, required=True)
    args = ap.parse_args()
    build_gap_analysis(args.old_run, args.new_run, args.gap_out)
    build_parity_report(args.old_run, args.new_run, args.parity_out)
    print(args.gap_out)
    print(args.parity_out)


if __name__ == "__main__":
    main()
