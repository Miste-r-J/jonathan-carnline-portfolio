from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FINAL_TAG = "modelrun77_prop_v3_20260622"
FINAL_PRESET = "es_modelrun77_prop_v3_paper"
EXPECTED_THRESHOLDS = {"p_setup_required": 0.20, "p_long_required": 0.66, "p_short_required": 0.75}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _closed_fill_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sum(
        bool((row.get("actual_entry_price") or row.get("entry_fill_price")) and (row.get("actual_exit_price") or row.get("exit_fill_price")))
        for row in rows
    )


def audit_run(run_dir: Path) -> dict[str, Any]:
    status = _load_json(run_dir / "status.json")
    health = _load_json(run_dir / "run_health_summary.json")
    config = _load_json(run_dir / "resolved_config.json")
    thresholds = config.get("resolved_thresholds") if isinstance(config.get("resolved_thresholds"), dict) else {}
    force_policy = config.get("phase2_force_open_policy") if isinstance(config.get("phase2_force_open_policy"), dict) else {}
    risk_limits = config.get("risk_limits") if isinstance(config.get("risk_limits"), dict) else {}
    closed_fills = _closed_fill_count(run_dir / "trades.csv")
    checks = {
        "tag": (config.get("phase2_tag") or status.get("phase2_tag")) == FINAL_TAG,
        "preset": (config.get("preset") or status.get("preset")) == FINAL_PRESET,
        "paper_policy": str(status.get("nt_exec_policy") or "").lower() == "paper",
        "manifest_thresholds": bool(config.get("phase2_manifest_thresholds_used")),
        "thresholds": all(abs(float(thresholds.get(key, -999)) - expected) < 1e-12 for key, expected in EXPECTED_THRESHOLDS.items()),
        "force_open_disabled": force_policy.get("enabled") is False,
        "setup_fail_entries_disabled": force_policy.get("allow_setup_fail_entries") is False,
        "risk_limits": (
            float(risk_limits.get("max_daily_loss_usd", -1)) == 500.0
            and float(risk_limits.get("max_risk_per_trade_usd", -1)) == 400.0
            and int(risk_limits.get("max_trades_per_day", -1)) == 6
            and int(risk_limits.get("max_losses_per_day", -1)) == 3
        ),
        "no_hard_lockout": not bool(status.get("hard_lockout_active")),
        "safe_verdict": str(health.get("verdict") or "") not in {"", "unsafe", "preflight_failed"},
    }
    return {
        "run_dir": str(run_dir.resolve()),
        "checks": checks,
        "pass": all(checks.values()),
        "closed_fill_lifecycles": closed_fills,
        "verdict": health.get("verdict"),
        "process_alive": health.get("process_alive"),
    }


def build_report(runs_root: Path, *, min_sessions: int = 2, min_lifecycles: int = 30) -> dict[str, Any]:
    candidates = sorted(
        (path for path in runs_root.glob("modelrun77_final_paper_cert_*") if path.is_dir()),
        key=lambda path: path.name,
    )
    runs = [audit_run(path) for path in candidates]
    passing = [run for run in runs if run["pass"]]
    lifecycle_total = sum(int(run["closed_fill_lifecycles"]) for run in passing)
    return {
        "schema_version": 1,
        "candidate_tag": FINAL_TAG,
        "runs_root": str(runs_root.resolve()),
        "requirements": {"minimum_passing_sessions": min_sessions, "minimum_closed_fill_lifecycles": min_lifecycles},
        "observed": {"candidate_sessions": len(runs), "passing_sessions": len(passing), "closed_fill_lifecycles": lifecycle_total},
        "certified": len(passing) >= min_sessions and lifecycle_total >= min_lifecycles,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ModelRun77 final-candidate paper certification evidence.")
    parser.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parents[2] / "runs" / "live")
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--min-lifecycles", type=int, default=30)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = build_report(args.runs_root, min_sessions=args.min_sessions, min_lifecycles=args.min_lifecycles)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
