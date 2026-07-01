from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the ES 5m live-reliability experiment matrix.")
    p.add_argument("--csv", default=str(REPO_ROOT / "data" / "intraday" / "es" / "ES6.csv"))
    p.add_argument("--artifact-root", default=str(ROOT / "artifacts" / "phase2" / "candidates"))
    p.add_argument("--summary-dir", default=str(ROOT / "runs" / "es_reliability_matrix"))
    p.add_argument("--tag-prefix", default="es_reliability")
    p.add_argument("--train-start", default="2021-01-14")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--val-start", default="2025-06-01")
    p.add_argument("--val-end", default="2025-10-31")
    p.add_argument("--test-start", default="2025-11-01")
    p.add_argument("--test-end", default="2026-01-16")
    p.add_argument("--walkforward-windows", type=int, default=3)
    p.add_argument("--live-shadow-summary-path", default="run_health_summary.json")
    p.add_argument("--require-safe-shadow-pass", dest="require_safe_shadow_pass", action="store_true", default=True)
    p.add_argument("--allow-unsafe-shadow-pass", dest="require_safe_shadow_pass", action="store_false")
    p.add_argument("--execute", action="store_true", help="Run training commands instead of only writing the matrix plan.")
    return p.parse_args()


def build_experiment_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return [
        {
            "name": "baseline_5m",
            "tag": f"{args.tag_prefix}_baseline_5m",
            "horizon": 8,
            "mtf_timeframes": "",
            "threshold_objective": "sharpe",
            "setup_threshold_multiplier": 1.75,
            "max_flip_rate": 0.20,
            "min_trades_val": 30,
        },
        {
            "name": "mtf_15m_context",
            "tag": f"{args.tag_prefix}_mtf_15m",
            "horizon": 8,
            "mtf_timeframes": "15m",
            "threshold_objective": "sharpe",
            "setup_threshold_multiplier": 1.75,
            "max_flip_rate": 0.20,
            "min_trades_val": 30,
        },
        {
            "name": "mtf_15m_60m_context",
            "tag": f"{args.tag_prefix}_mtf_15m_60m",
            "horizon": 8,
            "mtf_timeframes": "15m,60m",
            "threshold_objective": "sharpe",
            "setup_threshold_multiplier": 1.75,
            "max_flip_rate": 0.20,
            "min_trades_val": 30,
        },
        {
            "name": "cleaner_trade_h12",
            "tag": f"{args.tag_prefix}_mtf_15m_60m_h12",
            "horizon": 12,
            "mtf_timeframes": "15m,60m",
            "threshold_objective": "calmar",
            "setup_threshold_multiplier": 2.10,
            "max_flip_rate": 0.12,
            "min_trades_val": 20,
        },
    ]


def _train_cmd(args: argparse.Namespace, spec: Dict[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "train_phase2_grid.py"),
        "--csv",
        str(args.csv),
        "--instrument",
        "ES",
        "--tag",
        str(spec["tag"]),
        "--artifact-root",
        str(args.artifact_root),
        "--train-start",
        str(args.train_start),
        "--train-end",
        str(args.train_end),
        "--val-start",
        str(args.val_start),
        "--val-end",
        str(args.val_end),
        "--test-start",
        str(args.test_start),
        "--test-end",
        str(args.test_end),
        "--horizon",
        str(spec["horizon"]),
        "--label-mode",
        "horizon",
        "--label-threshold",
        "0.0015",
        "--threshold-objective",
        str(spec["threshold_objective"]),
        "--min-trades-val",
        str(spec["min_trades_val"]),
        "--max-flip-rate",
        str(spec["max_flip_rate"]),
        "--setup-threshold-multiplier",
        str(spec["setup_threshold_multiplier"]),
        "--walkforward-windows",
        str(args.walkforward_windows),
        "--live-shadow-summary-path",
        str(args.live_shadow_summary_path),
    ]
    if spec.get("mtf_timeframes"):
        cmd.extend(["--mtf-timeframes", str(spec["mtf_timeframes"])])
    if args.require_safe_shadow_pass:
        cmd.append("--require-safe-shadow-pass")
    return cmd


def _run(cmd: List[str], *, execute: bool) -> None:
    print("\n$ " + " ".join(f'"{item}"' if " " in item else item for item in cmd))
    if not execute:
        return
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)


def _manifest_summary(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {"manifest_path": str(manifest_path), "exists": False}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "manifest_path": str(manifest_path),
        "exists": True,
        "tag": payload.get("tag"),
        "rejected": bool(payload.get("rejected")),
        "rejected_reason": payload.get("rejected_reason"),
        "promotion_blocked": bool(payload.get("promotion_blocked")),
        "promotion_blocked_reason": payload.get("promotion_blocked_reason"),
        "trading_val": payload.get("trading_val"),
        "trading_test": payload.get("trading_test"),
        "walkforward": payload.get("walkforward"),
        "live_shadow_gate": payload.get("live_shadow_gate"),
        "config": payload.get("config"),
    }


def main() -> None:
    args = _parse_args()
    summary_dir = Path(args.summary_dir).expanduser().resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)

    specs = build_experiment_specs(args)
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        cmd = _train_cmd(args, spec)
        _run(cmd, execute=args.execute)
        manifest_path = Path(args.artifact_root).expanduser().resolve() / str(spec["tag"]) / "manifest.json"
        rows.append({**spec, **_manifest_summary(manifest_path)})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv": str(args.csv),
        "walkforward_windows": int(args.walkforward_windows),
        "live_shadow_summary_path": str(args.live_shadow_summary_path),
        "require_safe_shadow_pass": bool(args.require_safe_shadow_pass),
        "rows": rows,
    }
    (summary_dir / "es_reliability_matrix.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
