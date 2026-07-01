from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the retrain_v6 four-pass ES 5m Phase-2 training workflow.")
    p.add_argument("--csv", default=str(REPO_ROOT / "data" / "intraday" / "es" / "ES6.csv"))
    p.add_argument("--baseline-tag", default="retrain_v4")
    p.add_argument("--artifact-root", default=str(ROOT / "artifacts" / "phase2" / "candidates"))
    p.add_argument("--train-start", default="2021-01-14")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--val-start", default="2025-01-01")
    p.add_argument("--val-end", default="2025-10-31")
    p.add_argument("--test-start", default="2025-11-01")
    p.add_argument("--test-end", default="2026-01-16")
    p.add_argument("--max-grid-candidates", type=int, default=18, help="Cap pass-2 candidates for practical iteration.")
    p.add_argument("--n-estimators", type=int, default=2000)
    p.add_argument("--execute", action="store_true", help="Actually run training commands. Default prints the plan only.")
    p.add_argument("--skip-wf", action="store_true", help="Skip walk-forward pass after grid ranking.")
    p.add_argument("--summary-dir", default=str(ROOT / "runs" / "retrain_v6"))
    return p.parse_args()


def _run(cmd: List[str], *, execute: bool, cwd: Path = ROOT) -> subprocess.CompletedProcess[str] | None:
    print("\n$ " + " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    if not execute:
        return None
    return subprocess.run(cmd, cwd=str(cwd), text=True, check=True, capture_output=True)


def _candidate_grid(limit: int) -> Iterable[Dict[str, Any]]:
    raw = itertools.product(
        ["horizon", "exec"],
        [8, 10, 12],
        [0.0012, 0.0015, 0.0018],
        ["sharpe", "calmar", "ev"],
        [False, True],
    )
    ranked: List[Dict[str, Any]] = []
    for label_mode, horizon, threshold, objective, stack in raw:
        # Prefer current proven shape first, then nearby variations.
        distance = (
            (0 if label_mode == "horizon" else 1)
            + abs(horizon - 8)
            + abs(threshold - 0.0015) * 10000
            + (0 if objective == "sharpe" else 1)
            + (1 if stack else 0)
        )
        ranked.append(
            {
                "label_mode": label_mode,
                "horizon": horizon,
                "label_threshold": threshold,
                "threshold_objective": objective,
                "stack_setup_prob": stack,
                "distance": distance,
            }
        )
    ranked.sort(key=lambda row: (row["distance"], row["label_mode"], row["horizon"], row["label_threshold"]))
    return ranked[: max(1, int(limit))]


def _train_cmd(args: argparse.Namespace, tag: str, cfg: Dict[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        "tools/train_phase2_grid.py",
        "--csv",
        str(args.csv),
        "--instrument",
        "ES",
        "--tag",
        tag,
        "--artifact-root",
        str(args.artifact_root),
        "--train-start",
        args.train_start,
        "--train-end",
        args.train_end,
        "--val-start",
        args.val_start,
        "--val-end",
        args.val_end,
        "--test-start",
        args.test_start,
        "--test-end",
        args.test_end,
        "--label-mode",
        str(cfg["label_mode"]),
        "--horizon",
        str(cfg["horizon"]),
        "--label-threshold",
        str(cfg["label_threshold"]),
        "--threshold-objective",
        str(cfg["threshold_objective"]),
        "--setup-threshold-multiplier",
        "1.75",
        "--max-bad-fade-rate",
        "0.18",
        "--recency-weighting",
        "none",
        "--commission-per-contract",
        "2.0",
        "--slippage-ticks",
        "1.0",
        "--n-estimators",
        str(args.n_estimators),
    ]
    if cfg.get("stack_setup_prob"):
        cmd.append("--stack_setup_prob")
    return cmd


def _sharp_cmd(args: argparse.Namespace, tag: str, slippage: float) -> List[str]:
    return [
        sys.executable,
        "tools/run_sharp.py",
        "--tag",
        tag,
        "--csv",
        str(args.csv),
        "--instrument",
        "ES",
        "--contracts",
        "1",
        "--trade-window-start",
        "00:00",
        "--trade-window-end",
        "23:59",
        "--max-hold-bars",
        "24",
        "--slippage_ticks",
        str(slippage),
        "--skip_store",
    ]


def _validate_cmd(args: argparse.Namespace, tag: str) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "tools" / "validate_features.py"),
        "--manifest",
        str(Path(args.artifact_root) / tag / "manifest.json"),
    ]


def _score_from_output(text: str) -> Dict[str, Any]:
    start = text.rfind("{")
    if start < 0:
        return {}
    try:
        return json.loads(text[start:])
    except Exception:
        return {}


def _evaluate(args: argparse.Namespace, tag: str) -> Dict[str, Any]:
    scores: Dict[str, Any] = {"tag": tag}
    for slip in (0.0, 1.0, 2.0):
        proc = _run(_sharp_cmd(args, tag, slip), execute=True)
        payload = _score_from_output((proc.stdout if proc else "") or "")
        scores[f"slip_{slip:g}"] = payload
    return scores


def _slip_metric(row: Dict[str, Any], metric: str, slip: str = "1") -> float:
    payload = row.get(f"slip_{slip}") or {}
    try:
        return float(payload.get(metric) or 0.0)
    except Exception:
        return 0.0


def _pick_champion(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    usable = [row for row in rows if _slip_metric(row, "trades") > 0]
    if not usable:
        return None
    return sorted(
        usable,
        key=lambda row: (
            _slip_metric(row, "sharpe"),
            _slip_metric(row, "pnl_usd"),
            -abs(_slip_metric(row, "max_dd")),
            -_slip_metric(row, "flip_rate_per_day"),
        ),
        reverse=True,
    )[0]


def _write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = _parse_args()
    summary_dir = Path(args.summary_dir)
    rows: List[Dict[str, Any]] = []

    baseline_cfg = {
        "label_mode": "horizon",
        "horizon": 8,
        "label_threshold": 0.0015,
        "threshold_objective": "sharpe",
        "stack_setup_prob": False,
    }
    pass1_tag = "retrain_v6_pass1_baseline"
    _run(_train_cmd(args, pass1_tag, baseline_cfg), execute=args.execute)
    if args.execute:
        _run(_validate_cmd(args, pass1_tag), execute=True)
        result = _evaluate(args, pass1_tag)
        result.update(baseline_cfg)
        rows.append(result)

    pass2_tags: List[str] = []
    for idx, cfg in enumerate(_candidate_grid(args.max_grid_candidates), start=1):
        tag = f"retrain_v6_pass2_grid_{idx:02d}"
        pass2_tags.append(tag)
        _run(_train_cmd(args, tag, cfg), execute=args.execute)
        if args.execute:
            _run(_validate_cmd(args, tag), execute=True)
            result = _evaluate(args, tag)
            result.update({k: v for k, v in cfg.items() if k != "distance"})
            rows.append(result)

    if args.execute:
        _write_summary(summary_dir / "pass2_scores.json", rows)
        champion = _pick_champion(rows)
        if champion:
            champion_cfg = {
                "label_mode": champion.get("label_mode", "horizon"),
                "horizon": int(champion.get("horizon", 8)),
                "label_threshold": float(champion.get("label_threshold", 0.0015)),
                "threshold_objective": champion.get("threshold_objective", "sharpe"),
                "stack_setup_prob": bool(champion.get("stack_setup_prob", False)),
            }
            champion_tag = "retrain_v6_pass4_champion"
            _run(_train_cmd(args, champion_tag, champion_cfg), execute=True)
            _run(_validate_cmd(args, champion_tag), execute=True)
            champ_eval = _evaluate(args, champion_tag)
            champ_eval.update(champion_cfg)
            rows.append(champ_eval)
            _write_summary(summary_dir / "final_scores.json", rows)
            compare_cmd = [
                sys.executable,
                "tools/compare_phase2_models.py",
                "--csv",
                str(args.csv),
                "--baseline-tag",
                args.baseline_tag,
                "--v2-tag",
                champion_tag,
                "--v3-tag",
                pass1_tag,
                "--test-start",
                args.test_start,
                "--test-end",
                args.test_end,
                "--out",
                str(summary_dir / "TRAINING_COMPARISON_RETRAIN_V6.md"),
            ]
            _run(compare_cmd, execute=True)

    if not args.skip_wf:
        wf_tag = "retrain_v6_pass3_wf"
        wf_cmd = [
            sys.executable,
            "tools/train_phase2_wf.py",
            "--csv",
            str(args.csv),
            "--tag",
            wf_tag,
            "--instrument",
            "ES",
            "--n-blocks",
            "5",
            "--min-train-days",
            "180",
            "--n-estimators",
            str(max(800, int(args.n_estimators // 2))),
            "--setup-threshold-multiplier",
            "1.75",
            "--max-bad-fade-rate",
            "0.18",
            "--recency-weighting",
            "none",
        ]
        _run(wf_cmd, execute=args.execute)

    print("\nWorkflow notes:")
    print("1. Dry-run mode prints commands only. Re-run with --execute to train.")
    print("2. Execute mode writes pass2_scores.json and final_scores.json under --summary-dir.")
    print("3. Do not promote retrain_v6_pass4_champion until compare/readiness/live-paper gates pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
