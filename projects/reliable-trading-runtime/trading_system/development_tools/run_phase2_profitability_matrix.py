import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "runtime_engine" / "modeling" / "train_phase2.py"


def _parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for item in str(raw or "").split(","):
        txt = item.strip()
        if not txt:
            continue
        out.append(float(txt))
    return out


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for item in str(raw or "").split(","):
        txt = item.strip()
        if not txt:
            continue
        out.append(int(txt))
    return out


def _matrix_rows(
    horizons: Iterable[int],
    label_thresholds: Iterable[float],
    setup_multipliers: Iterable[float],
    stack_setup_prob_modes: Iterable[bool],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0
    for horizon, label_threshold, setup_mult, stack_mode in itertools.product(
        horizons,
        label_thresholds,
        setup_multipliers,
        stack_setup_prob_modes,
    ):
        idx += 1
        rows.append(
            {
                "id": idx,
                "horizon": int(horizon),
                "label_threshold": float(label_threshold),
                "setup_threshold_multiplier": float(setup_mult),
                "stack_setup_prob": bool(stack_mode),
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Generate/execute controlled Phase-2 profitability matrix.")
    p.add_argument("--csv", required=True, help="Training CSV path.")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizons", default="2,4,8")
    p.add_argument("--label-thresholds", default="0.0015,0.0025,0.0035")
    p.add_argument("--setup-threshold-multipliers", default="1.25,1.5,1.75")
    p.add_argument("--stack-setup-prob-modes", default="false,true", help="Comma list of false/true.")
    p.add_argument("--out-dir", default=None, help="Output directory for matrix plan/results.")
    p.add_argument("--execute", action="store_true", help="Run training commands. Default writes plan only.")
    p.add_argument("--python", default=sys.executable, help="Python executable used for training invocations.")
    p.add_argument("--extra-args", default="", help="Extra raw args appended to each train_phase2 call.")
    args = p.parse_args()

    horizons = _parse_int_list(args.horizons)
    label_thresholds = _parse_float_list(args.label_thresholds)
    setup_multipliers = _parse_float_list(args.setup_threshold_multipliers)
    stack_modes = [s.strip().lower() == "true" for s in str(args.stack_setup_prob_modes).split(",") if s.strip()]
    if not horizons or not label_thresholds or not setup_multipliers or not stack_modes:
        raise ValueError("Matrix dimensions cannot be empty.")

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "runs" / f"phase2_profitability_matrix_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _matrix_rows(horizons, label_thresholds, setup_multipliers, stack_modes)
    plan_path = out_dir / "matrix_plan.json"
    commands_path = out_dir / "commands.txt"
    results_path = out_dir / "results.jsonl"

    commands: List[List[str]] = []
    for row in rows:
        run_slug = f"h{row['horizon']}_lt{row['label_threshold']}_sm{row['setup_threshold_multiplier']}_stack{int(row['stack_setup_prob'])}"
        run_dir = out_dir / run_slug
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(args.python),
            str(TRAIN_SCRIPT),
            "--csv",
            str(args.csv),
            "--instrument",
            str(args.instrument),
            "--horizon",
            str(row["horizon"]),
            "--label-threshold",
            str(row["label_threshold"]),
            "--setup-threshold-multiplier",
            str(row["setup_threshold_multiplier"]),
            "--setup-model-path",
            str(run_dir / "phase2_setup.pkl"),
            "--direction-model-path",
            str(run_dir / "phase2_direction.pkl"),
            "--close-model-path",
            str(run_dir / "phase2_close.pkl"),
            "--train-start",
            "2021-01-01",
            "--train-end",
            "2023-12-31",
            "--test-start",
            "2024-01-01",
            "--test-end",
            "2025-12-31",
        ]
        if row["stack_setup_prob"]:
            cmd.append("--stack-setup-prob")
        extra = [chunk for chunk in str(args.extra_args or "").split(" ") if chunk.strip()]
        cmd.extend(extra)
        commands.append(cmd)

    plan_payload = {
        "generated_utc": stamp,
        "train_script": str(TRAIN_SCRIPT),
        "matrix_size": len(rows),
        "rows": rows,
        "out_dir": str(out_dir),
        "execute": bool(args.execute),
    }
    plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
    commands_path.write_text("\n".join(" ".join(cmd) for cmd in commands) + "\n", encoding="utf-8")

    if not args.execute:
        print(json.dumps({"status": "planned", "out_dir": str(out_dir), "matrix_size": len(rows)}, ensure_ascii=True))
        return 0

    with results_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row, cmd in zip(rows, commands):
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            record = {
                **row,
                "returncode": int(proc.returncode),
                "command": cmd,
                "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
                "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
            }
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
            fh.flush()

    print(json.dumps({"status": "executed", "out_dir": str(out_dir), "matrix_size": len(rows)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
