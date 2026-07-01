from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading_system.development_tools.validate_phase1_run import validate_run


def _iter_run_dirs(path: Path) -> list[Path]:
    if path.is_dir() and (path / "status.json").exists():
        return [path]
    if not path.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "status.json").exists():
            out.append(child)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Replayable live-run stability audit.")
    parser.add_argument("path", type=Path, help="Run directory or parent folder containing run directories.")
    parser.add_argument("--require-order-lifecycle", action="store_true", help="Also require OPEN/ACK/FILL/protection lifecycle evidence.")
    args = parser.parse_args()

    run_dirs = _iter_run_dirs(args.path)
    if not run_dirs:
        print("AUDIT_NO_RUNS_FOUND")
        return 2

    any_fail = False
    summary: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        failures = validate_run(run_dir, require_order_lifecycle=bool(args.require_order_lifecycle))
        summary.append(
            {
                "run_dir": str(run_dir),
                "pass": not failures,
                "failure_count": len(failures),
                "codes": sorted({str(f.get("code")) for f in failures}),
            }
        )
        if failures:
            any_fail = True
            print(f"AUDIT_FAIL {run_dir}")
            for f in failures:
                print(json.dumps(f, ensure_ascii=True))
        else:
            print(f"AUDIT_PASS {run_dir}")
    print("AUDIT_SUMMARY")
    print(json.dumps(summary, ensure_ascii=True))
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
