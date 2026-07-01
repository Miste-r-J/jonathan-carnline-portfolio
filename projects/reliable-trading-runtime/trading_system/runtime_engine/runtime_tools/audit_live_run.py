from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from trading_system.development_tools.validate_phase1_run import validate_run


def audit_run(run_dir: Path, *, require_order_lifecycle: bool = False) -> Dict[str, Any]:
    failures = validate_run(Path(run_dir), require_order_lifecycle=bool(require_order_lifecycle))
    return {
        "run_dir": str(Path(run_dir)),
        "pass": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit a live run directory.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-order-lifecycle", action="store_true")
    args = parser.parse_args()
    payload = audit_run(args.run_dir, require_order_lifecycle=bool(args.require_order_lifecycle))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
