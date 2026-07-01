from __future__ import annotations

"""Compatibility shim for legacy lifecycle parity validation entrypoints."""

from pathlib import Path
from typing import Any, Dict

from trading_system.development_tools.audit_live_backfill_parity import audit_live_backfill_parity


def validate_lifecycle_parity(run_dir: Path) -> Dict[str, Any]:
    payload = audit_live_backfill_parity(run_dir, run_dir, run_dir / "parity_audit")
    return {
        "total_lifecycle_records": payload.get("live_decisions_total", 0),
        "total_signal_to_order_records": len(payload.get("order_recon", []) or []),
        "issues": payload.get("mismatches", []),
        "warnings": [],
        "verdict": "PASS" if not payload.get("mismatches") else "FAIL",
        "report": payload,
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Validate lifecycle parity artifacts.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = validate_lifecycle_parity(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
