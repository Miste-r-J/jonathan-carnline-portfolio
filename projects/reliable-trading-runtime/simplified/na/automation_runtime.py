from __future__ import annotations

import argparse
import json
from typing import Optional

from .premarket_planner import runtime as planner_runtime


AUTOMATION_COMMANDS = ["/planner", "/news"]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Discord automation bot runtime wrapper.")
    parser.add_argument("--config", default=None, help="Optional planner/automation config path")
    parser.add_argument("--readiness", action="store_true", help="Print readiness report and exit")
    parser.add_argument("--diagnostics", action="store_true", help="Print Discord auth diagnostics and exit")
    parser.add_argument("--log-level", default="INFO", help="Logging level passed through to scheduler")
    args = parser.parse_args(argv)

    if args.diagnostics:
        diagnostics = planner_runtime.build_auth_diagnostics(args.config)
        diagnostics["surface"] = "automation_bot"
        diagnostics["commands"] = list(AUTOMATION_COMMANDS)
        print(json.dumps(diagnostics, indent=2, sort_keys=True))
        return 0

    report = planner_runtime.build_readiness_report(args.config)
    report["surface"] = "automation_bot"
    report["commands"] = list(AUTOMATION_COMMANDS)
    report["host_runtime"] = "automation_bot"
    if args.readiness:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2
    if not report["ok"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    return planner_runtime.main(["--config", args.config, "--log-level", args.log_level] if args.config else ["--log-level", args.log_level])


if __name__ == "__main__":
    raise SystemExit(main())
