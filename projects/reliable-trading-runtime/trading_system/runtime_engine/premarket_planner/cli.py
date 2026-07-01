from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import PlannerConfig, load_config
from .core import generate_plan
from .embed import PlanPayload, build_plan_payload
from .chart_context import (
    serialize_plan_context,
    dump_plan_context,
    DEFAULT_PLAN_CONTEXT_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the premarket plan embed.")
    parser.add_argument("-c", "--config", help="Path to premarket planner YAML config.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip webhook publish even if configured.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Emit the payload JSON to stdout.",
    )
    parser.add_argument(
        "--plan-json",
        help="Optional path to write a structured planner context for charts.misterjtrades.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log = logging.getLogger("premarket_planner.cli")

    try:
        config = load_config(args.config)
    except Exception as exc:
        log.error("Failed to load config: %s", exc)
        return 1

    result = generate_plan(config, logger=log)
    payload = build_plan_payload(config, result)
    chart_context = serialize_plan_context(result, config)

    _print_summary(payload)

    if args.print_json:
        print(json.dumps(_payload_to_dict(payload), indent=2, sort_keys=True))

    webhook = config.output.discord_webhook or config.webhook_url
    if webhook and not args.dry_run:
        _dispatch_webhook(webhook, payload, log)

    plan_json_path = args.plan_json or os.getenv("PREMARKET_PLAN_PATH")
    target_path = Path(plan_json_path).expanduser() if plan_json_path else DEFAULT_PLAN_CONTEXT_PATH
    dump_plan_context(target_path, chart_context)

    return 0


def _print_summary(payload: PlanPayload) -> None:
    print(payload.content)
    print(payload.embed.description)
    for field in payload.embed.fields:
        print(f"\n{field.name}\n{field.value}")
    if payload.warnings:
        print("\nWarnings:")
        for item in payload.warnings:
            print(f" - {item}")


def _dispatch_webhook(url: str, payload: PlanPayload, log: logging.Logger) -> None:
    body = json.dumps(_payload_to_dict(payload)).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            log.info("Webhook dispatched with status %s", response.status)
    except HTTPError as exc:
        log.error("Webhook rejected with status %s: %s", exc.code, exc.reason)
    except URLError as exc:
        log.error("Webhook dispatch failed: %s", exc.reason)


def _payload_to_dict(payload: PlanPayload) -> dict:
    return {"content": payload.content, "embeds": [payload.embed.to_dict()]}


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
