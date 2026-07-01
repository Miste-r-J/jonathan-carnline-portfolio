from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Iterable, Optional

import aiohttp

from .bot import PremarketPlannerBot
from .config import load_planner_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Discord premarket planner bot.")
    parser.add_argument(
        "--token",
        default=None,
        help="Override the Discord bot token. Defaults to lookup via the configured environment variable.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = load_planner_config()
    token = (
        args.token
        or os.getenv(config.discord_token_env)
        or os.getenv("AUTOMATION_DISCORD_TOKEN")
        or os.getenv("AUTOMATION_DISCORD_BOT_TOKEN")
        or os.getenv("PREMARKET_DISCORD_TOKEN")
    )

    if not token:
        logging.getLogger("premarket_planner.scheduler").error(
            "Discord token not provided. Set %s or use --token.",
            config.discord_token_env,
        )
        sys.exit(2)

    if not config.enabled:
        logging.getLogger("premarket_planner.scheduler").warning(
            "Premarket planner disabled in config; bot will still start for manual commands."
        )

    bot = PremarketPlannerBot(config)
    logger = logging.getLogger("premarket_planner.scheduler")

    retryable_errors = (aiohttp.ClientError, OSError)
    attempt = 0

    while True:
        try:
            bot.run(token)
            break
        except KeyboardInterrupt:  # pragma: no cover - graceful shutdown via signal
            logger.info("Premarket planner bot stopped via keyboard interrupt.")
            break
        except retryable_errors as exc:
            attempt += 1
            delay = min(60, 5 * attempt)
            logger.warning(
                "Discord connection failed (%s); retrying in %s seconds", exc, delay
            )
            time.sleep(delay)
        except Exception:
            logger.exception("Premarket planner crashed with an unexpected error")
            raise


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
