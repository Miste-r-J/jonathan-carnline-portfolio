from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Optional

from trading_system.runtime_engine.integrations.nt_bridge import NTBridgeServer


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("nt_smoketest")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _input_loop(server: NTBridgeServer, logger: logging.Logger) -> None:
    while True:
        try:
            line = input().strip().lower()
        except EOFError:
            return
        if line == "ping":
            ok = server.send({"type": "PING", "ts": time.time()})
            logger.info("PING sent=%s", ok)
        elif line == "status":
            logger.info("STATUS %s", server.get_status())
        elif line in {"quit", "exit"}:
            server.shutdown()
            return


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="NinjaTrader TCP smoke test server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args(argv)

    logger = _build_logger()
    server = NTBridgeServer(args.host, args.port, logger=logger, log_path=None)
    server.start()

    logger.info("Type 'ping' to send a PING, 'status' to print status, 'exit' to quit.")
    input_thread = threading.Thread(target=_input_loop, args=(server, logger), daemon=True)
    input_thread.start()

    try:
        while True:
            status = server.get_status()
            logger.info(
                "status listening=%s connected=%s handshake_ok=%s remote=%s last_rx=%s last_tx=%s last_err=%s",
                status.get("listening"),
                status.get("connected"),
                status.get("handshake_ok"),
                status.get("remote"),
                status.get("last_rx_ts"),
                status.get("last_tx_ts"),
                status.get("last_err"),
            )
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
