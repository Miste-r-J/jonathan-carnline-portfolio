"""
Compatibility shim for ``trading_system.runtime_engine.data_streaming.live_csv_runner``.

In the full stack this module wires the CLI arguments into the asynchronous
runner.  The standalone gym embeds that logic directly inside
``live_trading_runtime`` so the helper simply validates the call signature and
invokes the provided callback.
"""

from __future__ import annotations

from typing import Callable, Sequence


def run_stream_from_csv(args: Sequence[str] | None = None, *, main_fn: Callable[[Sequence[str] | None], None] | None = None) -> None:
    """
    Execute the supplied ``main_fn`` with the original argv.
    """
    if main_fn is None:
        raise RuntimeError("live_trading_runtime is self-contained in the standalone gym")
    main_fn(args)


__all__ = ["run_stream_from_csv"]
