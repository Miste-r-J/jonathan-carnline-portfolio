"""
Compatibility shim for ``na.streaming.live_csv_runner``.

In the full stack this module wires the CLI arguments into the asynchronous
runner.  The simplified gym embeds that logic directly inside
``stream_live_csv`` so the helper simply validates the call signature and
invokes the provided callback.
"""

from __future__ import annotations

from typing import Callable, Sequence


def run_stream_from_csv(args: Sequence[str] | None = None, *, main_fn: Callable[[Sequence[str] | None], None] | None = None) -> None:
    """
    Execute the supplied ``main_fn`` with the original argv.
    """
    if main_fn is None:
        raise RuntimeError("stream_live_csv is self-contained in the simplified gym")
    main_fn(args)


__all__ = ["run_stream_from_csv"]
