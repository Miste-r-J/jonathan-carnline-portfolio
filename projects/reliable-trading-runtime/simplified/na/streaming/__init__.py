"""
Streaming helpers for the simplified gym.

Only the CSV runner shim is exposed to keep imports compatible with the
production CLI.
"""

from .live_csv_runner import run_stream_from_csv

__all__ = ["run_stream_from_csv"]
