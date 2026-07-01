from __future__ import annotations

import builtins
import io
import os
import threading
import warnings
from typing import Iterable, Optional, Set


DEFAULT_DENY_MODULES = {
    "na.discord_addons.nt_bridge",
    "na.discord_addons.nt_execution",
    "na.discord_addons.mjt_bridge",
    "na.discord_addons.mock_nt_transport",
    "na.bot.online_learning",
}

DEFAULT_DENY_FILES = {
    "status.json",
    "lockout.json",
    "stream_state.json",
    "nt_bridge.jsonl",
    "order_events.jsonl",
    "exec_events.jsonl",
    "execution_ledger.jsonl",
    "trades.csv",
}

_PRED_STAGE = threading.local()


def default_denylist() -> tuple[Set[str], Set[str]]:
    return set(DEFAULT_DENY_MODULES), set(DEFAULT_DENY_FILES)


class ExecutionAccessMonitor:
    """Guard against execution-module imports and execution file reads in prediction stage."""

    def __init__(
        self,
        *,
        deny_modules: Optional[Iterable[str]] = None,
        deny_files: Optional[Iterable[str]] = None,
        strict: bool = True,
    ) -> None:
        if deny_modules is None or deny_files is None:
            default_modules, default_files = default_denylist()
        else:
            default_modules, default_files = set(), set()
        self.deny_modules = set(deny_modules) if deny_modules is not None else set(default_modules)
        self.deny_files = {str(x).lower() for x in (deny_files if deny_files is not None else default_files)}
        self.strict = bool(strict)
        self._lock = threading.RLock()
        self._active = False
        self._orig_import = builtins.__import__
        self._orig_open = builtins.open
        self._orig_io_open = io.open

    def _deny(self, msg: str) -> None:
        if self.strict:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def _check_module(self, name: str) -> None:
        for prefix in self.deny_modules:
            if name == prefix or name.startswith(prefix + "."):
                self._deny(f"Execution module import blocked in prediction stage: {name}")
                return

    def _check_path(self, path: object, mode: Optional[str]) -> None:
        if not mode or "r" not in mode:
            return
        if path is None:
            return
        try:
            path_str = os.fspath(path)
        except Exception:
            path_str = str(path)
        base = os.path.basename(str(path_str)).lower()
        if base in self.deny_files:
            self._deny(f"Execution file read blocked in prediction stage: {base}")

    def __enter__(self) -> "ExecutionAccessMonitor":
        with self._lock:
            if self._active:
                return self
            self._active = True
            setattr(_PRED_STAGE, "active", True)

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                self._check_module(str(name))
                return self._orig_import(name, globals, locals, fromlist, level)

            def guarded_open(file, mode="r", *args, **kwargs):
                self._check_path(file, mode)
                return self._orig_open(file, mode, *args, **kwargs)

            def guarded_io_open(file, mode="r", *args, **kwargs):
                self._check_path(file, mode)
                return self._orig_io_open(file, mode, *args, **kwargs)

            builtins.__import__ = guarded_import  # type: ignore[assignment]
            builtins.open = guarded_open  # type: ignore[assignment]
            io.open = guarded_io_open  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        with self._lock:
            if not self._active:
                return
            builtins.__import__ = self._orig_import  # type: ignore[assignment]
            builtins.open = self._orig_open  # type: ignore[assignment]
            io.open = self._orig_io_open  # type: ignore[assignment]
            setattr(_PRED_STAGE, "active", False)
            self._active = False


def in_prediction_stage() -> bool:
    return bool(getattr(_PRED_STAGE, "active", False))
