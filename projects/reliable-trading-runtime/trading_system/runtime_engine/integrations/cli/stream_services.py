"""
stream_services.py — Domain service objects extracted from LiveCSVStreamer.

Each service encapsulates one slice of behavior that was previously embedded
inline in the 33k-line LiveCSVStreamer class. They are importable and testable
independently.

Phase 4 wiring: live_trading_runtime.py imports these services and delegates to them
instead of containing the logic directly. The LiveCSVStreamer constructor is not
changed — services are constructed from existing constructor arguments and stored
as self._<service> attributes.

Services provided:
  SubprocessRunner   — centralised subprocess.run() with mandatory timeouts
  FeatureGuard       — feature alignment and Phase2 hash validation
  RiskConfigLoader   — strict / non-strict YAML config loading
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SubprocessRunner
# ---------------------------------------------------------------------------

class SubprocessRunner:
    """Centralised subprocess.run() wrapper.

    Every subprocess call in live_trading_runtime.py should route through this
    class so that timeouts and error logging are consistent.  shell=True is
    never allowed — callers must pass a list of tokens.
    """

    DEFAULT_TIMEOUT_SEC: float = 30.0

    def __init__(self, default_timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> None:
        self._timeout = default_timeout_sec

    def run(
        self,
        cmd: List[str],
        *,
        timeout_sec: Optional[float] = None,
        check: bool = True,
        capture_output: bool = True,
        cwd: Optional[str | Path] = None,
    ) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
        """Run *cmd* (must be a list, never a string) with a hard timeout.

        Raises:
            RuntimeError: on timeout.
            subprocess.CalledProcessError: if check=True and exit code != 0.
        """
        if isinstance(cmd, str):
            raise TypeError(
                "SubprocessRunner.run() requires a list of tokens, not a string. "
                "Never pass shell=True."
            )
        timeout = timeout_sec if timeout_sec is not None else self._timeout
        try:
            return subprocess.run(
                cmd,
                timeout=timeout,
                check=check,
                capture_output=capture_output,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "Subprocess timed out after %.1fs: %s", timeout, " ".join(str(t) for t in cmd)
            )
            raise RuntimeError(
                f"Subprocess timed out after {timeout}s: {' '.join(str(t) for t in cmd)}"
            )

    def run_silent(
        self,
        cmd: List[str],
        *,
        timeout_sec: Optional[float] = None,
        cwd: Optional[str | Path] = None,
    ) -> Optional[subprocess.CompletedProcess]:  # type: ignore[type-arg]
        """Run cmd and swallow all errors, returning None on any failure.

        Use only for best-effort / diagnostic commands where failure is acceptable.
        """
        try:
            return self.run(cmd, timeout_sec=timeout_sec, check=False, cwd=cwd)
        except Exception as exc:
            logger.debug("Best-effort subprocess failed (suppressed): %s — %s", cmd, exc)
            return None


# ---------------------------------------------------------------------------
# FeatureGuard
# ---------------------------------------------------------------------------

class FeatureGuard:
    """Feature alignment and Phase2 hash validation service.

    Replaces the inline try/import/except FeatureMismatchError pattern that
    caused NameError when the import itself failed (Phase 1a fix).
    """

    def __init__(self) -> None:
        try:
            from trading_system.runtime_engine.modeling.exceptions import FeatureMismatchError as _FME
            from trading_system.runtime_engine.modeling.feature_constants import MANDATORY_MODEL_FEATURES as _MMF
            from trading_system.runtime_engine.modeling.feature_hash import validate_feature_schema_and_hash as _VFSH
            self._FeatureMismatchError = _FME
            self._mandatory_features: Sequence[str] = _MMF
            self._validate_hash = _VFSH
            self.available = True
        except ImportError as exc:
            logger.warning("FeatureGuard: modules unavailable, hard checks disabled: %s", exc)
            self._FeatureMismatchError = RuntimeError
            self._mandatory_features = []
            self._validate_hash = None
            self.available = False

    @property
    def FeatureMismatchError(self) -> type:
        return self._FeatureMismatchError

    def check_no_silent_zeros(
        self,
        feature_names: Sequence[str],
        present_columns: Sequence[str],
        warn_context: str = "model",
    ) -> None:
        """Raise FeatureMismatchError if any mandatory feature would be zero-filled.

        Replaces the buggy try-import-inside-try pattern in _align_X.
        """
        if not self.available:
            logger.error(
                "FeatureGuard.check_no_silent_zeros: guard disabled — "
                "missing model features will be silently zero-filled. context=%s",
                warn_context,
            )
            return
        mandatory = set(self._mandatory_features)
        silent_zeros = [
            name for name in feature_names
            if name not in present_columns and name in mandatory
        ]
        if silent_zeros:
            raise self._FeatureMismatchError(  # type: ignore[call-arg]
                f"_align_X would silently zero-fill {len(silent_zeros)} "
                f"model-required feature(s): {silent_zeros}"
            )

    def validate_phase2_hash(
        self,
        columns: Sequence[str],
        expected_hash: str,
    ) -> str:
        """Validate Phase2 feature schema hash. Returns the computed hash.

        Replaces the second buggy try-import-inside-try pattern in _poll_once.
        Raises FeatureMismatchError if validation fails.
        Raises RuntimeError if the guard is unavailable (fail closed for Phase2).
        """
        if not self.available or self._validate_hash is None:
            raise RuntimeError(
                "FeatureGuard: feature_hash module unavailable — "
                "cannot validate Phase2 feature schema. Refusing to trade."
            )
        return self._validate_hash(  # type: ignore[call-arg]
            columns,
            required_features=self._mandatory_features,
            expected_hash=expected_hash,
        )


# ---------------------------------------------------------------------------
# RiskConfigLoader
# ---------------------------------------------------------------------------

class RiskConfigLoader:
    """Strict/non-strict YAML risk config loader.

    Centralises the three _load_*_risk_* functions from live_trading_runtime.py
    so that strict-mode logic lives in one place and can be tested independently.

    strict_mode=True  → any missing or malformed file raises immediately
    strict_mode=False → log warning and return safe defaults (backward-compat)
    """

    def __init__(self, strict_mode: bool = False) -> None:
        self.strict_mode = strict_mode

    def _require_yaml(self) -> Any:
        try:
            import yaml  # type: ignore
            return yaml
        except ImportError as exc:
            if self.strict_mode:
                raise RuntimeError(
                    "NA_LIVE_STRICT_CONFIG_MODE=1: PyYAML is required but not installed."
                ) from exc
            return None

    def _resolve_path(self, path: Optional[str], label: str) -> Optional[Path]:
        if not path:
            if self.strict_mode:
                raise RuntimeError(
                    f"NA_LIVE_STRICT_CONFIG_MODE=1: {label} config path is required but was not provided."
                )
            return None
        cfg_path = Path(path).expanduser()
        if not cfg_path.exists():
            if self.strict_mode:
                raise FileNotFoundError(
                    f"NA_LIVE_STRICT_CONFIG_MODE=1: {label} config not found at {cfg_path}"
                )
            logger.warning("%s config not found at %s; using defaults.", label, cfg_path)
            return None
        return cfg_path

    def _load_yaml(self, cfg_path: Path, label: str) -> Dict[str, Any]:
        yaml = self._require_yaml()
        if yaml is None:
            logger.warning("PyYAML unavailable; skipping %s config at %s", label, cfg_path)
            return {}
        try:
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            if self.strict_mode:
                raise RuntimeError(
                    f"NA_LIVE_STRICT_CONFIG_MODE=1: failed to parse {label} YAML at {cfg_path}"
                ) from exc
            logger.warning("Failed to parse %s YAML at %s: %s", label, cfg_path, exc)
            return {}

    def load_raw(self, path: Optional[str], label: str) -> Dict[str, Any]:
        """Load any YAML config file and return the raw dict. Empty dict on failure."""
        cfg_path = self._resolve_path(path, label)
        if cfg_path is None:
            return {}
        return self._load_yaml(cfg_path, label)
