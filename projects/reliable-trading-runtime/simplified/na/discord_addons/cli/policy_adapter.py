"""Adapter bridging stream_live_csv with the RiskGuard addon."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

from zoneinfo import ZoneInfo

from na.discord_addons.addons.risk_guard import Context, PolicyDecision, RiskGuard

logger = logging.getLogger(__name__)

_guard: Optional[RiskGuard] = None
_tz: ZoneInfo = ZoneInfo("America/Denver")
_router: Optional[Any] = None
_panic_flatten_cmd: Optional[str] = None


def configure(
    policy_profile: str,
    risk_config: str,
    tz: str,
    news_cache: Optional[str],
    *,
    instrument: str = "",
    materialized_cfg: Optional[str] = None,
    risk_strict: bool = True,
    cli_overrides: Optional[Dict[str, Any]] = None,
    env_json_overrides_var: str = "RISK_JSON_OVERRIDES",
    state_overrides: Optional[Dict[str, Any]] = None,
    panic_flatten_cmd: Optional[str] = None,
    router: Optional[Any] = None,
) -> None:
    """Instantiate the shared RiskGuard instance."""
    global _guard, _tz, _panic_flatten_cmd, _router
    _tz = ZoneInfo(tz)
    _panic_flatten_cmd = panic_flatten_cmd
    _router = router
    try:
        _guard = RiskGuard(
            profile=policy_profile,
            yaml_path=risk_config,
            instrument=instrument or "",
            materialized_cfg=materialized_cfg,
            risk_strict=risk_strict,
            news_cache=news_cache,
            state_overrides=state_overrides,
            panic_lock_hook=on_panic_lock,
            cli_overrides=cli_overrides,
            env_json_overrides_var=env_json_overrides_var,
            default_tz=tz,
        )
        logger.info(
            "RiskGuard initialized profile=%s config=%s tz=%s news_cache=%s instrument=%s persist_override=%s",
            policy_profile,
            risk_config,
            tz,
            news_cache,
            instrument or "?",
            state_overrides,
        )
        try:
            summary = getattr(_guard, "config_summary", None)
            if callable(summary):
                logger.info("RiskGuard effective: %s", summary())
        except Exception:
            logger.debug("RiskGuard summary unavailable", exc_info=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to initialize RiskGuard: %s", exc)
        _guard = None


def guard_loaded() -> bool:
    """Return True if the guard is available."""
    return _guard is not None


def build_context(payload: Dict[str, Any]) -> Context:
    """Construct a Context dataclass from the provided mapping."""
    now = payload.get("now")
    if isinstance(now, datetime) and now.tzinfo is None:
        now = now.replace(tzinfo=_tz)
    if now is None:
        now = datetime.now(tz=_tz)
    return Context(
        instrument=payload["instrument"],
        tier_name=payload.get("tier_name"),
        strategy_name=payload.get("strategy_name"),
        now=now,
        price=payload.get("price", 0.0),
        vwap=payload.get("vwap"),
        vwap_slope=payload.get("vwap_slope"),
        ema21_5m=payload.get("ema21_5m"),
        ema50_5m=payload.get("ema50_5m"),
        ema20_15m=payload.get("ema20_15m"),
        ema50_15m=payload.get("ema50_15m"),
        atr_5m=payload.get("atr_5m"),
        ib_high=payload.get("ib_high"),
        ib_low=payload.get("ib_low"),
        prev_day_high=payload.get("prev_day_high"),
        prev_day_low=payload.get("prev_day_low"),
        premarket_high=payload.get("premarket_high"),
        premarket_low=payload.get("premarket_low"),
        last_trade_result=payload.get("last_trade_result"),
        last_trade_time=payload.get("last_trade_time"),
        last_direction=payload.get("last_direction"),
        realized_R_today=payload.get("realized_R_today", 0.0),
        realized_usd_today=payload.get("realized_usd_today", 0.0),
        losses_today=payload.get("losses_today", 0),
        wins_today=payload.get("wins_today", 0),
        signal_bars=payload.get("signal_bars"),
    )


def should_trade_now(now: datetime, instrument: str, tier: Optional[str] = None) -> PolicyDecision:
    """Fast pre-entry check to gate trading before heavy processing."""
    if _guard is None:
        return PolicyDecision(action="allow", reason="RiskGuard disabled")
    if now.tzinfo is None:
        now = now.replace(tzinfo=_tz)
    return _guard.should_trade_now(now, instrument=instrument, tier=tier)


def pretrade_gate(ctx: Context, side: str) -> PolicyDecision:
    """Invoke RiskGuard before emitting a trade signal."""
    if _guard is None:
        return PolicyDecision(action="allow", reason="RiskGuard disabled")
    return _guard.evaluate_entry(ctx, side)  # type: ignore[arg-type]


def posttrade_record(
    result: str,
    r: float,
    usd: float,
    when: datetime,
    side: str,
    *,
    instrument: Optional[str] = None,
) -> None:
    """Persist completed trade results into the guard's state."""
    if _guard is None:
        return
    if when.tzinfo is None:
        when = when.replace(tzinfo=_tz)
    _guard.record_fill(result=result, r=r, usd=usd, when=when, side=side, instrument=instrument)


def policy_snapshot(instrument: str) -> Dict[str, Any]:
    """Return summary data for Discord reporting."""
    if _guard is None:
        return {}
    return _guard.last_policy_snapshot(instrument)


def on_panic_lock(payload: Dict[str, Any]) -> None:
    """Respond to a panic lock by broadcasting and triggering flatten hooks."""
    instrument = payload.get("instrument") or "?"
    tier = payload.get("tier")
    state = payload.get("state") or {}
    lockout = state.get("lockout") or {}
    if not isinstance(lockout, dict):
        lockout = {}
    until_ts = lockout.get("until_ts")
    until_txt = ""
    if until_ts:
        try:
            until_dt = datetime.fromtimestamp(int(until_ts), tz=_tz)
            until_txt = until_dt.strftime("%H:%M %Z")
        except Exception:
            until_txt = str(until_ts)

    title = "PANIC LOCK"
    label = f"{instrument}" + (f" ({tier})" if tier else "")
    description = f"{label} hit the loss limit. Entries disabled for the session."
    fields = [
        {"name": "Losses", "value": str(state.get("losses", 0)), "inline": True},
        {"name": "Consec Losses", "value": str(state.get("consecutive_losses", 0)), "inline": True},
        {"name": "Realized R", "value": f"{float(state.get('realized_R', 0.0)):.2f}", "inline": True},
        {"name": "Realized $", "value": f"{float(state.get('realized_usd', 0.0)):.0f}", "inline": True},
    ]
    if until_txt:
        fields.append({"name": "Lockout Until", "value": until_txt, "inline": False})

    if _router is not None:
        publish_embed = getattr(_router, "publish_embed", None)
        publish_text = getattr(_router, "publish_text", None)
        try:
            if callable(publish_embed):
                publish_embed(title=title, description=description, fields=fields, color=0xCC0000)
            elif callable(publish_text):
                publish_text(f"{title}: {description}")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to publish panic lock alert: %s", exc)

    if _panic_flatten_cmd:
        try:
            result = subprocess.run(  # noqa: PLW1510 - blocking intentional
                _panic_flatten_cmd,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning("Panic flatten command failed rc=%s stderr=%s", result.returncode, result.stderr.strip())
            else:
                logger.info("Panic flatten command executed: %s", _panic_flatten_cmd)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to execute panic flatten command: %s", exc)
