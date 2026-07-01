from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from .config import load_effective_profile

logger = logging.getLogger("na.risk_guard")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] risk_guard %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class PolicyDecision:
    action: str
    reason: str
    size_multiplier: float = 1.0
    hard_stop_pts: Optional[float] = None
    target1_pts: Optional[float] = None
    trail_atr_mult: Optional[float] = None
    lockout_until: Optional[datetime] = None
    news_blackout: bool = False
    news_post_window_end: Optional[datetime] = None
    news_next: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    instrument: str
    tier_name: Optional[str]
    strategy_name: Optional[str]
    now: datetime
    price: float = 0.0
    vwap: Optional[float] = None
    vwap_slope: Optional[float] = None
    ema21_5m: Optional[float] = None
    ema50_5m: Optional[float] = None
    ema20_15m: Optional[float] = None
    ema50_15m: Optional[float] = None
    atr_5m: Optional[float] = None
    ib_high: Optional[float] = None
    ib_low: Optional[float] = None
    prev_day_high: Optional[float] = None
    prev_day_low: Optional[float] = None
    premarket_high: Optional[float] = None
    premarket_low: Optional[float] = None
    last_trade_result: Optional[str] = None
    last_trade_time: Optional[datetime] = None
    last_direction: Optional[str] = None
    realized_R_today: float = 0.0
    realized_usd_today: float = 0.0
    losses_today: int = 0
    wins_today: int = 0
    signal_bars: Optional[int] = None


class RiskGuard:
    """
    Simplified RiskGuard that enforces basic daily limits from the YAML config.

    It supports profile selection, CLI/environment overrides, loss-based lockouts,
    and broadcasts panic hooks when a lockout trips.
    """

    def __init__(
        self,
        *,
        profile: str,
        yaml_path: str | Path,
        instrument: str,
        materialized_cfg: Optional[str] = None,
        risk_strict: bool = True,
        news_cache: Optional[str] = None,
        state_overrides: Optional[Dict[str, Any]] = None,
        panic_lock_hook: Optional[Any] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
        env_json_overrides_var: str = "RISK_JSON_OVERRIDES",
        default_tz: str = "America/Chicago",
    ) -> None:
        profile_cfg = load_effective_profile(
            profile=profile,
            instrument=instrument,
            yaml_path=str(yaml_path),
            materialized_path=materialized_cfg,
            risk_strict=risk_strict,
            cli_overrides=cli_overrides,
            env_json_overrides_var=env_json_overrides_var,
            default_tz=default_tz,
        )
        self.profile = profile_cfg.name
        self.cfg = profile_cfg.data
        self.instrument = instrument or self.cfg.get("instrument", "") or ""
        self.state_overrides = state_overrides or {}
        self.panic_lock_hook = panic_lock_hook
        self._tz = ZoneInfo(str(self.cfg.get("tz") or default_tz))
        self._loss_limits = self.cfg.get("loss_limits") or {}
        self._max_losses = self._coerce_int(self._loss_limits.get("max_losses_per_day"))
        self._max_consec = self._coerce_int(self._loss_limits.get("max_consecutive_losses"))
        self._max_dollar_loss = self._coerce_float(self._loss_limits.get("max_dollar_loss"))
        self._max_r = self._coerce_float(self._loss_limits.get("max_R_per_day"))
        self._size_after_first_loss_mult = self._coerce_float(self._loss_limits.get("size_after_first_loss_mult"))
        self._max_trades_per_day = self._coerce_int(self._loss_limits.get("max_trades_per_day"))
        self._session = self.cfg.get("session") or {}
        self._rth_start = self._parse_time(self._session.get("rth_start"))
        self._rth_end = self._parse_time(self._session.get("rth_end"))
        confirmations = self.cfg.get("confirmations") or {}
        anti = self.cfg.get("anti_overtrade") or {}
        self._min_signal_bars = (
            self._coerce_int(confirmations.get("min_signal_bars")) if isinstance(confirmations, dict) else None
        )
        self._same_dir_dedupe_min = (
            self._coerce_int(anti.get("same_dir_dedupe_min")) if isinstance(anti, dict) else None
        )
        self._state: Dict[str, Any] = {
            "day": None,
            "losses": 0,
            "wins": 0,
            "consecutive_losses": 0,
            "trades": 0,
            "realized_R": 0.0,
            "realized_usd": 0.0,
            "lockout": False,
            "lockout_reason": None,
            "lockout_since": None,
            "last_trade_result": None,
            "last_trade_time": None,
            "last_direction": None,
            "last_entry_time": None,
            "last_entry_side": None,
        }
        self._last_snapshot: Dict[str, Any] = {}
        logger.info(
            "RiskGuard loaded profile=%s instrument=%s strict=%s yaml=%s",
            self.profile,
            self.instrument or "?",
            risk_strict,
            yaml_path,
        )
        try:
            logger.info("RiskGuard limits profile=%s max_losses=%s max_R=%s max_usd=%s min_signal_bars=%s same_dir_dedupe_min=%s",
                        self.profile,
                        self._max_losses,
                        self._max_r,
                        self._max_dollar_loss,
                        (self.cfg.get("confirmations") or {}).get("min_signal_bars") if isinstance(self.cfg.get("confirmations"), dict) else None,
                        (self.cfg.get("anti_overtrade") or {}).get("same_dir_dedupe_min") if isinstance(self.cfg.get("anti_overtrade"), dict) else None)
        except Exception:
            pass

    @staticmethod
    def _parse_time(value: Optional[str]) -> Optional[time]:
        if not value:
            return None
        try:
            hour, minute = [int(part) for part in value.split(":", 1)]
            return time(hour=hour, minute=minute)
        except Exception:
            return None

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except Exception:
            return None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except Exception:
            return None

    def _reset_day(self, today: date) -> None:
        self._state.update(
            {
                "day": today,
                "losses": 0,
                "wins": 0,
                "consecutive_losses": 0,
                "trades": 0,
                "realized_R": 0.0,
                "realized_usd": 0.0,
                "lockout": False,
                "lockout_reason": None,
                "lockout_since": None,
            }
        )

    def _ensure_day(self, now: datetime) -> None:
        local_today = now.astimezone(self._tz).date()
        if self._state.get("day") != local_today:
            self._reset_day(local_today)

    def _within_session(self, now: datetime) -> bool:
        local_time = now.astimezone(self._tz).time()
        if self._rth_start and self._rth_end:
            if self._rth_start <= self._rth_end:
                return self._rth_start <= local_time <= self._rth_end
            return local_time >= self._rth_start or local_time <= self._rth_end
        return True

    def _session_end(self, now: datetime) -> Optional[datetime]:
        if not self._rth_end:
            return None
        local_now = now.astimezone(self._tz)
        end_date = local_now.date()
        if self._rth_start and self._rth_end and self._rth_start > self._rth_end:
            # Session crosses midnight. If we're on the "start-day" portion (>= start),
            # then the session end is the next day.
            if local_now.time() >= self._rth_start:
                end_date = end_date + timedelta(days=1)
        return datetime.combine(end_date, self._rth_end, tzinfo=self._tz)

    def _lockout(self, reason: str, now: datetime) -> PolicyDecision:
        if not self._state.get("lockout"):
            self._state["lockout"] = True
            self._state["lockout_reason"] = reason
            self._state["lockout_since"] = now.astimezone(self._tz)
            if callable(self.panic_lock_hook):
                until_dt = self._session_end(now)
                lockout_payload = {"until_ts": int(until_dt.timestamp())} if until_dt else {}
                payload = {
                    "instrument": self.instrument or "?",
                    "reason": reason,
                    "state": {
                        **dict(self._state),
                        "lockout": lockout_payload,
                    },
                    "tier": self.state_overrides.get("scope"),
                }
                try:
                    self.panic_lock_hook(payload)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("panic_lock_hook failed")
        return PolicyDecision(
            action="lockout",
            reason=reason,
            lockout_until=self._session_end(now),
            details=dict(self._state),
        )

    def _check_limits(self, now: datetime) -> PolicyDecision:
        if self._state.get("lockout"):
            return PolicyDecision(
                action="lockout",
                reason=self._state.get("lockout_reason", "lockout_active"),
                lockout_until=self._session_end(now),
            )
        if not self._within_session(now):
            return PolicyDecision(action="block", reason="outside_session")
        if (
            self._max_trades_per_day is not None
            and self._max_trades_per_day > 0
            and self._state.get("trades", 0) >= self._max_trades_per_day
        ):
            return self._lockout("max_trades_reached", now)
        if (
            self._max_losses is not None
            and self._max_losses > 0
            and self._state["losses"] >= self._max_losses
        ):
            return self._lockout("max_losses_reached", now)
        if (
            self._max_consec is not None
            and self._max_consec > 0
            and self._state["consecutive_losses"] >= self._max_consec
        ):
            return self._lockout("max_consecutive_losses", now)
        if (
            self._max_dollar_loss is not None
            and self._max_dollar_loss < 0
            and self._state["realized_usd"] <= self._max_dollar_loss
        ):
            return self._lockout("max_dollar_loss", now)
        if self._max_r is not None and self._max_r < 0 and self._state["realized_R"] <= self._max_r:
            return self._lockout("max_R_loss", now)
        return PolicyDecision(action="allow", reason="within_limits")

    def _size_multiplier(self) -> float:
        size_mult = 1.0
        if self._size_after_first_loss_mult is not None and self._state.get("losses", 0) >= 1:
            size_mult *= float(self._size_after_first_loss_mult)
        return float(size_mult)

    def _check_confirmations(self, ctx: Context, side: str) -> Optional[str]:
        confirmations = self.cfg.get("confirmations") or {}
        if not isinstance(confirmations, dict):
            return None
        require_price_vs_vwap = bool(confirmations.get("require_price_vs_vwap", False))
        require_vwap_slope_sign = bool(confirmations.get("require_vwap_slope_sign", False))
        min_signal_bars = self._coerce_int(confirmations.get("min_signal_bars"))

        if min_signal_bars is not None and ctx.signal_bars is not None:
            try:
                if int(ctx.signal_bars) < min_signal_bars:
                    return "min_signal_bars"
            except Exception:
                pass

        if require_price_vs_vwap and ctx.vwap is not None:
            if side == "long" and ctx.price < ctx.vwap:
                return "price_below_vwap"
            if side == "short" and ctx.price > ctx.vwap:
                return "price_above_vwap"

        if require_vwap_slope_sign and ctx.vwap_slope is not None:
            if side == "long" and ctx.vwap_slope <= 0:
                return "vwap_slope_not_positive"
            if side == "short" and ctx.vwap_slope >= 0:
                return "vwap_slope_not_negative"

        ema_5m = confirmations.get("ema_5m")
        if isinstance(ema_5m, list) and len(ema_5m) >= 2 and ctx.ema21_5m is not None and ctx.ema50_5m is not None:
            if side == "long" and ctx.ema21_5m < ctx.ema50_5m:
                return "ema_5m_not_bullish"
            if side == "short" and ctx.ema21_5m > ctx.ema50_5m:
                return "ema_5m_not_bearish"

        ema_15m = confirmations.get("ema_15m")
        if isinstance(ema_15m, list) and len(ema_15m) >= 2 and ctx.ema20_15m is not None and ctx.ema50_15m is not None:
            if side == "long" and ctx.ema20_15m < ctx.ema50_15m:
                return "ema_15m_not_bullish"
            if side == "short" and ctx.ema20_15m > ctx.ema50_15m:
                return "ema_15m_not_bearish"

        return None

    def _check_anti_overtrade(self, ctx: Context, side: str) -> Optional[str]:
        anti = self.cfg.get("anti_overtrade") or {}
        if not isinstance(anti, dict):
            return None
        same_dir_dedupe_min = self._coerce_int(anti.get("same_dir_dedupe_min"))
        if same_dir_dedupe_min is not None:
            last_entry_time = self._state.get("last_entry_time")
            last_entry_side = self._state.get("last_entry_side")
            if isinstance(last_entry_time, datetime) and last_entry_side == side:
                delta_min = (ctx.now.astimezone(self._tz) - last_entry_time).total_seconds() / 60.0
                if delta_min < float(same_dir_dedupe_min):
                    return "same_dir_dedupe"

        cooldown_after_win_min = self._coerce_int(anti.get("cooldown_after_win_min"))
        cooldown_after_stop_min = self._coerce_int(anti.get("cooldown_after_stop_min"))
        last_trade_time = self._state.get("last_trade_time")
        last_trade_result = self._state.get("last_trade_result")
        if isinstance(last_trade_time, datetime) and isinstance(last_trade_result, str):
            cooldown = None
            if last_trade_result == "win" and cooldown_after_win_min is not None:
                cooldown = cooldown_after_win_min
            if last_trade_result == "loss" and cooldown_after_stop_min is not None:
                cooldown = cooldown_after_stop_min
            if cooldown is not None:
                delta_min = (ctx.now.astimezone(self._tz) - last_trade_time).total_seconds() / 60.0
                if delta_min < float(cooldown):
                    return "cooldown"
        return None

    def should_trade_now(self, now: datetime, *, instrument: str, tier: Optional[str] = None) -> PolicyDecision:
        self._ensure_day(now)
        decision = self._check_limits(now)
        if decision.action != "allow":
            decision.details.setdefault("instrument", instrument or self.instrument)
        decision.size_multiplier = self._size_multiplier()
        if decision.action == "allow" and decision.size_multiplier < 0.999:
            decision.action = "downsize"
            decision.reason = "downsize_after_loss"
        return decision

    def evaluate_entry(self, ctx: Context, side: str) -> PolicyDecision:
        self._ensure_day(ctx.now)
        decision = self._check_limits(ctx.now)
        decision.size_multiplier = self._size_multiplier()
        if decision.action == "allow":
            anti_reason = self._check_anti_overtrade(ctx, side)
            if anti_reason:
                decision = PolicyDecision(action="block", reason=anti_reason, size_multiplier=decision.size_multiplier)
            else:
                conf_reason = self._check_confirmations(ctx, side)
                if conf_reason:
                    decision = PolicyDecision(action="block", reason=conf_reason, size_multiplier=decision.size_multiplier)
                else:
                    self._state["last_entry_time"] = ctx.now.astimezone(self._tz)
                    self._state["last_entry_side"] = side
        if decision.action == "allow" and decision.size_multiplier < 0.999:
            decision.action = "downsize"
            decision.reason = "downsize_after_loss"
        self._last_snapshot = {
            "profile": self.profile,
            "instrument": ctx.instrument,
            "side": side,
            "timestamp": ctx.now.astimezone(self._tz).isoformat(),
            "loss_limit": self._max_losses,
            "trade_limit": self._max_trades_per_day,
            "trades_today": self._state.get("trades", 0),
            "losses_today": self._state.get("losses", 0),
            "realized_usd_today": self._state.get("realized_usd", 0.0),
            "realized_R_today": self._state.get("realized_R", 0.0),
            "wins_today": self._state.get("wins", 0),
            "last_trade_result": self._state.get("last_trade_result"),
            "last_trade_time": self._state.get("last_trade_time").isoformat() if isinstance(self._state.get("last_trade_time"), datetime) else None,
            "last_direction": self._state.get("last_direction"),
            "lockout_until": self._session_end(ctx.now).isoformat() if self._state.get("lockout") and self._session_end(ctx.now) else None,
            "persist_dir": self.state_overrides.get("persist_dir"),
        }
        decision.lockout_until = self._session_end(ctx.now) if decision.action == "lockout" else None
        return decision

    def record_fill(self, *, result: str, r: float, usd: float, when: datetime, side: str, instrument: Optional[str]) -> None:
        self._ensure_day(when)
        self._state["trades"] = int(self._state.get("trades", 0)) + 1
        result_txt = (result or "").lower()
        is_loss = result_txt.startswith("loss") or r < 0 or usd < 0
        if is_loss:
            self._state["losses"] += 1
            self._state["consecutive_losses"] += 1
        else:
            self._state["wins"] += 1
            self._state["consecutive_losses"] = 0
        self._state["realized_R"] += float(r)
        self._state["realized_usd"] += float(usd)
        self._state["last_trade_result"] = result_txt if result_txt in {"win", "loss", "flat"} else result
        self._state["last_trade_time"] = when.astimezone(self._tz)
        self._state["last_direction"] = side
        decision = self._check_limits(when)
        if decision.action == "lockout":
            logger.warning("RiskGuard lockout after fill: %s", decision.reason)

    def last_policy_snapshot(self, instrument: str) -> Dict[str, Any]:
        snapshot = dict(self._last_snapshot)
        snapshot.setdefault("profile", self.profile)
        snapshot.setdefault("loss_limit", self._max_losses)
        snapshot.setdefault("trade_limit", self._max_trades_per_day)
        snapshot.setdefault("trades_today", self._state.get("trades", 0))
        snapshot.setdefault("max_consecutive_losses", self._max_consec)
        snapshot.setdefault("max_R_per_day", self._max_r)
        snapshot.setdefault("max_dollar_loss", self._max_dollar_loss)
        snapshot.setdefault("min_signal_bars", self._min_signal_bars)
        snapshot.setdefault("same_dir_dedupe_min", self._same_dir_dedupe_min)
        snapshot.setdefault("size_after_first_loss_mult", self._size_after_first_loss_mult)
        snapshot.setdefault("losses_today", self._state.get("losses", 0))
        snapshot.setdefault("wins_today", self._state.get("wins", 0))
        snapshot.setdefault("realized_usd_today", self._state.get("realized_usd", 0.0))
        snapshot.setdefault("realized_R_today", self._state.get("realized_R", 0.0))
        snapshot.setdefault("last_trade_result", self._state.get("last_trade_result"))
        snapshot.setdefault(
            "last_trade_time",
            self._state.get("last_trade_time").isoformat() if isinstance(self._state.get("last_trade_time"), datetime) else None,
        )
        snapshot.setdefault("last_direction", self._state.get("last_direction"))
        snapshot.setdefault("lockout_until", "session_end" if self._state.get("lockout") else None)
        snapshot.setdefault("persist_dir", self.state_overrides.get("persist_dir"))
        snapshot.setdefault("cfg_sha256", "simplified")
        return snapshot

    def config_summary(self) -> Dict[str, Any]:
        confirmations = self.cfg.get("confirmations") or {}
        anti = self.cfg.get("anti_overtrade") or {}
        return {
            "profile": self.profile,
            "instrument": self.instrument,
            "tz": str(self.cfg.get("tz")) if self.cfg.get("tz") is not None else None,
            "max_losses_per_day": self._max_losses,
            "max_consecutive_losses": self._max_consec,
            "max_R_per_day": self._max_r,
            "max_dollar_loss": self._max_dollar_loss,
            "min_signal_bars": confirmations.get("min_signal_bars") if isinstance(confirmations, dict) else None,
            "same_dir_dedupe_min": anti.get("same_dir_dedupe_min") if isinstance(anti, dict) else None,
        }


__all__ = ["Context", "PolicyDecision", "RiskGuard"]
