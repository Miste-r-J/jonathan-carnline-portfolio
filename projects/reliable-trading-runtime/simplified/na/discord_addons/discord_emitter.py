from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from datetime import datetime
import time
import json
from typing import Any, Dict, Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

try:
    from .discord_delivery import DiscordDeliveryError, DiscordDeliveryQueue
    from .route_config import OpenClawDiscordRoutes, load_openclaw_routes
except ImportError:  # pragma: no cover - fallback for direct execution
    from discord_delivery import DiscordDeliveryError, DiscordDeliveryQueue  # type: ignore
    from route_config import OpenClawDiscordRoutes, load_openclaw_routes  # type: ignore

logger = logging.getLogger(__name__)

MT_TZ = ZoneInfo("America/Denver")
BAR_FMT = "%Y-%m-%d %H:%M"


class DiscordEmitter:
    """Synchronous Discord webhook helper with graceful fallbacks."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        *,
        username: Optional[str] = None,
        routes: Optional[OpenClawDiscordRoutes] = None,
    ) -> None:
        self.username = username
        self.url = (webhook_url or "").strip()
        self._routes = routes or load_openclaw_routes()
        self._discord_token = self._routes.token
        self._client: Optional[tuple[str, Any]] = None
        self._warned_no_webhook = False
        self._warned_no_client = False
        self._warned_no_route = False
        self._no_route_warn_last_ts: float = 0.0
        self._no_route_warn_suppressed_count: int = 0
        self._no_route_warn_interval_sec: float = 60.0
        self._warned_no_route_reason: Optional[str] = None
        self._no_route_warn_by_reason: Dict[str, int] = {"missing_token": 0, "missing_routes": 0}
        self._route_readiness_state: Optional[str] = None
        self._bootstrap_client()
        dead_letter_path = os.environ.get("DISCORD_DELIVERY_DEADLETTER_PATH", "").strip() or "logs/discord_deadletter.jsonl"
        self._delivery_queue = DiscordDeliveryQueue(
            send_callable=self._send_envelope_direct,
            dead_letter_path=Path(dead_letter_path),
        )

    # ------------------------------------------------------------------ lifecycle

    def _bootstrap_client(self) -> None:
        """Try httpx first (for HTTP/2), fall back to requests."""
        try:
            import httpx  # type: ignore

            self._client = ("httpx", httpx.Client(http2=True, timeout=10))
            self._warned_no_client = False
            return
        except Exception:
            pass

        try:
            import requests  # type: ignore

            self._client = ("requests", requests.Session())
            self._warned_no_client = False
        except Exception:
            self._client = None

    def enable(self, webhook_url: str) -> None:
        """Set (or update) the webhook URL."""
        self.url = webhook_url.strip()
        self._warned_no_webhook = False
        if self._client is None:
            self._bootstrap_client()

    # Backwards compatibility for older call-sites
    enable_discord = enable

    # ------------------------------------------------------------------ publishing helpers

    def publish_event(self, ev: Dict[str, Any]) -> None:
        payload: Dict[str, Any]
        try:
            embed = self._build_trade_embed(ev)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to format Discord embed: %s", exc)
            embed = None
        route = self._route_for_event(ev)

        if embed:
            payload = {"embeds": [embed]}
        else:
            payload = {
                "content": f"{ev.get('side', '?')} {ev.get('type', '?')} • p={self._fmt_prob(ev.get('prob'))} • grade={ev.get('grade', '')}"
            }
        self._post(payload, route=route)

    # Alias used by existing CLI code
    publish = publish_event

    def publish_text(
        self,
        content: str,
        *,
        dedupe_key: Optional[str] = None,
        route: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not content.strip():
            return
        self._post({"content": content}, dedupe_key=dedupe_key, kind="text", route=route)

    def publish_embed(
        self,
        *,
        title: str,
        description: str,
        fields: Optional[list[Dict[str, Any]]] = None,
        color: Optional[int] = None,
        footer: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
        dedupe_key: Optional[str] = None,
        route: Optional[Dict[str, Any]] = None,
    ) -> None:
        embed: Dict[str, Any] = {
            "title": title,
            "description": description,
            "fields": fields or [],
        }
        if color is not None:
            embed["color"] = int(color)
        if footer:
            embed["footer"] = footer
        if timestamp is not None:
            if isinstance(timestamp, datetime):
                embed["timestamp"] = timestamp.isoformat()
            else:
                embed["timestamp"] = str(timestamp)

        payload = {
            "embeds": [embed]
        }
        self._post(payload, dedupe_key=dedupe_key, kind="embed", route=route)

    # ------------------------------------------------------------------ formatting helpers

    @staticmethod
    def _fmt_price(value: Any) -> str:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return "—"
        if not math.isfinite(num):
            return "—"
        return f"{num:.2f}"

    @staticmethod
    def _fmt_prob(value: Any) -> str:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(num):
            return "n/a"
        return f"{num:.1%}"

    @staticmethod
    def _fmt_int(value: Any) -> str:
        try:
            num = int(value)
        except (TypeError, ValueError):
            return "—"
        return str(num)

    @staticmethod
    def _coerce_dt(ts: Any) -> Optional[datetime]:
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if hasattr(ts, "to_pydatetime"):
            try:
                return ts.to_pydatetime()
            except Exception:
                pass
        try:
            txt = str(ts).strip()
        except Exception:
            return None
        if not txt:
            return None
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(txt)
        except Exception:
            return None

    @staticmethod
    def _denver_naive(ts: Any) -> Optional[datetime]:
        base = DiscordEmitter._coerce_dt(ts)
        if base is None:
            return None
        if base.tzinfo is not None:
            base = base.astimezone(MT_TZ).replace(tzinfo=None)
        return base

    @staticmethod
    def _format_bar(ts: Any) -> str:
        dt_obj = DiscordEmitter._denver_naive(ts)
        if dt_obj is None:
            return "—"
        return dt_obj.strftime(BAR_FMT)

    @staticmethod
    def _fmt_short_time(ts: Any) -> str:
        dt_obj = DiscordEmitter._coerce_dt(ts)
        if dt_obj is None:
            return "—"
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=MT_TZ)
        else:
            dt_obj = dt_obj.astimezone(MT_TZ)
        return dt_obj.strftime("%H:%M")

    @staticmethod
    def _gate_icon(flag: Any) -> str:
        return "✅" if bool(flag) else "⚠️"

    def _build_trade_embed(self, ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        side = (ev.get("side") or "").upper()
        action = (ev.get("type") or "").upper()
        if not action:
            return None

        dt_bar = self._denver_naive(ev.get("datetime"))
        dt_for_embed = dt_bar.replace(tzinfo=MT_TZ) if dt_bar else None
        dt_display = self._format_bar(ev.get("datetime"))

        color_map = {
            ("OPEN", "LONG"): 0x2ecc71,
            ("OPEN", "SHORT"): 0xe74c3c,
            ("FLIP", "LONG"): 0x27ae60,
            ("FLIP", "SHORT"): 0xc0392b,
        }
        if action in ("CLOSE", "EXIT"):
            color = 0xbdc3c7
        else:
            color = color_map.get((action, side), 0x3498db if side != "SHORT" else 0xe74c3c)

        grade = ev.get("grade") or "—"
        prob_txt = self._fmt_prob(ev.get("prob"))
        tier = ev.get("tier") or ev.get("tier_name")
        instrument = ev.get("instrument") or ev.get("symbol")
        preset = ev.get("preset")
        market_mode = ev.get("market_mode") or ev.get("mode")
        mode_txt = market_mode.title() if isinstance(market_mode, str) and market_mode else None

        if action in ("CLOSE", "EXIT"):
            title_icon = "⚪"
        else:
            title_icon = "🟢" if side == "LONG" else ("🔴" if side == "SHORT" else "⚪")
        title_parts = [title_icon, action]
        if side:
            title_parts.append(side)
        if instrument:
            title_parts.append(str(instrument))
        title = " ".join(title_parts).strip()

        header_sections = []
        if tier:
            header_sections.append(f"Tier {tier}")
        if preset:
            header_sections.append(f"Preset {preset}")
        if mode_txt:
            header_sections.append(f"Mode {mode_txt}")
        header_sections.append(f"Grade {grade}")
        if dt_bar:
            header_sections.append(f"{dt_display} MT")
        header_sections.append(f"Prob {prob_txt}")
        description = " • ".join(s for s in header_sections if s)

        risk = ev.get("risk") or {}
        ctx = ev.get("ctx") or {}
        if not isinstance(ctx, dict):
            ctx = {}
        gates = ev.get("gates") or {
            "prob": ctx.get("gate_prob"),
            "vwap": ctx.get("gate_vwap"),
            "ema": ctx.get("gate_ema"),
            "tod": ctx.get("gate_tod"),
        }
        hold_text = ev.get("hold_text") or ctx.get("hold_est_text") or ""
        bars_in_trade = ev.get("bars_in_trade") or ctx.get("bars_in_trade")
        size_hint = ev.get("size_hint") or ctx.get("size_hint")
        success_prob = ev.get("success_prob") or ctx.get("success_prob")
        r_multiple = ev.get("r_multiple")

        confidence_txt = self._fmt_prob(success_prob if success_prob is not None else ev.get("prob"))
        try:
            r_value = float(r_multiple) if r_multiple is not None else None
            if r_value is not None and math.isfinite(r_value):
                r_txt = f"{r_value:.2f}"
            else:
                r_txt = "—"
        except Exception:
            r_txt = "—"

        signal_parts = [
            f"Side {side or '—'}",
            f"Entry {self._fmt_price(ev.get('price'))}",
            f"Stop {self._fmt_price(risk.get('stop'))}",
            f"Target {self._fmt_price(risk.get('target'))}",
            f"R {r_txt}",
            f"Conf {confidence_txt}",
        ]
        signal_field = {"name": "Signal", "value": " | ".join(signal_parts), "inline": False}

        stats_parts = []
        if size_hint is not None:
            stats_parts.append(f"Size {self._fmt_int(size_hint)}")
        if bars_in_trade is not None:
            stats_parts.append(f"Bars {int(bars_in_trade)}")
        if hold_text:
            stats_parts.append(str(hold_text))
        stats_value = " | ".join(stats_parts) if stats_parts else "—"
        stats_field = {"name": "Management Stats", "value": stats_value, "inline": False}

        management = ev.get("notes")
        narrative = ev.get("narrative")

        policy_info = ev.get("policy") or {}
        policy_flags = ev.get("policy_flags") or []
        block_reason = ev.get("policy_block")

        policy_summary_lines: list[str] = []
        if policy_info or policy_flags or block_reason:
            news_info = policy_info.get("news_blackout") or {}
            next_news = policy_info.get("news_next") or {}
            if news_info.get("active"):
                news_label = news_info.get("label") or "News"
                news_end = self._fmt_short_time(news_info.get("end"))
                policy_summary_lines.append(f"News blackout: {news_label} → {news_end}")
            else:
                next_title = next_news.get("title")
                if next_title:
                    next_time = self._fmt_short_time(next_news.get("time"))
                    policy_summary_lines.append(f"Next news {next_title} @ {next_time}")
                else:
                    policy_summary_lines.append("News: clear")

            trend_bits = [
                f"VWAP {self._gate_icon(gates.get('vwap'))}",
                f"EMA {self._gate_icon(gates.get('ema'))}",
            ]
            try:
                if ev.get("trend_score") is not None:
                    trend_bits.append(f"score {float(ev['trend_score']):.2f}")
            except Exception:
                pass
            policy_summary_lines.append("Trend: " + " | ".join(trend_bits))

            realized_r = policy_info.get("realized_R_today")
            if realized_r is not None:
                try:
                    usd_today = policy_info.get("realized_usd_today")
                    losses_today = policy_info.get("losses_today")
                    pl_line = f"P/L {float(realized_r):.2f}R"
                    if usd_today is not None:
                        pl_line += f" ${float(usd_today):.0f}"
                    if losses_today is not None:
                        pl_line += f" losses {int(losses_today)}"
                    policy_summary_lines.append(pl_line)
                except Exception:
                    pass

            if isinstance(policy_flags, list) and policy_flags:
                flag_parts: list[str] = []
                for item in policy_flags:
                    if isinstance(item, dict):
                        code = item.get("code") or "POLICY"
                        detail = item.get("detail")
                    else:
                        code = str(item)
                        detail = None
                    if detail:
                        flag_parts.append(f"{code}:{detail}")
                    else:
                        flag_parts.append(code)
                if flag_parts:
                    policy_summary_lines.append("Guards: " + "; ".join(flag_parts))

            stop_pts = policy_info.get("hard_stop_pts")
            target_pts = policy_info.get("target1_pts")
            trail_mult = policy_info.get("trail_atr_mult")
            stop_parts: list[str] = []
            try:
                if stop_pts is not None:
                    stop_parts.append(f"stop {float(stop_pts):.2f}")
                if target_pts is not None:
                    stop_parts.append(f"T1 {float(target_pts):.2f}")
                if trail_mult is not None:
                    stop_parts.append(f"trail x{float(trail_mult):.2f}")
            except Exception:
                stop_parts = []
            if stop_parts:
                policy_summary_lines.append("ATR: " + " | ".join(stop_parts))

            action = policy_info.get("action")
            reason = policy_info.get("reason")
            if action in {"downsize", "lockout", "block"}:
                action_line = f"Decision: {action}"
                if action == "downsize" and policy_info.get("size_multiplier") not in (None, 1.0):
                    try:
                        action_line += f" ×{float(policy_info['size_multiplier']):.2f}"
                    except Exception:
                        pass
                if reason:
                    action_line += f" ({reason})"
                policy_summary_lines.append(action_line)
            elif reason and action == "allow":
                policy_summary_lines.append(f"Reason: {reason}")

            details_list = policy_info.get("details")
            if isinstance(details_list, list):
                for detail in details_list[:3]:
                    policy_summary_lines.append(detail)

            if block_reason and all("Blocked" not in line for line in policy_summary_lines):
                policy_summary_lines.append(f"Blocked: {block_reason}")

        gates_line = " • ".join([
            f"Prob {self._gate_icon(gates.get('prob'))}",
            f"VWAP {self._gate_icon(gates.get('vwap'))}",
            f"EMA {self._gate_icon(gates.get('ema'))}",
            f"TOD {self._gate_icon(gates.get('tod'))}",
        ])

        fields = [signal_field, stats_field]
        if isinstance(management, str) and management.strip():
            fields.append({"name": "Management", "value": management.strip(), "inline": False})
        if isinstance(narrative, str) and narrative.strip():
            fields.append({"name": "Playbook", "value": narrative.strip(), "inline": False})
        if policy_summary_lines:
            fields.append({"name": "Policy", "value": "\n".join(policy_summary_lines), "inline": False})
        fields.append({"name": "Gates", "value": gates_line, "inline": False})

        embed: Dict[str, Any] = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
        }

        footer_bits = []
        run_name = ev.get("run_name")
        if run_name is None and isinstance(ctx, dict):
            run_name = ctx.get("run_name") or ctx.get("model_label")
        if run_name:
            footer_bits.append(str(run_name))
        run_id = ev.get("run_id")
        if run_id:
            footer_bits.append(str(run_id))
        model_run_id = ev.get("model_run_id")
        if model_run_id is None and isinstance(ctx, dict):
            model_run_id = ctx.get("model_run_id")
        if model_run_id is not None:
            footer_bits.append(f"model_run_id {model_run_id}")
        version = ev.get("version")
        if version:
            try:
                footer_bits.append(f"model {str(version)[:8]}")
            except Exception:
                footer_bits.append(f"model {version}")
        if footer_bits:
            embed["footer"] = {"text": " • ".join(filter(None, footer_bits))}
        if dt_for_embed:
            embed["timestamp"] = dt_for_embed.isoformat()
        return embed

    # ------------------------------------------------------------------ internal utils

    def _route_for_event(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        event_type = str(ev.get("event_type") or "").strip().lower()
        if not event_type:
            action = str(ev.get("type") or "").strip().upper()
            if action in {"OPEN", "FLIP"}:
                event_type = "signal"
            elif action in {"CLOSE", "EXIT"}:
                event_type = "signal_recap"
            elif action in {"LOCKOUT", "BLOCK"}:
                event_type = "lockout"
            else:
                event_type = "signal"
        audience = str(ev.get("audience") or ("ops" if event_type in {"lockout", "alert", "health_update", "audit_event"} else "pro")).strip().lower()
        return {
            "event_type": event_type,
            "instrument": ev.get("instrument") or ev.get("symbol"),
            "audience": audience,
            "channel_key": ev.get("channel_key"),
            "channel_keys": ev.get("channel_keys"),
            "include_recap": bool(ev.get("include_recap")),
        }

    def _resolve_destinations(self, route: Optional[Dict[str, Any]]) -> list[str]:
        route = dict(route or {})
        if not route:
            return []
        return self._routes.route_channels(
            event_type=route.get("event_type"),
            instrument=route.get("instrument"),
            audience=str(route.get("audience") or "pro"),
            channel_key=route.get("channel_key"),
            channel_keys=route.get("channel_keys"),
            include_recap=bool(route.get("include_recap")),
        )

    def _log_route_readiness(self, *, ready: bool, destinations: list[str], route: Optional[Dict[str, Any]]) -> None:
        has_token = bool(self._discord_token)
        state = "ready" if ready else ("missing_token" if not has_token else "missing_routes")
        if state == self._route_readiness_state:
            return
        self._route_readiness_state = state
        route_type = ""
        try:
            route_type = str((route or {}).get("event_type") or "")
        except Exception:
            route_type = ""
        logger.info(
            "DISCORD_ROUTE_STATUS|state=%s|mode=%s|has_token=%s|destinations=%d|event_type=%s",
            state,
            str(getattr(self._routes, "mode", "unknown")),
            str(has_token).lower(),
            len(destinations),
            route_type,
        )

    def _post(
        self,
        payload: Dict[str, Any],
        *,
        dedupe_key: Optional[str] = None,
        kind: str = "webhook",
        route: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._client is None:
            if not self._warned_no_client:
                logger.warning("Discord client unavailable; skipping webhook publish.")
                self._warned_no_client = True
            return

        envelope = {"allowed_mentions": {"parse": []}}
        if self.username:
            envelope["username"] = self.username
        envelope.update(payload)
        routing_key = dedupe_key or payload.get("dedupe_key")

        if self.url:
            self._delivery_queue.submit(
                channel_id="webhook",
                kind=kind,
                payload={"transport": "webhook", "body": envelope},
                dedupe_key=routing_key,
            )
            return

        destinations = self._resolve_destinations(route)
        self._log_route_readiness(ready=bool(destinations and self._discord_token), destinations=destinations, route=route)
        if not destinations or not self._discord_token:
            reason_key = "missing_token" if not self._discord_token else "missing_routes"
            event_type = ""
            try:
                event_type = str((route or {}).get("event_type") or "")
            except Exception:
                event_type = ""
            now_ts = time.time()
            reason_changed = str(reason_key) != str(self._warned_no_route_reason or "")
            can_emit = reason_changed or (not self._warned_no_route) or ((now_ts - float(self._no_route_warn_last_ts)) >= float(self._no_route_warn_interval_sec))
            if can_emit:
                suppressed = int(self._no_route_warn_suppressed_count)
                suppressed_missing_token = int(self._no_route_warn_by_reason.get("missing_token", 0) or 0)
                suppressed_missing_routes = int(self._no_route_warn_by_reason.get("missing_routes", 0) or 0)
                if suppressed > 0:
                    logger.warning(
                        (
                            "Discord routes/token unavailable; skipping channel publish. "
                            "(reason=%s mode=%s event_type=%s suppressed_total=%d suppressed_missing_token=%d suppressed_missing_routes=%d)"
                        ),
                        reason_key,
                        str(getattr(self._routes, "mode", "unknown")),
                        event_type,
                        suppressed,
                        suppressed_missing_token,
                        suppressed_missing_routes,
                    )
                else:
                    logger.warning(
                        "Discord routes/token unavailable; skipping channel publish. (reason=%s mode=%s event_type=%s)",
                        reason_key,
                        str(getattr(self._routes, "mode", "unknown")),
                        event_type,
                    )
                self._no_route_warn_last_ts = now_ts
                self._no_route_warn_suppressed_count = 0
                self._no_route_warn_by_reason = {"missing_token": 0, "missing_routes": 0}
                self._warned_no_route = True
                self._warned_no_route_reason = reason_key
            else:
                self._no_route_warn_suppressed_count += 1
                self._no_route_warn_by_reason[reason_key] = int(self._no_route_warn_by_reason.get(reason_key, 0) or 0) + 1
            return

        suppressed_total = int(self._no_route_warn_suppressed_count)
        suppressed_missing_token = int(self._no_route_warn_by_reason.get("missing_token", 0) or 0)
        suppressed_missing_routes = int(self._no_route_warn_by_reason.get("missing_routes", 0) or 0)
        if suppressed_total > 0 or self._warned_no_route:
            logger.info(
                (
                    "Discord route/token availability restored. "
                    "(suppressed_total=%d suppressed_missing_token=%d suppressed_missing_routes=%d mode=%s)"
                ),
                suppressed_total,
                suppressed_missing_token,
                suppressed_missing_routes,
                str(getattr(self._routes, "mode", "unknown")),
            )
        self._warned_no_route = False
        self._warned_no_route_reason = None
        self._no_route_warn_suppressed_count = 0
        self._no_route_warn_by_reason = {"missing_token": 0, "missing_routes": 0}
        for channel_id in destinations:
            channel_envelope = dict(envelope)
            channel_envelope.pop("username", None)
            channel_key = f"{routing_key}:{channel_id}" if routing_key else None
            self._delivery_queue.submit(
                channel_id=channel_id,
                kind=kind,
                payload={"transport": "channel", "channel_id": channel_id, "body": channel_envelope},
                dedupe_key=channel_key,
            )

    def _send_envelope_direct(self, envelope: Dict[str, Any]) -> None:
        transport = str(envelope.get("transport") or "webhook").strip().lower()
        body = envelope.get("body") if isinstance(envelope.get("body"), dict) else envelope
        kind, client = self._client
        try:
            if transport == "channel":
                channel_id = str(envelope.get("channel_id") or "").strip()
                if not channel_id:
                    raise DiscordDeliveryError("channel_id_missing", retryable=False)
                if not self._discord_token:
                    raise DiscordDeliveryError("discord_token_missing", retryable=False)
                url = f"https://discord.com/api/v10/channels/{quote(channel_id)}/messages"
                headers = {"Authorization": f"Bot {self._discord_token}"}
                if kind == "httpx":
                    resp = client.post(url, json=body, headers=headers)
                else:
                    resp = client.post(url, json=body, headers=headers, timeout=10)
            elif kind == "httpx":
                resp = client.post(self.url, json=body)
            else:
                resp = client.post(self.url, json=body, timeout=10)
        except Exception as exc:
            raise DiscordDeliveryError(str(exc), retryable=True) from exc
        status = getattr(resp, "status_code", 0)
        body = getattr(resp, "text", "")
        if status == 429:
            retry_after = 0.5
            try:
                data = json.loads(body)
                retry_after = float(data.get("retry_after", retry_after))
            except Exception:
                pass
            raise DiscordDeliveryError(
                "discord_rate_limited",
                retry_after=max(0.1, retry_after),
                retryable=True,
                status_code=status,
                body=body[:300],
            )
        if status in {408, 425} or status >= 500:
            raise DiscordDeliveryError(
                f"discord_transient_http_{status}",
                retryable=True,
                status_code=status,
                body=body[:300],
            )
        if status < 200 or status >= 300:
            raise DiscordDeliveryError(
                f"discord_http_{status}",
                retryable=False,
                status_code=status,
                body=body[:300],
            )

    @staticmethod
    def _fmt(value: Any) -> str:
        if value is None:
            return "—"
        try:
            if isinstance(value, (int, float)):
                num = float(value)
                if not math.isfinite(num):
                    return "—"
                return f"{num:.2f}"
            return str(value)
        except Exception:
            return str(value)


__all__ = ["DiscordEmitter"]
