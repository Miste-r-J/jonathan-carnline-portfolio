from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import discord
from discord import AllowedMentions, SyncWebhook, app_commands
from discord.ext import commands, tasks

from .config import PlannerConfig, PlannerTargetConfig
from .news import build_news_summary
from .planner import PlanPayload, build_plan_payload


@dataclass
class TargetState:
    message_id: Optional[int] = None
    via_webhook: bool = False
    backfill_attempted: bool = False


@dataclass(frozen=True)
class TargetDispatchFailure:
    key: str
    reason: str
    detail: str


class PlannerTargetPublishError(RuntimeError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class PremarketPlannerBot(commands.Bot):
    """
    Discord bot that publishes the premarket plan on schedule or via manual triggers.
    """

    def __init__(self, config: PlannerConfig) -> None:
        intents = discord.Intents.default()

        super().__init__(command_prefix="!", intents=intents)
        self.config = config
        self._log = logging.getLogger("premarket_planner.bot")
        self._publish_lock = asyncio.Lock()
        self._channel_cache: Dict[int, discord.abc.Messageable] = {}
        self._startup_dispatched = False
        self._webhook_client: Optional[SyncWebhook] = None
        self._target_states: Dict[str, TargetState] = {
            target.key: TargetState() for target in self.config.targets
        }
        interval_minutes = max(float(self.config.dispatch_interval_minutes), 1.0)
        self._dispatch_interval_minutes = interval_minutes
        self._dispatch_interval = timedelta(minutes=interval_minutes)
        self._last_run_at: Optional[datetime] = None

        # Prime the scheduler with defaults; actual cadence is applied in setup_hook.
        self.periodic_dispatch.change_interval(minutes=1.0)

        trigger_name = (self.config.command_trigger.strip().lstrip("/") or "ping").lower()
        self._slash_command_name = trigger_name

        existing = self.tree.get_command(trigger_name)
        if existing is not None:
            self.tree.remove_command(trigger_name, type=app_commands.Command)

        async def _planner_ping(interaction: discord.Interaction) -> None:
            await self._handle_ping_command(interaction)

        self._slash_command = app_commands.Command(
            name=trigger_name,
            description="Send the latest premarket plan to this channel",
            callback=_planner_ping,
        )
        self.tree.add_command(self._slash_command)
        self._register_news_command()

    def _register_news_command(self) -> None:
        existing = self.tree.get_command("news")
        if existing is not None:
            self.tree.remove_command("news", type=app_commands.Command)
        self.tree.add_command(
            app_commands.Command(
                name="news",
                description="Show scheduled news-risk events from the automation news source",
                callback=self._handle_news_command,
            )
        )

    async def _handle_news_command(self, interaction: discord.Interaction) -> None:
        if not self._is_authorized_user(interaction.user):
            await self._reply_interaction(
                interaction,
                "⛔ You are not authorized to use this command.",
                ephemeral=True,
            )
            return
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True, thinking=True)
        summary = await asyncio.to_thread(build_news_summary, self.config)
        await self._reply_interaction(interaction, summary[:1900], ephemeral=True)

    async def setup_hook(self) -> None:
        """
        Called by discord.py once the loop is running. Configure the schedule and start the task.
        """
        if not self.config.enabled:
            self._log.warning("Premarket planner disabled via config; scheduler not started.")
            return
        if not any(target.channel_id for target in self.config.targets) and not self.config.discord_channel_id and not self.config.webhook_url:
            self._log.error("No channel ID or webhook configured; scheduler not started.")
            return

        if not self.periodic_dispatch.is_running():
            self.periodic_dispatch.start()
            self._log.info(
                "Premarket planner cadence set to start at %s %s and repeat every %.1f minutes",
                self.config.run_time.strftime("%H:%M"),
                self.config.session_tz,
                self._dispatch_interval_minutes,
            )
        try:
            await self.tree.sync()
        except Exception as exc:  # pragma: no cover - safety
            self._log.warning("Unable to sync application commands: %s", exc)

    async def on_ready(self) -> None:
        user = self.user
        if user:
            self._log.info("Connected to Discord as %s (%s)", user.name, user.id)
        else:
            self._log.info("Connected to Discord.")

        try:
            for guild in self.guilds:
                await self.tree.sync(guild=guild)
        except Exception as exc:  # pragma: no cover - defensive
            self._log.warning("Unable to sync guild commands: %s", exc)

        for target_cfg in self.config.targets:
            if not target_cfg.channel_id:
                continue
            channel = await self._resolve_channel(target_cfg.channel_id)
            if channel is None:
                self._log.error("Unable to resolve planner target %s channel %s", target_cfg.key, target_cfg.channel_id)
            else:
                self._log.debug("Resolved planner target %s channel %s (%s)", target_cfg.key, channel, channel.id)

        if not self._startup_dispatched:
            success = await self._dispatch_plan(trigger="startup")
            if success:
                self._last_run_at = datetime.now(self.config.timezone)
            self._startup_dispatched = True

    @tasks.loop(minutes=1)
    async def periodic_dispatch(self) -> None:
        now = datetime.now(self.config.timezone)
        if self.config.skip_weekends and now.weekday() >= 5:
            return

        run_time = self.config.run_time
        run_dt = datetime.combine(now.date(), run_time, tzinfo=self.config.timezone)
        if now < run_dt:
            return

        if self._last_run_at is not None:
            if self._last_run_at.date() != now.date():
                self._last_run_at = None
            elif now - self._last_run_at < self._dispatch_interval:
                return

        success = await self._dispatch_plan(trigger="schedule")
        if success:
            self._last_run_at = now

    def _is_authorized_user(self, user: Optional[discord.abc.User]) -> bool:
        if not self.config.authorized_users:
            return True
        if user is None:
            return False

        tokens = {str(user.id)}
        name = getattr(user, "name", None)
        if isinstance(name, str):
            tokens.add(name.lower())
        global_name = getattr(user, "global_name", None)
        if isinstance(global_name, str):
            tokens.add(global_name.lower())
        display_name = getattr(user, "display_name", None)
        if isinstance(display_name, str):
            tokens.add(display_name.lower())

        allowed = set(self.config.authorized_users)
        return any(token in allowed for token in tokens)

    async def _handle_ping_command(self, interaction: discord.Interaction) -> None:
        if not self.config.enabled:
            await self._reply_interaction(interaction, "⚠️ Premarket planner is disabled.")
            return

        allowed_channels = {
            int(channel_id)
            for channel_id in [self.config.discord_channel_id, *(target.channel_id for target in self.config.targets)]
            if channel_id is not None
        }
        if allowed_channels and interaction.channel_id not in allowed_channels:
            await self._reply_interaction(
                interaction,
                "⛔ This command only works in the planner channel.",
            )
            return

        if not self._is_authorized_user(interaction.user):
            await self._reply_interaction(
                interaction,
                "⛔ You are not authorized to use this command.",
                ephemeral=True,
            )
            return

        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True)

        try:
            await self._dispatch_plan(
                trigger=f"/{self._slash_command_name}",
                interaction=interaction,
            )
        except Exception as exc:
            self._log.exception("Slash command dispatch failed")
            await self._reply_interaction(
                interaction,
                f"⚠️ Unable to refresh planner: {exc!s}",
            )

    async def _dispatch_plan(
        self,
        *,
        trigger: str,
        interaction: Optional[discord.Interaction] = None,
    ) -> bool:
        if not self.config.enabled:
            self._log.info("Planner disabled; ignoring %s trigger", trigger)
            return False
        if not any(target.channel_id for target in self.config.targets) and self.config.discord_channel_id is None and not self.config.webhook_url:
            self._log.error("No target channel available for %s trigger", trigger)
            return False

        if self._publish_lock.locked():
            self._log.warning("Publish already in progress; skipping %s trigger", trigger)
            return False

        mention = f"<@&{self.config.mention_role_id}>" if self.config.mention_role_id else None
        successes = 0
        failure_message: Optional[str] = None
        target_failures: list[TargetDispatchFailure] = []

        async with self._publish_lock:
            for index, target_cfg in enumerate(self.config.targets):
                target_state = self._target_states.setdefault(target_cfg.key, TargetState())
                try:
                    payload: PlanPayload = await asyncio.to_thread(
                        build_plan_payload,
                        self.config,
                        target_cfg,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    self._log.exception(
                        "Failed to build premarket plan for %s on %s trigger",
                        target_cfg.key,
                        trigger,
                    )
                    if failure_message is None:
                        failure_message = f"⚠️ Unable to refresh {target_cfg.instrument}: {exc!s}"
                    continue

                embed = payload.embed.to_discord_embed()
                if trigger.startswith("/"):
                    mention_to_send = None
                elif target_state.message_id is None and index == 0:
                    mention_to_send = mention
                else:
                    mention_to_send = None

                channel_obj = await self._resolve_channel(target_cfg.channel_id or self.config.discord_channel_id)
                try:
                    await self._publish_plan(channel_obj, target_cfg, mention_to_send, embed)
                except PlannerTargetPublishError as exc:
                    target_failures.append(
                        TargetDispatchFailure(
                            key=target_cfg.key,
                            reason=exc.reason,
                            detail=exc.detail,
                        )
                    )
                    self._log.warning(
                        "Planner target %s failed on %s trigger (%s): %s",
                        target_cfg.key,
                        trigger,
                        exc.reason,
                        exc.detail,
                    )
                    if failure_message is None:
                        failure_message = f"⚠️ Unable to refresh {target_cfg.instrument}: {exc.detail}"
                    continue
                successes += 1

        if interaction is not None:
            if successes and target_failures:
                await self._reply_interaction(
                    interaction,
                    f"⚠️ Premarket plans refreshed partially. Failed targets: {self._format_failed_targets(target_failures)}.",
                    ephemeral=True,
                )
            elif successes:
                await self._reply_interaction(
                    interaction,
                    "✅ Premarket plans refreshed in the planner channel.",
                    ephemeral=True,
                )
            elif failure_message is not None:
                await self._reply_interaction(interaction, failure_message)
            else:
                await self._reply_interaction(
                    interaction,
                    "⚠️ Premarket plan refresh failed.",
                )

        self._log.info(
            "Premarket plan dispatch completed (%s) — %d targets updated, %d failed",
            trigger,
            successes,
            len(target_failures),
        )
        return successes > 0

    def _format_failed_targets(self, failures: list[TargetDispatchFailure]) -> str:
        parts: list[str] = []
        for failure in failures:
            parts.append(f"{failure.key} ({failure.reason})")
        return ", ".join(parts[:3])

    async def _resolve_channel(self, channel_id: Optional[int]) -> Optional[discord.abc.Messageable]:
        if not channel_id:
            return None
        cached = self._channel_cache.get(channel_id)
        if cached is not None:
            return cached

        channel = self.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(channel_id)
            except discord.NotFound:
                self._log.error("Planner channel %s not found", channel_id)
                return None
            except discord.Forbidden:
                self._log.error("Insufficient permissions to fetch channel %s", channel_id)
                return None
            except discord.HTTPException as exc:
                self._log.error("HTTP error fetching channel %s: %s", channel_id, exc)
                return None

        if isinstance(channel, discord.abc.Messageable):
            self._channel_cache[channel_id] = channel
            return channel

        self._log.error("Resolved channel %s is not messageable (%s)", channel_id, type(channel).__name__)
        return None

    async def _reply_interaction(
        self,
        interaction: discord.Interaction,
        content: str,
        *,
        ephemeral: bool = True,
    ) -> None:
        try:
            if interaction.response.is_done():
                await interaction.followup.send(content, ephemeral=ephemeral)
            else:
                await interaction.response.send_message(content, ephemeral=ephemeral)
        except Exception as exc:  # pragma: no cover - defensive
            self._log.warning("Failed to send interaction reply: %s", exc)

    async def _publish_plan(
        self,
        channel: Optional[discord.abc.Messageable],
        target_cfg: PlannerTargetConfig,
        content: Optional[str],
        embed: discord.Embed,
    ) -> None:
        state = self._target_states.setdefault(target_cfg.key, TargetState())
        allowed = AllowedMentions(roles=True) if content else AllowedMentions.none()

        if channel is not None:
            if state.message_id and not state.via_webhook:
                try:
                    message = await channel.fetch_message(state.message_id)
                except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                    state.message_id = None
                else:
                    try:
                        await message.edit(content=content, embed=embed)
                    except discord.Forbidden as exc:
                        raise PlannerTargetPublishError("missing access", "Missing access to edit planner message") from exc
                    except discord.NotFound as exc:
                        raise PlannerTargetPublishError("not found", "Existing planner message was not found") from exc
                    except discord.HTTPException as exc:
                        raise PlannerTargetPublishError("discord http error", str(exc)) from exc
                    return

            if not state.via_webhook and not state.backfill_attempted:
                state.backfill_attempted = True
                existing = await self._locate_existing_message(channel, embed)
                if existing is not None:
                    try:
                        await existing.edit(content=content, embed=embed)
                    except discord.Forbidden as exc:
                        raise PlannerTargetPublishError("missing access", "Missing access to edit planner history message") from exc
                    except discord.NotFound as exc:
                        raise PlannerTargetPublishError("not found", "Planner history message was not found") from exc
                    except discord.HTTPException as exc:
                        raise PlannerTargetPublishError("discord http error", str(exc)) from exc
                    state.message_id = existing.id
                    state.via_webhook = False
                    return

            try:
                message = await channel.send(content=content, embed=embed, allowed_mentions=allowed)
            except discord.Forbidden as exc:
                raise PlannerTargetPublishError("missing access", "Missing access to send planner message") from exc
            except discord.NotFound as exc:
                raise PlannerTargetPublishError("not found", "Planner channel was not found") from exc
            except discord.HTTPException as exc:
                raise PlannerTargetPublishError("discord http error", str(exc)) from exc
            state.message_id = message.id
            state.via_webhook = False
            return

        if self.config.webhook_url:
            webhook = self._get_webhook()

            if state.message_id and state.via_webhook:
                try:
                    await asyncio.to_thread(
                        webhook.edit_message,
                        message_id=state.message_id,
                        content=content,
                        embed=embed,
                        allowed_mentions=allowed,
                    )
                except discord.NotFound as exc:
                    raise PlannerTargetPublishError("not found", "Webhook planner message was not found") from exc
                except discord.Forbidden as exc:
                    raise PlannerTargetPublishError("missing access", "Missing access to edit webhook planner message") from exc
                except discord.HTTPException as exc:
                    raise PlannerTargetPublishError("discord http error", str(exc)) from exc
                return

            try:
                message = await asyncio.to_thread(
                    webhook.send,
                    content=content,
                    embed=embed,
                    allowed_mentions=allowed,
                    username=self.config.webhook_username or "Premarket Planner",
                    wait=True,
                )
            except discord.Forbidden as exc:
                raise PlannerTargetPublishError("missing access", "Missing access to send planner webhook message") from exc
            except discord.NotFound as exc:
                raise PlannerTargetPublishError("not found", "Planner webhook destination was not found") from exc
            except discord.HTTPException as exc:
                raise PlannerTargetPublishError("discord http error", str(exc)) from exc
            state.message_id = message.id
            state.via_webhook = True
            return

        raise PlannerTargetPublishError("unresolved", "No target channel or webhook available")

    async def _locate_existing_message(
        self,
        channel: discord.abc.Messageable,
        embed: discord.Embed,
        limit: int = 20,
    ) -> Optional[discord.Message]:
        me = self.user
        if me is None:
            return None
        title = embed.title if embed else None

        try:
            # type: ignore[attr-defined] - TextChannel has history()
            history = channel.history(limit=limit)  # pragma: no cover - network call
        except AttributeError:
            return None
        except discord.Forbidden:
            self._log.warning(
                "Missing permissions to read history in channel %s",
                getattr(channel, "id", "unknown"),
            )
            return None
        except discord.HTTPException as exc:
            self._log.warning(
                "HTTP error while fetching history for channel %s: %s",
                getattr(channel, "id", "unknown"),
                exc,
            )
            return None

        try:
            async for message in history:  # pragma: no cover - network call
                if message.author.id != me.id:
                    continue
                if message.embeds and title:
                    if message.embeds[0].title == title:
                        return message
                else:
                    return message
        except discord.Forbidden:
            self._log.warning(
                "Lost permission to read history in channel %s during scan",
                getattr(channel, "id", "unknown"),
            )
        except discord.HTTPException as exc:
            self._log.warning(
                "HTTP error while scanning history for channel %s: %s",
                getattr(channel, "id", "unknown"),
                exc,
            )
        return None

    async def _send_failure(
        self,
        interaction: Optional[discord.Interaction],
        target: Optional[discord.abc.Messageable],
        content: str,
    ) -> None:
        if interaction is not None:
            await self._reply_interaction(interaction, content)
            return
        if target is not None:
            await target.send(content)
            return
        if self.config.webhook_url:
            webhook = self._get_webhook()
            await asyncio.to_thread(
                webhook.send,
                content=content,
                username=self.config.webhook_username or "Premarket Planner",
            )

    def _get_webhook(self) -> SyncWebhook:
        if self._webhook_client is None:
            self._webhook_client = SyncWebhook.from_url(self.config.webhook_url)
        return self._webhook_client
