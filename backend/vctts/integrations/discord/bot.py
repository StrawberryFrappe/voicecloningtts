"""Skeleton for a Discord voice bot that speaks with the cloned voice.

Not wired into the app yet. Outline of the finished version:

1. ``pip install -e backend[discord]`` and store the bot token
   (Settings → secrets name ``discord``, or the DISCORD_BOT_TOKEN env var).
2. On app startup, if enabled, run :class:`VoiceBot` on the app's asyncio loop.
3. The bot joins the configured voice channel and plays
   :class:`~vctts.integrations.discord.sink.DiscordVoiceSink` frames via a
   ``discord.AudioSource`` whose ``read()`` returns ``sink.read_frame()``.
4. Register the sink with ``state.sinks.add(sink)``. The conversation
   pipeline then speaks into Discord exactly like it speaks into the
   virtual mic. No pipeline changes needed.
5. Text commands (``!ask ...``, ``!say ...``) map to
   ``state.conversations.send(...)`` / ``speak_text(...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .sink import DiscordVoiceSink

if TYPE_CHECKING:  # pragma: no cover
    from ...state import AppState


class VoiceBot:  # pragma: no cover - placeholder
    def __init__(self, state: "AppState", sink: DiscordVoiceSink | None = None):
        self.state = state
        self.sink = sink or DiscordVoiceSink()

    async def start(self) -> None:
        raise NotImplementedError(
            "The Discord bot is not implemented yet. See vctts/integrations/discord/README.md."
        )

    async def stop(self) -> None:
        self.sink.set_enabled(False)
