"""Discord voice bot integration (prepared, not enabled).

See README.md in this folder for the plan. Nothing here runs unless a bot
token is configured *and* ``discord.py`` is installed.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ...state import AppState


def status(state: "AppState") -> dict:
    installed = importlib.util.find_spec("discord") is not None
    return {
        "enabled": False,
        "installed": installed,
        "token_configured": state.secrets.has("discord"),
        "guild_id": state.db.get_setting("discord_guild_id"),
        "voice_channel_id": state.db.get_setting("discord_voice_channel_id"),
        "note": "Discord bot support is scaffolded but not wired up yet.",
    }
