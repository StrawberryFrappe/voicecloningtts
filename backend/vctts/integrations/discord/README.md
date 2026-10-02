# Discord voice bot (scaffold)

The goal is for the assistant to sit in a Discord voice channel as its own bot user,
instead of talking through your microphone.

What already exists:

- `sink.py`: `DiscordVoiceSink`. It converts TTS audio into Discord's format
  (48 kHz, 16-bit stereo, 20 ms frames) and is covered by unit tests.
- The conversation pipeline sends audio to every registered `AudioSink` through
  `SinkRouter`. A Discord sink therefore needs no pipeline changes.
- `ConversationService` and `EventBus` don't depend on the UI, so a bot can drive chats
  the same way the desktop window does.
- The secret name `discord` (or the `DISCORD_BOT_TOKEN` env var) is reserved for the bot token.
- The settings keys `discord_guild_id` and `discord_voice_channel_id` are reserved.

What's left (see the docstring in `bot.py`):

1. Create the bot in the Discord developer portal and enable the *message content*
   intent if you want text commands. Invite it with the Connect and Speak permissions.
2. `pip install -e backend[discord]`. On Windows, discord.py ships the opus DLL it needs.
3. Implement `VoiceBot.start()`:
   - connect a `discord.Client`,
   - join the configured channel,
   - `vc.play(PCMSource(sink))`, where `PCMSource.read()` returns `sink.read_frame()`.
4. Add a toggle and the IDs to the Settings page, then start the bot from `AppState.startup()`.
