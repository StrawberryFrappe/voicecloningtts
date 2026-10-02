# VoiceCloningTTS

A Windows desktop app that lets you chat with an LLM. It speaks the replies in a cloned voice and can send
that voice into a virtual microphone, mixed with your real mic, so other apps hear it.

- **Chat with any of four providers:** OpenAI, Claude (Anthropic), Google Gemini or OpenRouter, using your own API key.
  - Replies stream in.
  - Conversations are saved.
  - The chat code has a tool registry ready for web search later.
- **Personas:** each one has an editable system prompt plus answer-length tuning (Short / Normal / Long / Custom).
  It also sets temperature, reasoning effort, model and voice.
- **Voice cloning from an mp3/mp4** (or wav, m4a, ogg, webm, mkv...).
  - Pick 10-30 s of the person speaking on a waveform.
  - Save it as a named voice.
  - Switch voices anytime, and export/import voices as `.voice.zip`.
- **Two local GPU engines:**
  - **Chatterbox** (Resemble AI, MIT): Turbo/English/Multilingual, 23 languages including Spanish.
  - **XTTS-v2** (Coqui, non-commercial license): 17 languages, streaming.
- **Virtual microphone:** the app mixes your real mic with the cloned voice into **VB-Audio Virtual Cable**.
  - Discord, Zoom, OBS, games and so on just select `CABLE Output` as their microphone.
  - Optionally duck your mic while the voice talks.
  - You can hear the voice in your headphones.
- **Record → transcribe → edit → send.** The Record button captures your **microphone** or **what's playing in your
  headphones** (WASAPI loopback). It transcribes the clip with Whisper (faster-whisper on the GPU) so you can fix
  the text before sending.
- **"Say it" box:** type text and have the voice speak it directly, without the LLM.
- **Real-time voice changer:** talk into your mic and come out in the cloned voice, live, through the virtual mic.
  - Uses [Seed-VC](https://github.com/Plachtaa/seed-vc), which is zero-shot: it reuses your saved voices, with no
    training.
  - Expect about 300-450 ms of latency on an RTX 3060-class GPU.
  - TTS replies still mix on top of it.
- **Discord bot (prepared, not active):** the audio pipeline already has a Discord-ready sink. See
  [`backend/vctts/integrations/discord/README.md`](backend/vctts/integrations/discord/README.md).

## Requirements

- Windows 10/11, with an NVIDIA GPU recommended (6 GB+ VRAM). It works on CPU too, but slowly.
- [Python 3.11](https://www.python.org/downloads/) (`winget install Python.Python.3.11`)
- [Node.js LTS](https://nodejs.org/) to build the UI (`winget install OpenJS.NodeJS.LTS`)
- [VB-Audio Virtual Cable](https://vb-audio.com/Cable/), free, for the virtual microphone

## Install

```powershell
git clone https://github.com/StrawberryFrappe/voicecloningtts.git
cd voicecloningtts
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

The script:

1. creates `.venv`,
2. installs PyTorch 2.6 (CUDA 12.4), the app, Chatterbox, faster-whisper and PyAudioWPatch,
   plus XTTS-v2 in a separate `.venv-xtts`, because Chatterbox and coqui-tts need incompatible
   `transformers` versions. The app runs XTTS as a background worker process.
   - The real-time voice changer gets its own `.venv-vc`, and a pinned Seed-VC snapshot goes in `vendor\seed-vc`.
3. builds the UI,
4. checks for VB-Cable,
5. adds a desktop shortcut.

Flags: `-Cpu` (no NVIDIA GPU), `-NoXtts`, `-NoStt`, `-NoVc`.

Then start the app with **`run.bat`** (or the desktop shortcut). The first time you use an engine, it downloads its
model weights from Hugging Face (a few GB).

### Installing VB-Cable

1. Download the zip from https://vb-audio.com/Cable/ and extract it.
2. Right-click `VBCABLE_Setup_x64.exe` → **Run as administrator** → **Install Driver**. Then reboot.
3. In the app, open **Virtual mic**, then set:
   - *Your microphone* = your real mic
   - *Virtual cable* = `CABLE Input (VB-Audio Virtual Cable)`
   - *Monitor* = your headphones
4. Press **Start virtual mic**.
5. In Discord, Zoom, OBS and similar apps, set the input device to **`CABLE Output (VB-Audio Virtual Cable)`**.
   In Discord, turn off Krisp noise suppression, or it may cut the voice.

## First run / something broke?

Run the self-check. It tests each part of the install and saves a report you can paste when asking for help. The
report never includes API keys.

- **Settings → Diagnose → Quick check** takes seconds and downloads nothing. It checks:
  - the GPU, in each of the three Python environments,
  - audio devices and VB-Cable,
  - ffmpeg,
  - which models are already downloaded,
  - network access,
  - API keys.
- **Deep check** loads the models, downloading them the first time, then:
  - speaks a test sentence (saved as `logs\doctor-tts.wav`),
  - measures the voice changer on your GPU,
  - transcribes the test sentence,
  - pings your LLM.

  This gives the real speed of your PC. It can take several minutes.
- If the app window won't open, double-click **`doctor.bat`** (or run `doctor.bat --deep`) for the same checks in a
  console.

Each ✗ or ⚠ comes with a suggested fix. Reports are saved in `%APPDATA%\VoiceCloningTTS\logs\doctor-*.txt`.

## Using it

1. **Settings:** paste an API key for at least one provider. Keys are stored in the Windows Credential Manager.
2. **Voices:**
   - Choose an mp3/mp4 and drag on the waveform to select 10-30 s of clean speech.
   - Pick an engine and language, then **Save voice**.
   - Use **Preview** to hear it, and **Edit** to tune the engine settings: exaggeration, CFG/pace and temperature for
     Chatterbox; speed and temperature for XTTS.
3. **Personas:** edit the system prompt and answer length. *Short* is the default because spoken replies work better
   short. *Speech-friendly output* tells the model not to use markdown or lists, since those don't read well aloud.
4. **Chat:**
   - Type, or press **● Record** (Microphone / Headphones), then **■ Stop**. The transcript lands in the box for
     editing.
   - Press Enter to send. The reply streams in, and speech starts after the first sentence.
   - The length buttons in the header override the persona for the next messages.
   - **🔊 Say it** speaks your text directly. **■ Stop** cancels both generation and speech.

### Real-time voice changer

1. On **Virtual mic**, choose your microphone and the cable, then start the virtual mic.
2. In **Real-time voice changer**, pick a voice and press **🎭 Start voice changer**.
   - The first start downloads the model (about 1.5 GB). After that it starts in seconds.
   - You can also toggle it from the sidebar.
3. Talk. Other apps hear the converted voice instead of your real one.
   - Turn on *Hear my own mic* to listen to yourself converted. Use headphones, or you'll get feedback.
4. Watch **Inference / block**. It has to stay below 100% load.
   - If you hear glitches, raise *Block size* (more latency) or lower *Quality (diffusion steps)*.
   - Setting *CFG rate* to 0 is about 1.5× faster.
   - Gaming or video on the same GPU slows it down.

Tips:
- A 5-10 s clean reference works best.
- The changer keeps your intonation and accent and swaps the timbre. Pitch differences carry over, so a deep voice
  converted to a high voice can sound unnatural.

### Fine-tuning a voice for the voice changer

Zero-shot conversion already works from the voice's reference clip. Fine-tuning trains Seed-VC on more audio of that
person, so the live voice changer sounds closer to them. It doesn't affect TTS.

1. **Voices** → a voice → **Fine-tune**.
2. **+ Add training audio**: mp3, mp4 or wav of that person speaking.
   - Long recordings are split at pauses into 3-12 s clips automatically.
   - Aim for 1-5 minutes of clean speech. Even 30 seconds helps.
3. Pick the number of steps, then press **Fine-tune**.
   - The default is 200 on 4 GB cards and 500 on bigger ones.
   - Progress, loss and an ETA appear on the card.
   - The GPU is reserved while it trains, so TTS and the voice changer wait.
4. When it finishes, the voice shows **fine-tuned ✓**.
   - The voice changer uses the fine-tuned model automatically. You can switch it off per voice.
   - **🎧 Compare** records 5 s of you and plays it converted with the base model, then with the fine-tuned one.
   - The fine-tuned model (about 100 MB) and the training clips are saved with the voice and included in
     `.voice.zip` exports.

### Low-end GPUs (GTX 1650 and other 4 GB cards)

The app detects the GPU at startup. You can override what it picks under **Settings → GPU**.

- **Low-VRAM mode** turns on automatically under 6 GB. Only one heavy model sits on the GPU at a time.
  - If the voice changer is live and a chat reply needs speaking, the changer pauses. The reply plays, and the
    changer comes back on its own. Each swap costs a few seconds of loading.
  - Whisper (the Record button) runs on the CPU, using the `small` model.
- **fp32 precision** is used automatically on GTX 16-series and older cards, which produce glitches or silence in
  half precision.
- **Voice changer presets:**

  | Preset | Latency | Notes |
  | --- | --- | --- |
  | Quality | ~520 ms | |
  | Balanced | ~380 ms | |
  | Low-end GPU | ~620 ms | The default on 4 GB cards |
  | Minimal | ~920 ms | |

  Each latency is the algorithm's delay, plus about 100 ms from the devices. Press **⚡ Auto-tune** to measure your
  GPU and pick the best preset that keeps up (under 75% load).
- Expect the Low-end preset to run live on a GTX 1650. Fine-tuning 200 steps should take roughly 10-25 minutes there;
  the card shows a measured ETA.
- Close games and video playback while using the voice changer, since they compete for the same GPU.

### Preview playback

If the virtual mic isn't running, the voice plays inside the app window. You can change this under Settings →
*Browser playback*.

### Tips for good clones

- Use one speaker, with no music or background noise. 10-30 seconds of natural speech works best.
- Clone each language from a clip in that language. For Spanish, use Chatterbox *Multilingual* (the auto default for
  non-English voices) or XTTS.
- If the clone talks too fast, lower Chatterbox's *CFG / pace* to around 0.3. For more emotion, raise *exaggeration*
  to 0.7 or higher.

## Architecture

```
frontend/ (React + TS + Vite)  ⇄  REST + WebSocket  ⇄  backend/vctts (FastAPI, one process, pywebview window)

ConversationService ─ LLM stream ─► SentenceChunker ─► TTSManager (GPU worker thread) ─► SinkRouter
       ▲                 │                                                        ├─ AudioEngine (mic + voice → VB-Cable, monitor)
       │                 └─► ToolRegistry (empty: web search goes here)           ├─ Browser preview
   desktop UI / future Discord bot                                                └─ DiscordVoiceSink (future)
```

| Path | What it does |
| --- | --- |
| `backend/vctts/llm/` | `ChatProvider` interface; OpenAI/OpenRouter, Anthropic and Gemini providers; `ToolRegistry` |
| `backend/vctts/personas.py` | Personas, length presets → system prompt and `max_tokens` |
| `backend/vctts/conversation.py` | UI-agnostic pipeline: LLM → chunker → TTS → sinks; stop, regenerate, tool loop |
| `backend/vctts/tts/` | `TTSEngine` interface, Chatterbox and XTTS engines, `TTSManager` worker, sentence chunker, out-of-process engine worker (`worker.py`/`remote.py`) |
| `backend/vctts/voices/` | ffmpeg ingest (mp3/mp4 → clean 24 kHz mono), voice library, zip export/import |
| `backend/vctts/audio/` | WASAPI device discovery, real-time mixer / virtual mic, sinks, recorder (mic + loopback) |
| `backend/vctts/stt/` | faster-whisper transcription |
| `backend/vctts/vc/` | Real-time voice changer: Seed-VC streaming converter (`seedvc.py`, runs in `.venv-vc`), worker process, app-side controller hooked into the mixer |
| `backend/vctts/integrations/discord/` | Discord bot scaffold (`DiscordVoiceSink`, outline in `bot.py`) |

Data (chats, personas, voices, logs) lives in `%APPDATA%\VoiceCloningTTS\`.

### Adding a tool (for example web search) later

```python
state.tools.register(
    ToolSpec("web_search", "Search the web", {"type": "object", "properties": {"query": {"type": "string"}},
                                              "required": ["query"]}),
    handler=my_search_function,   # sync or async, returns str/JSON
)
```

All four providers already convert tool definitions, tool calls and tool results, and the pipeline runs the loop.

## Development

```powershell
.venv\Scripts\pip install -e "backend[dev]"
.venv\Scripts\python -m pytest backend\tests      # unit + API tests (fake LLM/TTS, no GPU needed)
powershell -File scripts\dev.ps1                   # backend :8765 with reload + Vite :5173
```

`python -m vctts --browser` opens the UI in your browser instead of a window. `--server-only` runs just the API.

## Licensing notes

- Chatterbox: MIT. Its output includes Resemble's imperceptible Perth watermark.
- Seed-VC (voice changer): GPL-3.0. The setup script downloads it separately into `vendor/`, and it only runs in its own worker process.
- XTTS-v2 weights: [Coqui Public Model License](https://coqui.ai/cpml), **non-commercial only**. You have to accept it
  in Settings before XTTS loads.
- Only clone voices you have permission to use.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Voice not heard in Discord | Discord input = `CABLE Output`, the virtual mic is started, and Krisp is off |
| Echo or doubled voice | Turn off *Hear my own mic*; don't set the monitor to the cable |
| "PyTorch can't see the GPU" | Re-run setup. Check `nvidia-smi` and update the NVIDIA driver |
| Whisper fails on GPU | It falls back to CPU automatically. Set Settings → Speech-to-text → Run on: CPU |
| Headphone recording missing | `pip install PyAudioWPatch` (the setup script installs it) |
| Voice changer choppy on a 4 GB card | Press **Auto-tune**, or pick *Low-end GPU*; check Settings → GPU shows fp32 |
| Fine-tune fails with out of memory | Close other GPU apps and restart the app (batch size is already 1 in low-VRAM mode) |
| Voice changer choppy | Raise *Block size*, lower *Quality*, set *CFG rate* to 0, close GPU-heavy apps |
| Voice changer silent | Lower *Silence gate* (try -60 dB) and check the Mic meter moves |
| Logs | `%APPDATA%\VoiceCloningTTS\logs\vctts.log` |
