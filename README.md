<div align="center">

<img src="docs/screenshots/scriba-logo-v2.png" alt="Scriba" width="280" height="115" />

**Record. Transcribe. Flow.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

Scriba is an agent that turns your recordings into a searchable, queryable knowledge base and lets you ask questions across all of it. It builds a persistent knowledge graph so the agent always has context, not just the last thing you said.

Run entirely **offline** with local STT models (Parakeet, Whisper, SenseVoice) + Ollama, or bring your own **AI providers**: Anthropic, OpenAI, Google, or any **OpenAI-compatible endpoint** (DeepInfra, OpenRouter, Groq, Together, vLLM, LM Studio) for open-weight models.

<div align="center">
<img src="docs/screenshots/scriba-home-cut.png" alt="Scriba home" width="700" />
</div>

## How it works

1. **Record** — capture your microphone or system audio
2. **Transcribe** — local STT models (Parakeet, Whisper, SenseVoice) or a cloud speech API, with speaker labels (who said what) on both: cloud hosts return segment timestamps and the labelling itself stays on your machine. Scriba learns your voice during onboarding (or `scriba voice enroll <clip>`) and keeps learning it from your meetings, so it can name you in the transcript
3. **Enrich** — an LLM extracts summaries, topics, entities, and action items from every recording
4. **Ask** — an agent reasons across your entire history to answer questions, find connections, and take action

## Get started

**Requirements:** FFmpeg and (optionally) Ollama for Private mode.

### Install

macOS (Apple Silicon and Intel) and Linux (x86_64):

```bash
curl -fsSL https://raw.githubusercontent.com/giovannialberto/scriba/main/install.sh | sh
```

The script downloads the latest release, verifies its checksum, and installs `scriba` into `~/.local/bin` — no sudo. Set `SCRIBA_INSTALL_DIR` to change the location or `SCRIBA_VERSION` to pin a release. Scriba updates itself from the dashboard afterwards.

FFmpeg: `brew install ffmpeg` on macOS; `sudo apt install ffmpeg` (Debian/Ubuntu), `sudo dnf install ffmpeg` (Fedora/RHEL), or `sudo pacman -S ffmpeg` (Arch) on Linux.

<details>
<summary>Alternatives: Homebrew, direct download</summary>

```bash
brew install ffmpeg
brew tap giovannialberto/scriba
brew trust --tap giovannialberto/scriba   # Homebrew 6+: allow formulae from this tap
brew install scriba
```

Or grab a binary directly from [Releases](https://github.com/giovannialberto/scriba/releases).

**Already on Homebrew with Scriba 0.28.0 or earlier?** Homebrew 6 now requires trusting third-party taps, and the in-app updater in those versions can't do it for you. Run this once, then update normally:

```bash
brew trust --tap giovannialberto/scriba && brew update && brew upgrade scriba
```

</details>

### Run

```bash
scriba
```

On first run, Scriba walks you through an onboarding flow to choose your mode and configure your setup. Then **`Ctrl+R`** to record.

### Choosing an AI provider

Private mode uses Ollama. Cloud mode works with Anthropic, OpenAI, Google, or any OpenAI-compatible endpoint, so open-weight models hosted on DeepInfra, OpenRouter, Groq, or Together, or served locally by vLLM or LM Studio, are one setting away. Pick them in Settings or from the CLI:

```bash
scriba config set-provider deepinfra                          # known host: endpoint pre-filled
scriba config set-enrichment-model Qwen/Qwen3.5-397B-A17B
scriba config set-enrichment-key <your-key>                   # or export DEEPINFRA_API_KEY

scriba config set-provider custom --base-url http://localhost:8000/v1   # vLLM, LM Studio, ...
```

The Settings screen lists the models the endpoint advertises, so you can switch without looking up IDs.

Cloud transcription is OpenAI-compatible too. OpenAI is the default; Groq and DeepInfra are one flag away, and any host with an `/audio/transcriptions` endpoint works:

```bash
scriba config set-api <your-key> --preset groq                       # whisper-large-v3-turbo on Groq
scriba config set-api <your-key> --model gpt-4o-transcribe-diarize   # OpenAI, with speaker labels
scriba config set-api <your-key> --base-url http://stt.local:8000/v1 --model whisper-1
```

## Meeting detection

Scriba notices when a meeting starts — Zoom, Meet, Teams, or any app that opens your microphone — and offers to record it. No audio is analyzed for this: it watches which process holds the mic, so talking at your desk never triggers it.

- A notification card asks **Record** or **Ignore** the moment a call begins (unanswered cards are ignored — nothing is recorded without your consent).
- Recording stops by itself the instant you leave the call, then transcription and enrichment run in the background.
- While recording, a live indicator shows the app, elapsed time, and mic level; the recording appears in your list the moment it stops, with a spinner until the transcript is ready.
- **`Ctrl+R`** stops any recording in progress; the **Meeting Watch** toggle in Settings turns detection off entirely.

Detection runs whenever the dashboard is open, at effectively zero CPU cost. You can also run it standalone with `scriba watch` (see `scriba watch --help` for options such as `--no-confirm`).

Fine-tuning lives under `meeting_detection` in `~/scriba_recordings/config.json`: `confirm_timeout_seconds`, `cooldown_seconds`, and `ignored_processes` — a list of app/bundle-id substrings whose mic use should never count as a meeting (useful if you run another recording tool alongside Scriba).

Supported on macOS 14+ (per-app attribution via Core Audio) and Linux with PulseAudio/PipeWire.


## Ask Scriba

Ask *"what did we decide in last Tuesday's call?"* or *"who has mentioned the Q2 roadmap?"* and Scriba will search your transcripts, look up entities, and chain tool calls to get to a real answer. Ask from the home to query your entire history, or from within a transcript for recording-specific context.

<div align="center">
<img src="docs/screenshots/scriba-chat.png" alt="Ask Scriba chat" width="700" />
</div>

## MCP server

Scriba exposes your recordings to Claude Desktop via the Model Context Protocol:

```json
{
  "mcpServers": {
    "scriba": { "command": "scriba", "args": ["mcp"] }
  }
}
```

Add to your Claude Desktop config:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

## License

[MIT](https://github.com/giovannialberto/scriba/blob/main/LICENSE) — Copyright (c) 2026 Giovanni Alberto Falcione
