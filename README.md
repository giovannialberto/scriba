<div align="center">

<img src="docs/screenshots/scriba-logo-v2.png" alt="Scriba" width="160" height="65" />

**Record. Transcribe. Flow.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

Scriba is an agent that turns your recordings into a searchable, queryable knowledge base and lets you ask questions across all of it. It builds a persistent knowledge graph so the agent always has context, not just the last thing you said.

Run entirely **offline** with Whisper.cpp + Ollama, or bring your own **AI providers** (OpenAI, Anthropic, Google).

<div align="center">
<img src="docs/screenshots/scriba-home-cut.png" alt="Scriba home" width="700" />
</div>

## How it works

1. **Record** — capture your microphone or system audio
2. **Transcribe** — local Whisper models or the OpenAI Whisper API
3. **Enrich** — an LLM extracts summaries, topics, entities, and action items from every recording
4. **Ask** — an agent reasons across your entire history to answer questions, find connections, and take action

## Get started

**Requirements:** CMake and FFmpeg (`brew install cmake ffmpeg`). Ollama is optional, for Private mode.

```bash
brew tap giovannialberto/scriba
brew install scriba
scriba
```

On first run, Scriba walks you through an onboarding flow to choose your mode and configure your setup. Then **`Ctrl+R`** to record.


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

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS).

## License

[MIT](https://github.com/giovannialberto/scriba/blob/main/LICENSE) — Copyright (c) 2026 Giovanni Alberto Falcione
