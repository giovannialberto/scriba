# CLAUDE.md — Scriba

## What is Scriba?

Scriba is a Rust CLI/TUI tool for recording audio, transcribing it (locally via Whisper.cpp or via OpenAI API), and enriching transcripts with LLM-powered knowledge extraction (entities, summaries, topics). It maintains a persistent knowledge graph ("World Context") across recordings. It also exposes an MCP server for integration with Claude Desktop and has an in-app agentic chat.

## Build & Run

```bash
# Build (debug)
cargo build

# Build (release)
cargo build --release

# Run directly
cargo run -- <command>

# Install locally
cargo install --path .
```

**System requirements:** CMake (for whisper-rs C++ bindings), FFmpeg (audio compression/resampling), working microphone for recording. Optional: Ollama for local LLM enrichment.

## CLI Commands

```
scriba record       # Record audio from microphone
scriba transcribe   # Transcribe an existing recording or import external audio
scriba config       # Show/set configuration (transcription mode, enrichment provider, API keys)
scriba health       # System health check (--verbose for details)
scriba mcp          # Run MCP server over stdio
scriba enrich       # Run knowledge extraction on a recording
scriba entity       # Manage entities (list, show, rename, merge, delete)
scriba world        # Manage world context (show, set, seed)
scriba db rebuild   # Rebuild database from disk
```

The TUI dashboard launches with just `scriba` (no subcommand).

## Project Structure

```
src/
├── main.rs              # CLI entry point, all commands via structopt
├── lib.rs               # Module re-exports
├── core/                # Audio pipeline: recording, encoding, transcription, config, workflow
├── database/            # SQLite models (Recording, Transcript, Entity) and repository
├── tui/                 # Ratatui-based dashboard (app.rs) and chat interface (chat.rs)
├── enrichment/          # LLM provider trait + implementations (Anthropic, OpenAI, Google, Ollama)
├── entities/            # EntityRegistry (CRUD) and EntityLinker (cross-recording correlation)
├── agent/               # Agentic chat loop with tool-use, multi-provider support
├── tools/               # 15 canonical tool definitions shared between MCP and agent
├── mcp/                 # MCP server (JSON-RPC stdio transport)
├── errors.rs            # ScribaError type
└── utils.rs             # BASE_PATH constant, recording naming helpers
```

## Architecture

- **Layered design:** UI (TUI + CLI) → WorkflowManager → Core (recording, transcription, encoding) → Data (Database, FileManager)
- **Provider pattern:** `LlmProvider` trait with pluggable backends (Anthropic, OpenAI, Google, Ollama) in `src/enrichment/`
- **Shared tool layer:** 15 tools defined in `src/tools/definitions.rs`, executed by `src/tools/executor.rs`, used by both MCP server and agent chat
- **Async throughout:** Tokio runtime, channels for TUI ↔ recording communication
- **SQLite storage:** `rusqlite` with bundled SQLite, full-text search, entity relationship tracking

## Key Patterns

- Configuration lives at `~/scriba_recordings/config.json`; world context at `~/scriba_recordings/world.md`; database at `~/scriba_recordings/scriba.db`
- `BASE_PATH` in `src/utils.rs` is the root for all recordings and data
- Recordings are stored as directories under `~/scriba_recordings/` with audio files + metadata
- Entity extraction happens during enrichment and feeds into the World Context knowledge graph
- The `WorkflowManager` in `src/core/workflow.rs` orchestrates record → transcribe → enrich pipelines

## Testing

No formal test suite. Validate changes by:
1. `cargo build` — must compile without errors
2. `cargo run -- health --verbose` — runtime health check
3. Manual testing of affected commands/TUI flows

## Code Style

- Rust 2021 edition
- Error handling via `anyhow::Result` for application code, `thiserror` for library error types (`ScribaError`)
- CLI parsing with `structopt`
- TUI rendering with `ratatui` + `crossterm`
- Async traits via `async-trait` crate
