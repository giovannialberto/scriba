# Scriba

Rust TUI that records audio, transcribes it (local or cloud), enriches it with LLM-powered entity extraction and summaries, and lets you query your entire recording history through an agent chat.

## Build & Run

```bash
cargo build                  # debug build
cargo build --release        # release build
cargo run                    # launch TUI dashboard
cargo test                   # run all tests (~32 tests, <1s)
cargo fmt --all              # format
cargo clippy -- -D warnings  # lint
```

## Architecture

```
src/
  main.rs              CLI entry point (StructOpt)
  lib.rs               Public module exports
  core/                Business logic (no UI deps)
    recording.rs       Audio capture via cpal
    transcription.rs   STT: sherpa-onnx (local) + OpenAI API
    workflow.rs        Orchestration: record -> transcribe -> enrich
    loopback.rs        System audio capture (macOS: ScreenCaptureKit, Linux: PulseAudio/PipeWire)
    config.rs          ScribaConfig, TranscriptionMode, EnrichmentMode
  database/            SQLite persistence (schema.sql at repo root, included at compile time)
  enrichment/          LLM integration (Ollama, Anthropic, OpenAI, Google)
    world.rs           Knowledge graph (WorldData), single source of truth
    extractor.rs       Entity/topic extraction from transcripts
    prompts.rs         All LLM prompts for extraction and evolution
    chat_prompts.rs    Chat system prompts (agent + fallback)
  agent/               Agent loop with tool use (Anthropic provider)
    providers/         Per-provider implementations
  entities/            Entity registry and linking (LLM-driven, no fuzzy matching)
  tui/                 Terminal UI (ratatui)
    app.rs             Main dashboard, navigation, key handling
    chat.rs            Chat interface, streaming, rendering
    browse.rs          Recording browser
    entities.rs        Entity browser/editor
    onboarding.rs      First-run setup flow
    settings.rs        Settings UI
    transcript.rs      Transcript viewer
    recording.rs       Recording UI
  mcp/                 Model Context Protocol server for Claude Desktop
  tools/               Tool definitions and executor for agent
```

## Key conventions

- **Edition 2024** — match ergonomics require no `ref`/`ref mut` on auto-deref patterns; `unsafe` blocks required inside `unsafe fn`
- **Conventional Commits** — `feat:`, `fix:`, `chore:`, `docs:`. Subject ≤72 chars, imperative mood
- **No co-author trailers** in commits
- **Branches for PRs** — don't commit directly to main
- **Focused PRs** — one concern per PR, no kitchen-sink changes
- **`anyhow::Result`** throughout; `#[serde(default)]` on all LLM-facing struct fields
- **TUI is modular** — each view in its own file under `tui/`, keep it that way
- **World = single source of truth** — DB entities are a materialized index derived from `~/scriba_recordings/world.md`
- **Entity linking is LLM-driven** — no fuzzy matching, no substring scanning

## Release process

1. Bump version in `Cargo.toml`
2. Commit: `chore: bump version to X.Y.Z`
3. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
4. CI builds macOS (x86_64 + ARM64) and Linux (x86_64) binaries, creates GitHub release, updates Homebrew formula

## Platform support

- **macOS**: system audio via ScreenCaptureKit, mic via cpal, install via Homebrew
- **Linux**: system audio via PulseAudio/PipeWire monitor sources, mic via cpal/ALSA, install via `install.sh` or GitHub releases
- Platform-specific code gated with `#[cfg(target_os = "...")]` in `loopback.rs`

## Testing

- Unit tests inline: `#[cfg(test)] mod tests { ... }`
- Tests cover: world merging, prompt generation, JSON extraction, search queries, ring buffer, Ollama client
- Avoid external API calls in tests
