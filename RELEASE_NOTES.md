Scriba 0.13.0 — Modular Architecture Refactoring

Highlights

- **Modular Codebase**: Complete reorganization into `core/`, `database/`, `tui/`, and `mcp/` modules with clear separation of concerns.
- **DRY Database Layer**: Eliminated duplicate row-mapping code with shared helper functions in `database/repository.rs`.
- **Unified File Operations**: Single `FileManager` in `core/files.rs` provides consistent file handling across all features.
- **Clean Module Boundaries**: Core business logic has no UI dependencies; TUI depends on Core, not vice versa.
- **Backward Compatible**: Public API exports maintained in `lib.rs` for existing integrations.

Changelog

- refactor(core): extract audio, config, recording, transcription, files, types, and workflow modules
- refactor(database): split into models.rs and repository.rs with DRY row mappers
- refactor(tui): move dashboard to dedicated tui/ module
- refactor(mcp): move MCP server to mcp/ module
- docs: update AGENTS.md with new module structure

New Module Structure
```
src/
├── core/           # Business logic (no UI deps)
│   ├── audio.rs, config.rs, files.rs, recording.rs
│   ├── transcription.rs, types.rs, workflow.rs
├── database/       # Data persistence
│   ├── models.rs, repository.rs
├── tui/            # Terminal UI
│   └── app.rs
├── mcp/            # MCP server
└── lib.rs, main.rs, errors.rs, utils.rs
```

---

Scriba 0.12.1 — Real-Time MCP Data Access

Highlights

- **Real-Time Database Access**: MCP server now creates fresh database connections for each request, ensuring Claude Desktop and other MCP clients see new recordings immediately without restart.
- **Improved Data Consistency**: Eliminates stale data issues when multiple Scriba instances are running concurrently.
- **Better UX**: Record audio and immediately ask Claude about it - no restart required.

Changelog

- fix(mcp): ensure fresh database connection for real-time data access
- fix(mcp): remove cached database connection at startup
- fix(mcp): improve data consistency for concurrent Scriba instances

---

Scriba 0.12.0 — MCP Server Integration for AI Assistants

Highlights

- **MCP Server**: Added Model Context Protocol server integration for seamless access to transcripts from Claude Desktop and other MCP clients via `scriba mcp` command.
- **AI Assistant Tools**: Four specialized tools for transcript access: list_transcripts, get_transcript, search_transcripts, and get_recording_info with full JSON schema validation.
- **Production-Ready**: Optimized MCP server with startup database initialization, professional logging, and proper JSON-RPC 2.0 implementation.
- **Claude Desktop Integration**: Complete setup guide and configuration for immediate use with Claude Desktop AI assistant.

Changelog

- feat(mcp): add Model Context Protocol server with STDIO transport
- feat(mcp): implement list_transcripts, get_transcript, search_transcripts, get_recording_info tools
- feat(database): add get_recording method for MCP server support
- feat(cli): add mcp subcommand to start MCP server
- docs(mcp): add Claude Desktop integration guide
- docs: update README with MCP server documentation and usage examples

---

Scriba 0.11.1 — Unified Core, Smoother UX, Reliable Builds

Highlights

- Unified recording: Introduced a single `record_audio` entrypoint with `RecordOptions`, removing duplicate logic and thin wrappers. CLI/TUI differences are now configuration, not code paths.
- Unified transcription: Added `transcribe_audio` to handle both local (Whisper via whisper-rs) and OpenAI API modes with a simple verbose flag for CLI/TUI.
- TUI import UX: The “A” flow now imports + transcribes in the background with a live progress animation instead of blocking the UI.
- Clean transcripts: Standardized on `transcript.txt` everywhere; removed `recording.txt` legacy fallback.
- Accurate metadata: Centralized audio metadata extraction using `FileManager::extract_audio_metadata` for consistent DB entries across record/import.
- Robust stop handling: Fixed CLI Ctrl+C and TUI stop behavior by offloading waits to OS threads, keeping async tasks responsive and Send-safe.
- Core deduplication: Reduced duplication of silent vs. verbose flows in `core.rs` via internal helpers.
- Documentation: Updated README to reflect Local/API modes, dashboard “A” behavior, and progress indicators.
- CI reliability: Fixed macOS ARM build by disabling ggml i8mm in the release workflow for broader Apple Silicon compatibility.

Changelog

- feat(core): unify recording/transcription entrypoints and simplify workflows
- feat(tui): non-blocking import + transcription with progress
- refactor: remove legacy `recording.txt` and associated fallbacks
- refactor: centralize metadata extraction for DB save
- fix(cli/tui): stop signal handling without blocking the runtime
- docs: update README for modes, dashboard keys, and progress UX
- ci: disable ggml i8mm on aarch64 macOS to stabilize builds

Notes

- Local transcription auto-downloads Whisper models on first use if not found under `~/scriba_recordings/models/`.
- API transcription uses `OPENAI_API_KEY` from the dashboard settings or CLI `--api-key`.

