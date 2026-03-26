Scriba 0.22.0 — Clean Slate

Highlights

- **New identity**: Scriba sheds the owl persona and embraces a clean, professional identity with a bold block-art SCRIBA logo in both CLI and TUI. A hidden Ctrl+O easter egg pays homage to the owl era with an animated takeover sequence.
- **Modular TUI architecture**: The monolithic 7,519-line `app.rs` has been decomposed into 7 focused modules (browse, entities, onboarding, recording, settings, transcript, and core app), dramatically improving maintainability.
- **Redesigned UI**: Settings and World pages adopt a borderless full-screen layout. A new Privacy Mode setup wizard guides users through system checks and local model downloads.
- **14 bug fixes**: Panic fixes for empty recordings navigation, unwrap removals throughout, keyboard shortcut accessibility improvements, byte-boundary truncation guards, and platform-aware install hints.

Changelog

- feat: remove owl ASCII art, tagline, and personality from CLI banner and all chat system prompts
- feat: replace CLI banner with thick block-art SCRIBA logo matching the TUI
- feat: add hidden Ctrl+O easter egg — animated owl takeover with 5 sprite frames, flashing colors, and countdown bar
- feat: redesign Settings and World pages to borderless full-screen layout
- feat: add Privacy Mode setup wizard with system checks and model downloads
- refactor: decompose `src/tui/app.rs` into 7 focused modules (app, browse, entities, onboarding, recording, settings, transcript)
- fix: panic on empty recordings list in next/previous navigation
- fix: `WorkflowManager::new().unwrap()` replaced with proper error propagation
- fix: Ctrl+S/Ctrl+W shortcuts not accessible from Browse view
- fix: settings UI alignment for Anthropic API key label
- fix: Esc in entity Editing mode now cancels instead of saving
- fix: byte-boundary truncation panics in entity aliases/context display
- fix: merge-self guard when source entity has no DB id
- fix: file dialog stays open on validation error instead of closing
- fix: filesystem deletion failure now surfaces as notification
- fix: `reqwest::Client::build().unwrap()` panic in onboarding
- fix: platform-aware install hints (macOS/Linux/Windows) in onboarding
- fix: byte-safe API key display in settings (chars vs bytes)
- fix: `Database::new()` replaced with `self.db` in voice stop command
- fix: Ctrl+W shortcut on non-US keyboard layouts (Italian QWERTY)

---

Scriba 0.21.1 — Unified Tools & Multi-Provider Agent

Highlights

- **Unified tool layer**: Agent chat and MCP server now share a single set of 14 tools (9 read + 5 write), eliminating ~370 lines of duplication. Write tools (update, alias, merge, delete entities) are now available from the in-app agent chat too.
- **Multi-provider agent**: The agent loop now works across all LLM providers (Anthropic, OpenAI, Google, Ollama) — no longer Anthropic-only. Each provider handles its own streaming format and tool-use protocol.

Changelog

- refactor: extract shared `src/tools/` module with `ToolSchema`, `ToolResult`, and `execute_tool` dispatcher for all 14 tools
- refactor: slim `agent/tools.rs` (465 → 158 lines) and `mcp/mod.rs` (1022 → 247 lines) to thin wrappers
- feat: all 5 entity write tools now available in agent chat (update, add/remove alias, merge, delete)
- feat: legacy MCP tool names (`list_transcripts`, `get_recording_info`, `search_by_entity`) preserved via aliases
- feat: generalize agent loop across Anthropic, OpenAI, Google, and Ollama providers
- refactor: extract provider-specific streaming/tool logic into `agent/providers/` module

---

Scriba 0.19.4 — Wing-Cut & HOOOOOO!!

Highlights

- **Wing-cut phase**: The owl no longer magically materializes the sand trail — it flies right above the ground, scoring the sand with its wing (`\` / `/` alternating sawing motion) to create the trail itself.
- **Return pass**: After cutting, the owl flies back to the start before vacuuming the trail left-to-right.
- **HOOOOOO!! shout**: After devouring the trail, the owl screams `HOOOOOOOO!!` — the shout grows right-to-left filling the whole trail area while crazy-eye sprites cycle twice as fast.
- **30s cycle**: Extended from 25s to 30s (300 frames) to accommodate the new wing-cut and return phases.

Changelog

- feat(tui): wing-cut phase — owl creates the sand trail by scoring ground with its wing
- feat(tui): return phase — owl flies back to start after cutting, full trail visible
- feat(tui): growing "HOOOOOO!!" shout during celebration, right-to-left expansion
- feat(tui): faster celebration sprite cycling (every 2 frames)
- feat(tui): 300-frame cycle (30s) with 8 distinct animation phases

---

Scriba 0.19.3 — Trail-Eating Owl Animation

Highlights

- **New header animation: "Trail Eating" owl**: The owl's idle fly-out animation is replaced with a playful 25-second sequence. A sand trail of dots materializes after the logo, the owl notices it (eyes shift right), then hovers above the trail vacuuming up dots through its nose with a breathing rhythm. After devouring the trail, the owl celebrates with crazy eye sprites before flapping back to its perch.
- **Suction visual**: During eating, the owl sits on line 1 above the trail. Dots on line 2 show `'` and `^` suction characters rising into the owl's beak, styled in yellow against the gray trail.
- **Narrow terminal fallback**: Terminals too narrow for the trail gracefully skip the animation and show static idle.

Changelog

- feat(tui): trail-eating owl header animation with 7 phases (idle → trail appears → notices → eats → celebrates → flies back → idle)
- feat(tui): owl hovers above trail with breathing rhythm `(o,o)` / `(O,O)` and suction chars `'` / `^`
- feat(tui): crazy celebration sprites at trail end: `\(@,@)/`, `/(O,O)\`, `\(x,X)/`, `|(O,o)|`
- feat(tui): ease-in-out quadratic for eating, ease-out for fly-back
- fix(tui): narrow terminal graceful fallback (trail_len < 10 → static idle)

---

Scriba 0.19.2 — Hotfix: Chat Panel Crash + Input Wrapping

Fixes

- **fix: crash on long chat responses**: The app could panic with an index-out-of-bounds error when a streaming LLM response completed. A stale scroll offset caused the render loop to index past the end of the dynamic lines buffer. The scroll offset is now clamped after each content recomputation, with an additional bounds check as a safety net.
- **fix: input text wraps instead of overflowing**: Long messages typed in the Ask Scriba input now wrap to multiple lines instead of disappearing off the right edge. The chat content area dynamically shrinks to make room for the growing input.
- **fix: copy-to-clipboard notification in transcript view**: The "Copied to clipboard" notification was invisible in the per-recording Ask Scriba panel because the footer wasn't rendered in that view. A notification overlay is now shown in the transcript popup.

---

Scriba 0.19.1 — Hotfix: Database Migration

Fixes

- **fix: database migration no longer destroys data**: v0.19.0 could wipe the database on upgrade for users who installed v0.18.0 fresh (the migration tried to add columns that already existed, triggering a destructive auto-reset). The migration now checks for existing columns before altering tables, and the auto-reset fallback has been removed entirely.
- **feat: `scriba db rebuild`**: New CLI command that scans `~/scriba_recordings/` directories and rebuilds the database from audio files and transcripts on disk. Use this to recover if your database was lost.

---

Scriba 0.19.0 — Ask Scriba

Highlights

- **Ask Scriba chat panel**: Interactive chat panel in the TUI — ask questions about your recordings or your world context. Streaming markdown responses with syntax highlighting, text selection, and auto-copy.
- **Multi-provider LLM support**: Choose between Ollama (local), Anthropic (Claude), OpenAI (GPT), or Google (Gemini) for both enrichment and chat. Configure via `scriba config` or TUI settings.
- **Web search integration**: Scriba can search the web (DuckDuckGo) during enrichment to verify and enrich entity information.
- **Chat panel polish**: Headings render cleanly (no `#` symbols), scrollbar thumb is stable, and completed messages are cached for smooth 60fps scrolling.

Changelog

- feat(tui): Ask Scriba chat panel with streaming responses, markdown rendering, and text selection
- feat(enrichment): add Anthropic, OpenAI, and Google Gemini LLM provider clients
- feat(enrichment): common `LlmProvider` trait for unified multi-provider access
- feat(enrichment): DuckDuckGo web search for entity verification during enrichment
- feat(enrichment): chat system prompts with world context and recording-specific modes
- feat(tui): strip heading `#` prefix spans from markdown output
- feat(tui): fix scrollbar wiggle by removing begin/end arrow symbols
- perf(tui): cache completed-message rendering — only visible lines cloned per frame
- feat(config): `EnrichmentMode` enum (Local/Cloud) with automatic migration from legacy Ollama fields
- feat(config): `search_enabled`, `max_search_results`, `evolve_world` settings

Migration

- Seamless upgrade: all new config fields use `#[serde(default)]`. Existing configs load unchanged.
- Legacy `ollama_endpoint`/`ollama_model` fields automatically migrate to `EnrichmentMode::Local`.
- No database schema changes — existing recordings and entities are untouched.

---

Scriba 0.18.0 — Speaker Diarization

Highlights

- **Speaker diarization**: Automatically identify and label different speakers in recordings using pyannote-rs (ONNX). After transcription, an LLM pass maps speaker labels to real names from your world context.
- **Async model download**: Diarization ONNX models (~200MB) download automatically on first use without blocking the UI.

Changelog

- feat(core): speaker diarization with pyannote-rs and ONNX runtime
- feat(core): LLM-powered speaker identification using world context
- fix: async diarization model download + strip markdown fences in LLM parsing

---

Scriba 0.17.0 — TUI Settings & Silence Auto-Stop

Highlights

- **TUI Ollama settings**: Manage Ollama endpoint and model directly from the TUI Settings view — no manual config editing needed.
- **Silence auto-stop**: Recordings automatically stop after a configurable period of silence (default: 60s). Detects laptop lid close and stale audio callbacks.

Changelog

- feat(tui): manage Ollama settings from TUI Settings view
- feat(core): silence auto-stop with configurable timeout
- fix: detect stale audio callback for lid-close auto-stop
- fix: raise silence threshold to 0.005 for reliable lid-close detection

---

Scriba 0.16.0 — Whisper World Prompt

Highlights

- **Whisper world prompt**: Whisper transcription now receives a prompt built from your world context — names of people, organizations, and projects you've mentioned before. This dramatically improves recognition of proper nouns and domain-specific terms.
- **Entity quality fixes**: Better deduplication and alias handling during extraction.

Changelog

- feat(core): build Whisper prompt from world context for better proper noun recognition
- fix: entity quality improvements and smooth setup for new/existing users

---

Scriba 0.15.2 — Context Compaction

Highlights

- **LLM-powered entity context compaction** (#35): Entity descriptions are now automatically compacted during enrichment. Instead of appending sentence fragments from each recording, Scriba merges existing context + new information into a clean, self-contained description via a single LLM call. Entity context always reads as a polished summary — not an append log.

Changelog

- feat(enrichment): replace `append_new_facts()` with LLM-powered `compact_entity_context()` during world delta application
- feat(enrichment): new `build_context_compaction_prompt()` for single-call merge + summarize
- feat(core): `apply_world_delta_to_entities()` is now async with LLM compaction and graceful fallback
- refactor(core): remove `append_new_facts` import from workflow (no longer needed for entity updates)

---

Scriba 0.15.1 — Bug Fixes

Fixes

- **fix: Merged entities no longer recreated** (#34): After merging entities in the TUI, subsequent enrichment runs now correctly recognize aliases and link to the existing entity instead of creating a duplicate.
- **fix: Duplicate entities from single extraction** (#33): If the same person or organization appeared multiple times in a transcript, they could be created as separate entities. The linker now deduplicates extraction results before processing, and the LLM prompt explicitly instructs against duplicates.
- **fix: Alias-aware entity lookup**: New `get_entity_by_name_or_alias()` searches both canonical names and aliases across the entire entity pipeline (linker, world delta application).
- **fix: World context refreshed before extraction**: World.md is now rebuilt from entities before each extraction, ensuring the LLM sees up-to-date alias mappings and merged state.

---

Scriba 0.15.0 — Scriba the Owl

Highlights

- **Meet Scriba the Owl**: A charming ASCII owl character that guides you through the entire experience with personality, humor, and animations.
- **Animated Onboarding**: First-run users get a delightful, videogame-style onboarding conversation with the owl — it flies onto screen, asks questions, builds your world, and transitions to the dashboard with a sparkle dissolve effect.
- **Scriba's World (W)**: The entities view is now "Scriba's World" — same powerful entity table with a new animated owl header showing owner summary, entity counts, and contextual owl quips that react to your actions.
- **Add Entities (A)**: Create new entities directly from the World view with name, type, context, and aliases. The missing "C" in CRUD is here.
- **Owl Personality**: The owl reacts to everything — celebrates when you add or edit, quips when you delete ("Gone but not forgotten... well, actually forgotten."), scribbles along while you're editing, and blinks in idle mode.
- **Flying Owl Header**: The main dashboard header features an animated owl that occasionally takes flight across the navbar, bobbing up and down with eased motion before settling back next to the SCRIBA logo.
- **Compact Header**: New 3-line owl + thin block letter header replaces the old 5-line block letters. Tagline: "hoo remembers everything".
- **Graceful Ollama Fallback**: If Ollama isn't running during onboarding, your info is saved as-is with a friendly message — no crash, no confusion.

Changelog

- feat(tui): animated owl onboarding experience with entrance fly-in, typewriter text, speech bubbles, step dots, and magic sparkle transition
- feat(tui): "Scriba's World" view with animated owl header, owner summary, entity counts, and contextual quips
- feat(tui): add entity mode (A key) with name/type/context popup and entity creation
- feat(tui): owl mood system — Idle (blink), Thinking (scribble), Celebrating (wing-wave) — reacts to user actions
- feat(tui): flying owl animation in main dashboard header with sine-wave vertical bobbing
- feat(tui): compact owl + thin block letter header with "hoo remembers everything" tagline
- feat(core): extract reusable `initialize_world_from_seed()` for both CLI and TUI world initialization
- refactor(tui): replace E key with W for World view, update all footer hints
- refactor(main): simplify `WorldCommand::Init` to delegate to shared workflow function

---

Scriba 0.14.0 — Knowledge Extraction & Entity Management

Highlights

- **Knowledge Extraction**: Ollama LLM integration for automatic transcript enrichment — extracts summaries, key points, topics, and entities from recordings.
- **World Context**: Persistent knowledge graph (`world.md`) tracks people, organizations, and projects across all recordings, evolving as new transcripts are processed.
- **LLM-Driven Entity Linking**: Entities are resolved against world context by the LLM itself, producing accurate canonical names without fuzzy matching.
- **Entities as Source of Truth**: Entity database drives the world — `world.md` is rebuilt from entities after every change, not the other way around.
- **TUI Entity Management**: Full CRUD for entities directly from the dashboard — edit name/type/context (`E`), delete (`D`), and merge entities (`M`) with alias and context combination.
- **Non-Blocking Transcription**: Transcription runs in the background with an inline spinner in the recordings table. Browse, edit entities, and view transcripts while transcription completes.
- **Smart Fact Accumulation**: World evolution appends new facts rather than replacing existing knowledge, with sentence-level deduplication.
- **Improved Delete Popup**: Clean, consistent confirmation dialogs for both recordings and entities.

Changelog

- feat(enrichment): add Ollama LLM client and enrichment service with extraction + world evolution
- feat(enrichment): add world context system with seed, merge, and delta application
- feat(enrichment): LLM-driven entity extraction with `resolved_to` field for canonical linking
- feat(entities): entity registry with rename, merge, delete, alias management, and type update
- feat(workflow): `rebuild_world_from_entities()` — entities drive world.md content
- feat(workflow): `apply_world_delta_to_entities()` — LLM deltas applied to entity DB first
- feat(workflow): `enrich_recording()` orchestration with extraction, linking, and world evolution
- feat(tui): entity management view with edit/delete/merge and EntityMode state machine
- feat(tui): non-blocking background transcription with inline status spinner
- feat(tui): auto-dismissing footer notifications for transcription completion
- refactor(tui): consistent delete confirmation popups for recordings and entities
- refactor(enrichment): extract `append_new_facts()` for sentence-level fact dedup
- refactor(workflow): remove `sync_world_to_entities()` in favor of entities-first flow

New Module Structure
```
src/
├── enrichment/     # LLM integration
│   ├── extractor.rs, ollama.rs, prompts.rs
│   ├── world.rs, context.rs
├── entities/       # Entity management
│   ├── linker.rs, registry.rs
```

Notes

- Requires a running Ollama instance (`ollama serve`) with a model pulled (default: `llama3.2`, recommended: `mistral`).
- World context is stored at `~/scriba_recordings/world.md` as structured JSON.
- Use `scriba world init` to seed initial knowledge, then `scriba enrich <recording>` to process transcripts.
- Entity edits in the TUI automatically rebuild `world.md` to stay in sync.

---

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

