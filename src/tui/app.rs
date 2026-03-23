use crate::core::{
    CloudProvider, CompressionSettings, EnrichmentMode, LocalModelSize, RecordOptions, RecordingResult, ScribaConfig,
    TranscriptionMode, VoiceCommand, VoiceDetectorHandle, VoiceListeningState,
    WorkflowManager, record_audio, rebuild_world_from_entities, initialize_world_from_seed,
    start_voice_detector,
};
use crate::database::{Database, Entity, Recording, RecordingStats};
use crate::enrichment::{OllamaClient, WorldContext, WorldData, WorldEntityExtractionResult};
use crate::enrichment::chat_prompts;
use crate::entities::EntityRegistry;
use crate::utils::generate_recording_name;
use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table, TableState, Wrap},
    Frame, Terminal,
};
use std::collections::VecDeque;
use std::io;
use tokio::sync::mpsc;

use anyhow::Context;
use dirs::home_dir;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command as TokioCommand;

use super::chat::{
    ChatContext, ChatFocus, ChatMessage, ChatRole, ChatState, ChatStreamEvent,
    chat_agent_pipeline, ACCENT,
};

pub struct Dashboard {
    db: Database,
    recordings: Vec<Recording>,
    table_state: TableState,
    current_page: usize,
    page_size: usize,
    stats: Option<RecordingStats>,
    show_help: bool,
    current_view: DashboardView,
    search_mode: bool,
    search_query: String,
    show_message: bool,
    message: String,
    show_transcript: bool,
    transcript_content: String,
    show_delete_confirm: bool,
    delete_confirm_selection: usize,  // 0 = Yes, 1 = No
    delete_candidate: Option<Recording>,
    current_playback_pid: Option<u32>,
    playback_finished_rx: Option<mpsc::Receiver<()>>, // Channel to receive playback completion
    last_transcribe_warning: Option<usize>, // Track which recording showed overwrite warning
    progress_animation: Option<String>,     // Base message for progress animation
    progress_frame: usize,                  // Animation frame counter
    active_transcription: Option<ActiveTranscription>, // Currently running transcription
    transcription_queue: VecDeque<PendingTranscription>, // FIFO queue of pending transcriptions
    notification_message: Option<(String, usize)>, // (message, frames_remaining) — auto-dismiss
    recording_task: Option<tokio::task::JoinHandle<Result<RecordingResult, anyhow::Error>>>,
    recording_mode: Option<RecordingMode>, // Track if we should transcribe after recording
    recording_stop_tx: Option<mpsc::Sender<()>>, // Channel to stop recording
    recording_level_rx: Option<mpsc::Receiver<f32>>, // Channel to receive volume levels
    current_volume_level: f32,             // Current recording volume for display
    recording_start_instant: Option<std::time::Instant>, // When recording started (for elapsed time)
    volume_history: VecDeque<f32>,         // Recent volume samples for waveform display
    config: ScribaConfig,                  // App configuration
    settings_selection: usize,             // Current setting selection
    editing_api_key: bool,                 // Whether we're editing API key
    api_key_input: String,                 // API key input buffer
    model_picker_state: ModelPickerState,
    model_picker_items: Vec<ModelPickerItem>,
    model_picker_selection: usize,
    model_picker_custom_input: String,
    ollama_models_rx: Option<mpsc::Receiver<Result<Vec<String>, String>>>,
    editing_enrichment_endpoint: bool,     // Whether we're editing Ollama endpoint (local mode)
    enrichment_endpoint_input: String,     // Ollama endpoint input buffer (local mode)
    editing_enrichment_api_key: bool,      // Whether we're editing enrichment API key
    enrichment_api_key_input: String,      // Enrichment API key input buffer
    return_to_view: Option<DashboardView>, // View to return to after message dismissal
    // File import dialog state
    show_file_dialog: bool,
    file_path_input: String,
    file_name_input: String,
    file_dialog_stage: FileDialogStage, // Current stage of file import process

    // Entity view state
    entities: Vec<Entity>,
    entity_table_state: TableState,
    selected_entity: Option<Entity>,
    show_entity_detail: bool,
    entity_mode: EntityMode,
    entity_edit_field: EntityEditField,
    entity_edit_name: String,
    entity_edit_type: String,
    entity_edit_context: String,
    merge_source_entity: Option<Entity>,
    confirm_selection: usize,  // shared for entity delete/merge confirms: 0 = Yes, 1 = No
    // Add entity state
    entity_add_name: String,
    entity_add_type: String,
    entity_add_context: String,
    entity_add_aliases: String,
    entity_add_field: EntityEditField,
    // Transcript enrichment data
    transcript_summary: Option<String>,
    transcript_key_points: Option<Vec<String>>,
    transcript_topics: Option<Vec<String>>,
    transcript_entities: Option<Vec<(String, String)>>, // (name, type)

    // Onboarding state
    onboarding: Option<OnboardingState>,

    // Voice mode ("Scriba Forever") state
    voice_command_rx: Option<mpsc::Receiver<VoiceCommand>>,
    voice_detector_handle: Option<VoiceDetectorHandle>,
    voice_mode_active: bool,

    // Chat state ("Ask Scriba")
    chat: ChatState,
    global_chat_messages: Vec<ChatMessage>,
    // Track the currently-viewed recording for chat context
    current_transcript_recording: Option<Recording>,

    // Home screen greeting
    greeting_text: String,
    greeting_subtitle: String,
    owner_name: String,
}

#[derive(Debug, PartialEq)]
enum DashboardView {
    Main,
    Browse,
    Help,
    Settings,
    Entities,
    Onboarding,
}

// ─────────────────────────────────────────────────────────────────────────────
// Onboarding: Scriba the Owl
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
enum OnboardingStep {
    Entrance,
    Intro,
    ModeSelection,
    // Cloud flow
    ProviderSelection,
    ApiKeyEntry,
    ApiKeyValidation,
    // Privacy flow
    SystemCheck,
    ModelSetup,
    // Shared
    AskName,
    AskRole,
    Processing,
    Confirmation,
    Done,
}

#[derive(Clone, Debug)]
enum CheckStatus {
    Pending,
    Running,
    Passed,
    Failed(String),
}

#[derive(Clone, Debug)]
enum DownloadStatus {
    Pending,
    InProgress(u8),
    Done,
    Failed(String),
}

struct DownloadProgress {
    index: usize,
    status: DownloadStatus,
}

const WHISPER_MODELS: &[(LocalModelSize, &str, &str)] = &[
    (LocalModelSize::Turbo, "Turbo (Recommended)", "~1.4 GB"),
    (LocalModelSize::Large, "Large", "~2.9 GB"),
    (LocalModelSize::Medium, "Medium", "~1.5 GB"),
    (LocalModelSize::Small, "Small", "~466 MB"),
];

struct OnboardingState {
    step: OnboardingStep,
    // Text (instant, no typewriter)
    full_text: String,
    visible_chars: usize,
    text_complete: bool,
    // Animation
    anim_frame: usize,
    // Mode selection
    selected_mode: usize,         // 0 = Cloud, 1 = Privacy (local)
    selected_provider: usize,     // 0 = Anthropic, 1 = OpenAI, 2 = Google
    api_key_input: String,
    api_key_valid: Option<bool>,  // None = not checked, Some(true/false) = result
    validation_fail_selection: usize, // 0 = Try different key, 1 = Skip
    validation_task: Option<tokio::task::JoinHandle<Result<bool, anyhow::Error>>>,
    // User inputs
    user_name: String,
    user_role: String,
    // Processing
    /// Returns (world_result, provider_hint). Hint is set when provider is unavailable.
    processing_task: Option<tokio::task::JoinHandle<Result<(Option<(WorldData, WorldEntityExtractionResult)>, Option<String>), anyhow::Error>>>,
    processed_world: Option<WorldData>,
    processed_entities: Option<WorldEntityExtractionResult>,
    ollama_available: bool,
    // Transition
    transition_frame: usize,
    transitioning: bool,
    // Confirmation data (parsed from world)
    confirm_owner: String,
    confirm_role: String,
    confirm_org: String,
    confirm_people: String,
    confirm_selection: usize,     // 0 = Looks good, 1 = Fix

    // ── Privacy flow: SystemCheck ──
    system_checks: Vec<(String, CheckStatus)>,
    system_check_rx: Option<mpsc::UnboundedReceiver<(usize, bool, String)>>,
    system_check_task: Option<tokio::task::JoinHandle<()>>,
    system_check_done: bool,
    system_check_selection: usize,  // 0 = Check again, 1 = Continue anyway
    ollama_reachable: bool,

    // ── Privacy flow: ModelSetup ──
    setup_phase: u8,                    // 0=whisper, 1=ollama, 2=confirm, 3=downloading
    whisper_model_selection: usize,     // index into WHISPER_MODELS
    ollama_model_selection: usize,      // index into available models
    ollama_available_models: Vec<String>,
    ollama_models_fetched: bool,
    download_task: Option<tokio::task::JoinHandle<()>>,
    download_rx: Option<mpsc::UnboundedReceiver<DownloadProgress>>,
    download_items: Vec<(String, DownloadStatus)>,
}

impl OnboardingState {
    fn new() -> Self {
        Self {
            step: OnboardingStep::Entrance,
            full_text: String::new(),
            visible_chars: 0,
            text_complete: false,
            anim_frame: 0,
            selected_mode: 0,
            selected_provider: 0,
            api_key_input: String::new(),
            api_key_valid: None,
            validation_fail_selection: 0,
            validation_task: None,
            user_name: String::new(),
            user_role: String::new(),
            processing_task: None,
            processed_world: None,
            processed_entities: None,
            ollama_available: true,
            transition_frame: 0,
            transitioning: false,
            confirm_owner: String::new(),
            confirm_role: String::new(),
            confirm_org: String::new(),
            confirm_people: String::new(),
            confirm_selection: 0,
            // Privacy flow: SystemCheck
            system_checks: Vec::new(),
            system_check_rx: None,
            system_check_task: None,
            system_check_done: false,
            system_check_selection: 0,
            ollama_reachable: false,
            // Privacy flow: ModelSetup
            setup_phase: 0,
            whisper_model_selection: 0,
            ollama_model_selection: 0,
            ollama_available_models: Vec::new(),
            ollama_models_fetched: false,
            download_task: None,
            download_rx: None,
            download_items: Vec::new(),
        }
    }

    fn set_step_text(&mut self, text: &str, animated: bool) {
        self.full_text = text.to_string();
        if animated {
            self.visible_chars = 0;
            self.text_complete = false;
        } else {
            self.visible_chars = self.full_text.chars().count();
            self.text_complete = true;
        }
    }

    fn tick_typewriter_lines(&mut self) {
        if self.text_complete {
            return;
        }
        // Reveal one line every 2nd tick (~200ms per line)
        if self.anim_frame % 2 != 0 {
            return;
        }
        let total = self.full_text.chars().count();
        // Find the next '\n' at or after visible_chars, advance past it
        let mut found_nl = false;
        for (i, ch) in self.full_text.chars().enumerate() {
            if i >= self.visible_chars && ch == '\n' {
                self.visible_chars = i + 1; // include the newline
                found_nl = true;
                break;
            }
        }
        if !found_nl {
            self.visible_chars = total;
            self.text_complete = true;
        }
    }

    fn visible_text(&self) -> &str {
        let end = self.visible_chars.min(self.full_text.len());
        // Make sure we don't split a multi-byte char
        let mut byte_end = 0;
        for (i, (idx, _)) in self.full_text.char_indices().enumerate() {
            if i >= end {
                break;
            }
            byte_end = idx;
        }
        if end > 0 {
            // Include the last char
            if let Some((_, ch)) = self.full_text.char_indices().nth(end - 1) {
                byte_end += ch.len_utf8();
            }
        }
        &self.full_text[..byte_end]
    }
}

#[derive(Debug, PartialEq)]
enum FileDialogStage {
    FilePath, // Asking for file path
    FileName, // Asking for display name (optional)
}

#[derive(Debug, Clone)]
enum RecordingMode {
    RecordAndTranscribe,
}

#[derive(Debug, PartialEq, Clone)]
enum ModelPickerState {
    Closed,
    Open,
    EditingCustom,
}

#[derive(Debug, Clone)]
struct ModelPickerItem {
    display_name: String,
    /// None means this is the "Custom..." sentinel.
    model_id: Option<String>,
}

struct ActiveTranscription {
    task: tokio::task::JoinHandle<Result<(), anyhow::Error>>,
    recording_name: String,
}

enum PendingTranscription {
    Retranscribe {
        recording_name: String,
        transcription_mode: TranscriptionMode,
    },
    Import {
        source_path: PathBuf,
        display_name: String,
        transcription_mode: TranscriptionMode,
    },
}

impl PendingTranscription {
    fn recording_name(&self) -> &str {
        match self {
            PendingTranscription::Retranscribe { recording_name, .. } => recording_name,
            PendingTranscription::Import { display_name, .. } => display_name,
        }
    }
}

#[derive(Debug, PartialEq)]
enum EntityMode {
    Browse,
    Editing,
    Adding,
    DeleteConfirm,
    MergeSelectTarget,
    MergeConfirm,
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum EntityEditField {
    Name,
    Type,
    Context,
}

const ENTITY_TYPES: &[&str] = &["person", "organization", "project", "other"];

#[derive(Debug)]
enum DashboardAction {
    Continue,
    Quit,
    RecordAndTranscribe,
    AddExternalFile,
    TranscribeSelected,
}

impl Dashboard {
    pub fn new() -> Result<Self> {
        let db = Database::new()?;
        let mut table_state = TableState::default();
        table_state.select(Some(0));
        let config = ScribaConfig::load()?;

        Ok(Self {
            db,
            recordings: Vec::new(),
            table_state,
            current_page: 0,
            page_size: 50, // Show more recordings per page
            stats: None,
            show_help: false,
            current_view: DashboardView::Main,
            search_mode: false,
            search_query: String::new(),
            show_message: false,
            message: String::new(),
            show_transcript: false,
            transcript_content: String::new(),
            show_delete_confirm: false,
            delete_confirm_selection: 1, // default to No
            delete_candidate: None,
            current_playback_pid: None,
            playback_finished_rx: None,
            last_transcribe_warning: None,
            progress_animation: None,
            progress_frame: 0,
            active_transcription: None,
            transcription_queue: VecDeque::new(),
            notification_message: None,
            recording_task: None,
            recording_mode: None,
            recording_stop_tx: None,
            recording_level_rx: None,
            current_volume_level: 0.0,
            recording_start_instant: None,
            volume_history: VecDeque::with_capacity(48),
            config,
            settings_selection: 0,
            editing_api_key: false,
            api_key_input: String::new(),
            model_picker_state: ModelPickerState::Closed,
            model_picker_items: Vec::new(),
            model_picker_selection: 0,
            model_picker_custom_input: String::new(),
            ollama_models_rx: None,
            editing_enrichment_endpoint: false,
            enrichment_endpoint_input: String::new(),
            editing_enrichment_api_key: false,
            enrichment_api_key_input: String::new(),
            return_to_view: None,
            // File import dialog state
            show_file_dialog: false,
            file_path_input: String::new(),
            file_name_input: String::new(),
            file_dialog_stage: FileDialogStage::FilePath,

            // Entity view state
            entities: Vec::new(),
            entity_table_state: TableState::default(),
            selected_entity: None,
            show_entity_detail: false,
            entity_mode: EntityMode::Browse,
            entity_edit_field: EntityEditField::Name,
            entity_edit_name: String::new(),
            entity_edit_type: String::new(),
            entity_edit_context: String::new(),
            merge_source_entity: None,
            confirm_selection: 1, // default to No
            // Add entity state
            entity_add_name: String::new(),
            entity_add_type: "person".to_string(),
            entity_add_context: String::new(),
            entity_add_aliases: String::new(),
            entity_add_field: EntityEditField::Name,
            // Transcript enrichment data
            transcript_summary: None,
            transcript_key_points: None,
            transcript_topics: None,
            transcript_entities: None,

            // Onboarding
            onboarding: None,

            // Voice mode
            voice_command_rx: None,
            voice_detector_handle: None,
            voice_mode_active: false,

            // Chat
            chat: ChatState::new(),
            global_chat_messages: Vec::new(),
            current_transcript_recording: None,

            // Greeting
            greeting_text: String::new(),
            greeting_subtitle: String::new(),
            owner_name: String::new(),
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Load initial data
        self.load_recordings()?;
        self.load_stats()?;

        // Check if onboarding is needed (no world.md exists)
        if !WorldContext::exists() {
            self.current_view = DashboardView::Onboarding;
            self.onboarding = Some(OnboardingState::new());
        } else {
            // Initialize chat context for global view
            self.load_entities().ok();
            self.init_global_chat();
        }

        let result = self.run_app(&mut terminal).await;

        // Shut down voice detector if active
        if let Some(handle) = self.voice_detector_handle.take() {
            handle.shutdown();
        }

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        result
    }

    async fn run_app<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
    ) -> Result<()> {
        let mut last_anim_tick = std::time::Instant::now();
        let anim_interval = Duration::from_millis(100); // animations stay at ~10fps

        'main: loop {
            // ── Animation tick (throttled to ~100ms) ──────────────────────
            let now = std::time::Instant::now();
            let anim_tick = now.duration_since(last_anim_tick) >= anim_interval;
            if anim_tick {
                last_anim_tick = now;
            }

            // Check if recording task completed
            if let Some(task) = &mut self.recording_task {
                if task.is_finished() {
                    let completed_task = self.recording_task.take().unwrap();
                    let recording_mode = self.recording_mode.take();

                    // Clean up channels
                    self.recording_stop_tx = None;
                    self.recording_level_rx = None;
                    self.current_volume_level = 0.0;
                    self.recording_start_instant = None;
                    self.volume_history.clear();

                    match completed_task.await {
                        Ok(Ok(result)) => {
                            let recording_name = result.recording_name;
                            let auto_stopped = result.auto_stopped;

                            if auto_stopped {
                                self.notification_message = Some((
                                    "Silence detected \u{2014} recording stopped automatically.".to_string(),
                                    120,
                                ));
                            }

                            // Recording completed successfully
                            if let Some(RecordingMode::RecordAndTranscribe) = recording_mode {
                                // Dismiss recording modal — transcription runs non-blocking
                                self.stop_progress_animation();
                                self.show_message = false;
                                self.message.clear();
                                let _ = self.load_recordings();
                                let _ = self.load_stats();

                                // Enqueue auto-transcription
                                let transcription_mode = self.config.transcription.clone();
                                self.enqueue_transcription(PendingTranscription::Retranscribe {
                                    recording_name: recording_name.clone(),
                                    transcription_mode,
                                });
                            } else {
                                // Recording only mode - complete
                                self.stop_progress_animation();
                                self.message = "Recording complete.".to_string();
                                self.show_message = true;
                                // Reload data to show new recording
                                let _ = self.load_recordings();
                                let _ = self.load_stats();
                            }
                        }
                        Ok(Err(err)) => {
                            self.stop_progress_animation();
                            self.message = format!("Recording failed: {}", err);
                            self.show_message = true;
                        }
                        Err(_) => {
                            self.stop_progress_animation();
                            self.message = "Recording task failed.".to_string();
                            self.show_message = true;
                        }
                    }
                }
            }

            // Check if active transcription completed
            if let Some(ref active) = self.active_transcription {
                if active.task.is_finished() {
                    let completed = self.active_transcription.take().unwrap();
                    let name = completed.recording_name;
                    match completed.task.await {
                        Ok(Ok(())) => {
                            self.notification_message = Some((
                                format!("Transcription complete: {}", name),
                                30,
                            ));
                            let _ = self.load_recordings();
                            let _ = self.load_stats();
                        }
                        Ok(Err(err)) => {
                            self.notification_message = Some((
                                format!("Transcription failed ({}): {}", name, err),
                                50,
                            ));
                        }
                        Err(_) => {
                            self.notification_message = Some((
                                format!("Transcription task failed: {}", name),
                                50,
                            ));
                        }
                    }
                    // Start next queued transcription
                    self.drain_transcription_queue();
                }
            }

            // Receive volume levels from recording
            if let Some(level_rx) = &mut self.recording_level_rx {
                if let Ok(level) = level_rx.try_recv() {
                    self.current_volume_level = level;
                    // Keep a rolling history for the waveform display
                    if self.volume_history.len() >= 48 {
                        self.volume_history.pop_front();
                    }
                    self.volume_history.push_back(level);
                }
            }

            // Check for playback completion
            if let Some(finished_rx) = &mut self.playback_finished_rx {
                if finished_rx.try_recv().is_ok() {
                    self.current_playback_pid = None;
                    self.playback_finished_rx = None;
                }
            }

            // Check for voice commands
            if let Some(ref mut rx) = self.voice_command_rx {
                if let Ok(cmd) = rx.try_recv() {
                    match cmd {
                        VoiceCommand::Record => {
                            self.handle_voice_record_command().await;
                        }
                        VoiceCommand::Stop => {
                            self.handle_voice_stop_command().await;
                        }
                    }
                }
            }

            // Check for Ollama model list completion
            if let Some(ref mut rx) = self.ollama_models_rx {
                if let Ok(result) = rx.try_recv() {
                    // Check if we're in onboarding ModelSetup phase 1 — populate onboarding models
                    let in_onboarding_model_setup = self.onboarding.as_ref()
                        .map(|ob| ob.step == OnboardingStep::ModelSetup && ob.setup_phase == 1)
                        .unwrap_or(false);

                    if in_onboarding_model_setup {
                        if let Ok(names) = &result {
                            if let Some(ref mut ob) = self.onboarding {
                                ob.ollama_available_models = names.clone();
                                ob.ollama_model_selection = 0;
                            }
                        }
                    } else {
                        let current_model = self.config.enrichment.model_name().to_string();
                        match result {
                            Ok(names) if !names.is_empty() => {
                                let mut items: Vec<ModelPickerItem> = names
                                    .iter()
                                    .map(|n| ModelPickerItem {
                                        display_name: n.clone(),
                                        model_id: Some(n.clone()),
                                    })
                                    .collect();
                                items.push(ModelPickerItem {
                                    display_name: "Custom...".into(),
                                    model_id: None,
                                });
                                let sel = names
                                    .iter()
                                    .position(|n| *n == current_model)
                                    .unwrap_or(items.len() - 1);
                                self.model_picker_items = items;
                                self.model_picker_selection = sel;
                            }
                            _ => {
                                // Error or empty — show only Custom...
                                self.model_picker_items = vec![ModelPickerItem {
                                    display_name: "Custom...".into(),
                                    model_id: None,
                                }];
                                self.model_picker_selection = 0;
                            }
                        }
                    }
                    self.ollama_models_rx = None;
                }
            }

            // Update progress animation if active (throttled)
            if anim_tick && self.progress_animation.is_some() {
                self.update_progress_message();
            }

            // Tick progress frame for inline transcription animation (throttled)
            if anim_tick && (self.active_transcription.is_some() || !self.transcription_queue.is_empty()) {
                self.progress_frame = self.progress_frame.wrapping_add(1);
            }

            // Auto-dismiss notification countdown (throttled)
            if anim_tick {
                if let Some((_, ref mut frames)) = self.notification_message {
                    if *frames == 0 {
                        self.notification_message = None;
                    } else {
                        *frames -= 1;
                    }
                }
            }

            // Onboarding tick logic (runs every frame for typewriter + async polling)
            if let Some(ref mut ob) = self.onboarding {
                if anim_tick {
                    ob.anim_frame = ob.anim_frame.wrapping_add(1);
                }

                match ob.step {
                    OnboardingStep::Entrance => {
                        // Auto-advance after 25 frames (~2.5s)
                        if ob.anim_frame >= 25 {
                            ob.step = OnboardingStep::Intro;
                            ob.set_step_text(
                                "Welcome to Scriba.\n\n\
                                 I listen to your recordings, transcribe them,\n\
                                 and remember everything \u{2014} names, places, topics.\n\
                                 Think of me as your personal note-taker\n\
                                 with a very good memory.\n\n\
                                 Let me get to know you first.",
                                true,
                            );
                        }
                    }
                    OnboardingStep::Intro => {
                        // Line-by-line reveal after logo pause (3 frames = 300ms)
                        if ob.anim_frame >= 3 {
                            ob.tick_typewriter_lines();
                        }
                    }
                    OnboardingStep::AskName | OnboardingStep::AskRole => {
                        ob.tick_typewriter_lines();
                    }
                    OnboardingStep::ModeSelection | OnboardingStep::ProviderSelection
                    | OnboardingStep::ApiKeyEntry => {
                        // Instant text — no typewriter
                    }
                    OnboardingStep::ModelSetup => {
                        // Phase 3: drain download progress
                        if ob.setup_phase == 3 {
                            if let Some(ref mut rx) = ob.download_rx {
                                while let Ok(prog) = rx.try_recv() {
                                    if prog.index < ob.download_items.len() {
                                        ob.download_items[prog.index].1 = prog.status;
                                    }
                                }
                            }
                            if let Some(ref task) = ob.download_task {
                                if task.is_finished() {
                                    ob.download_task.take();
                                    ob.download_rx.take();
                                    for item in ob.download_items.iter_mut() {
                                        if matches!(item.1, DownloadStatus::Pending | DownloadStatus::InProgress(_)) {
                                            item.1 = DownloadStatus::Done;
                                        }
                                    }
                                }
                            }
                        }
                        // Phases 0, 1, 2: instant text, no typewriter
                    }
                    OnboardingStep::SystemCheck => {
                        // Drain system check channel
                        let mut all_resolved = false;
                        if let Some(ref mut rx) = ob.system_check_rx {
                            while let Ok((idx, passed, hint)) = rx.try_recv() {
                                if idx < ob.system_checks.len() {
                                    if hint.is_empty() && !passed {
                                        // "Running" signal (empty hint, false)
                                        ob.system_checks[idx].1 = CheckStatus::Running;
                                    } else if passed {
                                        ob.system_checks[idx].1 = CheckStatus::Passed;
                                        if idx == 2 {
                                            ob.ollama_reachable = true;
                                        }
                                    } else {
                                        ob.system_checks[idx].1 = CheckStatus::Failed(hint);
                                    }
                                }
                            }
                        }
                        // Check if task is done
                        if let Some(ref task) = ob.system_check_task {
                            if task.is_finished() {
                                ob.system_check_task.take();
                                ob.system_check_rx.take();
                                all_resolved = true;
                            }
                        }
                        if all_resolved {
                            let any_failed = ob.system_checks.iter().any(|(_, s)| matches!(s, CheckStatus::Failed(_)));
                            ob.system_check_done = true;
                            if any_failed {
                                ob.system_check_selection = 0;
                            } else {
                                // All passed — start linger counter (will auto-advance after ~1.5s)
                                ob.transition_frame = 0;

                                // Pre-fetch Ollama models while lingering
                                if ob.ollama_reachable {
                                    let endpoint = if let EnrichmentMode::Local { ollama_endpoint, .. } = &self.config.enrichment.mode {
                                        ollama_endpoint.clone()
                                    } else {
                                        "http://localhost:11434".to_string()
                                    };
                                    let (tx, rx) = mpsc::channel(1);
                                    self.ollama_models_rx = Some(rx);
                                    tokio::spawn(async move {
                                        let result = OllamaClient::fetch_models(&endpoint).await;
                                        let _ = tx.send(result.map_err(|e| e.to_string())).await;
                                    });
                                }
                            }
                        }

                        // Linger on all-green for ~1.5s before auto-advancing
                        let all_passed = ob.system_check_done
                            && !ob.system_checks.iter().any(|(_, s)| matches!(s, CheckStatus::Failed(_)));
                        if all_passed && anim_tick {
                            ob.transition_frame += 1;
                            if ob.transition_frame >= 15 {
                                ob.step = OnboardingStep::ModelSetup;
                                ob.anim_frame = 0;
                                ob.setup_phase = 0;
                                ob.transition_frame = 0;
                                ob.set_step_text("Choose your transcription model", false);
                            }
                        }
                    }
                    OnboardingStep::ApiKeyValidation => {
                        // Poll the async validation task
                        if let Some(ref task) = ob.validation_task {
                            if task.is_finished() {
                                let completed = ob.validation_task.take().unwrap();
                                match completed.await {
                                    Ok(Ok(valid)) => {
                                        ob.api_key_valid = Some(valid);
                                        if valid {
                                            ob.step = OnboardingStep::AskName;
                                            ob.anim_frame = 0;
                                            ob.set_step_text(
                                                "API key verified.\n\n\
                                                 What's your name?",
                                                true,
                                            );
                                        } else {
                                            ob.validation_fail_selection = 0;
                                            ob.set_step_text(
                                                "API key validation failed.",
                                                false,
                                            );
                                        }
                                    }
                                    _ => {
                                        ob.api_key_valid = Some(false);
                                        ob.validation_fail_selection = 0;
                                        ob.set_step_text(
                                            "API key validation failed.",
                                            false,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    OnboardingStep::Processing => {
                        // Check if processing task completed
                        if let Some(ref task) = ob.processing_task {
                            if task.is_finished() {
                                let completed = ob.processing_task.take().unwrap();
                                match completed.await {
                                    Ok(Ok((Some((world_data, entities)), _))) => {
                                        // Fill confirmation data
                                        ob.confirm_owner = world_data.owner.name.clone();
                                        ob.confirm_role = world_data.owner.role.clone();
                                        ob.confirm_org = world_data.owner.organization.clone();
                                        let people_names: Vec<String> = world_data.people.iter()
                                            .map(|p| p.name.clone()).collect();
                                        ob.confirm_people = if people_names.is_empty() {
                                            "(none detected)".to_string()
                                        } else {
                                            people_names.join(", ")
                                        };
                                        ob.processed_world = Some(world_data);
                                        ob.processed_entities = Some(entities);
                                        ob.ollama_available = true;
                                        // Advance to confirmation
                                        ob.step = OnboardingStep::Confirmation;
                                        ob.confirm_selection = 0;
                                        ob.set_step_text("Here's what I've got:", false);
                                    }
                                    Ok(Ok((None, hint))) => {
                                        ob.ollama_available = false;
                                        let msg = if let Some(hint) = hint {
                                            format!(
                                                "{}\n\n\
                                                 Your info has been saved.\n\
                                                 You can fix this later in Settings (Ctrl+S).",
                                                hint
                                            )
                                        } else {
                                            "Enrichment provider is not reachable.\n\n\
                                             Your info has been saved.\n\
                                             You can fix this later in Settings (Ctrl+S).".to_string()
                                        };
                                        ob.set_step_text(&msg, false);
                                    }
                                    Ok(Err(e)) => {
                                        ob.ollama_available = false;
                                        let err_msg = format!("{:#}", e);
                                        ob.set_step_text(
                                            &format!(
                                                "Something went wrong during setup:\n\
                                                 {}\n\n\
                                                 Your info has been saved.\n\
                                                 You can fix this later in Settings (Ctrl+S).",
                                                err_msg,
                                            ),
                                            false,
                                        );
                                    }
                                    Err(e) => {
                                        ob.ollama_available = false;
                                        let err_msg = format!("{:#}", e);
                                        ob.set_step_text(
                                            &format!(
                                                "Something went wrong during setup:\n\
                                                 {}\n\n\
                                                 Your info has been saved.\n\
                                                 You can fix this later in Settings (Ctrl+S).",
                                                err_msg,
                                            ),
                                            false,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    OnboardingStep::Confirmation => {
                        // Text is instant now
                    }
                    OnboardingStep::Done => {
                        ob.tick_typewriter_lines();
                        // Fade-out transition (triggered by Enter key, throttled to anim_tick ~100ms)
                        if ob.transitioning && anim_tick {
                            ob.transition_frame += 1;
                            if ob.transition_frame > 10 {
                                // Transition complete — go to main dashboard
                                self.onboarding = None;
                                self.current_view = DashboardView::Main;
                                self.load_entities().ok();
                                self.init_global_chat();
                            }
                        }
                    }
                }
            }

            // Poll chat stream events (non-blocking — every frame for smooth streaming)
            if self.chat.poll_stream() {
                // A pending message is ready to be resent
                let msg = self.chat.pending_message.take().unwrap();
                self.chat.input_buffer = msg;
                self.send_chat_message();
            }

            // Animation counters (throttled to ~10fps)
            if anim_tick {
                self.chat.spinner_frame = self.chat.spinner_frame.wrapping_add(1);


            }

            terminal.draw(|f| self.ui(f))?;

            // Process all pending events (drain queue for smooth scrolling)
            while event::poll(Duration::from_millis(0))? {
                match event::read()? {
                    Event::Key(key) => {
                        if key.kind == KeyEventKind::Press {
                            // Ctrl+C quits from anywhere
                            if key.code == KeyCode::Char('c')
                                && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
                            {
                                break 'main;
                            }
                            match self.handle_key_event(key.code, key.modifiers).await {
                                Ok(DashboardAction::Continue) => {}
                                Ok(DashboardAction::Quit) => break 'main,
                                Ok(action) => {
                                    self.handle_dashboard_action(action).await?;
                                }
                                Err(e) => return Err(e),
                            }
                        }
                    }
                    Event::Mouse(mouse) => {
                        self.handle_mouse_event(mouse);
                    }
                    _ => {}
                }
            }

            // Sleep to target ~30fps when no events pending
            tokio::time::sleep(Duration::from_millis(16)).await;
        }
        Ok(())
    }

    async fn handle_key_event(&mut self, key_code: KeyCode, modifiers: crossterm::event::KeyModifiers) -> Result<DashboardAction> {
        // If audio is playing and ESC is pressed, stop it immediately
        if matches!(key_code, KeyCode::Esc) {
            // Check if we're in audio playback mode (either with PID or with playback message)
            let is_playing_audio = self.current_playback_pid.is_some()
                || (self.show_message && self.message.contains("Playing:"));

            if is_playing_audio {
                // Try both methods to ensure reliable stopping
                if let Some(pid) = self.current_playback_pid {
                    self.stop_audio_playback(pid)?;
                }
                // Also use emergency stop as a fallback (in case PID method fails)
                self.emergency_stop_all_audio_players()?;

                // Clear playback state
                self.current_playback_pid = None;
                self.playback_finished_rx = None;
                self.show_message = false;
                self.message.clear();

                // Audio playback stops immediately - return to dashboard
                return Ok(DashboardAction::Continue);
            }
        }

        // If recording is active and Escape is pressed, stop recording
        if self.recording_task.is_some() && matches!(key_code, KeyCode::Esc) {
            // Send stop signal to recording task
            if let Some(stop_tx) = self.recording_stop_tx.take() {
                let _ = stop_tx.send(()).await;
            }
            // The recording task will handle cleanup and completion
            return Ok(DashboardAction::Continue);
        }

        // Onboarding key handling
        if self.current_view == DashboardView::Onboarding {
            return self.handle_onboarding_keys(key_code).await;
        }

        if self.show_file_dialog {
            return self.handle_file_dialog_keys(key_code).await;
        }

        // Dismiss notification on any keypress (without consuming the key)
        if self.notification_message.is_some() {
            self.notification_message = None;
        }

        if self.show_message {
            // Special-case: allow confirming re-transcribe overwrite with T while message is visible
            if matches!(key_code, KeyCode::Char('t') | KeyCode::Char('T'))
                && self.last_transcribe_warning.is_some()
                && self.progress_animation.is_none()
            {
                // Dismiss the warning and trigger the action
                self.show_message = false;
                self.message.clear();
                return Ok(DashboardAction::TranscribeSelected);
            }

            // Don't close message if progress animation is active
            if self.progress_animation.is_none() && matches!(key_code, KeyCode::Esc) {
                // Only Esc key closes the message popup (consistent behavior)
                self.show_message = false;
                self.message.clear();

                // Return to the previous view if one was set
                if let Some(return_view) = self.return_to_view.take() {
                    self.current_view = return_view;
                }
            }
            return Ok(DashboardAction::Continue);
        }

        if self.search_mode {
            return self.handle_search_input(key_code).await;
        }

        if self.show_help {
            self.show_help = false;
            self.current_view = DashboardView::Main;
            return Ok(DashboardAction::Continue);
        }

        if self.current_view == DashboardView::Settings {
            return self.handle_settings_keys(key_code).await;
        }

        if self.current_view == DashboardView::Entities {
            return self.handle_entities_keys(key_code).await;
        }

        if self.show_transcript {
            // Ctrl+Y: copy full transcript (works regardless of focus)
            if key_code == KeyCode::Char('y')
                && modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
            {
                match self.copy_transcript_to_clipboard() {
                    Ok(()) => {
                        self.notification_message = Some(("Transcript copied to clipboard".to_string(), 40));
                    }
                    Err(e) => {
                        self.notification_message = Some((format!("Copy failed: {}", e), 40));
                    }
                }
                return Ok(DashboardAction::Continue);
            }

            // Ctrl+T: re-transcribe (works regardless of focus)
            if key_code == KeyCode::Char('t')
                && modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
            {
                if let Some(recording) = self.get_current_recording() {
                    if self.has_active_transcription() {
                        self.message = "Transcription already in progress".to_string();
                        self.show_message = true;
                    } else {
                        self.show_transcript = false;
                        self.transcript_content.clear();
                        let transcription_mode = self.config.transcription.clone();
                        let directory_name = recording.directory_name.clone();
                        self.enqueue_transcription(PendingTranscription::Retranscribe {
                            recording_name: directory_name,
                            transcription_mode,
                        });
                    }
                }
                return Ok(DashboardAction::Continue);
            }

            // Chat keys in transcript view take priority when chat is focused
            if self.handle_transcript_chat_key(key_code) {
                return Ok(DashboardAction::Continue);
            }
            return self.handle_transcript_keys(key_code).await;
        }

        if self.show_delete_confirm {
            return self.handle_delete_confirmation(key_code).await;
        }

        // Browse view keys
        if self.current_view == DashboardView::Browse {
            return self.handle_browse_keys(key_code).await;
        }

        // Chat key handling (Tab → Browse, chat input when focused)
        // Global Ctrl+ shortcuts (work from any Main view state)
        let ctrl = modifiers.contains(crossterm::event::KeyModifiers::CONTROL);
        if ctrl {
            match key_code {
                KeyCode::Char('s') => {
                    self.current_view = DashboardView::Settings;
                    self.settings_selection = 0;
                    return Ok(DashboardAction::Continue);
                }
                KeyCode::Char('w') => {
                    self.load_entities()?;
                    self.current_view = DashboardView::Entities;
                    self.entity_table_state.select(Some(0));
                    return Ok(DashboardAction::Continue);
                }
                KeyCode::Char('r') => {
                    return Ok(DashboardAction::RecordAndTranscribe);
                }
                _ => {}
            }
        }

        // F1 / ? for help (? only when input is empty to avoid conflicts)
        if matches!(key_code, KeyCode::F(1))
            || (key_code == KeyCode::Char('?') && self.chat.input_buffer.is_empty())
        {
            self.show_help = true;
            self.current_view = DashboardView::Help;
            return Ok(DashboardAction::Continue);
        }

        if self.handle_chat_key(key_code) {
            return Ok(DashboardAction::Continue);
        }

        Ok(DashboardAction::Continue)
    }

    async fn handle_browse_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Tab | KeyCode::Esc => {
                self.current_view = DashboardView::Main;
                self.chat.focus = ChatFocus::ChatInput;
                self.chat.borderless = true;
                if self.chat.messages.is_empty() {
                    self.chat.show_home_screen = true;
                    self.chat.show_suggestions = true;
                }
            }
            KeyCode::Up => {
                self.previous_recording();
            }
            KeyCode::Down => {
                self.next_recording();
            }
            KeyCode::PageUp | KeyCode::Char('[') => {
                self.previous_page().await?;
            }
            KeyCode::PageDown | KeyCode::Char(']') => {
                self.next_page().await?;
            }
            KeyCode::Enter => match self.show_selected_transcript().await {
                Ok(()) => {}
                Err(e) => {
                    self.message = format!("Failed to load transcript: {}", e);
                    self.show_message = true;
                }
            },
            KeyCode::Char('r') | KeyCode::Char('R') => {
                return Ok(DashboardAction::RecordAndTranscribe);
            }
            KeyCode::Char('t') | KeyCode::Char('T') => {
                return Ok(DashboardAction::TranscribeSelected);
            }
            KeyCode::Char('p') | KeyCode::Char('P') => match self.play_selected_recording().await {
                Ok(()) => {}
                Err(e) => {
                    self.message = format!("Failed to play recording: {}", e);
                    self.show_message = true;
                }
            },
            KeyCode::Char('d') | KeyCode::Delete => {
                self.show_delete_confirmation();
            }
            KeyCode::Char('/') => {
                self.search_mode = true;
                self.search_query.clear();
            }
            _ => {}
        }
        Ok(DashboardAction::Continue)
    }

    async fn handle_dashboard_action(&mut self, action: DashboardAction) -> Result<()> {
        match action {
            DashboardAction::RecordAndTranscribe => {
                self.execute_record_and_transcribe().await?;
            }
            DashboardAction::AddExternalFile => {
                self.execute_add_external_file().await?;
            }
            DashboardAction::TranscribeSelected => {
                self.execute_transcribe_selected().await?;
            }
            _ => {}
        }
        Ok(())
    }

    // ── Transcription queue helpers ──────────────────────────────────────

    fn has_active_transcription(&self) -> bool {
        self.active_transcription.is_some()
    }

    fn is_transcription_pending_or_active(&self, dir_name: &str) -> bool {
        if let Some(ref active) = self.active_transcription {
            if active.recording_name == dir_name {
                return true;
            }
        }
        self.transcription_queue
            .iter()
            .any(|p| p.recording_name() == dir_name)
    }

    fn drain_transcription_queue(&mut self) {
        if self.active_transcription.is_some() {
            return;
        }
        if let Some(pending) = self.transcription_queue.pop_front() {
            let name = pending.recording_name().to_string();
            let task = match pending {
                PendingTranscription::Retranscribe {
                    recording_name,
                    transcription_mode,
                } => tokio::spawn(async move {
                    let mut workflow = WorkflowManager::new().unwrap();
                    workflow
                        .retranscribe_recording_silent(&recording_name, transcription_mode)
                        .await
                }),
                PendingTranscription::Import {
                    source_path,
                    display_name,
                    transcription_mode,
                } => tokio::spawn(async move {
                    let mut workflow = WorkflowManager::new().unwrap();
                    workflow
                        .complete_import_workflow_silent(
                            &source_path,
                            Some(display_name),
                            Some(transcription_mode),
                        )
                        .await
                        .map(|_| ())
                }),
            };
            self.active_transcription = Some(ActiveTranscription {
                task,
                recording_name: name,
            });
            self.progress_frame = 0;
        }
    }

    fn enqueue_transcription(&mut self, pending: PendingTranscription) {
        let name = pending.recording_name().to_string();
        if self.is_transcription_pending_or_active(&name) {
            return;
        }
        self.transcription_queue.push_back(pending);
        self.drain_transcription_queue();
    }

    fn load_recordings(&mut self) -> Result<()> {
        let offset = (self.current_page * self.page_size) as i64;

        self.recordings = if self.search_query.is_empty() {
            self.db
                .list_recordings(Some(self.page_size as i64), Some(offset))?
        } else {
            let search_results = self.db.search_transcripts(&self.search_query, None)?;
            search_results
                .into_iter()
                .map(|(recording, _)| recording)
                .collect()
        };

        if !self.recordings.is_empty() {
            self.table_state.select(Some(0));
        } else {
            self.table_state.select(None);
        }

        // Refresh greeting with updated recording data
        if !self.owner_name.is_empty() {
            self.generate_greeting();
        }

        Ok(())
    }

    fn load_stats(&mut self) -> Result<()> {
        self.stats = Some(self.db.get_stats()?);
        Ok(())
    }

    fn load_entities(&mut self) -> Result<()> {
        self.entities = self.db.list_entities(None, None)?;
        if !self.entities.is_empty() && self.entity_table_state.selected().is_none() {
            self.entity_table_state.select(Some(0));
        }
        Ok(())
    }

    async fn handle_entities_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match self.entity_mode {
            EntityMode::Browse => {
                if self.show_entity_detail {
                    match key_code {
                        KeyCode::Esc => {
                            self.show_entity_detail = false;
                            self.selected_entity = None;
                        }
                        _ => {}
                    }
                    return Ok(DashboardAction::Continue);
                }

                match key_code {
                    KeyCode::Esc | KeyCode::Char('q') => {
                        self.current_view = DashboardView::Main;
                        self.show_entity_detail = false;
                        self.selected_entity = None;
                    }
                    KeyCode::Up => {
                        self.entity_navigate_up();
                    }
                    KeyCode::Down => {
                        self.entity_navigate_down();
                    }
                    KeyCode::Enter => {
                        if let Some(idx) = self.entity_table_state.selected() {
                            if let Some(entity) = self.entities.get(idx) {
                                self.selected_entity = Some(entity.clone());
                                self.show_entity_detail = true;
                            }
                        }
                    }
                    KeyCode::Char('e') | KeyCode::Char('E') => {
                        self.start_entity_edit();
                    }
                    KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Delete => {
                        if let Some(idx) = self.entity_table_state.selected() {
                            if let Some(entity) = self.entities.get(idx) {
                                self.selected_entity = Some(entity.clone());
                                self.confirm_selection = 1; // default to No
                                self.entity_mode = EntityMode::DeleteConfirm;
                            }
                        }
                    }
                    KeyCode::Char('m') | KeyCode::Char('M') => {
                        if let Some(idx) = self.entity_table_state.selected() {
                            if let Some(entity) = self.entities.get(idx) {
                                self.merge_source_entity = Some(entity.clone());
                                self.entity_mode = EntityMode::MergeSelectTarget;
                            }
                        }
                    }
                    KeyCode::Char('a') | KeyCode::Char('A') => {
                        self.start_entity_add();
                    }
                    KeyCode::Char('r') | KeyCode::Char('R') => {
                        self.load_entities()?;
                    }
                    _ => {}
                }
            }
            EntityMode::Adding => {
                match key_code {
                    KeyCode::Esc => {
                        self.entity_mode = EntityMode::Browse;
                    }
                    KeyCode::Tab | KeyCode::Down => {
                        self.entity_add_field = match self.entity_add_field {
                            EntityEditField::Name => EntityEditField::Type,
                            EntityEditField::Type => EntityEditField::Context,
                            EntityEditField::Context => EntityEditField::Name,
                        };
                    }
                    KeyCode::BackTab | KeyCode::Up => {
                        self.entity_add_field = match self.entity_add_field {
                            EntityEditField::Name => EntityEditField::Context,
                            EntityEditField::Type => EntityEditField::Name,
                            EntityEditField::Context => EntityEditField::Type,
                        };
                    }
                    KeyCode::Enter => {
                        if self.entity_add_field == EntityEditField::Type {
                            // Cycle type on Enter
                            let current_idx = ENTITY_TYPES.iter().position(|t| *t == self.entity_add_type).unwrap_or(0);
                            self.entity_add_type = ENTITY_TYPES[(current_idx + 1) % ENTITY_TYPES.len()].to_string();
                        } else if !self.entity_add_name.trim().is_empty() {
                            // Save if name is not empty
                            self.save_entity_add()?;
                            self.entity_mode = EntityMode::Browse;
                        }
                    }
                    KeyCode::Char(' ') if self.entity_add_field == EntityEditField::Type => {
                        let current_idx = ENTITY_TYPES.iter().position(|t| *t == self.entity_add_type).unwrap_or(0);
                        self.entity_add_type = ENTITY_TYPES[(current_idx + 1) % ENTITY_TYPES.len()].to_string();
                    }
                    KeyCode::Backspace => {
                        match self.entity_add_field {
                            EntityEditField::Name => { self.entity_add_name.pop(); }
                            EntityEditField::Type => {}
                            EntityEditField::Context => { self.entity_add_context.pop(); }
                        }
                    }
                    KeyCode::Char(c) => {
                        match self.entity_add_field {
                            EntityEditField::Name => self.entity_add_name.push(c),
                            EntityEditField::Type => {}
                            EntityEditField::Context => self.entity_add_context.push(c),
                        }
                    }
                    _ => {}
                }
            }
            EntityMode::Editing => {
                match key_code {
                    KeyCode::Esc => {
                        self.save_entity_edit()?;
                        self.entity_mode = EntityMode::Browse;
                    }
                    KeyCode::Tab | KeyCode::Down => {
                        self.entity_edit_field = match self.entity_edit_field {
                            EntityEditField::Name => EntityEditField::Type,
                            EntityEditField::Type => EntityEditField::Context,
                            EntityEditField::Context => EntityEditField::Name,
                        };
                    }
                    KeyCode::BackTab | KeyCode::Up => {
                        self.entity_edit_field = match self.entity_edit_field {
                            EntityEditField::Name => EntityEditField::Context,
                            EntityEditField::Type => EntityEditField::Name,
                            EntityEditField::Context => EntityEditField::Type,
                        };
                    }
                    KeyCode::Enter => {
                        if self.entity_edit_field == EntityEditField::Type {
                            self.cycle_entity_type();
                        } else {
                            // Move to next field
                            self.entity_edit_field = match self.entity_edit_field {
                                EntityEditField::Name => EntityEditField::Type,
                                EntityEditField::Type => EntityEditField::Context,
                                EntityEditField::Context => EntityEditField::Name,
                            };
                        }
                    }
                    KeyCode::Char(' ') if self.entity_edit_field == EntityEditField::Type => {
                        self.cycle_entity_type();
                    }
                    KeyCode::Char(c) => {
                        match self.entity_edit_field {
                            EntityEditField::Name => self.entity_edit_name.push(c),
                            EntityEditField::Context => self.entity_edit_context.push(c),
                            EntityEditField::Type => {} // Type is cycled, not typed
                        }
                    }
                    KeyCode::Backspace => {
                        match self.entity_edit_field {
                            EntityEditField::Name => { self.entity_edit_name.pop(); }
                            EntityEditField::Context => { self.entity_edit_context.pop(); }
                            EntityEditField::Type => {}
                        }
                    }
                    _ => {}
                }
            }
            EntityMode::DeleteConfirm => {
                match key_code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        self.confirm_selection = 0;
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        self.confirm_selection = 1;
                    }
                    KeyCode::Enter => {
                        if self.confirm_selection == 0 {
                            self.perform_entity_delete()?;
                        } else {
                            self.selected_entity = None;
                        }
                        self.entity_mode = EntityMode::Browse;
                    }
                    KeyCode::Esc => {
                        self.selected_entity = None;
                        self.entity_mode = EntityMode::Browse;
                    }
                    _ => {}
                }
            }
            EntityMode::MergeSelectTarget => {
                match key_code {
                    KeyCode::Up => {
                        self.entity_navigate_up();
                    }
                    KeyCode::Down => {
                        self.entity_navigate_down();
                    }
                    KeyCode::Enter => {
                        if let Some(idx) = self.entity_table_state.selected() {
                            if let Some(target) = self.entities.get(idx) {
                                let source_id = self.merge_source_entity.as_ref()
                                    .and_then(|e| e.id);
                                if source_id != target.id {
                                    self.selected_entity = Some(target.clone());
                                    self.confirm_selection = 1; // default to No
                                    self.entity_mode = EntityMode::MergeConfirm;
                                }
                            }
                        }
                    }
                    KeyCode::Esc => {
                        self.merge_source_entity = None;
                        self.entity_mode = EntityMode::Browse;
                    }
                    _ => {}
                }
            }
            EntityMode::MergeConfirm => {
                match key_code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        self.confirm_selection = 0;
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        self.confirm_selection = 1;
                    }
                    KeyCode::Enter => {
                        if self.confirm_selection == 0 {
                            self.perform_entity_merge()?;
                        } else {
                            self.selected_entity = None;
                            self.merge_source_entity = None;
                        }
                        self.entity_mode = EntityMode::Browse;
                    }
                    KeyCode::Esc => {
                        self.selected_entity = None;
                        self.merge_source_entity = None;
                        self.entity_mode = EntityMode::Browse;
                    }
                    _ => {}
                }
            }
        }
        Ok(DashboardAction::Continue)
    }

    fn entity_navigate_up(&mut self) {
        let i = match self.entity_table_state.selected() {
            Some(i) => if i == 0 { self.entities.len().saturating_sub(1) } else { i - 1 },
            None => 0,
        };
        self.entity_table_state.select(Some(i));
    }

    fn entity_navigate_down(&mut self) {
        let i = match self.entity_table_state.selected() {
            Some(i) => if i >= self.entities.len().saturating_sub(1) { 0 } else { i + 1 },
            None => 0,
        };
        self.entity_table_state.select(Some(i));
    }

    fn start_entity_edit(&mut self) {
        if let Some(idx) = self.entity_table_state.selected() {
            if let Some(entity) = self.entities.get(idx) {
                self.selected_entity = Some(entity.clone());
                self.entity_edit_name = entity.canonical_name.clone();
                self.entity_edit_type = entity.entity_type.clone();
                self.entity_edit_context = entity.context.clone().unwrap_or_default();
                self.entity_edit_field = EntityEditField::Name;
                self.entity_mode = EntityMode::Editing;
            }
        }
    }

    fn cycle_entity_type(&mut self) {
        let current_idx = ENTITY_TYPES.iter()
            .position(|t| *t == self.entity_edit_type)
            .unwrap_or(0);
        let next_idx = (current_idx + 1) % ENTITY_TYPES.len();
        self.entity_edit_type = ENTITY_TYPES[next_idx].to_string();
    }

    fn save_entity_edit(&mut self) -> Result<()> {
        if let Some(entity) = &self.selected_entity {
            let entity_id = match entity.id {
                Some(id) => id,
                None => return Ok(()),
            };

            let mut registry = EntityRegistry::new(&mut self.db);

            // Update name if changed
            if self.entity_edit_name != entity.canonical_name && !self.entity_edit_name.is_empty() {
                registry.rename_entity(entity_id, &self.entity_edit_name)?;
            }

            // Update type if changed
            if self.entity_edit_type != entity.entity_type {
                registry.update_entity_type(entity_id, &self.entity_edit_type)?;
            }

            // Update context if changed
            let old_context = entity.context.as_deref().unwrap_or("");
            if self.entity_edit_context != old_context {
                let ctx = if self.entity_edit_context.is_empty() { "" } else { &self.entity_edit_context };
                registry.update_entity_context(entity_id, ctx)?;
            }

            drop(registry);
            let _ = rebuild_world_from_entities(&self.db);
            self.load_entities()?;
            self.selected_entity = None;
        }
        Ok(())
    }

    fn perform_entity_delete(&mut self) -> Result<()> {
        if let Some(entity) = &self.selected_entity {
            if let Some(id) = entity.id {
                self.db.delete_entity(id)?;
                let _ = rebuild_world_from_entities(&self.db);
                self.load_entities()?;
                // Adjust selection if needed
                if let Some(selected) = self.entity_table_state.selected() {
                    if selected >= self.entities.len() && !self.entities.is_empty() {
                        self.entity_table_state.select(Some(self.entities.len() - 1));
                    }
                }
            }
        }
        self.selected_entity = None;
        Ok(())
    }

    fn perform_entity_merge(&mut self) -> Result<()> {
        let source_id = self.merge_source_entity.as_ref().and_then(|e| e.id);
        let target_id = self.selected_entity.as_ref().and_then(|e| e.id);

        if let (Some(src), Some(tgt)) = (source_id, target_id) {
            let mut registry = EntityRegistry::new(&mut self.db);
            registry.merge_entities(src, tgt)?;
            drop(registry);
            let _ = rebuild_world_from_entities(&self.db);
            self.load_entities()?;
        }

        self.merge_source_entity = None;
        self.selected_entity = None;
        Ok(())
    }

    fn start_entity_add(&mut self) {
        self.entity_add_name.clear();
        self.entity_add_type = "person".to_string();
        self.entity_add_context.clear();
        self.entity_add_aliases.clear();
        self.entity_add_field = EntityEditField::Name;
        self.entity_mode = EntityMode::Adding;
    }

    fn save_entity_add(&mut self) -> Result<()> {
        if self.entity_add_name.trim().is_empty() {
            return Ok(());
        }
        let mut registry = EntityRegistry::new(&mut self.db);
        let entity = registry.create_entity(
            self.entity_add_type.trim(),
            self.entity_add_name.trim(),
            if self.entity_add_context.trim().is_empty() {
                None
            } else {
                Some(self.entity_add_context.trim())
            },
        )?;
        // Add aliases if provided
        if let Some(id) = entity.id {
            for alias in self.entity_add_aliases.split(',') {
                let alias = alias.trim();
                if !alias.is_empty() {
                    registry.add_entity_alias(id, alias)?;
                }
            }
        }
        drop(registry);
        let _ = rebuild_world_from_entities(&self.db);
        self.load_entities()?;
        // Select the new entity (should be last or find by name)
        if let Some(pos) = self.entities.iter().position(|e| e.canonical_name == self.entity_add_name.trim()) {
            self.entity_table_state.select(Some(pos));
        }
        Ok(())
    }

    async fn handle_search_input(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Esc => {
                self.search_mode = false;
                self.search_query.clear();
                self.load_recordings()?;
            }
            KeyCode::Enter => {
                self.search_mode = false;
                self.load_recordings()?;
            }
            KeyCode::Backspace => {
                self.search_query.pop();
            }
            KeyCode::Char(c) => {
                self.search_query.push(c);
            }
            _ => {}
        }
        Ok(DashboardAction::Continue)
    }

    fn next_recording(&mut self) {
        let i = match self.table_state.selected() {
            Some(i) => {
                if i >= self.recordings.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    fn previous_recording(&mut self) {
        let i = match self.table_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.recordings.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    async fn next_page(&mut self) -> Result<()> {
        // Try to load next page - if it has recordings, advance
        let old_page = self.current_page;
        self.current_page += 1;
        self.load_recordings()?;

        // If no recordings found on next page, go back to previous page
        if self.recordings.is_empty() {
            self.current_page = old_page;
            self.load_recordings()?;
        }

        Ok(())
    }

    async fn previous_page(&mut self) -> Result<()> {
        if self.current_page > 0 {
            self.current_page -= 1;
            self.load_recordings()?;
        }
        Ok(())
    }

    async fn show_selected_transcript(&mut self) -> Result<()> {
        if let Some(selected) = self.table_state.selected() {
            if let Some(recording) = self.recordings.get(selected).cloned() {
                if recording.has_transcript {
                    match self.load_transcript_content(&recording) {
                        Ok(content) => {
                            self.transcript_content = content;
                            self.show_transcript = true;
                            // Load enrichment data
                            self.load_enrichment_data(&recording);
                            // Initialize recording chat context
                            self.init_recording_chat(&recording);
                        }
                        Err(e) => {
                            self.message = format!("Failed to load transcript: {}", e);
                            self.show_message = true;
                        }
                    }
                } else {
                    self.message =
                        "No transcript available for this recording. Use P to play instead."
                            .to_string();
                    self.show_message = true;
                }
            }
        }
        Ok(())
    }

    fn load_enrichment_data(&mut self, recording: &Recording) {
        // Load summary and key points from recording
        self.transcript_summary = recording.summary.clone();
        self.transcript_key_points = recording
            .key_points
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok());

        // Load topics and entities from transcript
        if let Some(id) = recording.id {
            if let Ok(Some(transcript)) = self.db.get_transcript_by_recording_id(id) {
                self.transcript_topics = transcript
                    .topics
                    .as_ref()
                    .and_then(|s| serde_json::from_str(s).ok());

                // Parse entities JSON: [{"name": "...", "type": "..."}]
                self.transcript_entities = transcript.entities.as_ref().and_then(|s| {
                    let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(s);
                    parsed.ok().map(|entities| {
                        entities
                            .iter()
                            .filter_map(|e| {
                                let name = e.get("name")?.as_str()?.to_string();
                                let entity_type = e.get("type")?.as_str()?.to_string();
                                Some((name, entity_type))
                            })
                            .collect()
                    })
                });
            }
        }
    }

    fn clear_enrichment_data(&mut self) {
        self.transcript_summary = None;
        self.transcript_key_points = None;
        self.transcript_topics = None;
        self.transcript_entities = None;
    }

    async fn play_selected_recording(&mut self) -> Result<()> {
        use anyhow::anyhow;
        if let Some(selected) = self.table_state.selected() {
            if let Some(recording) = self.recordings.get(selected) {
                // Locate the audio file in ~/scriba_recordings/<directory_name>/
                let audio_path = self
                    .find_audio_file(recording)
                    .ok_or_else(|| anyhow!("Could not find an audio file for this recording"))?;

                // Determine file extension to choose optimal players
                let is_mp3 = audio_path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.to_lowercase() == "mp3")
                    .unwrap_or(false);

                // Candidate players differ by platform. For MP3 files, prioritize mpv/ffplay over afplay
                #[cfg(target_os = "macos")]
                let candidates: Vec<(&str, &[&str])> = if is_mp3 {
                    vec![
                        ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                        (
                            "ffplay",
                            &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"],
                        ),
                        ("afplay", &[]), // Last resort for MP3
                    ]
                } else {
                    vec![
                        ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                        (
                            "ffplay",
                            &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"],
                        ),
                        ("afplay", &[]), // Works well with WAV
                    ]
                };

                #[cfg(all(unix, not(target_os = "macos")))]
                let candidates: Vec<(&str, &[&str])> = vec![
                    ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                    (
                        "ffplay",
                        &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"],
                    ),
                    ("aplay", &["-c", "2"]), // Force stereo output
                ];

                #[cfg(target_os = "windows")]
                let candidates: Vec<(&str, &[&str])> = vec![
                    ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                    (
                        "ffplay",
                        &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"],
                    ),
                    (
                        "powershell",
                        &["-NoProfile", "-Command", "(New-Object Media.SoundPlayer '"],
                    ), // will be handled specially
                ];

                // Try each candidate until one spawns successfully
                let mut launched_with: Option<String> = None;

                #[cfg(not(target_os = "windows"))]
                for (prog, base_args) in candidates {
                    let mut cmd = TokioCommand::new(prog);
                    // Detach from TTY so player doesn't consume keyboard (Esc) input
                    cmd.stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null());

                    // For afplay on macOS, check if this is a mono WAV file and needs special handling
                    if prog == "afplay" && recording.channels == 1 && !is_mp3 {
                        // Create a temporary stereo version of the mono WAV file
                        if let Ok(stereo_path) = self.create_stereo_temp_file(&audio_path).await {
                            cmd.arg(stereo_path);
                        } else {
                            // Fallback to original mono file
                            cmd.arg(&audio_path);
                        }
                    } else {
                        for a in base_args {
                            cmd.arg(a);
                        }
                        cmd.arg(&audio_path);
                    }

                    match cmd.spawn() {
                        Ok(mut child) => {
                            launched_with = Some(prog.to_string());

                            // Store child process for potential termination - ensure we have a valid PID
                            if let Some(child_id) = child.id() {
                                // Store the process ID for killing on key press immediately
                                self.current_playback_pid = Some(child_id);

                                // Create channel for playback completion notification
                                let (finished_tx, finished_rx) = mpsc::channel(1);
                                self.playback_finished_rx = Some(finished_rx);

                                tokio::spawn(async move {
                                    let _ = child.wait().await;
                                    let _ = finished_tx.send(()).await;
                                });
                                break;
                            } else {
                                // If we can't get PID, we can't control the process
                                launched_with = None;
                                continue;
                            }
                        }
                        Err(e) => {
                            // Store error for debugging if no player works
                            if prog == "mpv" && is_mp3 {
                                self.message = format!("mpv failed to play MP3: {}", e);
                            }
                        }
                    }
                }

                #[cfg(target_os = "windows")]
                {
                    // Try standard players first (mpv, ffplay), then fallback to PowerShell
                    for (prog, base_args) in &candidates[..candidates.len() - 1] {
                        // All except powershell
                        let mut cmd = TokioCommand::new(prog);
                        // Detach from TTY so player doesn't consume keyboard (Esc) input
                        cmd.stdin(Stdio::null())
                            .stdout(Stdio::null())
                            .stderr(Stdio::null());
                        for a in base_args {
                            cmd.arg(a);
                        }
                        cmd.arg(&audio_path);
                        match cmd.spawn() {
                            Ok(mut child) => {
                                if let Some(child_id) = child.id() {
                                    launched_with = Some(prog.to_string());
                                    self.current_playback_pid = Some(child_id);

                                    // Create channel for playback completion notification
                                    let (finished_tx, finished_rx) = mpsc::channel(1);
                                    self.playback_finished_rx = Some(finished_rx);

                                    tokio::spawn(async move {
                                        let _ = child.wait().await;
                                        let _ = finished_tx.send(()).await;
                                    });
                                    break;
                                }
                            }
                            Err(_e) => continue,
                        }
                    }

                    // PowerShell SoundPlayer fallback if no other player worked
                    if launched_with.is_none() {
                        let escaped = audio_path.to_string_lossy().replace("'", "''");
                        let ps =
                            format!("$p=New-Object Media.SoundPlayer '{}';$p.Play();", escaped);
                        let mut pscmd = TokioCommand::new("powershell");
                        // Detach from TTY so player doesn't consume keyboard (Esc) input
                        pscmd
                            .stdin(Stdio::null())
                            .stdout(Stdio::null())
                            .stderr(Stdio::null());
                        match pscmd.arg("-NoProfile").arg("-Command").arg(ps).spawn() {
                            Ok(mut child) => {
                                if let Some(child_id) = child.id() {
                                    launched_with = Some("powershell".to_string());
                                    self.current_playback_pid = Some(child_id);

                                    // Create channel for playback completion notification
                                    let (finished_tx, finished_rx) = mpsc::channel(1);
                                    self.playback_finished_rx = Some(finished_rx);

                                    tokio::spawn(async move {
                                        let _ = child.wait().await;
                                        let _ = finished_tx.send(()).await;
                                    });
                                }
                            }
                            Err(_e) => {}
                        }
                    }
                }

                if let Some(player) = launched_with {
                    let name = recording
                        .display_name
                        .as_ref()
                        .unwrap_or(&recording.directory_name);
                    self.message = format!(
                        "▶ Playing: {}\nUsing player: {}\n\nPress ESC to stop playback",
                        name, player
                    );
                    self.show_message = true;
                    return Ok(());
                }

                // If we reach here, no player succeeded
                #[cfg(target_os = "macos")]
                let hint = "Install `mpv` (brew install mpv) or ensure `afplay` is available.";
                #[cfg(all(unix, not(target_os = "macos")))]
                let hint = "Install `mpv` or `ffmpeg` (ffplay).";
                #[cfg(target_os = "windows")]
                let hint = "Ensure PowerShell is available or install a player like mpv.";

                Err(anyhow!("No audio player found on PATH. {}", hint))
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    fn find_audio_file(&self, recording: &Recording) -> Option<PathBuf> {
        let base_path = home_dir()?
            .join("scriba_recordings")
            .join(&recording.directory_name);
        if !base_path.exists() {
            return None;
        }
        let exts = [
            "wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "aiff", "aif", "caf",
        ];
        if let Ok(read_dir) = std::fs::read_dir(base_path) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if exts.iter().any(|x| x.eq_ignore_ascii_case(ext)) {
                        return Some(path);
                    }
                }
            }
        }
        None
    }

    fn get_current_recording(&self) -> Option<Recording> {
        let selected_index = self.table_state.selected()?;
        self.recordings.get(selected_index).cloned()
    }

    fn show_delete_confirmation(&mut self) {
        if let Some(selected) = self.table_state.selected() {
            if let Some(recording) = self.recordings.get(selected).cloned() {
                self.delete_candidate = Some(recording);
                self.delete_confirm_selection = 1; // default to No
                self.show_delete_confirm = true;
            }
        }
    }

    async fn handle_transcript_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Esc => {
                self.show_transcript = false;
                self.transcript_content.clear();
                self.clear_enrichment_data();
                self.restore_global_chat();
                Ok(DashboardAction::Continue)
            }
            _ => Ok(DashboardAction::Continue),
        }
    }

    fn is_editing_settings_field(&self) -> bool {
        self.editing_api_key || self.model_picker_state != ModelPickerState::Closed || self.editing_enrichment_endpoint || self.editing_enrichment_api_key
    }

    fn save_enrichment_config(&mut self) -> Result<()> {
        self.config.save()?;
        self.config = ScribaConfig::load()?;
        Ok(())
    }

    fn close_model_picker(&mut self) {
        self.model_picker_state = ModelPickerState::Closed;
        self.model_picker_items.clear();
        self.model_picker_selection = 0;
        self.model_picker_custom_input.clear();
        self.ollama_models_rx = None;
    }

    fn open_model_picker(&mut self) {
        let current_model = self.config.enrichment.model_name().to_string();

        match &self.config.enrichment.mode {
            EnrichmentMode::Cloud { provider, .. } => {
                let curated = provider.available_models();
                let mut items: Vec<ModelPickerItem> = curated
                    .iter()
                    .map(|m| ModelPickerItem {
                        display_name: m.display_name.clone(),
                        model_id: Some(m.model_id.clone()),
                    })
                    .collect();
                items.push(ModelPickerItem {
                    display_name: "Custom...".into(),
                    model_id: None,
                });

                // Pre-select current model, or "Custom..." if not in the list
                let sel = curated
                    .iter()
                    .position(|m| m.model_id == current_model)
                    .unwrap_or(items.len() - 1);

                self.model_picker_items = items;
                self.model_picker_selection = sel;
                self.model_picker_state = ModelPickerState::Open;
            }
            EnrichmentMode::Local { ollama_endpoint, .. } => {
                // Show a loading placeholder while we fetch
                self.model_picker_items = vec![ModelPickerItem {
                    display_name: "Loading...".into(),
                    model_id: None,
                }];
                self.model_picker_selection = 0;
                self.model_picker_state = ModelPickerState::Open;

                let endpoint = ollama_endpoint.clone();
                let (tx, rx) = mpsc::channel(1);
                self.ollama_models_rx = Some(rx);

                tokio::spawn(async move {
                    let result = OllamaClient::fetch_models(&endpoint)
                        .await
                        .map_err(|e| e.to_string());
                    let _ = tx.send(result).await;
                });
            }
        }
    }

    async fn handle_settings_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        // Max settings index: 0=Mode, 1=ModeSpecific,
        //                    2=EnrichProvider, 3=EnrichModel, 4=EnrichKeyOrEndpoint,
        //                    5=SilenceAutoStop, 6=SilenceTimeout,
        //                    7=Diarization, 8=MaxSpeakers,
        //                    9=VoiceMode, 10=VoiceSensitivity
        let max_index = 10;

        match key_code {
            KeyCode::Esc => {
                if self.model_picker_state != ModelPickerState::Closed {
                    self.close_model_picker();
                } else if self.is_editing_settings_field() {
                    self.editing_api_key = false;
                    self.editing_enrichment_endpoint = false;
                    self.editing_enrichment_api_key = false;
                } else {
                    self.current_view = DashboardView::Main;
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Up => {
                if self.model_picker_state == ModelPickerState::Open {
                    self.model_picker_selection = self.model_picker_selection.saturating_sub(1);
                } else if !self.is_editing_settings_field() {
                    self.settings_selection = self.settings_selection.saturating_sub(1);
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Down => {
                if self.model_picker_state == ModelPickerState::Open {
                    if !self.model_picker_items.is_empty() {
                        self.model_picker_selection = std::cmp::min(
                            self.model_picker_selection + 1,
                            self.model_picker_items.len() - 1,
                        );
                    }
                } else if !self.is_editing_settings_field() {
                    self.settings_selection = std::cmp::min(self.settings_selection + 1, max_index);
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Enter => {
                if self.editing_api_key {
                    // Save API key
                    let new_mode = TranscriptionMode::Api {
                        api_key: self.api_key_input.clone(),
                    };
                    match self.config.set_transcription_mode(new_mode) {
                        Ok(()) => {
                            self.config = ScribaConfig::load()?;
                        }
                        Err(e) => {
                            self.message = format!("Failed to save API key: {}", e);
                            self.show_message = true;
                            self.return_to_view = Some(DashboardView::Settings);
                        }
                    }
                    self.editing_api_key = false;
                    self.api_key_input.clear();
                } else if self.model_picker_state == ModelPickerState::Open {
                    if let Some(item) = self.model_picker_items.get(self.model_picker_selection) {
                        if item.display_name == "Loading..." {
                            // no-op while loading
                        } else if let Some(ref id) = item.model_id {
                            // Selected a concrete model — save it
                            let new_model = id.clone();
                            match &mut self.config.enrichment.mode {
                                EnrichmentMode::Cloud { model, .. } => {
                                    *model = Some(new_model);
                                }
                                EnrichmentMode::Local { ollama_model, .. } => {
                                    *ollama_model = new_model;
                                }
                            }
                            if let Err(e) = self.save_enrichment_config() {
                                self.message = format!("Failed to save enrichment model: {}", e);
                                self.show_message = true;
                                self.return_to_view = Some(DashboardView::Settings);
                            }
                            self.close_model_picker();
                        } else {
                            // "Custom..." sentinel — switch to custom text entry
                            self.model_picker_state = ModelPickerState::EditingCustom;
                            self.model_picker_custom_input = self.config.enrichment.model_name().to_string();
                        }
                    }
                } else if self.model_picker_state == ModelPickerState::EditingCustom {
                    let new_model = self.model_picker_custom_input.trim().to_string();
                    if !new_model.is_empty() {
                        match &mut self.config.enrichment.mode {
                            EnrichmentMode::Cloud { model, .. } => {
                                *model = Some(new_model);
                            }
                            EnrichmentMode::Local { ollama_model, .. } => {
                                *ollama_model = new_model;
                            }
                        }
                        if let Err(e) = self.save_enrichment_config() {
                            self.message = format!("Failed to save enrichment model: {}", e);
                            self.show_message = true;
                            self.return_to_view = Some(DashboardView::Settings);
                        }
                    }
                    self.close_model_picker();
                } else if self.editing_enrichment_endpoint {
                    let new_endpoint = self.enrichment_endpoint_input.trim().to_string();
                    if !new_endpoint.is_empty() {
                        self.config.enrichment.set_ollama_endpoint(new_endpoint);
                        if let Err(e) = self.save_enrichment_config() {
                            self.message = format!("Failed to save Ollama endpoint: {}", e);
                            self.show_message = true;
                            self.return_to_view = Some(DashboardView::Settings);
                        }
                    }
                    self.editing_enrichment_endpoint = false;
                    self.enrichment_endpoint_input.clear();
                } else if self.editing_enrichment_api_key {
                    let new_key = self.enrichment_api_key_input.trim().to_string();
                    // Clone the provider before mutating
                    let provider_clone = self.config.enrichment.cloud_provider().cloned();
                    if let EnrichmentMode::Cloud { api_key, .. } = &mut self.config.enrichment.mode {
                        *api_key = new_key.clone();
                    }
                    if let Some(ref p) = provider_clone {
                        self.config.enrichment.save_key_for_provider(p, &new_key);
                    }
                    if let Err(e) = self.save_enrichment_config() {
                        self.message = format!("Failed to save API key: {}", e);
                        self.show_message = true;
                        self.return_to_view = Some(DashboardView::Settings);
                    }
                    self.editing_enrichment_api_key = false;
                    self.enrichment_api_key_input.clear();
                } else {
                    match self.settings_selection {
                        0 => {
                            // Toggle transcription mode
                            let new_mode = match &self.config.transcription {
                                TranscriptionMode::Local { .. } => {
                                    let api_key = self
                                        .config
                                        .last_api_key
                                        .as_ref()
                                        .map(|key| key.clone())
                                        .unwrap_or_else(String::new);
                                    TranscriptionMode::Api { api_key }
                                }
                                TranscriptionMode::Api { .. } => TranscriptionMode::Local {
                                    model_size: LocalModelSize::Medium,
                                },
                            };
                            match self.config.set_transcription_mode(new_mode) {
                                Ok(()) => {
                                    self.config = ScribaConfig::load()?;
                                    self.settings_selection = 0;
                                }
                                Err(e) => {
                                    self.message = format!("Failed to change mode: {}", e);
                                    self.show_message = true;
                                    self.return_to_view = Some(DashboardView::Settings);
                                }
                            }
                        }
                        1 => {
                            match &self.config.transcription {
                                TranscriptionMode::Local { model_size } => {
                                    let new_model = match model_size {
                                        LocalModelSize::Tiny => LocalModelSize::Base,
                                        LocalModelSize::Base => LocalModelSize::Small,
                                        LocalModelSize::Small => LocalModelSize::Medium,
                                        LocalModelSize::Medium => LocalModelSize::Large,
                                        LocalModelSize::Large => LocalModelSize::Turbo,
                                        LocalModelSize::Turbo => LocalModelSize::Tiny,
                                    };
                                    let new_mode = TranscriptionMode::Local {
                                        model_size: new_model,
                                    };
                                    match self.config.set_transcription_mode(new_mode) {
                                        Ok(()) => {
                                            self.config = ScribaConfig::load()?;
                                        }
                                        Err(e) => {
                                            self.message =
                                                format!("Failed to change model: {}", e);
                                            self.show_message = true;
                                            self.return_to_view = Some(DashboardView::Settings);
                                        }
                                    }
                                }
                                TranscriptionMode::Api { .. } => {
                                    self.editing_api_key = true;
                                    self.api_key_input = match &self.config.transcription {
                                        TranscriptionMode::Api { api_key } => api_key.clone(),
                                        _ => String::new(),
                                    };
                                }
                            }
                        }
                        2 => {
                            // Enrichment Provider — cycle: Anthropic → OpenAI → Google → Ollama → Anthropic
                            // Close picker if open (prevents stale state)
                            self.close_model_picker();
                            // Clone current state to avoid borrow issues
                            let (cur_provider, cur_key, cur_ollama) = match &self.config.enrichment.mode {
                                EnrichmentMode::Cloud { provider, api_key, .. } => {
                                    (Some(provider.clone()), api_key.clone(), None)
                                }
                                EnrichmentMode::Local { ollama_endpoint, ollama_model } => {
                                    (None, String::new(), Some((ollama_endpoint.clone(), ollama_model.clone())))
                                }
                            };
                            // Save current settings before switching
                            if let Some(ref p) = cur_provider {
                                self.config.enrichment.save_key_for_provider(p, &cur_key);
                            }
                            if let Some((ref ep, ref mdl)) = cur_ollama {
                                self.config.enrichment.last_ollama_endpoint = Some(ep.clone());
                                self.config.enrichment.last_ollama_model = Some(mdl.clone());
                            }
                            let new_mode = match cur_provider {
                                Some(CloudProvider::Anthropic) => {
                                    let next_key = self.config.enrichment.load_key_for_provider(&CloudProvider::OpenAI);
                                    EnrichmentMode::Cloud {
                                        provider: CloudProvider::OpenAI,
                                        api_key: next_key,
                                        model: None,
                                    }
                                }
                                Some(CloudProvider::OpenAI) => {
                                    let next_key = self.config.enrichment.load_key_for_provider(&CloudProvider::Google);
                                    EnrichmentMode::Cloud {
                                        provider: CloudProvider::Google,
                                        api_key: next_key,
                                        model: None,
                                    }
                                }
                                Some(CloudProvider::Google) => {
                                    // Google → Ollama (Local) — restore previous settings if available
                                    let ep = self.config.enrichment.last_ollama_endpoint.clone()
                                        .unwrap_or_else(|| "http://localhost:11434".to_string());
                                    let mdl = self.config.enrichment.last_ollama_model.clone()
                                        .unwrap_or_else(|| "mistral:latest".to_string());
                                    EnrichmentMode::Local {
                                        ollama_endpoint: ep,
                                        ollama_model: mdl,
                                    }
                                }
                                None => {
                                    // Ollama → Anthropic
                                    let next_key = self.config.enrichment.load_key_for_provider(&CloudProvider::Anthropic);
                                    EnrichmentMode::Cloud {
                                        provider: CloudProvider::Anthropic,
                                        api_key: next_key,
                                        model: None,
                                    }
                                }
                            };
                            self.config.enrichment.mode = new_mode;
                            if let Err(e) = self.save_enrichment_config() {
                                self.message = format!("Failed to save provider: {}", e);
                                self.show_message = true;
                                self.return_to_view = Some(DashboardView::Settings);
                            }
                        }
                        3 => {
                            // Enrichment Model — open picker
                            self.open_model_picker();
                        }
                        4 => {
                            // Enrichment API Key (cloud) or Ollama Endpoint (local)
                            if self.config.enrichment.is_local() {
                                self.editing_enrichment_endpoint = true;
                                self.enrichment_endpoint_input = self.config.enrichment.ollama_endpoint();
                            } else {
                                self.editing_enrichment_api_key = true;
                                self.enrichment_api_key_input = self.config.enrichment.api_key().unwrap_or("").to_string();
                            }
                        }
                        5 => {
                            // Toggle silence auto-stop enabled/disabled
                            self.config.silence_auto_stop.enabled = !self.config.silence_auto_stop.enabled;
                            if let Err(e) = self.config.save() {
                                self.message = format!("Failed to save setting: {}", e);
                                self.show_message = true;
                                self.return_to_view = Some(DashboardView::Settings);
                            } else {
                                self.config = ScribaConfig::load()?;
                            }
                        }
                        6 => {
                            // Cycle silence timeout: 30s → 60s → 120s → 300s
                            if self.config.silence_auto_stop.enabled {
                                self.config.silence_auto_stop.timeout_seconds = match self.config.silence_auto_stop.timeout_seconds {
                                    30 => 60,
                                    60 => 120,
                                    120 => 300,
                                    _ => 30,
                                };
                                if let Err(e) = self.config.save() {
                                    self.message = format!("Failed to save setting: {}", e);
                                    self.show_message = true;
                                    self.return_to_view = Some(DashboardView::Settings);
                                } else {
                                    self.config = ScribaConfig::load()?;
                                }
                            }
                        }
                        7 => {
                            // Toggle speaker diarization enabled/disabled
                            self.config.diarization.enabled = !self.config.diarization.enabled;
                            if let Err(e) = self.config.save() {
                                self.message = format!("Failed to save setting: {}", e);
                                self.show_message = true;
                                self.return_to_view = Some(DashboardView::Settings);
                            } else {
                                self.config = ScribaConfig::load()?;
                            }
                        }
                        8 => {
                            // Cycle max speakers: 2 → 4 → 6 → 8
                            if self.config.diarization.enabled {
                                self.config.diarization.max_speakers = match self.config.diarization.max_speakers {
                                    2 => 4,
                                    4 => 6,
                                    6 => 8,
                                    _ => 2,
                                };
                                if let Err(e) = self.config.save() {
                                    self.message = format!("Failed to save setting: {}", e);
                                    self.show_message = true;
                                    self.return_to_view = Some(DashboardView::Settings);
                                } else {
                                    self.config = ScribaConfig::load()?;
                                }
                            }
                        }
                        9 => {
                            // Toggle voice mode on/off
                            self.toggle_voice_mode().await;
                            self.config.voice.enabled = self.voice_mode_active;
                            if let Err(e) = self.config.save() {
                                self.message = format!("Failed to save setting: {}", e);
                                self.show_message = true;
                                self.return_to_view = Some(DashboardView::Settings);
                            }
                        }
                        10 => {
                            // Cycle voice sensitivity: 0.005 → 0.01 → 0.02 → 0.05
                            if self.voice_mode_active {
                                self.config.voice.vad_threshold = match self.config.voice.vad_threshold {
                                    t if t <= 0.005 => 0.01,
                                    t if t <= 0.01 => 0.02,
                                    t if t <= 0.02 => 0.05,
                                    _ => 0.005,
                                };
                                if let Err(e) = self.config.save() {
                                    self.message = format!("Failed to save setting: {}", e);
                                    self.show_message = true;
                                    self.return_to_view = Some(DashboardView::Settings);
                                } else {
                                    self.config = ScribaConfig::load()?;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Char(c) => {
                if self.editing_api_key {
                    self.api_key_input.push(c);
                } else if self.model_picker_state == ModelPickerState::EditingCustom {
                    self.model_picker_custom_input.push(c);
                } else if self.editing_enrichment_endpoint {
                    self.enrichment_endpoint_input.push(c);
                } else if self.editing_enrichment_api_key {
                    self.enrichment_api_key_input.push(c);
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Backspace => {
                if self.editing_api_key {
                    self.api_key_input.pop();
                } else if self.model_picker_state == ModelPickerState::EditingCustom {
                    self.model_picker_custom_input.pop();
                } else if self.editing_enrichment_endpoint {
                    self.enrichment_endpoint_input.pop();
                } else if self.editing_enrichment_api_key {
                    self.enrichment_api_key_input.pop();
                }
                Ok(DashboardAction::Continue)
            }
            _ => Ok(DashboardAction::Continue),
        }
    }

    async fn handle_delete_confirmation(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.delete_confirm_selection = 0;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.delete_confirm_selection = 1;
            }
            KeyCode::Enter => {
                if self.delete_confirm_selection == 0 {
                    if let Some(recording) = self.delete_candidate.take() {
                        match self.perform_delete_recording(recording).await {
                            Ok(()) => {}
                            Err(e) => {
                                self.message = format!("Failed to delete recording: {}", e);
                                self.show_message = true;
                            }
                        }
                    }
                }
                self.show_delete_confirm = false;
                self.delete_candidate = None;
            }
            KeyCode::Esc => {
                self.show_delete_confirm = false;
                self.delete_candidate = None;
            }
            _ => {}
        }
        Ok(DashboardAction::Continue)
    }

    async fn handle_file_dialog_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Esc => {
                // Cancel file import
                self.show_file_dialog = false;
                self.file_path_input.clear();
                self.file_name_input.clear();
                self.file_dialog_stage = FileDialogStage::FilePath;
                Ok(DashboardAction::Continue)
            }
            KeyCode::Enter => {
                match self.file_dialog_stage {
                    FileDialogStage::FilePath => {
                        // Validate file path
                        if self.file_path_input.trim().is_empty() {
                            self.message = "Please enter a file path".to_string();
                            self.show_message = true;
                            self.return_to_view = Some(DashboardView::Main);
                            self.show_file_dialog = false;
                            return Ok(DashboardAction::Continue);
                        }

                        // Check if file exists
                        let file_path = PathBuf::from(self.file_path_input.trim());
                        if !file_path.exists() {
                            self.message = "File not found. Please check the path.".to_string();
                            self.show_message = true;
                            self.return_to_view = Some(DashboardView::Main);
                            self.show_file_dialog = false;
                            return Ok(DashboardAction::Continue);
                        }

                        // Move to name input stage
                        self.file_dialog_stage = FileDialogStage::FileName;
                        // Pre-fill with file stem as default name
                        if let Some(stem) = file_path.file_stem().and_then(|s| s.to_str()) {
                            self.file_name_input = stem.to_string();
                        }
                    }
                    FileDialogStage::FileName => {
                        // Use file name or default to file stem
                        let display_name = if self.file_name_input.trim().is_empty() {
                            let file_path = PathBuf::from(self.file_path_input.trim());
                            file_path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("imported_audio")
                                .to_string()
                        } else {
                            self.file_name_input.trim().to_string()
                        };

                        // Start import process
                        self.show_file_dialog = false;
                        self.start_file_import(self.file_path_input.clone(), display_name)
                            .await?;
                    }
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Char(c) => {
                // Add character to current input
                match self.file_dialog_stage {
                    FileDialogStage::FilePath => {
                        self.file_path_input.push(c);
                    }
                    FileDialogStage::FileName => {
                        self.file_name_input.push(c);
                    }
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Backspace => {
                // Remove character from current input
                match self.file_dialog_stage {
                    FileDialogStage::FilePath => {
                        self.file_path_input.pop();
                    }
                    FileDialogStage::FileName => {
                        self.file_name_input.pop();
                    }
                }
                Ok(DashboardAction::Continue)
            }
            _ => Ok(DashboardAction::Continue),
        }
    }

    async fn perform_delete_recording(&mut self, recording: Recording) -> Result<()> {
        if let Some(id) = recording.id {
            match self.db.delete_recording(id) {
                Ok(()) => {
                    let base_path = home_dir()
                        .context("Could not find home directory")?
                        .join("scriba_recordings");
                    let recording_dir = base_path.join(&recording.directory_name);

                    if recording_dir.exists() {
                        std::fs::remove_dir_all(&recording_dir).ok();
                    }

                    self.load_recordings()?;
                    self.load_stats()?;
                    self.message = "Recording deleted.".to_string();
                    self.show_message = true;
                }
                Err(_e) => {
                    return Err(anyhow::anyhow!(
                        "Could not delete recording (ID: {}).\nHint: This often happens when there are related rows (e.g., transcripts) without ON DELETE CASCADE. Delete dependents first or enable cascading, then retry.",
                        id
                    ));
                }
            }
        } else {
            return Err(anyhow::anyhow!(
                "Selected recording has no database ID; cannot delete."
            ));
        }
        Ok(())
    }

    fn load_transcript_content(&self, recording: &Recording) -> Result<String> {
        // First try to load from database
        if let Some(id) = recording.id {
            if let Ok(Some(transcript)) = self.db.get_transcript_by_recording_id(id) {
                return Ok(transcript.content);
            }
        }

        // Fallback: try to load from file (standard transcript.txt)
        let base_path = home_dir()
            .context("Could not find home directory")?
            .join("scriba_recordings");
        let recording_dir = base_path.join(&recording.directory_name);

        // Try transcript.txt
        let transcript_path = recording_dir.join("transcript.txt");
        if transcript_path.exists() {
            return std::fs::read_to_string(&transcript_path)
                .context("Failed to read transcript.txt file");
        }

        Err(anyhow::anyhow!("No transcript file found (expected transcript.txt)"))
    }

    fn copy_transcript_to_clipboard(&self) -> Result<()> {
        use arboard::Clipboard;

        let mut clipboard = Clipboard::new().context("Failed to access clipboard")?;

        clipboard
            .set_text(&self.transcript_content)
            .context("Failed to copy text to clipboard")?;

        Ok(())
    }

    fn ui(&mut self, f: &mut Frame) {
        match self.current_view {
            DashboardView::Main => self.render_main_dashboard(f),
            DashboardView::Browse => self.render_browse_view(f),
            DashboardView::Help => self.render_help(f, f.size()),
            DashboardView::Settings => self.render_settings(f, f.size()),
            DashboardView::Entities => self.render_entities_view(f, f.size()),
            DashboardView::Onboarding => self.render_onboarding(f, f.size()),
        }
    }

    fn render_main_dashboard(&mut self, f: &mut Frame) {
        let size = f.size();

        if self.recording_task.is_some() {
            self.render_recording_view(f, size);
            return;
        }

        if self.show_file_dialog {
            self.render_file_dialog_popup(f, size);
            return;
        }

        if self.show_message {
            self.render_message_popup(f, size);
            return;
        }

        if self.show_transcript {
            self.render_transcript_popup(f, size);
            return;
        }

        if self.show_delete_confirm {
            self.render_delete_confirmation_popup(f, size);
            return;
        }

        // Center-constrain only the home screen; full width once chatting
        let content_area = if self.chat.show_home_screen && self.chat.messages.is_empty() {
            let max_width: u16 = 90;
            let h_pad = size.width.saturating_sub(max_width) / 2;
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(h_pad),
                    Constraint::Min(1),
                    Constraint::Length(h_pad),
                ])
                .split(size)[1]
        } else {
            size
        };

        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(6),   // Chat
                Constraint::Length(2), // Footer
            ])
            .split(content_area);

        self.chat.render(f, main_chunks[0]);
        self.render_home_footer(f, main_chunks[1]);

        if self.search_mode {
            self.render_search_input(f, size);
        }
    }

    fn render_browse_view(&mut self, f: &mut Frame) {
        let size = f.size();

        if self.recording_task.is_some() {
            self.render_recording_view(f, size);
            return;
        }

        if self.show_file_dialog {
            self.render_file_dialog_popup(f, size);
            return;
        }

        if self.show_message {
            self.render_message_popup(f, size);
            return;
        }

        if self.show_transcript {
            self.render_transcript_popup(f, size);
            return;
        }

        if self.show_delete_confirm {
            self.render_delete_confirmation_popup(f, size);
            return;
        }

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(6),   // Recording list
                Constraint::Length(2), // Footer (separator + line)
            ])
            .split(size);

        self.render_recording_list(f, chunks[0]);
        self.render_browse_footer(f, chunks[1]);

        if self.search_mode {
            self.render_search_input(f, size);
        }
    }

    fn render_home_footer(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        // Align footer with chat box: 1 char left pad, 3 chars right pad
        let aligned = Rect {
            x: area.x + 1,
            width: area.width.saturating_sub(4),
            ..area
        };

        // Notification override
        if let Some((ref msg, _)) = self.notification_message {
            let is_error = msg.contains("failed") || msg.contains("Failed");
            let style = if is_error {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            };
            let para = Paragraph::new(msg.as_str())
                .style(style)
                .alignment(Alignment::Center);
            f.render_widget(para, aligned);
            return;
        }

        // Separator line
        let sep = "\u{2500}".repeat(aligned.width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: aligned.x, y: aligned.y, width: aligned.width, height: 1 },
        );

        // Footer content area (below separator)
        let footer_area = if aligned.height > 1 {
            Rect { x: aligned.x, y: aligned.y + 1, width: aligned.width, height: aligned.height - 1 }
        } else {
            aligned
        };

        // Left side: ▸ scriba · v0.21.2
        let version = env!("CARGO_PKG_VERSION");
        let left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(format!("scriba \u{00B7} v{}", version), Style::default().fg(Color::DarkGray)),
        ];

        // Right side: shortcuts (context-dependent)
        let in_chat = !self.chat.show_home_screen && !self.chat.messages.is_empty();
        let mut right_spans: Vec<Span> = Vec::new();
        if in_chat {
            right_spans.extend([
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Esc", Style::default().fg(Color::White)),
                Span::styled("] Home  ", Style::default().fg(Color::DarkGray)),
            ]);
        }
        right_spans.extend([
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Tab", Style::default().fg(Color::White)),
            Span::styled("] Browse  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Ctrl+R", Style::default().fg(Color::White)),
            Span::styled("] Record  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("?", Style::default().fg(Color::White)),
            Span::styled("] Help ", Style::default().fg(Color::DarkGray)),
        ]);

        // Compute widths to fill gap with spaces
        let left_width: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_width: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_area.width as usize).saturating_sub(left_width + right_width);

        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);

        let line = Line::from(spans);
        f.render_widget(Paragraph::new(line), footer_area);
    }

    fn render_browse_footer(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        // Notification override
        if let Some((ref msg, _)) = self.notification_message {
            let is_error = msg.contains("failed") || msg.contains("Failed");
            let style = if is_error {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            };
            let para = Paragraph::new(msg.as_str())
                .style(style)
                .alignment(Alignment::Center);
            f.render_widget(para, area);
            return;
        }

        // Separator line
        let sep = "\u{2500}".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: area.x, y: area.y, width: area.width, height: 1 },
        );

        // Footer content area (below separator)
        let footer_area = if area.height > 1 {
            Rect { x: area.x, y: area.y + 1, width: area.width, height: area.height - 1 }
        } else {
            area
        };

        // Left side: ▸ scriba · v0.21.2 · N recordings · Xh Ym
        let version = env!("CARGO_PKG_VERSION");
        let rec_count = self.stats.as_ref().map(|s| s.total_recordings).unwrap_or(0);
        let total_secs = self.stats.as_ref().map(|s| s.total_duration_seconds).unwrap_or(0);
        let total_h = total_secs / 3600;
        let total_m = (total_secs % 3600) / 60;
        let duration_str = if total_h > 0 {
            format!("{}h {}m", total_h, total_m)
        } else {
            format!("{}m", total_m)
        };

        let mut left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(
                format!("scriba \u{00B7} v{} \u{00B7} {} recordings \u{00B7} {}", version, rec_count, duration_str),
                Style::default().fg(Color::DarkGray),
            ),
        ];

        // Transcribing indicator
        if self.active_transcription.is_some() {
            left_spans.push(Span::styled(
                " \u{00B7} Transcribing...",
                Style::default().fg(Color::Yellow),
            ));
        }

        // Right side: shortcuts
        let right_spans = vec![
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Tab", Style::default().fg(Color::White)),
            Span::styled("] Home  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Ctrl+R", Style::default().fg(Color::White)),
            Span::styled("] Record  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("/", Style::default().fg(Color::White)),
            Span::styled("] Search ", Style::default().fg(Color::DarkGray)),
        ];

        // Compute widths to fill gap
        let left_width: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_width: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_area.width as usize).saturating_sub(left_width + right_width);

        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);

        let line = Line::from(spans);
        f.render_widget(Paragraph::new(line), footer_area);
    }


    fn render_recording_list(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        if self.recordings.is_empty() {
            let msg = if self.search_query.is_empty() {
                "No recordings yet. Press Ctrl+R to start recording."
            } else {
                "No recordings match your search."
            };
            let para = Paragraph::new(msg)
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            f.render_widget(para, area);
            return;
        }

        let selected_idx = self.table_state.selected().unwrap_or(0);
        let today = chrono::Local::now().date_naive();
        let yesterday = today - chrono::Duration::days(1);

        // Build lines and track which line maps to which recording index
        let mut lines: Vec<Line> = Vec::new();
        let mut line_to_recording: Vec<Option<usize>> = Vec::new();
        let mut current_group: Option<String> = None;

        let name_col_width = (area.width as usize).saturating_sub(16); // leave room for prefix + duration + time

        for (i, recording) in self.recordings.iter().enumerate() {
            let rec_date = recording.created_at.with_timezone(&chrono::Local).date_naive();
            let group_label = if rec_date == today {
                "Today".to_string()
            } else if rec_date == yesterday {
                "Yesterday".to_string()
            } else {
                rec_date.format("%b %-d").to_string()
            };

            if current_group.as_ref() != Some(&group_label) {
                // Blank line before group (except first)
                if current_group.is_some() {
                    lines.push(Line::from(""));
                    line_to_recording.push(None);
                }
                // Date header
                lines.push(Line::from(Span::styled(
                    format!("  {}", group_label),
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
                )));
                line_to_recording.push(None);
                current_group = Some(group_label);
            }

            let is_selected = i == selected_idx;
            let is_active = self.active_transcription.as_ref()
                .is_some_and(|a| a.recording_name == recording.directory_name);
            let is_queued = !is_active
                && self.transcription_queue.iter().any(|p| p.recording_name() == recording.directory_name);

            // Selection marker
            let marker = if is_selected { "\u{25B8} " } else { "  " };
            let marker_style = Style::default().fg(Color::White);

            // Status dot
            let (dot, dot_style) = if is_active {
                let spinner = match self.progress_frame % 4 {
                    0 => "\u{25D0}",
                    1 => "\u{25D3}",
                    2 => "\u{25D1}",
                    _ => "\u{25D2}",
                };
                (spinner, Style::default().fg(Color::Yellow))
            } else if is_queued {
                ("\u{25CB}", Style::default().fg(Color::Yellow))
            } else if recording.has_transcript {
                ("\u{25CF}", Style::default().fg(ACCENT))
            } else {
                ("\u{25CB}", Style::default().fg(Color::DarkGray))
            };

            // Name
            let display_name = recording.display_name.as_ref()
                .unwrap_or(&recording.directory_name);
            let name_style = if is_selected {
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Indexed(249))
            };

            // Duration (short format)
            let duration = recording.duration_seconds
                .map(|d| {
                    let mins = d / 60;
                    if mins >= 60 {
                        format!("{}h{}m", mins / 60, mins % 60)
                    } else {
                        format!("{}m", mins)
                    }
                })
                .unwrap_or_default();

            // Time
            let time = recording.created_at
                .with_timezone(&chrono::Local)
                .format("%H:%M")
                .to_string();

            // Truncate name to fit
            let right_part = format!("{:>6}  {}", duration, time);
            let max_name = name_col_width.saturating_sub(right_part.len() + 6); // 6 = "  ▸ ● "
            let truncated_name: String = if display_name.chars().count() > max_name {
                display_name.chars().take(max_name.saturating_sub(1)).collect::<String>() + "\u{2026}"
            } else {
                display_name.clone()
            };
            let name_padding = max_name.saturating_sub(truncated_name.chars().count());

            let spans = vec![
                Span::styled(format!("  {}", marker), marker_style),
                Span::styled(format!("{} ", dot), dot_style),
                Span::styled(truncated_name, name_style),
                Span::raw(" ".repeat(name_padding)),
                Span::styled(right_part, Style::default().fg(Color::DarkGray)),
            ];

            lines.push(Line::from(spans));
            line_to_recording.push(Some(i));
        }

        // Find the line index for the selected recording to keep it in view
        let selected_line = line_to_recording.iter()
            .position(|r| *r == Some(selected_idx))
            .unwrap_or(0);

        // Calculate scroll offset
        let visible_height = area.height as usize;
        let scroll = if selected_line >= visible_height {
            selected_line - visible_height + 2 // keep 1 line below selected visible
        } else {
            0
        };

        let para = Paragraph::new(lines).scroll((scroll as u16, 0));
        f.render_widget(para, area);
    }

    fn render_help(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(area)[1];

        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let help_text = vec![
            Line::from("SCRIBA HELP"),
            Line::from(""),
            Line::from("Global Shortcuts:"),
            Line::from("  Ctrl+R     - Record audio (Esc to stop)"),
            Line::from("  Tab        - Browse recordings"),
            Line::from("  Ctrl+S     - Settings"),
            Line::from("  Ctrl+W     - World (entities)"),
            Line::from("  ?/F1       - Show this help"),
            Line::from("  Esc        - Back / Quit"),
            Line::from(""),
            Line::from("Home:"),
            Line::from("  Type in the chat box to ask questions about your recordings"),
            Line::from(""),
            Line::from("Browse:"),
            Line::from("  \u{2191}/\u{2193}        - Navigate recordings"),
            Line::from("  PgUp/PgDn  - Change pages"),
            Line::from("  Enter      - View transcript"),
            Line::from("  P          - Play recording"),
            Line::from("  D          - Delete recording"),
            Line::from("  /          - Search recordings"),
            Line::from("  A          - Add external audio file"),
            Line::from(""),
            Line::from("Transcript Viewer:"),
            Line::from("  \u{2191}/\u{2193}        - Scroll"),
            Line::from("  Ctrl+Y     - Copy transcript"),
            Line::from("  Ctrl+T     - Re-transcribe"),
            Line::from("  Esc        - Close"),
            Line::from(""),
            Line::from("Press Esc to close this help."),
        ];

        let help_paragraph = Paragraph::new(help_text)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Yellow))
                    .title("Help"),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(help_paragraph, popup_area);
    }

    fn render_settings(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        // Full-screen borderless layout: header + body + footer
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // Header
                Constraint::Min(10),   // Body
                Constraint::Length(2),  // Footer
            ])
            .split(area);

        // ── Header ──────────────────────────────────────────────────
        let header_line = Line::from(vec![
            Span::raw("  "),
            Span::styled("Settings", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]);
        f.render_widget(Paragraph::new(header_line), Rect { x: chunks[0].x, y: chunks[0].y, width: chunks[0].width, height: 1 });
        let sep = "\u{2500}".repeat(chunks[0].width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: chunks[0].x, y: chunks[0].y + 1, width: chunks[0].width, height: 1 },
        );

        // ── Body ────────────────────────────────────────────────────
        let label_style = Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD);
        let val_normal = Style::default().fg(Color::White);
        let val_selected = Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD);
        let val_editing = Style::default().fg(Color::Green).add_modifier(Modifier::BOLD);
        let hint_style = Style::default().fg(Color::DarkGray);
        let val_disabled = Style::default().fg(Color::DarkGray);
        let section_style = Style::default().fg(ACCENT).add_modifier(Modifier::BOLD);
        let pad = 20; // label column width

        let sel = self.settings_selection;
        let mut lines: Vec<Line> = Vec::new();

        // Helper closure-like macro for building a setting line
        macro_rules! setting_line {
            ($label:expr, $value:expr, $idx:expr, $hint:expr, $style_override:expr) => {{
                let padded_label = format!("  {:<width$}", $label, width = pad);
                let is_sel = sel == $idx;
                let v_style = if let Some(s) = $style_override { s } else if is_sel { val_selected } else { val_normal };
                let mut spans = vec![
                    Span::styled(padded_label, label_style),
                    Span::styled($value.to_string(), v_style),
                ];
                if is_sel && !$hint.is_empty() {
                    spans.push(Span::styled(format!("  {}", $hint), hint_style));
                }
                lines.push(Line::from(spans));
            }};
        }

        // ── TRANSCRIPTION ───────────────────────────────────────────
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::raw("  "), Span::styled("TRANSCRIPTION", section_style)]));

        let mode_value = match &self.config.transcription {
            TranscriptionMode::Local { model_size } => format!("Local (Whisper {})", model_size),
            TranscriptionMode::Api { .. } => "OpenAI API".to_string(),
        };
        setting_line!("Mode", mode_value, 0, "\u{2190} Enter to change", None::<Style>);

        // Model size (only for local mode)
        if let TranscriptionMode::Local { model_size } = &self.config.transcription {
            setting_line!("Model Size", model_size, 1, "\u{2190} Enter to cycle", None::<Style>);
        }

        // API key (only for API mode)
        if let TranscriptionMode::Api { api_key } = &self.config.transcription {
            let api_key_display = if self.editing_api_key {
                format!("{}_", self.api_key_input)
            } else if api_key.is_empty() {
                "[Not Set]".to_string()
            } else {
                format!("{}******", &api_key[..api_key.len().min(4)])
            };
            let style_override = if sel == 1 && self.editing_api_key { Some(val_editing) } else { None };
            setting_line!("OpenAI API Key", api_key_display, 1, "\u{2190} Enter to edit", style_override);
        }

        // ── ENRICHMENT ──────────────────────────────────────────────
        lines.push(Line::from(""));
        let enrichment_label = if self.config.enrichment.is_local() { "ENRICHMENT (Privacy)" } else { "ENRICHMENT" };
        lines.push(Line::from(vec![Span::raw("  "), Span::styled(enrichment_label, section_style)]));

        setting_line!("Provider", self.config.enrichment.provider_display_name(), 2, "\u{2190} Enter to cycle", None::<Style>);

        // Model (index 3)
        if self.model_picker_state == ModelPickerState::Closed {
            setting_line!("Model", self.config.enrichment.model_name(), 3, "\u{2190} Enter to choose", None::<Style>);
        } else {
            // Picker is open
            lines.push(Line::from(vec![
                Span::styled(format!("  {:<width$}", "Model", width = pad), label_style),
                Span::styled("(select below)", hint_style),
            ]));

            let current_model = self.config.enrichment.model_name().to_string();
            for (i, item) in self.model_picker_items.iter().enumerate() {
                let is_cursor = i == self.model_picker_selection;
                let is_current = item.model_id.as_deref() == Some(current_model.as_str());
                let is_custom_sentinel = item.model_id.is_none() && item.display_name == "Custom...";

                let arrow = if is_cursor { "    \u{25B8} " } else { "      " };

                let style = if is_cursor {
                    if self.model_picker_state == ModelPickerState::EditingCustom && is_custom_sentinel {
                        val_editing
                    } else {
                        val_selected
                    }
                } else if is_current {
                    Style::default().fg(ACCENT)
                } else {
                    val_normal
                };

                if is_custom_sentinel && self.model_picker_state == ModelPickerState::EditingCustom {
                    lines.push(Line::from(vec![
                        Span::styled(arrow, style),
                        Span::styled(format!("Custom: {}_", self.model_picker_custom_input), style),
                    ]));
                } else {
                    let suffix = if is_current && !is_cursor { " (current)" } else { "" };
                    lines.push(Line::from(vec![
                        Span::styled(arrow, style),
                        Span::styled(format!("{}{}", item.display_name, suffix), style),
                    ]));
                }
            }
        }

        // API Key or Ollama Endpoint (index 4)
        if self.config.enrichment.is_local() {
            let endpoint_display = if self.editing_enrichment_endpoint {
                format!("{}_", self.enrichment_endpoint_input)
            } else {
                self.config.enrichment.ollama_endpoint().to_string()
            };
            let style_override = if sel == 4 && self.editing_enrichment_endpoint { Some(val_editing) } else { None };
            setting_line!("Ollama Server", endpoint_display, 4, "\u{2190} Enter to edit", style_override);
        } else {
            let key_label = match self.config.enrichment.cloud_provider() {
                Some(p) => format!("{} API Key", p.display_name()),
                None => "API Key".to_string(),
            };
            let env_hint = match self.config.enrichment.cloud_provider() {
                Some(p) => format!(" (or set {})", p.env_var_name()),
                None => String::new(),
            };
            let key_display = if self.editing_enrichment_api_key {
                format!("{}_", self.enrichment_api_key_input)
            } else {
                match self.config.enrichment.api_key() {
                    Some(key) if key.len() >= 4 => format!("{}******", &key[..4]),
                    Some(_) => "******".to_string(),
                    None => format!("[Not Set]{}", env_hint),
                }
            };
            let style_override = if sel == 4 && self.editing_enrichment_api_key { Some(val_editing) } else { None };
            setting_line!(key_label, key_display, 4, "\u{2190} Enter to edit", style_override);
        }

        // ── RECORDING ───────────────────────────────────────────────
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::raw("  "), Span::styled("RECORDING", section_style)]));

        let silence_enabled = self.config.silence_auto_stop.enabled;
        let silence_value = if silence_enabled { "Enabled" } else { "Disabled" };
        setting_line!("Auto-Stop", silence_value, 5, "\u{2190} Enter to toggle", None::<Style>);

        let timeout_secs = self.config.silence_auto_stop.timeout_seconds;
        let timeout_display = match timeout_secs {
            s if s < 60 => format!("{}s", s),
            s if s % 60 == 0 => format!("{}m", s / 60),
            s => format!("{}m {}s", s / 60, s % 60),
        };
        let timeout_style_override = if !silence_enabled { Some(val_disabled) } else { None };
        let timeout_hint = if silence_enabled { "\u{2190} Enter to cycle" } else { "(enable auto-stop first)" };
        setting_line!("Timeout", timeout_display, 6, timeout_hint, timeout_style_override);

        // ── DIARIZATION ─────────────────────────────────────────────
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::raw("  "), Span::styled("DIARIZATION", section_style)]));

        let diarization_enabled = self.config.diarization.enabled;
        let diar_value = if diarization_enabled { "Enabled" } else { "Disabled" };
        setting_line!("Speaker Diarization", diar_value, 7, "\u{2190} Enter to toggle", None::<Style>);

        let max_speakers = self.config.diarization.max_speakers;
        let speakers_style_override = if !diarization_enabled { Some(val_disabled) } else { None };
        let speakers_hint = if diarization_enabled { "\u{2190} Enter to cycle" } else { "(enable diarization first)" };
        setting_line!("Max Speakers", max_speakers, 8, speakers_hint, speakers_style_override);

        // ── VOICE MODE ──────────────────────────────────────────────
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::raw("  "), Span::styled("VOICE MODE", section_style)]));

        let voice_enabled = self.voice_mode_active;
        let voice_value = if voice_enabled { "Active" } else { "Off" };
        setting_line!("Voice Activation", voice_value, 9, "\u{2190} Enter to toggle", None::<Style>);

        let sensitivity_label = match self.config.voice.vad_threshold {
            t if t <= 0.005 => "Very High (0.005)",
            t if t <= 0.01 => "High (0.01)",
            t if t <= 0.02 => "Medium (0.02)",
            _ => "Low (0.05)",
        };
        let sens_style_override = if !voice_enabled { Some(val_disabled) } else { None };
        let sens_hint = if voice_enabled { "\u{2190} Enter to cycle" } else { "(enable voice mode first)" };
        setting_line!("Sensitivity", sensitivity_label, 10, sens_hint, sens_style_override);

        let body = Paragraph::new(lines).style(Style::default().fg(Color::White));
        f.render_widget(body, chunks[1]);

        // ── Footer ──────────────────────────────────────────────────
        let sep2 = "\u{2500}".repeat(chunks[2].width as usize);
        f.render_widget(
            Paragraph::new(sep2).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: chunks[2].x, y: chunks[2].y, width: chunks[2].width, height: 1 },
        );

        let footer_area = Rect { x: chunks[2].x, y: chunks[2].y + 1, width: chunks[2].width, height: 1 };
        let version = env!("CARGO_PKG_VERSION");
        let left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(format!("scriba \u{00B7} v{}", version), Style::default().fg(Color::DarkGray)),
        ];
        let right_spans = vec![
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("\u{2191}\u{2193}", Style::default().fg(Color::White)),
            Span::styled("] Navigate  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Enter", Style::default().fg(Color::White)),
            Span::styled("] Change  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Esc", Style::default().fg(Color::White)),
            Span::styled("] Back ", Style::default().fg(Color::DarkGray)),
        ];
        let left_w: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_w: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_area.width as usize).saturating_sub(left_w + right_w);
        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);
        f.render_widget(Paragraph::new(Line::from(spans)), footer_area);
    }

    fn render_entities_view(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        use ratatui::text::{Line, Span};

        // Full-screen borderless layout: header + table + footer
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // Header
                Constraint::Min(8),    // Entity table
                Constraint::Length(2), // Footer
            ])
            .split(area);

        // ── Header ──────────────────────────────────────────────────
        let people_count = self.entities.iter().filter(|e| e.entity_type == "person").count();
        let org_count = self.entities.iter().filter(|e| e.entity_type == "organization").count();
        let project_count = self.entities.iter().filter(|e| e.entity_type == "project").count();

        let mut count_parts: Vec<String> = Vec::new();
        if people_count > 0 { count_parts.push(format!("{} people", people_count)); }
        if org_count > 0 { count_parts.push(format!("{} orgs", org_count)); }
        if project_count > 0 { count_parts.push(format!("{} projects", project_count)); }
        let counts_suffix = if count_parts.is_empty() {
            String::new()
        } else {
            format!(" \u{00B7} {}", count_parts.join(" \u{00B7} "))
        };

        let header_line = Line::from(vec![
            Span::raw("  "),
            Span::styled("Scriba's World", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(counts_suffix, Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(Paragraph::new(header_line), Rect { x: main_chunks[0].x, y: main_chunks[0].y, width: main_chunks[0].width, height: 1 });

        let sep = "\u{2500}".repeat(main_chunks[0].width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: main_chunks[0].x, y: main_chunks[0].y + 1, width: main_chunks[0].width, height: 1 },
        );

        // ── Entity table ────────────────────────────────────────────
        let header_cells = ["ID", "Type", "Name", "Aliases", "Context", "Mentions"]
            .iter()
            .map(|h| {
                Cell::from(*h).style(
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )
            });

        let header_row = Row::new(header_cells)
            .style(Style::default())
            .height(1)
            .bottom_margin(1);

        let merge_source_id = self.merge_source_entity.as_ref().and_then(|e| e.id);
        let rows: Vec<Row> = self
            .entities
            .iter()
            .map(|entity| {
                let is_merge_source = self.entity_mode == EntityMode::MergeSelectTarget
                    && merge_source_id == entity.id;

                let aliases = entity.aliases_list().join(", ");
                let aliases_display = if aliases.is_empty() {
                    "-".to_string()
                } else if aliases.len() > 20 {
                    format!("{}...", &aliases[..17])
                } else {
                    aliases
                };

                let context_display = entity
                    .context
                    .as_ref()
                    .map(|c| {
                        if c.len() > 30 {
                            format!("{}...", &c[..27])
                        } else {
                            c.clone()
                        }
                    })
                    .unwrap_or_else(|| "-".to_string());

                let type_color = if is_merge_source {
                    Color::DarkGray
                } else {
                    match entity.entity_type.as_str() {
                        "person" => Color::Green,
                        "organization" => Color::Blue,
                        _ => Color::Gray,
                    }
                };

                let name_style = if is_merge_source {
                    Style::default().fg(Color::DarkGray)
                } else {
                    Style::default().add_modifier(Modifier::BOLD)
                };

                let dim = if is_merge_source { Style::default().fg(Color::DarkGray) } else { Style::default().fg(Color::Gray) };

                let cells = vec![
                    Cell::from(entity.id.unwrap_or(0).to_string()),
                    Cell::from(entity.entity_type.clone()).style(Style::default().fg(type_color)),
                    Cell::from(entity.canonical_name.clone()).style(name_style),
                    Cell::from(aliases_display).style(dim),
                    Cell::from(context_display).style(dim),
                    Cell::from(entity.mention_count.to_string())
                        .style(if is_merge_source { Style::default().fg(Color::DarkGray) } else { Style::default().fg(Color::Yellow) }),
                ];

                Row::new(cells).height(1).bottom_margin(0)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(4),   // ID
                Constraint::Length(12),  // Type
                Constraint::Length(20),  // Name
                Constraint::Length(22),  // Aliases
                Constraint::Min(20),     // Context
                Constraint::Length(8),   // Mentions
            ],
        )
        .header(header_row)
        .block(Block::default())
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("\u{25B6} ");

        f.render_stateful_widget(table, main_chunks[1], &mut self.entity_table_state);

        // ── Footer ──────────────────────────────────────────────────
        let sep2 = "\u{2500}".repeat(main_chunks[2].width as usize);
        f.render_widget(
            Paragraph::new(sep2).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: main_chunks[2].x, y: main_chunks[2].y, width: main_chunks[2].width, height: 1 },
        );

        let footer_area = Rect { x: main_chunks[2].x, y: main_chunks[2].y + 1, width: main_chunks[2].width, height: 1 };
        let version = env!("CARGO_PKG_VERSION");
        let entity_count = self.entities.len();
        let left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(
                format!("scriba \u{00B7} v{} \u{00B7} {} entities", version, entity_count),
                Style::default().fg(Color::DarkGray),
            ),
        ];

        let right_spans: Vec<Span> = match self.entity_mode {
            EntityMode::Browse => vec![
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("A", Style::default().fg(Color::White)),
                Span::styled("] Add  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("E", Style::default().fg(Color::White)),
                Span::styled("] Edit  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("D", Style::default().fg(Color::White)),
                Span::styled("] Del  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("M", Style::default().fg(Color::White)),
                Span::styled("] Merge  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("R", Style::default().fg(Color::White)),
                Span::styled("] Refresh  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Esc", Style::default().fg(Color::White)),
                Span::styled("] Back ", Style::default().fg(Color::DarkGray)),
            ],
            EntityMode::MergeSelectTarget => vec![
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Enter", Style::default().fg(Color::White)),
                Span::styled("] Confirm  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Esc", Style::default().fg(Color::White)),
                Span::styled("] Cancel ", Style::default().fg(Color::DarkGray)),
            ],
            EntityMode::DeleteConfirm => vec![
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Y", Style::default().fg(Color::White)),
                Span::styled("] Confirm  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("N", Style::default().fg(Color::White)),
                Span::styled("] Cancel ", Style::default().fg(Color::DarkGray)),
            ],
            EntityMode::MergeConfirm => vec![
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Y", Style::default().fg(Color::White)),
                Span::styled("] Confirm  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("N", Style::default().fg(Color::White)),
                Span::styled("] Cancel ", Style::default().fg(Color::DarkGray)),
            ],
            EntityMode::Adding | EntityMode::Editing => vec![
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Tab", Style::default().fg(Color::White)),
                Span::styled("] Next field  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Space", Style::default().fg(Color::White)),
                Span::styled("] Cycle type  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Esc", Style::default().fg(Color::White)),
                Span::styled("] Done ", Style::default().fg(Color::DarkGray)),
            ],
        };

        let left_w: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_w: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_area.width as usize).saturating_sub(left_w + right_w);
        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);
        f.render_widget(Paragraph::new(Line::from(spans)), footer_area);

        // Popups (overlays stay unchanged)
        if self.show_entity_detail {
            self.render_entity_detail_popup(f, area);
        }
        if self.entity_mode == EntityMode::Adding {
            self.render_entity_add_popup(f, area);
        }
        if self.entity_mode == EntityMode::Editing {
            self.render_entity_edit_popup(f, area);
        }
        if self.entity_mode == EntityMode::DeleteConfirm {
            self.render_entity_delete_confirm(f, area);
        }
        if self.entity_mode == EntityMode::MergeConfirm {
            self.render_entity_merge_confirm(f, area);
        }
    }

    fn render_entity_detail_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(10),
                Constraint::Percentage(80),
                Constraint::Percentage(10),
            ])
            .split(area)[1];

        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(10),
                Constraint::Percentage(80),
                Constraint::Percentage(10),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        if let Some(entity) = &self.selected_entity {
            let aliases = entity.aliases_list().join(", ");
            let aliases_display = if aliases.is_empty() {
                "-".to_string()
            } else {
                aliases
            };

            let context_display = entity
                .context
                .as_ref()
                .map(|c| c.clone())
                .unwrap_or_else(|| "(no context)".to_string());

            let type_label = match entity.entity_type.as_str() {
                "person" => "person",
                "organization" => "org",
                _ => "other",
            };

            let content = vec![
                Line::from(vec![
                    Span::styled(
                        format!("[{}] ", type_label),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        &entity.canonical_name,
                        Style::default()
                            .fg(ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Type: ", Style::default().fg(Color::Green)),
                    Span::styled(
                        &entity.entity_type,
                        Style::default().fg(Color::White),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("ID: ", Style::default().fg(Color::Green)),
                    Span::styled(
                        entity.id.unwrap_or(0).to_string(),
                        Style::default().fg(Color::White),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Mentions: ", Style::default().fg(Color::Green)),
                    Span::styled(
                        entity.mention_count.to_string(),
                        Style::default().fg(Color::Yellow),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Aliases: ", Style::default().fg(Color::Green)),
                    Span::styled(
                        aliases_display,
                        Style::default().fg(Color::Gray),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![Span::styled(
                    "Context:",
                    Style::default().fg(Color::Green),
                )]),
                Line::from(vec![Span::styled(
                    context_display,
                    Style::default().fg(Color::White),
                )]),
                Line::from(""),
                Line::from(""),
                Line::from(vec![Span::styled(
                    "Press ESC to close | E to edit | D to delete | M to merge",
                    Style::default().fg(Color::Blue),
                )]),
            ];

            let detail_paragraph = Paragraph::new(content)
                .style(Style::default().fg(Color::White))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!("Entity Details - {}", entity.canonical_name))
                        .title_style(
                            Style::default()
                                .fg(ACCENT)
                                .add_modifier(Modifier::BOLD),
                        )
                        .border_style(Style::default().fg(ACCENT)),
                )
                .wrap(Wrap { trim: true });

            f.render_widget(detail_paragraph, popup_area);
        }
    }

    fn render_entity_add_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(area)[1];
        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let cursor = "\u{2588}";
        let name_style = if self.entity_add_field == EntityEditField::Name {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        let context_style = if self.entity_add_field == EntityEditField::Context {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let name_display = if self.entity_add_field == EntityEditField::Name {
            format!("{}{}", self.entity_add_name, cursor)
        } else if self.entity_add_name.is_empty() {
            "(type a name)".to_string()
        } else {
            self.entity_add_name.clone()
        };

        let type_display: Vec<Span> = ENTITY_TYPES.iter().map(|t| {
            if *t == self.entity_add_type {
                Span::styled(format!(" [{}] ", t), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
            } else {
                Span::styled(format!("  {}  ", t), Style::default().fg(Color::Gray))
            }
        }).collect();

        let context_display = if self.entity_add_field == EntityEditField::Context {
            format!("{}{}", self.entity_add_context, cursor)
        } else if self.entity_add_context.is_empty() {
            "(optional)".to_string()
        } else {
            self.entity_add_context.clone()
        };

        let content = vec![
            Line::from(vec![
                Span::styled("Name: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![Span::styled(name_display, name_style)]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Type: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(type_display),
            Line::from(""),
            Line::from(vec![
                Span::styled("Context: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![Span::styled(context_display, context_style)]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Tab/↑↓: Switch Field | Space: Cycle Type | Enter: Save | Esc: Cancel", Style::default().fg(Color::DarkGray)),
            ]),
        ];

        let add_paragraph = Paragraph::new(content)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" ADD NEW ENTITY ")
                    .title_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                    .border_style(Style::default().fg(Color::Yellow)),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(add_paragraph, popup_area);
    }

    fn render_entity_edit_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(area)[1];
        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let name_style = if self.entity_edit_field == EntityEditField::Name {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        let context_style = if self.entity_edit_field == EntityEditField::Context {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let cursor = "█";
        let name_display = if self.entity_edit_field == EntityEditField::Name {
            format!("{}{}", self.entity_edit_name, cursor)
        } else {
            self.entity_edit_name.clone()
        };

        let type_display: Vec<Span> = ENTITY_TYPES.iter().map(|t| {
            if *t == self.entity_edit_type {
                Span::styled(format!(" [{}] ", t), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
            } else {
                Span::styled(format!("  {}  ", t), Style::default().fg(Color::Gray))
            }
        }).collect();

        let context_display = if self.entity_edit_field == EntityEditField::Context {
            format!("{}{}", self.entity_edit_context, cursor)
        } else {
            if self.entity_edit_context.is_empty() { "(empty)".to_string() } else { self.entity_edit_context.clone() }
        };

        let content = vec![
            Line::from(vec![
                Span::styled("Name: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![Span::styled(name_display, name_style)]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Type: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(type_display),
            Line::from(""),
            Line::from(vec![
                Span::styled("Context: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![Span::styled(context_display, context_style)]),
        ];

        let edit_paragraph = Paragraph::new(content)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Edit Entity")
                    .title_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                    .border_style(Style::default().fg(Color::Yellow)),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(edit_paragraph, popup_area);
    }

    fn render_entity_delete_confirm(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let entity_name = self.selected_entity.as_ref()
            .map(|e| e.canonical_name.as_str())
            .unwrap_or("?");

        let sel = self.confirm_selection;
        let sel_bg = Color::Indexed(236);
        let labels = ["Yes, delete", "No, keep it"];
        let label_width = labels.iter().map(|l| l.chars().count() + 2).max().unwrap_or(0);

        let mut lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled("Delete entity: ", Style::default().fg(Color::White)),
                Span::styled(entity_name, Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::styled("?", Style::default().fg(Color::White)),
            ]),
            Line::from(""),
        ];
        for (i, label) in labels.iter().enumerate() {
            if sel == i {
                let pad = " ".repeat(label_width.saturating_sub(label.chars().count() + 2));
                lines.push(Line::from(vec![
                    Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                    Span::styled(format!("{}{}", label, pad), Style::default().fg(Color::White).bg(sel_bg)),
                ]));
            } else {
                lines.push(Line::from(Span::styled(format!("  {}", label), Style::default().fg(Color::DarkGray))));
            }
        }

        let max_w = lines.iter().map(|l| l.width() as u16).max().unwrap_or(0).max(30);
        let content_width = max_w.min(area.width);
        let left_pad = (area.width.saturating_sub(content_width)) / 2;
        let line_count = lines.len();
        let top_pad = (area.height as usize).saturating_sub(line_count) / 2;
        let mut centered: Vec<Line> = Vec::with_capacity(top_pad + line_count);
        for _ in 0..top_pad { centered.push(Line::from("")); }
        centered.extend(lines);

        f.render_widget(Clear, area);
        let body = Rect { x: area.x + left_pad, width: content_width, ..area };
        f.render_widget(Paragraph::new(centered).alignment(Alignment::Left), body);
    }

    fn render_entity_merge_confirm(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let source_name = self.merge_source_entity.as_ref()
            .map(|e| e.canonical_name.as_str())
            .unwrap_or("?");
        let target_name = self.selected_entity.as_ref()
            .map(|e| e.canonical_name.as_str())
            .unwrap_or("?");

        let sel = self.confirm_selection;
        let sel_bg = Color::Indexed(236);
        let labels = ["Yes, merge", "No, cancel"];
        let label_width = labels.iter().map(|l| l.chars().count() + 2).max().unwrap_or(0);

        let mut lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled("Merge ", Style::default().fg(Color::White)),
                Span::styled(source_name, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::styled(" into ", Style::default().fg(Color::White)),
                Span::styled(target_name, Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
                Span::styled("?", Style::default().fg(Color::White)),
            ]),
            Line::from(Span::styled(
                format!("'{}' becomes an alias. Mentions transferred.", source_name),
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
        ];
        for (i, label) in labels.iter().enumerate() {
            if sel == i {
                let pad = " ".repeat(label_width.saturating_sub(label.chars().count() + 2));
                lines.push(Line::from(vec![
                    Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                    Span::styled(format!("{}{}", label, pad), Style::default().fg(Color::White).bg(sel_bg)),
                ]));
            } else {
                lines.push(Line::from(Span::styled(format!("  {}", label), Style::default().fg(Color::DarkGray))));
            }
        }

        let max_w = lines.iter().map(|l| l.width() as u16).max().unwrap_or(0).max(30);
        let content_width = max_w.min(area.width);
        let left_pad = (area.width.saturating_sub(content_width)) / 2;
        let line_count = lines.len();
        let top_pad = (area.height as usize).saturating_sub(line_count) / 2;
        let mut centered: Vec<Line> = Vec::with_capacity(top_pad + line_count);
        for _ in 0..top_pad { centered.push(Line::from("")); }
        centered.extend(lines);

        f.render_widget(Clear, area);
        let body = Rect { x: area.x + left_pad, width: content_width, ..area };
        f.render_widget(Paragraph::new(centered).alignment(Alignment::Left).wrap(Wrap { trim: false }), body);
    }

    fn render_recording_view(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // Header
                Constraint::Min(6),    // Body
                Constraint::Length(2),  // Footer
            ])
            .split(area);

        // ── Header ──────────────────────────────────────────────────
        let header_line = Line::from(vec![
            Span::raw("  "),
            Span::styled("Recording", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]);
        f.render_widget(Paragraph::new(header_line), Rect { x: chunks[0].x, y: chunks[0].y, width: chunks[0].width, height: 1 });
        let sep = "\u{2500}".repeat(chunks[0].width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: chunks[0].x, y: chunks[0].y + 1, width: chunks[0].width, height: 1 },
        );

        // ── Body (vertically centered) ─────────────────────────────
        let body = chunks[1];
        // 3 lines: spinner, blank, waveform
        let content_height: u16 = 3;
        let v_offset = body.height.saturating_sub(content_height) / 2;

        // Spinner + elapsed time
        let spinners = ["\u{280B}", "\u{2819}", "\u{2839}", "\u{2838}", "\u{283C}", "\u{2834}", "\u{2826}", "\u{2827}", "\u{2807}", "\u{280F}"];
        let spinner = spinners[self.progress_frame % spinners.len()];
        let elapsed = self.recording_start_instant
            .map(|t| t.elapsed())
            .unwrap_or_default();
        let mins = elapsed.as_secs() / 60;
        let secs = elapsed.as_secs() % 60;
        let elapsed_str = format!("{}m {:02}s", mins, secs);

        let spinner_line = Line::from(vec![
            Span::styled(format!("{}  Recording \u{00B7} {}", spinner, elapsed_str), Style::default().fg(Color::White)),
        ]);
        let spinner_y = body.y + v_offset;
        f.render_widget(
            Paragraph::new(spinner_line).alignment(Alignment::Center),
            Rect { x: body.x, y: spinner_y, width: body.width, height: 1 },
        );

        // Scrolling waveform using block-height characters
        let wave_chars = [' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}'];
        let wave_width: usize = 40;

        let mut spans: Vec<Span> = Vec::with_capacity(wave_width);
        let hist_len = self.volume_history.len();
        for i in 0..wave_width {
            let level = if i < wave_width.saturating_sub(hist_len) {
                0.0_f32
            } else {
                let idx = hist_len.saturating_sub(wave_width.saturating_sub(i));
                // Average with neighbors for a smoother look
                let raw = self.volume_history.get(idx).copied().unwrap_or(0.0);
                let prev = if idx > 0 { self.volume_history.get(idx - 1).copied().unwrap_or(raw) } else { raw };
                (raw * 0.7 + prev * 0.3).min(1.0)
            };
            let scaled = (level * 50.0).min(1.0);
            let char_idx = (scaled * (wave_chars.len() - 1) as f32).round() as usize;
            let ch = wave_chars[char_idx.min(wave_chars.len() - 1)];
            // Dim bars vs bright bars based on intensity
            let color = if char_idx <= 1 {
                Color::Indexed(237) // dark gray for silence
            } else {
                ACCENT
            };
            spans.push(Span::styled(String::from(ch), Style::default().fg(color)));
        }

        let wave_line = Line::from(spans);
        let wave_y = spinner_y + 2;
        if wave_y < body.y + body.height {
            f.render_widget(
                Paragraph::new(wave_line).alignment(Alignment::Center),
                Rect { x: body.x, y: wave_y, width: body.width, height: 1 },
            );
        }

        // ── Footer ─────────────────────────────────────────────────
        let sep2 = "\u{2500}".repeat(chunks[2].width as usize);
        f.render_widget(
            Paragraph::new(sep2).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: chunks[2].x, y: chunks[2].y, width: chunks[2].width, height: 1 },
        );
        let footer_area = Rect { x: chunks[2].x, y: chunks[2].y + 1, width: chunks[2].width, height: 1 };
        let version = env!("CARGO_PKG_VERSION");
        let left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(format!("scriba \u{00B7} v{}", version), Style::default().fg(Color::DarkGray)),
        ];
        let right_spans = vec![
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Esc", Style::default().fg(Color::White)),
            Span::styled("] Stop ", Style::default().fg(Color::DarkGray)),
        ];
        let left_w: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_w: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_area.width as usize).saturating_sub(left_w + right_w);
        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);
        f.render_widget(Paragraph::new(Line::from(spans)), footer_area);
    }

    fn render_message_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(30),
                Constraint::Length(5),
                Constraint::Percentage(65),
            ])
            .split(area)[1];

        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(20),
                Constraint::Percentage(60),
                Constraint::Percentage(20),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let para = Paragraph::new(self.message.clone())
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Red))
                    .title("Message (press Esc)"),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(para, popup_area);
    }

    fn render_transcript_popup(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        // Dynamic chat panel height
        let has_chat_content = !self.chat.messages.is_empty()
            || !self.chat.pending_blocks.is_empty()
            || self.chat.is_generating;

        let chat_h: u16 = if has_chat_content {
            (area.height * 45 / 100).max(8)
        } else {
            (self.chat.suggestions.len() as u16 + 4).max(5).min(8)
        };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),      // header
                Constraint::Min(5),         // content (enrichment + transcript preview)
                Constraint::Length(2),      // blank + separator
                Constraint::Length(chat_h), // chat panel
                Constraint::Length(2),      // footer
            ])
            .split(area);

        // ── Header ──────────────────────────────────────────────────────
        self.render_transcript_header(f, chunks[0]);

        // ── Content (enrichment + transcript preview) ───────────────────
        self.render_transcript_body(f, chunks[1]);

        // ── Separator (blank line + ─) ───────────────────────────────────
        let sep = "\u{2500}".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: chunks[2].x, y: chunks[2].y + 1, width: chunks[2].width, height: 1 },
        );

        // ── Chat panel ──────────────────────────────────────────────────
        self.chat.render(f, chunks[3]);

        // ── Footer ──────────────────────────────────────────────────────
        self.render_transcript_footer(f, chunks[4]);

        // Notification overlay (e.g. "Copied to clipboard")
        if let Some((ref msg, _)) = self.notification_message {
            let is_error = msg.contains("failed") || msg.contains("Failed");
            let style = if is_error {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            };
            let notif_area = Rect {
                x: area.x,
                y: area.y + area.height.saturating_sub(1),
                width: area.width,
                height: 1,
            };
            let para = Paragraph::new(msg.as_str())
                .style(style)
                .alignment(Alignment::Center);
            f.render_widget(para, notif_area);
        }
    }

    fn render_transcript_header(&self, f: &mut Frame, area: Rect) {
        // Row 0: "  Name · Xm · Mar 8, 12:30"
        let recording = self.current_transcript_recording.as_ref();
        let name = recording
            .and_then(|r| r.display_name.as_deref())
            .or(recording.map(|r| r.directory_name.as_str()))
            .unwrap_or("Recording");
        let dur_secs = recording.and_then(|r| r.duration_seconds).unwrap_or(0);
        let dur_str = if dur_secs >= 3600 {
            format!("{}h {}m", dur_secs / 3600, (dur_secs % 3600) / 60)
        } else {
            format!("{}m", dur_secs / 60)
        };
        let date_str = recording
            .map(|r| r.created_at.format("%b %-d, %H:%M").to_string())
            .unwrap_or_default();

        let header_line = Line::from(vec![
            Span::raw("  "),
            Span::styled(name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" \u{00B7} {} \u{00B7} {}", dur_str, date_str), Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(Paragraph::new(header_line), Rect { x: area.x, y: area.y, width: area.width, height: 1 });

        // Row 1: separator
        let sep = "\u{2500}".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: area.x, y: area.y + 1, width: area.width, height: 1 },
        );
    }

    /// Build combined enrichment + transcript lines and render as a single scrollable Paragraph.
    fn render_transcript_body(&self, f: &mut Frame, area: Rect) {
        let content_width = area.width.saturating_sub(4) as usize; // 2 padding each side
        let mut lines: Vec<Line> = Vec::new();

        // ── Enrichment metadata (inline) ────────────────────────────────
        let has_enrichment = self.transcript_summary.is_some()
            || self.transcript_topics.is_some()
            || self.transcript_entities.is_some();

        if has_enrichment {
            lines.push(Line::from(""));

            // Summary
            if let Some(summary) = &self.transcript_summary {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled("Summary", Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
                ]));
                // Wrap summary text
                let wrapped = textwrap::wrap(summary, content_width);
                for wl in &wrapped {
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(wl.to_string(), Style::default().fg(Color::Indexed(249))),
                    ]));
                }
                lines.push(Line::from(""));
            }

            // Topics
            if let Some(topics) = &self.transcript_topics {
                if !topics.is_empty() {
                    let joined = topics.join(" \u{00B7} ");
                    Self::push_labeled_wrap(&mut lines, "Topics  ", &joined, content_width);
                }
            }

            // People / Entities
            if let Some(entities) = &self.transcript_entities {
                if !entities.is_empty() {
                    let people: Vec<&str> = entities.iter()
                        .filter(|(_, t)| t == "person")
                        .map(|(n, _)| n.as_str())
                        .collect();
                    let orgs: Vec<&str> = entities.iter()
                        .filter(|(_, t)| t == "organization")
                        .map(|(n, _)| n.as_str())
                        .collect();
                    let others: Vec<&str> = entities.iter()
                        .filter(|(_, t)| t != "person" && t != "organization")
                        .map(|(n, _)| n.as_str())
                        .collect();

                    if !people.is_empty() {
                        Self::push_labeled_wrap(&mut lines, "People  ", &people.join(" \u{00B7} "), content_width);
                    }
                    if !orgs.is_empty() {
                        Self::push_labeled_wrap(&mut lines, "Orgs    ", &orgs.join(" \u{00B7} "), content_width);
                    }
                    if !others.is_empty() {
                        Self::push_labeled_wrap(&mut lines, "Entities ", &others.join(" \u{00B7} "), content_width);
                    }
                }
            }

            // Key takeaway (first point only)
            if let Some(key_points) = &self.transcript_key_points {
                if !key_points.is_empty() {
                    lines.push(Line::from(""));
                    let wrapped = textwrap::wrap(&key_points[0], content_width);
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled("Key takeaway", Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
                    ]));
                    for wl in &wrapped {
                        lines.push(Line::from(vec![
                            Span::raw("  "),
                            Span::styled(wl.to_string(), Style::default().fg(Color::Indexed(249))),
                        ]));
                    }
                }
            }

            // Separator between enrichment and transcript
            lines.push(Line::from(""));
            let sep = "\u{2500}".repeat(area.width as usize);
            lines.push(Line::from(Span::styled(sep, Style::default().fg(Color::Indexed(237)))));
        }

        // ── Transcript preview (compact, max ~15 lines, faded tail) ────
        lines.push(Line::from(""));
        let max_preview = 15usize;
        let mut preview_lines: Vec<Line> = Vec::new();
        let transcript_lines_total = self.transcript_content.lines().count();
        let mut source_lines_used = 0usize;
        for text_line in self.transcript_content.lines() {
            if preview_lines.len() >= max_preview {
                break;
            }
            source_lines_used += 1;
            if text_line.is_empty() {
                preview_lines.push(Line::from(""));
            } else {
                let wrapped = textwrap::wrap(text_line, content_width);
                for wl in &wrapped {
                    if preview_lines.len() >= max_preview {
                        break;
                    }
                    preview_lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(wl.to_string(), Style::default().fg(Color::White)),
                    ]));
                }
            }
        }

        // Truncated if we didn't consume all source lines, OR if wrapping
        // filled the preview before we finished even the lines we did visit.
        let remaining_source = transcript_lines_total.saturating_sub(source_lines_used);
        let is_truncated = remaining_source > 0 || preview_lines.len() >= max_preview;

        // Fade out the last few lines + hint when transcript is truncated
        if is_truncated {
            let fade_count = 3usize;
            let fade_colors: [Color; 3] = [
                Color::Indexed(249),
                Color::Indexed(245),
                Color::Indexed(240),
            ];
            let total_preview = preview_lines.len();
            if total_preview > fade_count {
                for (i, color) in fade_colors.iter().enumerate() {
                    let idx = total_preview - fade_count + i;
                    if idx < total_preview {
                        let text: String = preview_lines[idx].spans.iter().map(|s| s.content.to_string()).collect();
                        preview_lines[idx] = Line::from(Span::styled(text, Style::default().fg(*color)));
                    }
                }
            }
        }

        let shown_count = preview_lines.len();
        lines.extend(preview_lines);

        // Hint line for truncated transcripts
        if is_truncated {
            lines.push(Line::from(""));
            // Count total wrapped display lines for accurate "X more lines"
            let total_display_lines: usize = self.transcript_content.lines().map(|l| {
                if l.is_empty() { 1 } else { textwrap::wrap(l, content_width).len() }
            }).sum();
            let more = total_display_lines.saturating_sub(shown_count);

            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{} more lines not shown \u{00B7} ", more),
                    Style::default().fg(Color::Indexed(240)),
                ),
                Span::styled("[", Style::default().fg(Color::Indexed(240))),
                Span::styled("Ctrl+Y", Style::default().fg(Color::Indexed(249))),
                Span::styled("] copy full transcript", Style::default().fg(Color::Indexed(240))),
            ]));
        }

        let para = Paragraph::new(lines);
        f.render_widget(para, area);
    }

    fn render_transcript_footer(&self, f: &mut Frame, area: Rect) {
        // Separator
        let sep = "\u{2500}".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: area.x, y: area.y, width: area.width, height: 1 },
        );

        let footer_area = if area.height > 1 {
            Rect { x: area.x, y: area.y + 1, width: area.width, height: area.height - 1 }
        } else {
            area
        };

        // Left: ▸ scriba · vX.Y.Z
        let version = env!("CARGO_PKG_VERSION");
        let left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(format!("scriba \u{00B7} v{}", version), Style::default().fg(Color::DarkGray)),
        ];

        // Right: [Ctrl+Y] Copy  [Ctrl+T] Re-transcribe  [Esc] Back
        let right_spans = vec![
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Ctrl+Y", Style::default().fg(Color::White)),
            Span::styled("] Copy  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Ctrl+T", Style::default().fg(Color::White)),
            Span::styled("] Re-transcribe  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Esc", Style::default().fg(Color::White)),
            Span::styled("] Back ", Style::default().fg(Color::DarkGray)),
        ];

        let left_width: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_width: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_area.width as usize).saturating_sub(left_width + right_width);

        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);

        f.render_widget(Paragraph::new(Line::from(spans)), footer_area);
    }

    /// Push a "Label  value text..." line that wraps, with continuation lines indented to align.
    fn push_labeled_wrap(lines: &mut Vec<Line<'_>>, label: &str, value: &str, max_width: usize) {
        let prefix_len = 2 + label.len(); // "  " + label
        let value_width = max_width.saturating_sub(prefix_len);
        if value_width == 0 {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(label.to_string(), Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
                Span::styled(value.to_string(), Style::default().fg(Color::Indexed(249))),
            ]));
            return;
        }
        let wrapped = textwrap::wrap(value, value_width);
        for (i, wl) in wrapped.iter().enumerate() {
            if i == 0 {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(label.to_string(), Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
                    Span::styled(wl.to_string(), Style::default().fg(Color::Indexed(249))),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::raw(" ".repeat(prefix_len)),
                    Span::styled(wl.to_string(), Style::default().fg(Color::Indexed(249))),
                ]));
            }
        }
    }

    fn render_delete_confirmation_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let recording_name = if let Some(recording) = &self.delete_candidate {
            recording
                .display_name
                .as_ref()
                .unwrap_or(&recording.directory_name)
                .clone()
        } else {
            "?".to_string()
        };

        let sel = self.delete_confirm_selection;
        let sel_bg = Color::Indexed(236);
        let labels = ["Yes, delete", "No, keep it"];
        let label_width = labels.iter().map(|l| l.chars().count() + 2).max().unwrap_or(0);

        let mut lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled("Delete recording: ", Style::default().fg(Color::White)),
                Span::styled(&recording_name, Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::styled("?", Style::default().fg(Color::White)),
            ]),
            Line::from(""),
        ];
        for (i, label) in labels.iter().enumerate() {
            if sel == i {
                let pad = " ".repeat(label_width.saturating_sub(label.chars().count() + 2));
                lines.push(Line::from(vec![
                    Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                    Span::styled(format!("{}{}", label, pad), Style::default().fg(Color::White).bg(sel_bg)),
                ]));
            } else {
                lines.push(Line::from(Span::styled(format!("  {}", label), Style::default().fg(Color::DarkGray))));
            }
        }

        let max_w = lines.iter().map(|l| l.width() as u16).max().unwrap_or(0).max(30);
        let content_width = max_w.min(area.width);
        let left_pad = (area.width.saturating_sub(content_width)) / 2;
        let line_count = lines.len();
        let top_pad = (area.height as usize).saturating_sub(line_count) / 2;
        let mut centered: Vec<Line> = Vec::with_capacity(top_pad + line_count);
        for _ in 0..top_pad { centered.push(Line::from("")); }
        centered.extend(lines);

        f.render_widget(Clear, area);
        let body = Rect { x: area.x + left_pad, width: content_width, ..area };
        f.render_widget(Paragraph::new(centered).alignment(Alignment::Left), body);
    }

    fn render_search_input(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(40),
                Constraint::Length(3),
                Constraint::Percentage(40),
            ])
            .split(area)[1];

        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(20),
                Constraint::Percentage(60),
                Constraint::Percentage(20),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let search_input = Paragraph::new(format!("Search: {}", self.search_query))
            .style(Style::default().fg(Color::Yellow))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(ACCENT))
                    .title("Search Recordings"),
            );

        f.render_widget(search_input, popup_area);
    }

    async fn execute_record_and_transcribe(&mut self) -> Result<()> {
        // Check if already recording (transcription can run concurrently)
        if self.recording_task.is_some() {
            self.message = "Recording already in progress".to_string();
            self.show_message = true;
            return Ok(());
        }

        // Show immediate progress animation
        self.progress_animation = Some("Recording... (Press Esc to stop)".to_string());
        self.progress_frame = 0;
        self.show_message = true;
        self.recording_mode = Some(RecordingMode::RecordAndTranscribe);
        self.recording_start_instant = Some(std::time::Instant::now());

        // Generate filename and start recording task (no name prompt, consistent with A command)
        let recording_name = generate_recording_name(None);
        self.start_recording_task(recording_name).await?;

        Ok(())
    }

    async fn execute_add_external_file(&mut self) -> Result<()> {
        // Check if already recording (transcription can run concurrently)
        if self.recording_task.is_some() {
            self.message = "Recording already in progress".to_string();
            self.show_message = true;
            return Ok(());
        }

        // Show file dialog for importing audio file
        self.show_file_dialog = true;
        self.file_dialog_stage = FileDialogStage::FilePath;
        self.file_path_input.clear();
        self.file_name_input.clear();

        Ok(())
    }

    async fn start_file_import(&mut self, file_path: String, display_name: String) -> Result<()> {
        let source_path = PathBuf::from(file_path.trim());
        let transcription_mode = self.config.transcription.clone();

        self.enqueue_transcription(PendingTranscription::Import {
            source_path,
            display_name,
            transcription_mode,
        });

        Ok(())
    }

    // Removed unused execute_transcribe_file; dashboard uses TranscribeSelected (T) instead

    async fn execute_transcribe_selected(&mut self) -> Result<()> {
        // Check if transcription is already running
        if self.has_active_transcription() {
            self.message = "Transcription already in progress. Please wait...".to_string();
            self.show_message = true;
            return Ok(());
        }

        // Get the selected recording
        let selected_index = match self.table_state.selected() {
            Some(i) => i,
            None => {
                self.message = "No recording selected".to_string();
                self.show_message = true;
                return Ok(());
            }
        };

        let selected_recording = match self.recordings.get(selected_index) {
            Some(recording) => recording.clone(),
            None => {
                self.message = "Invalid recording selection".to_string();
                self.show_message = true;
                return Ok(());
            }
        };

        // Check if transcript already exists
        let has_transcript = if let Some(id) = selected_recording.id {
            self.db
                .get_transcript_by_recording_id(id)
                .is_ok_and(|t| t.is_some())
        } else {
            false
        };

        if has_transcript {
            // Check if this is the second press on the same recording
            if self.last_transcribe_warning == Some(selected_index) {
                // User confirmed overwrite - proceed with transcription
                self.last_transcribe_warning = None;
            } else {
                // First press - show warning and remember this recording
                self.last_transcribe_warning = Some(selected_index);
                self.message =
                    "Recording already has transcript. Press T again to overwrite.".to_string();
                self.show_message = true;
                return Ok(());
            }
        } else {
            // Clear any previous warning state
            self.last_transcribe_warning = None;
        }

        // Enqueue transcription
        let directory_name = selected_recording.directory_name.clone();
        let transcription_mode = self.config.transcription.clone();

        self.enqueue_transcription(PendingTranscription::Retranscribe {
            recording_name: directory_name,
            transcription_mode,
        });

        Ok(())
    }

    async fn create_stereo_temp_file(
        &self,
        mono_file_path: &std::path::Path,
    ) -> Result<std::path::PathBuf> {
        use std::fs;

        // Create a temporary file path for the stereo version
        let temp_dir = std::env::temp_dir();
        let temp_filename = format!(
            "scriba_stereo_{}.wav",
            mono_file_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
        );
        let temp_path = temp_dir.join(temp_filename);

        // Use Rust's hound crate to convert mono to stereo
        let mono_reader =
            hound::WavReader::open(mono_file_path).context("Failed to open mono audio file")?;

        let spec = mono_reader.spec();

        // Create stereo spec (2 channels)
        let stereo_spec = hound::WavSpec {
            channels: 2,
            sample_rate: spec.sample_rate,
            bits_per_sample: spec.bits_per_sample,
            sample_format: spec.sample_format,
        };

        let mut stereo_writer = hound::WavWriter::create(&temp_path, stereo_spec)
            .context("Failed to create stereo audio file")?;

        // Convert samples based on format
        match spec.sample_format {
            hound::SampleFormat::Float => {
                // 32-bit float samples
                for sample in mono_reader.into_samples::<f32>() {
                    match sample {
                        Ok(s) => {
                            // Write the same sample to both left and right channels
                            stereo_writer.write_sample(s)?; // Left
                            stereo_writer.write_sample(s)?; // Right
                        }
                        Err(e) => {
                            return Err(anyhow::anyhow!("Error processing audio sample: {}", e))
                        }
                    }
                }
            }
            hound::SampleFormat::Int => {
                // Integer samples (16-bit or 24-bit)
                if spec.bits_per_sample == 16 {
                    for sample in mono_reader.into_samples::<i16>() {
                        match sample {
                            Ok(s) => {
                                stereo_writer.write_sample(s)?; // Left
                                stereo_writer.write_sample(s)?; // Right
                            }
                            Err(e) => {
                                return Err(anyhow::anyhow!(
                                    "Error processing audio sample: {}",
                                    e
                                ));
                            }
                        }
                    }
                } else if spec.bits_per_sample == 24 {
                    for sample in mono_reader.into_samples::<i32>() {
                        match sample {
                            Ok(s) => {
                                stereo_writer.write_sample(s)?; // Left
                                stereo_writer.write_sample(s)?; // Right
                            }
                            Err(e) => {
                                return Err(anyhow::anyhow!(
                                    "Error processing audio sample: {}",
                                    e
                                ));
                            }
                        }
                    }
                } else {
                    return Err(anyhow::anyhow!(
                        "Unsupported bit depth: {}",
                        spec.bits_per_sample
                    ));
                }
            }
        }

        // Finalize the stereo file
        stereo_writer
            .finalize()
            .context("Failed to finalize stereo audio file")?;

        // Schedule cleanup of temp file after a delay
        let temp_path_clone = temp_path.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
            let _ = fs::remove_file(&temp_path_clone);
        });

        Ok(temp_path)
    }

    fn emergency_stop_all_audio_players(&self) -> Result<()> {
        // Kill all common audio players as a fallback when PID is not available
        #[cfg(unix)]
        {
            use std::process::Command;
            // Try to kill common audio players
            let players = ["mpv", "ffplay", "afplay"];
            for player in &players {
                let _ = Command::new("killall").arg(player).output();
            }
        }

        #[cfg(windows)]
        {
            use std::process::Command;
            // Try to kill common audio players on Windows
            let players = ["mpv.exe", "ffplay.exe"];
            for player in &players {
                let _ = Command::new("taskkill")
                    .arg("/IM")
                    .arg(player)
                    .arg("/F")
                    .output();
            }
        }
        Ok(())
    }

    fn stop_audio_playback(&self, pid: u32) -> Result<()> {
        #[cfg(unix)]
        {
            use std::process::Command;
            // Use SIGTERM first for graceful shutdown, then SIGKILL if needed
            let _ = Command::new("kill")
                .arg("-TERM")
                .arg(pid.to_string())
                .output();

            // Give a very brief moment for graceful shutdown
            std::thread::sleep(std::time::Duration::from_millis(10));

            // Use SIGKILL immediately for faster termination (audio players can be stubborn)
            let kill_result = Command::new("kill")
                .arg("-KILL")
                .arg(pid.to_string())
                .output();

            // Also try pkill in case the process spawned children
            let _ = Command::new("pkill")
                .arg("-P")
                .arg(pid.to_string())
                .output();

            match kill_result {
                Ok(_) => Ok(()),
                Err(_e) => {
                    // If direct kill fails, try killall on common audio players
                    let _ = Command::new("killall")
                        .arg("mpv")
                        .arg("ffplay")
                        .arg("afplay")
                        .output();
                    Ok(())
                }
            }
        }

        #[cfg(windows)]
        {
            use std::process::Command;
            let _ = Command::new("taskkill")
                .arg("/PID")
                .arg(pid.to_string())
                .arg("/F")
                .output();
            Ok(())
        }
    }

    // start_progress_animation not used; progress is updated directly via fields

    fn stop_progress_animation(&mut self) {
        self.progress_animation = None;
    }

    fn update_progress_message(&mut self) {
        if let Some(base_msg) = &self.progress_animation {
            let spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let spinner = spinners[self.progress_frame % spinners.len()];

            // If recording is active, show volume level instead of progress bar
            if self.recording_task.is_some() {
                let volume_bar = self.create_volume_bar(self.current_volume_level);
                self.message = format!("{} {} [{}]", spinner, base_msg, volume_bar);
            } else {
                // Regular progress bar for transcription
                let bar_width = 20;
                let progress_pos = (self.progress_frame / 2) % (bar_width * 2);
                let mut bar = vec!["▱"; bar_width];

                if progress_pos < bar_width {
                    for i in 0..=progress_pos.min(bar_width - 1) {
                        bar[i] = "▰";
                    }
                } else {
                    let reverse_pos = (bar_width * 2 - 1) - progress_pos;
                    for i in reverse_pos..bar_width {
                        bar[i] = "▰";
                    }
                }

                let bar_str = bar.join("");
                self.message = format!("{} {} [{}]", spinner, base_msg, bar_str);
            }

            self.progress_frame += 1;
        }
    }

    fn create_volume_bar(&self, level: f32) -> String {
        let bar_width = 20;
        // Scale the level (0.0 to 1.0) to bar width and apply some amplification for visibility
        let scaled_level = (level * 50.0).min(1.0); // Amplify for visibility
        let filled_chars = (scaled_level * bar_width as f32) as usize;

        let mut bar = vec!["▱"; bar_width];
        for i in 0..filled_chars.min(bar_width) {
            bar[i] = "▰";
        }

        format!("{}|{}%", bar.join(""), (scaled_level * 100.0) as u8)
    }

    async fn start_recording_task(&mut self, recording_name: String) -> Result<()> {
        // Create channels for recording control
        let (stop_tx, stop_rx) = mpsc::channel(1);
        let (level_tx, level_rx) = mpsc::channel(100);

        // Store the channels for control and feedback
        self.recording_stop_tx = Some(stop_tx);
        self.recording_level_rx = Some(level_rx);

        // Use speech-optimized compression settings
        let compression_settings = CompressionSettings::speech_optimized();

        // Determine if auto-transcription is enabled based on recording mode
        let _auto_transcribe = matches!(
            self.recording_mode,
            Some(RecordingMode::RecordAndTranscribe)
        );

        // Silence auto-stop timeout from config
        let silence_timeout = if self.config.silence_auto_stop.enabled {
            Some(Duration::from_secs(self.config.silence_auto_stop.timeout_seconds as u64))
        } else {
            None
        };

        // Use unified recording function with TUI control channels
        let output_path = PathBuf::from(&recording_name);

        self.recording_task = Some(tokio::spawn(async move {
            record_audio(
                output_path,
                RecordOptions {
                    compression_settings: Some(compression_settings),
                    stop_rx: Some(stop_rx),
                    level_tx: Some(level_tx),
                    verbose: false,
                    silence_timeout,
                },
            )
            .await
        }));

        Ok(())
    }

    // ── Voice Mode ("Scriba Forever") ────────────────────────────────────

    async fn toggle_voice_mode(&mut self) {
        if self.voice_mode_active {
            // Shut down voice detector
            if let Some(handle) = self.voice_detector_handle.take() {
                // If recording, stop it first
                if handle.listening_state() == VoiceListeningState::Recording {
                    let _ = handle.stop_recording();
                }
                handle.shutdown();
            }
            self.voice_command_rx = None;
            self.voice_mode_active = false;
            self.notification_message = Some(("Voice mode disabled".to_string(), 30));
        } else {
            // Start voice detector
            let (tx, rx) = mpsc::channel(8);
            match start_voice_detector(&self.config.voice, tx).await {
                Ok(handle) => {
                    self.voice_detector_handle = Some(handle);
                    self.voice_command_rx = Some(rx);
                    self.voice_mode_active = true;
                    self.notification_message = Some((
                        "Voice mode active -- say \"Scriba record\" to start".to_string(),
                        60,
                    ));
                }
                Err(e) => {
                    self.notification_message = Some((
                        format!("Failed to start voice mode: {}", e),
                        60,
                    ));
                }
            }
        }
    }

    async fn handle_voice_record_command(&mut self) {
        // Don't start if already recording
        if self.recording_task.is_some() {
            return;
        }

        let recording_name = generate_recording_name(None);

        if let Some(ref handle) = self.voice_detector_handle {
            if let Err(e) = handle.start_recording(&recording_name) {
                self.notification_message = Some((
                    format!("Voice record failed: {}", e),
                    60,
                ));
                return;
            }
        }

        self.notification_message = Some((
            "Voice: Recording started! Say \"Scriba stop\" to finish.".to_string(),
            60,
        ));
    }

    async fn handle_voice_stop_command(&mut self) {
        let result = if let Some(ref handle) = self.voice_detector_handle {
            handle.stop_recording()
        } else {
            return;
        };

        match result {
            Ok(Some((dir_name, wav_path))) => {
                self.notification_message = Some((
                    "Voice: Recording stopped. Transcribing...".to_string(),
                    60,
                ));

                // Save to database
                let dir_name_clone = dir_name.clone();
                if let Ok(mut db) = Database::new() {
                    let meta = crate::core::FileManager::extract_audio_metadata(&wav_path);
                    if let Ok(meta) = meta {
                        let recording = Recording {
                            id: None,
                            directory_name: dir_name.clone(),
                            display_name: None,
                            created_at: chrono::Utc::now(),
                            updated_at: chrono::Utc::now(),
                            duration_seconds: meta.duration_seconds,
                            file_size_bytes: meta.file_size_bytes,
                            audio_format: meta.audio_format,
                            sample_rate: meta.sample_rate,
                            channels: meta.channels,
                            has_transcript: false,
                            transcript_status: "pending".to_string(),
                            language_code: "auto".to_string(),
                            model_used: "whisper.cpp".to_string(),
                            tags: None,
                            summary: None,
                            key_points: None,
                            action_items: None,
                            speakers: None,
                            sentiment_score: None,
                            search_index: None,
                            categories: None,
                            confidence_score: None,
                            audio_path: wav_path
                                .file_name()
                                .unwrap()
                                .to_string_lossy()
                                .to_string(),
                            transcript_path: None,
                        };
                        let _ = db.insert_recording(&recording);
                    }
                }

                // Enqueue transcription pipeline
                let transcription_mode = self.config.transcription.clone();
                let _ = self.load_recordings();
                let _ = self.load_stats();

                self.enqueue_transcription(PendingTranscription::Retranscribe {
                    recording_name: dir_name_clone,
                    transcription_mode,
                });
            }
            Ok(None) => {
                self.notification_message = Some(("Voice: No recording to stop.".to_string(), 30));
            }
            Err(e) => {
                self.notification_message = Some((
                    format!("Voice stop failed: {}", e),
                    60,
                ));
            }
        }
    }

    fn render_file_dialog_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Length(12),
                Constraint::Percentage(65),
            ])
            .split(area)[1];

        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let (title, prompt, current_input, hint) = match self.file_dialog_stage {
            FileDialogStage::FilePath => (
                "Import Audio File - Step 1/2",
                "Enter the full path to the audio file:",
                &self.file_path_input,
                "Example: /path/to/your/audio.mp3 or ~/Downloads/recording.wav",
            ),
            FileDialogStage::FileName => (
                "Import Audio File - Step 2/2",
                "Enter a display name for this recording:",
                &self.file_name_input,
                "This name will be shown in your recordings list",
            ),
        };

        let content = vec![
            Line::from(vec![Span::styled(
                prompt,
                Style::default().fg(Color::White),
            )]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Input: ", Style::default().fg(Color::Green)),
                Span::styled(
                    format!("{}_", current_input),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(hint, Style::default().fg(Color::Gray))]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Press ENTER to continue, ESC to cancel",
                Style::default().fg(Color::Blue),
            )]),
        ];

        let dialog_paragraph = Paragraph::new(content)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title)
                    .title_style(
                        Style::default()
                            .fg(ACCENT)
                            .add_modifier(Modifier::BOLD),
                    )
                    .border_style(Style::default().fg(ACCENT)),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(dialog_paragraph, popup_area);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Ask Scriba: Chat Implementation
    // ─────────────────────────────────────────────────────────────────────────

    fn generate_suggestions(&self) -> Vec<String> {
        match &self.chat.context {
            ChatContext::Global => {
                let mut suggestions = vec!["What have I been talking about recently?".to_string()];

                // Top entity suggestion
                if !self.entities.is_empty() {
                    if let Some(top) = self.entities.first() {
                        suggestions.push(format!("Summarize my conversations about {}", top.canonical_name));
                    }
                }

                // Action items
                let has_action_items = self.recordings.iter().any(|r| {
                    r.action_items.as_ref().map_or(false, |a| !a.is_empty() && a != "[]")
                });
                if has_action_items {
                    suggestions.push("What action items do I have pending?".to_string());
                }

                // People
                let has_people = self.entities.iter().any(|e| e.entity_type == "person");
                if has_people {
                    suggestions.push("Who have I been meeting with most?".to_string());
                }

                suggestions.truncate(4);
                suggestions
            }
            ChatContext::Recording { .. } => {
                let mut suggestions = vec!["Summarize the key takeaways".to_string()];

                // Draft email if action items exist
                if let Some(recording) = &self.current_transcript_recording {
                    let has_actions = recording.action_items.as_ref()
                        .map_or(false, |a| !a.is_empty() && a != "[]");
                    let has_key_points = recording.key_points.as_ref()
                        .map_or(false, |k| !k.is_empty() && k != "[]");

                    if has_actions || has_key_points {
                        suggestions.push("Draft a follow-up email".to_string());
                    }

                    if has_actions {
                        suggestions.push("What were the action items?".to_string());
                    }
                }

                // Entity cross-reference
                if let Some(entities) = &self.transcript_entities {
                    if let Some((name, _)) = entities.first() {
                        suggestions.push(format!("What other recordings mention {}?", name));
                    }
                }

                suggestions.truncate(4);
                suggestions
            }
        }
    }

    fn generate_greeting(&mut self) {
        use crate::tui::chat::HomeRecording;

        // Owner first name
        let first_name = self.owner_name.split_whitespace().next().unwrap_or("").to_string();

        // Count today's recordings and total duration
        let today = chrono::Local::now().date_naive();
        let today_recordings: Vec<&Recording> = self.recordings.iter().filter(|r| {
            r.created_at.with_timezone(&chrono::Local).date_naive() == today
        }).collect();
        let today_count = today_recordings.len();

        // Greeting text
        self.greeting_text = if first_name.is_empty() {
            "Welcome back.".to_string()
        } else {
            format!("Welcome back, {}.", first_name)
        };

        // Subtitle
        self.greeting_subtitle = if today_count == 0 {
            "No recordings today yet.".to_string()
        } else if today_count == 1 {
            "You had 1 recording today.".to_string()
        } else {
            format!("You had {} recordings today.", today_count)
        };

        // Build home recordings (most recent first, cap at 5)
        let mut home_recs: Vec<HomeRecording> = today_recordings.iter().take(5).map(|r| {
            let name = r.display_name.as_ref()
                .unwrap_or(&r.directory_name)
                .clone();
            let duration_mins = r.duration_seconds.unwrap_or(0) / 60;
            let summary_line = r.summary.as_ref().and_then(|s| {
                let first = s.split(&['.', '!', '?'][..]).next().unwrap_or("").trim();
                if first.is_empty() { None } else { Some(format!("{}.", first)) }
            });
            let recording_id = r.id.unwrap_or(0);
            HomeRecording { recording_id, name, duration_mins, summary_line }
        }).collect();
        // If no duration info, show at least 1m
        for rec in &mut home_recs {
            if rec.duration_mins == 0 {
                rec.duration_mins = 1;
            }
        }

        // Copy into chat state
        self.chat.greeting_text = self.greeting_text.clone();
        self.chat.greeting_subtitle = self.greeting_subtitle.clone();
        self.chat.home_recordings = home_recs;
        self.chat.selected_action = 0;

        self.chat.placeholder = "Ask anything...".to_string();
    }

    fn init_global_chat(&mut self) {
        // Load world context for owner name
        let world = WorldContext::load().ok()
            .and_then(|wc| WorldData::from_json(&wc.content).ok())
            .unwrap_or_default();

        self.owner_name = if world.owner.name.is_empty() {
            // Fall back to system username, capitalize first letter
            let sys_user = std::env::var("USER").unwrap_or_default();
            if sys_user.is_empty() {
                String::new()
            } else {
                let mut chars = sys_user.chars();
                match chars.next() {
                    Some(c) => c.to_uppercase().to_string() + chars.as_str(),
                    None => String::new(),
                }
            }
        } else {
            world.owner.name.clone()
        };

        // All providers use agent prompt (tools handle data fetching)
        self.chat.system_prompt = chat_prompts::build_agent_global_prompt(&self.owner_name);

        self.chat.context = ChatContext::Global;
        self.chat.suggestions = self.generate_suggestions();
        self.chat.show_suggestions = self.chat.messages.is_empty();
        self.chat.show_home_screen = self.chat.messages.is_empty();
        self.chat.borderless = true;
        self.chat.focus = ChatFocus::ChatInput;

        self.generate_greeting();
    }

    fn init_recording_chat(&mut self, recording: &Recording) {
        // Stash global messages
        self.global_chat_messages = self.chat.messages.clone();

        let world = WorldContext::load().ok()
            .and_then(|wc| WorldData::from_json(&wc.content).ok())
            .unwrap_or_default();

        let owner_name = if world.owner.name.is_empty() {
            "User".to_string()
        } else {
            world.owner.name.clone()
        };

        let recording_name = recording.display_name.as_ref()
            .unwrap_or(&recording.directory_name)
            .clone();

        let summary = recording.summary.as_deref().unwrap_or("");

        // All providers use agent prompt (tools handle data fetching)
        self.chat.system_prompt = chat_prompts::build_agent_recording_prompt(
            &owner_name,
            recording.id.unwrap_or(0),
            &recording_name,
            summary,
        );

        self.chat.context = ChatContext::Recording {
            recording_id: recording.id.unwrap_or(0),
            recording_name,
        };
        self.chat.messages.clear();
        self.chat.input_buffer.clear();
        self.chat.pending_blocks.clear();
        self.chat.current_status = None;
        self.chat.is_generating = false;
        self.chat.scroll_offset = 0;

        self.current_transcript_recording = Some(recording.clone());
        self.chat.suggestions = self.generate_suggestions();
        self.chat.show_suggestions = true;
        self.chat.borderless = true;
        self.chat.show_home_screen = false;
    }

    fn restore_global_chat(&mut self) {
        self.chat.messages = std::mem::take(&mut self.global_chat_messages);
        self.chat.context = ChatContext::Global;
        self.chat.input_buffer.clear();
        self.chat.pending_blocks.clear();
        self.chat.current_status = None;
        self.chat.is_generating = false;
        self.chat.scroll_offset = 0;
        self.current_transcript_recording = None;
        self.chat.suggestions = self.generate_suggestions();
        self.chat.show_suggestions = self.chat.messages.is_empty();
        self.chat.show_home_screen = self.chat.messages.is_empty();
        // Restore home-screen chat state when returning to Main
        if self.current_view == DashboardView::Main {
            self.chat.borderless = true;
            self.chat.focus = ChatFocus::ChatInput;
        } else {
            self.chat.borderless = false;
            self.chat.focus = ChatFocus::Table;
        }
    }

    fn send_chat_message(&mut self) {
        let user_msg = if self.chat.show_suggestions && !self.chat.suggestions.is_empty() {
            // Check if "Ask Scriba anything..." (last option) is selected
            if self.chat.selected_suggestion >= self.chat.suggestions.len() {
                // Switch to free-form input mode
                self.chat.show_suggestions = false;
                return;
            }
            // Use selected suggestion
            let msg = self.chat.suggestions[self.chat.selected_suggestion].clone();
            self.chat.show_suggestions = false;
            msg
        } else if !self.chat.input_buffer.is_empty() {
            let msg = self.chat.input_buffer.clone();
            self.chat.input_buffer.clear();
            self.chat.show_suggestions = false;
            self.chat.show_home_screen = false;
            msg
        } else {
            return;
        };

        // Add user message
        self.chat.messages.push(ChatMessage::text(ChatRole::User, user_msg.clone()));

        // Spawn generation pipeline
        let (event_tx, event_rx) = mpsc::channel::<ChatStreamEvent>(100);
        self.chat.stream_rx = Some(event_rx);
        self.chat.is_generating = true;
        self.chat.auto_scroll = true; // re-engage for new response
        self.chat.pending_blocks.clear();
        self.chat.current_status = Some("Preparing...".to_string());

        let config = self.config.enrichment.clone();
        let system_prompt = self.chat.system_prompt.clone();
        let messages: Vec<(String, String)> = self.chat.messages.iter().map(|m| {
            let role = match m.role {
                ChatRole::User => "User",
                ChatRole::Assistant => "Assistant",
                ChatRole::System => "System",
            };
            (role.to_string(), m.content())
        }).collect();

        let needs_compaction = self.chat.needs_compaction();
        self.chat.pending_blocks.clear();
        self.chat.generation_task = Some(tokio::spawn(async move {
            chat_agent_pipeline(config, system_prompt, messages, user_msg, needs_compaction, event_tx).await;
        }));
    }

    /// Execute the selected action from the home screen quick-action menu.
    fn execute_home_action(&mut self) {
        let idx = self.chat.selected_action.min(
            self.chat.home_recordings.len().saturating_sub(1),
        );
        let rec = self.chat.home_recordings[idx].clone();
        self.chat.action_menu_open = false;

        match self.chat.action_menu_selection {
            0 => {
                // View transcript — find the recording and open transcript popup
                if let Some(recording) = self.recordings.iter()
                    .find(|r| r.id == Some(rec.recording_id))
                    .cloned()
                {
                    if recording.has_transcript {
                        if let Ok(content) = self.load_transcript_content(&recording) {
                            self.transcript_content = content;
                            self.show_transcript = true;
                            self.load_enrichment_data(&recording);
                            self.init_recording_chat(&recording);
                        }
                    }
                }
            }
            1 | 2 => {
                // Fresh chat for this recording
                self.chat.messages.clear();
                self.chat.pending_blocks.clear();
                self.chat.current_status = None;
                self.chat.scroll_offset = 0;
                self.chat.auto_scroll = true;
                self.chat.invalidate_cache();
                self.chat.show_home_screen = false;
                self.chat.show_suggestions = false;
                self.global_chat_messages.clear();

                if self.chat.action_menu_selection == 1 {
                    // Summarize — send immediately
                    self.chat.input_buffer = format!("Summarize my {} recording (recording id={}).", rec.name, rec.recording_id);
                    self.send_chat_message();
                } else {
                    // Ask about it — fill input, let user edit
                    self.chat.input_buffer = format!("What happened in my {}? (recording id={})", rec.name, rec.recording_id);
                }
            }
            _ => {}
        }
    }

    fn handle_chat_key(&mut self, key_code: KeyCode) -> bool {
        match key_code {
            KeyCode::Tab => {
                if self.current_view == DashboardView::Main {
                    // Main → Browse
                    self.current_view = DashboardView::Browse;
                } else {
                    // Transcript view: toggle focus
                    self.chat.focus = match self.chat.focus {
                        ChatFocus::Table => ChatFocus::ChatInput,
                        ChatFocus::ChatInput => ChatFocus::Table,
                    };
                }
                true
            }
            _ if self.chat.focus == ChatFocus::ChatInput => {
                match key_code {
                    KeyCode::Enter => {
                        let on_home = self.chat.show_home_screen
                            && self.chat.input_buffer.is_empty()
                            && !self.chat.home_recordings.is_empty();
                        if on_home && !self.chat.action_menu_open {
                            // Open action menu for selected recording
                            self.chat.action_menu_open = true;
                            self.chat.action_menu_selection = 0;
                        } else if on_home && self.chat.action_menu_open {
                            // Execute selected action
                            self.execute_home_action();
                        } else if self.chat.is_generating {
                            if !self.chat.input_buffer.is_empty() {
                                self.chat.pending_message = Some(self.chat.input_buffer.clone());
                                self.chat.input_buffer.clear();
                            }
                        } else {
                            self.send_chat_message();
                        }
                    }
                    KeyCode::Esc => {
                        if self.chat.action_menu_open {
                            self.chat.action_menu_open = false;
                        } else if !self.chat.input_buffer.is_empty() {
                            self.chat.input_buffer.clear();
                        } else if self.current_view == DashboardView::Main
                            && !self.chat.show_home_screen
                            && !self.chat.is_generating
                        {
                            // Return to home screen from chat
                            self.chat.messages.clear();
                            self.chat.pending_blocks.clear();
                            self.chat.current_status = None;
                            self.chat.scroll_offset = 0;
                            self.chat.show_home_screen = true;
                            self.chat.show_suggestions = true;
                            self.chat.selected_suggestion = 0;
                            self.chat.selected_action = 0;
                            self.chat.action_menu_open = false;
                            self.chat.auto_scroll = true;
                            self.chat.invalidate_cache();
                            self.generate_greeting();
                        }
                        else if self.current_view != DashboardView::Main {
                            self.chat.focus = ChatFocus::Table;
                        }
                        // On home screen with nothing to dismiss — do nothing
                    }
                    KeyCode::Backspace => {
                        self.chat.input_buffer.pop();
                    }
                    KeyCode::Up => {
                        if self.chat.action_menu_open {
                            if self.chat.action_menu_selection > 0 {
                                self.chat.action_menu_selection -= 1;
                            }
                        } else if self.chat.show_home_screen
                            && self.chat.input_buffer.is_empty()
                            && !self.chat.home_recordings.is_empty()
                        {
                            if self.chat.selected_action > 0 {
                                self.chat.selected_action -= 1;
                            }
                        } else if self.chat.show_suggestions && !self.chat.suggestions.is_empty() {
                            if self.chat.selected_suggestion > 0 {
                                self.chat.selected_suggestion -= 1;
                            }
                        } else {
                            if self.chat.auto_scroll {
                                self.chat.scroll_offset = usize::MAX;
                            }
                            self.chat.auto_scroll = false;
                            self.chat.scroll_offset = self.chat.scroll_offset.saturating_sub(1);
                        }
                    }
                    KeyCode::Down => {
                        if self.chat.action_menu_open {
                            if self.chat.action_menu_selection < 2 {
                                self.chat.action_menu_selection += 1;
                            }
                        } else if self.chat.show_home_screen
                            && self.chat.input_buffer.is_empty()
                            && !self.chat.home_recordings.is_empty()
                        {
                            let max_idx = self.chat.home_recordings.len().saturating_sub(1);
                            if self.chat.selected_action < max_idx {
                                self.chat.selected_action += 1;
                            }
                        } else if self.chat.show_suggestions && !self.chat.suggestions.is_empty() {
                            let max_idx = self.chat.suggestions.len();
                            if self.chat.selected_suggestion < max_idx {
                                self.chat.selected_suggestion += 1;
                            }
                        } else if !self.chat.auto_scroll {
                            self.chat.scroll_offset += 1;
                            self.chat.auto_scroll = true;
                        }
                    }
                    KeyCode::Char(c) => {
                        self.chat.input_buffer.push(c);
                        self.chat.show_suggestions = false;
                        self.chat.action_menu_open = false;
                    }
                    _ => {}
                }
                true
            }
            _ => false,
        }
    }

    fn handle_transcript_chat_key(&mut self, key_code: KeyCode) -> bool {
        if key_code == KeyCode::Tab {
            self.chat.focus = match self.chat.focus {
                ChatFocus::Table => ChatFocus::ChatInput, // Table = transcript scroll in this context
                ChatFocus::ChatInput => ChatFocus::Table,
            };
            return true;
        }

        if self.chat.focus != ChatFocus::ChatInput {
            return false;
        }

        match key_code {
            KeyCode::Enter => {
                if self.chat.is_generating {
                    if !self.chat.input_buffer.is_empty() {
                        self.chat.pending_message = Some(self.chat.input_buffer.clone());
                        self.chat.input_buffer.clear();
                    }
                } else {
                    self.send_chat_message();
                }
            }
            KeyCode::Backspace => {
                self.chat.input_buffer.pop();
            }
            KeyCode::Up => {
                if self.chat.show_suggestions && !self.chat.suggestions.is_empty() {
                    if self.chat.selected_suggestion > 0 {
                        self.chat.selected_suggestion -= 1;
                    }
                } else {
                    if self.chat.auto_scroll {
                        self.chat.scroll_offset = usize::MAX;
                    }
                    self.chat.auto_scroll = false;
                    self.chat.scroll_offset = self.chat.scroll_offset.saturating_sub(1);
                }
            }
            KeyCode::Down => {
                if self.chat.show_suggestions && !self.chat.suggestions.is_empty() {
                    let max_idx = self.chat.suggestions.len();
                    if self.chat.selected_suggestion < max_idx {
                        self.chat.selected_suggestion += 1;
                    }
                } else if !self.chat.auto_scroll {
                    self.chat.scroll_offset += 1;
                    self.chat.auto_scroll = true;
                }
            }
            KeyCode::Char(c) => {
                self.chat.input_buffer.push(c);
                self.chat.show_suggestions = false;
            }
            KeyCode::Esc => {
                if !self.chat.input_buffer.is_empty() {
                    self.chat.input_buffer.clear();
                } else {
                    self.chat.focus = ChatFocus::Table;
                }
            }
            _ => {}
        }
        true
    }

    // ── Mouse Handling ────────────────────────────────────────────────────

    fn handle_mouse_event(&mut self, mouse: crossterm::event::MouseEvent) {
        let rect = self.chat.panel_rect;
        if rect.width == 0 || rect.height == 0 {
            return;
        }
        // Only handle if mouse is within the chat panel
        if mouse.column < rect.x || mouse.column >= rect.x + rect.width
            || mouse.row < rect.y || mouse.row >= rect.y + rect.height
        {
            return;
        }

        match mouse.kind {
            MouseEventKind::ScrollUp => {
                if self.chat.auto_scroll {
                    self.chat.scroll_offset = self.chat.total_content_lines;
                }
                self.chat.auto_scroll = false;
                self.chat.scroll_offset = self.chat.scroll_offset.saturating_sub(3);
            }
            MouseEventKind::ScrollDown => {
                if !self.chat.auto_scroll {
                    self.chat.scroll_offset += 3;
                    let inner_height = rect.height.saturating_sub(2) as usize;
                    let has_conv = !self.chat.messages.is_empty() || self.chat.is_generating;
                    let reserved = if has_conv { 2 } else { 1 };
                    let chat_height = inner_height.saturating_sub(reserved);
                    if self.chat.scroll_offset + chat_height >= self.chat.total_content_lines {
                        self.chat.auto_scroll = true;
                    }
                }
            }
            MouseEventKind::Down(crossterm::event::MouseButton::Left) => {
                // Start text selection
                if let Some(pos) = self.chat.mouse_to_content_pos(mouse.column, mouse.row) {
                    self.chat.selection_anchor = Some(pos);
                    self.chat.selection_end = None; // reset until drag
                    self.chat.focus = ChatFocus::ChatInput;
                }
            }
            MouseEventKind::Drag(crossterm::event::MouseButton::Left) => {
                // Extend selection + auto-scroll at edges
                if self.chat.selection_anchor.is_some() {
                    let edge_zone = 2u16; // rows from edge that trigger auto-scroll
                    let top_edge = rect.y + 1; // inside top border
                    let inner_height = rect.height.saturating_sub(2) as usize;
                    let has_conv = !self.chat.messages.is_empty() || self.chat.is_generating;
                    let reserved = if has_conv { 2 } else { 1 };
                    let chat_height = inner_height.saturating_sub(reserved);
                    let bottom_edge = rect.y + 1 + chat_height as u16;

                    if mouse.row < top_edge + edge_zone && mouse.row >= rect.y {
                        // Dragging near top — scroll up
                        if self.chat.auto_scroll {
                            self.chat.scroll_offset = self.chat.total_content_lines;
                        }
                        self.chat.auto_scroll = false;
                        self.chat.scroll_offset = self.chat.scroll_offset.saturating_sub(2);
                    } else if mouse.row >= bottom_edge.saturating_sub(edge_zone) && mouse.row < rect.y + rect.height {
                        // Dragging near bottom — scroll down
                        if !self.chat.auto_scroll {
                            self.chat.scroll_offset += 2;
                            if self.chat.scroll_offset + chat_height >= self.chat.total_content_lines {
                                self.chat.auto_scroll = true;
                            }
                        }
                    }

                    if let Some(pos) = self.chat.mouse_to_content_pos(mouse.column, mouse.row) {
                        self.chat.selection_end = Some(pos);
                    }
                }
            }
            MouseEventKind::Up(crossterm::event::MouseButton::Left) => {
                // Finalize selection and copy
                if let (Some(anchor), Some(end)) = (self.chat.selection_anchor, self.chat.selection_end) {
                    let selected = self.chat.extract_selected_text(anchor, end);
                    if !selected.trim().is_empty() {
                        use arboard::Clipboard;
                        if let Ok(mut clipboard) = Clipboard::new() {
                            let _ = clipboard.set_text(&selected);
                            self.notification_message = Some(("Copied to clipboard".to_string(), 15));
                        }
                    }
                    // Keep selection visible (don't clear anchor/end) — cleared on next click
                } else {
                    // Single click (no drag) — clear any existing selection
                    self.chat.selection_anchor = None;
                    self.chat.selection_end = None;
                }
            }
            _ => {}
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Onboarding
    // ─────────────────────────────────────────────────────────────────────────

    async fn handle_onboarding_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        let ob = match self.onboarding.as_mut() {
            Some(ob) => ob,
            None => return Ok(DashboardAction::Continue),
        };

        // Esc at any step → skip onboarding
        if matches!(key_code, KeyCode::Esc) {
            self.onboarding = None;
            self.current_view = DashboardView::Main;
            return Ok(DashboardAction::Continue);
        }

        match ob.step {
            OnboardingStep::Entrance => {
                // No key handling during entrance animation
            }
            OnboardingStep::Intro => {
                if !ob.text_complete {
                    ob.visible_chars = ob.full_text.chars().count();
                    ob.text_complete = true;
                } else if matches!(key_code, KeyCode::Enter) {
                    ob.step = OnboardingStep::ModeSelection;
                    ob.anim_frame = 0;
                    ob.set_step_text("I use AI to extract names, topics, and summaries\nfrom your recordings. How should I do it?", false);
                }
            }
            OnboardingStep::ModeSelection => {
                match key_code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        ob.selected_mode = 0;
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        ob.selected_mode = 1;
                    }
                    KeyCode::Enter => {
                        if ob.selected_mode == 0 {
                            ob.step = OnboardingStep::ProviderSelection;
                            ob.anim_frame = 0;
                            ob.set_step_text("Which cloud provider?", false);
                        } else {
                            ob.selected_mode = 1;
                            // Set local mode defaults in config
                            self.config.enrichment.mode = EnrichmentMode::Local {
                                ollama_endpoint: "http://localhost:11434".to_string(),
                                ollama_model: "mistral:latest".to_string(),
                            };
                            let _ = self.config.save();

                            // Start system checks
                            ob.step = OnboardingStep::SystemCheck;
                            ob.anim_frame = 0;
                            ob.set_step_text("Checking your setup...", false);
                            ob.system_checks = vec![
                                ("FFmpeg".to_string(), CheckStatus::Pending),
                                ("Ollama".to_string(), CheckStatus::Pending),
                                ("Ollama server".to_string(), CheckStatus::Pending),
                            ];
                            ob.system_check_done = false;
                            ob.system_check_selection = 0;
                            ob.ollama_reachable = false;

                            let (tx, rx) = mpsc::unbounded_channel();
                            ob.system_check_rx = Some(rx);
                            ob.system_check_task = Some(tokio::spawn(async move {
                                // Check 1: FFmpeg
                                let _ = tx.send((0, false, String::new())); // mark running
                                let ffmpeg_ok = crate::core::transcription::find_ffmpeg().is_ok();
                                let _ = tx.send((0, ffmpeg_ok, if ffmpeg_ok { String::new() } else {
                                    "Install with: brew install ffmpeg".to_string()
                                }));

                                // Check 2: Ollama binary
                                let _ = tx.send((1, false, String::new())); // mark running
                                let ollama_bin = tokio::process::Command::new("which")
                                    .arg("ollama")
                                    .output()
                                    .await
                                    .map(|o| o.status.success())
                                    .unwrap_or(false);
                                let _ = tx.send((1, ollama_bin, if ollama_bin { String::new() } else {
                                    "Install with: brew install ollama".to_string()
                                }));

                                if !ollama_bin {
                                    // Skip server check if binary not found
                                    let _ = tx.send((2, false, "Install Ollama first".to_string()));
                                    return;
                                }

                                // Check 3: Ollama server responding
                                let _ = tx.send((2, false, String::new())); // mark running
                                let client = reqwest::Client::builder()
                                    .timeout(std::time::Duration::from_secs(5))
                                    .build()
                                    .unwrap();
                                let server_ok = client
                                    .get("http://localhost:11434/api/tags")
                                    .send()
                                    .await
                                    .map(|r| r.status().is_success())
                                    .unwrap_or(false);
                                let _ = tx.send((2, server_ok, if server_ok { String::new() } else {
                                    "Start with: ollama serve".to_string()
                                }));
                            }));
                        }
                    }
                    _ => {}
                }
            }
            OnboardingStep::ProviderSelection => {
                match key_code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        ob.selected_provider = ob.selected_provider.saturating_sub(1);
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        ob.selected_provider = (ob.selected_provider + 1).min(2);
                    }
                    KeyCode::Enter => {
                        let p = match ob.selected_provider {
                            0 => CloudProvider::Anthropic,
                            1 => CloudProvider::OpenAI,
                            _ => CloudProvider::Google,
                        };
                        self.config.enrichment.mode = EnrichmentMode::Cloud {
                            provider: p.clone(),
                            api_key: String::new(),
                            model: None,
                        };
                        ob.step = OnboardingStep::ApiKeyEntry;
                        ob.anim_frame = 0;
                        ob.set_step_text(&format!(
                            "Enter your {} API key.\n\n\
                             Paste it below:",
                            p.display_name()
                        ), false);
                    }
                    _ => {}
                }
            }
            OnboardingStep::ApiKeyEntry => {
                match key_code {
                    KeyCode::Enter => {
                        if !ob.api_key_input.trim().is_empty() {
                            let key = ob.api_key_input.trim().to_string();
                            if let EnrichmentMode::Cloud { api_key, .. } = &mut self.config.enrichment.mode {
                                *api_key = key.clone();
                            }
                            let _ = self.config.save();

                            // Start validation
                            ob.step = OnboardingStep::ApiKeyValidation;
                            ob.anim_frame = 0;
                            ob.api_key_valid = None;
                            ob.set_step_text("Testing your API key...", false);

                            let config = self.config.enrichment.clone();
                            ob.validation_task = Some(tokio::spawn(async move {
                                let provider = crate::enrichment::create_provider(&config);
                                match provider.health_check().await {
                                    Ok(()) => Ok(true),
                                    Err(_) => Ok(false),
                                }
                            }));
                        }
                    }
                    KeyCode::Char(c) => {
                        ob.api_key_input.push(c);
                    }
                    KeyCode::Backspace => {
                        ob.api_key_input.pop();
                    }
                    _ => {}
                }
            }
            OnboardingStep::ApiKeyValidation => {
                // Validation result is resolved by the tick handler.
                // Here we only handle retry/skip choices after validation fails.
                if ob.api_key_valid == Some(false) {
                    match key_code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            ob.validation_fail_selection = 0;
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            ob.validation_fail_selection = 1;
                        }
                        KeyCode::Enter => {
                            if ob.validation_fail_selection == 0 {
                                ob.step = OnboardingStep::ApiKeyEntry;
                                ob.api_key_input.clear();
                                ob.anim_frame = 0;
                                let provider_name = self.config.enrichment.provider_display_name().to_string();
                                ob.set_step_text(&format!(
                                    "Let's try again. Paste your {} API key:",
                                    provider_name
                                ), false);
                            } else {
                                ob.step = OnboardingStep::AskName;
                                ob.anim_frame = 0;
                                ob.set_step_text(
                                    "No problem. You can set the key later in Settings.\n\n\
                                     What's your name?",
                                    true,
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
            OnboardingStep::SystemCheck => {
                if ob.system_check_done {
                    match key_code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            ob.system_check_selection = 0;
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            ob.system_check_selection = 1;
                        }
                        KeyCode::Enter => {
                            if ob.system_check_selection == 0 {
                                // "Check again" — re-run all checks
                                ob.system_check_done = false;
                                ob.system_check_selection = 0;
                                ob.ollama_reachable = false;
                                ob.set_step_text("Checking your setup...", false);
                                ob.system_checks = vec![
                                    ("FFmpeg".to_string(), CheckStatus::Pending),
                                    ("Ollama".to_string(), CheckStatus::Pending),
                                    ("Ollama server".to_string(), CheckStatus::Pending),
                                ];

                                let (tx, rx) = mpsc::unbounded_channel();
                                ob.system_check_rx = Some(rx);
                                ob.system_check_task = Some(tokio::spawn(async move {
                                    let _ = tx.send((0, false, String::new()));
                                    let ffmpeg_ok = crate::core::transcription::find_ffmpeg().is_ok();
                                    let _ = tx.send((0, ffmpeg_ok, if ffmpeg_ok { String::new() } else {
                                        "Install with: brew install ffmpeg".to_string()
                                    }));

                                    let _ = tx.send((1, false, String::new()));
                                    let ollama_bin = tokio::process::Command::new("which")
                                        .arg("ollama")
                                        .output()
                                        .await
                                        .map(|o| o.status.success())
                                        .unwrap_or(false);
                                    let _ = tx.send((1, ollama_bin, if ollama_bin { String::new() } else {
                                        "Install with: brew install ollama".to_string()
                                    }));

                                    if !ollama_bin {
                                        let _ = tx.send((2, false, "Install Ollama first".to_string()));
                                        return;
                                    }

                                    let _ = tx.send((2, false, String::new()));
                                    let client = reqwest::Client::builder()
                                        .timeout(std::time::Duration::from_secs(5))
                                        .build()
                                        .unwrap();
                                    let server_ok = client
                                        .get("http://localhost:11434/api/tags")
                                        .send()
                                        .await
                                        .map(|r| r.status().is_success())
                                        .unwrap_or(false);
                                    let _ = tx.send((2, server_ok, if server_ok { String::new() } else {
                                        "Start with: ollama serve".to_string()
                                    }));
                                }));
                            } else {
                                // "Continue anyway" — advance to ModelSetup
                                ob.step = OnboardingStep::ModelSetup;
                                ob.anim_frame = 0;
                                ob.setup_phase = 0;
                                ob.set_step_text("Choose your transcription model", false);
                            }
                        }
                        _ => {}
                    }
                }
            }
            OnboardingStep::ModelSetup => {
                match ob.setup_phase {
                    0 => {
                        // Whisper model selection
                        match key_code {
                            KeyCode::Up | KeyCode::Char('k') => {
                                ob.whisper_model_selection = ob.whisper_model_selection.saturating_sub(1);
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                ob.whisper_model_selection = (ob.whisper_model_selection + 1).min(WHISPER_MODELS.len() - 1);
                            }
                            KeyCode::Enter => {
                                if ob.ollama_reachable {
                                    ob.setup_phase = 1;
                                    ob.set_step_text("Choose your enrichment model", false);
                                    // Fetch available models if not already done
                                    if !ob.ollama_models_fetched {
                                        ob.ollama_models_fetched = true;
                                        let endpoint = if let EnrichmentMode::Local { ollama_endpoint, .. } = &self.config.enrichment.mode {
                                            ollama_endpoint.clone()
                                        } else {
                                            "http://localhost:11434".to_string()
                                        };
                                        let (tx, rx) = mpsc::channel(1);
                                        self.ollama_models_rx = Some(rx);
                                        tokio::spawn(async move {
                                            let result = OllamaClient::fetch_models(&endpoint).await;
                                            let _ = tx.send(result.map_err(|e| e.to_string())).await;
                                        });
                                    }
                                } else {
                                    // Skip Ollama model selection, go to confirm
                                    ob.setup_phase = 2;
                                    ob.system_check_selection = 0;
                                    ob.set_step_text("Ready to set up:", false);
                                }
                            }
                            _ => {}
                        }
                    }
                    1 => {
                        // Ollama model selection
                        let model_count = ob.ollama_available_models.len().max(1);
                        match key_code {
                            KeyCode::Up | KeyCode::Char('k') => {
                                ob.ollama_model_selection = ob.ollama_model_selection.saturating_sub(1);
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                ob.ollama_model_selection = (ob.ollama_model_selection + 1).min(model_count - 1);
                            }
                            KeyCode::Enter => {
                                // Set the selected model in config
                                if let Some(model_name) = ob.ollama_available_models.get(ob.ollama_model_selection) {
                                    let model_with_tag = if model_name.contains(':') {
                                        model_name.clone()
                                    } else {
                                        format!("{}:latest", model_name)
                                    };
                                    if let EnrichmentMode::Local { ollama_model, .. } = &mut self.config.enrichment.mode {
                                        *ollama_model = model_with_tag;
                                    }
                                    let _ = self.config.save();
                                }
                                ob.setup_phase = 2;
                                ob.set_step_text("Ready to set up:", false);
                            }
                            _ => {}
                        }
                    }
                    2 => {
                        // Download confirmation: "Download now" / "Skip for now"
                        match key_code {
                            KeyCode::Up | KeyCode::Char('k') => {
                                ob.system_check_selection = 0;
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                ob.system_check_selection = 1;
                            }
                            KeyCode::Enter => {
                                if ob.system_check_selection == 0 {
                                    // "Download now"
                                    ob.setup_phase = 3;
                                    ob.set_step_text("Setting up your models...", false);

                                    // Build download items
                                    let whisper_size = WHISPER_MODELS[ob.whisper_model_selection].0;
                                    let whisper_label = WHISPER_MODELS[ob.whisper_model_selection].1.to_string();
                                    let whisper_already = crate::core::transcription::check_whisper_model(whisper_size);

                                    let mut items: Vec<(String, DownloadStatus)> = Vec::new();
                                    items.push((
                                        format!("Whisper {}", whisper_label.replace(" (Recommended)", "")),
                                        if whisper_already { DownloadStatus::Done } else { DownloadStatus::Pending },
                                    ));

                                    let ollama_model = if let EnrichmentMode::Local { ollama_model, .. } = &self.config.enrichment.mode {
                                        ollama_model.clone()
                                    } else {
                                        "mistral:latest".to_string()
                                    };

                                    let need_ollama_pull = ob.ollama_reachable;
                                    if need_ollama_pull {
                                        items.push((
                                            format!("Pulling {}", ollama_model.split(':').next().unwrap_or(&ollama_model)),
                                            DownloadStatus::Pending,
                                        ));
                                    }
                                    ob.download_items = items;

                                    // Start downloads
                                    let (tx, rx) = mpsc::unbounded_channel();
                                    ob.download_rx = Some(rx);

                                    let endpoint = if let EnrichmentMode::Local { ollama_endpoint, .. } = &self.config.enrichment.mode {
                                        ollama_endpoint.clone()
                                    } else {
                                        "http://localhost:11434".to_string()
                                    };
                                    let ollama_model_clone = ollama_model.clone();

                                    ob.download_task = Some(tokio::spawn(async move {
                                        // Download Whisper model
                                        if !whisper_already {
                                            let (wtx, mut wrx) = mpsc::unbounded_channel();
                                            let tx_w = tx.clone();
                                            let whisper_handle = tokio::spawn(async move {
                                                let result = crate::core::transcription::download_whisper_with_progress(whisper_size, wtx).await;
                                                if let Err(_) = result {
                                                    let _ = tx_w.send(DownloadProgress {
                                                        index: 0,
                                                        status: DownloadStatus::Failed("Download failed".to_string()),
                                                    });
                                                }
                                            });

                                            // Forward whisper progress
                                            let tx_wf = tx.clone();
                                            tokio::spawn(async move {
                                                while let Some(pct) = wrx.recv().await {
                                                    if pct >= 100 {
                                                        let _ = tx_wf.send(DownloadProgress { index: 0, status: DownloadStatus::Done });
                                                    } else {
                                                        let _ = tx_wf.send(DownloadProgress { index: 0, status: DownloadStatus::InProgress(pct) });
                                                    }
                                                }
                                            });

                                            let _ = whisper_handle.await;
                                        }

                                        // Pull Ollama model
                                        if need_ollama_pull {
                                            let (otx, mut orx) = mpsc::unbounded_channel();
                                            let tx_o = tx.clone();
                                            let endpoint_clone = endpoint.clone();
                                            let model_clone = ollama_model_clone.clone();
                                            let ollama_handle = tokio::spawn(async move {
                                                let result = crate::enrichment::pull_model_with_progress(&endpoint_clone, &model_clone, otx).await;
                                                if let Err(_) = result {
                                                    let _ = tx_o.send(DownloadProgress {
                                                        index: 1,
                                                        status: DownloadStatus::Failed("Pull failed".to_string()),
                                                    });
                                                }
                                            });

                                            // Forward ollama progress
                                            let tx_of = tx.clone();
                                            tokio::spawn(async move {
                                                while let Some((_status, pct)) = orx.recv().await {
                                                    if let Some(p) = pct {
                                                        if p >= 100 {
                                                            let _ = tx_of.send(DownloadProgress { index: 1, status: DownloadStatus::Done });
                                                        } else {
                                                            let _ = tx_of.send(DownloadProgress { index: 1, status: DownloadStatus::InProgress(p) });
                                                        }
                                                    }
                                                }
                                            });

                                            let _ = ollama_handle.await;
                                        }
                                    }));
                                } else {
                                    // "Skip for now"
                                    // Save chosen whisper model to config
                                    let whisper_size = WHISPER_MODELS[ob.whisper_model_selection].0;
                                    self.config.transcription = TranscriptionMode::Local { model_size: whisper_size };
                                    let _ = self.config.save();

                                    ob.step = OnboardingStep::AskName;
                                    ob.anim_frame = 0;
                                    ob.set_step_text(
                                        "No problem. Models will download on first use.\n\n\
                                         What's your name?",
                                        true,
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                    3 => {
                        // Downloading phase — check for completion or failure
                        let all_done = ob.download_items.iter().all(|(_, s)| matches!(s, DownloadStatus::Done));
                        let any_failed = ob.download_items.iter().any(|(_, s)| matches!(s, DownloadStatus::Failed(_)));
                        if all_done || any_failed {
                            match key_code {
                                KeyCode::Enter => {
                                    // Save whisper model to config
                                    let whisper_size = WHISPER_MODELS[ob.whisper_model_selection].0;
                                    self.config.transcription = TranscriptionMode::Local { model_size: whisper_size };
                                    let _ = self.config.save();

                                    ob.step = OnboardingStep::AskName;
                                    ob.anim_frame = 0;
                                    if all_done {
                                        ob.set_step_text(
                                            "All set!\n\nWhat's your name?",
                                            true,
                                        );
                                    } else {
                                        ob.set_step_text(
                                            "Some downloads had issues. You can retry later.\n\n\
                                             What's your name?",
                                            true,
                                        );
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
            OnboardingStep::AskName => {
                if !ob.text_complete {
                    ob.visible_chars = ob.full_text.chars().count();
                    ob.text_complete = true;
                } else {
                    match key_code {
                        KeyCode::Enter => {
                            if !ob.user_name.trim().is_empty() {
                                ob.step = OnboardingStep::AskRole;
                                ob.anim_frame = 0;
                                let name = ob.user_name.clone();
                                ob.set_step_text(&format!(
                                    "Nice to meet you, {}.\n\n\
                                     Tell me about yourself \u{2014}\n\
                                     What do you do? What's your company?\n\
                                     Who do you work with?\n\n\
                                     Just write naturally.",
                                    name
                                ), true);
                            }
                        }
                        KeyCode::Char(c) => {
                            ob.user_name.push(c);
                        }
                        KeyCode::Backspace => {
                            ob.user_name.pop();
                        }
                        _ => {}
                    }
                }
            }
            OnboardingStep::AskRole => {
                if !ob.text_complete {
                    ob.visible_chars = ob.full_text.chars().count();
                    ob.text_complete = true;
                } else {
                    match key_code {
                        KeyCode::Enter => {
                            if !ob.user_role.trim().is_empty() {
                                ob.step = OnboardingStep::Processing;
                                ob.anim_frame = 0;
                                ob.set_step_text("Setting up your world...", false);
                                self.start_onboarding_processing();
                            }
                        }
                        KeyCode::Char(c) => {
                            ob.user_role.push(c);
                        }
                        KeyCode::Backspace => {
                            ob.user_role.pop();
                        }
                        _ => {}
                    }
                }
            }
            OnboardingStep::Processing => {
                // If provider failed and text is complete, Enter advances
                if ob.text_complete && ob.processing_task.is_none() && !ob.ollama_available {
                    if matches!(key_code, KeyCode::Enter) {
                        ob.step = OnboardingStep::Done;
                        ob.anim_frame = 0;
                        ob.set_step_text(
                            "Your world is ready.\n\n\
                             Every recording you make will be enriched\n\
                             with what I know about you and your world.\n\n\
                             Let's go.",
                            true,
                        );
                    }
                }
            }
            OnboardingStep::Confirmation => {
                match key_code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        ob.confirm_selection = 0;
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        ob.confirm_selection = 1;
                    }
                    KeyCode::Enter => {
                        if ob.confirm_selection == 0 {
                            ob.step = OnboardingStep::Done;
                            ob.anim_frame = 0;
                            ob.set_step_text(
                                "Your world is ready!\n\n\
                                 I'll remember all of this. Every recording\n\
                                 you make, I'll enrich with what I know about\n\
                                 you and your world.\n\n\
                                 Time to fly!",
                                true,
                            );
                        } else {
                            // Go back to AskName with values preserved
                            ob.step = OnboardingStep::AskName;
                            ob.anim_frame = 0;
                            ob.set_step_text("What's your name?", true);
                            // Delete the world.md that was created during processing
                            let _ = std::fs::remove_file(WorldContext::file_path());
                        }
                    }
                    _ => {}
                }
            }
            OnboardingStep::Done => {
                if !ob.text_complete {
                    // Skip typewriter
                    ob.visible_chars = ob.full_text.chars().count();
                    ob.text_complete = true;
                } else if !ob.transitioning && matches!(key_code, KeyCode::Enter) {
                    // Start fade-out transition
                    ob.transitioning = true;
                    ob.transition_frame = 0;
                }
            }
        }

        Ok(DashboardAction::Continue)
    }

    fn start_onboarding_processing(&mut self) {
        let ob = match self.onboarding.as_mut() {
            Some(ob) => ob,
            None => return,
        };

        // Build seed content from user inputs
        let mut seed = format!("My name is {}. ", ob.user_name.trim());
        seed.push_str(ob.user_role.trim());

        let config = self.config.clone();

        ob.processing_task = Some(tokio::spawn(async move {
            let mut db = Database::new()?;

            // For local mode, retry up to 3 times with a short delay
            // (Ollama may need a moment after a fresh model pull)
            let max_attempts = if config.enrichment.is_local() { 3 } else { 1 };
            let mut result = None;
            for attempt in 0..max_attempts {
                if attempt > 0 {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
                match initialize_world_from_seed(&mut db, &config, &seed).await? {
                    Some(data) => {
                        result = Some(data);
                        break;
                    }
                    None => {}
                }
            }

            if let Some(data) = result {
                Ok((Some(data), None))
            } else {
                // Provider unavailable — diagnose why
                let hint = if config.enrichment.is_local() {
                    let endpoint = config.enrichment.ollama_endpoint();
                    let model = config.enrichment.ollama_model();
                    let client = OllamaClient::new(&endpoint, &model);
                    let status = client.diagnose().await;
                    match status.hint() {
                        Some(h) => Some(h),
                        None => {
                            // Diagnose says Ready but LLM call still failed
                            Some(format!(
                                "Ollama is running but the LLM call failed.\n\
                                 Model: {}\n\n\
                                 Try running a test query:\n\
                                   ollama run {} \"hello\"",
                                model, model.split(':').next().unwrap_or(&model)
                            ))
                        }
                    }
                } else {
                    let provider_name = config.enrichment.provider_display_name().to_string();
                    if config.enrichment.resolve_api_key().is_none() {
                        Some(format!(
                            "No API key set for {}.",
                            provider_name
                        ))
                    } else {
                        Some(format!(
                            "{} is not reachable.\n\
                             Check your API key and network connection.",
                            provider_name
                        ))
                    }
                };
                Ok((None, hint))
            }
        }));
    }

    fn render_onboarding(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let ob = match &self.onboarding {
            Some(ob) => ob,
            None => return,
        };

        // Exit dim fade
        if ob.transitioning {
            self.render_dim_fade(f, area, ob);
            return;
        }

        // Entrance: block-art SCRIBA logo, 3-stage fade (~2.5s, 25 frames)
        if ob.step == OnboardingStep::Entrance {
            let frame = ob.anim_frame.min(24);
            if frame < 6 {
                // Blank
                return;
            }
            let logo_lines = [
                "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} \u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}  \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} ",
                "\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}",
                "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}     \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2554}\u{255D}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2554}\u{255D}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2551}",
                "\u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2551}     \u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2551}",
                "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2551}\u{255A}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}  \u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2554}\u{255D}\u{2588}\u{2588}\u{2551}  \u{2588}\u{2588}\u{2551}",
                "\u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D} \u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D}\u{255A}\u{2550}\u{255D}  \u{255A}\u{2550}\u{255D}\u{255A}\u{2550}\u{255D}\u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D} \u{255A}\u{2550}\u{255D}  \u{255A}\u{2550}\u{255D}",
            ];
            let color = if frame <= 12 {
                Color::Indexed(235) // very dim
            } else if frame <= 18 {
                Color::Indexed(237) // dim
            } else {
                Color::Indexed(240) // medium
            };
            let logo_width: u16 = 43;
            let logo_height: u16 = logo_lines.len() as u16;
            let cx = area.x + area.width.saturating_sub(logo_width) / 2;
            let cy = area.y + area.height.saturating_sub(logo_height) / 2;
            let style = Style::default().fg(color);
            for (i, line) in logo_lines.iter().enumerate() {
                let line_area = Rect { x: cx, y: cy + i as u16, width: logo_width, height: 1 };
                f.render_widget(Paragraph::new(*line).style(style), line_area);
            }
            return;
        }

        // Full-screen borderless layout: header + body + dots + footer
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // Header
                Constraint::Min(6),    // Body
                Constraint::Length(1),  // Step dots
                Constraint::Length(2),  // Footer
            ])
            .split(area);

        // ── Header ──────────────────────────────────────────────────
        let header_title = match ob.step {
            OnboardingStep::Entrance | OnboardingStep::Intro => "Welcome",
            OnboardingStep::ModeSelection => "Setup",
            OnboardingStep::ProviderSelection => "Setup \u{00B7} Provider",
            OnboardingStep::ApiKeyEntry | OnboardingStep::ApiKeyValidation => "Setup \u{00B7} API Key",
            OnboardingStep::SystemCheck => "Setup \u{00B7} System Check",
            OnboardingStep::ModelSetup => "Setup \u{00B7} Models",
            OnboardingStep::AskName | OnboardingStep::AskRole => "Setup \u{00B7} About You",
            OnboardingStep::Processing => "Setup \u{00B7} Processing",
            OnboardingStep::Confirmation => "Setup \u{00B7} Confirm",
            OnboardingStep::Done => "Ready",
        };
        let header_line = Line::from(vec![
            Span::raw("  "),
            Span::styled(header_title, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]);
        f.render_widget(Paragraph::new(header_line), Rect { x: chunks[0].x, y: chunks[0].y, width: chunks[0].width, height: 1 });
        let sep = "\u{2500}".repeat(chunks[0].width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: chunks[0].x, y: chunks[0].y + 1, width: chunks[0].width, height: 1 },
        );

        // ── Body ────────────────────────────────────────────────────
        let body = chunks[1];
        let visible = ob.visible_text();

        // Build body content lines
        let mut lines: Vec<Line> = Vec::new();

        // Braille spinner for processing with task running
        let is_processing_active = ob.step == OnboardingStep::Processing && ob.processing_task.is_some();
        let is_validating = ob.step == OnboardingStep::ApiKeyValidation && ob.validation_task.is_some();
        let is_done = ob.step == OnboardingStep::Done;

        // Step-specific rendering
        if ob.step == OnboardingStep::ModeSelection {
            // Line selector for mode
            for text_line in visible.split('\n') {
                lines.push(Line::from(Span::styled(text_line, Style::default().fg(Color::White))));
            }
            lines.push(Line::from(""));

            let sel = ob.selected_mode;
            let sel_bg = Color::Indexed(236);

            let mode_blocks: [(&str, &[&str]); 2] = [
                ("Cloud Provider", &["Uses Anthropic, OpenAI or Google.", "Best quality. Needs an API key."]),
                ("Privacy Mode (local)", &["Runs entirely on your machine.", "Fully private \u{2014} no data leaves", "your computer."]),
            ];
            // Find max visible width across all blocks for uniform padding
            let block_width = mode_blocks.iter().flat_map(|(title, descs)| {
                std::iter::once(title.chars().count() + 2) // "▸ " prefix
                    .chain(descs.iter().map(|d| d.chars().count() + 2)) // "  " prefix
            }).max().unwrap_or(0);

            for (i, (title, descs)) in mode_blocks.iter().enumerate() {
                if i > 0 { lines.push(Line::from("")); }
                if sel == i {
                    let title_text = format!("{}{}", title, " ".repeat(block_width.saturating_sub(title.chars().count() + 2)));
                    lines.push(Line::from(vec![
                        Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                        Span::styled(title_text, Style::default().fg(Color::White).bg(sel_bg)),
                    ]));
                    let ds = Style::default().fg(Color::Indexed(245)).bg(sel_bg);
                    for d in *descs {
                        let padded = format!("  {}{}", d, " ".repeat(block_width.saturating_sub(d.chars().count() + 2)));
                        lines.push(Line::from(Span::styled(padded, ds)));
                    }
                } else {
                    lines.push(Line::from(Span::styled(format!("  {}", title), Style::default().fg(Color::DarkGray))));
                    let ds = Style::default().fg(Color::DarkGray);
                    for d in *descs {
                        lines.push(Line::from(Span::styled(format!("  {}", d), ds)));
                    }
                }
            }
        } else if ob.step == OnboardingStep::ProviderSelection {
            // Line selector for provider
            for text_line in visible.split('\n') {
                lines.push(Line::from(Span::styled(text_line, Style::default().fg(Color::White))));
            }
            lines.push(Line::from(""));

            let providers = [
                ("Anthropic (Claude)", "Best for nuanced understanding"),
                ("OpenAI (GPT)", "Widely used, reliable"),
                ("Google (Gemini)", "Fast and cost-effective"),
            ];
            let sel_bg = Color::Indexed(236);
            // Find max width across all provider entries for uniform padding
            let block_width = providers.iter().flat_map(|(name, desc)| {
                std::iter::once(name.chars().count() + 2) // "▸ " prefix
                    .chain(std::iter::once(desc.chars().count() + 2)) // "  " prefix
            }).max().unwrap_or(0);

            for (i, (name, desc)) in providers.iter().enumerate() {
                if i > 0 {
                    lines.push(Line::from(""));
                }
                if ob.selected_provider == i {
                    let name_pad = " ".repeat(block_width.saturating_sub(name.chars().count() + 2));
                    lines.push(Line::from(vec![
                        Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                        Span::styled(format!("{}{}", name, name_pad), Style::default().fg(Color::White).bg(sel_bg)),
                    ]));
                    let desc_pad = " ".repeat(block_width.saturating_sub(desc.chars().count() + 2));
                    lines.push(Line::from(Span::styled(format!("  {}{}", desc, desc_pad), Style::default().fg(Color::Indexed(245)).bg(sel_bg))));
                } else {
                    lines.push(Line::from(Span::styled(format!("  {}", name), Style::default().fg(Color::DarkGray))));
                    lines.push(Line::from(Span::styled(format!("  {}", desc), Style::default().fg(Color::DarkGray))));
                }
            }
        } else if ob.step == OnboardingStep::SystemCheck {
            // System check checklist with spinners/results
            for text_line in visible.split('\n') {
                lines.push(Line::from(Span::styled(text_line, Style::default().fg(Color::White))));
            }
            lines.push(Line::from(""));

            let braille = ['\u{28F7}', '\u{28EF}', '\u{28DF}', '\u{28BF}', '\u{287F}', '\u{28FE}', '\u{28FD}', '\u{28FB}'];
            for (name, status) in &ob.system_checks {
                let (icon, style) = match status {
                    CheckStatus::Pending => ("  ".to_string(), Style::default().fg(Color::DarkGray)),
                    CheckStatus::Running => (
                        format!("{} ", braille[ob.anim_frame % braille.len()]),
                        Style::default().fg(ACCENT),
                    ),
                    CheckStatus::Passed => ("\u{2713} ".to_string(), Style::default().fg(Color::Green)),
                    CheckStatus::Failed(_) => ("\u{2717} ".to_string(), Style::default().fg(Color::Red)),
                };
                lines.push(Line::from(vec![
                    Span::styled(icon, style),
                    Span::styled(name.as_str(), Style::default().fg(Color::White)),
                ]));
            }

            // If done with failures, show instructions for the first failing check
            if ob.system_check_done {
                let first_fail = ob.system_checks.iter().find_map(|(name, s)| {
                    if let CheckStatus::Failed(hint) = s {
                        Some((name.clone(), hint.clone()))
                    } else {
                        None
                    }
                });

                if let Some((_name, hint)) = first_fail {
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        "Almost there!",
                        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                    )));
                    lines.push(Line::from(""));
                    for hint_line in hint.split('\n') {
                        lines.push(Line::from(Span::styled(
                            format!("  {}", hint_line),
                            Style::default().fg(Color::Indexed(245)),
                        )));
                    }
                    lines.push(Line::from(""));

                    // Arrow selector: Check again / Continue anyway
                    let sel_bg = Color::Indexed(236);
                    let options = ["Check again", "Continue anyway"];
                    let opt_width = options.iter().map(|l| l.chars().count() + 2).max().unwrap_or(0);
                    for (i, label) in options.iter().enumerate() {
                        if ob.system_check_selection == i {
                            let pad = " ".repeat(opt_width.saturating_sub(label.chars().count() + 2));
                            lines.push(Line::from(vec![
                                Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                                Span::styled(format!("{}{}", label, pad), Style::default().fg(Color::White).bg(sel_bg)),
                            ]));
                        } else {
                            lines.push(Line::from(Span::styled(format!("  {}", label), Style::default().fg(Color::DarkGray))));
                        }
                    }
                }
            }
        } else if ob.step == OnboardingStep::ModelSetup {
            // Phase-dependent rendering
            for text_line in visible.split('\n') {
                lines.push(Line::from(Span::styled(text_line, Style::default().fg(Color::White))));
            }
            lines.push(Line::from(""));

            let sel_bg = Color::Indexed(236);

            match ob.setup_phase {
                0 => {
                    // Whisper model selector
                    let block_width = WHISPER_MODELS.iter()
                        .map(|(_, name, size)| name.chars().count() + size.chars().count() + 6)
                        .max().unwrap_or(0);

                    for (i, (_, name, size)) in WHISPER_MODELS.iter().enumerate() {
                        let label = format!("{}    {}", name, size);
                        if ob.whisper_model_selection == i {
                            let pad = " ".repeat(block_width.saturating_sub(label.chars().count() + 2));
                            lines.push(Line::from(vec![
                                Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                                Span::styled(format!("{}{}", label, pad), Style::default().fg(Color::White).bg(sel_bg)),
                            ]));
                        } else {
                            lines.push(Line::from(Span::styled(format!("  {}", label), Style::default().fg(Color::DarkGray))));
                        }
                    }
                }
                1 => {
                    // Ollama model selector
                    if ob.ollama_available_models.is_empty() {
                        let braille = ['\u{28F7}', '\u{28EF}', '\u{28DF}', '\u{28BF}', '\u{287F}', '\u{28FE}', '\u{28FD}', '\u{28FB}'];
                        lines.push(Line::from(vec![
                            Span::styled(format!("{} ", braille[ob.anim_frame % braille.len()]), Style::default().fg(ACCENT)),
                            Span::styled("Loading models...", Style::default().fg(Color::DarkGray)),
                        ]));
                    } else {
                        for (i, model_name) in ob.ollama_available_models.iter().enumerate() {
                            let display = model_name.clone();
                            if ob.ollama_model_selection == i {
                                let pad = " ".repeat(30usize.saturating_sub(display.chars().count() + 2));
                                lines.push(Line::from(vec![
                                    Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                                    Span::styled(format!("{}{}", display, pad), Style::default().fg(Color::White).bg(sel_bg)),
                                ]));
                            } else {
                                lines.push(Line::from(Span::styled(format!("  {}", display), Style::default().fg(Color::DarkGray))));
                            }
                        }
                    }
                }
                2 => {
                    // Download confirmation
                    let whisper_name = WHISPER_MODELS[ob.whisper_model_selection].1
                        .replace(" (Recommended)", "");
                    let whisper_size = WHISPER_MODELS[ob.whisper_model_selection].2;
                    lines.push(Line::from(vec![
                        Span::styled("  Whisper ", Style::default().fg(Color::DarkGray)),
                        Span::styled(whisper_name, Style::default().fg(Color::White)),
                        Span::styled(format!("    {}", whisper_size), Style::default().fg(Color::Indexed(245))),
                    ]));

                    if ob.ollama_reachable {
                        let model = if let EnrichmentMode::Local { ollama_model, .. } = &self.config.enrichment.mode {
                            ollama_model.split(':').next().unwrap_or(ollama_model).to_string()
                        } else {
                            "mistral".to_string()
                        };
                        let model_line = Line::from(vec![
                            Span::styled("  ", Style::default().fg(Color::DarkGray)),
                            Span::styled(model, Style::default().fg(Color::White)),
                            Span::styled("    pull from Ollama", Style::default().fg(Color::Indexed(245))),
                        ]);
                        lines.push(model_line);
                    }
                    lines.push(Line::from(""));

                    let options = ["Download now", "Skip for now"];
                    let opt_width = options.iter().map(|l| l.chars().count() + 2).max().unwrap_or(0);
                    for (i, label) in options.iter().enumerate() {
                        if ob.system_check_selection == i {
                            let pad = " ".repeat(opt_width.saturating_sub(label.chars().count() + 2));
                            lines.push(Line::from(vec![
                                Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                                Span::styled(format!("{}{}", label, pad), Style::default().fg(Color::White).bg(sel_bg)),
                            ]));
                        } else {
                            lines.push(Line::from(Span::styled(format!("  {}", label), Style::default().fg(Color::DarkGray))));
                        }
                    }
                }
                3 => {
                    // Downloading with progress
                    let braille = ['\u{28F7}', '\u{28EF}', '\u{28DF}', '\u{28BF}', '\u{287F}', '\u{28FE}', '\u{28FD}', '\u{28FB}'];
                    for (name, status) in &ob.download_items {
                        let (icon, detail) = match status {
                            DownloadStatus::Pending => (
                                "  ".to_string(),
                                String::new(),
                            ),
                            DownloadStatus::InProgress(pct) => (
                                format!("{} ", braille[ob.anim_frame % braille.len()]),
                                format!(" {}%", pct),
                            ),
                            DownloadStatus::Done => (
                                "\u{2713} ".to_string(),
                                String::new(),
                            ),
                            DownloadStatus::Failed(msg) => (
                                "\u{2717} ".to_string(),
                                format!(" — {}", msg),
                            ),
                        };

                        let icon_style = match status {
                            DownloadStatus::Done => Style::default().fg(Color::Green),
                            DownloadStatus::Failed(_) => Style::default().fg(Color::Red),
                            DownloadStatus::InProgress(_) => Style::default().fg(ACCENT),
                            _ => Style::default().fg(Color::DarkGray),
                        };

                        lines.push(Line::from(vec![
                            Span::styled(icon, icon_style),
                            Span::styled(name.as_str(), Style::default().fg(Color::White)),
                            Span::styled(detail, Style::default().fg(Color::Indexed(245))),
                        ]));
                    }

                    // Show "Continue" or "Retry" when done
                    let all_done = ob.download_items.iter().all(|(_, s)| matches!(s, DownloadStatus::Done));
                    let any_failed = ob.download_items.iter().any(|(_, s)| matches!(s, DownloadStatus::Failed(_)));
                    if all_done || any_failed {
                        lines.push(Line::from(""));
                        let label = if all_done { "Continue" } else { "Continue anyway" };
                        lines.push(Line::from(Span::styled(
                            format!("  [Enter] {}", label),
                            Style::default().fg(Color::DarkGray),
                        )));
                    }
                }
                _ => {}
            }
        } else if ob.step == OnboardingStep::ApiKeyValidation && ob.api_key_valid == Some(false) {
            // Validation failed — arrow selector for retry/skip
            for text_line in visible.split('\n') {
                lines.push(Line::from(Span::styled(text_line, Style::default().fg(Color::White))));
            }
            lines.push(Line::from(""));

            let sel_bg = Color::Indexed(236);
            let fail_labels = ["Try a different key", "Skip for now (set it later)"];
            let fail_width = fail_labels.iter().map(|l| l.chars().count() + 2).max().unwrap_or(0);
            for (i, label) in fail_labels.iter().enumerate() {
                if ob.validation_fail_selection == i {
                    let pad = " ".repeat(fail_width.saturating_sub(label.chars().count() + 2));
                    lines.push(Line::from(vec![
                        Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                        Span::styled(format!("{}{}", label, pad), Style::default().fg(Color::White).bg(sel_bg)),
                    ]));
                } else {
                    lines.push(Line::from(Span::styled(format!("  {}", label), Style::default().fg(Color::DarkGray))));
                }
            }
        } else if ob.step == OnboardingStep::Confirmation {
            // Structured label/value layout
            for text_line in visible.split('\n') {
                lines.push(Line::from(Span::styled(text_line, Style::default().fg(Color::White))));
            }
            lines.push(Line::from(""));

            let label_style = Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD);
            let value_style = Style::default().fg(Color::White);
            let fields: [(&str, &str); 4] = [
                ("Name   ", &ob.confirm_owner),
                ("Role   ", &ob.confirm_role),
                ("Org    ", &ob.confirm_org),
                ("Known  ", &ob.confirm_people),
            ];
            for (label, value) in &fields {
                lines.push(Line::from(vec![
                    Span::styled(format!("  {}", label), label_style),
                    Span::styled(*value, value_style),
                ]));
            }

            lines.push(Line::from(""));

            // Line selector: Looks good / Let me fix that
            let sel_bg = Color::Indexed(236);
            let confirm_labels = ["Looks good", "Let me fix that"];
            let confirm_width = confirm_labels.iter().map(|l| l.chars().count() + 2).max().unwrap_or(0);
            for (i, label) in confirm_labels.iter().enumerate() {
                if ob.confirm_selection == i {
                    let pad = " ".repeat(confirm_width.saturating_sub(label.chars().count() + 2));
                    lines.push(Line::from(vec![
                        Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                        Span::styled(format!("{}{}", label, pad), Style::default().fg(Color::White).bg(sel_bg)),
                    ]));
                } else {
                    lines.push(Line::from(Span::styled(format!("  {}", label), Style::default().fg(Color::DarkGray))));
                }
            }
        } else {
            // Default text rendering for all other steps
            for text_line in visible.split('\n') {
                if is_processing_active || is_validating {
                    let braille = ['\u{28F7}', '\u{28EF}', '\u{28DF}', '\u{28BF}', '\u{287F}', '\u{28FE}', '\u{28FD}', '\u{28FB}'];
                    let spinner = braille[ob.anim_frame % braille.len()];
                    lines.push(Line::from(vec![
                        Span::styled(format!("{} ", spinner), Style::default().fg(ACCENT)),
                        Span::styled(text_line, Style::default().fg(Color::White)),
                    ]));
                } else if is_done {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2713} ", Style::default().fg(ACCENT)),
                        Span::styled(text_line, Style::default().fg(Color::White)),
                    ]));
                } else {
                    lines.push(Line::from(Span::styled(text_line, Style::default().fg(Color::White))));
                }
            }

            // Input field for input steps (only after text reveal is complete)
            let show_input = ob.text_complete && matches!(
                ob.step,
                OnboardingStep::ApiKeyEntry | OnboardingStep::AskName | OnboardingStep::AskRole
            );

            if show_input {
                let input_value = match ob.step {
                    OnboardingStep::ApiKeyEntry => &ob.api_key_input,
                    OnboardingStep::AskName => &ob.user_name,
                    OnboardingStep::AskRole => &ob.user_role,
                    _ => &ob.user_name,
                };

                // Mask API key input
                let display_value = if ob.step == OnboardingStep::ApiKeyEntry && !input_value.is_empty() {
                    let vis = input_value.chars().take(4).collect::<String>();
                    let hidden = "*".repeat(input_value.len().saturating_sub(4));
                    format!("{}{}", vis, hidden)
                } else {
                    input_value.to_string()
                };

                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled("\u{25B8} ", Style::default().fg(ACCENT)),
                    Span::styled(format!("{}_", display_value), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
                ]));
            }
        }

        // Vertical centering: prepend empty lines
        let line_count = lines.len();
        let top_pad = (body.height as usize).saturating_sub(line_count) / 2;
        let mut centered_lines: Vec<Line> = Vec::with_capacity(top_pad + line_count);
        for _ in 0..top_pad {
            centered_lines.push(Line::from(""));
        }
        centered_lines.extend(lines);

        let max_line_width = centered_lines.iter().map(|l| l.width() as u16).max().unwrap_or(0);
        let content_width = max_line_width.max(30).min(body.width); // floor 30, cap at body
        let left_pad = (body.width.saturating_sub(content_width)) / 2;
        let centered_body = Rect {
            x: body.x + left_pad,
            width: content_width,
            ..body
        };
        let body_para = Paragraph::new(centered_lines)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false });
        f.render_widget(body_para, centered_body);

        // ── Step dots ───────────────────────────────────────────────
        // Both privacy and cloud flows have 9 steps:
        // Privacy: Intro(0) → Mode(1) → SystemCheck(2) → ModelSetup(3) → Name(4) → Role(5) → Processing(6) → Confirm(7) → Done(8)
        // Cloud:   Intro(0) → Mode(1) → Provider(2) → ApiKey(3) → Name(4) → Role(5) → Processing(6) → Confirm(7) → Done(8)
        let step_count = 9;
        let current_idx = match ob.step {
            OnboardingStep::Entrance | OnboardingStep::Intro => 0,
            OnboardingStep::ModeSelection => 1,
            OnboardingStep::ProviderSelection | OnboardingStep::SystemCheck => 2,
            OnboardingStep::ApiKeyEntry | OnboardingStep::ApiKeyValidation | OnboardingStep::ModelSetup => 3,
            OnboardingStep::AskName => 4,
            OnboardingStep::AskRole => 5,
            OnboardingStep::Processing => 6,
            OnboardingStep::Confirmation => 7,
            OnboardingStep::Done => 8,
        };

        let mut dots: Vec<Span> = Vec::new();
        for i in 0..step_count {
            let (ch, color) = if i == current_idx {
                ("\u{25CF}", ACCENT)
            } else if i < current_idx {
                ("\u{25CF}", Color::DarkGray)
            } else {
                ("\u{25CB}", Color::Indexed(237))
            };
            dots.push(Span::styled(format!(" {} ", ch), Style::default().fg(color)));
        }
        let dots_line = Paragraph::new(Line::from(dots)).alignment(Alignment::Center);
        f.render_widget(dots_line, chunks[2]);

        // ── Footer ──────────────────────────────────────────────────
        let footer_area = chunks[3];
        let footer_sep = "\u{2500}".repeat(footer_area.width as usize);
        f.render_widget(
            Paragraph::new(footer_sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: footer_area.x, y: footer_area.y, width: footer_area.width, height: 1 },
        );

        let footer_content = if footer_area.height > 1 {
            Rect { x: footer_area.x, y: footer_area.y + 1, width: footer_area.width, height: footer_area.height - 1 }
        } else {
            footer_area
        };

        let version = env!("CARGO_PKG_VERSION");
        let left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(format!("scriba \u{00B7} v{}", version), Style::default().fg(Color::DarkGray)),
        ];

        let right_hint = match ob.step {
            OnboardingStep::Intro => "[Enter] Continue",
            OnboardingStep::ModeSelection | OnboardingStep::ProviderSelection
            | OnboardingStep::Confirmation => "[Up/Down] Select  [Enter] Confirm",
            OnboardingStep::ApiKeyEntry => "[Enter] Validate",
            OnboardingStep::ApiKeyValidation => {
                if ob.validation_task.is_some() { "Validating..." }
                else if ob.api_key_valid == Some(false) { "[Up/Down] Select  [Enter] Confirm" }
                else { "" }
            }
            OnboardingStep::SystemCheck => {
                if ob.system_check_done { "[Up/Down] Select  [Enter] Confirm" }
                else { "Checking..." }
            }
            OnboardingStep::ModelSetup => {
                match ob.setup_phase {
                    0 | 1 => "[Up/Down] Select  [Enter] Confirm",
                    2 => "[Up/Down] Select  [Enter] Confirm",
                    3 => {
                        let all_done = ob.download_items.iter().all(|(_, s)| matches!(s, DownloadStatus::Done));
                        let any_failed = ob.download_items.iter().any(|(_, s)| matches!(s, DownloadStatus::Failed(_)));
                        if all_done || any_failed { "[Enter] Continue" } else { "Downloading..." }
                    }
                    _ => "",
                }
            }
            OnboardingStep::AskName | OnboardingStep::AskRole => "[Enter] Continue",
            OnboardingStep::Processing => {
                if ob.processing_task.is_some() { "" }
                else if !ob.ollama_available { "[Enter] Continue" }
                else { "" }
            }
            OnboardingStep::Done => {
                if ob.text_complete && !ob.transitioning { "[Enter] Let's go" } else { "" }
            }
            _ => "",
        };

        let mut right_spans: Vec<Span> = Vec::new();
        if ob.step != OnboardingStep::Done {
            right_spans.extend([
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Esc", Style::default().fg(Color::White)),
                Span::styled("] Skip", Style::default().fg(Color::DarkGray)),
            ]);
        }
        if !right_hint.is_empty() {
            right_spans.push(Span::styled("  ", Style::default()));
            right_spans.push(Span::styled(right_hint, Style::default().fg(Color::DarkGray)));
        }
        right_spans.push(Span::raw(" "));

        let left_width: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_width: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_content.width as usize).saturating_sub(left_width + right_width);

        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);

        f.render_widget(Paragraph::new(Line::from(spans)), footer_content);
    }

    fn render_dim_fade(&self, f: &mut Frame, area: ratatui::layout::Rect, ob: &OnboardingState) {
        // 10-frame fade-out (~1s)
        let frame = ob.transition_frame.min(10);
        let color = match frame {
            0 => Color::Indexed(252),
            1 => Color::Indexed(249),
            2 => Color::Indexed(245),
            3 => Color::Indexed(242),
            4 => Color::Indexed(240),
            5 => Color::Indexed(238),
            6 => Color::Indexed(237),
            7 => Color::Indexed(236),
            8 => Color::Indexed(235),
            _ => return, // blank
        };

        // Use same layout as normal render to keep text in the body region
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // Header
                Constraint::Min(6),    // Body
                Constraint::Length(1),  // Step dots
                Constraint::Length(2),  // Footer
            ])
            .split(area);
        let body = chunks[1];

        // Re-render body text with dimming color
        let visible = ob.visible_text();
        let mut lines: Vec<Line> = Vec::new();
        let is_done = ob.step == OnboardingStep::Done;
        for text_line in visible.split('\n') {
            if is_done {
                lines.push(Line::from(vec![
                    Span::styled("\u{2713} ", Style::default().fg(color)),
                    Span::styled(text_line, Style::default().fg(color)),
                ]));
            } else {
                lines.push(Line::from(Span::styled(text_line, Style::default().fg(color))));
            }
        }

        // Vertical centering
        let line_count = lines.len();
        let top_pad = (body.height as usize).saturating_sub(line_count) / 2;
        let mut centered_lines: Vec<Line> = Vec::with_capacity(top_pad + line_count);
        for _ in 0..top_pad { centered_lines.push(Line::from("")); }
        centered_lines.extend(lines);

        // Horizontal centering
        let max_line_width = centered_lines.iter().map(|l| l.width() as u16).max().unwrap_or(0);
        let content_width = max_line_width.max(30).min(body.width);
        let left_pad = (body.width.saturating_sub(content_width)) / 2;
        let centered_body = Rect {
            x: body.x + left_pad,
            width: content_width,
            ..body
        };
        let p = Paragraph::new(centered_lines)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false });
        f.render_widget(p, centered_body);
    }
}

