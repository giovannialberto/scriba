use crate::core::{
    AudioPlayer, EnrichmentMode, RecordingResult, ScribaConfig, TranscriptionMode,
};
use crate::database::{Database, Entity, Recording, RecordingStats};
use crate::enrichment::{OllamaClient, WorldContext, WorldData};
use crate::enrichment::chat_prompts;
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
    widgets::{Block, Borders, Clear, Paragraph, TableState, Wrap},
    Frame, Terminal,
};
use std::collections::VecDeque;
use std::io;
use std::time::Duration;
use tokio::sync::mpsc;

use super::chat::{
    ChatContext, ChatFocus, ChatMessage, ChatRole, ChatState, ChatStreamEvent,
    chat_agent_pipeline, ACCENT,
};

use super::browse::FileDialogStage;
use super::entities::{EntityMode, EntityEditField};
use super::onboarding::{OnboardingState, OnboardingStep, OnboardingTickResult};
use super::recording::{RecordingMode, ActiveTranscription, PendingTranscription};
use super::settings::{ModelPickerState, ModelPickerItem};

pub struct Dashboard {
    pub(super) db: Database,
    pub(super) recordings: Vec<Recording>,
    pub(super) table_state: TableState,
    pub(super) current_page: usize,
    pub(super) page_size: usize,
    pub(super) stats: Option<RecordingStats>,
    pub(super) show_help: bool,
    pub(super) current_view: DashboardView,
    pub(super) search_mode: bool,
    pub(super) search_query: String,
    pub(super) show_message: bool,
    pub(super) message: String,
    pub(super) show_transcript: bool,
    pub(super) transcript_content: String,
    pub(super) show_delete_confirm: bool,
    pub(super) delete_confirm_selection: usize,  // 0 = Yes, 1 = No
    pub(super) delete_candidate: Option<Recording>,
    pub(super) audio_player: Option<AudioPlayer>, // Native rodio playback (replaces subprocess players)
    pub(super) last_transcribe_warning: Option<usize>, // Track which recording showed overwrite warning
    pub(super) progress_animation: Option<String>,     // Base message for progress animation
    pub(super) progress_frame: usize,                  // Animation frame counter
    pub(super) active_transcription: Option<ActiveTranscription>, // Currently running transcription
    pub(super) transcription_queue: VecDeque<PendingTranscription>, // FIFO queue of pending transcriptions
    pub(super) notification_message: Option<(String, usize)>, // (message, frames_remaining) -- auto-dismiss
    pub(super) recording_task: Option<tokio::task::JoinHandle<Result<RecordingResult, anyhow::Error>>>,
    pub(super) recording_mode: Option<RecordingMode>, // Track if we should transcribe after recording
    pub(super) recording_stop_tx: Option<mpsc::Sender<()>>, // Channel to stop recording
    pub(super) recording_level_rx: Option<mpsc::Receiver<f32>>, // Channel to receive volume levels
    pub(super) current_volume_level: f32,             // Current recording volume for display
    pub(super) recording_start_instant: Option<std::time::Instant>, // When recording started (for elapsed time)
    pub(super) volume_history: VecDeque<f32>,         // Recent volume samples for waveform display
    pub(super) config: ScribaConfig,                  // App configuration
    pub(super) settings_selection: usize,             // Current setting selection
    pub(super) editing_api_key: bool,                 // Whether we're editing API key
    pub(super) api_key_input: String,                 // API key input buffer
    pub(super) model_picker_state: ModelPickerState,
    pub(super) model_picker_items: Vec<ModelPickerItem>,
    pub(super) model_picker_selection: usize,
    pub(super) model_picker_custom_input: String,
    pub(super) ollama_models_rx: Option<mpsc::Receiver<Result<Vec<String>, String>>>,
    pub(super) editing_enrichment_endpoint: bool,     // Whether we're editing Ollama endpoint (local mode)
    pub(super) enrichment_endpoint_input: String,     // Ollama endpoint input buffer (local mode)
    pub(super) editing_enrichment_api_key: bool,      // Whether we're editing enrichment API key
    pub(super) enrichment_api_key_input: String,      // Enrichment API key input buffer
    pub(super) return_to_view: Option<DashboardView>, // View to return to after message dismissal
    // File import dialog state
    pub(super) show_file_dialog: bool,
    pub(super) file_path_input: String,
    pub(super) file_name_input: String,
    pub(super) file_dialog_stage: FileDialogStage, // Current stage of file import process
    pub(super) file_dialog_error: Option<String>,  // Inline error for current step

    // Entity view state
    pub(super) entities: Vec<Entity>,
    pub(super) entity_table_state: TableState,
    pub(super) selected_entity: Option<Entity>,
    pub(super) show_entity_detail: bool,
    pub(super) entity_mode: EntityMode,
    pub(super) entity_edit_field: EntityEditField,
    pub(super) entity_edit_name: String,
    pub(super) entity_edit_type: String,
    pub(super) entity_edit_context: String,
    pub(super) merge_source_entity: Option<Entity>,
    pub(super) confirm_selection: usize,  // shared for entity delete/merge confirms: 0 = Yes, 1 = No
    // Add entity state
    pub(super) entity_add_name: String,
    pub(super) entity_add_type: String,
    pub(super) entity_add_context: String,
    pub(super) entity_add_aliases: String,
    pub(super) entity_add_field: EntityEditField,
    // Transcript enrichment data
    pub(super) transcript_summary: Option<String>,
    pub(super) transcript_key_points: Option<Vec<String>>,
    pub(super) transcript_topics: Option<Vec<String>>,
    pub(super) transcript_entities: Option<Vec<(String, String)>>, // (name, type)

    // Onboarding state
    pub(super) onboarding: Option<OnboardingState>,

    // Chat state ("Ask Scriba")
    pub(super) chat: ChatState,
    pub(super) global_chat_messages: Vec<ChatMessage>,
    // Track the currently-viewed recording for chat context
    pub(super) current_transcript_recording: Option<Recording>,

    // Home screen greeting
    pub(super) greeting_text: String,
    pub(super) greeting_subtitle: String,
    pub(super) owner_name: String,

    // Easter egg
    pub(super) owl_easter_egg_frame: Option<usize>,

    // Update checker
    pub(super) update_available: Option<String>,                // Some("0.26.0") when newer version exists
    pub(super) update_check_rx: Option<mpsc::Receiver<Option<String>>>,
    pub(super) update_in_progress: bool,
    pub(super) update_task: Option<tokio::task::JoinHandle<Result<String, String>>>,
    pub(super) update_completed: Option<Result<String, String>>,  // Ok(version) or Err(message)
}

#[derive(Debug, PartialEq, Clone)]
pub(super) enum DashboardView {
    Main,
    Browse,
    Help,
    Settings,
    Entities,
    Onboarding,
}

#[derive(Debug)]
pub(super) enum DashboardAction {
    Continue,
    Quit,
    RecordAndTranscribe,
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
            audio_player: None,
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
            file_dialog_error: None,

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

            // Chat
            chat: ChatState::new(),
            global_chat_messages: Vec::new(),
            current_transcript_recording: None,

            // Greeting
            greeting_text: String::new(),
            greeting_subtitle: String::new(),
            owner_name: String::new(),

            // Easter egg
            owl_easter_egg_frame: None,

            // Update checker
            update_available: None,
            update_check_rx: None,
            update_in_progress: false,
            update_task: None,
            update_completed: None,
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

        // Spawn async update check (non-blocking, silent on failure)
        if self.config.check_for_updates {
            let (tx, rx) = mpsc::channel(1);
            self.update_check_rx = Some(rx);
            let current_version = env!("CARGO_PKG_VERSION").to_string();
            tokio::spawn(async move {
                let result = check_for_update(&current_version).await;
                let _ = tx.send(result).await;
            });
        }

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
            // -- Animation tick (throttled to ~100ms) --
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
                                // Dismiss recording modal -- transcription runs non-blocking
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
            if let Some(ref player) = self.audio_player {
                if !player.is_playing() {
                    self.audio_player = None;
                    self.show_message = false;
                    self.message.clear();
                }
            }

            // Check for Ollama model list completion
            if let Some(ref mut rx) = self.ollama_models_rx {
                if let Ok(result) = rx.try_recv() {
                    // Check if we're in onboarding ModelSetup phase 1 -- populate onboarding models
                    let in_onboarding_model_setup = self.onboarding.as_ref()
                        .map(|ob| ob.step == OnboardingStep::ModelSetup && ob.setup_phase == 1)
                        .unwrap_or(false);

                    if in_onboarding_model_setup {
                        let local_names: Vec<String> = match &result {
                            Ok(names) => names.clone(),
                            Err(_) => Vec::new(),
                        };
                        if let Some(ref mut ob) = self.onboarding {
                            use super::onboarding::{RECOMMENDED_OLLAMA_MODELS, OllamaModelOption};
                            let local_set: std::collections::HashSet<&str> = local_names.iter().map(|s| s.as_str()).collect();
                            // Build list: recommended models first, then extra local models
                            let mut options: Vec<OllamaModelOption> = RECOMMENDED_OLLAMA_MODELS.iter().map(|&(id, label, size)| {
                                OllamaModelOption {
                                    id: id.to_string(),
                                    label: label.to_string(),
                                    size: size.to_string(),
                                    installed: local_set.contains(id),
                                }
                            }).collect();
                            // Add locally installed models not in the recommended list
                            for name in &local_names {
                                if !RECOMMENDED_OLLAMA_MODELS.iter().any(|(id, _, _)| *id == name.as_str()) {
                                    options.push(OllamaModelOption {
                                        id: name.clone(),
                                        label: name.clone(),
                                        size: String::new(),
                                        installed: true,
                                    });
                                }
                            }
                            ob.ollama_available_models = options;
                            ob.ollama_model_selection = 0;
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
                                // Error or empty -- show only Custom...
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

            // Check for update version check completion
            if let Some(ref mut rx) = self.update_check_rx {
                if let Ok(result) = rx.try_recv() {
                    if let Some(version) = result {
                        self.update_available = Some(version);
                    }
                    self.update_check_rx = None;
                }
            }

            // Check for update task completion
            if let Some(ref task) = self.update_task {
                if task.is_finished() {
                    let completed = self.update_task.take().unwrap();
                    self.update_in_progress = false;
                    match completed.await {
                        Ok(Ok(version)) => {
                            self.update_available = None;
                            self.update_completed = Some(Ok(version));
                        }
                        Ok(Err(msg)) => {
                            self.update_completed = Some(Err(msg));
                        }
                        Err(_) => {
                            self.update_completed = Some(Err("Update task failed".to_string()));
                        }
                    }
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

                // Easter egg animation tick
                if let Some(ref mut frame) = self.owl_easter_egg_frame {
                    *frame += 1;
                    if *frame > 150 {
                        self.owl_easter_egg_frame = None;
                    }
                }
            }

            // Onboarding tick logic (runs every frame for typewriter + async polling)
            if let Some(ref mut ob) = self.onboarding {
                if let Some(result) = ob.tick(anim_tick).await {
                    match result {
                        OnboardingTickResult::Complete => {
                            self.onboarding = None;
                            self.current_view = DashboardView::Main;
                            self.load_entities().ok();
                            self.init_global_chat();
                        }
                        OnboardingTickResult::SaveWhisperKey(key) => {
                            self.config.transcription = TranscriptionMode::Api {
                                api_key: key,
                            };
                            let _ = self.config.save();
                        }
                        OnboardingTickResult::FetchOllamaModels => {
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
            }

            // Poll chat stream events (non-blocking -- every frame for smooth streaming)
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
            if self.audio_player.is_some() {
                if let Some(player) = self.audio_player.take() {
                    player.stop();
                }
                self.show_message = false;
                self.message.clear();
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

        // Dismiss easter egg on any keypress
        if self.owl_easter_egg_frame.is_some() {
            self.owl_easter_egg_frame = None;
            return Ok(DashboardAction::Continue);
        }

        // Onboarding key handling
        if self.current_view == DashboardView::Onboarding {
            return self.handle_onboarding_keys(key_code).await;
        }

        if self.show_file_dialog {
            return self.handle_file_dialog_keys(key_code).await;
        }

        // Update banner: u to update, Esc to dismiss
        if self.update_available.is_some() && !self.update_in_progress {
            match key_code {
                KeyCode::Char('u') | KeyCode::Char('U') => {
                    let version = self.update_available.as_ref().unwrap().clone();
                    self.update_in_progress = true;
                    self.update_task = Some(tokio::spawn(async move {
                        perform_update(&version).await
                    }));
                    return Ok(DashboardAction::Continue);
                }
                KeyCode::Esc => {
                    self.update_available = None;
                    return Ok(DashboardAction::Continue);
                }
                _ => {} // other keys fall through
            }
        }
        // Dismiss update completion message on any keypress
        if self.update_completed.is_some() {
            self.update_completed = None;
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

        // Global Ctrl+ shortcuts (work from any view)
        let ctrl = modifiers.contains(crossterm::event::KeyModifiers::CONTROL);
        if ctrl {
            match key_code {
                KeyCode::Char('s') => {
                    self.current_view = DashboardView::Settings;
                    self.settings_selection = 0;
                    return Ok(DashboardAction::Continue);
                }
                // Also match 'z' — on Italian (and other non-US QWERTY) keyboards,
                // crossterm reports physical key positions, and W/Z are swapped.
                KeyCode::Char('w') | KeyCode::Char('z') => {
                    self.load_entities()?;
                    self.current_view = DashboardView::Entities;
                    self.entity_table_state.select(Some(0));
                    return Ok(DashboardAction::Continue);
                }
                KeyCode::Char('r') => {
                    return Ok(DashboardAction::RecordAndTranscribe);
                }
                KeyCode::Char('o') => {
                    self.owl_easter_egg_frame = Some(0);
                    return Ok(DashboardAction::Continue);
                }
                _ => {}
            }
        }

        // Browse view keys
        if self.current_view == DashboardView::Browse {
            return self.handle_browse_keys(key_code).await;
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

    async fn handle_dashboard_action(&mut self, action: DashboardAction) -> Result<()> {
        match action {
            DashboardAction::RecordAndTranscribe => {
                self.execute_record_and_transcribe().await?;
            }
            DashboardAction::TranscribeSelected => {
                self.execute_transcribe_selected().await?;
            }
            _ => {}
        }
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

        // Easter egg overlay (renders on top of everything)
        if let Some(frame) = self.owl_easter_egg_frame {
            self.render_owl_easter_egg(f, f.size(), frame);
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

    fn render_home_footer(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        // Align footer with chat box: symmetric 2-char margin each side
        let aligned = Rect {
            x: area.x + 2,
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

        // Update completion message
        if let Some(ref result) = self.update_completed {
            let (msg, style) = match result {
                Ok(v) => (
                    format!("Updated to v{} \u{2014} restart Scriba to use it", v),
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                ),
                Err(e) => (
                    format!("Update failed: {}", e),
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
            };
            let para = Paragraph::new(msg).style(style).alignment(Alignment::Center);
            f.render_widget(para, aligned);
            return;
        }

        // Update available banner
        if let Some(ref version) = self.update_available {
            if self.update_in_progress {
                let braille = ['\u{28F7}', '\u{28EF}', '\u{28DF}', '\u{28BF}', '\u{287F}', '\u{28FE}', '\u{28FD}', '\u{28FB}'];
                let frame = self.progress_frame;
                let line = Line::from(vec![
                    Span::styled(
                        format!("{} ", braille[frame % braille.len()]),
                        Style::default().fg(ACCENT),
                    ),
                    Span::styled(
                        format!("Updating to v{}...", version),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]);
                f.render_widget(Paragraph::new(line).alignment(Alignment::Center), aligned);
            } else {
                let line = Line::from(vec![
                    Span::styled(
                        format!("v{} available ", version),
                        Style::default().fg(ACCENT),
                    ),
                    Span::styled("[", Style::default().fg(Color::DarkGray)),
                    Span::styled("u", Style::default().fg(Color::White)),
                    Span::styled("] Update  ", Style::default().fg(Color::DarkGray)),
                    Span::styled("[", Style::default().fg(Color::DarkGray)),
                    Span::styled("Esc", Style::default().fg(Color::White)),
                    Span::styled("] Dismiss", Style::default().fg(Color::DarkGray)),
                ]);
                f.render_widget(Paragraph::new(line).alignment(Alignment::Center), aligned);
            }
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

        // Left side: > scriba . v0.21.2
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
            Line::from("  \u{2191}/\u{2193}        - Navigate recordings (auto-pages)"),
            Line::from("  Enter      - View transcript"),
            Line::from("  T          - Transcribe selected"),
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

    // -------------------------------------------------------------------------
    // Ask Scriba: Chat Implementation
    // -------------------------------------------------------------------------

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

    pub(super) fn generate_greeting(&mut self) {
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

    pub(super) fn init_recording_chat(&mut self, recording: &Recording) {
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

    pub(super) fn restore_global_chat(&mut self) {
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
                // View transcript -- find the recording and open transcript popup
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
                    // Summarize -- send immediately
                    self.chat.input_buffer = format!("Summarize my {} recording (recording id={}).", rec.name, rec.recording_id);
                    self.send_chat_message();
                } else {
                    // Ask about it -- fill input, let user edit
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
                    // Main -> Browse
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
                        // On home screen with nothing to dismiss -- do nothing
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

    // -- Mouse Handling --

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
                        // Dragging near top -- scroll up
                        if self.chat.auto_scroll {
                            self.chat.scroll_offset = self.chat.total_content_lines;
                        }
                        self.chat.auto_scroll = false;
                        self.chat.scroll_offset = self.chat.scroll_offset.saturating_sub(2);
                    } else if mouse.row >= bottom_edge.saturating_sub(edge_zone) && mouse.row < rect.y + rect.height {
                        // Dragging near bottom -- scroll down
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
                    // Keep selection visible (don't clear anchor/end) -- cleared on next click
                } else {
                    // Single click (no drag) -- clear any existing selection
                    self.chat.selection_anchor = None;
                    self.chat.selection_end = None;
                }
            }
            _ => {}
        }
    }

    fn render_owl_easter_egg(&self, f: &mut Frame, area: ratatui::layout::Rect, frame: usize) {
        // The owl is back, and it's UNHINGED. Full-screen takeover.
        let owl_sprites: &[&[&str]] = &[
            // Frame set 0: eyes wide, wings spread
            &[
                r"                                          ",
                r"        ,_,      ,_,                      ",
                r"       (O,O)    (O,O)     HOOT HOOT!!     ",
                r"      /{   }\  /{   }\                    ",
                r#"       -"-"-    -"-"-     I NEVER LEFT!!  "#,
            ],
            // Frame set 1: crazy eyes, wings up
            &[
                r"                                          ",
                r"       \(@,@)/ \(@,@)/                    ",
                r"        /{_}\   /{_}\   YOU THOUGHT YOU   ",
                r#"        -"-"-   -"-"-   COULD GET RID     "#,
                r"                        OF ME?! HOOOO!!   ",
            ],
            // Frame set 2: spinning
            &[
                r"                                          ",
                r"          /(O,O)\     /(O,O)\             ",
                r"           |\_/|       |\_/|              ",
                r#"           -"-"-       -"-"-              "#,
                r"        *AGGRESSIVELY HOOTING*            ",
            ],
            // Frame set 3: maximum chaos
            &[
                r"                                          ",
                r"    \(O,O)/  (o,o)  \(@,@)/  /(O,O)\     ",
                r#"     |][|   {{`"'}}    |><|    |}{|       "#,
                r#"     -"-"-  -"-"-   -"-"-   -"-"-        "#,
                r"   THE OWLS ARE NOT WHAT THEY SEEM!!!     ",
            ],
            // Frame set 4: single menacing owl
            &[
                r"                                          ",
                r"              ,___,                        ",
                r"             (O   O)   hoo hoo hoo...     ",
                r"              /)_(\                        ",
                r"             ' - - `   i see everything   ",
                r"                                          ",
                r"          ...hoo remembers everything...  ",
            ],
        ];

        let sprite_idx = (frame / 30) % owl_sprites.len();
        let sprite = owl_sprites[sprite_idx];

        // Flash background between colors for extra chaos
        let bg = match (frame / 8) % 4 {
            0 => Color::Black,
            1 => Color::Indexed(17),  // dark blue
            2 => Color::Black,
            _ => Color::Indexed(52),  // dark red
        };
        let fg = match (frame / 5) % 5 {
            0 => Color::Yellow,
            1 => Color::Magenta,
            2 => Color::Cyan,
            3 => Color::Red,
            _ => Color::Green,
        };

        // Fill background
        f.render_widget(Clear, area);
        let bg_fill = " ".repeat(area.width as usize);
        for y in area.y..area.y + area.height {
            f.render_widget(
                Paragraph::new(bg_fill.clone()).style(Style::default().bg(bg)),
                Rect { x: area.x, y, width: area.width, height: 1 },
            );
        }

        // Center and render the owl
        let sprite_height = sprite.len() as u16;
        let sprite_width = sprite.iter().map(|l| l.len()).max().unwrap_or(0) as u16;
        let cx = area.x + area.width.saturating_sub(sprite_width) / 2;
        let cy = area.y + area.height.saturating_sub(sprite_height) / 2;

        for (i, line) in sprite.iter().enumerate() {
            let y = cy + i as u16;
            if y < area.y + area.height {
                f.render_widget(
                    Paragraph::new(*line).style(Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD)),
                    Rect { x: cx, y, width: sprite_width.min(area.width), height: 1 },
                );
            }
        }

        // Countdown bar at bottom
        let remaining = 150_usize.saturating_sub(frame);
        let bar_width = (remaining as u16 * area.width) / 150;
        if bar_width > 0 {
            let bar = "\u{2588}".repeat(bar_width as usize);
            f.render_widget(
                Paragraph::new(bar).style(Style::default().fg(Color::DarkGray)),
                Rect { x: area.x, y: area.y + area.height - 1, width: area.width, height: 1 },
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Update checker
// ─────────────────────────────────────────────────────────────────────────────

/// Check GitHub releases for a newer version. Returns `Some("X.Y.Z")` if available.
async fn check_for_update(current_version: &str) -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .ok()?;
    let resp = client
        .get("https://api.github.com/repos/giovannialberto/scriba/releases/latest")
        .header("User-Agent", "scriba")
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let json: serde_json::Value = resp.json().await.ok()?;
    let tag = json["tag_name"].as_str()?;
    let latest = tag.strip_prefix('v').unwrap_or(tag);
    if version_newer(latest, current_version) {
        Some(latest.to_string())
    } else {
        None
    }
}

/// Simple semver comparison: is `a` newer than `b`?
fn version_newer(a: &str, b: &str) -> bool {
    let parse = |v: &str| -> Vec<u32> {
        v.split('.').filter_map(|s| s.parse().ok()).collect()
    };
    let va = parse(a);
    let vb = parse(b);
    va > vb
}

/// Perform the actual update. Returns the new version string on success.
async fn perform_update(version: &str) -> Result<String, String> {
    let version_tag = if version.starts_with('v') {
        version.to_string()
    } else {
        format!("v{}", version)
    };

    // Check if installed via Homebrew
    if cfg!(target_os = "macos") {
        let brew_check = tokio::process::Command::new("brew")
            .args(["list", "scriba"])
            .output()
            .await;
        if let Ok(output) = brew_check {
            if output.status.success() {
                // Update via Homebrew
                let result = tokio::process::Command::new("brew")
                    .args(["upgrade", "scriba"])
                    .output()
                    .await
                    .map_err(|e| format!("brew upgrade failed: {}", e))?;
                if result.status.success() {
                    return Ok(version.to_string());
                }
                return Err(String::from_utf8_lossy(&result.stderr).to_string());
            }
        }
    }

    // Direct binary replacement (macOS non-Homebrew + Linux)
    let target = if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            "aarch64-apple-darwin"
        } else {
            "x86_64-apple-darwin"
        }
    } else if cfg!(target_os = "linux") {
        "x86_64-unknown-linux-gnu"
    } else {
        return Err("Unsupported platform".to_string());
    };

    let url = format!(
        "https://github.com/giovannialberto/scriba/releases/download/{}/scriba-{}",
        version_tag, target
    );

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?;

    let resp = client
        .get(&url)
        .header("User-Agent", "scriba")
        .send()
        .await
        .map_err(|e| format!("Download failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Download failed: HTTP {}", resp.status()));
    }

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Download failed: {}", e))?;

    // Write to temp file, then replace current binary
    let current_exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let tmp_path = current_exe.with_extension("update");

    tokio::fs::write(&tmp_path, &bytes)
        .await
        .map_err(|e| format!("Failed to write update: {}", e))?;

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&tmp_path, perms)
            .map_err(|e| format!("Failed to set permissions: {}", e))?;
    }

    // Atomic replace
    std::fs::rename(&tmp_path, &current_exe)
        .map_err(|e| format!("Failed to replace binary: {}", e))?;

    Ok(version.to_string())
}
