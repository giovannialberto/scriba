use crate::core::{
    CloudProvider, EnrichmentMode, LocalModelSize,
    TranscriptionMode, initialize_world_from_seed,
};
use crate::database::Database;
use crate::enrichment::{OllamaClient, WorldContext, WorldData, WorldEntityExtractionResult};
use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
    Frame,
};
use tokio::sync::mpsc;

use super::chat::ACCENT;
use super::app::{Dashboard, DashboardAction, DashboardView};

// ─────────────────────────────────────────────────────────────────────────────
// Onboarding
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
pub(super) enum OnboardingStep {
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
pub(super) enum CheckStatus {
    Pending,
    Running,
    Passed,
    Failed(String),
}

#[derive(Clone, Debug)]
pub(super) enum DownloadStatus {
    Pending,
    InProgress(u8),
    Done,
    Failed(String),
}

pub(super) struct DownloadProgress {
    pub(super) index: usize,
    pub(super) status: DownloadStatus,
}

pub(super) const WHISPER_MODELS: &[(LocalModelSize, &str, &str)] = &[
    (LocalModelSize::Turbo, "Turbo (Recommended)", "~1.4 GB"),
    (LocalModelSize::Large, "Large", "~2.9 GB"),
    (LocalModelSize::Medium, "Medium", "~1.5 GB"),
    (LocalModelSize::Small, "Small", "~466 MB"),
];

pub(super) struct OnboardingState {
    pub(super) step: OnboardingStep,
    pub(super) full_text: String,
    pub(super) visible_chars: usize,
    pub(super) text_complete: bool,
    pub(super) anim_frame: usize,
    pub(super) selected_mode: usize,
    pub(super) selected_provider: usize,
    pub(super) api_key_input: String,
    pub(super) api_key_valid: Option<bool>,
    pub(super) validation_fail_selection: usize,
    pub(super) validation_task: Option<tokio::task::JoinHandle<Result<bool, anyhow::Error>>>,
    pub(super) user_name: String,
    pub(super) user_role: String,
    pub(super) processing_task: Option<tokio::task::JoinHandle<Result<(Option<(WorldData, WorldEntityExtractionResult)>, Option<String>), anyhow::Error>>>,
    pub(super) processed_world: Option<WorldData>,
    pub(super) processed_entities: Option<WorldEntityExtractionResult>,
    pub(super) ollama_available: bool,
    pub(super) transition_frame: usize,
    pub(super) transitioning: bool,
    pub(super) confirm_owner: String,
    pub(super) confirm_role: String,
    pub(super) confirm_org: String,
    pub(super) confirm_people: String,
    pub(super) confirm_selection: usize,
    pub(super) system_checks: Vec<(String, CheckStatus)>,
    pub(super) system_check_rx: Option<mpsc::UnboundedReceiver<(usize, bool, String)>>,
    pub(super) system_check_task: Option<tokio::task::JoinHandle<()>>,
    pub(super) system_check_done: bool,
    pub(super) system_check_selection: usize,
    pub(super) ollama_reachable: bool,
    pub(super) setup_phase: u8,
    pub(super) whisper_model_selection: usize,
    pub(super) ollama_model_selection: usize,
    pub(super) ollama_available_models: Vec<String>,
    pub(super) ollama_models_fetched: bool,
    pub(super) download_task: Option<tokio::task::JoinHandle<()>>,
    pub(super) download_rx: Option<mpsc::UnboundedReceiver<DownloadProgress>>,
    pub(super) download_items: Vec<(String, DownloadStatus)>,
}

impl OnboardingState {
    pub(super) fn new() -> Self {
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

    pub(super) fn set_step_text(&mut self, text: &str, animated: bool) {
        self.full_text = text.to_string();
        if animated {
            self.visible_chars = 0;
            self.text_complete = false;
        } else {
            self.visible_chars = self.full_text.chars().count();
            self.text_complete = true;
        }
    }

    pub(super) fn tick_typewriter_lines(&mut self) {
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

    pub(super) fn visible_text(&self) -> &str {
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

    /// Reset system-check state and spawn the async check task.
    pub(super) fn start_system_checks(&mut self) {
        self.system_checks = vec![
            ("FFmpeg".to_string(), CheckStatus::Pending),
            ("Ollama".to_string(), CheckStatus::Pending),
            ("Ollama server".to_string(), CheckStatus::Pending),
        ];
        self.system_check_done = false;
        self.system_check_selection = 0;
        self.ollama_reachable = false;
        self.set_step_text("Checking your setup...", false);

        let (tx, rx) = mpsc::unbounded_channel();
        self.system_check_rx = Some(rx);
        self.system_check_task = Some(tokio::spawn(async move {
            // Check 1: FFmpeg
            let _ = tx.send((0, false, String::new())); // mark running
            let ffmpeg_ok = crate::core::transcription::find_ffmpeg().is_ok();
            let _ = tx.send((0, ffmpeg_ok, if ffmpeg_ok { String::new() } else {
                if cfg!(target_os = "macos") {
                    "Install with: brew install ffmpeg".to_string()
                } else if cfg!(target_os = "windows") {
                    "Install ffmpeg: https://ffmpeg.org/download.html".to_string()
                } else {
                    "Install with your package manager (e.g. apt install ffmpeg)".to_string()
                }
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
                if cfg!(target_os = "macos") {
                    "Install with: brew install ollama".to_string()
                } else {
                    "Install Ollama: https://ollama.ai/download".to_string()
                }
            }));

            if !ollama_bin {
                // Skip server check if binary not found
                let _ = tx.send((2, false, "Install Ollama first".to_string()));
                return;
            }

            // Check 3: Ollama server responding
            let _ = tx.send((2, false, String::new())); // mark running
            let client = match reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build() {
                Ok(c) => c,
                Err(_) => {
                    let _ = tx.send((2, false, "Failed to create HTTP client".to_string()));
                    return;
                }
            };
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

impl Dashboard {
    pub(super) async fn handle_onboarding_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
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
                    ob.set_step_text("Scriba uses AI to extract names, topics, and summaries\nfrom your recordings. Choose a mode:", false);
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
                            ob.start_system_checks();
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
                                ob.start_system_checks();
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
                             with what Scriba knows about you and your world.\n\n\
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
                                 Every recording you make will be enriched\n\
                                 with what Scriba knows about you and your world.\n\n\
                                 Let's get started.",
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

    pub(super) fn start_onboarding_processing(&mut self) {
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

    pub(super) fn render_onboarding(&self, f: &mut Frame, area: ratatui::layout::Rect) {
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

    pub(super) fn render_dim_fade(&self, f: &mut Frame, area: ratatui::layout::Rect, ob: &OnboardingState) {
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
