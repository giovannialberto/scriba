use crate::core::{
    CloudProvider, EnrichmentMode, LocalModelSize, ScribaConfig, TranscriptionMode,
};
use crate::enrichment::OllamaClient;
use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use tokio::sync::mpsc;

use super::app::{Dashboard, DashboardAction, DashboardView};
use super::chat::ACCENT;

// ─── Settings types ──────────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Clone)]
pub(super) enum ModelPickerState {
    Closed,
    Open,
    EditingCustom,
}

#[derive(Debug, Clone)]
pub(super) struct ModelPickerItem {
    pub(super) display_name: String,
    /// None means this is the "Custom..." sentinel.
    pub(super) model_id: Option<String>,
}

// ─── Settings impl block ─────────────────────────────────────────────────────

impl Dashboard {
    pub(super) fn is_editing_settings_field(&self) -> bool {
        self.editing_api_key || self.model_picker_state != ModelPickerState::Closed || self.editing_enrichment_endpoint || self.editing_enrichment_api_key
    }

    pub(super) fn save_enrichment_config(&mut self) -> Result<()> {
        self.config.save()?;
        self.config = ScribaConfig::load()?;
        Ok(())
    }

    pub(super) fn close_model_picker(&mut self) {
        self.model_picker_state = ModelPickerState::Closed;
        self.model_picker_items.clear();
        self.model_picker_selection = 0;
        self.model_picker_custom_input.clear();
        self.ollama_models_rx = None;
    }

    pub(super) fn open_model_picker(&mut self) {
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

    pub(super) async fn handle_settings_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
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
                                        .cloned()
                                        .unwrap_or_default();
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

    pub(super) fn render_settings(&self, f: &mut Frame, area: ratatui::layout::Rect) {
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
        let pad = 28; // label column width (must fit "Anthropic (Claude) API Key")

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
                let prefix: String = api_key.chars().take(4).collect();
                format!("{}******", prefix)
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
                    Some(key) if key.len() >= 4 => {
                        let prefix: String = key.chars().take(4).collect();
                        format!("{}******", prefix)
                    }
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
}
