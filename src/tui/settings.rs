use crate::core::{
    CloudProvider, EnrichmentMode, LocalModel, TranscriptionMode,
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

// ─── Settings index layout ───────────────────────────────────────────────────
//
// Index 0 is always the mode toggle. Mode-specific items follow (3 for Private,
// 4 for Cloud), then 2 shared items (Recording).

const IDX_MODE: usize = 0;
/// Number of mode-specific items in Private mode (STT Model, Ollama Model, Ollama Server).
const PRIVATE_MODE_ITEMS: usize = 3;
/// Number of mode-specific items in Cloud mode (Whisper API Key, LLM Provider, Model, Provider API Key).
const CLOUD_MODE_ITEMS: usize = 4;
/// Number of shared items (Auto-Stop, Timeout).
const SHARED_ITEMS: usize = 2;

/// First shared-section index for a given mode.
fn shared_offset(is_private: bool) -> usize {
    1 + if is_private { PRIVATE_MODE_ITEMS } else { CLOUD_MODE_ITEMS }
}

/// Maximum selectable index for a given mode.
fn max_index(is_private: bool) -> usize {
    shared_offset(is_private) + SHARED_ITEMS - 1
}

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
        self.config.save()
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
        // Dual-mode settings layout:
        let is_private = self.config.is_private_mode();
        let max_idx = max_index(is_private);
        let shared_off = shared_offset(is_private);

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
                    self.settings_selection = std::cmp::min(self.settings_selection + 1, max_idx);
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Enter => {
                // Handle active editing states first (these are mode-independent)
                if self.editing_api_key {
                    // Save OpenAI transcription API key
                    let new_mode = TranscriptionMode::Api {
                        api_key: self.api_key_input.clone(),
                    };
                    if let Err(e) = self.config.set_transcription_mode(new_mode) {
                        self.message = format!("Failed to save API key: {}", e);
                        self.show_message = true;
                        self.return_to_view = Some(DashboardView::Settings);
                    }
                    self.editing_api_key = false;
                    self.api_key_input.clear();
                } else if self.model_picker_state == ModelPickerState::Open {
                    if let Some(item) = self.model_picker_items.get(self.model_picker_selection) {
                        if item.display_name == "Loading..." {
                            // no-op while loading
                        } else if let Some(ref id) = item.model_id {
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
                    // Update both the inline api_key (authoritative at runtime, read by
                    // resolve_api_key) and the per-provider map (for cross-provider
                    // persistence when cycling providers).
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
                    let idx = self.settings_selection;

                    if idx == IDX_MODE {
                        // ── Atomic mode switch (Private ↔ Cloud) ────────
                        // Build new config on a clone so the original stays
                        // consistent if save() fails.
                        self.close_model_picker();
                        let mut new_cfg = self.config.clone();
                        if is_private {
                            // Private → Cloud
                            new_cfg.enrichment.last_ollama_endpoint = Some(new_cfg.enrichment.ollama_endpoint());
                            new_cfg.enrichment.last_ollama_model = Some(new_cfg.enrichment.ollama_model());
                            let provider = new_cfg.last_cloud_provider.clone().unwrap_or(CloudProvider::Anthropic);
                            let transcription_key = new_cfg.last_api_key.clone().unwrap_or_default();
                            let enrichment_key = new_cfg.enrichment.load_key_for_provider(&provider);
                            let enrichment_model = new_cfg.enrichment.load_model_for_provider(&provider);
                            new_cfg.enrichment.mode = EnrichmentMode::Cloud {
                                provider,
                                api_key: enrichment_key,
                                model: enrichment_model,
                            };
                            // set_transcription_mode preserves last_local_model_size & last_api_key, then saves
                            new_cfg.set_transcription_mode(TranscriptionMode::Api { api_key: transcription_key })?;
                        } else {
                            // Cloud → Private
                            if let Some(p) = new_cfg.enrichment.cloud_provider().cloned() {
                                new_cfg.last_cloud_provider = Some(p.clone());
                                let key = new_cfg.enrichment.api_key().unwrap_or("").to_string();
                                new_cfg.enrichment.save_key_for_provider(&p, &key);
                                let model = match &new_cfg.enrichment.mode {
                                    EnrichmentMode::Cloud { model, .. } => model.clone(),
                                    _ => None,
                                };
                                new_cfg.enrichment.save_model_for_provider(&p, &model);
                            }
                            let model = new_cfg.last_local_model.unwrap_or(LocalModel::ParakeetTdt);
                            let ep = new_cfg.enrichment.last_ollama_endpoint.clone()
                                .unwrap_or_else(|| "http://localhost:11434".to_string());
                            let mdl = new_cfg.enrichment.last_ollama_model.clone()
                                .unwrap_or_else(|| "mistral:latest".to_string());
                            new_cfg.enrichment.mode = EnrichmentMode::Local {
                                ollama_endpoint: ep,
                                ollama_model: mdl,
                            };
                            // set_transcription_mode preserves last_api_key & last_local_model, then saves
                            new_cfg.set_transcription_mode(TranscriptionMode::Local { model })?;
                        }
                        // Save succeeded — adopt the new config
                        self.config = new_cfg;
                        self.settings_selection = IDX_MODE;

                    } else if idx < shared_off {
                        // ── Mode-specific items ─────────────────────────
                        match (idx, is_private) {
                            (1, true) => {
                                // Cycle transcription model
                                if let TranscriptionMode::Local { model } = &self.config.transcription {
                                    let models = LocalModel::all_models();
                                    let current_idx = models.iter().position(|m| m == model).unwrap_or(0);
                                    let next_idx = (current_idx + 1) % models.len();
                                    let new_mode = TranscriptionMode::Local { model: models[next_idx] };
                                    if let Err(e) = self.config.set_transcription_mode(new_mode) {
                                        self.message = format!("Failed to change model: {}", e);
                                        self.show_message = true;
                                        self.return_to_view = Some(DashboardView::Settings);
                                    }
                                }
                            }
                            (2, true) => {
                                // Open Ollama model picker
                                self.open_model_picker();
                            }
                            (3, true) => {
                                // Edit Ollama server URL
                                self.editing_enrichment_endpoint = true;
                                self.enrichment_endpoint_input = self.config.enrichment.ollama_endpoint();
                            }
                            (1, false) => {
                                // Edit OpenAI transcription API key
                                self.editing_api_key = true;
                                self.api_key_input = match &self.config.transcription {
                                    TranscriptionMode::Api { api_key } => api_key.clone(),
                                    _ => String::new(),
                                };
                            }
                            (2, false) => {
                                // Cycle cloud enrichment provider: Anthropic → OpenAI → Google → Anthropic
                                self.close_model_picker();
                                let (cur_provider, cur_key, cur_model) = match &self.config.enrichment.mode {
                                    EnrichmentMode::Cloud { provider, api_key, model } => {
                                        (provider.clone(), api_key.clone(), model.clone())
                                    }
                                    _ => (CloudProvider::Anthropic, String::new(), None),
                                };
                                self.config.enrichment.save_key_for_provider(&cur_provider, &cur_key);
                                self.config.enrichment.save_model_for_provider(&cur_provider, &cur_model);
                                let next_provider = match cur_provider {
                                    CloudProvider::Anthropic => CloudProvider::OpenAI,
                                    CloudProvider::OpenAI => CloudProvider::Google,
                                    CloudProvider::Google => CloudProvider::Anthropic,
                                };
                                let next_key = self.config.enrichment.load_key_for_provider(&next_provider);
                                let next_model = self.config.enrichment.load_model_for_provider(&next_provider);
                                self.config.enrichment.mode = EnrichmentMode::Cloud {
                                    provider: next_provider,
                                    api_key: next_key,
                                    model: next_model,
                                };
                                if let Err(e) = self.save_enrichment_config() {
                                    self.message = format!("Failed to save provider: {}", e);
                                    self.show_message = true;
                                    self.return_to_view = Some(DashboardView::Settings);
                                }
                            }
                            (3, false) => {
                                // Open cloud model picker
                                self.open_model_picker();
                            }
                            (4, false) => {
                                // Edit enrichment API key
                                self.editing_enrichment_api_key = true;
                                self.enrichment_api_key_input = self.config.enrichment.api_key().unwrap_or("").to_string();
                            }
                            _ => {}
                        }

                    } else {
                        // ── Shared sections (Recording) ─
                        match idx - shared_off {
                            0 => {
                                // Toggle silence auto-stop
                                self.config.silence_auto_stop.enabled = !self.config.silence_auto_stop.enabled;
                                if let Err(e) = self.config.save() {
                                    self.message = format!("Failed to save setting: {}", e);
                                    self.show_message = true;
                                    self.return_to_view = Some(DashboardView::Settings);
                                }
                            }
                            1 => {
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
                                    }
                                }
                            }
                            _ => {}
                        }
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

    /// Render the inline model picker items into the given `lines` vec.
    fn render_model_picker_items<'a>(
        &self,
        lines: &mut Vec<Line<'a>>,
        val_selected: Style,
        val_editing: Style,
        val_normal: Style,
    ) {
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
                    Span::styled(arrow.to_string(), style),
                    Span::styled(format!("Custom: {}_", self.model_picker_custom_input), style),
                ]));
            } else {
                let suffix = if is_current && !is_cursor { " (current)" } else { "" };
                lines.push(Line::from(vec![
                    Span::styled(arrow.to_string(), style),
                    Span::styled(format!("{}{}", item.display_name, suffix), style),
                ]));
            }
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
        let marker_style = Style::default().fg(ACCENT);
        let pad = 28; // label column width (must fit "Anthropic (Claude) API Key")

        let is_private = self.config.is_private_mode();
        let sel = self.settings_selection;
        let mut lines: Vec<Line> = Vec::new();

        // Shared section indices depend on mode
        let shared_off = shared_offset(is_private);

        // Helper macro for building a setting line with ▸ selection marker
        macro_rules! setting_line {
            ($label:expr, $value:expr, $idx:expr, $hint:expr, $style_override:expr) => {{
                let is_sel = sel == $idx;
                let marker = if is_sel { "\u{25B8} " } else { "  " };
                let padded_label = format!("{:<width$}", $label, width = pad);
                let v_style = if let Some(s) = $style_override { s } else if is_sel { val_selected } else { val_normal };
                let mut spans = vec![
                    Span::styled(marker, if is_sel { marker_style } else { Style::default() }),
                    Span::styled(padded_label, label_style),
                    Span::styled($value.to_string(), v_style),
                ];
                if is_sel && !$hint.is_empty() {
                    spans.push(Span::styled(format!("  {}", $hint), hint_style));
                }
                lines.push(Line::from(spans));
            }};
        }

        // ── MODE TOGGLE (index 0) ───────────────────────────────────
        lines.push(Line::from(""));
        {
            let (priv_style, cloud_style) = if is_private {
                (
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    Style::default().fg(Color::DarkGray),
                )
            } else {
                (
                    Style::default().fg(Color::DarkGray),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                )
            };
            let marker = if sel == IDX_MODE { "\u{25B8} " } else { "  " };
            let m_style = if sel == IDX_MODE { marker_style } else { Style::default() };
            let mut mode_spans = vec![
                Span::styled(marker, m_style),
                Span::styled(format!("{:<width$}", "Mode", width = pad), label_style),
                Span::styled("PRIVATE", priv_style),
                Span::styled("   ", Style::default()),
                Span::styled("CLOUD", cloud_style),
            ];
            if sel == IDX_MODE {
                mode_spans.push(Span::styled("  \u{2190} Enter to switch", hint_style));
            }
            lines.push(Line::from(mode_spans));
        }

        // ── MODE-SPECIFIC SECTION ───────────────────────────────────
        if is_private {
            // ── PRIVATE ─────────────────────────────────────────────
            lines.push(Line::from(""));
            lines.push(Line::from(vec![Span::raw("  "), Span::styled("PRIVATE", section_style)]));

            // Index 1: Whisper Model Size
            if let TranscriptionMode::Local { model } = &self.config.transcription {
                setting_line!("STT Model", model.display_name(), 1, "\u{2190} Enter to cycle", None::<Style>);
            }

            // Index 2: Ollama Model (picker)
            let model_idx: usize = 2;
            if self.model_picker_state == ModelPickerState::Closed {
                setting_line!("Ollama Model", self.config.enrichment.model_name(), model_idx, "\u{2190} Enter to choose", None::<Style>);
            } else {
                let picker_marker = if sel == model_idx { "\u{25B8} " } else { "  " };
                lines.push(Line::from(vec![
                    Span::styled(picker_marker, if sel == model_idx { marker_style } else { Style::default() }),
                    Span::styled(format!("{:<width$}", "Ollama Model", width = pad), label_style),
                    Span::styled("(select below)", hint_style),
                ]));
                self.render_model_picker_items(&mut lines, val_selected, val_editing, val_normal);
            }

            // Index 3: Ollama Server
            let endpoint_display = if self.editing_enrichment_endpoint {
                format!("{}_", self.enrichment_endpoint_input)
            } else {
                self.config.enrichment.ollama_endpoint().to_string()
            };
            let style_override = if sel == 3 && self.editing_enrichment_endpoint { Some(val_editing) } else { None };
            setting_line!("Ollama Server", endpoint_display, 3, "\u{2190} Enter to edit", style_override);

        } else {
            // ── CLOUD ───────────────────────────────────────────────
            lines.push(Line::from(""));
            lines.push(Line::from(vec![Span::raw("  "), Span::styled("CLOUD", section_style)]));

            // Index 1: Whisper API Key (transcription)
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
                setting_line!("Whisper API Key", api_key_display, 1, "\u{2190} Enter to edit", style_override);
            }

            // Index 2: LLM Provider (cycle)
            setting_line!("LLM Provider", self.config.enrichment.provider_display_name(), 2, "\u{2190} Enter to cycle", None::<Style>);

            // Index 3: Model (picker)
            let model_idx: usize = 3;
            if self.model_picker_state == ModelPickerState::Closed {
                setting_line!("Model", self.config.enrichment.model_name(), model_idx, "\u{2190} Enter to choose", None::<Style>);
            } else {
                let picker_marker = if sel == model_idx { "\u{25B8} " } else { "  " };
                lines.push(Line::from(vec![
                    Span::styled(picker_marker, if sel == model_idx { marker_style } else { Style::default() }),
                    Span::styled(format!("{:<width$}", "Model", width = pad), label_style),
                    Span::styled("(select below)", hint_style),
                ]));
                self.render_model_picker_items(&mut lines, val_selected, val_editing, val_normal);
            }

            // Index 4: Enrichment API Key
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
        setting_line!("Auto-Stop", silence_value, shared_off, "\u{2190} Enter to toggle", None::<Style>);

        let timeout_secs = self.config.silence_auto_stop.timeout_seconds;
        let timeout_display = match timeout_secs {
            s if s < 60 => format!("{}s", s),
            s if s % 60 == 0 => format!("{}m", s / 60),
            s => format!("{}m {}s", s / 60, s % 60),
        };
        let timeout_style_override = if !silence_enabled { Some(val_disabled) } else { None };
        let timeout_hint = if silence_enabled { "\u{2190} Enter to cycle" } else { "(enable auto-stop first)" };
        setting_line!("Timeout", timeout_display, shared_off + 1, timeout_hint, timeout_style_override);

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
