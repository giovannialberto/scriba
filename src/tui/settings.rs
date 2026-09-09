//! Settings UI.
//!
//! Two independent cards, Speech to text and Assistant, each shaped the same
//! way: Provider, Model, then the fields that provider needs (endpoint, key,
//! server). Every choice opens the same list picker; text fields edit inline.
//! The first row summarizes the combination (Fully private / Cloud / Mixed)
//! and offers one-shot profiles. Recording and General sections follow.

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};
use tokio::sync::mpsc;

use super::app::{Dashboard, DashboardAction, DashboardView};
use super::chat::ACCENT;
use super::onboarding::LOCAL_MODELS;
use crate::core::transcription::check_model_downloaded;
use crate::core::{
    CloudProvider, DEFAULT_OLLAMA_ENDPOINT, DEFAULT_OLLAMA_MODEL, EndpointPreset, EnrichmentMode,
    LocalModel, OPENAI_TRANSCRIPTION_MODELS, ScribaConfig, TranscriptionMode, TranscriptionPreset,
};
use crate::enrichment::OllamaClient;
use crate::llm::{self, LlmTarget, ModelListEntry, Protocol};

/// Label column width. Labels are fixed and short; provider names live in the value column.
const LABEL_WIDTH: usize = 14;
/// Rows shown in a picker before it scrolls.
const PICKER_VISIBLE: usize = 9;
/// Pickers longer than this show the type-to-filter line up front.
const PICKER_FILTER_THRESHOLD: usize = 10;

// ─── Rows ────────────────────────────────────────────────────────────────────

/// Every selectable row. Which ones appear depends on the configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Row {
    Setup,
    SpeechProvider,
    SpeechModel,
    SpeechEndpoint,
    SpeechKey,
    AssistantProvider,
    AssistantModel,
    AssistantServer,
    AssistantEndpoint,
    AssistantKey,
    AutoStop,
    Timeout,
    MeetingWatch,
    CheckUpdates,
}

impl Row {
    fn label(self) -> &'static str {
        match self {
            Row::Setup => "Setup",
            Row::SpeechProvider | Row::AssistantProvider => "Provider",
            Row::SpeechModel | Row::AssistantModel => "Model",
            Row::SpeechEndpoint | Row::AssistantEndpoint => "Endpoint",
            Row::SpeechKey | Row::AssistantKey => "API key",
            Row::AssistantServer => "Server",
            Row::AutoStop => "Auto-stop",
            Row::Timeout => "Timeout",
            Row::MeetingWatch => "Meeting watch",
            Row::CheckUpdates => "Check updates",
        }
    }

    /// Section header shown above the first row of each group.
    fn section(self) -> Option<&'static str> {
        match self {
            Row::Setup => None,
            Row::SpeechProvider | Row::SpeechModel | Row::SpeechEndpoint | Row::SpeechKey => {
                Some("SPEECH TO TEXT")
            }
            Row::AssistantProvider
            | Row::AssistantModel
            | Row::AssistantServer
            | Row::AssistantEndpoint
            | Row::AssistantKey => Some("ASSISTANT"),
            Row::AutoStop | Row::Timeout | Row::MeetingWatch => Some("RECORDING"),
            Row::CheckUpdates => Some("GENERAL"),
        }
    }
}

/// The two provider cards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Card {
    Speech,
    Assistant,
}

/// Rows for the current configuration, in display order.
fn settings_rows(config: &ScribaConfig) -> Vec<Row> {
    let mut rows = vec![Row::Setup, Row::SpeechProvider, Row::SpeechModel];
    match speech_provider(config) {
        SpeechProvider::Local => {}
        SpeechProvider::Custom => rows.extend([Row::SpeechEndpoint, Row::SpeechKey]),
        _ => rows.push(Row::SpeechKey),
    }
    rows.extend([Row::AssistantProvider, Row::AssistantModel]);
    match assistant_provider(config) {
        AssistantProvider::Ollama => rows.push(Row::AssistantServer),
        AssistantProvider::Custom => rows.extend([Row::AssistantEndpoint, Row::AssistantKey]),
        _ => rows.push(Row::AssistantKey),
    }
    rows.extend([
        Row::AutoStop,
        Row::Timeout,
        Row::MeetingWatch,
        Row::CheckUpdates,
    ]);
    rows
}

// ─── Providers as the UI sees them ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpeechProvider {
    Local,
    OpenAI,
    Groq,
    DeepInfra,
    Custom,
}

impl SpeechProvider {
    const ALL: [SpeechProvider; 5] = [
        SpeechProvider::Local,
        SpeechProvider::OpenAI,
        SpeechProvider::Groq,
        SpeechProvider::DeepInfra,
        SpeechProvider::Custom,
    ];

    fn label(self) -> &'static str {
        match self {
            SpeechProvider::Local => "Local (on this Mac)",
            SpeechProvider::OpenAI => "OpenAI",
            SpeechProvider::Groq => "Groq",
            SpeechProvider::DeepInfra => "DeepInfra",
            SpeechProvider::Custom => "Custom endpoint",
        }
    }

    fn detail(self) -> &'static str {
        match self {
            SpeechProvider::Local => "nothing leaves your computer",
            SpeechProvider::OpenAI => "whisper-1, gpt-4o-transcribe",
            SpeechProvider::Groq => "fast hosted Whisper",
            SpeechProvider::DeepInfra => "hosted Whisper, low cost",
            SpeechProvider::Custom => "any OpenAI-compatible speech API",
        }
    }

    fn preset(self) -> Option<&'static TranscriptionPreset> {
        match self {
            SpeechProvider::OpenAI => TranscriptionPreset::by_name("openai"),
            SpeechProvider::Groq => TranscriptionPreset::by_name("groq"),
            SpeechProvider::DeepInfra => TranscriptionPreset::by_name("deepinfra"),
            _ => None,
        }
    }

    fn slot(self) -> &'static str {
        match self {
            SpeechProvider::Local => "local",
            SpeechProvider::OpenAI => "openai",
            SpeechProvider::Groq => "groq",
            SpeechProvider::DeepInfra => "deepinfra",
            SpeechProvider::Custom => "custom",
        }
    }
}

fn speech_provider(config: &ScribaConfig) -> SpeechProvider {
    match &config.transcription {
        TranscriptionMode::Local { .. } => SpeechProvider::Local,
        TranscriptionMode::Api { .. } => {
            match TranscriptionPreset::for_url(&config.transcription_base_url()).map(|p| p.name) {
                Some("openai") => SpeechProvider::OpenAI,
                Some("groq") => SpeechProvider::Groq,
                Some("deepinfra") => SpeechProvider::DeepInfra,
                _ => SpeechProvider::Custom,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AssistantProvider {
    Ollama,
    Anthropic,
    OpenAI,
    Google,
    DeepInfra,
    OpenRouter,
    Groq,
    Together,
    Custom,
}

impl AssistantProvider {
    const ALL: [AssistantProvider; 9] = [
        AssistantProvider::Ollama,
        AssistantProvider::Anthropic,
        AssistantProvider::OpenAI,
        AssistantProvider::Google,
        AssistantProvider::DeepInfra,
        AssistantProvider::OpenRouter,
        AssistantProvider::Groq,
        AssistantProvider::Together,
        AssistantProvider::Custom,
    ];

    fn label(self) -> &'static str {
        match self {
            AssistantProvider::Ollama => "Ollama (local)",
            AssistantProvider::Anthropic => "Anthropic",
            AssistantProvider::OpenAI => "OpenAI",
            AssistantProvider::Google => "Google",
            AssistantProvider::DeepInfra => "DeepInfra",
            AssistantProvider::OpenRouter => "OpenRouter",
            AssistantProvider::Groq => "Groq",
            AssistantProvider::Together => "Together",
            AssistantProvider::Custom => "Custom endpoint",
        }
    }

    fn detail(self) -> &'static str {
        match self {
            AssistantProvider::Ollama => "runs on this Mac",
            AssistantProvider::Anthropic => "Claude",
            AssistantProvider::OpenAI => "GPT",
            AssistantProvider::Google => "Gemini",
            AssistantProvider::DeepInfra => "open-weight models",
            AssistantProvider::OpenRouter => "one key, many models",
            AssistantProvider::Groq => "fast open-weight models",
            AssistantProvider::Together => "open-weight models",
            AssistantProvider::Custom => "vLLM, LM Studio, any OpenAI-compatible server",
        }
    }

    fn cloud_provider(self) -> Option<CloudProvider> {
        match self {
            AssistantProvider::Ollama => None,
            AssistantProvider::Anthropic => Some(CloudProvider::Anthropic),
            AssistantProvider::OpenAI => Some(CloudProvider::OpenAI),
            AssistantProvider::Google => Some(CloudProvider::Google),
            _ => Some(CloudProvider::OpenAICompatible),
        }
    }

    fn endpoint_preset(self) -> Option<&'static EndpointPreset> {
        match self {
            AssistantProvider::DeepInfra => EndpointPreset::by_name("deepinfra"),
            AssistantProvider::OpenRouter => EndpointPreset::by_name("openrouter"),
            AssistantProvider::Groq => EndpointPreset::by_name("groq"),
            AssistantProvider::Together => EndpointPreset::by_name("together"),
            _ => None,
        }
    }

    fn slot(self) -> &'static str {
        match self {
            AssistantProvider::Ollama => "ollama",
            AssistantProvider::Anthropic => "anthropic",
            AssistantProvider::OpenAI => "openai",
            AssistantProvider::Google => "google",
            AssistantProvider::DeepInfra => "deepinfra",
            AssistantProvider::OpenRouter => "openrouter",
            AssistantProvider::Groq => "groq",
            AssistantProvider::Together => "together",
            AssistantProvider::Custom => "custom",
        }
    }

    /// Whether models are listed live by the host (vs. a curated list).
    fn lists_models_live(self) -> bool {
        !matches!(
            self,
            AssistantProvider::Anthropic | AssistantProvider::OpenAI | AssistantProvider::Google
        )
    }
}

fn assistant_provider(config: &ScribaConfig) -> AssistantProvider {
    match &config.enrichment.mode {
        EnrichmentMode::Local { .. } => AssistantProvider::Ollama,
        EnrichmentMode::Cloud { provider, .. } => match provider {
            CloudProvider::Anthropic => AssistantProvider::Anthropic,
            CloudProvider::OpenAI => AssistantProvider::OpenAI,
            CloudProvider::Google => AssistantProvider::Google,
            CloudProvider::OpenAICompatible => {
                let url = config.enrichment.effective_base_url().unwrap_or_default();
                match EndpointPreset::for_url(&url).map(|p| p.name) {
                    Some("deepinfra") => AssistantProvider::DeepInfra,
                    Some("openrouter") => AssistantProvider::OpenRouter,
                    Some("groq") => AssistantProvider::Groq,
                    Some("together") => AssistantProvider::Together,
                    _ => AssistantProvider::Custom,
                }
            }
        },
    }
}

/// One-shot setups behind the Setup row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Profile {
    FullyPrivate,
    Cloud,
    LocalSpeechClaude,
    KeepCustom,
}

/// What the combination of the two cards amounts to.
fn setup_badge(config: &ScribaConfig) -> (&'static str, &'static str) {
    let speech_local = speech_provider(config) == SpeechProvider::Local;
    let assistant_local = assistant_provider(config) == AssistantProvider::Ollama;
    match (speech_local, assistant_local) {
        (true, true) => ("\u{25CF}", "Fully private"),
        (false, false) => ("\u{25CF}", "Cloud"),
        _ => ("\u{25D0}", "Mixed"),
    }
}

// ─── Editing state ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(super) enum PickerValue {
    Speech(SpeechProvider),
    Assistant(AssistantProvider),
    LocalStt(LocalModel),
    Model(String),
    /// "Type a model id…" sentinel.
    TypeModel,
    Profile(Profile),
}

#[derive(Debug, Clone)]
pub(super) struct PickerItem {
    pub(super) label: String,
    pub(super) detail: String,
    pub(super) value: PickerValue,
}

/// What the settings screen is doing besides navigating.
#[derive(Debug, Clone)]
pub(super) enum SettingsEdit {
    None,
    /// Inline text entry on `row`.
    Text {
        row: Row,
        buffer: String,
        secret: bool,
        note: Option<String>,
    },
    /// List picker opened from `row`.
    Picker {
        row: Row,
        title: String,
        items: Vec<PickerItem>,
        selection: usize,
        filter: String,
        loading: bool,
        message: Option<String>,
    },
}

/// Result of probing an API key against its host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum KeyStatus {
    Checking,
    Valid,
    Invalid(String),
}

/// Ellipsize `text` to at most `max` characters.
fn fit(text: &str, max: usize) -> String {
    let count = text.chars().count();
    if count <= max {
        return text.to_string();
    }
    if max <= 1 {
        return "\u{2026}".to_string();
    }
    let mut out: String = text.chars().take(max - 1).collect();
    out.push('\u{2026}');
    out
}

/// Mask a secret: bullets plus the last four characters.
fn mask_secret(secret: &str) -> String {
    let count = secret.chars().count();
    if count == 0 {
        return String::new();
    }
    if count <= 4 {
        return "\u{2022}".repeat(count);
    }
    let tail: String = secret.chars().skip(count - 4).collect();
    format!("{}\u{2026}{}", "\u{2022}".repeat(8), tail)
}

fn visible_picker_items<'a>(items: &'a [PickerItem], filter: &str) -> Vec<(usize, &'a PickerItem)> {
    let needle = filter.trim().to_lowercase();
    items
        .iter()
        .enumerate()
        .filter(|(_, item)| {
            needle.is_empty()
                || matches!(item.value, PickerValue::TypeModel)
                || item.label.to_lowercase().contains(&needle)
                || item.detail.to_lowercase().contains(&needle)
        })
        .collect()
}

// ─── Dashboard: settings behavior ────────────────────────────────────────────

impl Dashboard {
    pub(super) fn is_editing_settings_field(&self) -> bool {
        !matches!(self.settings_edit, SettingsEdit::None)
    }

    /// Called when the Settings view opens: probe keys that have not been checked yet.
    pub(super) fn on_settings_opened(&mut self) {
        self.settings_edit = SettingsEdit::None;
        self.settings_selection = 0;
        if self.speech_key_status.is_none() {
            self.probe_key(Card::Speech);
        }
        if self.assistant_key_status.is_none() {
            self.probe_key(Card::Assistant);
        }
    }

    fn settings_error(&mut self, what: &str, err: impl std::fmt::Display) {
        self.message = format!("Failed to save {}: {}", what, err);
        self.show_message = true;
        self.return_to_view = Some(DashboardView::Settings);
    }

    fn save_settings(&mut self, what: &str) {
        if let Err(e) = self.config.save() {
            self.settings_error(what, e);
        }
    }

    fn clamp_settings_selection(&mut self) {
        let max = settings_rows(&self.config).len().saturating_sub(1);
        if self.settings_selection > max {
            self.settings_selection = max;
        }
    }

    // ── Key handling ──────────────────────────────────────────────────

    pub(super) async fn handle_settings_keys(
        &mut self,
        key_code: KeyCode,
    ) -> Result<DashboardAction> {
        let rows = settings_rows(&self.config);

        match key_code {
            KeyCode::Esc => {
                if self.is_editing_settings_field() {
                    self.settings_edit = SettingsEdit::None;
                    self.model_list_rx = None;
                } else {
                    self.current_view = DashboardView::Main;
                }
            }
            KeyCode::Up => match &mut self.settings_edit {
                SettingsEdit::Picker { selection, .. } => *selection = selection.saturating_sub(1),
                SettingsEdit::Text { .. } => {}
                SettingsEdit::None => {
                    self.settings_selection = self.settings_selection.saturating_sub(1);
                }
            },
            KeyCode::Down => match &mut self.settings_edit {
                SettingsEdit::Picker {
                    items,
                    selection,
                    filter,
                    ..
                } => {
                    let visible = visible_picker_items(items, filter).len();
                    if visible > 0 {
                        *selection = (*selection + 1).min(visible - 1);
                    }
                }
                SettingsEdit::Text { .. } => {}
                SettingsEdit::None => {
                    self.settings_selection = (self.settings_selection + 1).min(rows.len() - 1);
                }
            },
            KeyCode::Enter => {
                match std::mem::replace(&mut self.settings_edit, SettingsEdit::None) {
                    SettingsEdit::Text { row, buffer, .. } => self.commit_text(row, buffer),
                    SettingsEdit::Picker {
                        row,
                        items,
                        selection,
                        filter,
                        loading,
                        ..
                    } => {
                        if loading {
                            // Keep waiting for the list.
                            self.settings_edit = SettingsEdit::Picker {
                                row,
                                title: String::new(),
                                items,
                                selection,
                                filter,
                                loading,
                                message: None,
                            };
                            self.restore_picker_title(row);
                        } else {
                            let picked = visible_picker_items(&items, &filter)
                                .get(selection)
                                .map(|(_, item)| item.value.clone());
                            if let Some(value) = picked {
                                self.pick(row, value);
                            }
                        }
                    }
                    SettingsEdit::None => {
                        if let Some(row) = rows.get(self.settings_selection).copied() {
                            self.activate_row(row);
                        }
                    }
                }
            }
            KeyCode::Char(c) => match &mut self.settings_edit {
                SettingsEdit::Text { buffer, .. } => buffer.push(c),
                SettingsEdit::Picker {
                    filter, selection, ..
                } => {
                    filter.push(c);
                    *selection = 0;
                }
                SettingsEdit::None => {}
            },
            KeyCode::Backspace => match &mut self.settings_edit {
                SettingsEdit::Text { buffer, .. } => {
                    buffer.pop();
                }
                SettingsEdit::Picker {
                    filter, selection, ..
                } => {
                    filter.pop();
                    *selection = 0;
                }
                SettingsEdit::None => {}
            },
            _ => {}
        }
        self.clamp_settings_selection();
        Ok(DashboardAction::Continue)
    }

    fn restore_picker_title(&mut self, row: Row) {
        let title = self.picker_title(row);
        if let SettingsEdit::Picker { title: t, .. } = &mut self.settings_edit {
            *t = title;
        }
    }

    fn picker_title(&self, row: Row) -> String {
        match row {
            Row::Setup => "Change setup".to_string(),
            Row::SpeechProvider => "Choose a speech-to-text provider".to_string(),
            Row::AssistantProvider => "Choose an assistant provider".to_string(),
            Row::SpeechModel => format!(
                "Choose a model \u{00B7} {}",
                speech_provider(&self.config).label()
            ),
            Row::AssistantModel => {
                format!(
                    "Choose a model \u{00B7} {}",
                    assistant_provider(&self.config).label()
                )
            }
            _ => String::new(),
        }
    }

    fn open_picker(&mut self, row: Row, items: Vec<PickerItem>, current: usize) {
        self.settings_edit = SettingsEdit::Picker {
            row,
            title: self.picker_title(row),
            items,
            selection: current,
            filter: String::new(),
            loading: false,
            message: None,
        };
    }

    fn open_loading_picker(&mut self, row: Row) {
        self.settings_edit = SettingsEdit::Picker {
            row,
            title: self.picker_title(row),
            items: Vec::new(),
            selection: 0,
            filter: String::new(),
            loading: true,
            message: None,
        };
    }

    fn open_text(&mut self, row: Row, buffer: String, secret: bool, note: Option<String>) {
        self.settings_edit = SettingsEdit::Text {
            row,
            buffer,
            secret,
            note,
        };
    }

    // ── Row activation ────────────────────────────────────────────────

    fn activate_row(&mut self, row: Row) {
        match row {
            Row::Setup => {
                let items = vec![
                    PickerItem {
                        label: "Fully private".into(),
                        detail: "local speech + Ollama".into(),
                        value: PickerValue::Profile(Profile::FullyPrivate),
                    },
                    PickerItem {
                        label: "Cloud".into(),
                        detail: "OpenAI speech + Claude".into(),
                        value: PickerValue::Profile(Profile::Cloud),
                    },
                    PickerItem {
                        label: "Local speech + Claude".into(),
                        detail: "Parakeet here, Claude for the assistant".into(),
                        value: PickerValue::Profile(Profile::LocalSpeechClaude),
                    },
                    PickerItem {
                        label: "Keep my custom mix".into(),
                        detail: String::new(),
                        value: PickerValue::Profile(Profile::KeepCustom),
                    },
                ];
                self.open_picker(row, items, 0);
            }
            Row::SpeechProvider => {
                let current = speech_provider(&self.config);
                let items: Vec<PickerItem> = SpeechProvider::ALL
                    .iter()
                    .map(|p| PickerItem {
                        label: p.label().into(),
                        detail: self.speech_provider_detail(*p),
                        value: PickerValue::Speech(*p),
                    })
                    .collect();
                let idx = SpeechProvider::ALL
                    .iter()
                    .position(|p| *p == current)
                    .unwrap_or(0);
                self.open_picker(row, items, idx);
            }
            Row::AssistantProvider => {
                let current = assistant_provider(&self.config);
                let items: Vec<PickerItem> = AssistantProvider::ALL
                    .iter()
                    .map(|p| PickerItem {
                        label: p.label().into(),
                        detail: self.assistant_provider_detail(*p),
                        value: PickerValue::Assistant(*p),
                    })
                    .collect();
                let idx = AssistantProvider::ALL
                    .iter()
                    .position(|p| *p == current)
                    .unwrap_or(0);
                self.open_picker(row, items, idx);
            }
            Row::SpeechModel | Row::AssistantModel => self.open_model_picker(row),
            Row::SpeechEndpoint => {
                let current = self.config.transcription_base_url();
                self.open_text(
                    row,
                    current,
                    false,
                    Some("full API root, e.g. http://host:8000/v1".into()),
                );
            }
            Row::AssistantEndpoint => {
                let current = self
                    .config
                    .enrichment
                    .effective_base_url()
                    .unwrap_or_default();
                self.open_text(
                    row,
                    current,
                    false,
                    Some("full API root, e.g. http://host:8000/v1".into()),
                );
            }
            Row::AssistantServer => {
                let current = self.config.enrichment.ollama_endpoint();
                self.open_text(row, current, false, None);
            }
            Row::SpeechKey => {
                let stored = self.config.get_api_key().unwrap_or("").to_string();
                let (buffer, note) = if stored.is_empty() {
                    match self.same_host_key(Card::Speech) {
                        Some(k) => (
                            k,
                            Some("prefilled from the Assistant key for the same host".into()),
                        ),
                        None => (String::new(), None),
                    }
                } else {
                    (stored, None)
                };
                self.open_text(row, buffer, true, note);
            }
            Row::AssistantKey => {
                let stored = self.config.enrichment.api_key().unwrap_or("").to_string();
                let (buffer, note) = if stored.is_empty() {
                    match self.same_host_key(Card::Assistant) {
                        Some(k) => (
                            k,
                            Some("prefilled from the Speech to text key for the same host".into()),
                        ),
                        None => (String::new(), None),
                    }
                } else {
                    (stored, None)
                };
                self.open_text(row, buffer, true, note);
            }
            Row::AutoStop => {
                self.config.silence_auto_stop.enabled = !self.config.silence_auto_stop.enabled;
                self.save_settings("setting");
            }
            Row::Timeout => {
                if self.config.silence_auto_stop.enabled {
                    self.config.silence_auto_stop.timeout_seconds =
                        match self.config.silence_auto_stop.timeout_seconds {
                            30 => 60,
                            60 => 120,
                            120 => 300,
                            _ => 30,
                        };
                    self.save_settings("setting");
                }
            }
            Row::MeetingWatch => {
                self.config.meeting_detection.enabled = !self.config.meeting_detection.enabled;
                self.save_settings("setting");
                self.sync_autopilot();
            }
            Row::CheckUpdates => {
                self.config.check_for_updates = !self.config.check_for_updates;
                self.save_settings("setting");
            }
        }
    }

    /// Detail shown in the provider picker: whether a key is already saved for that host.
    fn speech_provider_detail(&self, p: SpeechProvider) -> String {
        let base = p.detail().to_string();
        if p == SpeechProvider::Local {
            return base;
        }
        let has_key = if speech_provider(&self.config) == p {
            self.config.resolve_transcription_api_key().is_some()
        } else {
            self.config.stt_key_for_slot(p.slot()).is_some()
        };
        if has_key {
            format!("{base} \u{00B7} key saved \u{2713}")
        } else {
            base
        }
    }

    fn assistant_provider_detail(&self, p: AssistantProvider) -> String {
        let base = p.detail().to_string();
        if p == AssistantProvider::Ollama {
            return base;
        }
        let has_key = if assistant_provider(&self.config) == p {
            self.config.enrichment.resolve_api_key().is_some()
        } else {
            !self
                .config
                .enrichment
                .load_key_for_slot(p.slot())
                .is_empty()
        };
        if has_key {
            format!("{base} \u{00B7} key saved \u{2713}")
        } else {
            base
        }
    }

    /// Key the other card uses when both point at the same host (OpenAI, Groq, DeepInfra).
    fn same_host_key(&self, card: Card) -> Option<String> {
        let speech_slot = speech_provider(&self.config).slot();
        let assistant_slot = assistant_provider(&self.config).slot();
        if speech_slot != assistant_slot || speech_slot == "local" || speech_slot == "custom" {
            return None;
        }
        match card {
            Card::Speech => self
                .config
                .enrichment
                .api_key()
                .map(str::to_string)
                .filter(|k| !k.is_empty()),
            Card::Assistant => self
                .config
                .get_api_key()
                .map(str::to_string)
                .filter(|k| !k.is_empty()),
        }
    }

    // ── Picking and committing ────────────────────────────────────────

    fn pick(&mut self, row: Row, value: PickerValue) {
        match value {
            PickerValue::Profile(p) => self.apply_profile(p),
            PickerValue::Speech(p) => self.apply_speech_provider(p),
            PickerValue::Assistant(p) => self.apply_assistant_provider(p),
            PickerValue::LocalStt(model) => {
                if let Err(e) = self
                    .config
                    .set_transcription_mode(TranscriptionMode::Local { model })
                {
                    self.settings_error("model", e);
                }
            }
            PickerValue::Model(id) => self.commit_text(row, id),
            PickerValue::TypeModel => {
                let current = match row {
                    Row::SpeechModel => self.config.transcription_model(),
                    _ => self.config.enrichment.model_name().to_string(),
                };
                self.open_text(
                    row,
                    current,
                    false,
                    Some("model id exactly as the host lists it".into()),
                );
            }
        }
    }

    fn commit_text(&mut self, row: Row, raw: String) {
        let value = raw.trim().to_string();
        let opt = Some(value.clone()).filter(|v| !v.is_empty());
        match row {
            Row::SpeechModel => {
                self.config.set_transcription_model(opt);
                self.config.remember_stt_settings();
                self.save_settings("model");
            }
            Row::SpeechEndpoint => {
                self.config.set_transcription_base_url(opt);
                self.config.remember_stt_settings();
                self.save_settings("endpoint");
                self.speech_key_status = None;
                self.probe_key(Card::Speech);
            }
            Row::SpeechKey => {
                let mode = self.config.api_mode_with_key(value);
                if let Err(e) = self.config.set_transcription_mode(mode) {
                    self.settings_error("API key", e);
                }
                self.config.remember_stt_settings();
                self.save_settings("API key");
                self.speech_key_status = None;
                self.probe_key(Card::Speech);
            }
            Row::AssistantModel => {
                match &mut self.config.enrichment.mode {
                    EnrichmentMode::Cloud { model, .. } => *model = opt,
                    EnrichmentMode::Local { ollama_model, .. } => {
                        if let Some(m) = opt {
                            *ollama_model = m;
                        }
                    }
                }
                self.config.enrichment.remember_cloud_settings();
                self.save_settings("model");
            }
            Row::AssistantEndpoint => {
                self.config.enrichment.set_base_url(opt);
                self.config.enrichment.remember_cloud_settings();
                self.save_settings("endpoint");
                self.assistant_key_status = None;
                self.probe_key(Card::Assistant);
            }
            Row::AssistantServer => {
                if let Some(url) = opt {
                    self.config
                        .enrichment
                        .set_ollama_endpoint(url.trim_end_matches('/').to_string());
                    self.save_settings("server");
                }
            }
            Row::AssistantKey => {
                if let EnrichmentMode::Cloud { api_key, .. } = &mut self.config.enrichment.mode {
                    *api_key = value;
                }
                self.config.enrichment.remember_cloud_settings();
                self.save_settings("API key");
                self.assistant_key_status = None;
                self.probe_key(Card::Assistant);
            }
            _ => {}
        }
    }

    fn apply_profile(&mut self, profile: Profile) {
        match profile {
            Profile::FullyPrivate => {
                self.apply_speech_provider(SpeechProvider::Local);
                self.apply_assistant_provider(AssistantProvider::Ollama);
            }
            Profile::Cloud => {
                self.apply_speech_provider(SpeechProvider::OpenAI);
                self.apply_assistant_provider(AssistantProvider::Anthropic);
            }
            Profile::LocalSpeechClaude => {
                self.apply_speech_provider(SpeechProvider::Local);
                self.apply_assistant_provider(AssistantProvider::Anthropic);
            }
            Profile::KeepCustom => {}
        }
    }

    fn apply_speech_provider(&mut self, p: SpeechProvider) {
        if speech_provider(&self.config) == p {
            return;
        }
        self.config.remember_stt_settings();
        let mode = match p {
            SpeechProvider::Local => TranscriptionMode::Local {
                model: self
                    .config
                    .last_local_model
                    .unwrap_or(LocalModel::ParakeetTdt),
            },
            _ => {
                let slot = p.slot();
                let api_key = self
                    .config
                    .stt_key_for_slot(slot)
                    .or_else(|| {
                        // Older configs kept a single OpenAI key here.
                        (slot == "openai")
                            .then(|| self.config.last_api_key.clone())
                            .flatten()
                    })
                    .unwrap_or_default();
                let base_url = match p {
                    SpeechProvider::Custom => self.config.stt_custom_base_url.clone(),
                    SpeechProvider::OpenAI => None,
                    _ => p.preset().map(|pr| pr.base_url.to_string()),
                };
                let model = self
                    .config
                    .stt_model_for_slot(slot)
                    .or_else(|| p.preset().map(|pr| pr.model.to_string()));
                TranscriptionMode::Api {
                    api_key,
                    base_url,
                    model,
                }
            }
        };
        if let Err(e) = self.config.set_transcription_mode(mode) {
            self.settings_error("provider", e);
        }
        self.speech_key_status = None;
        self.probe_key(Card::Speech);
    }

    fn apply_assistant_provider(&mut self, p: AssistantProvider) {
        if assistant_provider(&self.config) == p {
            return;
        }
        let e = &mut self.config.enrichment;
        e.remember_cloud_settings();
        if let EnrichmentMode::Local {
            ollama_endpoint,
            ollama_model,
        } = &e.mode
        {
            e.last_ollama_endpoint = Some(ollama_endpoint.clone());
            e.last_ollama_model = Some(ollama_model.clone());
        }
        let slot = p.slot();
        let mut open_models = false;
        e.mode = match p.cloud_provider() {
            None => EnrichmentMode::Local {
                ollama_endpoint: e
                    .last_ollama_endpoint
                    .clone()
                    .unwrap_or_else(|| DEFAULT_OLLAMA_ENDPOINT.to_string()),
                ollama_model: e
                    .last_ollama_model
                    .clone()
                    .unwrap_or_else(|| DEFAULT_OLLAMA_MODEL.to_string()),
            },
            Some(provider) => {
                let base_url = match p {
                    AssistantProvider::Custom => e.load_base_url_for_slot("custom"),
                    _ => p.endpoint_preset().map(|pr| pr.base_url.to_string()),
                };
                let model = e.load_model_for_slot(slot);
                // Hosts without a sensible default model: ask right away.
                open_models = model.is_none()
                    && matches!(
                        p,
                        AssistantProvider::OpenRouter
                            | AssistantProvider::Groq
                            | AssistantProvider::Together
                            | AssistantProvider::Custom
                    );
                EnrichmentMode::Cloud {
                    provider,
                    api_key: e.load_key_for_slot(slot),
                    model,
                    base_url,
                }
            }
        };
        self.save_settings("provider");
        self.assistant_key_status = None;
        self.probe_key(Card::Assistant);
        if open_models {
            // Put the cursor on the row the picker belongs to.
            if let Some(idx) = settings_rows(&self.config)
                .iter()
                .position(|r| *r == Row::AssistantModel)
            {
                self.settings_selection = idx;
            }
            self.open_model_picker(Row::AssistantModel);
        }
    }

    // ── Model pickers ─────────────────────────────────────────────────

    fn open_model_picker(&mut self, row: Row) {
        match row {
            Row::SpeechModel => match speech_provider(&self.config) {
                SpeechProvider::Local => {
                    let current = self.config.get_local_model();
                    let mut items: Vec<PickerItem> = Vec::new();
                    let mut push = |model: LocalModel, label: &str, size: &str| {
                        let installed = if check_model_downloaded(model) {
                            " \u{00B7} installed \u{2713}"
                        } else {
                            ""
                        };
                        items.push(PickerItem {
                            label: label.replace(" (Recommended)", "").to_string(),
                            detail: format!("{}{}", size.trim_start_matches('~'), installed),
                            value: PickerValue::LocalStt(model),
                        });
                    };
                    for (model, label, size) in LOCAL_MODELS {
                        push(*model, label, size);
                    }
                    for model in LocalModel::all_models().iter().copied() {
                        if !LOCAL_MODELS.iter().any(|(m, _, _)| *m == model) {
                            push(model, model.display_name(), "");
                        }
                    }
                    let idx = items
                        .iter()
                        .position(
                            |i| matches!(i.value, PickerValue::LocalStt(m) if Some(m) == current),
                        )
                        .unwrap_or(0);
                    self.open_picker(row, items, idx);
                }
                SpeechProvider::OpenAI => {
                    let current = self.config.transcription_model();
                    let mut items: Vec<PickerItem> = OPENAI_TRANSCRIPTION_MODELS
                        .iter()
                        .map(|(id, label)| PickerItem {
                            label: label.to_string(),
                            detail: id.to_string(),
                            value: PickerValue::Model(id.to_string()),
                        })
                        .collect();
                    let idx = items
                        .iter()
                        .position(|i| matches!(&i.value, PickerValue::Model(m) if *m == current))
                        .unwrap_or(0);
                    items.push(type_model_item());
                    self.open_picker(row, items, idx);
                }
                _ => {
                    let base = self.config.transcription_base_url();
                    let target = LlmTarget {
                        protocol: Protocol::OpenAI,
                        endpoint: format!("{}/", base.trim_end_matches('/')),
                        api_key: self.config.resolve_transcription_api_key(),
                        api_key_env: Some(self.config.transcription_api_key_env()),
                        model: self.config.transcription_model(),
                        display_name: self.config.transcription_host_display(),
                    };
                    self.open_loading_picker(row);
                    self.spawn_model_list(async move {
                        llm::list_models(&target)
                            .await
                            .map(|names| names.into_iter().map(Into::into).collect())
                            .map_err(|e| e.to_string())
                    });
                }
            },
            Row::AssistantModel => {
                let provider = assistant_provider(&self.config);
                match provider {
                    AssistantProvider::Ollama => {
                        let endpoint = self.config.enrichment.ollama_endpoint();
                        self.open_loading_picker(row);
                        self.spawn_model_list(async move {
                            OllamaClient::fetch_model_infos(&endpoint)
                                .await
                                .map(|infos| infos.into_iter().map(Into::into).collect())
                                .map_err(|e| e.to_string())
                        });
                    }
                    p if p.lists_models_live() => {
                        let target = LlmTarget::from_config(&self.config.enrichment);
                        self.open_loading_picker(row);
                        self.spawn_model_list(async move {
                            llm::list_models(&target)
                                .await
                                .map(|names| names.into_iter().map(Into::into).collect())
                                .map_err(|e| e.to_string())
                        });
                    }
                    _ => {
                        let current = self.config.enrichment.model_name().to_string();
                        let curated = provider
                            .cloud_provider()
                            .map(|c| c.available_models())
                            .unwrap_or_default();
                        let mut items: Vec<PickerItem> = curated
                            .iter()
                            .map(|m| PickerItem {
                                label: m.display_name.clone(),
                                detail: m.model_id.clone(),
                                value: PickerValue::Model(m.model_id.clone()),
                            })
                            .collect();
                        let idx = curated
                            .iter()
                            .position(|m| m.model_id == current)
                            .unwrap_or(items.len());
                        items.push(type_model_item());
                        let selection = idx.min(items.len() - 1);
                        self.open_picker(row, items, selection);
                    }
                }
            }
            _ => {}
        }
    }

    fn spawn_model_list<F>(&mut self, fetch: F)
    where
        F: std::future::Future<Output = Result<Vec<ModelListEntry>, String>> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel(1);
        self.model_list_rx = Some(rx);
        tokio::spawn(async move {
            let _ = tx.send(fetch.await).await;
        });
    }

    /// A model list arrived for the open picker.
    pub(super) fn on_model_list(&mut self, result: Result<Vec<ModelListEntry>, String>) {
        let SettingsEdit::Picker {
            row, loading: true, ..
        } = &self.settings_edit
        else {
            return;
        };
        let row = *row;
        let current = match row {
            Row::SpeechModel => self.config.transcription_model(),
            _ => self.config.enrichment.model_name().to_string(),
        };
        let (items, message) = match result {
            Ok(entries) if !entries.is_empty() => {
                let mut items: Vec<PickerItem> = entries
                    .iter()
                    .map(|e| PickerItem {
                        label: e.id.clone(),
                        detail: match e.supports_tools {
                            Some(true) => "tools \u{2713}".to_string(),
                            Some(false) => "no tools".to_string(),
                            None => String::new(),
                        },
                        value: PickerValue::Model(e.id.clone()),
                    })
                    .collect();
                items.push(type_model_item());
                (items, None)
            }
            Ok(_) => (
                vec![type_model_item()],
                Some("The host listed no models".to_string()),
            ),
            Err(e) => (
                vec![type_model_item()],
                Some(format!(
                    "Could not list models \u{00B7} {}",
                    compact_reason(&e)
                )),
            ),
        };
        let selection = items
            .iter()
            .position(|i| matches!(&i.value, PickerValue::Model(m) if *m == current))
            .unwrap_or(0);
        if let SettingsEdit::Picker {
            items: slot,
            selection: sel,
            loading,
            message: msg,
            ..
        } = &mut self.settings_edit
        {
            *slot = items;
            *sel = selection;
            *loading = false;
            *msg = message;
        }
    }

    // ── Key probes ────────────────────────────────────────────────────

    /// Check the card's key against its host in the background.
    fn probe_key(&mut self, card: Card) {
        let tx = self.key_probe_tx.clone();
        match card {
            Card::Speech => {
                if speech_provider(&self.config) == SpeechProvider::Local {
                    self.speech_key_status = None;
                    return;
                }
                let Some(key) = self.config.resolve_transcription_api_key() else {
                    self.speech_key_status = None;
                    return;
                };
                let url = format!("{}/models", self.config.transcription_base_url());
                self.speech_key_status = Some(KeyStatus::Checking);
                tokio::spawn(async move {
                    let result = probe_bearer_endpoint(&url, &key).await;
                    let _ = tx.send((Card::Speech, result)).await;
                });
            }
            Card::Assistant => {
                if assistant_provider(&self.config) == AssistantProvider::Ollama
                    || self.config.enrichment.resolve_api_key().is_none()
                {
                    self.assistant_key_status = None;
                    return;
                }
                let config = self.config.enrichment.clone();
                self.assistant_key_status = Some(KeyStatus::Checking);
                tokio::spawn(async move {
                    let provider = crate::enrichment::create_provider(&config);
                    let result = provider.health_check().await.map_err(|e| e.to_string());
                    let _ = tx.send((Card::Assistant, result)).await;
                });
            }
        }
    }

    /// A probe finished.
    pub(super) fn on_key_probe(&mut self, card: Card, result: Result<(), String>) {
        let status = match result {
            Ok(()) => KeyStatus::Valid,
            Err(e) => KeyStatus::Invalid(e),
        };
        match card {
            Card::Speech => self.speech_key_status = Some(status),
            Card::Assistant => self.assistant_key_status = Some(status),
        }
    }

    // ── Rendering ─────────────────────────────────────────────────────

    pub(super) fn render_settings(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2), // Header
                Constraint::Min(10),   // Body
                Constraint::Length(2), // Footer
            ])
            .split(area);

        // ── Header ──────────────────────────────────────────────────
        let (badge_icon, badge_text) = setup_badge(&self.config);
        let title = Span::styled(
            "Settings",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        );
        let badge = format!("{} {}", badge_icon, badge_text);
        let gap = (chunks[0].width as usize)
            .saturating_sub(2 + "Settings".len() + badge.chars().count() + 2);
        let header_line = Line::from(vec![
            Span::raw("  "),
            title,
            Span::raw(" ".repeat(gap)),
            Span::styled(badge, Style::default().fg(ACCENT)),
            Span::raw("  "),
        ]);
        f.render_widget(
            Paragraph::new(header_line),
            Rect {
                x: chunks[0].x,
                y: chunks[0].y,
                width: chunks[0].width,
                height: 1,
            },
        );
        let sep = "\u{2500}".repeat(chunks[0].width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect {
                x: chunks[0].x,
                y: chunks[0].y + 1,
                width: chunks[0].width,
                height: 1,
            },
        );

        // ── Body ────────────────────────────────────────────────────
        let label_style = Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::BOLD);
        let val_normal = Style::default().fg(Color::White);
        let val_selected = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let val_editing = Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD);
        let val_disabled = Style::default().fg(Color::DarkGray);
        let hint_style = Style::default().fg(Color::DarkGray);
        let section_style = Style::default().fg(ACCENT).add_modifier(Modifier::BOLD);
        let marker_style = Style::default().fg(ACCENT);
        let ok_style = Style::default().fg(Color::Green);
        let bad_style = Style::default().fg(Color::Red);

        let width = chunks[1].width as usize;
        let rows = settings_rows(&self.config);
        let sel = self.settings_selection;
        let mut lines: Vec<Line> = Vec::new();
        let mut last_section: Option<&str> = None;

        for (idx, row) in rows.iter().enumerate() {
            let row = *row;
            if row.section() != last_section {
                if let Some(section) = row.section() {
                    lines.push(Line::from(""));
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(section, section_style),
                    ]));
                }
                last_section = row.section();
            }
            let is_sel = idx == sel;
            let (value, detail, detail_style, hint, editing_value) = self.row_content(row, is_sel);

            // Layout: marker(2) + label + value + detail + hint, fitted to the width.
            // The hint is short and always shown; the value keeps at least half of
            // what is left; the detail takes the rest or disappears if too cramped.
            let hint_text = if is_sel && !hint.is_empty() {
                format!("  {}", hint)
            } else {
                String::new()
            };
            let remaining = width
                .saturating_sub(2 + LABEL_WIDTH + 1)
                .saturating_sub(hint_text.chars().count());
            let value_len = value.chars().count();
            let value_max = value_len.min((remaining * 55 / 100).max(12));
            let value_fitted = fit(&value, value_max);
            let detail_room = remaining.saturating_sub(value_fitted.chars().count());
            let detail_fitted = if detail.is_empty() || detail_room < 10 {
                String::new()
            } else {
                fit(&format!("  {}", detail), detail_room)
            };

            let v_style = if editing_value {
                val_editing
            } else if is_sel {
                val_selected
            } else if row == Row::Timeout && !self.config.silence_auto_stop.enabled {
                val_disabled
            } else {
                val_normal
            };
            let mut spans = vec![
                Span::styled(
                    if is_sel { "\u{25B8} " } else { "  " },
                    if is_sel {
                        marker_style
                    } else {
                        Style::default()
                    },
                ),
                Span::styled(
                    format!("{:<width$}", row.label(), width = LABEL_WIDTH),
                    label_style,
                ),
                Span::styled(value_fitted, v_style),
            ];
            if !detail_fitted.is_empty() {
                let style = match detail_style {
                    DetailTone::Neutral => hint_style,
                    DetailTone::Good => ok_style,
                    DetailTone::Bad => bad_style,
                };
                spans.push(Span::styled(detail_fitted, style));
            }
            if !hint_text.is_empty() {
                spans.push(Span::styled(hint_text, hint_style));
            }
            lines.push(Line::from(spans));

            // Inline editor / picker under the active row
            match &self.settings_edit {
                SettingsEdit::Text {
                    row: r,
                    note: Some(note),
                    ..
                } if *r == row => {
                    lines.push(Line::from(Span::styled(
                        format!("{}{}", " ".repeat(2 + LABEL_WIDTH), note),
                        hint_style,
                    )));
                }
                SettingsEdit::Picker {
                    row: r,
                    title,
                    items,
                    selection,
                    filter,
                    loading,
                    message,
                } if *r == row => {
                    let indent = " ".repeat(2 + LABEL_WIDTH);
                    lines.push(Line::from(Span::styled(
                        format!("{}\u{250C} {} ", indent, title),
                        hint_style,
                    )));
                    if *loading {
                        lines.push(Line::from(Span::styled(
                            format!("{}\u{2502} Loading\u{2026}", indent),
                            hint_style,
                        )));
                    } else {
                        if !filter.is_empty() || items.len() > PICKER_FILTER_THRESHOLD {
                            lines.push(Line::from(vec![
                                Span::styled(format!("{}\u{2502} filter: ", indent), hint_style),
                                Span::styled(format!("{}_", filter), val_editing),
                            ]));
                        }
                        let visible = visible_picker_items(items, filter);
                        let label_w = visible
                            .iter()
                            .map(|(_, i)| i.label.chars().count())
                            .max()
                            .unwrap_or(0)
                            .min(44);
                        let start = selection
                            .saturating_sub(PICKER_VISIBLE / 2)
                            .min(visible.len().saturating_sub(PICKER_VISIBLE));
                        let end = (start + PICKER_VISIBLE).min(visible.len());
                        if visible.is_empty() {
                            lines.push(Line::from(Span::styled(
                                format!("{}\u{2502} no match", indent),
                                hint_style,
                            )));
                        }
                        for (vis_idx, (_, item)) in visible.iter().enumerate().take(end).skip(start)
                        {
                            let is_cursor = vis_idx == *selection;
                            let arrow = if is_cursor { "\u{25B8} " } else { "  " };
                            let style = if is_cursor { val_selected } else { val_normal };
                            let label = format!("{:<w$}", fit(&item.label, label_w), w = label_w);
                            let detail = fit(
                                &item.detail,
                                width.saturating_sub(2 + LABEL_WIDTH + 4 + label_w + 2),
                            );
                            lines.push(Line::from(vec![
                                Span::styled(format!("{}\u{2502} {}", indent, arrow), hint_style),
                                Span::styled(label, style),
                                Span::styled(format!("  {}", detail), hint_style),
                            ]));
                        }
                        if end < visible.len() {
                            lines.push(Line::from(Span::styled(
                                format!("{}\u{2502} \u{2026} {} more", indent, visible.len() - end),
                                hint_style,
                            )));
                        }
                    }
                    if let Some(msg) = message {
                        lines.push(Line::from(Span::styled(
                            format!("{}\u{2502} {}", indent, msg),
                            bad_style,
                        )));
                    }
                    lines.push(Line::from(Span::styled(
                        format!("{}\u{2514}", indent),
                        hint_style,
                    )));
                }
                _ => {}
            }
        }

        let body = Paragraph::new(lines).style(Style::default().fg(Color::White));
        f.render_widget(body, chunks[1]);

        // ── Footer ──────────────────────────────────────────────────
        let sep2 = "\u{2500}".repeat(chunks[2].width as usize);
        f.render_widget(
            Paragraph::new(sep2).style(Style::default().fg(Color::Indexed(237))),
            Rect {
                x: chunks[2].x,
                y: chunks[2].y,
                width: chunks[2].width,
                height: 1,
            },
        );
        let footer_area = Rect {
            x: chunks[2].x,
            y: chunks[2].y + 1,
            width: chunks[2].width,
            height: 1,
        };
        let version = env!("CARGO_PKG_VERSION");
        let left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(
                format!("scriba \u{00B7} v{}", version),
                Style::default().fg(Color::DarkGray),
            ),
        ];
        let keys: &[(&str, &str)] = match &self.settings_edit {
            SettingsEdit::None => &[
                ("\u{2191}\u{2193}", "Navigate"),
                ("Enter", "Change"),
                ("Esc", "Back"),
            ],
            SettingsEdit::Text { .. } => &[("Enter", "Save"), ("Esc", "Cancel")],
            SettingsEdit::Picker { .. } => &[
                ("\u{2191}\u{2193}", "Select"),
                ("Enter", "Pick"),
                ("type", "Filter"),
                ("Esc", "Cancel"),
            ],
        };
        let mut right_spans = Vec::new();
        for (key, action) in keys {
            right_spans.push(Span::styled("[", Style::default().fg(Color::DarkGray)));
            right_spans.push(Span::styled(*key, Style::default().fg(Color::White)));
            right_spans.push(Span::styled(
                format!("] {}  ", action),
                Style::default().fg(Color::DarkGray),
            ));
        }
        let left_w: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_w: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_area.width as usize).saturating_sub(left_w + right_w);
        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);
        f.render_widget(Paragraph::new(Line::from(spans)), footer_area);
    }

    /// Value, detail (with tone), hint, and whether the value is being edited, for one row.
    fn row_content(
        &self,
        row: Row,
        is_sel: bool,
    ) -> (String, String, DetailTone, &'static str, bool) {
        // Inline text editing takes over the value column.
        if let SettingsEdit::Text {
            row: r,
            buffer,
            secret,
            ..
        } = &self.settings_edit
        {
            if *r == row {
                let shown = if *secret {
                    mask_secret(buffer)
                } else {
                    buffer.clone()
                };
                return (
                    format!("{}_", shown),
                    String::new(),
                    DetailTone::Neutral,
                    "",
                    true,
                );
            }
        }
        let _ = is_sel;
        let config = &self.config;
        match row {
            Row::Setup => {
                let (_, text) = setup_badge(config);
                (
                    text.to_string(),
                    String::new(),
                    DetailTone::Neutral,
                    "\u{2190} Enter to change setup",
                    false,
                )
            }
            Row::SpeechProvider => (
                speech_provider(config).label().to_string(),
                String::new(),
                DetailTone::Neutral,
                "\u{2190} Enter to choose",
                false,
            ),
            Row::SpeechModel => match speech_provider(config) {
                SpeechProvider::Local => {
                    let model = config.get_local_model().unwrap_or(LocalModel::ParakeetTdt);
                    let size = LOCAL_MODELS
                        .iter()
                        .find(|(m, _, _)| *m == model)
                        .map(|(_, _, s)| s.trim_start_matches('~').to_string())
                        .unwrap_or_default();
                    let installed = check_model_downloaded(model);
                    let detail = match (size.is_empty(), installed) {
                        (false, true) => format!("{} \u{00B7} installed \u{2713}", size),
                        (false, false) => format!("{} \u{00B7} downloads on first use", size),
                        (true, true) => "installed \u{2713}".to_string(),
                        (true, false) => "downloads on first use".to_string(),
                    };
                    (
                        model.display_name().to_string(),
                        detail,
                        DetailTone::Neutral,
                        "\u{2190} Enter to choose",
                        false,
                    )
                }
                SpeechProvider::OpenAI => {
                    let model = config.transcription_model();
                    let detail = if model.contains("diarize") {
                        "speaker labels"
                    } else {
                        "curated"
                    };
                    (
                        model,
                        detail.to_string(),
                        DetailTone::Neutral,
                        "\u{2190} Enter to choose",
                        false,
                    )
                }
                _ => (
                    config.transcription_model(),
                    "listed by host".to_string(),
                    DetailTone::Neutral,
                    "\u{2190} Enter to choose",
                    false,
                ),
            },
            Row::SpeechEndpoint => (
                config.transcription_base_url(),
                String::new(),
                DetailTone::Neutral,
                "\u{2190} Enter to edit",
                false,
            ),
            Row::SpeechKey => {
                let stored = config.get_api_key().unwrap_or("");
                let env = config.transcription_api_key_env();
                key_row(
                    stored,
                    config.resolve_transcription_api_key().is_some(),
                    &env,
                    &self.speech_key_status,
                )
            }
            Row::AssistantProvider => (
                assistant_provider(config).label().to_string(),
                String::new(),
                DetailTone::Neutral,
                "\u{2190} Enter to choose",
                false,
            ),
            Row::AssistantModel => {
                let provider = assistant_provider(config);
                let unset = matches!(
                    &config.enrichment.mode,
                    EnrichmentMode::Cloud { model: None, .. }
                ) && matches!(
                    provider,
                    AssistantProvider::OpenRouter
                        | AssistantProvider::Groq
                        | AssistantProvider::Together
                        | AssistantProvider::Custom
                );
                if unset {
                    // No sensible default exists for these hosts; the built-in fallback would be wrong.
                    return (
                        "not chosen yet".to_string(),
                        String::new(),
                        DetailTone::Bad,
                        "\u{2190} Enter to choose",
                        false,
                    );
                }
                let detail = if provider.lists_models_live() {
                    "listed by host"
                } else {
                    "curated"
                };
                (
                    config.enrichment.model_name().to_string(),
                    detail.to_string(),
                    DetailTone::Neutral,
                    "\u{2190} Enter to choose",
                    false,
                )
            }
            Row::AssistantServer => (
                config.enrichment.ollama_endpoint(),
                String::new(),
                DetailTone::Neutral,
                "\u{2190} Enter to edit",
                false,
            ),
            Row::AssistantEndpoint => (
                config.enrichment.effective_base_url().unwrap_or_default(),
                String::new(),
                DetailTone::Neutral,
                "\u{2190} Enter to edit",
                false,
            ),
            Row::AssistantKey => {
                let stored = config.enrichment.api_key().unwrap_or("");
                let env = config.enrichment.api_key_env_var().unwrap_or_default();
                key_row(
                    stored,
                    config.enrichment.resolve_api_key().is_some(),
                    &env,
                    &self.assistant_key_status,
                )
            }
            Row::AutoStop => (
                if config.silence_auto_stop.enabled {
                    "Enabled"
                } else {
                    "Disabled"
                }
                .to_string(),
                String::new(),
                DetailTone::Neutral,
                "\u{2190} Enter to toggle",
                false,
            ),
            Row::Timeout => {
                let secs = config.silence_auto_stop.timeout_seconds;
                let value = match secs {
                    s if s < 60 => format!("{}s", s),
                    s if s % 60 == 0 => format!("{}m", s / 60),
                    s => format!("{}m {}s", s / 60, s % 60),
                };
                let hint = if config.silence_auto_stop.enabled {
                    "\u{2190} Enter to cycle"
                } else {
                    "(enable auto-stop first)"
                };
                (value, String::new(), DetailTone::Neutral, hint, false)
            }
            Row::MeetingWatch => (
                if config.meeting_detection.enabled {
                    "Enabled"
                } else {
                    "Disabled"
                }
                .to_string(),
                String::new(),
                DetailTone::Neutral,
                "\u{2190} Enter to toggle",
                false,
            ),
            Row::CheckUpdates => (
                if config.check_for_updates {
                    "Enabled"
                } else {
                    "Disabled"
                }
                .to_string(),
                String::new(),
                DetailTone::Neutral,
                "\u{2190} Enter to toggle",
                false,
            ),
        }
    }
}

/// Color tone for the detail column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DetailTone {
    Neutral,
    Good,
    Bad,
}

fn type_model_item() -> PickerItem {
    PickerItem {
        label: "Type a model id\u{2026}".into(),
        detail: String::new(),
        value: PickerValue::TypeModel,
    }
}

/// Value/detail for an API key row from what is stored, what resolves, and the probe result.
fn key_row(
    stored: &str,
    resolves: bool,
    env: &str,
    status: &Option<KeyStatus>,
) -> (String, String, DetailTone, &'static str, bool) {
    let value = if !stored.is_empty() {
        mask_secret(stored)
    } else if resolves {
        format!("not set \u{00B7} using {} from environment", env)
    } else {
        "not set".to_string()
    };
    let (detail, tone) = match status {
        Some(KeyStatus::Checking) => ("checking\u{2026}".to_string(), DetailTone::Neutral),
        Some(KeyStatus::Valid) => ("\u{2713} valid".to_string(), DetailTone::Good),
        Some(KeyStatus::Invalid(msg)) => (
            format!("\u{2717} rejected \u{00B7} {}", compact_reason(msg)),
            DetailTone::Bad,
        ),
        None if !stored.is_empty() => ("saved in config".to_string(), DetailTone::Neutral),
        None => (String::new(), DetailTone::Neutral),
    };
    let hint = if stored.is_empty() && !resolves {
        "\u{2190} Enter to paste, or export the variable"
    } else {
        "\u{2190} Enter to edit"
    };
    (value, detail, tone, hint, false)
}

fn first_line(msg: &str) -> String {
    msg.lines().next().unwrap_or("").trim().to_string()
}

/// Boil a provider error down to a few words for the key row.
fn compact_reason(msg: &str) -> String {
    let lower = msg.to_lowercase();
    if lower.contains("authentication")
        || lower.contains("401")
        || lower.contains("403")
        || lower.contains("api key")
    {
        "invalid key".to_string()
    } else if lower.contains("rate limit") || lower.contains("429") {
        "rate limited".to_string()
    } else if lower.contains("could not reach")
        || lower.contains("network")
        || lower.contains("timeout")
        || lower.contains("connect")
    {
        "host unreachable".to_string()
    } else {
        fit(&first_line(msg), 40)
    }
}

/// `GET {url}` with a bearer token; Ok on any 2xx.
async fn probe_bearer_endpoint(url: &str, key: &str) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    let response = client
        .get(url)
        .bearer_auth(key)
        .send()
        .await
        .map_err(|e| format!("could not reach host: {}", e))?;
    let status = response.status();
    if status.is_success() {
        Ok(())
    } else {
        Err(format!("HTTP {}", status.as_u16()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with(transcription: TranscriptionMode, mode: EnrichmentMode) -> ScribaConfig {
        let mut config = ScribaConfig::default();
        config.transcription = transcription;
        config.enrichment.mode = mode;
        config
    }

    #[test]
    fn rows_follow_the_configuration() {
        let private = config_with(
            TranscriptionMode::Local {
                model: LocalModel::ParakeetTdt,
            },
            EnrichmentMode::Local {
                ollama_endpoint: DEFAULT_OLLAMA_ENDPOINT.into(),
                ollama_model: DEFAULT_OLLAMA_MODEL.into(),
            },
        );
        let rows = settings_rows(&private);
        assert!(!rows.contains(&Row::SpeechKey), "local speech needs no key");
        assert!(rows.contains(&Row::AssistantServer));
        assert!(!rows.contains(&Row::AssistantKey));
        assert_eq!(setup_badge(&private).1, "Fully private");

        let mixed = config_with(
            TranscriptionMode::Local {
                model: LocalModel::ParakeetTdt,
            },
            EnrichmentMode::Cloud {
                provider: CloudProvider::Anthropic,
                api_key: String::new(),
                model: None,
                base_url: None,
            },
        );
        let rows = settings_rows(&mixed);
        assert!(rows.contains(&Row::AssistantKey));
        assert!(!rows.contains(&Row::AssistantEndpoint));
        assert_eq!(setup_badge(&mixed).1, "Mixed");

        let custom = config_with(
            TranscriptionMode::Api {
                api_key: String::new(),
                base_url: Some("http://stt:8000/v1".into()),
                model: None,
            },
            EnrichmentMode::Cloud {
                provider: CloudProvider::OpenAICompatible,
                api_key: String::new(),
                model: None,
                base_url: Some("http://llm:8000/v1".into()),
            },
        );
        let rows = settings_rows(&custom);
        assert!(rows.contains(&Row::SpeechEndpoint) && rows.contains(&Row::SpeechKey));
        assert!(rows.contains(&Row::AssistantEndpoint) && rows.contains(&Row::AssistantKey));
        assert_eq!(speech_provider(&custom), SpeechProvider::Custom);
        assert_eq!(assistant_provider(&custom), AssistantProvider::Custom);
        assert_eq!(setup_badge(&custom).1, "Cloud");
        assert_eq!(rows[0], Row::Setup);
        assert_eq!(*rows.last().unwrap(), Row::CheckUpdates);
    }

    #[test]
    fn providers_are_detected_from_endpoints() {
        let groq_stt = config_with(
            TranscriptionMode::Api {
                api_key: "k".into(),
                base_url: Some("https://api.groq.com/openai/v1/".into()),
                model: None,
            },
            EnrichmentMode::Cloud {
                provider: CloudProvider::OpenAICompatible,
                api_key: String::new(),
                model: None,
                base_url: Some("https://openrouter.ai/api/v1".into()),
            },
        );
        assert_eq!(speech_provider(&groq_stt), SpeechProvider::Groq);
        assert_eq!(assistant_provider(&groq_stt), AssistantProvider::OpenRouter);
        let deepinfra_default = config_with(
            TranscriptionMode::api("k"),
            EnrichmentMode::Cloud {
                provider: CloudProvider::OpenAICompatible,
                api_key: String::new(),
                model: None,
                base_url: None,
            },
        );
        assert_eq!(speech_provider(&deepinfra_default), SpeechProvider::OpenAI);
        assert_eq!(
            assistant_provider(&deepinfra_default),
            AssistantProvider::DeepInfra
        );
    }

    #[test]
    fn labels_fit_the_label_column() {
        let all = [
            Row::Setup,
            Row::SpeechProvider,
            Row::SpeechModel,
            Row::SpeechEndpoint,
            Row::SpeechKey,
            Row::AssistantProvider,
            Row::AssistantModel,
            Row::AssistantServer,
            Row::AssistantEndpoint,
            Row::AssistantKey,
            Row::AutoStop,
            Row::Timeout,
            Row::MeetingWatch,
            Row::CheckUpdates,
        ];
        for row in all {
            assert!(row.label().chars().count() < LABEL_WIDTH, "{:?}", row);
        }
    }

    #[test]
    fn text_helpers() {
        assert_eq!(fit("abcdef", 10), "abcdef");
        assert_eq!(fit("abcdefghij", 5), "abcd\u{2026}");
        assert_eq!(fit("abc", 1), "\u{2026}");
        assert_eq!(mask_secret(""), "");
        assert_eq!(mask_secret("abc"), "\u{2022}\u{2022}\u{2022}");
        assert_eq!(
            mask_secret("sk-ant-1234567890abcd"),
            "\u{2022}\u{2022}\u{2022}\u{2022}\u{2022}\u{2022}\u{2022}\u{2022}\u{2026}abcd"
        );
    }

    #[test]
    fn picker_filter_keeps_the_type_sentinel() {
        let items = vec![
            PickerItem {
                label: "Qwen/Qwen3.5".into(),
                detail: String::new(),
                value: PickerValue::Model("q".into()),
            },
            PickerItem {
                label: "meta-llama/Llama-3.3".into(),
                detail: "tools".into(),
                value: PickerValue::Model("l".into()),
            },
            type_model_item(),
        ];
        let visible = visible_picker_items(&items, "qwen");
        assert_eq!(visible.len(), 2);
        assert!(matches!(visible[1].1.value, PickerValue::TypeModel));
        assert_eq!(visible_picker_items(&items, "TOOLS").len(), 2);
        assert_eq!(visible_picker_items(&items, "").len(), 3);
    }

    #[test]
    fn key_rows_explain_where_the_key_comes_from() {
        let (value, detail, tone, hint, _) = key_row("", false, "GROQ_API_KEY", &None);
        assert_eq!(value, "not set");
        assert!(detail.is_empty());
        assert!(hint.contains("export"));
        assert_eq!(tone, DetailTone::Neutral);
        let (value, _, _, _, _) = key_row("", true, "GROQ_API_KEY", &None);
        assert!(value.contains("GROQ_API_KEY"));
        let (value, detail, tone, _, _) = key_row(
            "sk-ant-1234567890",
            true,
            "X",
            &Some(KeyStatus::Invalid("HTTP 401\nmore".into())),
        );
        assert!(value.starts_with('\u{2022}'));
        assert_eq!(detail, "\u{2717} rejected \u{00B7} invalid key");
        assert_eq!(compact_reason("Rate limited: slow down"), "rate limited");
        assert_eq!(
            compact_reason("Network error: could not reach host: dns"),
            "host unreachable"
        );
        assert_eq!(
            compact_reason("API error (500): {\"very\": \"long body\"}\nmore"),
            "API error (500): {\"very\": \"long body\"}"
        );
        assert_eq!(tone, DetailTone::Bad);
    }
}
