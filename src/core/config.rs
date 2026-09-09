//! Configuration management for Scriba.

use anyhow::{Context, Result};
use dirs::home_dir;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Transcription mode configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TranscriptionMode {
    Local {
        /// Accepts both new `model` field and legacy `model_size` field from old configs.
        #[serde(alias = "model_size")]
        model: LocalModel,
    },
    /// Any OpenAI-compatible `/audio/transcriptions` endpoint: OpenAI itself,
    /// Groq, DeepInfra, or a self-hosted server.
    Api {
        api_key: String,
        /// API root, e.g. `https://api.groq.com/openai/v1`. None = OpenAI.
        #[serde(default)]
        base_url: Option<String>,
        /// Transcription model id. None = `DEFAULT_TRANSCRIPTION_MODEL`.
        #[serde(default)]
        model: Option<String>,
    },
}

impl TranscriptionMode {
    /// OpenAI transcription with the default model.
    pub fn api(api_key: impl Into<String>) -> Self {
        TranscriptionMode::Api {
            api_key: api_key.into(),
            base_url: None,
            model: None,
        }
    }
}

/// Default API root for cloud transcription.
pub const DEFAULT_TRANSCRIPTION_BASE_URL: &str = "https://api.openai.com/v1";
/// Default cloud transcription model.
pub const DEFAULT_TRANSCRIPTION_MODEL: &str = "whisper-1";

/// OpenAI transcription models offered in the UI (id, label).
pub const OPENAI_TRANSCRIPTION_MODELS: &[(&str, &str)] = &[
    ("whisper-1", "Whisper (whisper-1)"),
    ("gpt-4o-mini-transcribe", "GPT-4o mini transcribe"),
    ("gpt-4o-transcribe", "GPT-4o transcribe"),
    ("gpt-transcribe", "GPT transcribe"),
    ("gpt-4o-transcribe-diarize", "GPT-4o transcribe with speaker labels"),
];

/// A known host for OpenAI-compatible speech-to-text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptionPreset {
    pub name: &'static str,
    pub display: &'static str,
    pub base_url: &'static str,
    pub model: &'static str,
    pub env_var: &'static str,
}

/// Hosts with a drop-in `/audio/transcriptions` endpoint.
pub const TRANSCRIPTION_PRESETS: &[TranscriptionPreset] = &[
    TranscriptionPreset {
        name: "openai",
        display: "OpenAI",
        base_url: DEFAULT_TRANSCRIPTION_BASE_URL,
        model: DEFAULT_TRANSCRIPTION_MODEL,
        env_var: "OPENAI_API_KEY",
    },
    TranscriptionPreset {
        name: "groq",
        display: "Groq",
        base_url: "https://api.groq.com/openai/v1",
        model: "whisper-large-v3-turbo",
        env_var: "GROQ_API_KEY",
    },
    TranscriptionPreset {
        name: "deepinfra",
        display: "DeepInfra",
        base_url: "https://api.deepinfra.com/v1/openai",
        model: "openai/whisper-large-v3-turbo",
        env_var: "DEEPINFRA_API_KEY",
    },
];

impl TranscriptionPreset {
    pub fn by_name(name: &str) -> Option<&'static TranscriptionPreset> {
        let lower = name.to_lowercase();
        TRANSCRIPTION_PRESETS.iter().find(|p| p.name == lower)
    }

    pub fn for_url(url: &str) -> Option<&'static TranscriptionPreset> {
        let normalized = url.trim().trim_end_matches('/').to_lowercase();
        TRANSCRIPTION_PRESETS
            .iter()
            .find(|p| normalized.starts_with(&p.base_url.to_lowercase()))
    }

    pub fn names() -> Vec<&'static str> {
        TRANSCRIPTION_PRESETS.iter().map(|p| p.name).collect()
    }
}

/// Available local transcription models.
///
/// Each variant has a `serde(alias)` matching the old `LocalModelSize` enum value
/// so that existing config files (e.g. `"model_size": "Medium"`) deserialize correctly.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LocalModel {
    #[serde(alias = "Tiny")]
    WhisperTiny,
    #[serde(alias = "Base")]
    WhisperBase,
    #[serde(alias = "Small")]
    WhisperSmall,
    #[serde(alias = "Medium")]
    WhisperMedium,
    #[serde(alias = "Large")]
    WhisperLarge,
    #[serde(alias = "Turbo")]
    WhisperTurbo,
    SenseVoice,
    ParakeetTdt,
}

impl LocalModel {
    /// User-friendly display name for UI.
    pub fn display_name(&self) -> &str {
        match self {
            LocalModel::WhisperTiny => "Whisper Tiny",
            LocalModel::WhisperBase => "Whisper Base",
            LocalModel::WhisperSmall => "Whisper Small",
            LocalModel::WhisperMedium => "Whisper Medium",
            LocalModel::WhisperLarge => "Whisper Large",
            LocalModel::WhisperTurbo => "Whisper Turbo",
            LocalModel::SenseVoice => "SenseVoice",
            LocalModel::ParakeetTdt => "Parakeet TDT 0.6B",
        }
    }

    /// All models available for selection in the UI.
    pub fn all_models() -> &'static [LocalModel] {
        &[
            LocalModel::ParakeetTdt,
            LocalModel::WhisperTurbo,
            LocalModel::SenseVoice,
            LocalModel::WhisperTiny,
            LocalModel::WhisperBase,
            LocalModel::WhisperSmall,
            LocalModel::WhisperMedium,
            LocalModel::WhisperLarge,
        ]
    }

    /// The recommended default model for new users.
    pub fn recommended() -> Self {
        LocalModel::ParakeetTdt
    }
}

impl std::fmt::Display for LocalModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            LocalModel::WhisperTiny => "tiny",
            LocalModel::WhisperBase => "base",
            LocalModel::WhisperSmall => "small",
            LocalModel::WhisperMedium => "medium",
            LocalModel::WhisperLarge => "large",
            LocalModel::WhisperTurbo => "turbo",
            LocalModel::SenseVoice => "sensevoice",
            LocalModel::ParakeetTdt => "parakeet",
        };
        write!(f, "{}", s)
    }
}

impl std::str::FromStr for LocalModel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "tiny" | "whispertiny" => Ok(LocalModel::WhisperTiny),
            "base" | "whisperbase" => Ok(LocalModel::WhisperBase),
            "small" | "whispersmall" => Ok(LocalModel::WhisperSmall),
            "medium" | "whispermedium" => Ok(LocalModel::WhisperMedium),
            "large" | "whisperlarge" => Ok(LocalModel::WhisperLarge),
            "turbo" | "whisperturbo" => Ok(LocalModel::WhisperTurbo),
            "sensevoice" => Ok(LocalModel::SenseVoice),
            "parakeet" | "parakeettdt" => Ok(LocalModel::ParakeetTdt),
            _ => Err(anyhow::anyhow!("Invalid model: {}. Use: tiny, base, small, medium, large, turbo, sensevoice, parakeet", s)),
        }
    }
}

/// Legacy type alias for backward compatibility with code that references LocalModelSize.
#[deprecated(note = "use LocalModel instead")]
pub type LocalModelSize = LocalModel;

/// Main Scriba configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScribaConfig {
    pub transcription: TranscriptionMode,
    pub audio_settings: AudioSettings,
    /// Stores the last used API key to preserve it when switching modes.
    pub last_api_key: Option<String>,
    /// Knowledge extraction and enrichment settings.
    #[serde(default)]
    pub enrichment: EnrichmentConfig,
    /// Silence auto-stop settings.
    #[serde(default)]
    pub silence_auto_stop: SilenceAutoStopConfig,
    /// Automatic meeting detection settings (`scriba watch`).
    #[serde(default)]
    pub meeting_detection: MeetingDetectionConfig,
    /// Speaker diarization settings.
    #[serde(default)]
    pub diarization: DiarizationConfig,
    /// Whether the one-time "learn your voice" introduction was shown to a
    /// user who set up Scriba before voice profiles existed.
    #[serde(default)]
    pub voice_intro_seen: bool,
    /// Voice-activated recording settings (reserved for future use).
    #[serde(default, skip_serializing)]
    pub voice: VoiceConfig,
    /// Preserved local model when switching from Private to Cloud mode.
    #[serde(default, alias = "last_local_model_size")]
    pub last_local_model: Option<LocalModel>,
    /// Preserved cloud provider when switching from Cloud to Private mode.
    #[serde(default)]
    pub last_cloud_provider: Option<CloudProvider>,
    /// Preserved cloud transcription endpoint when switching to Private mode.
    #[serde(default)]
    pub last_transcription_base_url: Option<String>,
    /// Preserved cloud transcription model when switching to Private mode.
    #[serde(default)]
    pub last_transcription_model: Option<String>,
    /// Transcription API keys per host slot ("openai", "groq", "deepinfra", "custom").
    #[serde(default)]
    pub stt_api_keys: HashMap<String, String>,
    /// Transcription model per host slot.
    #[serde(default)]
    pub stt_models: HashMap<String, String>,
    /// Last endpoint typed for the "custom" transcription slot.
    #[serde(default)]
    pub stt_custom_base_url: Option<String>,
    /// Check for updates on launch (default: true).
    #[serde(default = "default_true")]
    pub check_for_updates: bool,
}

fn default_true() -> bool {
    true
}

/// Configuration for voice-activated recording ("Scriba Forever" mode).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Whether voice-activated recording mode is enabled.
    pub enabled: bool,
    /// RMS threshold for speech detection (VAD).
    pub vad_threshold: f32,
    /// Seconds of audio to keep in the rolling pre-buffer.
    pub pre_buffer_seconds: f32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            vad_threshold: 0.01,
            pre_buffer_seconds: 3.0,
        }
    }
}

/// Configuration for silence-based auto-stop during recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SilenceAutoStopConfig {
    /// Whether silence auto-stop is enabled.
    pub enabled: bool,
    /// Seconds of continuous silence before auto-stopping.
    pub timeout_seconds: u32,
}

impl Default for SilenceAutoStopConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_seconds: 60,
        }
    }
}

/// Configuration for automatic meeting detection (`scriba watch`).
///
/// The watcher detects the start and end of a meeting by watching whether a
/// microphone is in use by another process (e.g. Zoom or Google Meet opening
/// the mic for capture on join and releasing it on leave) rather than by
/// listening to audio levels, so casual talking in front of the computer never
/// triggers a false detection. When a meeting starts it can fire a desktop
/// notification and optionally auto-record it; the recording stops the moment
/// the meeting app releases the mic (where the OS supports per-process
/// attribution: macOS 14+, PulseAudio/PipeWire), with a silence timeout as a
/// fallback net.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingDetectionConfig {
    /// Whether meeting detection is enabled when running `scriba watch`.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// When true, automatically start recording when a meeting is detected and
    /// stop it when the meeting ends. When false, only fire a desktop
    /// notification.
    #[serde(default = "default_true")]
    pub auto_record: bool,
    /// Ask before recording: show a Record/Ignore dialog when a meeting is
    /// detected instead of recording immediately. An unanswered dialog counts
    /// as Ignore — nothing is recorded without explicit consent.
    #[serde(default = "default_true")]
    pub confirm_before_record: bool,
    /// Seconds before the Record/Ignore dialog gives up (and ignores the
    /// meeting).
    #[serde(default = "default_confirm_timeout")]
    pub confirm_timeout_seconds: u32,
    /// Fallback net: seconds of continuous silence after which an
    /// auto-recorded meeting's recording stops anyway. The primary stop signal
    /// is the meeting app releasing the mic; this only kicks in when that
    /// signal is unavailable (pre-14 macOS) or missed.
    #[serde(default = "default_min_silence")]
    pub min_silence_seconds: u32,
    /// Seconds after a recording finishes during which new detections are
    /// ignored. Guards against feedback loops with other recording tools on
    /// the same machine, whose reaction to Scriba's capture would otherwise
    /// look like a new meeting.
    #[serde(default = "default_cooldown")]
    pub cooldown_seconds: u32,
    /// Processes whose mic capture never counts as a meeting (case-insensitive
    /// substring match on the bundle ID / process name, e.g. "granola" or
    /// "us.zoom"). Add other recording tools here to avoid feedback loops.
    #[serde(default)]
    pub ignored_processes: Vec<String>,
    /// Preferred input device name (Linux: filter the PulseAudio/PipeWire
    /// source by name). On macOS all input devices are monitored and this is
    /// currently ignored.
    #[serde(default)]
    pub input_device: Option<String>,
}

fn default_confirm_timeout() -> u32 {
    15
}

fn default_min_silence() -> u32 {
    90
}

fn default_cooldown() -> u32 {
    120
}

impl Default for MeetingDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_record: true,
            confirm_before_record: true,
            confirm_timeout_seconds: default_confirm_timeout(),
            min_silence_seconds: default_min_silence(),
            cooldown_seconds: default_cooldown(),
            ignored_processes: Vec::new(),
            input_device: None,
        }
    }
}

/// Cloud LLM provider for enrichment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CloudProvider {
    Anthropic,
    OpenAI,
    Google,
    /// Any host speaking the OpenAI Chat Completions protocol (DeepInfra,
    /// OpenRouter, Groq, Together, vLLM, LM Studio, ...). The endpoint lives
    /// in `EnrichmentMode::Cloud::base_url`.
    OpenAICompatible,
}

impl std::fmt::Display for CloudProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloudProvider::Anthropic => write!(f, "anthropic"),
            CloudProvider::OpenAI => write!(f, "openai"),
            CloudProvider::Google => write!(f, "google"),
            CloudProvider::OpenAICompatible => write!(f, "custom"),
        }
    }
}

impl std::str::FromStr for CloudProvider {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let lower = s.to_lowercase();
        match lower.as_str() {
            "anthropic" | "claude" => Ok(CloudProvider::Anthropic),
            "openai" | "gpt" => Ok(CloudProvider::OpenAI),
            "google" | "gemini" => Ok(CloudProvider::Google),
            "custom" | "openai-compatible" | "compatible" => Ok(CloudProvider::OpenAICompatible),
            _ if EndpointPreset::by_name(&lower).is_some() => Ok(CloudProvider::OpenAICompatible),
            _ => Err(anyhow::anyhow!(
                "Invalid cloud provider: {}. Use: anthropic, openai, google, custom, or one of {}",
                s,
                EndpointPreset::names().join(", ")
            )),
        }
    }
}

impl CloudProvider {
    /// All cloud providers in the order the UI cycles through them.
    pub const ALL: [CloudProvider; 4] = [
        CloudProvider::Anthropic,
        CloudProvider::OpenAI,
        CloudProvider::Google,
        CloudProvider::OpenAICompatible,
    ];

    /// The provider that follows this one in the UI cycle.
    pub fn next(&self) -> CloudProvider {
        let idx = Self::ALL.iter().position(|p| p == self).unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()].clone()
    }

    /// Default model for this provider.
    pub fn default_model(&self) -> &str {
        match self {
            CloudProvider::Anthropic => "claude-sonnet-4-6",
            CloudProvider::OpenAI => "gpt-5.2",
            CloudProvider::Google => "gemini-2.5-flash",
            CloudProvider::OpenAICompatible => DEFAULT_COMPATIBLE_MODEL,
        }
    }

    /// Display name for UI.
    pub fn display_name(&self) -> &str {
        match self {
            CloudProvider::Anthropic => "Anthropic (Claude)",
            CloudProvider::OpenAI => "OpenAI (GPT)",
            CloudProvider::Google => "Google (Gemini)",
            CloudProvider::OpenAICompatible => "OpenAI-compatible",
        }
    }

    /// Env var name for this provider's API key.
    ///
    /// For `OpenAICompatible` this is the generic fallback; prefer
    /// [`EnrichmentConfig::api_key_env_var`], which knows the endpoint.
    pub fn env_var_name(&self) -> &str {
        match self {
            CloudProvider::Anthropic => "ANTHROPIC_API_KEY",
            CloudProvider::OpenAI => "OPENAI_API_KEY",
            CloudProvider::Google => "GOOGLE_API_KEY",
            CloudProvider::OpenAICompatible => "SCRIBA_LLM_API_KEY",
        }
    }

    /// Whether this provider needs a user-supplied base URL.
    pub fn uses_custom_endpoint(&self) -> bool {
        matches!(self, CloudProvider::OpenAICompatible)
    }

    /// Curated list of models for this provider.
    pub fn available_models(&self) -> Vec<ModelDef> {
        match self {
            CloudProvider::Anthropic => vec![
                ModelDef { display_name: "Claude Opus 4.6".into(), model_id: "claude-opus-4-6".into() },
                ModelDef { display_name: "Claude Sonnet 4.6".into(), model_id: "claude-sonnet-4-6".into() },
                ModelDef { display_name: "Claude Haiku 4.5".into(), model_id: "claude-haiku-4-5-20251001".into() },
            ],
            CloudProvider::OpenAI => vec![
                ModelDef { display_name: "GPT-5.2".into(), model_id: "gpt-5.2".into() },
                ModelDef { display_name: "GPT-5.1 Mini".into(), model_id: "gpt-5.1-mini".into() },
                ModelDef { display_name: "o3".into(), model_id: "o3".into() },
                ModelDef { display_name: "o4-mini".into(), model_id: "o4-mini".into() },
            ],
            CloudProvider::Google => vec![
                ModelDef { display_name: "Gemini 2.5 Pro".into(), model_id: "gemini-2.5-pro".into() },
                ModelDef { display_name: "Gemini 2.5 Flash".into(), model_id: "gemini-2.5-flash".into() },
                ModelDef { display_name: "Gemini 2.5 Flash-Lite".into(), model_id: "gemini-2.5-flash-lite".into() },
                ModelDef { display_name: "Gemini 3.1 Pro Preview".into(), model_id: "gemini-3.1-pro-preview".into() },
            ],
            // Model catalogs differ per host; the UI lists them live from `{base_url}/models`.
            CloudProvider::OpenAICompatible => vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelDef {
    pub display_name: String,
    pub model_id: String,
}

/// Default Ollama server for Private mode.
pub const DEFAULT_OLLAMA_ENDPOINT: &str = "http://localhost:11434";
/// Default Ollama model for Private mode: current, tool-capable, fits a 16 GB machine.
pub const DEFAULT_OLLAMA_MODEL: &str = "gemma4:12b";

/// Default endpoint for `CloudProvider::OpenAICompatible` when none is configured.
pub const DEFAULT_COMPATIBLE_BASE_URL: &str = "https://api.deepinfra.com/v1/openai";
/// Default model for `CloudProvider::OpenAICompatible` (open-weight, tool-capable, hosted on DeepInfra).
pub const DEFAULT_COMPATIBLE_MODEL: &str = "Qwen/Qwen3.5-397B-A17B";

/// A well-known OpenAI-compatible host, so users can pick it by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointPreset {
    /// Short name accepted by the CLI (`scriba config set-provider deepinfra`).
    pub name: &'static str,
    /// Human-readable name.
    pub display: &'static str,
    /// Base URL of the OpenAI-compatible API.
    pub base_url: &'static str,
    /// Conventional environment variable for the API key.
    pub env_var: &'static str,
}

/// Known OpenAI-compatible hosts. Any other URL works too; these only supply
/// defaults and nicer labels.
pub const ENDPOINT_PRESETS: &[EndpointPreset] = &[
    EndpointPreset {
        name: "deepinfra",
        display: "DeepInfra",
        base_url: "https://api.deepinfra.com/v1/openai",
        env_var: "DEEPINFRA_API_KEY",
    },
    EndpointPreset {
        name: "openrouter",
        display: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        env_var: "OPENROUTER_API_KEY",
    },
    EndpointPreset {
        name: "groq",
        display: "Groq",
        base_url: "https://api.groq.com/openai/v1",
        env_var: "GROQ_API_KEY",
    },
    EndpointPreset {
        name: "together",
        display: "Together",
        base_url: "https://api.together.xyz/v1",
        env_var: "TOGETHER_API_KEY",
    },
];

impl EndpointPreset {
    /// Look up a preset by its short name (case-insensitive).
    pub fn by_name(name: &str) -> Option<&'static EndpointPreset> {
        let lower = name.to_lowercase();
        ENDPOINT_PRESETS.iter().find(|p| p.name == lower)
    }

    /// Find the preset whose base URL matches the given endpoint (ignoring
    /// trailing slashes and scheme case).
    pub fn for_url(url: &str) -> Option<&'static EndpointPreset> {
        let normalized = url.trim().trim_end_matches('/').to_lowercase();
        ENDPOINT_PRESETS
            .iter()
            .find(|p| normalized.starts_with(&p.base_url.to_lowercase()))
    }

    /// Short names of all presets, for help text.
    pub fn names() -> Vec<&'static str> {
        ENDPOINT_PRESETS.iter().map(|p| p.name).collect()
    }
}

/// Enrichment mode configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnrichmentMode {
    Cloud {
        provider: CloudProvider,
        api_key: String,
        /// None = use provider default model.
        #[serde(default)]
        model: Option<String>,
        /// Base URL override. Required in spirit for `OpenAICompatible`
        /// (falls back to `DEFAULT_COMPATIBLE_BASE_URL`); optional proxy
        /// override for the first-party providers.
        #[serde(default)]
        base_url: Option<String>,
    },
    Local {
        ollama_endpoint: String,
        ollama_model: String,
    },
}

impl Default for EnrichmentMode {
    fn default() -> Self {
        EnrichmentMode::Cloud {
            provider: CloudProvider::Anthropic,
            api_key: String::new(),
            model: None,
            base_url: None,
        }
    }
}

/// Configuration for knowledge extraction and enrichment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentConfig {
    /// Whether automatic enrichment is enabled after transcription.
    pub enabled: bool,
    /// Enrichment provider mode.
    #[serde(default)]
    pub mode: EnrichmentMode,

    // Legacy fields — used only for migration from old config format.
    // Kept with serde(default) so old configs deserialize, then converted to EnrichmentMode::Local.
    #[serde(default, skip_serializing)]
    ollama_endpoint: Option<String>,
    #[serde(default, skip_serializing)]
    ollama_model: Option<String>,

    /// Per-provider API keys so switching providers doesn't lose keys.
    #[serde(default)]
    pub cloud_api_keys: HashMap<String, String>,
    /// Per-provider model selections so switching providers doesn't lose model choice.
    #[serde(default)]
    pub cloud_models: HashMap<String, String>,
    /// Per-provider base URLs so switching providers doesn't lose the endpoint.
    #[serde(default)]
    pub cloud_base_urls: HashMap<String, String>,

    /// Preserved Ollama endpoint so cycling away from Local doesn't lose it.
    #[serde(default)]
    pub last_ollama_endpoint: Option<String>,
    /// Preserved Ollama model so cycling away from Local doesn't lose it.
    #[serde(default)]
    pub last_ollama_model: Option<String>,

    /// Confidence threshold for automatic entity linking (0.0-1.0).
    pub auto_link_threshold: f32,
    /// Whether to evolve the world description after each enrichment.
    #[serde(default = "default_evolve_world")]
    pub evolve_world: bool,
    /// Whether to search the web for unresolved entities.
    #[serde(default = "default_search_enabled")]
    pub search_enabled: bool,
    /// Maximum number of web search results per unresolved entity.
    #[serde(default = "default_max_search_results")]
    pub max_search_results: usize,
}

fn default_evolve_world() -> bool {
    true
}

fn default_search_enabled() -> bool {
    true
}

fn default_max_search_results() -> usize {
    3
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: EnrichmentMode::default(),
            ollama_endpoint: None,
            ollama_model: None,
            cloud_api_keys: HashMap::new(),
            cloud_models: HashMap::new(),
            cloud_base_urls: HashMap::new(),
            last_ollama_endpoint: None,
            last_ollama_model: None,
            auto_link_threshold: 0.8,
            evolve_world: true,
            search_enabled: true,
            max_search_results: 3,
        }
    }
}

impl EnrichmentConfig {
    /// Migrate legacy config fields to the new mode format.
    /// Called after deserialization.
    pub fn migrate_legacy(&mut self) {
        // If legacy fields are present and mode is the default Cloud with empty key,
        // this was an old config — convert to Local mode.
        if let Some(endpoint) = self.ollama_endpoint.take() {
            let model = self.ollama_model.take().unwrap_or_else(|| DEFAULT_OLLAMA_MODEL.to_string());
            // Only migrate if mode looks like the default (empty cloud key)
            if matches!(&self.mode, EnrichmentMode::Cloud { api_key, .. } if api_key.is_empty()) {
                self.mode = EnrichmentMode::Local {
                    ollama_endpoint: endpoint,
                    ollama_model: model,
                };
            }
        }

        // Seed cloud_api_keys from the existing api_key field if the map is
        // empty (first run after upgrade).  This ensures the key is attributed
        // to the correct provider instead of being carried to the wrong one on
        // the first provider cycle.
        if self.cloud_api_keys.is_empty() {
            if let EnrichmentMode::Cloud { provider, api_key, .. } = &self.mode {
                if !api_key.is_empty() {
                    self.cloud_api_keys.insert(provider.to_string(), api_key.clone());
                }
            }
        }
    }

    /// Storage slot for a provider: the provider name, except OpenAI-compatible
    /// hosts get their preset name ("deepinfra", "groq", ...) or "custom", so
    /// keys and models for different hosts do not overwrite each other.
    pub fn provider_slot(provider: &CloudProvider, base_url: Option<&str>) -> String {
        if provider.uses_custom_endpoint() {
            let url = base_url
                .map(str::to_string)
                .unwrap_or_else(|| DEFAULT_COMPATIBLE_BASE_URL.to_string());
            EndpointPreset::for_url(&url)
                .map(|p| p.name.to_string())
                .unwrap_or_else(|| "custom".to_string())
        } else {
            provider.to_string()
        }
    }

    /// Slot of the current cloud configuration, if any.
    pub fn current_slot(&self) -> Option<String> {
        match &self.mode {
            EnrichmentMode::Cloud { provider, base_url, .. } => {
                Some(Self::provider_slot(provider, base_url.as_deref()))
            }
            EnrichmentMode::Local { .. } => None,
        }
    }

    /// Remember the current cloud key, model and endpoint under their slot.
    pub fn remember_cloud_settings(&mut self) {
        if let EnrichmentMode::Cloud { provider, api_key, model, base_url } = self.mode.clone() {
            let slot = Self::provider_slot(&provider, base_url.as_deref());
            self.save_key_for_slot(&slot, &api_key);
            self.save_model_for_slot(&slot, &model);
            if slot == "custom" {
                self.save_base_url_for_slot(&slot, &base_url);
            }
        }
    }

    pub fn save_key_for_slot(&mut self, slot: &str, key: &str) {
        if key.trim().is_empty() {
            self.cloud_api_keys.remove(slot);
        } else {
            self.cloud_api_keys.insert(slot.to_string(), key.trim().to_string());
        }
    }

    /// Key for a slot; OpenAI-compatible hosts fall back to the shared
    /// "custom" slot older configs used.
    pub fn load_key_for_slot(&self, slot: &str) -> String {
        self.cloud_api_keys
            .get(slot)
            .or_else(|| if slot != "custom" { self.cloud_api_keys.get("custom") } else { None })
            .cloned()
            .unwrap_or_default()
    }

    pub fn save_model_for_slot(&mut self, slot: &str, model: &Option<String>) {
        match model.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
            Some(m) => {
                self.cloud_models.insert(slot.to_string(), m.to_string());
            }
            None => {
                self.cloud_models.remove(slot);
            }
        }
    }

    pub fn load_model_for_slot(&self, slot: &str) -> Option<String> {
        self.cloud_models.get(slot).cloned()
    }

    pub fn save_base_url_for_slot(&mut self, slot: &str, base_url: &Option<String>) {
        match base_url.as_deref().map(str::trim).filter(|u| !u.is_empty()) {
            Some(u) => {
                self.cloud_base_urls.insert(slot.to_string(), u.to_string());
            }
            None => {
                self.cloud_base_urls.remove(slot);
            }
        }
    }

    pub fn load_base_url_for_slot(&self, slot: &str) -> Option<String> {
        self.cloud_base_urls.get(slot).cloned()
    }

    /// Save an API key for a specific provider into the per-provider map.
    pub fn save_key_for_provider(&mut self, provider: &CloudProvider, key: &str) {
        self.save_key_for_slot(&provider.to_string(), key);
    }

    /// Load a previously-stored API key for a specific provider.
    pub fn load_key_for_provider(&self, provider: &CloudProvider) -> String {
        self.cloud_api_keys
            .get(&provider.to_string())
            .cloned()
            .unwrap_or_default()
    }

    /// Save a model selection for a specific provider into the per-provider map.
    pub fn save_model_for_provider(&mut self, provider: &CloudProvider, model: &Option<String>) {
        if let Some(m) = model {
            self.cloud_models.insert(provider.to_string(), m.clone());
        } else {
            self.cloud_models.remove(&provider.to_string());
        }
    }

    /// Load a previously-stored model selection for a specific provider.
    /// Returns `None` if no explicit selection was saved (use provider default).
    pub fn load_model_for_provider(&self, provider: &CloudProvider) -> Option<String> {
        self.cloud_models.get(&provider.to_string()).cloned()
    }

    /// Save a base URL for a specific provider into the per-provider map.
    pub fn save_base_url_for_provider(&mut self, provider: &CloudProvider, base_url: &Option<String>) {
        match base_url.as_deref().map(str::trim).filter(|u| !u.is_empty()) {
            Some(u) => {
                self.cloud_base_urls.insert(provider.to_string(), u.to_string());
            }
            None => {
                self.cloud_base_urls.remove(&provider.to_string());
            }
        }
    }

    /// Load a previously-stored base URL for a specific provider.
    pub fn load_base_url_for_provider(&self, provider: &CloudProvider) -> Option<String> {
        self.cloud_base_urls.get(&provider.to_string()).cloned()
    }

    /// Explicitly configured base URL, if any.
    pub fn base_url(&self) -> Option<&str> {
        match &self.mode {
            EnrichmentMode::Cloud { base_url: Some(u), .. } if !u.trim().is_empty() => Some(u.trim()),
            _ => None,
        }
    }

    /// Base URL that will actually be used: the explicit override, or the
    /// default host for `OpenAICompatible`. `None` means the provider's own
    /// first-party endpoint.
    pub fn effective_base_url(&self) -> Option<String> {
        if let Some(u) = self.base_url() {
            return Some(u.to_string());
        }
        match &self.mode {
            EnrichmentMode::Cloud { provider, .. } if provider.uses_custom_endpoint() => {
                Some(DEFAULT_COMPATIBLE_BASE_URL.to_string())
            }
            _ => None,
        }
    }

    /// Set the base URL (only effective in Cloud mode). Empty clears it.
    pub fn set_base_url(&mut self, url: Option<String>) {
        if let EnrichmentMode::Cloud { base_url, .. } = &mut self.mode {
            *base_url = url.map(|u| u.trim().trim_end_matches('/').to_string()).filter(|u| !u.is_empty());
        }
    }

    /// Whether the current provider needs a user-supplied endpoint.
    pub fn has_custom_endpoint(&self) -> bool {
        matches!(&self.mode, EnrichmentMode::Cloud { provider, .. } if provider.uses_custom_endpoint())
    }

    /// Environment variable consulted for the API key: the known host's
    /// conventional variable when the endpoint matches a preset, otherwise the
    /// provider's own.
    pub fn api_key_env_var(&self) -> Option<String> {
        match &self.mode {
            EnrichmentMode::Cloud { provider, .. } => {
                if provider.uses_custom_endpoint()
                    && let Some(preset) = self.effective_base_url().as_deref().and_then(EndpointPreset::for_url)
                {
                    return Some(preset.env_var.to_string());
                }
                Some(provider.env_var_name().to_string())
            }
            EnrichmentMode::Local { .. } => None,
        }
    }

    /// Get the current cloud provider, if in cloud mode.
    pub fn cloud_provider(&self) -> Option<&CloudProvider> {
        match &self.mode {
            EnrichmentMode::Cloud { provider, .. } => Some(provider),
            _ => None,
        }
    }

    /// Get the provider display name. For OpenAI-compatible endpoints this
    /// names the host when it is a known preset (e.g. "DeepInfra").
    pub fn provider_display_name(&self) -> String {
        match &self.mode {
            EnrichmentMode::Cloud { provider, .. } if provider.uses_custom_endpoint() => {
                match self.effective_base_url().as_deref().and_then(EndpointPreset::for_url) {
                    Some(preset) => format!("{} (OpenAI-compatible)", preset.display),
                    None => provider.display_name().to_string(),
                }
            }
            EnrichmentMode::Cloud { provider, .. } => provider.display_name().to_string(),
            EnrichmentMode::Local { .. } => "Ollama (Local)".to_string(),
        }
    }

    /// Whether the current mode needs an API key.
    pub fn needs_api_key(&self) -> bool {
        matches!(&self.mode, EnrichmentMode::Cloud { .. })
    }

    /// Get the API key if in cloud mode.
    pub fn api_key(&self) -> Option<&str> {
        match &self.mode {
            EnrichmentMode::Cloud { api_key, .. } if !api_key.is_empty() => Some(api_key),
            _ => None,
        }
    }

    /// Resolve the effective API key: config value > env var.
    pub fn resolve_api_key(&self) -> Option<String> {
        match &self.mode {
            EnrichmentMode::Cloud { api_key, .. } => {
                if !api_key.is_empty() {
                    return Some(api_key.clone());
                }
                // Fallback to env var
                let env = self.api_key_env_var()?;
                std::env::var(env).ok().filter(|k| !k.trim().is_empty())
            }
            EnrichmentMode::Local { .. } => None,
        }
    }

    /// Get the model name in use (explicit or provider default).
    pub fn model_name(&self) -> &str {
        match &self.mode {
            EnrichmentMode::Cloud { provider, model, .. } => {
                model.as_deref().unwrap_or_else(|| provider.default_model())
            }
            EnrichmentMode::Local { ollama_model, .. } => ollama_model,
        }
    }

    /// Whether the mode is local (Ollama).
    pub fn is_local(&self) -> bool {
        matches!(&self.mode, EnrichmentMode::Local { .. })
    }

    /// Get the Ollama endpoint (only meaningful in Local mode).
    /// Returns a default if not in Local mode.
    pub fn ollama_endpoint(&self) -> String {
        match &self.mode {
            EnrichmentMode::Local { ollama_endpoint, .. } => ollama_endpoint.clone(),
            _ => DEFAULT_OLLAMA_ENDPOINT.to_string(),
        }
    }

    /// Get the Ollama model (only meaningful in Local mode).
    /// Returns a default if not in Local mode.
    pub fn ollama_model(&self) -> String {
        match &self.mode {
            EnrichmentMode::Local { ollama_model, .. } => ollama_model.clone(),
            _ => DEFAULT_OLLAMA_MODEL.to_string(),
        }
    }

    /// Set the Ollama model (only effective in Local mode).
    pub fn set_ollama_model(&mut self, model: String) {
        if let EnrichmentMode::Local { ollama_model, .. } = &mut self.mode {
            *ollama_model = model;
        }
    }

    /// Set the Ollama endpoint (only effective in Local mode).
    pub fn set_ollama_endpoint(&mut self, endpoint: String) {
        if let EnrichmentMode::Local { ollama_endpoint, .. } = &mut self.mode {
            *ollama_endpoint = endpoint;
        }
    }
}

/// Configuration for speaker diarization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationConfig {
    /// Whether to label who said what after local transcription.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Reserved: upper bound on detected speakers (clustering is automatic today).
    #[serde(default = "default_max_speakers")]
    pub max_speakers: u32,
}

fn default_max_speakers() -> u32 {
    6
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_speakers: default_max_speakers(),
        }
    }
}

/// Audio recording settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSettings {
    pub sample_rate: u32,
    pub bitrate: u32,
    pub channels: u16,
    pub speech_optimized: bool,
    /// Preferred input device name. When set, Scriba will try to use this device
    /// instead of the system default. Useful for headphone/external microphones.
    /// Use `scriba health --verbose` to list available devices.
    #[serde(default)]
    pub input_device: Option<String>,
    /// System audio loopback device for capturing the other side of a call.
    /// On macOS, set to "screencapturekit" (or omit) to use the native
    /// ScreenCaptureKit API. On Linux, set to a PulseAudio/PipeWire monitor
    /// source name (e.g. "Monitor of Built-in Audio").
    /// Use `scriba health --verbose` to detect available loopback sources.
    #[serde(default)]
    pub loopback_device: Option<String>,
}

impl Default for ScribaConfig {
    fn default() -> Self {
        Self {
            transcription: TranscriptionMode::Local {
                model: LocalModel::ParakeetTdt,
            },
            audio_settings: AudioSettings {
                sample_rate: 48000,
                bitrate: 128,
                channels: 1,
                speech_optimized: true,
                input_device: None,
                loopback_device: None,
            },
            last_api_key: None,
            enrichment: EnrichmentConfig::default(),
            silence_auto_stop: SilenceAutoStopConfig::default(),
            meeting_detection: MeetingDetectionConfig::default(),
            diarization: DiarizationConfig::default(),
            voice_intro_seen: false,
            voice: VoiceConfig::default(),
            last_local_model: None,
            last_cloud_provider: None,
            last_transcription_base_url: None,
            last_transcription_model: None,
            stt_api_keys: HashMap::new(),
            stt_models: HashMap::new(),
            stt_custom_base_url: None,
            check_for_updates: true,
        }
    }
}

impl ScribaConfig {
    /// Get the path to the configuration file.
    pub fn config_path() -> Result<PathBuf> {
        let home = home_dir().context("Failed to get home directory")?;
        Ok(home.join("scriba_recordings").join("config.json"))
    }

    /// Load configuration from disk, creating default if it doesn't exist.
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if !config_path.exists() {
            let config = Self::default();
            config.save()?;
            return Ok(config);
        }

        let content = fs::read_to_string(&config_path).context("Failed to read config file")?;
        let mut config: Self = serde_json::from_str(&content).context("Failed to parse config file")?;

        // Migrate legacy enrichment config (ollama_endpoint/ollama_model fields → EnrichmentMode::Local)
        config.enrichment.migrate_legacy();

        Ok(config)
    }

    /// Save configuration to disk.
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent).context("Failed to create config directory")?;
        }

        let content = serde_json::to_string_pretty(self).context("Failed to serialize config")?;
        fs::write(&config_path, content).context("Failed to write config file")?;

        Ok(())
    }

    /// Set the transcription mode and save.
    pub fn set_transcription_mode(&mut self, mode: TranscriptionMode) -> Result<()> {
        // Save current API settings if switching away from API mode
        if let TranscriptionMode::Api { api_key, base_url, model } = &self.transcription {
            if !api_key.is_empty() {
                self.last_api_key = Some(api_key.clone());
            }
            self.last_transcription_base_url = base_url.clone();
            self.last_transcription_model = model.clone();
        }
        // Save current local model if switching away from Local mode
        if let TranscriptionMode::Local { model } = &self.transcription {
            self.last_local_model = Some(*model);
        }

        self.transcription = mode;
        self.save()
    }

    /// Check if in Private mode (local STT + Ollama).
    pub fn is_private_mode(&self) -> bool {
        matches!(self.transcription, TranscriptionMode::Local { .. })
    }

    /// Get the API key if in API mode.
    pub fn get_api_key(&self) -> Option<&str> {
        match &self.transcription {
            TranscriptionMode::Api { api_key, .. } => Some(api_key),
            _ => None,
        }
    }

    /// Effective cloud transcription API root.
    pub fn transcription_base_url(&self) -> String {
        match &self.transcription {
            TranscriptionMode::Api { base_url: Some(u), .. } if !u.trim().is_empty() => {
                u.trim().trim_end_matches('/').to_string()
            }
            _ => DEFAULT_TRANSCRIPTION_BASE_URL.to_string(),
        }
    }

    /// Effective cloud transcription model.
    pub fn transcription_model(&self) -> String {
        match &self.transcription {
            TranscriptionMode::Api { model: Some(m), .. } if !m.trim().is_empty() => m.trim().to_string(),
            _ => DEFAULT_TRANSCRIPTION_MODEL.to_string(),
        }
    }

    /// Environment variable consulted for the transcription key.
    pub fn transcription_api_key_env(&self) -> String {
        TranscriptionPreset::for_url(&self.transcription_base_url())
            .map(|p| p.env_var)
            .unwrap_or("SCRIBA_STT_API_KEY")
            .to_string()
    }

    /// Transcription key: config value, else the host's environment variable.
    pub fn resolve_transcription_api_key(&self) -> Option<String> {
        match &self.transcription {
            TranscriptionMode::Api { api_key, .. } if !api_key.trim().is_empty() => Some(api_key.clone()),
            TranscriptionMode::Api { .. } => std::env::var(self.transcription_api_key_env())
                .ok()
                .filter(|k| !k.trim().is_empty()),
            TranscriptionMode::Local { .. } => None,
        }
    }

    /// Human-readable transcription host ("OpenAI", "Groq", or the domain).
    pub fn transcription_host_display(&self) -> String {
        let url = self.transcription_base_url();
        if let Some(preset) = TranscriptionPreset::for_url(&url) {
            return preset.display.to_string();
        }
        url.split("//")
            .nth(1)
            .and_then(|rest| rest.split('/').next())
            .unwrap_or(&url)
            .to_string()
    }

    /// Host slot of the current cloud transcription endpoint: a preset name
    /// ("openai", "groq", "deepinfra") or "custom". `None` in Local mode.
    pub fn stt_slot(&self) -> Option<String> {
        match &self.transcription {
            TranscriptionMode::Local { .. } => None,
            TranscriptionMode::Api { .. } => Some(Self::stt_slot_for_url(&self.transcription_base_url())),
        }
    }

    /// Slot name for a transcription endpoint URL.
    pub fn stt_slot_for_url(url: &str) -> String {
        TranscriptionPreset::for_url(url)
            .map(|p| p.name.to_string())
            .unwrap_or_else(|| "custom".to_string())
    }

    /// Remember the current cloud transcription key, model and custom endpoint
    /// under their host slot so switching hosts never loses them.
    pub fn remember_stt_settings(&mut self) {
        if let TranscriptionMode::Api { api_key, model, base_url } = &self.transcription {
            let slot = Self::stt_slot_for_url(&self.transcription_base_url());
            if api_key.trim().is_empty() {
                self.stt_api_keys.remove(&slot);
            } else {
                self.stt_api_keys.insert(slot.clone(), api_key.clone());
            }
            match model.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
                Some(m) => {
                    self.stt_models.insert(slot.clone(), m.to_string());
                }
                None => {
                    self.stt_models.remove(&slot);
                }
            }
            if slot == "custom" {
                self.stt_custom_base_url = base_url.clone();
            }
        }
    }

    /// Stored transcription key for a host slot.
    pub fn stt_key_for_slot(&self, slot: &str) -> Option<String> {
        self.stt_api_keys.get(slot).cloned().filter(|k| !k.is_empty())
    }

    /// Stored transcription model for a host slot.
    pub fn stt_model_for_slot(&self, slot: &str) -> Option<String> {
        self.stt_models.get(slot).cloned().filter(|m| !m.is_empty())
    }

    /// API mode with `key`, keeping the configured (or last used) endpoint and model.
    pub fn api_mode_with_key(&self, key: String) -> TranscriptionMode {
        let (base_url, model) = match &self.transcription {
            TranscriptionMode::Api { base_url, model, .. } => (base_url.clone(), model.clone()),
            TranscriptionMode::Local { .. } => (
                self.last_transcription_base_url.clone(),
                self.last_transcription_model.clone(),
            ),
        };
        TranscriptionMode::Api { api_key: key, base_url, model }
    }

    /// Set the cloud transcription endpoint (only effective in API mode; empty clears).
    pub fn set_transcription_base_url(&mut self, url: Option<String>) {
        if let TranscriptionMode::Api { base_url, .. } = &mut self.transcription {
            *base_url = url
                .map(|u| u.trim().trim_end_matches('/').to_string())
                .filter(|u| !u.is_empty());
        }
    }

    /// Set the cloud transcription model (only effective in API mode; empty clears).
    pub fn set_transcription_model(&mut self, model: Option<String>) {
        if let TranscriptionMode::Api { model: m, .. } = &mut self.transcription {
            *m = model.map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        }
    }

    /// Get the local model if in local mode.
    pub fn get_local_model(&self) -> Option<LocalModel> {
        match &self.transcription {
            TranscriptionMode::Local { model } => Some(*model),
            _ => None,
        }
    }

    /// Backward-compatible alias.
    #[deprecated(note = "use get_local_model() instead")]
    #[allow(deprecated)]
    pub fn get_local_model_size(&self) -> Option<LocalModel> {
        self.get_local_model()
    }
}

/// Resolve transcription mode from CLI flags and config.
/// Priority: force_local > api_key > model > config default
pub fn resolve_transcription_mode(
    force_local: bool,
    model: Option<LocalModel>,
    api_key: Option<String>,
    config: &ScribaConfig,
) -> Result<TranscriptionMode> {
    if force_local {
        let model = model.unwrap_or(LocalModel::ParakeetTdt);
        return Ok(TranscriptionMode::Local { model });
    }

    if let Some(key) = api_key {
        return Ok(config.api_mode_with_key(key));
    }

    if let Some(model) = model {
        return Ok(TranscriptionMode::Local { model });
    }

    // Use config default
    Ok(config.transcription.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::field_reassign_with_default)]
    fn cloud(provider: CloudProvider, base_url: Option<&str>) -> EnrichmentConfig {
        let mut config = EnrichmentConfig::default();
        config.mode = EnrichmentMode::Cloud {
            provider,
            api_key: String::new(),
            model: None,
            base_url: base_url.map(str::to_string),
        };
        config
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn transcription_defaults_and_overrides() {
        let mut config = ScribaConfig::default();
        config.transcription = TranscriptionMode::api("sk-openai");
        assert_eq!(config.transcription_base_url(), DEFAULT_TRANSCRIPTION_BASE_URL);
        assert_eq!(config.transcription_model(), DEFAULT_TRANSCRIPTION_MODEL);
        assert_eq!(config.transcription_host_display(), "OpenAI");
        assert_eq!(config.transcription_api_key_env(), "OPENAI_API_KEY");
        assert_eq!(config.resolve_transcription_api_key().as_deref(), Some("sk-openai"));

        config.set_transcription_base_url(Some("https://api.groq.com/openai/v1/".into()));
        config.set_transcription_model(Some(" whisper-large-v3-turbo ".into()));
        assert_eq!(config.transcription_base_url(), "https://api.groq.com/openai/v1");
        assert_eq!(config.transcription_model(), "whisper-large-v3-turbo");
        assert_eq!(config.transcription_host_display(), "Groq");
        assert_eq!(config.transcription_api_key_env(), "GROQ_API_KEY");

        config.set_transcription_base_url(Some("http://stt.local:8000/v1".into()));
        assert_eq!(config.transcription_host_display(), "stt.local:8000");
        assert_eq!(config.transcription_api_key_env(), "SCRIBA_STT_API_KEY");
        config.set_transcription_model(Some("  ".into()));
        assert_eq!(config.transcription_model(), DEFAULT_TRANSCRIPTION_MODEL);
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn api_mode_with_key_keeps_endpoint_across_mode_switches() {
        let mut config = ScribaConfig::default();
        config.transcription = TranscriptionMode::Api {
            api_key: "k".into(),
            base_url: Some("https://api.groq.com/openai/v1".into()),
            model: Some("whisper-large-v3-turbo".into()),
        };
        // Switching to Local must remember the endpoint/model (without saving to disk).
        if let TranscriptionMode::Api { base_url, model, .. } = &config.transcription {
            config.last_transcription_base_url = base_url.clone();
            config.last_transcription_model = model.clone();
        }
        config.transcription = TranscriptionMode::Local { model: LocalModel::ParakeetTdt };
        let restored = config.api_mode_with_key("k2".into());
        match restored {
            TranscriptionMode::Api { api_key, base_url, model } => {
                assert_eq!(api_key, "k2");
                assert_eq!(base_url.as_deref(), Some("https://api.groq.com/openai/v1"));
                assert_eq!(model.as_deref(), Some("whisper-large-v3-turbo"));
            }
            _ => panic!("expected Api mode"),
        }
    }

    #[test]
    fn legacy_api_transcription_config_deserializes() {
        let json = r#"{"Api":{"api_key":"sk"}}"#;
        let mode: TranscriptionMode = serde_json::from_str(json).unwrap();
        assert!(matches!(mode, TranscriptionMode::Api { base_url: None, model: None, .. }));
        assert_eq!(TranscriptionPreset::by_name("Groq").unwrap().model, "whisper-large-v3-turbo");
        assert!(TranscriptionPreset::for_url("https://api.deepinfra.com/v1/openai/").is_some());
    }

    #[test]
    fn slots_keep_compatible_hosts_apart() {
        assert_eq!(EnrichmentConfig::provider_slot(&CloudProvider::Anthropic, None), "anthropic");
        assert_eq!(
            EnrichmentConfig::provider_slot(&CloudProvider::OpenAICompatible, Some("https://api.groq.com/openai/v1")),
            "groq"
        );
        assert_eq!(EnrichmentConfig::provider_slot(&CloudProvider::OpenAICompatible, None), "deepinfra");
        assert_eq!(
            EnrichmentConfig::provider_slot(&CloudProvider::OpenAICompatible, Some("http://box:8000/v1")),
            "custom"
        );
        let mut e = EnrichmentConfig::default();
        e.save_key_for_slot("custom", "old-shared-key");
        assert_eq!(e.load_key_for_slot("groq"), "old-shared-key", "legacy shared slot is a fallback");
        e.save_key_for_slot("groq", "gsk");
        assert_eq!(e.load_key_for_slot("groq"), "gsk");
        assert_eq!(ScribaConfig::stt_slot_for_url("https://api.deepinfra.com/v1/openai"), "deepinfra");
        assert_eq!(ScribaConfig::stt_slot_for_url("http://stt.local/v1"), "custom");
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn stt_settings_are_remembered_per_host() {
        let mut config = ScribaConfig::default();
        config.transcription = TranscriptionMode::Api {
            api_key: "gsk".into(),
            base_url: Some("https://api.groq.com/openai/v1".into()),
            model: Some("whisper-large-v3-turbo".into()),
        };
        config.remember_stt_settings();
        assert_eq!(config.stt_slot().as_deref(), Some("groq"));
        assert_eq!(config.stt_key_for_slot("groq").as_deref(), Some("gsk"));
        assert_eq!(config.stt_model_for_slot("groq").as_deref(), Some("whisper-large-v3-turbo"));
        assert!(config.stt_key_for_slot("openai").is_none());
    }

    #[test]
    fn provider_names_parse() {
        assert_eq!("claude".parse::<CloudProvider>().unwrap(), CloudProvider::Anthropic);
        assert_eq!("custom".parse::<CloudProvider>().unwrap(), CloudProvider::OpenAICompatible);
        assert_eq!("DeepInfra".parse::<CloudProvider>().unwrap(), CloudProvider::OpenAICompatible);
        assert_eq!("groq".parse::<CloudProvider>().unwrap(), CloudProvider::OpenAICompatible);
        assert!("nope".parse::<CloudProvider>().is_err());
    }

    #[test]
    fn provider_cycle_visits_every_provider_once() {
        let mut seen = vec![CloudProvider::Anthropic];
        let mut cur = CloudProvider::Anthropic;
        for _ in 0..CloudProvider::ALL.len() - 1 {
            cur = cur.next();
            assert!(!seen.contains(&cur));
            seen.push(cur.clone());
        }
        assert_eq!(cur.next(), CloudProvider::Anthropic);
    }

    #[test]
    fn presets_match_urls_loosely() {
        assert_eq!(EndpointPreset::for_url("https://api.deepinfra.com/v1/openai/").unwrap().name, "deepinfra");
        assert_eq!(EndpointPreset::for_url("HTTPS://openrouter.ai/api/v1").unwrap().name, "openrouter");
        assert!(EndpointPreset::for_url("http://localhost:8000/v1").is_none());
        assert_eq!(EndpointPreset::by_name("Together").unwrap().base_url, "https://api.together.xyz/v1");
    }

    #[test]
    fn compatible_provider_defaults_and_env_var() {
        let config = cloud(CloudProvider::OpenAICompatible, None);
        assert!(config.has_custom_endpoint());
        assert_eq!(config.effective_base_url().as_deref(), Some(DEFAULT_COMPATIBLE_BASE_URL));
        assert_eq!(config.api_key_env_var().as_deref(), Some("DEEPINFRA_API_KEY"));
        assert_eq!(config.provider_display_name(), "DeepInfra (OpenAI-compatible)");
        assert_eq!(config.model_name(), DEFAULT_COMPATIBLE_MODEL);

        let config = cloud(CloudProvider::OpenAICompatible, Some("http://localhost:8000/v1/"));
        assert_eq!(config.base_url(), Some("http://localhost:8000/v1/"));
        assert_eq!(config.api_key_env_var().as_deref(), Some("SCRIBA_LLM_API_KEY"));
        assert_eq!(config.provider_display_name(), "OpenAI-compatible");
    }

    #[test]
    fn first_party_providers_ignore_base_url_defaults() {
        let config = cloud(CloudProvider::Anthropic, None);
        assert!(!config.has_custom_endpoint());
        assert!(config.effective_base_url().is_none());
        assert_eq!(config.api_key_env_var().as_deref(), Some("ANTHROPIC_API_KEY"));
        assert_eq!(config.provider_display_name(), "Anthropic (Claude)");
    }

    #[test]
    fn set_base_url_normalizes_and_clears() {
        let mut config = cloud(CloudProvider::OpenAICompatible, None);
        config.set_base_url(Some("  https://openrouter.ai/api/v1/  ".to_string()));
        assert_eq!(config.base_url(), Some("https://openrouter.ai/api/v1"));
        assert_eq!(config.provider_display_name(), "OpenRouter (OpenAI-compatible)");
        config.set_base_url(Some("   ".to_string()));
        assert!(config.base_url().is_none());
    }

    #[test]
    fn base_urls_persist_per_provider() {
        let mut config = cloud(CloudProvider::OpenAICompatible, None);
        let p = CloudProvider::OpenAICompatible;
        config.save_base_url_for_provider(&p, &Some("http://box:8000/v1".to_string()));
        assert_eq!(config.load_base_url_for_provider(&p).as_deref(), Some("http://box:8000/v1"));
        config.save_base_url_for_provider(&p, &None);
        assert!(config.load_base_url_for_provider(&p).is_none());
    }

    #[test]
    fn legacy_cloud_config_without_base_url_deserializes() {
        let json = r#"{"enabled":true,"mode":{"Cloud":{"provider":"Anthropic","api_key":"k","model":null}},"auto_link_threshold":0.8}"#;
        let config: EnrichmentConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(config.mode, EnrichmentMode::Cloud { base_url: None, .. }));
        assert_eq!(config.api_key(), Some("k"));
    }
}
