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
    Local { model_size: LocalModelSize },
    Api { api_key: String },
}

/// Available local Whisper model sizes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LocalModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    Turbo,
}

impl std::fmt::Display for LocalModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            LocalModelSize::Tiny => "tiny",
            LocalModelSize::Base => "base",
            LocalModelSize::Small => "small",
            LocalModelSize::Medium => "medium",
            LocalModelSize::Large => "large",
            LocalModelSize::Turbo => "turbo",
        };
        write!(f, "{}", s)
    }
}

impl std::str::FromStr for LocalModelSize {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(LocalModelSize::Tiny),
            "base" => Ok(LocalModelSize::Base),
            "small" => Ok(LocalModelSize::Small),
            "medium" => Ok(LocalModelSize::Medium),
            "large" => Ok(LocalModelSize::Large),
            "turbo" => Ok(LocalModelSize::Turbo),
            _ => Err(anyhow::anyhow!("Invalid model size: {}", s)),
        }
    }
}

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
    /// Speaker diarization settings.
    #[serde(default)]
    pub diarization: DiarizationConfig,
    /// Voice-activated recording ("Scriba Forever") settings.
    #[serde(default)]
    pub voice: VoiceConfig,
    /// Preserved local model size when switching from Private to Cloud mode.
    #[serde(default)]
    pub last_local_model_size: Option<LocalModelSize>,
    /// Preserved cloud provider when switching from Cloud to Private mode.
    #[serde(default)]
    pub last_cloud_provider: Option<CloudProvider>,
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

/// Cloud LLM provider for enrichment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CloudProvider {
    Anthropic,
    OpenAI,
    Google,
}

impl std::fmt::Display for CloudProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloudProvider::Anthropic => write!(f, "anthropic"),
            CloudProvider::OpenAI => write!(f, "openai"),
            CloudProvider::Google => write!(f, "google"),
        }
    }
}

impl std::str::FromStr for CloudProvider {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "anthropic" | "claude" => Ok(CloudProvider::Anthropic),
            "openai" | "gpt" => Ok(CloudProvider::OpenAI),
            "google" | "gemini" => Ok(CloudProvider::Google),
            _ => Err(anyhow::anyhow!("Invalid cloud provider: {}. Use: anthropic, openai, google", s)),
        }
    }
}

impl CloudProvider {
    /// Default model for this provider.
    pub fn default_model(&self) -> &str {
        match self {
            CloudProvider::Anthropic => "claude-sonnet-4-6",
            CloudProvider::OpenAI => "gpt-5.2",
            CloudProvider::Google => "gemini-2.5-flash",
        }
    }

    /// Display name for UI.
    pub fn display_name(&self) -> &str {
        match self {
            CloudProvider::Anthropic => "Anthropic (Claude)",
            CloudProvider::OpenAI => "OpenAI (GPT)",
            CloudProvider::Google => "Google (Gemini)",
        }
    }

    /// Env var name for this provider's API key.
    pub fn env_var_name(&self) -> &str {
        match self {
            CloudProvider::Anthropic => "ANTHROPIC_API_KEY",
            CloudProvider::OpenAI => "OPENAI_API_KEY",
            CloudProvider::Google => "GOOGLE_API_KEY",
        }
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelDef {
    pub display_name: String,
    pub model_id: String,
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
            let model = self.ollama_model.take().unwrap_or_else(|| "mistral:latest".to_string());
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

    /// Save an API key for a specific provider into the per-provider map.
    pub fn save_key_for_provider(&mut self, provider: &CloudProvider, key: &str) {
        if key.is_empty() {
            self.cloud_api_keys.remove(&provider.to_string());
        } else {
            self.cloud_api_keys.insert(provider.to_string(), key.to_string());
        }
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

    /// Get the current cloud provider, if in cloud mode.
    pub fn cloud_provider(&self) -> Option<&CloudProvider> {
        match &self.mode {
            EnrichmentMode::Cloud { provider, .. } => Some(provider),
            _ => None,
        }
    }

    /// Get the provider display name.
    pub fn provider_display_name(&self) -> &str {
        match &self.mode {
            EnrichmentMode::Cloud { provider, .. } => provider.display_name(),
            EnrichmentMode::Local { .. } => "Ollama (Local)",
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
            EnrichmentMode::Cloud { provider, api_key, .. } => {
                if !api_key.is_empty() {
                    return Some(api_key.clone());
                }
                // Fallback to env var
                std::env::var(provider.env_var_name()).ok()
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
            _ => "http://localhost:11434".to_string(),
        }
    }

    /// Get the Ollama model (only meaningful in Local mode).
    /// Returns a default if not in Local mode.
    pub fn ollama_model(&self) -> String {
        match &self.mode {
            EnrichmentMode::Local { ollama_model, .. } => ollama_model.clone(),
            _ => "mistral:latest".to_string(),
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
    /// Whether speaker diarization is enabled during transcription.
    pub enabled: bool,
    /// Maximum number of speakers to detect.
    pub max_speakers: u32,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_speakers: 6,
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
                model_size: LocalModelSize::Medium,
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
            diarization: DiarizationConfig::default(),
            voice: VoiceConfig::default(),
            last_local_model_size: None,
            last_cloud_provider: None,
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
        // Save current API key if switching away from API mode
        if let TranscriptionMode::Api { api_key } = &self.transcription {
            if !api_key.is_empty() {
                self.last_api_key = Some(api_key.clone());
            }
        }
        // Save current local model size if switching away from Local mode
        if let TranscriptionMode::Local { model_size } = &self.transcription {
            self.last_local_model_size = Some(*model_size);
        }

        self.transcription = mode;
        self.save()
    }

    /// Check if in Private mode (local Whisper + Ollama).
    pub fn is_private_mode(&self) -> bool {
        matches!(self.transcription, TranscriptionMode::Local { .. })
    }

    /// Get the API key if in API mode.
    pub fn get_api_key(&self) -> Option<&str> {
        match &self.transcription {
            TranscriptionMode::Api { api_key } => Some(api_key),
            _ => None,
        }
    }

    /// Get the local model size if in local mode.
    pub fn get_local_model_size(&self) -> Option<LocalModelSize> {
        match &self.transcription {
            TranscriptionMode::Local { model_size } => Some(*model_size),
            _ => None,
        }
    }
}

/// Resolve transcription mode from CLI flags and config.
/// Priority: force_local > api_key > model > config default
pub fn resolve_transcription_mode(
    force_local: bool,
    model: Option<LocalModelSize>,
    api_key: Option<String>,
    config: &ScribaConfig,
) -> Result<TranscriptionMode> {
    if force_local {
        let model_size = model.unwrap_or(LocalModelSize::Medium);
        return Ok(TranscriptionMode::Local { model_size });
    }

    if let Some(key) = api_key {
        return Ok(TranscriptionMode::Api { api_key: key });
    }

    if let Some(model_size) = model {
        return Ok(TranscriptionMode::Local { model_size });
    }

    // Use config default
    Ok(config.transcription.clone())
}
