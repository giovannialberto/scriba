//! Configuration management for Scriba.

use anyhow::{Context, Result};
use dirs::home_dir;
use serde::{Deserialize, Serialize};
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
}

/// Audio recording settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSettings {
    pub sample_rate: u32,
    pub bitrate: u32,
    pub channels: u16,
    pub speech_optimized: bool,
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
            },
            last_api_key: None,
        }
    }
}

impl ScribaConfig {
    /// Get the path to the configuration file.
    pub fn config_path() -> PathBuf {
        home_dir()
            .expect("Failed to get home directory")
            .join("scriba_recordings")
            .join("config.json")
    }

    /// Load configuration from disk, creating default if it doesn't exist.
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();

        if !config_path.exists() {
            let config = Self::default();
            config.save()?;
            return Ok(config);
        }

        let content = fs::read_to_string(&config_path).context("Failed to read config file")?;
        let config: Self = serde_json::from_str(&content).context("Failed to parse config file")?;

        Ok(config)
    }

    /// Save configuration to disk.
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path();

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

        self.transcription = mode;
        self.save()
    }

    /// Check if using local transcription mode.
    pub fn is_local_mode(&self) -> bool {
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
