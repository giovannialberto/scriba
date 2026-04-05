//! Core module for Scriba - business logic without UI dependencies.
//!
//! This module contains:
//! - Audio recording and encoding
//! - Transcription (local sherpa-onnx and OpenAI API)
//! - Configuration management
//! - File operations
//! - Workflow orchestration

pub mod audio;
pub mod config;
pub mod files;
pub mod loopback;
pub mod playback;
pub mod recording;
pub mod ring_buffer;
pub mod transcription;
pub mod types;
pub mod workflow;

// Re-export commonly used types for convenience
pub use audio::{AudioEncoder, AudioFormat, CompressionSettings, merge_wav_files};
pub use loopback::{detect_loopback_sources, start_loopback_capture};
pub use playback::AudioPlayer;
#[allow(deprecated)]
pub use config::{resolve_transcription_mode, CloudProvider, EnrichmentConfig, EnrichmentMode, LocalModel, LocalModelSize, ModelDef, ScribaConfig, SilenceAutoStopConfig, TranscriptionMode};
pub use files::FileManager;
pub use recording::{record_audio, list_input_devices, resolve_input_device, AudioLevelMonitor, RecordOptions, RecordingResult};
pub use transcription::{transcribe_audio, TranscriptionProgress};
pub use types::{ManagedRecording, RecordingConfig, RecordingMetadata, RecordingMode};
pub use workflow::{DatabaseManager, HealthStatus, HealthStatusLevel, WorkflowManager, rebuild_world_from_entities, initialize_world_from_seed};
