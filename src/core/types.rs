//! Shared types for Scriba core operations.

use std::path::PathBuf;

use std::time::Duration;

use super::audio::CompressionSettings;
use super::config::TranscriptionMode;

/// High-level recording workflow parameters.
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    pub name: Option<String>,
    pub compression: Option<CompressionSettings>,
    pub auto_transcribe: bool,
    pub transcription_mode: Option<TranscriptionMode>,
}

/// Recording execution mode.
#[derive(Debug)]
pub enum RecordingMode {
    /// CLI mode: blocks until Ctrl+C.
    Cli,
    /// TUI mode: controlled via channels.
    Tui {
        stop_rx: tokio::sync::mpsc::Receiver<()>,
        level_tx: tokio::sync::mpsc::Sender<f32>,
        silence_timeout: Option<Duration>,
    },
    /// Meeting mode: stopped via `stop_rx` when the meeting watcher sees the
    /// meeting app release the mic, with `silence_timeout` as a fallback net.
    /// Used by the meeting autopilot; `verbose` must be false when it runs
    /// inside the TUI (stdout would corrupt the ratatui screen).
    Meeting {
        stop_rx: tokio::sync::mpsc::Receiver<()>,
        silence_timeout: Option<Duration>,
        verbose: bool,
    },
}

/// Represents a managed recording with metadata and file operations.
pub struct ManagedRecording {
    pub directory_name: String,
    pub display_name: Option<String>,
    pub audio_path: PathBuf,
    pub transcript_path: Option<PathBuf>,
    pub metadata: RecordingMetadata,
}

/// Audio file metadata extracted from various sources.
#[derive(Debug, Clone)]
pub struct RecordingMetadata {
    pub duration_seconds: Option<i64>,
    pub file_size_bytes: Option<i64>,
    pub audio_format: String,
    pub sample_rate: i64,
    pub channels: i64,
}
