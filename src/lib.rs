//! Scriba - A CLI & TUI for recording and transcribing audio.
//!
//! This crate provides:
//! - Audio recording with configurable compression
//! - Transcription via local Whisper or OpenAI API
//! - SQLite database for storing recordings and transcripts
//! - Interactive TUI dashboard
//! - MCP server for AI assistant integration

pub mod core;
pub mod database;
pub mod errors;
pub mod mcp;
pub mod tui;
pub mod utils;

// Re-export commonly used types for backward compatibility and convenience
pub use core::{
    AudioEncoder, AudioFormat, CompressionSettings, LocalModelSize, ScribaConfig,
    TranscriptionMode, WorkflowManager,
};
pub use database::{Database, Recording, RecordingStats, Transcript};
pub use tui::Dashboard;
