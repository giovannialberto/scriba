//! Scriba - A CLI & TUI for recording and transcribing audio.
//!
//! This crate provides:
//! - Audio recording with configurable compression
//! - Transcription via local sherpa-onnx (Whisper, SenseVoice, etc.) or OpenAI API
//! - SQLite database for storing recordings and transcripts
//! - Interactive TUI dashboard
//! - MCP server for AI assistant integration

pub mod agent;
pub mod core;
pub mod database;
pub mod enrichment;
pub mod entities;
pub mod errors;
pub mod mcp;
pub mod tools;
pub mod tui;
pub mod utils;

// Re-export commonly used types for backward compatibility and convenience
pub use core::{
    AudioEncoder, AudioFormat, CompressionSettings, EnrichmentConfig, LocalModel, LocalModelSize,
    ScribaConfig, TranscriptionMode, WorkflowManager,
};
pub use database::{Database, Entity, Recording, RecordingStats, Transcript};
pub use enrichment::{EnrichmentService, ExtractionResult, OllamaClient};
pub use entities::{EntityLinker, EntityRegistry};
pub use tui::Dashboard;
