//! TUI module for Scriba - Terminal User Interface.
//!
//! This module provides an interactive terminal dashboard for:
//! - Viewing and managing recordings
//! - Recording audio
//! - Transcribing recordings
//! - Playing audio
//! - Searching transcripts

mod app;
mod browse;
pub mod chat;
mod entities;
mod onboarding;
mod recording;
mod settings;
mod transcript;

pub use app::Dashboard;
