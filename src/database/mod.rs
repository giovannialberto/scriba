//! Database module for Scriba.
//!
//! Provides data persistence with SQLite, including:
//! - Recording and transcript storage
//! - Full-text search
//! - Statistics aggregation

mod models;
mod repository;

pub use models::{Entity, EntityMentionRecord, Recording, RecordingStats, SpeakerProfile, Tag, Transcript};
pub use repository::Database;
