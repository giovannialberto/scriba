//! Database models for Scriba recordings and transcripts.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A recording entry in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recording {
    pub id: Option<i64>,
    pub directory_name: String,
    pub display_name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub duration_seconds: Option<i64>,
    pub file_size_bytes: Option<i64>,
    pub audio_format: String,
    pub sample_rate: i64,
    pub channels: i64,
    pub has_transcript: bool,
    pub transcript_status: String,
    pub language_code: String,
    pub model_used: String,
    pub tags: Option<String>,
    pub summary: Option<String>,
    pub key_points: Option<String>,
    pub action_items: Option<String>,
    pub speakers: Option<String>,
    pub sentiment_score: Option<f64>,
    pub search_index: Option<String>,
    pub categories: Option<String>,
    pub confidence_score: Option<f64>,
    pub audio_path: String,
    pub transcript_path: Option<String>,
    /// Error message from the last failed transcription attempt.
    pub transcript_error: Option<String>,
}

/// A transcript entry in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub id: Option<i64>,
    pub recording_id: i64,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub word_count: Option<i64>,
    pub character_count: Option<i64>,
    pub language_detected: Option<String>,
    pub confidence_scores: Option<String>,
    pub segments: Option<String>,
    pub entities: Option<String>,
    pub topics: Option<String>,
}

/// A tag for organizing recordings.
#[derive(Debug, Clone)]
pub struct Tag {
    pub id: Option<i64>,
    pub name: String,
    pub color: String,
    pub created_at: DateTime<Utc>,
    pub usage_count: i64,
}

/// An entity (person, organization, etc.) in the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: Option<i64>,
    pub entity_type: String,
    pub canonical_name: String,
    pub aliases: Option<String>,    // JSON array
    pub context: Option<String>,
    pub metadata: Option<String>,   // JSON object
    pub mention_count: i64,
    pub first_seen_at: Option<DateTime<Utc>>,
    pub last_seen_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A mention of an entity in a recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMentionRecord {
    pub id: Option<i64>,
    pub entity_id: Option<i64>,
    pub recording_id: i64,
    pub mention_text: String,
    pub context_snippet: Option<String>,
    pub confidence: f64,
    pub linked_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

impl Entity {
    /// Parse aliases from JSON string.
    pub fn aliases_list(&self) -> Vec<String> {
        self.aliases
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default()
    }

    /// Set aliases from a list.
    pub fn set_aliases(&mut self, aliases: Vec<String>) {
        self.aliases = Some(serde_json::to_string(&aliases).unwrap_or_default());
    }

    /// Add an alias if not already present.
    pub fn add_alias(&mut self, alias: &str) {
        let mut aliases = self.aliases_list();
        if !aliases.iter().any(|a| a.eq_ignore_ascii_case(alias)) {
            aliases.push(alias.to_string());
            self.set_aliases(aliases);
        }
    }

    /// Remove an alias if present.
    pub fn remove_alias(&mut self, alias: &str) {
        let aliases: Vec<String> = self
            .aliases_list()
            .into_iter()
            .filter(|a| !a.eq_ignore_ascii_case(alias))
            .collect();
        self.set_aliases(aliases);
    }
}

/// Aggregated statistics for recordings.
#[derive(Debug)]
pub struct RecordingStats {
    pub total_recordings: i64,
    pub total_duration_seconds: i64,
    pub total_size_bytes: i64,
    pub transcribed_count: i64,
    pub total_words: i64,
}

impl RecordingStats {
    pub fn format_duration(&self) -> String {
        let hours = self.total_duration_seconds / 3600;
        let minutes = (self.total_duration_seconds % 3600) / 60;
        let seconds = self.total_duration_seconds % 60;

        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else {
            format!("{}s", seconds)
        }
    }

    pub fn format_size(&self) -> String {
        const KB: i64 = 1024;
        const MB: i64 = KB * 1024;
        const GB: i64 = MB * 1024;

        if self.total_size_bytes >= GB {
            format!("{:.1} GB", self.total_size_bytes as f64 / GB as f64)
        } else if self.total_size_bytes >= MB {
            format!("{:.1} MB", self.total_size_bytes as f64 / MB as f64)
        } else if self.total_size_bytes >= KB {
            format!("{:.1} KB", self.total_size_bytes as f64 / KB as f64)
        } else {
            format!("{} bytes", self.total_size_bytes)
        }
    }
}
