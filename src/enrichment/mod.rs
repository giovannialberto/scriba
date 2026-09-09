//! Knowledge extraction and enrichment module.
//!
//! This module provides AI-powered extraction of metadata from transcripts,
//! including summaries, topics, entities (people, organizations), and action items.
//!
//! The module also manages "Scriba's World" - an evolving understanding of
//! the owner that grows with every conversation.
//!
//! Model calls go through [`crate::llm`]; this module owns prompts, parsing,
//! and the Ollama management helpers (model listing, pulling, diagnostics).

pub mod chat_prompts;
mod extractor;
pub mod ollama;
mod prompts;
pub mod provider;
pub mod search;
pub mod world;

pub use extractor::{
    EnrichmentService, ExtractedEntity, ExtractionResult, WorldEntityExtractionResult,
    WorldEntityOrganization, WorldEntityPerson,
};
pub use ollama::{OllamaClient, OllamaError, OllamaModelInfo, OllamaStatus, pull_model_with_progress};
pub use provider::{LlmProvider, ProviderError};
pub use world::{WorldContext, WorldData, append_new_facts};

use crate::core::config::EnrichmentConfig;

/// Create an LLM provider from enrichment configuration.
pub fn create_provider(config: &EnrichmentConfig) -> Box<dyn LlmProvider> {
    Box::new(crate::llm::GenaiProvider::from_config(config))
}
