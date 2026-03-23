//! Knowledge extraction and enrichment module.
//!
//! This module provides AI-powered extraction of metadata from transcripts,
//! including summaries, topics, entities (people, organizations), and action items.
//!
//! The module also manages "Scriba's World" - an evolving understanding of
//! the owner that grows with every conversation.

mod anthropic;
pub mod chat_prompts;
mod context;
mod extractor;
mod google;
mod ollama;
mod openai;
mod prompts;
pub mod provider;
pub mod search;
pub mod world;

pub use extractor::{
    EnrichmentService, ExtractedEntity, ExtractionResult,
    WorldEntityExtractionResult, WorldEntityOrganization, WorldEntityPerson,
};
pub use ollama::{OllamaClient, OllamaError, OllamaStatus, pull_model_with_progress};
pub use provider::{LlmProvider, ProviderError, ProviderKind};
pub use world::{WorldContext, WorldData, append_new_facts};

use crate::core::config::{CloudProvider, EnrichmentConfig, EnrichmentMode};

/// Create an LLM provider from enrichment configuration.
pub fn create_provider(config: &EnrichmentConfig) -> Box<dyn LlmProvider> {
    match &config.mode {
        EnrichmentMode::Cloud { provider, api_key, model } => {
            let effective_key = if api_key.is_empty() {
                // Try env var fallback
                std::env::var(provider.env_var_name()).unwrap_or_default()
            } else {
                api_key.clone()
            };
            let model_ref = model.as_deref();

            match provider {
                CloudProvider::Anthropic => {
                    Box::new(anthropic::AnthropicClient::new(&effective_key, model_ref))
                }
                CloudProvider::OpenAI => {
                    Box::new(openai::OpenAIClient::new(&effective_key, model_ref))
                }
                CloudProvider::Google => {
                    Box::new(google::GoogleClient::new(&effective_key, model_ref))
                }
            }
        }
        EnrichmentMode::Local { ollama_endpoint, ollama_model } => {
            Box::new(OllamaClient::new(ollama_endpoint, ollama_model))
        }
    }
}
