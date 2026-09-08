//! Agent-side provider abstraction.
//!
//! The agent loop only needs three things from a model: stream one turn,
//! summarize history for compaction, and a display name. Everything
//! provider-specific (wire format, tool schema translation, streaming
//! parsers) is handled by [`crate::llm::GenaiProvider`].

use async_trait::async_trait;
use genai::chat::{ChatMessage, MessageContent, Tool};
use tokio::sync::mpsc;

use super::loop_runner::AgentEvent;
use crate::core::config::EnrichmentConfig;
use crate::enrichment::ProviderError;
use crate::llm::GenaiProvider;

/// Result of a single agent turn (one API call).
pub struct AgentTurnResult {
    /// Assistant content: text, tool calls, and any provider-specific parts
    /// (e.g. Gemini thought signatures) that must round-trip into history.
    pub content: MessageContent,
    /// Whether the model signaled it is done (no more tool calls desired).
    pub should_stop: bool,
    /// Prompt tokens for this turn, including cached tokens where reported.
    pub input_tokens: u32,
    /// Completion tokens for this turn.
    pub output_tokens: u32,
}

/// Trait implemented by the model transport used by the agent loop.
#[async_trait]
pub trait AgentProvider: Send + Sync {
    /// Send one turn to the LLM. Streams text chunks via `tx` during the call.
    async fn send_turn(
        &self,
        system_prompt: &str,
        messages: &[ChatMessage],
        tools: &[Tool],
        tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<AgentTurnResult, ProviderError>;

    /// Make a non-streaming call to summarize messages for compaction.
    async fn compact_history(&self, prompt: &str) -> Result<String, ProviderError>;

    /// Provider display name for status messages.
    fn display_name(&self) -> &str;
}

/// Create an `AgentProvider` from enrichment configuration.
pub fn create_agent_provider(config: &EnrichmentConfig) -> Box<dyn AgentProvider> {
    Box::new(GenaiProvider::from_config(config))
}
