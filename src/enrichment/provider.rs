//! Common LLM provider trait and error type.
//!
//! The single implementation lives in [`crate::llm::GenaiProvider`]; this
//! trait exists so enrichment code can be tested against fakes without a
//! network.

use async_trait::async_trait;
use thiserror::Error;

/// Errors that can occur when interacting with an LLM provider.
#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("Authentication failed: {message}")]
    AuthFailure { message: String },

    #[error("Rate limited: {message}")]
    RateLimited { message: String },

    #[error("API error ({status}): {message}")]
    HttpStatus { status: u16, message: String },

    #[error("Request timeout after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("Network error: {message}")]
    Network { message: String },

    #[error("Failed to parse response: {message}")]
    ParseError { message: String },

    #[error("Provider error: {message}")]
    Other { message: String },
}

impl ProviderError {
    /// Whether retrying the same request later has a reasonable chance of
    /// succeeding (rate limits, timeouts, transport errors, server errors).
    pub fn is_retryable(&self) -> bool {
        match self {
            ProviderError::RateLimited { .. }
            | ProviderError::Timeout { .. }
            | ProviderError::Network { .. } => true,
            ProviderError::HttpStatus { status, .. } => *status >= 500,
            ProviderError::AuthFailure { .. }
            | ProviderError::ParseError { .. }
            | ProviderError::Other { .. } => false,
        }
    }
}

/// Trait for LLM providers that can generate text from prompts.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Generate a JSON response from a prompt.
    async fn generate(&self, prompt: &str) -> Result<String, ProviderError>;

    /// Generate a plain text response (no JSON format constraint).
    async fn generate_text(&self, prompt: &str) -> Result<String, ProviderError>;

    /// Check if the provider is reachable and configured.
    async fn health_check(&self) -> Result<(), ProviderError>;

    /// Provider display name (for UI).
    fn display_name(&self) -> &str;

    /// Model name in use.
    fn model(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_classification() {
        assert!(
            ProviderError::RateLimited {
                message: String::new()
            }
            .is_retryable()
        );
        assert!(ProviderError::Timeout { seconds: 1 }.is_retryable());
        assert!(
            ProviderError::Network {
                message: String::new()
            }
            .is_retryable()
        );
        assert!(
            ProviderError::HttpStatus {
                status: 503,
                message: String::new()
            }
            .is_retryable()
        );
        assert!(
            !ProviderError::HttpStatus {
                status: 400,
                message: String::new()
            }
            .is_retryable()
        );
        assert!(
            !ProviderError::AuthFailure {
                message: String::new()
            }
            .is_retryable()
        );
        assert!(
            !ProviderError::ParseError {
                message: String::new()
            }
            .is_retryable()
        );
    }
}
