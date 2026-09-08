//! Unified LLM transport built on [`genai`].
//!
//! Every model call in Scriba — enrichment extraction, agent chat, context
//! compaction, health checks — goes through [`GenaiProvider`]. Provider
//! selection is a pure function of [`EnrichmentConfig`], resolved into an
//! [`LlmTarget`] that pins the wire protocol, endpoint, credential and model.
//! Nothing here guesses a provider from a model name.

use std::future::Future;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use genai::adapter::AdapterKind;
use genai::chat::{
    CacheControl, ChatMessage, ChatOptions, ChatRequest, ChatResponseFormat, ChatStreamEvent,
    MessageContent, StopReason, Tool,
};
use genai::resolver::{AuthData, Endpoint, ProviderConfig};
use genai::{Client, ModelIden, ServiceTarget, WebConfig};
use tokio::sync::mpsc;

use crate::agent::loop_runner::AgentEvent;
use crate::agent::provider::{AgentProvider, AgentTurnResult};
use crate::core::config::{CloudProvider, EnrichmentConfig, EnrichmentMode};
use crate::enrichment::{LlmProvider, OllamaClient, OllamaModelInfo, ProviderError};
use tokio::sync::OnceCell;

/// Overall HTTP timeout for a single model call.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// Timeout for the lightweight connectivity probe in `health_check`.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(15);
/// Output budget for JSON extraction calls.
const JSON_MAX_TOKENS: u32 = 8192;
/// Output budget for free-text calls (world evolution, entity context).
const TEXT_MAX_TOKENS: u32 = 2048;
/// Output budget for one agent turn.
const AGENT_MAX_TOKENS: u32 = 8192;
/// Output budget for history compaction summaries.
const COMPACTION_MAX_TOKENS: u32 = 1024;
/// Sampling temperature for providers that accept one.
const EXTRACTION_TEMPERATURE: f64 = 0.3;
/// Longest error body we echo back to the user.
const MAX_ERROR_BODY_CHARS: usize = 500;

/// How transient failures are retried. One 429 or 503 must not kill an
/// enrichment run or a chat turn.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Total attempts including the first one.
    pub max_attempts: u32,
    /// Delay before the first retry; doubles each time.
    pub base_delay: Duration,
    /// Upper bound for any single delay (also caps `Retry-After`).
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(20),
        }
    }
}

impl RetryPolicy {
    /// Delay before retry number `retry` (1-based), honoring a server hint.
    fn delay_for(&self, retry: u32, err: &ProviderError) -> Duration {
        let hinted = match err {
            ProviderError::RateLimited {
                retry_after_secs: Some(secs),
                ..
            } => Some(Duration::from_secs(*secs)),
            _ => None,
        };
        let backoff = self
            .base_delay
            .checked_mul(1u32 << retry.saturating_sub(1).min(16))
            .unwrap_or(self.max_delay);
        hinted.unwrap_or(backoff).min(self.max_delay)
    }
}

/// Run `op` until it succeeds, fails with a non-retryable error, or the
/// policy is exhausted. `on_retry(next_attempt, error, delay)` is called
/// before each wait so callers can surface progress.
pub async fn with_retry<T, F, Fut, N>(
    policy: &RetryPolicy,
    mut op: F,
    mut on_retry: N,
) -> Result<T, ProviderError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ProviderError>>,
    N: FnMut(u32, &ProviderError, Duration),
{
    let mut attempt = 1;
    loop {
        match op().await {
            Ok(v) => return Ok(v),
            Err(err) if err.is_retryable() && attempt < policy.max_attempts.max(1) => {
                let delay = policy.delay_for(attempt, &err);
                attempt += 1;
                on_retry(attempt, &err, delay);
                tokio::time::sleep(delay).await;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Wire protocol spoken to the endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// Anthropic Messages API.
    Anthropic,
    /// OpenAI Chat Completions (also spoken by most open-weight hosts).
    OpenAI,
    /// Google Gemini generateContent.
    Gemini,
    /// Ollama native `/api/chat`.
    Ollama,
}

impl Protocol {
    fn adapter_kind(self) -> AdapterKind {
        match self {
            Protocol::Anthropic => AdapterKind::Anthropic,
            Protocol::OpenAI => AdapterKind::OpenAI,
            Protocol::Gemini => AdapterKind::Gemini,
            Protocol::Ollama => AdapterKind::Ollama,
        }
    }

    fn default_endpoint(self) -> &'static str {
        match self {
            Protocol::Anthropic => "https://api.anthropic.com/v1/",
            Protocol::OpenAI => "https://api.openai.com/v1/",
            Protocol::Gemini => "https://generativelanguage.googleapis.com/v1beta/",
            Protocol::Ollama => crate::core::config::DEFAULT_OLLAMA_ENDPOINT,
        }
    }

    /// Whether the endpoint needs an API key.
    fn requires_api_key(self) -> bool {
        !matches!(self, Protocol::Ollama)
    }

    /// Whether pinning a sampling temperature is safe. Current Anthropic and
    /// OpenAI reasoning models reject the parameter outright.
    fn accepts_temperature(self) -> bool {
        matches!(self, Protocol::Gemini | Protocol::Ollama)
    }

    /// Whether the adapter has a dedicated JSON-mode flag.
    fn supports_json_mode(self) -> bool {
        matches!(self, Protocol::OpenAI | Protocol::Ollama)
    }
}

/// Fully resolved destination for model calls.
#[derive(Debug, Clone)]
pub struct LlmTarget {
    pub protocol: Protocol,
    /// Base URL, always with a trailing slash.
    pub endpoint: String,
    /// Credential, if the protocol needs one and one was found.
    pub api_key: Option<String>,
    /// Environment variable consulted for the credential (for error hints).
    pub api_key_env: Option<String>,
    pub model: String,
    pub display_name: String,
}

impl LlmTarget {
    /// Resolve the target from the enrichment configuration.
    pub fn from_config(config: &EnrichmentConfig) -> Self {
        match &config.mode {
            EnrichmentMode::Cloud {
                provider, model, ..
            } => {
                let protocol = match provider {
                    CloudProvider::Anthropic => Protocol::Anthropic,
                    CloudProvider::OpenAI | CloudProvider::OpenAICompatible => Protocol::OpenAI,
                    CloudProvider::Google => Protocol::Gemini,
                };
                let endpoint = config
                    .effective_base_url()
                    .unwrap_or_else(|| protocol.default_endpoint().to_string());
                Self {
                    protocol,
                    endpoint: normalize_base_url(&endpoint),
                    api_key: config.resolve_api_key().filter(|k| !k.trim().is_empty()),
                    api_key_env: config.api_key_env_var(),
                    model: model
                        .clone()
                        .unwrap_or_else(|| provider.default_model().to_string()),
                    display_name: config.provider_display_name(),
                }
            }
            EnrichmentMode::Local {
                ollama_endpoint,
                ollama_model,
            } => Self {
                protocol: Protocol::Ollama,
                endpoint: normalize_base_url(ollama_endpoint),
                api_key: None,
                api_key_env: None,
                model: ollama_model.clone(),
                display_name: "Ollama (Local)".to_string(),
            },
        }
    }

    fn service_target(&self) -> ServiceTarget {
        let auth = match &self.api_key {
            Some(key) => AuthData::from_single(key.clone()),
            // Ollama ignores the credential but genai requires one.
            None => AuthData::from_single("ollama"),
        };
        ServiceTarget {
            endpoint: Endpoint::from_owned(self.endpoint.clone()),
            auth,
            model: ModelIden::new(self.protocol.adapter_kind(), self.model.as_str()),
        }
    }
}

/// One model offered by an endpoint, as shown in model pickers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelListEntry {
    pub id: String,
    /// `Some(false)` when the host says the model cannot call tools.
    pub supports_tools: Option<bool>,
}

impl ModelListEntry {
    /// Label for pickers: the id, flagged when the agent cannot use the model.
    pub fn display_name(&self) -> String {
        if self.supports_tools == Some(false) {
            format!("{} (no tools)", self.id)
        } else {
            self.id.clone()
        }
    }
}

impl From<OllamaModelInfo> for ModelListEntry {
    fn from(info: OllamaModelInfo) -> Self {
        Self {
            id: info.name,
            supports_tools: info.supports_tools,
        }
    }
}

impl From<String> for ModelListEntry {
    fn from(id: String) -> Self {
        Self {
            id,
            supports_tools: None,
        }
    }
}

/// List the models an endpoint advertises (`GET {base_url}/models` for
/// OpenAI-compatible hosts). Ollama has its own listing in `OllamaClient`.
pub async fn list_models(target: &LlmTarget) -> Result<Vec<String>, ProviderError> {
    let auth = match &target.api_key {
        Some(key) => AuthData::from_single(key.clone()),
        None => AuthData::from_single("none"),
    };
    let provider_config =
        ProviderConfig::from_endpoint(Endpoint::from_owned(target.endpoint.clone()))
            .with_auth(auth);
    let mut names = Client::default()
        .all_model_names(target.protocol.adapter_kind(), provider_config)
        .await
        .map_err(map_error)?;
    names.sort();
    names.dedup();
    Ok(names)
}

/// Ensure a base URL ends with exactly one slash so adapters can append paths.
fn normalize_base_url(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    format!("{trimmed}/")
}

/// Map a `genai` error onto Scriba's provider error taxonomy.
pub fn map_error(err: genai::Error) -> ProviderError {
    use genai::webc::Error as WebError;

    fn from_web(web: &WebError) -> Option<ProviderError> {
        match web {
            WebError::ResponseFailedStatus {
                status,
                body,
                headers,
            } => {
                let retry_after = headers
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.trim().parse::<u64>().ok());
                Some(status_error_with_hint(status.as_u16(), body, retry_after))
            }
            WebError::Reqwest(e) if e.is_timeout() => Some(ProviderError::Timeout {
                seconds: REQUEST_TIMEOUT.as_secs(),
            }),
            WebError::Reqwest(e) if e.is_connect() => Some(ProviderError::Network {
                message: e.to_string(),
            }),
            _ => None,
        }
    }

    match &err {
        genai::Error::WebAdapterCall { webc_error, .. }
        | genai::Error::WebModelCall { webc_error, .. } => {
            from_web(webc_error).unwrap_or_else(|| ProviderError::Other {
                message: err.to_string(),
            })
        }
        genai::Error::HttpError { status, body, .. } => status_error(status.as_u16(), body),
        // Streaming requests surface HTTP failures as a stringified cause.
        genai::Error::WebStream { cause, .. } => match parse_stream_http_status(cause) {
            Some((status, body)) => status_error(status, body),
            None => ProviderError::Network {
                message: trim_body(cause),
            },
        },
        genai::Error::StreamParse { serde_error, .. } => ProviderError::ParseError {
            message: serde_error.to_string(),
        },
        genai::Error::ChatResponse { body, .. } => ProviderError::Other {
            message: format!("Stream error: {body}"),
        },
        genai::Error::RequiresApiKey { .. }
        | genai::Error::NoAuthData { .. }
        | genai::Error::NoAuthResolver { .. } => ProviderError::AuthFailure {
            message: err.to_string(),
        },
        _ => ProviderError::Other {
            message: err.to_string(),
        },
    }
}

/// Classify an HTTP status into the provider error taxonomy.
fn status_error(status: u16, body: &str) -> ProviderError {
    status_error_with_hint(status, body, None)
}

/// Like [`status_error`], carrying a `Retry-After` hint for 429 responses.
fn status_error_with_hint(status: u16, body: &str, retry_after_secs: Option<u64>) -> ProviderError {
    let message = trim_body(body);
    match status {
        401 | 403 => ProviderError::AuthFailure { message },
        // Gemini reports a bad key as 400 INVALID_ARGUMENT rather than 401.
        400 if body.contains("API_KEY_INVALID") || body.contains("API key not valid") => {
            ProviderError::AuthFailure { message }
        }
        429 => ProviderError::RateLimited {
            message,
            retry_after_secs,
        },
        _ => ProviderError::HttpStatus { status, message },
    }
}

/// Recover `(status, body)` from genai's stringified stream HTTP error,
/// which looks like `"HTTP error.\nStatus: 400 Bad Request\nBody: {...}"`.
fn parse_stream_http_status(cause: &str) -> Option<(u16, &str)> {
    let rest = cause.split("Status:").nth(1)?;
    let status: u16 = rest.split_whitespace().next()?.parse().ok()?;
    let body = rest
        .split_once("Body:")
        .map(|(_, b)| b.trim())
        .unwrap_or("");
    Some((status, body))
}

fn trim_body(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.chars().count() <= MAX_ERROR_BODY_CHARS {
        return trimmed.to_string();
    }
    let mut s: String = trimmed.chars().take(MAX_ERROR_BODY_CHARS).collect();
    s.push('…');
    s
}

/// The one model transport used by enrichment and the agent.
pub struct GenaiProvider {
    target: LlmTarget,
    service_target: ServiceTarget,
    client: Client,
    /// Cached result of the Ollama tool-capability probe.
    tools_supported: OnceCell<bool>,
    retry: RetryPolicy,
}

impl GenaiProvider {
    /// Build a provider for a resolved target.
    pub fn new(target: LlmTarget) -> Self {
        let service_target = target.service_target();
        let client = Client::builder()
            .with_web_config(WebConfig::default().with_timeout(REQUEST_TIMEOUT))
            .build();
        Self {
            target,
            service_target,
            client,
            tools_supported: OnceCell::new(),
            retry: RetryPolicy::default(),
        }
    }

    /// Override the retry policy (tests, one-shot CLI probes).
    pub fn with_retry_policy(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Whether the model can take tool definitions. Ollama rejects requests
    /// that include tools for models without the capability, so ask once and
    /// remember. Cloud providers are assumed capable; a probe failure is too.
    async fn tools_supported(&self) -> bool {
        if self.target.protocol != Protocol::Ollama {
            return true;
        }
        *self
            .tools_supported
            .get_or_init(|| async {
                OllamaClient::new(&self.target.endpoint, &self.target.model)
                    .supports_tools()
                    .await
                    .ok()
                    .flatten()
                    .unwrap_or(true)
            })
            .await
    }

    /// Build a provider from the enrichment configuration.
    pub fn from_config(config: &EnrichmentConfig) -> Self {
        Self::new(LlmTarget::from_config(config))
    }

    /// The resolved target this provider talks to.
    pub fn target(&self) -> &LlmTarget {
        &self.target
    }

    /// Fail fast with a helpful message instead of a bare 401 when no key is set.
    fn ensure_credentials(&self) -> Result<(), ProviderError> {
        if self.target.protocol.requires_api_key() && self.target.api_key.is_none() {
            let hint = match &self.target.api_key_env {
                Some(env) => format!(" Add one in Settings or set {env}."),
                None => String::new(),
            };
            return Err(ProviderError::AuthFailure {
                message: format!(
                    "No API key configured for {}.{hint}",
                    self.target.display_name
                ),
            });
        }
        Ok(())
    }

    fn base_options(&self) -> ChatOptions {
        let mut options = ChatOptions::default();
        if self.target.protocol.accepts_temperature() {
            options = options.with_temperature(EXTRACTION_TEMPERATURE);
        }
        options
    }

    /// Non-streaming call returning the concatenated text parts.
    async fn complete_text(
        &self,
        request: ChatRequest,
        options: ChatOptions,
    ) -> Result<String, ProviderError> {
        self.ensure_credentials()?;
        let response = with_retry(
            &self.retry,
            || async {
                self.client
                    .exec_chat(self.service_target.clone(), request.clone(), Some(&options))
                    .await
                    .map_err(map_error)
            },
            |_, _, _| {},
        )
        .await?;
        let text = response.content.into_texts().join("");
        if text.trim().is_empty() {
            return Err(ProviderError::ParseError {
                message: format!("Empty response from {}", self.target.display_name),
            });
        }
        Ok(text)
    }
}

#[async_trait]
impl LlmProvider for GenaiProvider {
    async fn generate(&self, prompt: &str) -> Result<String, ProviderError> {
        let prompt = format!(
            "{prompt}\n\nRespond with valid JSON only. No markdown fences, no explanation."
        );
        let mut options = self.base_options().with_max_tokens(JSON_MAX_TOKENS);
        if self.target.protocol.supports_json_mode() {
            options = options.with_response_format(ChatResponseFormat::JsonMode);
        }
        self.complete_text(ChatRequest::from_user(prompt), options)
            .await
    }

    async fn generate_text(&self, prompt: &str) -> Result<String, ProviderError> {
        let options = self.base_options().with_max_tokens(TEXT_MAX_TOKENS);
        self.complete_text(ChatRequest::from_user(prompt), options)
            .await
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        if self.target.protocol == Protocol::Ollama {
            return OllamaClient::new(&self.target.endpoint, &self.target.model)
                .health_check()
                .await
                .map_err(Into::into);
        }
        self.ensure_credentials()?;
        let options = ChatOptions::default().with_max_tokens(64);
        let call = self.client.exec_chat(
            self.service_target.clone(),
            ChatRequest::from_user("Say OK"),
            Some(&options),
        );
        match tokio::time::timeout(HEALTH_TIMEOUT, call).await {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(e)) => Err(map_error(e)),
            Err(_) => Err(ProviderError::Timeout {
                seconds: HEALTH_TIMEOUT.as_secs(),
            }),
        }
    }

    fn display_name(&self) -> &str {
        &self.target.display_name
    }

    fn model(&self) -> &str {
        &self.target.model
    }
}

#[async_trait]
impl AgentProvider for GenaiProvider {
    async fn send_turn(
        &self,
        system_prompt: &str,
        messages: &[ChatMessage],
        tools: &[Tool],
        tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<AgentTurnResult, ProviderError> {
        self.ensure_credentials()?;

        // The system prompt and tool schemas are the stable prefix of every
        // turn; marking the system block cacheable lets Anthropic serve both
        // from cache. Other adapters ignore the hint.
        let system = ChatMessage::system(system_prompt).with_options(CacheControl::Ephemeral);
        let mut all = Vec::with_capacity(messages.len() + 1);
        all.push(system);
        all.extend(messages.iter().cloned());
        let mut request = ChatRequest::new(all);
        if self.tools_supported().await {
            request = request.with_tools(tools.to_vec());
        } else {
            let _ = tx
                .send(AgentEvent::Warning(format!(
                    "{} cannot call tools, so Scriba is answering without looking at your \
                     recordings. Pick a tools-capable model in Settings (for example {}).",
                    self.target.model,
                    crate::core::config::DEFAULT_OLLAMA_MODEL
                )))
                .await;
        }

        let options = self
            .base_options()
            .with_max_tokens(AGENT_MAX_TOKENS)
            .with_capture_usage(true)
            .with_capture_content(true)
            .with_capture_tool_calls(true);

        // Retry only the request itself: once chunks have been streamed to the
        // UI there is no way to take them back.
        let display_name = self.target.display_name.clone();
        let mut response = with_retry(
            &self.retry,
            || async {
                self.client
                    .exec_chat_stream(self.service_target.clone(), request.clone(), Some(&options))
                    .await
                    .map_err(map_error)
            },
            |attempt, err, delay| {
                let reason = match err {
                    ProviderError::RateLimited { .. } => "rate limited".to_string(),
                    ProviderError::Timeout { .. } => "timed out".to_string(),
                    ProviderError::Network { .. } => "connection failed".to_string(),
                    ProviderError::HttpStatus { status, .. } => format!("returned HTTP {status}"),
                    other => other.to_string(),
                };
                let _ = tx.try_send(AgentEvent::Status(format!(
                    "{display_name} {reason}; retrying in {}s (attempt {attempt}/{})",
                    delay.as_secs(),
                    self.retry.max_attempts
                )));
            },
        )
        .await?;

        let mut streamed_text = String::new();
        let mut end = None;
        while let Some(event) = response.stream.next().await {
            match event.map_err(map_error)? {
                ChatStreamEvent::Chunk(chunk) => {
                    if !chunk.content.is_empty() {
                        streamed_text.push_str(&chunk.content);
                        let _ = tx.send(AgentEvent::Chunk(chunk.content)).await;
                    }
                }
                ChatStreamEvent::End(e) => end = Some(e),
                ChatStreamEvent::Start
                | ChatStreamEvent::ToolCallChunk(_)
                | ChatStreamEvent::ReasoningChunk(_)
                | ChatStreamEvent::ThoughtSignatureChunk(_) => {}
            }
        }
        let end = end.unwrap_or_default();

        let content = end
            .captured_content
            .filter(|c| !c.is_empty())
            .unwrap_or_else(|| MessageContent::from_text(streamed_text));
        let has_tool_calls = !content.tool_calls().is_empty();
        // Local models may write a tool call as text instead of a tool_use part; only
        // real tool calls continue the loop.

        let truncated = matches!(end.captured_stop_reason, Some(StopReason::MaxTokens(_)));
        if truncated {
            let _ = tx
                .send(AgentEvent::Status(
                    "Response hit the output limit".to_string(),
                ))
                .await;
        }

        let (input_tokens, output_tokens) = end
            .captured_usage
            .as_ref()
            .map(|u| {
                (
                    u.prompt_tokens.unwrap_or(0).max(0) as u32,
                    u.completion_tokens.unwrap_or(0).max(0) as u32,
                )
            })
            .unwrap_or((0, 0));

        Ok(AgentTurnResult {
            content,
            should_stop: !has_tool_calls || truncated,
            input_tokens,
            output_tokens,
        })
    }

    async fn compact_history(&self, prompt: &str) -> Result<String, ProviderError> {
        let options = self.base_options().with_max_tokens(COMPACTION_MAX_TOKENS);
        self.complete_text(ChatRequest::from_user(prompt), options)
            .await
    }

    fn display_name(&self) -> &str {
        &self.target.display_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::field_reassign_with_default)]
    fn cloud_config(provider: CloudProvider, model: Option<&str>) -> EnrichmentConfig {
        let mut config = EnrichmentConfig::default();
        config.mode = EnrichmentMode::Cloud {
            provider,
            api_key: "sk-test".to_string(),
            model: model.map(str::to_string),
            base_url: None,
        };
        config
    }

    #[test]
    fn compatible_target_defaults_to_deepinfra() {
        let target = LlmTarget::from_config(&cloud_config(CloudProvider::OpenAICompatible, None));
        assert_eq!(target.protocol, Protocol::OpenAI);
        assert_eq!(target.endpoint, "https://api.deepinfra.com/v1/openai/");
        assert_eq!(target.model, crate::core::config::DEFAULT_COMPATIBLE_MODEL);
        assert_eq!(target.api_key_env.as_deref(), Some("DEEPINFRA_API_KEY"));
        assert_eq!(target.display_name, "DeepInfra (OpenAI-compatible)");
    }

    #[test]
    fn compatible_target_honors_custom_base_url() {
        let mut config = cloud_config(CloudProvider::OpenAICompatible, Some("my-model"));
        config.set_base_url(Some("http://localhost:8000/v1/".to_string()));
        let target = LlmTarget::from_config(&config);
        assert_eq!(target.endpoint, "http://localhost:8000/v1/");
        assert_eq!(target.model, "my-model");
        assert_eq!(target.api_key_env.as_deref(), Some("SCRIBA_LLM_API_KEY"));
        assert_eq!(target.display_name, "OpenAI-compatible");
    }

    #[test]
    fn base_url_override_applies_to_first_party_providers() {
        let mut config = cloud_config(CloudProvider::Anthropic, None);
        config.set_base_url(Some("https://proxy.example.com/anthropic".to_string()));
        let target = LlmTarget::from_config(&config);
        assert_eq!(target.protocol, Protocol::Anthropic);
        assert_eq!(target.endpoint, "https://proxy.example.com/anthropic/");
    }

    #[test]
    fn cloud_target_uses_provider_defaults() {
        let target = LlmTarget::from_config(&cloud_config(CloudProvider::Anthropic, None));
        assert_eq!(target.protocol, Protocol::Anthropic);
        assert_eq!(target.endpoint, "https://api.anthropic.com/v1/");
        assert_eq!(target.model, CloudProvider::Anthropic.default_model());
        assert_eq!(target.api_key.as_deref(), Some("sk-test"));
        assert_eq!(target.api_key_env.as_deref(), Some("ANTHROPIC_API_KEY"));
    }

    #[test]
    fn cloud_target_respects_explicit_model() {
        let target = LlmTarget::from_config(&cloud_config(CloudProvider::Google, Some("gemini-x")));
        assert_eq!(target.protocol, Protocol::Gemini);
        assert_eq!(target.model, "gemini-x");
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn local_target_normalizes_endpoint() {
        let mut config = EnrichmentConfig::default();
        config.mode = EnrichmentMode::Local {
            ollama_endpoint: "http://box:11434//".to_string(),
            ollama_model: "gemma4:12b".to_string(),
        };
        let target = LlmTarget::from_config(&config);
        assert_eq!(target.protocol, Protocol::Ollama);
        assert_eq!(target.endpoint, "http://box:11434/");
        assert_eq!(target.model, "gemma4:12b");
        assert!(target.api_key.is_none());
    }

    #[test]
    fn missing_cloud_key_fails_fast() {
        let target = LlmTarget {
            protocol: Protocol::OpenAI,
            endpoint: normalize_base_url(Protocol::OpenAI.default_endpoint()),
            api_key: None,
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            model: "gpt-test".to_string(),
            display_name: "OpenAI (GPT)".to_string(),
        };
        let provider = GenaiProvider::new(target);
        let err = provider.ensure_credentials().unwrap_err();
        assert!(matches!(err, ProviderError::AuthFailure { .. }));
        assert!(err.to_string().contains("OPENAI_API_KEY"));
    }

    #[test]
    fn ollama_needs_no_key() {
        let target = LlmTarget {
            protocol: Protocol::Ollama,
            endpoint: "http://localhost:11434/".to_string(),
            api_key: None,
            api_key_env: None,
            model: "m".to_string(),
            display_name: "Ollama (Local)".to_string(),
        };
        assert!(GenaiProvider::new(target).ensure_credentials().is_ok());
    }

    #[test]
    fn model_list_entries_flag_missing_tool_support() {
        let capable: ModelListEntry = "gemma4:12b".to_string().into();
        assert_eq!(capable.display_name(), "gemma4:12b");
        let entry = ModelListEntry::from(OllamaModelInfo {
            name: "gemma3:12b".to_string(),
            supports_tools: Some(false),
        });
        assert_eq!(entry.display_name(), "gemma3:12b (no tools)");
    }

    fn fast_policy() -> RetryPolicy {
        RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(4),
        }
    }

    #[tokio::test]
    async fn retries_transient_errors_then_succeeds() {
        let calls = std::cell::Cell::new(0);
        let mut retries = Vec::new();
        let result = with_retry(
            &fast_policy(),
            || {
                calls.set(calls.get() + 1);
                let n = calls.get();
                async move {
                    if n < 3 {
                        Err(ProviderError::HttpStatus {
                            status: 503,
                            message: "busy".into(),
                        })
                    } else {
                        Ok(n)
                    }
                }
            },
            |attempt, _, delay| retries.push((attempt, delay)),
        )
        .await;
        assert_eq!(result.unwrap(), 3);
        assert_eq!(
            retries,
            vec![(2, Duration::from_millis(1)), (3, Duration::from_millis(2))]
        );
    }

    #[tokio::test]
    async fn does_not_retry_permanent_errors() {
        let calls = std::cell::Cell::new(0);
        let result: Result<(), _> = with_retry(
            &fast_policy(),
            || {
                calls.set(calls.get() + 1);
                async {
                    Err(ProviderError::AuthFailure {
                        message: "bad key".into(),
                    })
                }
            },
            |_, _, _| panic!("must not retry"),
        )
        .await;
        assert!(matches!(result, Err(ProviderError::AuthFailure { .. })));
        assert_eq!(calls.get(), 1);
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts() {
        let calls = std::cell::Cell::new(0);
        let result: Result<(), _> = with_retry(
            &fast_policy(),
            || {
                calls.set(calls.get() + 1);
                async { Err(ProviderError::Timeout { seconds: 1 }) }
            },
            |_, _, _| {},
        )
        .await;
        assert!(matches!(result, Err(ProviderError::Timeout { .. })));
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn retry_delays_honor_server_hint_and_cap() {
        let policy = RetryPolicy {
            max_attempts: 5,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
        };
        let generic = ProviderError::Network {
            message: String::new(),
        };
        assert_eq!(policy.delay_for(1, &generic), Duration::from_secs(1));
        assert_eq!(policy.delay_for(2, &generic), Duration::from_secs(2));
        assert_eq!(policy.delay_for(3, &generic), Duration::from_secs(4));
        assert_eq!(policy.delay_for(4, &generic), Duration::from_secs(5));
        let hinted = ProviderError::RateLimited {
            message: String::new(),
            retry_after_secs: Some(3),
        };
        assert_eq!(policy.delay_for(1, &hinted), Duration::from_secs(3));
        let huge_hint = ProviderError::RateLimited {
            message: String::new(),
            retry_after_secs: Some(600),
        };
        assert_eq!(policy.delay_for(1, &huge_hint), Duration::from_secs(5));
    }

    #[test]
    fn retry_after_header_is_carried() {
        assert!(matches!(
            status_error_with_hint(429, "slow down", Some(7)),
            ProviderError::RateLimited {
                retry_after_secs: Some(7),
                ..
            }
        ));
    }

    #[test]
    fn status_codes_map_to_error_kinds() {
        assert!(matches!(
            status_error(401, ""),
            ProviderError::AuthFailure { .. }
        ));
        assert!(matches!(
            status_error(403, ""),
            ProviderError::AuthFailure { .. }
        ));
        assert!(matches!(
            status_error(429, ""),
            ProviderError::RateLimited { .. }
        ));
        assert!(matches!(
            status_error(503, "busy"),
            ProviderError::HttpStatus { status: 503, .. }
        ));
        assert!(status_error(503, "busy").is_retryable());
        assert!(!status_error(400, "bad").is_retryable());
    }

    #[test]
    fn gemini_invalid_key_is_an_auth_failure() {
        let body = r#"{"error": {"code": 400, "status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}}"#;
        assert!(matches!(
            status_error(400, body),
            ProviderError::AuthFailure { .. }
        ));
        assert!(matches!(
            status_error(400, "malformed request"),
            ProviderError::HttpStatus { status: 400, .. }
        ));
    }

    #[test]
    fn stream_http_errors_are_parsed() {
        let cause = "HTTP error.\nStatus: 429 Too Many Requests\nBody: {\"slow\": true}";
        let (status, body) = parse_stream_http_status(cause).unwrap();
        assert_eq!(status, 429);
        assert_eq!(body, "{\"slow\": true}");
        assert!(matches!(
            status_error(status, body),
            ProviderError::RateLimited { .. }
        ));
        assert!(parse_stream_http_status("connection reset").is_none());
    }

    #[test]
    fn error_bodies_are_truncated() {
        let long = "x".repeat(MAX_ERROR_BODY_CHARS + 50);
        let msg = trim_body(&long);
        assert_eq!(msg.chars().count(), MAX_ERROR_BODY_CHARS + 1);
        assert!(msg.ends_with('…'));
        assert_eq!(trim_body("  short  "), "short");
    }

    #[test]
    fn temperature_only_where_safe() {
        assert!(!Protocol::Anthropic.accepts_temperature());
        assert!(!Protocol::OpenAI.accepts_temperature());
        assert!(Protocol::Gemini.accepts_temperature());
        assert!(Protocol::Ollama.accepts_temperature());
    }
}
