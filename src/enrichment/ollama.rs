//! Ollama management client: health checks, diagnostics, model listing and pulls.
//!
//! Inference itself goes through [`crate::llm`].

use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;
use thiserror::Error;

use super::provider::ProviderError;

/// Errors that can occur when interacting with Ollama.
#[derive(Error, Debug)]
pub enum OllamaError {
    #[error("Ollama is not running at {endpoint}. Please start Ollama first.")]
    NotRunning { endpoint: String },

    #[error("Model '{model}' is not available. Run: ollama pull {model}")]
    ModelNotFound { model: String },

    #[error("Ollama request failed: {message}")]
    RequestFailed { message: String },

    #[error("Failed to parse Ollama response: {message}")]
    ParseError { message: String },

    #[error("Request timeout after {seconds}s")]
    Timeout { seconds: u64 },
}

/// Response from Ollama tags API (list models).
#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

/// Information about an available model.
#[derive(Debug, Deserialize)]
struct ModelInfo {
    name: String,
}

/// An installed Ollama model and whether it can call tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OllamaModelInfo {
    pub name: String,
    /// `None` when the server did not report capabilities.
    pub supports_tools: Option<bool>,
}

/// Extract the `capabilities` array from an `/api/show` response body.
fn parse_capabilities(body: &serde_json::Value) -> Vec<String> {
    body.get("capabilities")
        .and_then(|c| c.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

/// Structured diagnosis of Ollama readiness.
#[derive(Debug)]
pub enum OllamaStatus {
    /// Ollama is running and the model is available.
    Ready,
    /// The `ollama` binary was not found in PATH.
    NotInstalled,
    /// Ollama binary exists but the server is not responding.
    NotRunning { endpoint: String },
    /// Server is running but the configured model is not pulled.
    ModelMissing { model: String },
}

impl OllamaStatus {
    /// Return a user-facing hint describing how to fix the issue.
    pub fn hint(&self) -> Option<String> {
        match self {
            OllamaStatus::Ready => None,
            OllamaStatus::NotInstalled => Some(
                "I need Ollama to think!\n\nInstall it with:\n  brew install ollama".to_string(),
            ),
            OllamaStatus::NotRunning { .. } => Some(
                "Ollama is installed but sleeping.\n\nStart it with:\n  ollama serve".to_string(),
            ),
            OllamaStatus::ModelMissing { model } => Some(format!(
                "Almost there! Pull the model:\n  ollama pull {}",
                model
            )),
        }
    }
}

/// Client for interacting with the Ollama API.
#[derive(Clone)]
pub struct OllamaClient {
    client: Client,
    endpoint: String,
    model: String,
}

impl OllamaClient {
    /// Create a new Ollama client.
    pub fn new(endpoint: &str, model: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300)) // 5 minute timeout for generation
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    /// Check if Ollama is running and the model is available.
    pub async fn health_check(&self) -> Result<(), OllamaError> {
        // Check if Ollama is running
        let tags_url = format!("{}/api/tags", self.endpoint);
        let response = self
            .client
            .get(&tags_url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|_| OllamaError::NotRunning {
                endpoint: self.endpoint.clone(),
            })?;

        if !response.status().is_success() {
            return Err(OllamaError::NotRunning {
                endpoint: self.endpoint.clone(),
            });
        }

        // Check if the model is available
        let tags: TagsResponse = response.json().await.map_err(|e| OllamaError::ParseError {
            message: e.to_string(),
        })?;

        let model_base = self.model.split(':').next().unwrap_or(&self.model);
        let model_available = tags.models.iter().any(|m| {
            let name_base = m.name.split(':').next().unwrap_or(&m.name);
            name_base == model_base || m.name == self.model
        });

        if !model_available {
            return Err(OllamaError::ModelNotFound {
                model: self.model.clone(),
            });
        }

        Ok(())
    }

    /// Diagnose Ollama readiness with actionable status.
    pub async fn diagnose(&self) -> OllamaStatus {
        // Check if ollama binary is in PATH
        let binary_found = std::process::Command::new("which")
            .arg("ollama")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        if !binary_found {
            return OllamaStatus::NotInstalled;
        }

        // Check if server is responding
        let tags_url = format!("{}/api/tags", self.endpoint);
        let response = match self
            .client
            .get(&tags_url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => r,
            _ => {
                return OllamaStatus::NotRunning {
                    endpoint: self.endpoint.clone(),
                };
            }
        };

        // Check if model is available
        if let Ok(tags) = response.json::<TagsResponse>().await {
            let model_base = self.model.split(':').next().unwrap_or(&self.model);
            let model_available = tags.models.iter().any(|m| {
                let name_base = m.name.split(':').next().unwrap_or(&m.name);
                name_base == model_base || m.name == self.model
            });
            if !model_available {
                return OllamaStatus::ModelMissing {
                    model: self.model.clone(),
                };
            }
        }

        OllamaStatus::Ready
    }

    /// Fetch available model names from an Ollama endpoint.
    /// Creates a temporary client with a 5-second timeout.
    pub async fn fetch_models(endpoint: &str) -> Result<Vec<String>, OllamaError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| OllamaError::RequestFailed { message: e.to_string() })?;

        let url = format!("{}/api/tags", endpoint.trim_end_matches('/'));
        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|_| OllamaError::NotRunning { endpoint: endpoint.to_string() })?;

        if !response.status().is_success() {
            return Err(OllamaError::NotRunning { endpoint: endpoint.to_string() });
        }

        let tags: TagsResponse = response.json().await.map_err(|e| OllamaError::ParseError {
            message: e.to_string(),
        })?;

        let names: Vec<String> = tags.models.into_iter().map(|m| m.name).collect();
        Ok(names)
    }

    /// Fetch the installed models together with whether each can call tools,
    /// so the UI can steer users away from models the agent cannot use.
    /// Falls back to `supports_tools: None` when `/api/show` is unavailable.
    pub async fn fetch_model_infos(endpoint: &str) -> Result<Vec<OllamaModelInfo>, OllamaError> {
        let names = Self::fetch_models(endpoint).await?;
        let mut infos = Vec::with_capacity(names.len());
        for name in names {
            let supports_tools = Self::show_capabilities(endpoint, &name)
                .await
                .ok()
                .map(|caps| caps.iter().any(|c| c == "tools"));
            infos.push(OllamaModelInfo { name, supports_tools });
        }
        Ok(infos)
    }

    /// Capabilities Ollama reports for a model (`completion`, `tools`,
    /// `thinking`, `vision`, ...), via `POST /api/show`.
    pub async fn show_capabilities(endpoint: &str, model: &str) -> Result<Vec<String>, OllamaError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| OllamaError::RequestFailed { message: e.to_string() })?;

        let url = format!("{}/api/show", endpoint.trim_end_matches('/'));
        let response = client
            .post(&url)
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await
            .map_err(|_| OllamaError::NotRunning { endpoint: endpoint.to_string() })?;

        if response.status().as_u16() == 404 {
            return Err(OllamaError::ModelNotFound { model: model.to_string() });
        }
        if !response.status().is_success() {
            return Err(OllamaError::RequestFailed {
                message: format!("HTTP {} from /api/show", response.status()),
            });
        }

        let body: serde_json::Value = response.json().await.map_err(|e| OllamaError::ParseError {
            message: e.to_string(),
        })?;
        Ok(parse_capabilities(&body))
    }

    /// Whether the configured model can call tools. `Ok(None)` means Ollama
    /// did not report capabilities (older server), so nothing can be assumed.
    pub async fn supports_tools(&self) -> Result<Option<bool>, OllamaError> {
        let caps = Self::show_capabilities(&self.endpoint, &self.model).await?;
        if caps.is_empty() {
            return Ok(None);
        }
        Ok(Some(caps.iter().any(|c| c == "tools")))
    }

    /// Get the configured model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get the configured endpoint.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

/// Pull an Ollama model with streaming progress.
///
/// Sends progress as `(status_text, percentage)` where percentage is 0–100
/// or `None` for status-only updates. Completes when pull is done.
pub async fn pull_model_with_progress(
    endpoint: &str,
    model: &str,
    tx: tokio::sync::mpsc::UnboundedSender<(String, Option<u8>)>,
) -> Result<(), OllamaError> {
    use futures_util::StreamExt;

    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|e| OllamaError::RequestFailed { message: e.to_string() })?;

    let url = format!("{}/api/pull", endpoint.trim_end_matches('/'));
    let body = serde_json::json!({ "name": model, "stream": true });

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|_| OllamaError::NotRunning { endpoint: endpoint.to_string() })?;

    if !response.status().is_success() {
        let status = response.status();
        let body_text = response.text().await.unwrap_or_default();
        return Err(OllamaError::RequestFailed {
            message: format!("HTTP {}: {}", status, body_text),
        });
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| OllamaError::RequestFailed { message: e.to_string() })?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].trim().to_string();
            buffer = buffer[newline_pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                let status_text = val.get("status")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string();

                let pct = match (val.get("completed"), val.get("total")) {
                    (Some(c), Some(t)) => {
                        let completed = c.as_u64().unwrap_or(0);
                        let total = t.as_u64().unwrap_or(1).max(1);
                        Some(((completed as f64 / total as f64) * 100.0).min(100.0) as u8)
                    }
                    _ => None,
                };

                let _ = tx.send((status_text, pct));
            }
        }
    }

    let _ = tx.send(("success".to_string(), Some(100)));
    Ok(())
}

impl From<OllamaError> for ProviderError {
    fn from(err: OllamaError) -> Self {
        match err {
            OllamaError::NotRunning { endpoint } => ProviderError::Network {
                message: format!("Ollama not running at {}", endpoint),
            },
            OllamaError::ModelNotFound { model } => ProviderError::Other {
                message: format!("Model '{}' not available", model),
            },
            OllamaError::RequestFailed { message } => ProviderError::Other { message },
            OllamaError::ParseError { message } => ProviderError::ParseError { message },
            OllamaError::Timeout { seconds } => ProviderError::Timeout { seconds },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_are_parsed_from_show_response() {
        let body = serde_json::json!({
            "capabilities": ["completion", "tools", "thinking"],
            "model_info": {"gemma4.context_length": 262144}
        });
        assert_eq!(parse_capabilities(&body), vec!["completion", "tools", "thinking"]);
        assert!(parse_capabilities(&serde_json::json!({"details": {}})).is_empty());
        assert!(parse_capabilities(&serde_json::json!({"capabilities": "tools"})).is_empty());
    }

    #[test]
    fn test_client_creation() {
        let client = OllamaClient::new("http://localhost:11434", "llama3.2");
        assert_eq!(client.endpoint(), "http://localhost:11434");
        assert_eq!(client.model(), "llama3.2");
    }

    #[test]
    fn test_endpoint_trailing_slash() {
        let client = OllamaClient::new("http://localhost:11434/", "llama3.2");
        assert_eq!(client.endpoint(), "http://localhost:11434");
    }
}
