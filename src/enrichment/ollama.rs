//! Ollama API client for local LLM inference.

use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

use super::provider::{LlmProvider, ProviderError};

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

/// Request body for Ollama generate API.
#[derive(Debug, Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'a str>,
    options: GenerateOptions,
}

/// Generation options for Ollama.
#[derive(Debug, Serialize)]
struct GenerateOptions {
    temperature: f32,
    num_predict: i32,
}

/// Response from Ollama generate API.
#[derive(Debug, Deserialize)]
struct GenerateResponse {
    response: String,
    #[allow(dead_code)]
    #[serde(default)]
    done: bool,
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
    timeout_seconds: u64,
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
            timeout_seconds: 300,
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

    /// Generate a response from the LLM.
    pub async fn generate(&self, prompt: &str) -> Result<String, OllamaError> {
        let url = format!("{}/api/generate", self.endpoint);

        let request = GenerateRequest {
            model: &self.model,
            prompt,
            stream: false,
            format: Some("json"),
            options: GenerateOptions {
                temperature: 0.3, // Low temperature for more consistent extraction
                num_predict: 4096, // Allow enough tokens for full extraction
            },
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(self.timeout_seconds))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    OllamaError::Timeout {
                        seconds: self.timeout_seconds,
                    }
                } else if e.is_connect() {
                    OllamaError::NotRunning {
                        endpoint: self.endpoint.clone(),
                    }
                } else {
                    OllamaError::RequestFailed {
                        message: e.to_string(),
                    }
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(OllamaError::RequestFailed {
                message: format!("HTTP {}: {}", status, body),
            });
        }

        let result: GenerateResponse =
            response.json().await.map_err(|e| OllamaError::ParseError {
                message: e.to_string(),
            })?;

        Ok(result.response)
    }

    /// Generate a response without JSON format constraint.
    pub async fn generate_text(&self, prompt: &str) -> Result<String, OllamaError> {
        let url = format!("{}/api/generate", self.endpoint);

        let request = GenerateRequest {
            model: &self.model,
            prompt,
            stream: false,
            format: None,
            options: GenerateOptions {
                temperature: 0.3,
                num_predict: 2048,
            },
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(self.timeout_seconds))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    OllamaError::Timeout {
                        seconds: self.timeout_seconds,
                    }
                } else if e.is_connect() {
                    OllamaError::NotRunning {
                        endpoint: self.endpoint.clone(),
                    }
                } else {
                    OllamaError::RequestFailed {
                        message: e.to_string(),
                    }
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(OllamaError::RequestFailed {
                message: format!("HTTP {}: {}", status, body),
            });
        }

        let result: GenerateResponse =
            response.json().await.map_err(|e| OllamaError::ParseError {
                message: e.to_string(),
            })?;

        Ok(result.response)
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

#[async_trait]
impl LlmProvider for OllamaClient {
    async fn generate(&self, prompt: &str) -> Result<String, ProviderError> {
        OllamaClient::generate(self, prompt).await.map_err(Into::into)
    }

    async fn generate_text(&self, prompt: &str) -> Result<String, ProviderError> {
        OllamaClient::generate_text(self, prompt).await.map_err(Into::into)
    }

    async fn generate_text_stream(
        &self,
        prompt: &str,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<(), ProviderError> {
        use futures_util::StreamExt;

        let url = format!("{}/api/generate", self.endpoint);

        let request = GenerateRequest {
            model: &self.model,
            prompt,
            stream: true,
            format: None,
            options: GenerateOptions {
                temperature: 0.3,
                num_predict: 4096,
            },
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(std::time::Duration::from_secs(self.timeout_seconds))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProviderError::Timeout { seconds: self.timeout_seconds }
                } else if e.is_connect() {
                    ProviderError::Network { message: format!("Ollama not running at {}", self.endpoint) }
                } else {
                    ProviderError::Other { message: e.to_string() }
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::Other {
                message: format!("HTTP {}: {}", status, body),
            });
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ProviderError::Network { message: e.to_string() })?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines (NDJSON)
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                if let Ok(resp) = serde_json::from_str::<GenerateResponse>(&line) {
                    if !resp.response.is_empty() {
                        let _ = tx.send(resp.response).await;
                    }
                    if resp.done {
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        OllamaClient::health_check(self).await.map_err(Into::into)
    }

    fn display_name(&self) -> &str {
        "Ollama (Local)"
    }

    fn model(&self) -> &str {
        OllamaClient::model(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
