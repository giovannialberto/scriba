//! Manual smoke test for the LLM provider layer: health check, JSON
//! generation, and one agent turn with a tool call.
//!
//! Hits real providers, so it is not part of `cargo test`. Run it against a
//! throwaway HOME so the agent sees an empty database:
//!
//! ```text
//! cargo build --example llm_smoke
//! HOME=/tmp/scriba-smoke target/debug/examples/llm_smoke ollama qwen2.5:7b
//! HOME=/tmp/scriba-smoke SMOKE_KEY=sk-... target/debug/examples/llm_smoke anthropic claude-sonnet-4-6
//! HOME=/tmp/scriba-smoke SMOKE_KEY=...    target/debug/examples/llm_smoke google gemini-2.5-flash
//! HOME=/tmp/scriba-smoke SMOKE_KEY=...    target/debug/examples/llm_smoke deepinfra Qwen/Qwen3.5-397B-A17B
//! HOME=/tmp/scriba-smoke SMOKE_KEY=x SMOKE_BASE_URL=http://localhost:11434/v1 target/debug/examples/llm_smoke custom qwen2.5:7b
//! ```

use scriba::agent::loop_runner::AgentEvent;
use scriba::agent::{create_agent_provider, run_agent_loop};
use scriba::core::config::{CloudProvider, EnrichmentConfig, EnrichmentMode};
use scriba::enrichment::create_provider;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args.get(1).map(String::as_str).unwrap_or("ollama");
    let model = args.get(2).cloned();

    let mut config = EnrichmentConfig::default();
    config.mode = match kind {
        "ollama" => EnrichmentMode::Local {
            ollama_endpoint: "http://localhost:11434".to_string(),
            ollama_model: model.unwrap_or_else(|| "qwen2.5:7b".to_string()),
        },
        other => {
            let provider: CloudProvider = other.parse().expect("provider");
            EnrichmentMode::Cloud {
                provider,
                api_key: std::env::var("SMOKE_KEY").unwrap_or_default(),
                model,
                base_url: std::env::var("SMOKE_BASE_URL").ok(),
            }
        }
    };

    // 1. Enrichment-style JSON generation
    let llm = create_provider(&config);
    println!("== provider: {} / {}", llm.display_name(), llm.model());
    match llm.health_check().await {
        Ok(()) => println!("== health: ok"),
        Err(e) => println!("== health: FAILED: {e}"),
    }
    if config.has_custom_endpoint() {
        let target = scriba::llm::LlmTarget::from_config(&config);
        match scriba::llm::list_models(&target).await {
            Ok(names) => println!("== models: {} listed, e.g. {:?}", names.len(), names.iter().take(3).collect::<Vec<_>>()),
            Err(e) => println!("== models: FAILED: {e}"),
        }
    }
    match llm
        .generate("Return a JSON object with keys \"city\" (string) and \"population\" (integer) for Rome.")
        .await
    {
        Ok(json) => {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(json.trim());
            println!("== generate: {} chars, valid json = {}", json.len(), parsed.is_ok());
            if parsed.is_err() {
                println!("   raw: {json}");
            }
        }
        Err(e) => println!("== generate: FAILED: {e}"),
    }

    // 2. Agent loop with tools
    let provider = create_agent_provider(&config);
    let (tx, mut rx) = mpsc::channel(100);
    let system = "You are Scriba, an assistant over the user's recordings. Always use tools to answer questions about the data.".to_string();
    let question = "How many recordings do I have in total? Use the get_stats tool, then answer in one sentence.".to_string();
    let handle = tokio::spawn(async move {
        run_agent_loop(provider, system, Vec::new(), question, tx).await;
    });

    let mut text = String::new();
    let mut tool_calls = 0;
    let mut errors = 0;
    while let Some(ev) = rx.recv().await {
        match ev {
            AgentEvent::Status(s) => println!("   [status] {s}"),
            AgentEvent::Warning(w) => println!("   [warning] {w}"),
            AgentEvent::Chunk(c) => text.push_str(&c),
            AgentEvent::ToolCall {
                name,
                input_summary,
            } => {
                tool_calls += 1;
                println!("   [tool call] {name}({input_summary})");
            }
            AgentEvent::ToolResult {
                name,
                output_summary,
            } => {
                println!("   [tool result] {name} -> {output_summary}");
            }
            AgentEvent::Usage {
                input_tokens,
                output_tokens,
            } => {
                println!("   [usage] in={input_tokens} out={output_tokens}");
            }
            AgentEvent::Done => println!("   [done]"),
            AgentEvent::Error(e) => {
                errors += 1;
                println!("   [ERROR] {e}");
            }
        }
    }
    let _ = handle.await;
    println!("== agent: tool_calls={tool_calls} errors={errors}");
    println!("== answer: {}", text.trim());
}
