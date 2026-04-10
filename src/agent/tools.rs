//! Agent-side tool wrappers.
//!
//! Delegates to `crate::tools` for definitions and execution,
//! and provides Anthropic-format wrapping + TUI display helpers.

use crate::database::Database;
use crate::tools;
use serde_json::{json, Value};

/// Return all tool definitions in Anthropic Messages API format.
pub fn all_tool_definitions() -> Vec<Value> {
    tools::all_tool_schemas()
        .into_iter()
        .map(|s| {
            json!({
                "name": s.name,
                "description": s.description,
                "input_schema": s.input_schema,
            })
        })
        .collect()
}

/// Execute a tool and return the output string.
pub fn execute_tool(name: &str, input: &Value, db: &mut Database) -> String {
    tools::execute_tool(name, input, db).output
}

/// Summarize a tool result for display in the UI (short one-liner).
pub fn summarize_tool_result(name: &str, result: &str) -> String {
    let char_count = result.len();
    match name {
        "list_recordings" => {
            let count = result.matches("\"id\"").count();
            format!("{} recordings", count)
        }
        "get_recording" => {
            if result.contains("not found") {
                "not found".to_string()
            } else {
                // Show what metadata is available
                let mut has = Vec::new();
                if result.contains("\"summary\"") { has.push("summary"); }
                if result.contains("\"topics\"") { has.push("topics"); }
                if result.contains("\"key_points\"") { has.push("key points"); }
                if result.contains("\"action_items\"") { has.push("actions"); }
                if has.is_empty() { "metadata".to_string() } else { has.join(", ") }
            }
        }
        "get_transcript" => {
            let word_count = result.split_whitespace().count();
            format!("transcript, {} words", word_count)
        }
        "search_transcripts" => {
            let count = result.matches("\"recording_id\"").count();
            format!("{} results", count)
        }
        "list_entities" => {
            let count = result.matches("\"id\"").count();
            format!("{} entities", count)
        }
        "get_entity" => {
            if result.contains("not found") {
                "not found".to_string()
            } else {
                format!("{} chars", char_count)
            }
        }
        "get_recordings_for_entity" => {
            let count = result.matches("\"id\"").count();
            format!("{} recordings", count)
        }
        "get_world_context" => {
            if result.is_empty() || result == "(no world context)" {
                "empty".to_string()
            } else {
                let word_count = result.split_whitespace().count();
                format!("{} words", word_count)
            }
        }
        "get_stats" => "ok".to_string(),
        // Write tools
        "create_entity" | "update_entity" | "add_entity_alias" | "remove_entity_alias"
        | "merge_entities" | "delete_entity" => {
            if result.contains("\"success\": true") {
                "ok".to_string()
            } else {
                "error".to_string()
            }
        }
        _ => format!("{} chars", char_count),
    }
}

/// Summarize tool input for display in the TUI.
pub fn summarize_input(name: &str, input: &Value) -> String {
    match name {
        "search_transcripts" => {
            let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("?");
            let mut parts = vec![format!("\"{}\"", query)];
            if let Some(from) = input.get("from_date").and_then(|v| v.as_str()) {
                parts.push(format!("from {}", from));
            }
            if let Some(to) = input.get("to_date").and_then(|v| v.as_str()) {
                parts.push(format!("to {}", to));
            }
            parts.join(", ")
        }
        "get_recording" | "get_transcript" => {
            let id = input
                .get("id")
                .or_else(|| input.get("recording_id"))
                .and_then(|v| v.as_i64())
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".to_string());
            format!("id={}", id)
        }
        "get_entity" => {
            let id = input
                .get("id")
                .and_then(|v| v.as_i64())
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".to_string());
            format!("id={}", id)
        }
        "get_recordings_for_entity" => {
            let id = input
                .get("entity_id")
                .and_then(|v| v.as_i64())
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".to_string());
            format!("entity_id={}", id)
        }
        "list_recordings" => {
            let mut parts = Vec::new();
            if let Some(from) = input.get("from_date").and_then(|v| v.as_str()) {
                parts.push(format!("from {}", from));
            }
            if let Some(to) = input.get("to_date").and_then(|v| v.as_str()) {
                parts.push(format!("to {}", to));
            }
            if let Some(limit) = input.get("limit").and_then(|v| v.as_i64()) {
                parts.push(format!("limit={}", limit));
            }
            if parts.is_empty() { "all".to_string() } else { parts.join(", ") }
        }
        "list_entities" => {
            let mut parts = Vec::new();
            if let Some(t) = input.get("entity_type").and_then(|v| v.as_str()) {
                parts.push(format!("type={}", t));
            }
            if let Some(l) = input.get("limit").and_then(|v| v.as_i64()) {
                parts.push(format!("limit={}", l));
            }
            if parts.is_empty() {
                "all".to_string()
            } else {
                parts.join(", ")
            }
        }
        // Write tools
        "create_entity" => {
            let name = input.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            let etype = input.get("entity_type").and_then(|v| v.as_str()).unwrap_or("?");
            format!("{} ({})", name, etype)
        }
        "update_entity" | "add_entity_alias" | "remove_entity_alias" | "delete_entity" => {
            let id = input
                .get("entity_id")
                .and_then(|v| v.as_i64())
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".to_string());
            format!("entity_id={}", id)
        }
        "merge_entities" => {
            let src = input.get("source_id").and_then(|v| v.as_i64()).unwrap_or(0);
            let tgt = input.get("target_id").and_then(|v| v.as_i64()).unwrap_or(0);
            format!("{} -> {}", src, tgt)
        }
        _ => String::new(),
    }
}
