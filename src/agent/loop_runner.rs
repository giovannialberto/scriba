//! Agent loop — orchestrates tool-use chat across any LLM provider.
//!
//! The loop sends messages to the LLM via an `AgentProvider`, collects tool
//! calls from the response, executes them, appends the results, and continues
//! until the model stops calling tools or the iteration limit is reached.

use genai::chat::{ChatMessage, ToolCall, ToolResponse};
use tokio::sync::mpsc;

use super::provider::AgentProvider;
use super::tools;
use crate::database::Database;

/// Events emitted by the agent loop for the TUI to display.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Status message (e.g. "Thinking...")
    Status(String),
    /// Non-fatal problem the user should see and keep (e.g. degraded mode)
    Warning(String),
    /// Text chunk from the assistant's response
    Chunk(String),
    /// Agent is calling a tool
    ToolCall { name: String, input_summary: String },
    /// Tool returned a result
    ToolResult {
        name: String,
        output_summary: String,
    },
    /// Token usage from the API response
    Usage {
        input_tokens: u32,
        output_tokens: u32,
    },
    /// Agent finished
    Done,
    /// Error occurred
    Error(String),
}

const MAX_ITERATIONS: usize = 10;

/// Run the agent loop. Streams events to `tx` for the TUI.
///
/// `provider` — the model transport.
/// `system_prompt` — the system message for the agent.
/// `history` — previous conversation turns as `("User" | "Assistant", text)`.
/// `user_message` — the latest user message.
/// `tx` — channel to send events to the TUI.
pub async fn run_agent_loop(
    provider: Box<dyn AgentProvider>,
    system_prompt: String,
    history: Vec<(String, String)>,
    user_message: String,
    tx: mpsc::Sender<AgentEvent>,
) {
    let mut db = match Database::new() {
        Ok(db) => db,
        Err(e) => {
            let _ = tx
                .send(AgentEvent::Error(format!("Database error: {}", e)))
                .await;
            return;
        }
    };

    let tool_defs = tools::all_tool_definitions();
    let mut messages = messages_from_history(&history, &user_message);

    let mut iterations = 0;
    let mut has_emitted_text = false;

    loop {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            let _ = tx
                .send(AgentEvent::Status("Reached maximum iterations".to_string()))
                .await;
            let _ = tx.send(AgentEvent::Done).await;
            break;
        }

        // If this is a continuation after tool use and we already emitted text,
        // add a line break so the next text doesn't stick to the previous one.
        if iterations > 1 && has_emitted_text {
            let _ = tx.send(AgentEvent::Chunk("\n\n".to_string())).await;
        }

        let _ = tx.send(AgentEvent::Status("Thinking...".to_string())).await;

        let turn = match provider
            .send_turn(&system_prompt, &messages, &tool_defs, &tx)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let _ = tx
                    .send(AgentEvent::Error(format!(
                        "{} error: {}",
                        provider.display_name(),
                        e
                    )))
                    .await;
                return;
            }
        };

        // Emit usage so the TUI can update the context window bar
        let _ = tx
            .send(AgentEvent::Usage {
                input_tokens: turn.input_tokens,
                output_tokens: turn.output_tokens,
            })
            .await;

        if turn.content.texts().iter().any(|t| !t.is_empty()) {
            has_emitted_text = true;
        }

        let tool_calls: Vec<ToolCall> = turn.content.tool_calls().into_iter().cloned().collect();

        // Keep the assistant turn verbatim (text + tool calls + provider parts)
        messages.push(ChatMessage::assistant(turn.content));

        if turn.should_stop || tool_calls.is_empty() {
            let _ = tx.send(AgentEvent::Done).await;
            break;
        }

        // Execute each tool and collect results
        let mut responses: Vec<ToolResponse> = Vec::with_capacity(tool_calls.len());
        for call in &tool_calls {
            let input_summary = tools::summarize_input(&call.fn_name, &call.fn_arguments);
            let _ = tx
                .send(AgentEvent::ToolCall {
                    name: call.fn_name.clone(),
                    input_summary,
                })
                .await;

            let output = tools::execute_tool(&call.fn_name, &call.fn_arguments, &mut db);
            let output_summary = tools::summarize_tool_result(&call.fn_name, &output);

            let _ = tx
                .send(AgentEvent::ToolResult {
                    name: call.fn_name.clone(),
                    output_summary,
                })
                .await;

            responses.push(ToolResponse::from_tool_call(call, output));
        }

        // All results for this batch go back in a single tool message
        messages.push(ChatMessage::from(responses));
    }
}

/// Convert the TUI's flat text history into model messages, then append the
/// new user message. Empty turns are dropped because some providers reject
/// empty content blocks.
pub fn messages_from_history(history: &[(String, String)], user_message: &str) -> Vec<ChatMessage> {
    let mut messages = Vec::with_capacity(history.len() + 1);
    for (role, content) in history {
        if content.trim().is_empty() {
            continue;
        }
        match role.as_str() {
            "User" => messages.push(ChatMessage::user(content.clone())),
            "Assistant" => messages.push(ChatMessage::assistant(content.clone())),
            _ => {}
        }
    }
    messages.push(ChatMessage::user(user_message.to_string()));
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use genai::chat::ChatRole;

    fn text_of(msg: &ChatMessage) -> String {
        msg.content.texts().join("")
    }

    #[test]
    fn history_maps_roles_and_appends_user_message() {
        let history = vec![
            ("User".to_string(), "hello".to_string()),
            ("Assistant".to_string(), "hi there".to_string()),
        ];
        let messages = messages_from_history(&history, "what's new?");

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, ChatRole::User);
        assert_eq!(text_of(&messages[0]), "hello");
        assert_eq!(messages[1].role, ChatRole::Assistant);
        assert_eq!(text_of(&messages[1]), "hi there");
        assert_eq!(messages[2].role, ChatRole::User);
        assert_eq!(text_of(&messages[2]), "what's new?");
    }

    #[test]
    fn history_drops_empty_and_unknown_turns() {
        let history = vec![
            ("User".to_string(), "   ".to_string()),
            ("System".to_string(), "ignored".to_string()),
            ("Assistant".to_string(), "kept".to_string()),
        ];
        let messages = messages_from_history(&history, "q");

        assert_eq!(messages.len(), 2);
        assert_eq!(text_of(&messages[0]), "kept");
        assert_eq!(text_of(&messages[1]), "q");
    }
}
