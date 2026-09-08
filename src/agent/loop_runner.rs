//! Agent loop — orchestrates tool-use chat across any LLM provider.
//!
//! The loop sends messages to the LLM via an `AgentProvider`, collects tool
//! calls from the response, executes them (off the async runtime), appends
//! the results, and continues until the model stops calling tools. When the
//! tool-call budget runs out it asks the model for a final answer instead of
//! stopping silently. The messages added during a turn are handed back to the
//! caller so the next turn can continue from the full transcript.

use genai::chat::{ChatMessage, ChatRole, ContentPart, Tool, ToolCall, ToolResponse};
use serde_json::Value;
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
    ToolResult { name: String, output_summary: String },
    /// Token usage from the API response
    Usage { input_tokens: u32, output_tokens: u32 },
    /// Messages appended to the model-facing transcript during this turn
    /// (user message, assistant turns, tool results). Emitted before `Done`
    /// or `Error` so the caller can keep history for the next turn.
    Transcript(Vec<ChatMessage>),
    /// Agent finished
    Done,
    /// Error occurred
    Error(String),
}

/// Maximum rounds of tool calls per user turn before the model is asked to
/// answer with what it has.
pub const MAX_TOOL_ROUNDS: usize = 12;

const WRAP_UP_INSTRUCTION: &str = "You have used all the tool calls available for this turn. \
Answer now with what you have gathered so far and say plainly what you could not verify. \
Do not call any more tools.";

/// Executes tool calls on behalf of the loop. Runs on a blocking thread so a
/// slow query never stalls streaming.
pub trait ToolRunner: Send + 'static {
    fn run(&mut self, name: &str, input: &Value) -> String;
}

/// The production runner: every tool works against the SQLite database.
pub struct DbToolRunner {
    db: Database,
}

impl DbToolRunner {
    pub fn open() -> anyhow::Result<Self> {
        Ok(Self { db: Database::new()? })
    }
}

impl ToolRunner for DbToolRunner {
    fn run(&mut self, name: &str, input: &Value) -> String {
        tools::execute_tool(name, input, &mut self.db)
    }
}

/// Run the agent loop against the database. Streams events to `tx`.
///
/// `history` — the model-facing transcript from previous turns.
/// `user_message` — the latest user message.
pub async fn run_agent_loop(
    provider: Box<dyn AgentProvider>,
    system_prompt: String,
    history: Vec<ChatMessage>,
    user_message: String,
    tx: mpsc::Sender<AgentEvent>,
) {
    let runner = match DbToolRunner::open() {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(AgentEvent::Error(format!("Database error: {}", e))).await;
            return;
        }
    };
    run_agent_loop_with(provider, system_prompt, history, user_message, tx, runner).await;
}

/// Run the agent loop with a custom tool runner (used by tests).
pub async fn run_agent_loop_with<R: ToolRunner>(
    provider: Box<dyn AgentProvider>,
    system_prompt: String,
    history: Vec<ChatMessage>,
    user_message: String,
    tx: mpsc::Sender<AgentEvent>,
    mut runner: R,
) {
    let tool_defs = tools::all_tool_definitions();
    let mut messages = history;
    let start = messages.len();
    messages.push(ChatMessage::user(user_message));

    let mut rounds = 0;
    let mut has_emitted_text = false;

    loop {
        // If this is a continuation after tool use and we already emitted text,
        // add a line break so the next text doesn't stick to the previous one.
        if rounds > 0 && has_emitted_text {
            let _ = tx.send(AgentEvent::Chunk("\n\n".to_string())).await;
        }

        let wrapping_up = rounds >= MAX_TOOL_ROUNDS;
        if wrapping_up {
            let _ = tx
                .send(AgentEvent::Status(
                    "Reached the tool-call limit, answering with what was found".to_string(),
                ))
                .await;
            messages.push(ChatMessage::user(WRAP_UP_INSTRUCTION));
        } else {
            let _ = tx.send(AgentEvent::Status("Thinking...".to_string())).await;
        }

        let tools_for_turn: &[Tool] = if wrapping_up { &[] } else { &tool_defs };
        let turn = match provider
            .send_turn(&system_prompt, &messages, tools_for_turn, &tx)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                // Keep only the user's message: an assistant turn without its
                // tool results would make the transcript invalid.
                let _ = tx
                    .send(AgentEvent::Transcript(vec![messages[start].clone()]))
                    .await;
                let _ = tx
                    .send(AgentEvent::Error(format!("{} error: {}", provider.display_name(), e)))
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

        if turn.should_stop || tool_calls.is_empty() || wrapping_up {
            break;
        }
        rounds += 1;

        // Execute the tools on a blocking thread; the UI keeps rendering.
        let blocking_tx = tx.clone();
        let executed = tokio::task::spawn_blocking(move || {
            let mut responses: Vec<ToolResponse> = Vec::with_capacity(tool_calls.len());
            for call in &tool_calls {
                let input_summary = tools::summarize_input(&call.fn_name, &call.fn_arguments);
                let _ = blocking_tx.blocking_send(AgentEvent::ToolCall {
                    name: call.fn_name.clone(),
                    input_summary,
                });

                let output = runner.run(&call.fn_name, &call.fn_arguments);
                let output_summary = tools::summarize_tool_result(&call.fn_name, &output);

                let _ = blocking_tx.blocking_send(AgentEvent::ToolResult {
                    name: call.fn_name.clone(),
                    output_summary,
                });

                responses.push(ToolResponse::from_tool_call(call, output));
            }
            (runner, responses)
        })
        .await;

        let responses = match executed {
            Ok((r, responses)) => {
                runner = r;
                responses
            }
            Err(e) => {
                let _ = tx
                    .send(AgentEvent::Transcript(vec![messages[start].clone()]))
                    .await;
                let _ = tx.send(AgentEvent::Error(format!("Tool execution failed: {}", e))).await;
                return;
            }
        };

        // All results for this batch go back in a single tool message
        messages.push(ChatMessage::from(responses));
    }

    let _ = tx.send(AgentEvent::Transcript(messages[start..].to_vec())).await;
    let _ = tx.send(AgentEvent::Done).await;
}

/// Render the model-facing transcript as `(role, text)` pairs for the
/// compaction prompt. Tool calls and results are summarized rather than
/// dumped, so the summary stays about what was learned, not raw payloads.
pub fn history_as_text_pairs(history: &[ChatMessage]) -> Vec<(String, String)> {
    const MAX_RESULT_CHARS: usize = 300;
    history
        .iter()
        .filter_map(|msg| {
            let role = match msg.role {
                ChatRole::User => "User",
                ChatRole::Assistant => "Assistant",
                ChatRole::Tool => "Tool",
                ChatRole::System => return None,
            };
            let mut parts: Vec<String> = Vec::new();
            for part in msg.content.parts() {
                match part {
                    ContentPart::Text(t) if !t.trim().is_empty() => parts.push(t.trim().to_string()),
                    ContentPart::ToolCall(call) => {
                        parts.push(format!("[called {}({})]", call.fn_name, call.fn_arguments))
                    }
                    ContentPart::ToolResponse(resp) => {
                        let name = resp.fn_name.as_deref().unwrap_or("tool");
                        let body: String = resp.content.chars().take(MAX_RESULT_CHARS).collect();
                        let ellipsis = if resp.content.chars().count() > MAX_RESULT_CHARS { "…" } else { "" };
                        parts.push(format!("[{} result: {}{}]", name, body, ellipsis));
                    }
                    _ => {}
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some((role.to_string(), parts.join("\n")))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::provider::AgentTurnResult;
    use crate::enrichment::ProviderError;
    use async_trait::async_trait;
    use genai::chat::MessageContent;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    /// Scripted provider: pops one pre-baked turn per call and records how
    /// many tools each call was offered.
    struct FakeProvider {
        turns: Mutex<VecDeque<Result<AgentTurnResult, ProviderError>>>,
        tools_offered: Arc<Mutex<Vec<usize>>>,
    }

    #[async_trait]
    impl AgentProvider for FakeProvider {
        async fn send_turn(
            &self,
            _system_prompt: &str,
            _messages: &[ChatMessage],
            tools: &[Tool],
            _tx: &mpsc::Sender<AgentEvent>,
        ) -> Result<AgentTurnResult, ProviderError> {
            self.tools_offered.lock().unwrap().push(tools.len());
            self.turns
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Ok(text_turn("fallback")))
        }

        async fn compact_history(&self, _prompt: &str) -> Result<String, ProviderError> {
            Ok("summary".into())
        }

        fn display_name(&self) -> &str {
            "Fake"
        }
    }

    struct FakeRunner {
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl ToolRunner for FakeRunner {
        fn run(&mut self, name: &str, _input: &Value) -> String {
            self.calls.lock().unwrap().push(name.to_string());
            r#"{"total_recordings": 3}"#.to_string()
        }
    }

    fn text_turn(text: &str) -> AgentTurnResult {
        AgentTurnResult {
            content: MessageContent::from_text(text),
            should_stop: true,
            input_tokens: 10,
            output_tokens: 5,
        }
    }

    fn tool_turn(name: &str) -> AgentTurnResult {
        AgentTurnResult {
            content: MessageContent::from_tool_calls(vec![ToolCall {
                call_id: format!("call-{name}"),
                fn_name: name.to_string(),
                fn_arguments: serde_json::json!({}),
                thought_signatures: None,
            }]),
            should_stop: false,
            input_tokens: 10,
            output_tokens: 5,
        }
    }

    async fn drive(
        turns: Vec<Result<AgentTurnResult, ProviderError>>,
        history: Vec<ChatMessage>,
    ) -> (Vec<AgentEvent>, Vec<usize>, Vec<String>) {
        let tools_offered = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let provider = Box::new(FakeProvider {
            turns: Mutex::new(turns.into()),
            tools_offered: tools_offered.clone(),
        });
        let runner = FakeRunner { calls: calls.clone() };
        let (tx, mut rx) = mpsc::channel(256);
        run_agent_loop_with(provider, "sys".into(), history, "question".into(), tx, runner).await;
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        let offered = tools_offered.lock().unwrap().clone();
        let calls = calls.lock().unwrap().clone();
        (events, offered, calls)
    }

    fn transcript(events: &[AgentEvent]) -> Vec<ChatMessage> {
        events
            .iter()
            .find_map(|e| match e {
                AgentEvent::Transcript(m) => Some(m.clone()),
                _ => None,
            })
            .expect("transcript emitted")
    }

    #[tokio::test]
    async fn tool_round_then_answer_yields_full_transcript() {
        let (events, offered, calls) =
            drive(vec![Ok(tool_turn("get_stats")), Ok(text_turn("You have 3."))], vec![]).await;

        assert_eq!(calls, vec!["get_stats"]);
        assert_eq!(offered.len(), 2);
        assert!(offered.iter().all(|n| *n > 0), "tools offered on every normal turn");
        assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolCall { name, .. } if name == "get_stats")));
        assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolResult { .. })));
        assert!(matches!(events.last(), Some(AgentEvent::Done)));

        let t = transcript(&events);
        assert_eq!(t.len(), 4, "user, assistant(tool call), tool results, assistant(text)");
        assert_eq!(t[0].role, ChatRole::User);
        assert_eq!(t[1].role, ChatRole::Assistant);
        assert_eq!(t[2].role, ChatRole::Tool);
        assert_eq!(t[3].role, ChatRole::Assistant);
        assert_eq!(t[3].content.texts().join(""), "You have 3.");
    }

    #[tokio::test]
    async fn prior_history_is_not_re_emitted() {
        let history = vec![ChatMessage::user("earlier"), ChatMessage::assistant("ok")];
        let (events, _, _) = drive(vec![Ok(text_turn("hi"))], history).await;
        let t = transcript(&events);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].content.texts().join(""), "question");
    }

    #[tokio::test]
    async fn tool_budget_ends_with_a_wrap_up_turn_without_tools() {
        let turns: Vec<_> = (0..MAX_TOOL_ROUNDS + 1)
            .map(|_| Ok(tool_turn("list_recordings")))
            .collect();
        let (events, offered, calls) = drive(turns, vec![]).await;

        assert_eq!(calls.len(), MAX_TOOL_ROUNDS);
        assert_eq!(offered.len(), MAX_TOOL_ROUNDS + 1);
        assert_eq!(*offered.last().unwrap(), 0, "wrap-up turn gets no tools");
        assert!(matches!(events.last(), Some(AgentEvent::Done)));

        let t = transcript(&events);
        let wrap_up = t.iter().rev().nth(1).unwrap();
        assert_eq!(wrap_up.role, ChatRole::User);
        assert!(wrap_up.content.texts().join("").contains("Do not call any more tools"));
    }

    #[tokio::test]
    async fn provider_error_keeps_only_the_user_message() {
        let err = ProviderError::AuthFailure { message: "nope".into() };
        let (events, _, calls) = drive(vec![Err(err)], vec![]).await;
        assert!(calls.is_empty());
        let t = transcript(&events);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].role, ChatRole::User);
        assert!(matches!(events.last(), Some(AgentEvent::Error(msg)) if msg.contains("Fake error")));
    }

    #[test]
    fn history_pairs_summarize_tool_traffic() {
        let long_result = "x".repeat(400);
        let history = vec![
            ChatMessage::user("what happened?"),
            ChatMessage::assistant(MessageContent::from_tool_calls(vec![ToolCall {
                call_id: "1".into(),
                fn_name: "get_stats".into(),
                fn_arguments: serde_json::json!({"limit": 5}),
                thought_signatures: None,
            }])),
            ChatMessage::from(vec![ToolResponse {
                call_id: "1".into(),
                fn_name: Some("get_stats".into()),
                content: long_result,
            }]),
            ChatMessage::assistant("Three recordings."),
            ChatMessage::system("ignored"),
        ];
        let pairs = history_as_text_pairs(&history);
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0], ("User".to_string(), "what happened?".to_string()));
        assert_eq!(pairs[1].1, r#"[called get_stats({"limit":5})]"#);
        assert!(pairs[2].1.starts_with("[get_stats result: xxx"));
        assert!(pairs[2].1.ends_with("…]"));
        assert!(pairs[2].1.len() < 340);
        assert_eq!(pairs[3].1, "Three recordings.");
    }
}
