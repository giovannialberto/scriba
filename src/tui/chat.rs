//! Chat module for "Ask Scriba" — types, state, rendering, and pipeline functions.

use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
    Frame,
};
use tokio::sync::mpsc;

use crate::enrichment::chat_prompts;

/// Accent color used throughout the UI (lavender/indigo #af87ff)
pub const ACCENT: Color = Color::Indexed(141);

// ─────────────────────────────────────────────────────────────────────────────
// Chat types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    Status(String),
    Chunk(String),
    ToolCall { name: String, input_summary: String },
    ToolResult { name: String, output_summary: String },
    Usage { input_tokens: u32, output_tokens: u32 },
    Compacted { summary: String, removed_count: usize },
    Done,
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatContext {
    Global,
    Recording { recording_id: i64, recording_name: String },
}

#[derive(Debug, Clone)]
pub struct ToolCallDisplay {
    pub name: String,
    pub input_summary: String,
    pub output_summary: String,
    pub is_complete: bool,
}

#[derive(Debug, Clone)]
pub enum ChatBlock {
    Text(String),
    ToolCall(ToolCallDisplay),
    CompactionMarker,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub blocks: Vec<ChatBlock>,
}

impl ChatMessage {
    /// Create a simple text-only message (user, system, or fallback assistant).
    pub fn text(role: ChatRole, content: String) -> Self {
        Self { role, blocks: vec![ChatBlock::Text(content)] }
    }

    /// Concatenate all text blocks for LLM conversation history.
    pub fn content(&self) -> String {
        let mut s = String::new();
        for block in &self.blocks {
            if let ChatBlock::Text(t) = block {
                s.push_str(t);
            }
        }
        s
    }
}

#[derive(Debug, Clone)]
pub struct HomeRecording {
    pub recording_id: i64,
    pub name: String,
    pub duration_mins: i64,
    pub summary_line: Option<String>,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ChatFocus {
    Table,
    ChatInput,
}

// ─────────────────────────────────────────────────────────────────────────────
// ChatState
// ─────────────────────────────────────────────────────────────────────────────

pub struct ChatState {
    pub context: ChatContext,
    pub messages: Vec<ChatMessage>,
    pub input_buffer: String,
    pub scroll_offset: usize,

    // Streaming state
    pub stream_rx: Option<mpsc::Receiver<ChatStreamEvent>>,
    pub current_status: Option<String>,
    pub is_generating: bool,

    // Blocks accumulated during the current generation (text + tool calls, in order)
    pub pending_blocks: Vec<ChatBlock>,

    // Generation task handle
    pub generation_task: Option<tokio::task::JoinHandle<()>>,

    // Suggestions
    pub suggestions: Vec<String>,
    pub show_suggestions: bool,
    pub selected_suggestion: usize,

    // Pre-built system prompt
    pub system_prompt: String,

    // Focus
    pub focus: ChatFocus,

    // Home screen greeting (set from Dashboard)
    pub greeting_text: String,
    pub greeting_subtitle: String,

    // Borderless mode (no Block borders when true)
    pub borderless: bool,

    // Home screen timeline
    pub home_recordings: Vec<HomeRecording>,
    pub selected_action: usize,
    pub show_home_screen: bool,

    // Quick action menu (shown on Enter for a home recording)
    pub action_menu_open: bool,
    pub action_menu_selection: usize, // 0=transcript, 1=summarize, 2=ask

    // Placeholder hint shown in input box when empty
    pub placeholder: String,

    // Spinner frame
    pub spinner_frame: usize,

    // Auto-scroll: stays true until user manually scrolls up
    pub auto_scroll: bool,

    // Queued message: submitted while generating, will auto-send when done
    pub pending_message: Option<String>,

    // Last rendered chat panel area (for mouse hit-testing)
    pub panel_rect: Rect,

    // Total content lines (for scroll clamping in mouse handler)
    pub total_content_lines: usize,

    // Text selection state (click-drag to select, auto-copy on release)
    pub selection_anchor: Option<(usize, usize)>, // (content_line, char_col) where drag started
    pub selection_end: Option<(usize, usize)>,    // (content_line, char_col) current drag position
    pub content_texts: Vec<String>,               // plain text of each content line for selection highlight
    pub content_texts_raw: Vec<String>,           // same lines but without UI prefix chrome (for clipboard)

    // Render geometry cached for mouse hit-testing
    pub content_top_pad: usize,       // blank lines above content (home screen centering)
    pub content_border_overhead: u16, // border lines subtracted from height (0 if borderless)

    // Context window tracking (Anthropic only)
    pub context_window_max: u32,
    pub context_input_tokens: u32,
    pub context_output_tokens: u32,
    usage_baseline_set: bool, // true after first Usage event per generation

    // Rendering cache for completed messages
    cached_msg_lines: Vec<Line<'static>>,
    cached_msg_texts: Vec<String>,
    cached_msg_raw_texts: Vec<String>, // prefix-free version of cached_msg_texts for clipboard
    cached_msg_count: usize,
    cached_width: usize,
}

impl ChatState {
    pub fn new() -> Self {
        Self {
            context: ChatContext::Global,
            messages: Vec::new(),
            input_buffer: String::new(),
            scroll_offset: 0,
            stream_rx: None,
            current_status: None,
            is_generating: false,
            pending_blocks: Vec::new(),
            generation_task: None,
            suggestions: Vec::new(),
            show_suggestions: true,
            selected_suggestion: 0,
            system_prompt: String::new(),
            focus: ChatFocus::ChatInput,
            greeting_text: String::new(),
            greeting_subtitle: String::new(),
            borderless: false,
            home_recordings: Vec::new(),
            selected_action: 0,
            show_home_screen: true,
            action_menu_open: false,
            action_menu_selection: 0,
            placeholder: String::new(),
            spinner_frame: 0,
            auto_scroll: true,
            pending_message: None,
            panel_rect: Rect::default(),
            total_content_lines: 0,
            selection_anchor: None,
            selection_end: None,
            content_texts: Vec::new(),
            content_texts_raw: Vec::new(),
            content_top_pad: 0,
            content_border_overhead: 2,
            context_window_max: std::env::var("SCRIBA_CTX_LIMIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(200_000),
            context_input_tokens: 0,
            context_output_tokens: 0,
            usage_baseline_set: false,
            cached_msg_lines: Vec::new(),
            cached_msg_texts: Vec::new(),
            cached_msg_raw_texts: Vec::new(),
            cached_msg_count: 0,
            cached_width: 0,
        }
    }

    /// Invalidate the render cache (e.g. after clearing messages).
    pub fn invalidate_cache(&mut self) {
        self.cached_msg_count = 0;
    }

    pub fn context_usage_fraction(&self) -> f64 {
        let total = self.context_input_tokens + self.context_output_tokens;
        total as f64 / self.context_window_max as f64
    }

    pub fn needs_compaction(&self) -> bool {
        self.context_usage_fraction() > 0.80
    }

    // ── Stream Polling ──────────────────────────────────────────────────────

    /// Poll the chat stream for new events. Returns `true` if a pending message
    /// should be re-sent (i.e. generation finished and a queued message exists).
    pub fn poll_stream(&mut self) -> bool {
        let mut should_resend = false;
        if let Some(ref mut rx) = self.stream_rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    ChatStreamEvent::Status(msg) => {
                        self.current_status = Some(msg);
                    }
                    ChatStreamEvent::Chunk(text) => {
                        // Append text to the last Text block, or create one
                        if let Some(ChatBlock::Text(ref mut s)) = self.pending_blocks.last_mut() {
                            s.push_str(&text);
                        } else {
                            self.pending_blocks.push(ChatBlock::Text(text));
                        }
                        // Text is rendered progressively — clear status so it doesn't compete
                        self.current_status = None;
                    }
                    ChatStreamEvent::ToolCall { name, input_summary } => {
                        self.current_status = None; // tool call line is the indicator
                        self.pending_blocks.push(ChatBlock::ToolCall(ToolCallDisplay {
                            name,
                            input_summary,
                            output_summary: String::new(),
                            is_complete: false,
                        }));
                    }
                    ChatStreamEvent::ToolResult { name, output_summary } => {
                        // Find last matching incomplete tool call in blocks
                        for block in self.pending_blocks.iter_mut().rev() {
                            if let ChatBlock::ToolCall(ref mut tc) = block {
                                if tc.name == name && !tc.is_complete {
                                    tc.output_summary = output_summary.clone();
                                    tc.is_complete = true;
                                    break;
                                }
                            }
                        }
                        self.current_status = Some("Thinking...".to_string());
                    }
                    ChatStreamEvent::Usage { input_tokens, output_tokens } => {
                        // Only update input_tokens from the FIRST API call of each
                        // generation. That call reflects the true conversation baseline.
                        // Subsequent calls within the same turn include transient tool-use
                        // overhead that would inflate the bar and cause apparent "resets"
                        // when the next turn starts with a smaller baseline.
                        if !self.usage_baseline_set {
                            self.context_input_tokens = input_tokens;
                            self.usage_baseline_set = true;
                        }
                        // Always update output_tokens (current response contribution)
                        self.context_output_tokens = output_tokens;
                    }
                    ChatStreamEvent::Compacted { summary, removed_count } => {
                        let total_msgs = self.messages.len();
                        if removed_count <= total_msgs {
                            let remaining: Vec<ChatMessage> = self.messages.drain(removed_count..).collect();
                            self.messages.clear();
                            self.messages.push(ChatMessage::text(
                                ChatRole::System,
                                format!("Context compacted: {} messages summarized, {} kept",
                                    removed_count, remaining.len()),
                            ));
                            self.messages.push(ChatMessage {
                                role: ChatRole::System,
                                blocks: vec![ChatBlock::CompactionMarker],
                            });
                            self.messages.push(ChatMessage::text(ChatRole::System, summary));
                            self.messages.extend(remaining);
                        } else {
                            // Mismatch — just insert the summary without restructuring
                            self.messages.push(ChatMessage {
                                role: ChatRole::System,
                                blocks: vec![ChatBlock::CompactionMarker],
                            });
                            self.messages.push(ChatMessage::text(ChatRole::System, summary));
                        }
                        self.cached_msg_count = 0;
                        self.context_input_tokens = 0;
                        self.context_output_tokens = 0;
                        self.usage_baseline_set = false;
                    }
                    ChatStreamEvent::Done => {
                        let blocks = std::mem::take(&mut self.pending_blocks);
                        self.messages.push(ChatMessage {
                            role: ChatRole::Assistant,
                            blocks,
                        });
                        self.is_generating = false;
                        self.stream_rx = None;
                        self.current_status = None;
                        self.usage_baseline_set = false;
                        if self.pending_message.is_some() {
                            should_resend = true;
                        }
                        return should_resend;
                    }
                    ChatStreamEvent::Error(msg) => {
                        self.pending_blocks.clear();
                        self.messages.push(ChatMessage::text(
                            ChatRole::System,
                            format!("Error: {}", msg),
                        ));
                        self.is_generating = false;
                        self.stream_rx = None;
                        self.current_status = None;
                        self.usage_baseline_set = false;
                        self.pending_message = None;
                        return false;
                    }
                }
            }
        }
        should_resend
    }

    // ── Mouse Helpers ───────────────────────────────────────────────────────

    /// Map a mouse position to a (content_line, char_col) in the chat content.
    pub fn mouse_to_content_pos(&self, mouse_col: u16, mouse_row: u16) -> Option<(usize, usize)> {
        let rect = self.panel_rect;
        // Subtract the actual border overhead (0 if borderless, 1 if bordered)
        let border_top = if self.content_border_overhead > 0 { 1u16 } else { 0u16 };
        let border_left = border_top;
        let click_row = (mouse_row.saturating_sub(rect.y + border_top)) as usize;
        let char_col = (mouse_col.saturating_sub(rect.x + border_left)) as usize;

        let inner_height = rect.height.saturating_sub(self.content_border_overhead) as usize;
        let has_conv = !self.messages.is_empty() || self.is_generating;
        let reserved = if has_conv { 2 } else { 1 };
        let chat_height = inner_height.saturating_sub(reserved);

        // Account for home-screen vertical centering top padding
        let top_pad = self.content_top_pad;
        if click_row < top_pad {
            return None;
        }
        let click_row_in_content = click_row - top_pad;

        let scroll_y = if self.auto_scroll || self.total_content_lines <= chat_height {
            self.total_content_lines.saturating_sub(chat_height)
        } else {
            let max_scroll = self.total_content_lines.saturating_sub(chat_height);
            self.scroll_offset.min(max_scroll)
        };

        let lines_to_show = self.total_content_lines.saturating_sub(scroll_y);
        let pad = chat_height.saturating_sub(lines_to_show.min(chat_height));
        if click_row_in_content < pad {
            return None;
        }
        let content_line = scroll_y + (click_row_in_content - pad);
        if content_line >= self.total_content_lines {
            return None;
        }
        Some((content_line, char_col))
    }

    /// Extract the plain text between two content positions.
    /// Uses `content_texts_raw` (no UI prefixes) so pasted text is clean.
    pub fn extract_selected_text(&self, anchor: (usize, usize), end: (usize, usize)) -> String {
        let (start, end) = if anchor <= end { (anchor, end) } else { (end, anchor) };
        let (start_line, start_col) = start;
        let (end_line, end_col) = end;

        // Use raw (prefix-free) texts for clipboard; fall back to content_texts if not populated
        let texts = if !self.content_texts_raw.is_empty() {
            &self.content_texts_raw
        } else {
            &self.content_texts
        };
        if texts.is_empty() || start_line >= texts.len() {
            return String::new();
        }

        // Compute per-line prefix lengths from the visual content_texts so selection
        // coordinates (which reference the visual lines) map correctly into the raw text.
        let prefix_lens: Vec<usize> = self.content_texts.iter().zip(texts.iter()).map(|(vis, raw)| {
            vis.chars().count().saturating_sub(raw.chars().count())
        }).collect();

        let mut lines: Vec<String> = Vec::new();
        for line_idx in start_line..=end_line.min(texts.len() - 1) {
            let raw_text = &texts[line_idx];
            let raw_chars: Vec<char> = raw_text.chars().collect();
            let prefix_len = prefix_lens.get(line_idx).copied().unwrap_or(0);

            // Translate visual column coordinates into raw-text column coordinates
            let raw_from = if line_idx == start_line {
                start_col.saturating_sub(prefix_len).min(raw_chars.len())
            } else {
                0
            };
            let raw_to = if line_idx == end_line {
                end_col.saturating_sub(prefix_len).min(raw_chars.len())
            } else {
                raw_chars.len()
            };

            if raw_from < raw_to {
                let slice: String = raw_chars[raw_from..raw_to].iter().collect();
                lines.push(slice.trim_end().to_string());
            } else {
                // Empty/blank line — preserve it as a paragraph separator
                lines.push(String::new());
            }
        }

        // Join lines, then trim trailing blank lines only
        lines.join("\n").trim_end().to_string()
    }

    // ── Rendering ───────────────────────────────────────────────────────────

    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        self.panel_rect = area;
        if area.height < 3 {
            return;
        }

        let is_focused = self.focus == ChatFocus::ChatInput;
        let border_color = if is_focused { ACCENT } else { Color::DarkGray };

        let title = match &self.context {
            ChatContext::Global => "Ask Scriba",
            ChatContext::Recording { .. } => "Ask about this recording",
        };

        let border_overhead = if self.borderless { 0u16 } else { 2u16 };
        self.content_border_overhead = border_overhead;
        self.content_top_pad = 0; // reset; home screen will update below
        let inner_height = area.height.saturating_sub(border_overhead) as usize;
        let has_conversation = !self.messages.is_empty() || self.is_generating;
        let padding = if self.borderless { 2usize } else { 4usize };
        let content_width = area.width.saturating_sub(padding as u16) as usize;

        // Home screen timeline (replaces old suggestions in borderless mode)
        let show_home = self.show_home_screen
            && self.borderless
            && self.messages.is_empty()
            && self.pending_blocks.is_empty()
            && !self.is_generating;

        // On home screen the input is rendered inline — no bottom reservation needed
        let reserved = if show_home {
            0
        } else {
            let input_line_count = if !self.input_buffer.is_empty() {
                let display = format!("{}\u{2588}", self.input_buffer);
                let wrap_width = content_width.saturating_sub(4);
                if wrap_width > 0 { textwrap::wrap(&display, wrap_width).len() } else { 1 }
            } else {
                1
            };
            if has_conversation { 1 + input_line_count } else { input_line_count }
        };
        let chat_height = inner_height.saturating_sub(reserved);

        let mut final_lines: Vec<Line> = Vec::with_capacity(inner_height);

        let show_suggestions = self.show_suggestions
            && !self.suggestions.is_empty()
            && self.messages.is_empty()
            && self.pending_blocks.is_empty()
            && !self.is_generating;

        // Compute selection range (normalized: start <= end)
        let selection = match (self.selection_anchor, self.selection_end) {
            (Some(a), Some(e)) => {
                let (start, end) = if a <= e { (a, e) } else { (e, a) };
                Some((start, end))
            }
            _ => None,
        };

        if show_home {
            // ── Home screen: ASCII logo + timeline + recording tree ──────
            let mut all_lines: Vec<Line> = Vec::new();
            let mut content_texts: Vec<String> = Vec::new();
            let margin = "   ";
            let dim = Style::default().fg(Color::DarkGray);

            // Block art logo (centered, lavender accent)
            let logo_lines = [
                "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} \u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}  \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} ",
                "\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}",
                "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}     \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2554}\u{255D}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2554}\u{255D}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2551}",
                "\u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2551}     \u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2551}",
                "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2551}\u{255A}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}  \u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2554}\u{255D}\u{2588}\u{2588}\u{2551}  \u{2588}\u{2588}\u{2551}",
                "\u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D} \u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D}\u{255A}\u{2550}\u{255D}  \u{255A}\u{2550}\u{255D}\u{255A}\u{2550}\u{255D}\u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D} \u{255A}\u{2550}\u{255D}  \u{255A}\u{2550}\u{255D}",
            ];
            let logo_width = 43;
            let logo_pad = content_width.saturating_sub(logo_width) / 2;
            let logo_padding = " ".repeat(logo_pad);
            let logo_style = Style::default().fg(Color::Indexed(60)); // muted indigo, subtle
            for logo_line in &logo_lines {
                let text = format!("{}{}", logo_padding, logo_line);
                content_texts.push(text.clone());
                all_lines.push(Line::from(Span::styled(text, logo_style)));
            }

            // Blank line
            all_lines.push(Line::from(""));
            content_texts.push(String::new());

            // Timeline entry 1: System initialization
            let init_text = format!("{}\u{25CB}  System initialization complete.", margin);
            content_texts.push(init_text.clone());
            all_lines.push(Line::from(Span::styled(init_text, dim)));

            // Blank line
            all_lines.push(Line::from(""));
            content_texts.push(String::new());

            // Timeline entry 2: Welcome greeting
            let greeting = format!("{}\u{25CB}  {}", margin, self.greeting_text);
            content_texts.push(greeting.clone());
            all_lines.push(Line::from(Span::styled(
                greeting,
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )));

            // Subtitle
            if !self.greeting_subtitle.is_empty() {
                let sub = format!("{}   {}", margin, self.greeting_subtitle);
                content_texts.push(sub.clone());
                all_lines.push(Line::from(Span::styled(sub, dim)));
            }

            // Recording tree — constrain to chat box width
            let rec_count = self.home_recordings.len();
            let action_labels = ["View transcript", "Summarize", "Ask about it"];
            let tree_max = content_width.saturating_sub(2); // match chat box width
            for (i, rec) in self.home_recordings.iter().enumerate() {
                let is_last = i == rec_count - 1;
                let connector = if is_last { "\u{2514}\u{2500}\u{2500}" } else { "\u{251C}\u{2500}\u{2500}" };
                let is_selected = i == self.selected_action;

                // Recording name line — truncate name to fit
                let prefix = format!("{}   {} ", margin, connector);
                let suffix = format!(" ({}m)", rec.duration_mins);
                let max_name = tree_max.saturating_sub(prefix.chars().count() + suffix.chars().count());
                let name: String = if rec.name.chars().count() > max_name {
                    rec.name.chars().take(max_name.saturating_sub(1)).collect::<String>() + "\u{2026}"
                } else {
                    rec.name.clone()
                };
                let rec_line = format!("{}{}{}", prefix, name, suffix);
                content_texts.push(rec_line.clone());
                let rec_style = if is_selected {
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                all_lines.push(Line::from(Span::styled(rec_line, rec_style)));

                // Summary line if present — truncate to fit
                let vert = if is_last { " " } else { "\u{2502}" };
                if let Some(ref summary) = rec.summary_line {
                    let sum_prefix = format!("{}   {}   ", margin, vert);
                    let max_sum = tree_max.saturating_sub(sum_prefix.chars().count());
                    let truncated: String = if summary.chars().count() > max_sum {
                        summary.chars().take(max_sum.saturating_sub(1)).collect::<String>() + "\u{2026}"
                    } else {
                        summary.clone()
                    };
                    let sum_line = format!("{}{}", sum_prefix, truncated);
                    content_texts.push(sum_line.clone());
                    all_lines.push(Line::from(Span::styled(sum_line, dim)));
                }

                // Quick action menu (inline, under the selected recording)
                if is_selected && self.action_menu_open {
                    for (ai, label) in action_labels.iter().enumerate() {
                        let bullet = if ai == self.action_menu_selection { "\u{25B8}" } else { " " };
                        let menu_line = format!("{}   {}    {} {}", margin, vert, bullet, label);
                        content_texts.push(menu_line.clone());
                        let menu_style = if ai == self.action_menu_selection {
                            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::DarkGray)
                        };
                        all_lines.push(Line::from(Span::styled(menu_line, menu_style)));
                    }
                }
            }

            // ── Chat input area (grey background, no borders) ─────────────
            let box_height = 7; // total lines for the input area
            let bg = Color::DarkGray; // matches the summary text color tone
            let box_width = content_width.saturating_sub(2);
            let inner_width = box_width.saturating_sub(6); // prompt + side padding

            // Blank line before input area
            all_lines.push(Line::from(""));
            content_texts.push(String::new());

            // Build input content lines
            let cursor_visible = is_focused && (self.spinner_frame % 5 < 3);
            let cursor_char = if cursor_visible { "\u{2588}" } else { " " };
            let bg_style = Style::default().bg(bg);

            let mut input_lines: Vec<Line> = Vec::new();
            let mut input_texts: Vec<String> = Vec::new();

            // Top padding line (empty, with background)
            let full_bg = " ".repeat(box_width);
            input_texts.push(full_bg.clone());
            input_lines.push(Line::from(Span::styled(format!(" {}", full_bg), bg_style)));

            // Target total width per line: 1 + box_width (leading space + fill)
            let pad_left = "   "; // 3-char left padding inside box
            let pad_len = pad_left.len(); // 3
            if !self.input_buffer.is_empty() {
                let display = format!("{}{}", self.input_buffer, cursor_char);
                let wrap_w = inner_width + 2; // reclaim prompt space
                let wrapped = textwrap::wrap(&display, wrap_w);
                for w in &wrapped {
                    let w_len = w.chars().count();
                    // " " (1) + pad (3) + text + fill = 1 + box_width
                    let right_fill = (1 + box_width).saturating_sub(1 + pad_len + w_len);
                    let text = format!("{}{}{}", pad_left, w, " ".repeat(right_fill));
                    input_texts.push(text.clone());
                    input_lines.push(Line::from(vec![
                        Span::styled(" ", bg_style),
                        Span::styled(pad_left.to_string(), bg_style),
                        Span::styled(w.to_string(), Style::default().fg(Color::White).bg(bg)),
                        Span::styled(" ".repeat(right_fill), bg_style),
                    ]));
                }
            } else {
                // Empty input: cursor + placeholder hint (or just cursor)
                let hint = if !self.placeholder.is_empty() { self.placeholder.as_str() } else { "" };
                let hint_len = hint.chars().count();
                let total_content = 1 + hint_len; // cursor + hint
                let right_fill = (1 + box_width).saturating_sub(1 + pad_len + total_content);
                let text = format!("{}{}{}{}", pad_left, cursor_char, hint, " ".repeat(right_fill));
                input_texts.push(text.clone());
                input_lines.push(Line::from(vec![
                    Span::styled(" ", bg_style),
                    Span::styled(pad_left.to_string(), bg_style),
                    Span::styled(cursor_char.to_string(), Style::default().fg(Color::White).bg(bg)),
                    Span::styled(hint.to_string(), Style::default().fg(Color::Indexed(246)).bg(bg)),
                    Span::styled(" ".repeat(right_fill), bg_style),
                ]));
            }

            // Add input lines, then fill remaining height with empty bg rows
            let used = input_lines.len();
            for l in input_lines {
                all_lines.push(l);
            }
            content_texts.extend(input_texts);
            for _ in used..box_height {
                let bg_fill = " ".repeat(box_width);
                content_texts.push(bg_fill.clone());
                all_lines.push(Line::from(Span::styled(format!(" {}", bg_fill), bg_style)));
            }

            // ── Vertical centering (bias upper third) ────────────────────
            let total_content = all_lines.len();
            self.content_texts_raw = content_texts.clone(); // home screen: no meaningful prefixes
            self.content_texts = content_texts;
            self.total_content_lines = total_content;
            // Use inner_height (not chat_height) since we included the input inline
            let avail = inner_height;
            let top_pad = if total_content < avail {
                (avail - total_content) / 3
            } else {
                0
            };
            self.content_top_pad = top_pad;
            for _ in 0..top_pad {
                final_lines.push(Line::from(""));
            }
            let remaining = avail.saturating_sub(top_pad);
            for line in all_lines.into_iter().take(remaining) {
                final_lines.push(line);
            }
            let used = top_pad + remaining.min(total_content);
            for _ in used..avail {
                final_lines.push(Line::from(""));
            }
        } else if show_suggestions && !self.show_home_screen {
            // ── Chat suggestions (recording view or non-home) ────────────
            let mut all_lines: Vec<Line> = Vec::new();
            let mut content_texts: Vec<String> = Vec::new();

            let total_options = self.suggestions.len() + 1;
            for (i, s) in self.suggestions.iter().enumerate() {
                let bullet = if i == self.selected_suggestion { "\u{25B8}" } else { " " };
                let text = format!("  {} {}", bullet, s);
                let style = if i == self.selected_suggestion {
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                content_texts.push(text.clone());
                all_lines.push(Line::from(Span::styled(text, style)));
            }
            let free_form_idx = total_options - 1;
            let bullet = if self.selected_suggestion == free_form_idx { "\u{25B8}" } else { " " };
            let text = format!("  {} Ask Scriba anything...", bullet);
            let style = if self.selected_suggestion == free_form_idx {
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            content_texts.push(text.clone());
            all_lines.push(Line::from(Span::styled(text, style)));

            self.content_texts_raw = content_texts.clone(); // suggestions: raw = display (no meaningful prefix)
            self.content_texts = content_texts;
            self.content_top_pad = 0;
            let total_content = all_lines.len();
            self.total_content_lines = total_content;
            let scroll_y: u16 = if self.auto_scroll || total_content <= chat_height {
                total_content.saturating_sub(chat_height) as u16
            } else {
                let max_scroll = total_content.saturating_sub(chat_height);
                self.scroll_offset.min(max_scroll) as u16
            };
            let lines_to_show = total_content.saturating_sub(scroll_y as usize);
            let pad = chat_height.saturating_sub(lines_to_show.min(chat_height));
            for _ in 0..pad {
                final_lines.push(Line::from(""));
            }
            for (vis_idx, line) in all_lines.into_iter().skip(scroll_y as usize).take(chat_height).enumerate() {
                let content_idx = scroll_y as usize + vis_idx;
                if let Some(((sel_start_line, sel_start_col), (sel_end_line, sel_end_col))) = selection {
                    if content_idx >= sel_start_line && content_idx <= sel_end_line {
                        let line_start = if content_idx == sel_start_line { sel_start_col } else { 0 };
                        let line_text_len = self.content_texts.get(content_idx)
                            .map(|t| t.chars().count()).unwrap_or(0);
                        let line_end = if content_idx == sel_end_line { sel_end_col } else { line_text_len };
                        final_lines.push(apply_selection_highlight(line, line_start, line_end));
                        continue;
                    }
                }
                final_lines.push(line);
            }
        } else {
            // ── Phase 1: Completed messages (cached) ───────────────────────
            let msg_count = self.messages.len();
            let cache_valid = msg_count == self.cached_msg_count
                && content_width == self.cached_width;

            if !cache_valid {
                let mut cached_lines: Vec<Line<'static>> = Vec::new();
                let mut cached_texts: Vec<String> = Vec::new();
                let mut cached_raw_texts: Vec<String> = Vec::new();
                let wrap_width = content_width.saturating_sub(2);

                for msg in &self.messages {
                    match msg.role {
                        ChatRole::User => {
                            let content = msg.content();
                            for line in content.lines() {
                                let wrapped = textwrap::wrap(line, wrap_width);
                                if wrapped.is_empty() {
                                    cached_texts.push("  ".to_string());
                                    cached_raw_texts.push(String::new());
                                    cached_lines.push(Line::from("  ".to_string()));
                                } else {
                                    for w in wrapped.iter() {
                                        let text = format!(" \u{2502} {}", w);
                                        cached_texts.push(text.clone());
                                        cached_raw_texts.push(w.to_string());
                                        cached_lines.push(Line::from(vec![
                                            Span::styled(" \u{2502} ", Style::default().fg(Color::Indexed(60))),
                                            Span::styled(w.to_string(), Style::default().fg(Color::Indexed(249))),
                                        ]));
                                    }
                                }
                            }
                            cached_texts.push(String::new());
                            cached_raw_texts.push(String::new());
                            cached_lines.push(Line::from(""));
                        }
                        ChatRole::Assistant => {
                            // Render blocks in chronological order — no header label
                            for block in &msg.blocks {
                                match block {
                                    ChatBlock::ToolCall(tc) => {
                                        let before = cached_texts.len();
                                        render_tool_call_cached(tc, &mut cached_lines, &mut cached_texts);
                                        // For tool calls, raw text = same as display (no meaningful prefix to strip)
                                        let added = cached_texts.len() - before;
                                        for i in (cached_texts.len() - added)..cached_texts.len() {
                                            cached_raw_texts.push(cached_texts[i].trim_start().to_string());
                                        }
                                    }
                                    ChatBlock::Text(text) => {
                                        for wl in safe_markdown_lines(text, wrap_width) {
                                            let plain: String =
                                                wl.spans.iter().map(|s| s.content.as_ref()).collect();
                                            cached_texts.push(format!("  {}", plain));
                                            cached_raw_texts.push(plain.clone());
                                            let mut indented = vec![Span::raw("  ".to_string())];
                                            indented.extend(wl.spans);
                                            cached_lines.push(Line::from(indented));
                                        }
                                    }
                                    ChatBlock::CompactionMarker => {
                                        let dashes = "─".repeat((content_width.saturating_sub(22)) / 2);
                                        let marker = format!("  {} context compacted {}", dashes, dashes);
                                        cached_texts.push(marker.clone());
                                        cached_raw_texts.push(String::new()); // don't copy compaction markers
                                        cached_lines.push(Line::from(Span::styled(
                                            marker,
                                            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                                        )));
                                    }
                                }
                            }
                            cached_texts.push(String::new());
                            cached_raw_texts.push(String::new());
                            cached_lines.push(Line::from(""));
                        }
                        ChatRole::System => {
                            // Check for CompactionMarker blocks first
                            let has_marker = msg.blocks.iter().any(|b| matches!(b, ChatBlock::CompactionMarker));
                            if has_marker {
                                for block in &msg.blocks {
                                    match block {
                                        ChatBlock::CompactionMarker => {
                                            let dashes = "─".repeat((content_width.saturating_sub(22)) / 2);
                                            let marker = format!("  {} context compacted {}", dashes, dashes);
                                            cached_texts.push(marker.clone());
                                            cached_raw_texts.push(String::new());
                                            cached_lines.push(Line::from(Span::styled(
                                                marker,
                                                Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                                            )));
                                        }
                                        ChatBlock::Text(text) if !text.is_empty() => {
                                            let style = Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC);
                                            for line in text.lines() {
                                                let wrapped = textwrap::wrap(line, content_width);
                                                for w in &wrapped {
                                                    let full = w.to_string();
                                                    cached_texts.push(full.clone());
                                                    cached_raw_texts.push(full.clone());
                                                    cached_lines.push(Line::from(Span::styled(full, style)));
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            } else {
                                let content = msg.content();
                                let style = Style::default().fg(Color::Red).add_modifier(Modifier::ITALIC);
                                for line in content.lines() {
                                    let wrapped = textwrap::wrap(line, content_width);
                                    if wrapped.is_empty() {
                                        cached_texts.push(String::new());
                                        cached_raw_texts.push(String::new());
                                        cached_lines.push(Line::from(Span::styled(String::new(), style)));
                                    } else {
                                        for w in &wrapped {
                                            let full = w.to_string();
                                            cached_texts.push(full.clone());
                                            cached_raw_texts.push(full.clone());
                                            cached_lines.push(Line::from(Span::styled(full, style)));
                                        }
                                    }
                                }
                            }
                            cached_texts.push(String::new());
                            cached_raw_texts.push(String::new());
                            cached_lines.push(Line::from(""));
                        }
                    }
                }

                self.cached_msg_lines = cached_lines;
                self.cached_msg_texts = cached_texts;
                self.cached_msg_raw_texts = cached_raw_texts;
                self.cached_msg_count = msg_count;
                self.cached_width = content_width;
            }

            // ── Phase 2: Dynamic content (progressive block rendering) ────
            // Blocks render as they arrive: text with markdown, tool calls
            // with spinners. Builds up the answer in real-time.
            let mut dynamic_lines: Vec<Line<'static>> = Vec::new();
            let mut dynamic_texts: Vec<String> = Vec::new();
            let mut dynamic_raw_texts: Vec<String> = Vec::new();
            let wrap_width = content_width.saturating_sub(2);

            for block in &self.pending_blocks {
                match block {
                    ChatBlock::Text(text) => {
                        // Render text progressively with markdown — no header label
                        for wl in safe_markdown_lines(text, wrap_width) {
                            let plain: String =
                                wl.spans.iter().map(|s| s.content.as_ref()).collect();
                            dynamic_texts.push(format!("  {}", plain));
                            dynamic_raw_texts.push(plain.clone());
                            let mut indented = vec![Span::raw("  ".to_string())];
                            indented.extend(wl.spans);
                            dynamic_lines.push(Line::from(indented));
                        }
                    }
                    ChatBlock::ToolCall(tc) => {
                        let mut spans: Vec<Span<'static>> = Vec::new();
                        if tc.is_complete {
                            spans.push(Span::styled("  \u{2713} ", Style::default().fg(Color::Green)));
                        } else {
                            let spinners = ['\u{25D0}', '\u{25D1}', '\u{25D2}', '\u{25D3}'];
                            let icon = spinners[self.spinner_frame % spinners.len()];
                            spans.push(Span::styled(
                                format!("  {} ", icon),
                                Style::default().fg(Color::Magenta),
                            ));
                        }
                        spans.push(Span::styled(
                            tc.name.clone(),
                            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                        ));
                        if !tc.input_summary.is_empty() {
                            spans.push(Span::styled(
                                format!("({})", tc.input_summary),
                                Style::default().fg(Color::DarkGray),
                            ));
                        }
                        if tc.is_complete && !tc.output_summary.is_empty() {
                            spans.push(Span::styled(
                                format!(" \u{2192} {}", tc.output_summary),
                                Style::default().fg(Color::DarkGray),
                            ));
                        }
                        let text: String = spans.iter().map(|s| s.content.as_ref()).collect();
                        dynamic_raw_texts.push(text.trim_start().to_string());
                        dynamic_texts.push(text);
                        dynamic_lines.push(Line::from(spans));
                    }
                    ChatBlock::CompactionMarker => {
                        let dashes = "─".repeat((content_width.saturating_sub(22)) / 2);
                        let marker = format!("  {} context compacted {}", dashes, dashes);
                        dynamic_texts.push(marker.clone());
                        dynamic_raw_texts.push(String::new());
                        dynamic_lines.push(Line::from(Span::styled(
                            marker,
                            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                        )));
                    }
                }
            }

            if let Some(status) = &self.current_status {
                let spinners = ['◐', '◑', '◒', '◓'];
                let spinner = spinners[self.spinner_frame % spinners.len()];
                let text = format!(" {} {}", spinner, status);
                dynamic_raw_texts.push(String::new()); // don't copy status spinner lines
                dynamic_texts.push(text);
                dynamic_lines.push(Line::from(vec![
                    Span::styled(
                        format!(" {} ", spinner),
                        Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(status.clone(), Style::default().fg(Color::Yellow)),
                ]));
            }

            // ── Phase 3: Assemble content_texts, compute total ─────────────
            let cached_len = self.cached_msg_lines.len();
            let total_content = cached_len + dynamic_lines.len();
            self.total_content_lines = total_content;

            // Clamp stale scroll offset to valid range
            let max_offset = total_content.saturating_sub(chat_height);
            if self.scroll_offset > max_offset {
                self.scroll_offset = max_offset;
            }

            let mut content_texts = self.cached_msg_texts.clone();
            content_texts.extend(dynamic_texts);
            self.content_texts = content_texts;

            let mut content_raw_texts = self.cached_msg_raw_texts.clone();
            content_raw_texts.extend(dynamic_raw_texts);
            self.content_texts_raw = content_raw_texts;

            // ── Scroll calculation ─────────────────────────────────────────
            let scroll_y: u16 = if self.auto_scroll || total_content <= chat_height {
                total_content.saturating_sub(chat_height) as u16
            } else {
                let max_scroll = total_content.saturating_sub(chat_height);
                self.scroll_offset.min(max_scroll) as u16
            };
            let lines_to_show = total_content.saturating_sub(scroll_y as usize);
            let pad = chat_height.saturating_sub(lines_to_show.min(chat_height));
            for _ in 0..pad {
                final_lines.push(Line::from(""));
            }

            // ── Phase 4: Build visible window ──────────────────────────────
            let start = scroll_y as usize;
            let end = (start + chat_height).min(total_content);
            for i in start..end {
                let line = if i < cached_len {
                    self.cached_msg_lines[i].clone()
                } else {
                    let dyn_idx = i - cached_len;
                    if dyn_idx < dynamic_lines.len() {
                        dynamic_lines[dyn_idx].clone()
                    } else {
                        Line::from("")
                    }
                };
                let content_idx = i;
                if let Some(((sel_start_line, sel_start_col), (sel_end_line, sel_end_col))) = selection {
                    if content_idx >= sel_start_line && content_idx <= sel_end_line {
                        let line_start = if content_idx == sel_start_line { sel_start_col } else { 0 };
                        let line_text_len = self.content_texts.get(content_idx)
                            .map(|t| t.chars().count()).unwrap_or(0);
                        let line_end = if content_idx == sel_end_line { sel_end_col } else { line_text_len };
                        final_lines.push(apply_selection_highlight(line, line_start, line_end));
                        continue;
                    }
                }
                final_lines.push(line);
            }
        }

        // ── Separator + Input line (skip for home screen — it's rendered inline) ──
        if !show_home {
            if has_conversation {
                let sep_width = content_width.min(area.width.saturating_sub(4) as usize);
                let sep = "\u{2500}".repeat(sep_width);
                final_lines.push(Line::from(Span::styled(
                    format!("  {}", sep),
                    Style::default().fg(Color::DarkGray),
                )));
            }

            // Blinking block cursor: visible for 3 frames, hidden for 2 (~500ms cycle at 10fps)
            let cursor_visible = is_focused && (self.spinner_frame % 5 < 3);
            let cursor_char = if cursor_visible { "\u{2588}" } else { " " };

            let has_pending = self.pending_message.is_some();
            if has_pending && self.input_buffer.is_empty() {
                let queued_msg = self.pending_message.as_deref().unwrap_or("");
                final_lines.push(Line::from(vec![
                    Span::styled(" \u{2502} ", Style::default().fg(Color::Indexed(60))),
                    Span::styled(
                        format!("{} ", queued_msg),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::styled("(queued)", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
                ]));
            } else if !self.input_buffer.is_empty() {
                let display = format!("{}{}", self.input_buffer, cursor_char);
                let wrap_width = content_width.saturating_sub(5);
                let wrapped = textwrap::wrap(&display, wrap_width);
                for w in wrapped.iter() {
                    final_lines.push(Line::from(vec![
                        Span::styled(" \u{2502} ", Style::default().fg(Color::Indexed(60))),
                        Span::styled(w.to_string(), Style::default().fg(Color::White)),
                    ]));
                }
            } else {
                final_lines.push(Line::from(vec![
                    Span::styled(" \u{2502} ", Style::default().fg(Color::Indexed(60))),
                    Span::styled(cursor_char.to_string(), Style::default().fg(Color::White)),
                ]));
            };
        }

        let mut chat_block = if self.borderless {
            Block::default()
        } else {
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border_color))
                .title(title)
                .title_style(Style::default().fg(if is_focused { ACCENT } else { Color::DarkGray }))
        };

        if !self.borderless && self.context_input_tokens > 0 {
            let fraction = self.context_usage_fraction().min(1.0);
            let pct = (fraction * 100.0) as u32;
            let bar_len: usize = 10;
            let filled = ((bar_len as f64) * fraction) as usize;
            let empty = bar_len.saturating_sub(filled);

            let bar_color = if fraction <= 0.60 {
                Color::Green
            } else if fraction <= 0.80 {
                Color::Yellow
            } else {
                Color::Red
            };

            chat_block = chat_block.title_bottom(
                Line::from(vec![
                    Span::styled(" ctx [", Style::default().fg(Color::DarkGray)),
                    Span::styled("█".repeat(filled), Style::default().fg(bar_color)),
                    Span::styled("░".repeat(empty), Style::default().fg(Color::Indexed(237))),
                    Span::styled(format!("] {}% ", pct), Style::default().fg(Color::DarkGray)),
                ]).right_aligned()
            );
        }

        let para = Paragraph::new(final_lines).block(chat_block);
        f.render_widget(para, area);

        // ── Scroll position indicator ───────────────────────────────────────
        let total_content = self.total_content_lines;
        if total_content > chat_height && chat_height > 0 {
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .thumb_symbol("█")
                .track_symbol(Some("│"))
                .begin_symbol(None)
                .end_symbol(None)
                .thumb_style(Style::default().fg(Color::Indexed(245)))
                .track_style(Style::default().fg(Color::Indexed(237)));
            let scroll_y = if self.auto_scroll || total_content <= chat_height {
                total_content.saturating_sub(chat_height)
            } else {
                let max_scroll = total_content.saturating_sub(chat_height);
                self.scroll_offset.min(max_scroll)
            };
            let mut scrollbar_state = ScrollbarState::new(total_content)
                .position(scroll_y);
            let sb_inset = if self.borderless { 0 } else { 1 };
            let scrollbar_area = Rect {
                x: area.x,
                y: area.y + sb_inset,
                width: area.width,
                height: area.height.saturating_sub(sb_inset * 2),
            };
            f.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
        }

    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Static helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Render a completed tool call into the cached lines/texts.
fn render_tool_call_cached(
    tc: &ToolCallDisplay,
    cached_lines: &mut Vec<Line<'static>>,
    cached_texts: &mut Vec<String>,
) {
    let mut spans = vec![
        Span::styled("  \u{2713} ", Style::default().fg(Color::Green)),
        Span::styled(
            tc.name.clone(),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
    ];
    if !tc.input_summary.is_empty() {
        spans.push(Span::styled(
            format!("({})", tc.input_summary),
            Style::default().fg(Color::DarkGray),
        ));
    }
    if !tc.output_summary.is_empty() {
        spans.push(Span::styled(
            format!(" \u{2192} {}", tc.output_summary),
            Style::default().fg(Color::DarkGray),
        ));
    }
    let text: String = spans.iter().map(|s| s.content.as_ref()).collect();
    cached_texts.push(text);
    cached_lines.push(Line::from(spans));
}


/// Wrap a styled Line to fit within max_width, preserving span styles.
fn wrap_styled_line(line: Line<'static>, max_width: usize) -> Vec<Line<'static>> {
    let char_count: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
    if char_count <= max_width || max_width == 0 {
        return vec![line];
    }

    let styled_chars: Vec<(char, Style)> = line
        .spans
        .iter()
        .flat_map(|span| span.content.chars().map(move |c| (c, span.style)))
        .collect();

    let mut result = Vec::new();
    let mut pos = 0;

    while pos < styled_chars.len() {
        let end = (pos + max_width).min(styled_chars.len());
        let actual_end = if end >= styled_chars.len() {
            end
        } else {
            styled_chars[pos..end]
                .iter()
                .rposition(|(c, _)| *c == ' ')
                .map(|p| pos + p + 1)
                .unwrap_or(end)
        };

        let chunk = &styled_chars[pos..actual_end];
        let mut spans: Vec<Span<'static>> = Vec::new();
        if let Some(&(first_c, first_style)) = chunk.first() {
            let mut current_text = String::new();
            current_text.push(first_c);
            let mut current_style = first_style;
            for &(c, style) in &chunk[1..] {
                if style != current_style {
                    spans.push(Span::styled(current_text, current_style));
                    current_text = String::new();
                    current_style = style;
                }
                current_text.push(c);
            }
            spans.push(Span::styled(current_text, current_style));
        }
        result.push(Line::from(spans));

        pos = actual_end;
        while pos < styled_chars.len() && styled_chars[pos].0 == ' ' {
            pos += 1;
        }
    }

    result
}

/// Apply a highlight background to a portion of a Line (for text selection).
fn apply_selection_highlight(line: Line<'static>, sel_start: usize, sel_end: usize) -> Line<'static> {
    if sel_start >= sel_end {
        return line;
    }
    let highlight_bg = Color::Indexed(237);
    let mut result_spans: Vec<Span<'static>> = Vec::new();
    let mut col: usize = 0;

    for span in line.spans {
        let span_char_count = span.content.chars().count();
        let span_start = col;
        let span_end = col + span_char_count;

        if span_end <= sel_start || span_start >= sel_end {
            result_spans.push(span);
        } else {
            let chars: Vec<char> = span.content.chars().collect();

            let hl_start = sel_start.saturating_sub(span_start);
            if hl_start > 0 {
                let before: String = chars[..hl_start].iter().collect();
                result_spans.push(Span::styled(before, span.style));
            }

            let hl_end = (sel_end - span_start).min(chars.len());
            let selected: String = chars[hl_start..hl_end].iter().collect();
            result_spans.push(Span::styled(selected, span.style.bg(highlight_bg)));

            if hl_end < chars.len() {
                let after: String = chars[hl_end..].iter().collect();
                result_spans.push(Span::styled(after, span.style));
            }
        }

        col = span_end;
    }

    Line::from(result_spans)
}

/// Safe markdown renderer — handles headers, bold, italic, inline code, code blocks,
/// and list items with styled spans. Cannot panic (no third-party markdown crate).
fn safe_markdown_lines(text: &str, wrap_width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code_block = false;

    for raw_line in text.lines() {
        // Code fence toggle
        if raw_line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
            // Don't render the fence line itself
            continue;
        }

        if in_code_block {
            // Render code block content with a distinct style
            let styled = Line::from(Span::styled(
                raw_line.to_string(),
                Style::default().fg(Color::Green),
            ));
            lines.extend(wrap_styled_line(styled, wrap_width));
            continue;
        }

        // Headers: strip leading `#`s and style as bold
        if let Some(rest) = strip_heading_md(raw_line) {
            let styled = Line::from(Span::styled(
                rest,
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ));
            lines.extend(wrap_styled_line(styled, wrap_width));
            continue;
        }

        // List items: `- text` or `* text`
        let trimmed = raw_line.trim_start();
        let is_list = trimmed.starts_with("- ") || trimmed.starts_with("* ");
        let (prefix, body) = if is_list {
            let indent = raw_line.len() - trimmed.len();
            let bullet_prefix = format!("{}\u{2022} ", " ".repeat(indent));
            (bullet_prefix, &trimmed[2..])
        } else {
            (String::new(), raw_line)
        };

        // Parse inline styles: **bold**, *italic*, `code`
        let mut spans = Vec::new();
        if !prefix.is_empty() {
            spans.push(Span::styled(prefix, Style::default().fg(Color::Yellow)));
        }
        parse_inline_markdown(body, &mut spans);

        let styled_line = Line::from(spans);
        lines.extend(wrap_styled_line(styled_line, wrap_width));
    }

    if lines.is_empty() {
        lines.push(Line::from(""));
    }
    lines
}

/// Strip `# ` / `## ` / `### ` etc. from the start of a line, returning the heading text.
fn strip_heading_md(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    if hashes > 0 && hashes <= 6 {
        let rest = trimmed[hashes..].trim_start();
        if !rest.is_empty() {
            return Some(rest.to_string());
        }
    }
    None
}

/// Parse inline markdown (`**bold**`, `*italic*`, `` `code` ``) into styled spans.
fn parse_inline_markdown(text: &str, spans: &mut Vec<Span<'static>>) {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut plain = String::new();
    let base_style = Style::default();

    while i < len {
        // Inline code: `...`
        if chars[i] == '`' {
            if !plain.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut plain), base_style));
            }
            i += 1;
            let mut code = String::new();
            while i < len && chars[i] != '`' {
                code.push(chars[i]);
                i += 1;
            }
            if i < len { i += 1; } // skip closing `
            spans.push(Span::styled(code, Style::default().fg(Color::Green)));
            continue;
        }

        // Bold: **...**
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            if !plain.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut plain), base_style));
            }
            i += 2;
            let mut bold = String::new();
            while i + 1 < len && !(chars[i] == '*' && chars[i + 1] == '*') {
                bold.push(chars[i]);
                i += 1;
            }
            if i + 1 < len { i += 2; } // skip closing **
            spans.push(Span::styled(bold, Style::default().add_modifier(Modifier::BOLD)));
            continue;
        }

        // Italic: *...*
        if chars[i] == '*' {
            if !plain.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut plain), base_style));
            }
            i += 1;
            let mut italic = String::new();
            while i < len && chars[i] != '*' {
                italic.push(chars[i]);
                i += 1;
            }
            if i < len { i += 1; } // skip closing *
            spans.push(Span::styled(italic, Style::default().add_modifier(Modifier::ITALIC)));
            continue;
        }

        plain.push(chars[i]);
        i += 1;
    }

    if !plain.is_empty() {
        spans.push(Span::styled(plain, base_style));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Async pipeline functions
// ─────────────────────────────────────────────────────────────────────────────

pub async fn chat_agent_pipeline(
    config: crate::core::config::EnrichmentConfig,
    system_prompt: String,
    messages: Vec<(String, String)>,
    user_message: String,
    needs_compaction: bool,
    tx: mpsc::Sender<ChatStreamEvent>,
) {
    use crate::agent::loop_runner::{AgentEvent, run_agent_loop};
    use crate::agent::providers::create_agent_provider;

    let provider = create_agent_provider(&config);

    let mut effective_system_prompt = system_prompt;
    let mut effective_messages = messages;

    // Auto-compact if context is above 80% and there are enough messages
    if needs_compaction && effective_messages.len() >= 4 {
        let _ = tx.send(ChatStreamEvent::Status(
            format!("Compacting context ({} messages)...", effective_messages.len()),
        )).await;

        let keep_count = 4; // keep last 4 messages
        let compact_end = effective_messages.len() - keep_count;
        let to_compact: Vec<(String, String)> = effective_messages[..compact_end].to_vec();
        let remaining: Vec<(String, String)> = effective_messages[compact_end..].to_vec();

        let prompt = chat_prompts::build_compaction_prompt(&to_compact);
        match provider.compact_history(&prompt).await {
            Ok(summary) => {
                let removed_count = compact_end;
                let _ = tx.send(ChatStreamEvent::Compacted {
                    summary: summary.clone(),
                    removed_count,
                }).await;
                let _ = tx.send(ChatStreamEvent::Status(
                    format!("Compacted {} messages into summary, {} kept", removed_count, remaining.len()),
                )).await;

                // Augment system prompt with the summary
                effective_system_prompt = format!(
                    "{}\n\n## Previous Conversation Summary\n{}",
                    effective_system_prompt, summary
                );
                effective_messages = remaining;
            }
            Err(e) => {
                let _ = tx.send(ChatStreamEvent::Error(
                    format!("Compaction failed: {}", e),
                )).await;
                return;
            }
        }
    }

    let (agent_tx, mut agent_rx) = mpsc::channel::<AgentEvent>(100);

    // Spawn the agent loop
    let agent_handle = tokio::spawn(async move {
        run_agent_loop(provider, effective_system_prompt, effective_messages, user_message, agent_tx).await;
    });

    // Bridge AgentEvent -> ChatStreamEvent
    while let Some(event) = agent_rx.recv().await {
        let chat_event = match event {
            AgentEvent::Status(msg) => ChatStreamEvent::Status(msg),
            AgentEvent::Chunk(text) => ChatStreamEvent::Chunk(text),
            AgentEvent::ToolCall { name, input_summary } => {
                ChatStreamEvent::ToolCall { name, input_summary }
            }
            AgentEvent::ToolResult { name, output_summary } => {
                ChatStreamEvent::ToolResult { name, output_summary }
            }
            AgentEvent::Usage { input_tokens, output_tokens } => {
                ChatStreamEvent::Usage { input_tokens, output_tokens }
            }
            AgentEvent::Done => ChatStreamEvent::Done,
            AgentEvent::Error(msg) => ChatStreamEvent::Error(msg),
        };
        if tx.send(chat_event).await.is_err() {
            break;
        }
    }

    // If the agent loop panicked, report the error
    if let Err(e) = agent_handle.await {
        let _ = tx.send(ChatStreamEvent::Error(format!("Agent error: {}", e))).await;
    }
}
