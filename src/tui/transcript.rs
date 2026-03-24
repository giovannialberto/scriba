//! Transcript popup rendering and key handling for the Dashboard.
//!
//! This module contains all methods related to displaying a recording's transcript,
//! including enrichment metadata (summary, topics, entities, key points),
//! clipboard support, and the transcript popup layout.

use crate::database::Recording;
use anyhow::{Context, Result};
use crossterm::event::KeyCode;
use dirs::home_dir;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use super::app::{Dashboard, DashboardAction};
use super::chat::ACCENT;

impl Dashboard {
    pub(super) async fn handle_transcript_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Esc => {
                self.show_transcript = false;
                self.transcript_content.clear();
                self.clear_enrichment_data();
                self.restore_global_chat();
                Ok(DashboardAction::Continue)
            }
            _ => Ok(DashboardAction::Continue),
        }
    }

    pub(super) async fn show_selected_transcript(&mut self) -> Result<()> {
        if let Some(selected) = self.table_state.selected() {
            if let Some(recording) = self.recordings.get(selected).cloned() {
                if recording.has_transcript {
                    match self.load_transcript_content(&recording) {
                        Ok(content) => {
                            self.transcript_content = content;
                            self.show_transcript = true;
                            // Load enrichment data
                            self.load_enrichment_data(&recording);
                            // Initialize recording chat context
                            self.init_recording_chat(&recording);
                        }
                        Err(e) => {
                            self.message = format!("Failed to load transcript: {}", e);
                            self.show_message = true;
                        }
                    }
                } else {
                    self.message =
                        "No transcript available for this recording. Use P to play instead."
                            .to_string();
                    self.show_message = true;
                }
            }
        }
        Ok(())
    }

    pub(super) fn load_enrichment_data(&mut self, recording: &Recording) {
        // Load summary and key points from recording
        self.transcript_summary = recording.summary.clone();
        self.transcript_key_points = recording
            .key_points
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok());

        // Load topics and entities from transcript
        if let Some(id) = recording.id {
            if let Ok(Some(transcript)) = self.db.get_transcript_by_recording_id(id) {
                self.transcript_topics = transcript
                    .topics
                    .as_ref()
                    .and_then(|s| serde_json::from_str(s).ok());

                // Parse entities JSON: [{"name": "...", "type": "..."}]
                self.transcript_entities = transcript.entities.as_ref().and_then(|s| {
                    let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(s);
                    parsed.ok().map(|entities| {
                        entities
                            .iter()
                            .filter_map(|e| {
                                let name = e.get("name")?.as_str()?.to_string();
                                let entity_type = e.get("type")?.as_str()?.to_string();
                                Some((name, entity_type))
                            })
                            .collect()
                    })
                });
            }
        }
    }

    pub(super) fn clear_enrichment_data(&mut self) {
        self.transcript_summary = None;
        self.transcript_key_points = None;
        self.transcript_topics = None;
        self.transcript_entities = None;
    }

    pub(super) fn load_transcript_content(&self, recording: &Recording) -> Result<String> {
        // First try to load from database
        if let Some(id) = recording.id {
            if let Ok(Some(transcript)) = self.db.get_transcript_by_recording_id(id) {
                return Ok(transcript.content);
            }
        }

        // Fallback: try to load from file (standard transcript.txt)
        let base_path = home_dir()
            .context("Could not find home directory")?
            .join("scriba_recordings");
        let recording_dir = base_path.join(&recording.directory_name);

        // Try transcript.txt
        let transcript_path = recording_dir.join("transcript.txt");
        if transcript_path.exists() {
            return std::fs::read_to_string(&transcript_path)
                .context("Failed to read transcript.txt file");
        }

        Err(anyhow::anyhow!("No transcript file found (expected transcript.txt)"))
    }

    pub(super) fn copy_transcript_to_clipboard(&self) -> Result<()> {
        use arboard::Clipboard;

        let mut clipboard = Clipboard::new().context("Failed to access clipboard")?;

        clipboard
            .set_text(&self.transcript_content)
            .context("Failed to copy text to clipboard")?;

        Ok(())
    }

    pub(super) fn render_transcript_popup(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        // Dynamic chat panel height
        let has_chat_content = !self.chat.messages.is_empty()
            || !self.chat.pending_blocks.is_empty()
            || self.chat.is_generating;

        let chat_h: u16 = if has_chat_content {
            (area.height * 45 / 100).max(8)
        } else {
            (self.chat.suggestions.len() as u16 + 4).max(5).min(8)
        };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),      // header
                Constraint::Min(5),         // content (enrichment + transcript preview)
                Constraint::Length(2),      // blank + separator
                Constraint::Length(chat_h), // chat panel
                Constraint::Length(2),      // footer
            ])
            .split(area);

        // ── Header ──────────────────────────────────────────────────────
        self.render_transcript_header(f, chunks[0]);

        // ── Content (enrichment + transcript preview) ───────────────────
        self.render_transcript_body(f, chunks[1]);

        // ── Separator (blank line + ─) ───────────────────────────────────
        let sep = "\u{2500}".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: chunks[2].x, y: chunks[2].y + 1, width: chunks[2].width, height: 1 },
        );

        // ── Chat panel ──────────────────────────────────────────────────
        self.chat.render(f, chunks[3]);

        // ── Footer ──────────────────────────────────────────────────────
        self.render_transcript_footer(f, chunks[4]);

        // Notification overlay (e.g. "Copied to clipboard")
        if let Some((ref msg, _)) = self.notification_message {
            let is_error = msg.contains("failed") || msg.contains("Failed");
            let style = if is_error {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            };
            let notif_area = Rect {
                x: area.x,
                y: area.y + area.height.saturating_sub(1),
                width: area.width,
                height: 1,
            };
            let para = Paragraph::new(msg.as_str())
                .style(style)
                .alignment(Alignment::Center);
            f.render_widget(para, notif_area);
        }
    }

    pub(super) fn render_transcript_header(&self, f: &mut Frame, area: Rect) {
        // Row 0: "  Name · Xm · Mar 8, 12:30"
        let recording = self.current_transcript_recording.as_ref();
        let name = recording
            .and_then(|r| r.display_name.as_deref())
            .or(recording.map(|r| r.directory_name.as_str()))
            .unwrap_or("Recording");
        let dur_secs = recording.and_then(|r| r.duration_seconds).unwrap_or(0);
        let dur_str = if dur_secs >= 3600 {
            format!("{}h {}m", dur_secs / 3600, (dur_secs % 3600) / 60)
        } else {
            format!("{}m", dur_secs / 60)
        };
        let date_str = recording
            .map(|r| r.created_at.format("%b %-d, %H:%M").to_string())
            .unwrap_or_default();

        let header_line = Line::from(vec![
            Span::raw("  "),
            Span::styled(name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" \u{00B7} {} \u{00B7} {}", dur_str, date_str), Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(Paragraph::new(header_line), Rect { x: area.x, y: area.y, width: area.width, height: 1 });

        // Row 1: separator
        let sep = "\u{2500}".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: area.x, y: area.y + 1, width: area.width, height: 1 },
        );
    }

    /// Build combined enrichment + transcript lines and render as a single scrollable Paragraph.
    pub(super) fn render_transcript_body(&self, f: &mut Frame, area: Rect) {
        let content_width = area.width.saturating_sub(4) as usize; // 2 padding each side
        let mut lines: Vec<Line> = Vec::new();

        // ── Enrichment metadata (inline) ────────────────────────────────
        let has_enrichment = self.transcript_summary.is_some()
            || self.transcript_topics.is_some()
            || self.transcript_entities.is_some();

        if has_enrichment {
            lines.push(Line::from(""));

            // Summary
            if let Some(summary) = &self.transcript_summary {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled("Summary", Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
                ]));
                // Wrap summary text
                let wrapped = textwrap::wrap(summary, content_width);
                for wl in &wrapped {
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(wl.to_string(), Style::default().fg(Color::Indexed(249))),
                    ]));
                }
                lines.push(Line::from(""));
            }

            // Topics
            if let Some(topics) = &self.transcript_topics {
                if !topics.is_empty() {
                    let joined = topics.join(" \u{00B7} ");
                    Self::push_labeled_wrap(&mut lines, "Topics  ", &joined, content_width);
                }
            }

            // People / Entities
            if let Some(entities) = &self.transcript_entities {
                if !entities.is_empty() {
                    let people: Vec<&str> = entities.iter()
                        .filter(|(_, t)| t == "person")
                        .map(|(n, _)| n.as_str())
                        .collect();
                    let orgs: Vec<&str> = entities.iter()
                        .filter(|(_, t)| t == "organization")
                        .map(|(n, _)| n.as_str())
                        .collect();
                    let others: Vec<&str> = entities.iter()
                        .filter(|(_, t)| t != "person" && t != "organization")
                        .map(|(n, _)| n.as_str())
                        .collect();

                    if !people.is_empty() {
                        Self::push_labeled_wrap(&mut lines, "People  ", &people.join(" \u{00B7} "), content_width);
                    }
                    if !orgs.is_empty() {
                        Self::push_labeled_wrap(&mut lines, "Orgs    ", &orgs.join(" \u{00B7} "), content_width);
                    }
                    if !others.is_empty() {
                        Self::push_labeled_wrap(&mut lines, "Entities ", &others.join(" \u{00B7} "), content_width);
                    }
                }
            }

            // Key takeaway (first point only)
            if let Some(key_points) = &self.transcript_key_points {
                if !key_points.is_empty() {
                    lines.push(Line::from(""));
                    let wrapped = textwrap::wrap(&key_points[0], content_width);
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled("Key takeaway", Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
                    ]));
                    for wl in &wrapped {
                        lines.push(Line::from(vec![
                            Span::raw("  "),
                            Span::styled(wl.to_string(), Style::default().fg(Color::Indexed(249))),
                        ]));
                    }
                }
            }

            // Separator between enrichment and transcript
            lines.push(Line::from(""));
            let sep = "\u{2500}".repeat(area.width as usize);
            lines.push(Line::from(Span::styled(sep, Style::default().fg(Color::Indexed(237)))));
        }

        // ── Transcript preview (compact, max ~15 lines, faded tail) ────
        lines.push(Line::from(""));
        let max_preview = 15usize;
        let mut preview_lines: Vec<Line> = Vec::new();
        let transcript_lines_total = self.transcript_content.lines().count();
        let mut source_lines_used = 0usize;
        for text_line in self.transcript_content.lines() {
            if preview_lines.len() >= max_preview {
                break;
            }
            source_lines_used += 1;
            if text_line.is_empty() {
                preview_lines.push(Line::from(""));
            } else {
                let wrapped = textwrap::wrap(text_line, content_width);
                for wl in &wrapped {
                    if preview_lines.len() >= max_preview {
                        break;
                    }
                    preview_lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(wl.to_string(), Style::default().fg(Color::White)),
                    ]));
                }
            }
        }

        // Truncated if we didn't consume all source lines, OR if wrapping
        // filled the preview before we finished even the lines we did visit.
        let remaining_source = transcript_lines_total.saturating_sub(source_lines_used);
        let is_truncated = remaining_source > 0 || preview_lines.len() >= max_preview;

        // Fade out the last few lines + hint when transcript is truncated
        if is_truncated {
            let fade_count = 3usize;
            let fade_colors: [Color; 3] = [
                Color::Indexed(249),
                Color::Indexed(245),
                Color::Indexed(240),
            ];
            let total_preview = preview_lines.len();
            if total_preview > fade_count {
                for (i, color) in fade_colors.iter().enumerate() {
                    let idx = total_preview - fade_count + i;
                    if idx < total_preview {
                        let text: String = preview_lines[idx].spans.iter().map(|s| s.content.to_string()).collect();
                        preview_lines[idx] = Line::from(Span::styled(text, Style::default().fg(*color)));
                    }
                }
            }
        }

        let shown_count = preview_lines.len();
        lines.extend(preview_lines);

        // Hint line for truncated transcripts
        if is_truncated {
            lines.push(Line::from(""));
            // Count total wrapped display lines for accurate "X more lines"
            let total_display_lines: usize = self.transcript_content.lines().map(|l| {
                if l.is_empty() { 1 } else { textwrap::wrap(l, content_width).len() }
            }).sum();
            let more = total_display_lines.saturating_sub(shown_count);

            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{} more lines not shown \u{00B7} ", more),
                    Style::default().fg(Color::Indexed(240)),
                ),
                Span::styled("[", Style::default().fg(Color::Indexed(240))),
                Span::styled("Ctrl+Y", Style::default().fg(Color::Indexed(249))),
                Span::styled("] copy full transcript", Style::default().fg(Color::Indexed(240))),
            ]));
        }

        let para = Paragraph::new(lines);
        f.render_widget(para, area);
    }

    pub(super) fn render_transcript_footer(&self, f: &mut Frame, area: Rect) {
        // Separator
        let sep = "\u{2500}".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: area.x, y: area.y, width: area.width, height: 1 },
        );

        let footer_area = if area.height > 1 {
            Rect { x: area.x, y: area.y + 1, width: area.width, height: area.height - 1 }
        } else {
            area
        };

        // Left: ▸ scriba · vX.Y.Z
        let version = env!("CARGO_PKG_VERSION");
        let left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(format!("scriba \u{00B7} v{}", version), Style::default().fg(Color::DarkGray)),
        ];

        // Right: [Ctrl+Y] Copy  [Ctrl+T] Re-transcribe  [Esc] Back
        let right_spans = vec![
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Ctrl+Y", Style::default().fg(Color::White)),
            Span::styled("] Copy  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Ctrl+T", Style::default().fg(Color::White)),
            Span::styled("] Re-transcribe  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Esc", Style::default().fg(Color::White)),
            Span::styled("] Back ", Style::default().fg(Color::DarkGray)),
        ];

        let left_width: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_width: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_area.width as usize).saturating_sub(left_width + right_width);

        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);

        f.render_widget(Paragraph::new(Line::from(spans)), footer_area);
    }

    /// Push a "Label  value text..." line that wraps, with continuation lines indented to align.
    pub(super) fn push_labeled_wrap(lines: &mut Vec<Line<'_>>, label: &str, value: &str, max_width: usize) {
        let prefix_len = 2 + label.len(); // "  " + label
        let value_width = max_width.saturating_sub(prefix_len);
        if value_width == 0 {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(label.to_string(), Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
                Span::styled(value.to_string(), Style::default().fg(Color::Indexed(249))),
            ]));
            return;
        }
        let wrapped = textwrap::wrap(value, value_width);
        for (i, wl) in wrapped.iter().enumerate() {
            if i == 0 {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(label.to_string(), Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
                    Span::styled(wl.to_string(), Style::default().fg(Color::Indexed(249))),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::raw(" ".repeat(prefix_len)),
                    Span::styled(wl.to_string(), Style::default().fg(Color::Indexed(249))),
                ]));
            }
        }
    }
}
