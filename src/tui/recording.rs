use crate::core::{
    CompressionSettings, RecordOptions,
    TranscriptionMode, WorkflowManager, record_audio,
};
use crate::utils::generate_recording_name;
use anyhow::Result;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;

use super::chat::ACCENT;
use super::app::Dashboard;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(super) enum RecordingMode {
    RecordAndTranscribe,
}

pub(super) struct ActiveTranscription {
    pub(super) task: tokio::task::JoinHandle<Result<(), anyhow::Error>>,
    pub(super) recording_name: String,
}

pub(super) enum PendingTranscription {
    Retranscribe {
        recording_name: String,
        transcription_mode: TranscriptionMode,
    },
    Import {
        source_path: PathBuf,
        display_name: String,
        transcription_mode: TranscriptionMode,
    },
}

impl PendingTranscription {
    pub(super) fn recording_name(&self) -> &str {
        match self {
            PendingTranscription::Retranscribe { recording_name, .. } => recording_name,
            PendingTranscription::Import { display_name, .. } => display_name,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recording / transcription methods
// ─────────────────────────────────────────────────────────────────────────────

impl Dashboard {
    pub(super) fn has_active_transcription(&self) -> bool {
        self.active_transcription.is_some()
    }

    pub(super) fn is_transcription_pending_or_active(&self, dir_name: &str) -> bool {
        if let Some(ref active) = self.active_transcription {
            if active.recording_name == dir_name {
                return true;
            }
        }
        self.transcription_queue
            .iter()
            .any(|p| p.recording_name() == dir_name)
    }

    pub(super) fn drain_transcription_queue(&mut self) {
        if self.active_transcription.is_some() {
            return;
        }
        if let Some(pending) = self.transcription_queue.pop_front() {
            let name = pending.recording_name().to_string();
            let task = match pending {
                PendingTranscription::Retranscribe {
                    recording_name,
                    transcription_mode,
                } => tokio::spawn(async move {
                    let mut workflow = WorkflowManager::new()?;
                    workflow
                        .retranscribe_recording_silent(&recording_name, transcription_mode)
                        .await
                }),
                PendingTranscription::Import {
                    source_path,
                    display_name,
                    transcription_mode,
                } => tokio::spawn(async move {
                    let mut workflow = WorkflowManager::new()?;
                    workflow
                        .complete_import_workflow_silent(
                            &source_path,
                            Some(display_name),
                            Some(transcription_mode),
                        )
                        .await
                        .map(|_| ())
                }),
            };
            self.active_transcription = Some(ActiveTranscription {
                task,
                recording_name: name,
            });
            self.progress_frame = 0;
        }
    }

    pub(super) fn enqueue_transcription(&mut self, pending: PendingTranscription) {
        let name = pending.recording_name().to_string();
        if self.is_transcription_pending_or_active(&name) {
            return;
        }
        self.transcription_queue.push_back(pending);
        self.drain_transcription_queue();
    }

    pub(super) async fn execute_record_and_transcribe(&mut self) -> Result<()> {
        // Check if already recording (transcription can run concurrently)
        if self.recording_task.is_some() {
            self.message = "Recording already in progress".to_string();
            self.show_message = true;
            return Ok(());
        }

        // Show immediate progress animation
        self.progress_animation = Some("Recording... (Press Esc to stop)".to_string());
        self.progress_frame = 0;
        self.show_message = true;
        self.recording_mode = Some(RecordingMode::RecordAndTranscribe);
        self.recording_start_instant = Some(std::time::Instant::now());

        // Generate filename and start recording task (no name prompt, consistent with A command)
        let recording_name = generate_recording_name(None);
        self.start_recording_task(recording_name).await?;

        Ok(())
    }

    pub(super) async fn execute_transcribe_selected(&mut self) -> Result<()> {
        // Check if transcription is already running
        if self.has_active_transcription() {
            self.message = "Transcription already in progress. Please wait...".to_string();
            self.show_message = true;
            return Ok(());
        }

        // Get the selected recording
        let selected_index = match self.table_state.selected() {
            Some(i) => i,
            None => {
                self.message = "No recording selected".to_string();
                self.show_message = true;
                return Ok(());
            }
        };

        let selected_recording = match self.recordings.get(selected_index) {
            Some(recording) => recording.clone(),
            None => {
                self.message = "Invalid recording selection".to_string();
                self.show_message = true;
                return Ok(());
            }
        };

        // Check if transcript already exists
        let has_transcript = if let Some(id) = selected_recording.id {
            self.db
                .get_transcript_by_recording_id(id)
                .is_ok_and(|t| t.is_some())
        } else {
            false
        };

        if has_transcript {
            // Check if this is the second press on the same recording
            if self.last_transcribe_warning == Some(selected_index) {
                // User confirmed overwrite - proceed with transcription
                self.last_transcribe_warning = None;
            } else {
                // First press - show warning and remember this recording
                self.last_transcribe_warning = Some(selected_index);
                self.message =
                    "Recording already has transcript. Press T again to overwrite.".to_string();
                self.show_message = true;
                return Ok(());
            }
        } else {
            // Clear any previous warning state
            self.last_transcribe_warning = None;
        }

        // Enqueue transcription
        let directory_name = selected_recording.directory_name.clone();
        let transcription_mode = self.config.transcription.clone();

        self.enqueue_transcription(PendingTranscription::Retranscribe {
            recording_name: directory_name,
            transcription_mode,
        });

        Ok(())
    }

    pub(super) async fn start_recording_task(&mut self, recording_name: String) -> Result<()> {
        // Create channels for recording control
        let (stop_tx, stop_rx) = mpsc::channel(1);
        let (level_tx, level_rx) = mpsc::channel(100);

        // Store the channels for control and feedback
        self.recording_stop_tx = Some(stop_tx);
        self.recording_level_rx = Some(level_rx);

        // Use speech-optimized compression settings
        let compression_settings = CompressionSettings::speech_optimized();

        // Determine if auto-transcription is enabled based on recording mode
        let _auto_transcribe = matches!(
            self.recording_mode,
            Some(RecordingMode::RecordAndTranscribe)
        );

        // Silence auto-stop timeout from config
        let silence_timeout = if self.config.silence_auto_stop.enabled {
            Some(Duration::from_secs(self.config.silence_auto_stop.timeout_seconds as u64))
        } else {
            None
        };

        // Use unified recording function with TUI control channels
        let output_path = PathBuf::from(&recording_name);
        let input_device = self.config.audio_settings.input_device.clone();
        let loopback_device = self.config.audio_settings.loopback_device.clone();

        self.recording_task = Some(tokio::spawn(async move {
            record_audio(
                output_path,
                RecordOptions {
                    compression_settings: Some(compression_settings),
                    stop_rx: Some(stop_rx),
                    level_tx: Some(level_tx),
                    verbose: false,
                    silence_timeout,
                    input_device,
                    loopback_device,
                },
            )
            .await
        }));

        Ok(())
    }

    pub(super) async fn start_file_import(&mut self, file_path: String, display_name: String) -> Result<()> {
        let source_path = PathBuf::from(file_path.trim());
        let transcription_mode = self.config.transcription.clone();

        self.enqueue_transcription(PendingTranscription::Import {
            source_path,
            display_name,
            transcription_mode,
        });

        Ok(())
    }

    pub(super) fn stop_progress_animation(&mut self) {
        self.progress_animation = None;
    }

    pub(super) fn update_progress_message(&mut self) {
        if let Some(base_msg) = &self.progress_animation {
            let spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let spinner = spinners[self.progress_frame % spinners.len()];

            // If recording is active, show volume level instead of progress bar
            if self.recording_task.is_some() {
                let volume_bar = self.create_volume_bar(self.current_volume_level);
                self.message = format!("{} {} [{}]", spinner, base_msg, volume_bar);
            } else {
                // Regular progress bar for transcription
                let bar_width = 20;
                let progress_pos = (self.progress_frame / 2) % (bar_width * 2);
                let mut bar = vec!["▱"; bar_width];

                if progress_pos < bar_width {
                    for i in 0..=progress_pos.min(bar_width - 1) {
                        bar[i] = "▰";
                    }
                } else {
                    let reverse_pos = (bar_width * 2 - 1) - progress_pos;
                    for i in reverse_pos..bar_width {
                        bar[i] = "▰";
                    }
                }

                let bar_str = bar.join("");
                self.message = format!("{} {} [{}]", spinner, base_msg, bar_str);
            }

            self.progress_frame += 1;
        }
    }

    pub(super) fn create_volume_bar(&self, level: f32) -> String {
        let bar_width = 20;
        // Scale the level (0.0 to 1.0) to bar width and apply some amplification for visibility
        let scaled_level = (level * 50.0).min(1.0); // Amplify for visibility
        let filled_chars = (scaled_level * bar_width as f32) as usize;

        let mut bar = vec!["▱"; bar_width];
        for i in 0..filled_chars.min(bar_width) {
            bar[i] = "▰";
        }

        format!("{}|{}%", bar.join(""), (scaled_level * 100.0) as u8)
    }

    pub(super) fn render_recording_view(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // Header
                Constraint::Min(6),    // Body
                Constraint::Length(2),  // Footer
            ])
            .split(area);

        // ── Header ──────────────────────────────────────────────────
        let header_line = Line::from(vec![
            Span::raw("  "),
            Span::styled("Recording", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]);
        f.render_widget(Paragraph::new(header_line), Rect { x: chunks[0].x, y: chunks[0].y, width: chunks[0].width, height: 1 });
        let sep = "\u{2500}".repeat(chunks[0].width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: chunks[0].x, y: chunks[0].y + 1, width: chunks[0].width, height: 1 },
        );

        // ── Body (vertically centered) ─────────────────────────────
        let body = chunks[1];
        // 3 lines: spinner, blank, waveform
        let content_height: u16 = 3;
        let v_offset = body.height.saturating_sub(content_height) / 2;

        // Spinner + elapsed time
        let spinners = ["\u{280B}", "\u{2819}", "\u{2839}", "\u{2838}", "\u{283C}", "\u{2834}", "\u{2826}", "\u{2827}", "\u{2807}", "\u{280F}"];
        let spinner = spinners[self.progress_frame % spinners.len()];
        let elapsed = self.recording_start_instant
            .map(|t| t.elapsed())
            .unwrap_or_default();
        let mins = elapsed.as_secs() / 60;
        let secs = elapsed.as_secs() % 60;
        let elapsed_str = format!("{}m {:02}s", mins, secs);

        let spinner_line = Line::from(vec![
            Span::styled(format!("{}  Recording \u{00B7} {}", spinner, elapsed_str), Style::default().fg(Color::White)),
        ]);
        let spinner_y = body.y + v_offset;
        f.render_widget(
            Paragraph::new(spinner_line).alignment(Alignment::Center),
            Rect { x: body.x, y: spinner_y, width: body.width, height: 1 },
        );

        // Scrolling waveform using block-height characters
        let wave_chars = [' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}'];
        let wave_width: usize = 40;

        let mut spans: Vec<Span> = Vec::with_capacity(wave_width);
        let hist_len = self.volume_history.len();
        for i in 0..wave_width {
            let level = if i < wave_width.saturating_sub(hist_len) {
                0.0_f32
            } else {
                let idx = hist_len.saturating_sub(wave_width.saturating_sub(i));
                // Average with neighbors for a smoother look
                let raw = self.volume_history.get(idx).copied().unwrap_or(0.0);
                let prev = if idx > 0 { self.volume_history.get(idx - 1).copied().unwrap_or(raw) } else { raw };
                (raw * 0.7 + prev * 0.3).min(1.0)
            };
            let scaled = (level * 50.0).min(1.0);
            let char_idx = (scaled * (wave_chars.len() - 1) as f32).round() as usize;
            let ch = wave_chars[char_idx.min(wave_chars.len() - 1)];
            // Dim bars vs bright bars based on intensity
            let color = if char_idx <= 1 {
                Color::Indexed(237) // dark gray for silence
            } else {
                ACCENT
            };
            spans.push(Span::styled(String::from(ch), Style::default().fg(color)));
        }

        let wave_line = Line::from(spans);
        let wave_y = spinner_y + 2;
        if wave_y < body.y + body.height {
            f.render_widget(
                Paragraph::new(wave_line).alignment(Alignment::Center),
                Rect { x: body.x, y: wave_y, width: body.width, height: 1 },
            );
        }

        // ── Footer ─────────────────────────────────────────────────
        let sep2 = "\u{2500}".repeat(chunks[2].width as usize);
        f.render_widget(
            Paragraph::new(sep2).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: chunks[2].x, y: chunks[2].y, width: chunks[2].width, height: 1 },
        );
        let footer_area = Rect { x: chunks[2].x, y: chunks[2].y + 1, width: chunks[2].width, height: 1 };
        let version = env!("CARGO_PKG_VERSION");
        let left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(format!("scriba \u{00B7} v{}", version), Style::default().fg(Color::DarkGray)),
        ];
        let right_spans = vec![
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Esc", Style::default().fg(Color::White)),
            Span::styled("] Stop ", Style::default().fg(Color::DarkGray)),
        ];
        let left_w: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_w: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_area.width as usize).saturating_sub(left_w + right_w);
        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);
        f.render_widget(Paragraph::new(Line::from(spans)), footer_area);
    }
}
