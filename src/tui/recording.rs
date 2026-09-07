use crate::core::{
    CompressionSettings, RecordOptions, RecordingKind, RecordingPhase, TranscriptionMode,
    WorkflowManager, record_audio,
};
use crate::utils::generate_recording_name;
use anyhow::Result;
use ratatui::{
    layout::Rect,
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
            self.notification_message =
                Some(("Already recording \u{2014} Ctrl+R stops it.".to_string(), 30));
            return Ok(());
        }

        // Non-blocking: the recording strip above the current view shows
        // progress; the user keeps navigating.
        self.progress_frame = 0;
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
        // Never double-capture: the meeting autopilot may be auto-recording.
        if !self
            .recording_guard
            .try_begin(RecordingKind::Manual, None)
        {
            self.recording_mode = None;
            self.notification_message = Some((
                "A meeting is being auto-recorded \u{2014} Ctrl+R stops it.".to_string(),
                40,
            ));
            return Ok(());
        }

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

    /// Whether audio is being captured right now (manual or meeting). Only
    /// capture gets the strip; post-capture work is indicated on the
    /// recording's own row instead.
    pub(super) fn recording_indicator_visible(&self) -> bool {
        self.recording_task.is_some()
            || self
                .recording_guard
                .snapshot()
                .is_some_and(|i| i.phase == RecordingPhase::Recording)
    }

    /// Ctrl+R: start a manual recording, or stop whichever recording (manual
    /// or meeting) is in progress.
    pub(super) async fn toggle_recording(&mut self) -> Result<()> {
        if self.recording_task.is_some() {
            if let Some(stop_tx) = self.recording_stop_tx.take() {
                let _ = stop_tx.send(()).await;
            }
            return Ok(());
        }
        if let Some(info) = self.recording_guard.snapshot()
            && info.kind == RecordingKind::Meeting
        {
            if info.phase == RecordingPhase::Recording {
                self.recording_guard.request_stop();
                self.notification_message =
                    Some(("Stopping meeting recording\u{2026}".to_string(), 30));
            } else {
                self.notification_message =
                    Some(("Meeting recording is being processed.".to_string(), 30));
            }
            return Ok(());
        }
        self.execute_record_and_transcribe().await
    }

    /// Directory of a meeting recording whose capture is done but whose
    /// transcription/enrichment is still running in the autopilot.
    pub(super) fn processing_directory(&self) -> Option<String> {
        self.recording_guard
            .snapshot()
            .filter(|i| i.phase == RecordingPhase::Processing)
            .and_then(|i| i.directory)
    }

    /// Whether background work (TUI transcription queue or autopilot
    /// finalization) is running for a recording.
    pub(super) fn is_recording_busy(&self, directory_name: &str) -> bool {
        self.is_transcription_pending_or_active(directory_name)
            || self.processing_directory().as_deref() == Some(directory_name)
    }

    /// Keep the home list's per-row busy flags in sync with current state.
    pub(super) fn refresh_home_busy(&mut self) {
        let busy: Vec<bool> = self
            .chat
            .home_recordings
            .iter()
            .map(|r| self.is_recording_busy(&r.directory_name))
            .collect();
        for (rec, b) in self.chat.home_recordings.iter_mut().zip(busy) {
            rec.busy = b;
        }
    }

    /// Non-blocking recording indicator rendered at the top of non-home views:
    /// one content line plus a separator.
    pub(super) fn render_recording_strip(&self, f: &mut Frame, area: Rect) {
        let aligned = Rect {
            x: area.x + 2,
            width: area.width.saturating_sub(4),
            ..area
        };
        let line = self.recording_strip_line(aligned.width as usize, "");
        f.render_widget(
            Paragraph::new(line),
            Rect { x: aligned.x, y: aligned.y, width: aligned.width, height: 1 },
        );
        if area.height > 1 {
            let sep = "\u{2500}".repeat(aligned.width as usize);
            f.render_widget(
                Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
                Rect { x: aligned.x, y: aligned.y + 1, width: aligned.width, height: 1 },
            );
        }
    }

    /// The recording indicator as a single line: pulsing dot, kind (with the
    /// source app for meetings), elapsed time, live waveform, and a right-
    /// aligned Ctrl+R hint.
    pub(super) fn recording_strip_line(&self, width: usize, margin: &str) -> Line<'static> {
        let info = self.recording_guard.snapshot();
        let (kind_label, source, elapsed) = match &info {
            Some(i) => (
                match i.kind {
                    RecordingKind::Manual => "Recording",
                    RecordingKind::Meeting => "Recording meeting",
                },
                i.source.clone(),
                i.elapsed(),
            ),
            None => (
                "Recording",
                None,
                self.recording_start_instant.map(|t| t.elapsed()).unwrap_or_default(),
            ),
        };
        let elapsed_str = format!("{:02}:{:02}", elapsed.as_secs() / 60, elapsed.as_secs() % 60);
        let dim = Style::default().fg(Color::DarkGray);
        let bold = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);

        // Pulsing red dot (~1s period at the 100ms animation tick).
        let dot_on = (self.progress_frame / 5) % 2 == 0;
        let dot_color = if dot_on { Color::Red } else { Color::Indexed(88) };
        let mut spans: Vec<Span<'static>> = vec![
            Span::raw(margin.to_string()),
            Span::styled("\u{25CF} ", Style::default().fg(dot_color)),
            Span::styled(kind_label, bold),
        ];
        if let Some(src) = &source {
            spans.push(Span::styled(format!(" \u{00B7} {src}"), Style::default().fg(ACCENT)));
        }
        spans.push(Span::styled(format!("  {elapsed_str}  "), dim));
        spans.extend(self.waveform_spans(24));

        let hint: Vec<Span<'static>> = vec![
            Span::styled("[", dim),
            Span::styled("Ctrl+R", Style::default().fg(Color::White)),
            Span::styled("] Stop", dim),
        ];
        let left_width: usize = spans.iter().map(|s| s.content.chars().count()).sum();
        let right_width: usize = hint.iter().map(|s| s.content.chars().count()).sum();
        let gap = width.saturating_sub(left_width + right_width);
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(hint);
        Line::from(spans)
    }

    /// Recent mic levels as block-height characters, newest on the right.
    fn waveform_spans(&self, width: usize) -> Vec<Span<'static>> {
        let wave_chars = [' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}'];
        let hist_len = self.volume_history.len();
        let mut spans = Vec::with_capacity(width);
        for i in 0..width {
            let level = if i < width.saturating_sub(hist_len) {
                0.0_f32
            } else {
                let idx = hist_len.saturating_sub(width.saturating_sub(i));
                let raw = self.volume_history.get(idx).copied().unwrap_or(0.0);
                let prev = if idx > 0 { self.volume_history.get(idx - 1).copied().unwrap_or(raw) } else { raw };
                (raw * 0.7 + prev * 0.3).min(1.0)
            };
            let scaled = (level * 50.0).min(1.0);
            let char_idx = ((scaled * (wave_chars.len() - 1) as f32).round() as usize).min(wave_chars.len() - 1);
            let color = if char_idx <= 1 { Color::Indexed(237) } else { ACCENT };
            spans.push(Span::styled(String::from(wave_chars[char_idx]), Style::default().fg(color)));
        }
        spans
    }
}
