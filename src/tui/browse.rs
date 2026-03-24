use crate::database::Recording;
use anyhow::{Context, Result};
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};
use std::path::PathBuf;
use std::process::Stdio;
use tokio::sync::mpsc;
use tokio::process::Command as TokioCommand;
use dirs::home_dir;

use super::chat::ACCENT;
use super::app::{Dashboard, DashboardAction, DashboardView};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
pub(super) enum FileDialogStage {
    FilePath, // Asking for file path
    FileName, // Asking for display name (optional)
}

// ─────────────────────────────────────────────────────────────────────────────
// Browse-related Dashboard methods
// ─────────────────────────────────────────────────────────────────────────────

impl Dashboard {
    pub(super) async fn handle_browse_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Tab | KeyCode::Esc => {
                self.current_view = DashboardView::Main;
                self.chat.focus = super::chat::ChatFocus::ChatInput;
                self.chat.borderless = true;
                if self.chat.messages.is_empty() {
                    self.chat.show_home_screen = true;
                    self.chat.show_suggestions = true;
                }
            }
            KeyCode::Up => {
                self.previous_recording();
            }
            KeyCode::Down => {
                self.next_recording();
            }
            KeyCode::PageUp | KeyCode::Char('[') => {
                self.previous_page().await?;
            }
            KeyCode::PageDown | KeyCode::Char(']') => {
                self.next_page().await?;
            }
            KeyCode::Enter => match self.show_selected_transcript().await {
                Ok(()) => {}
                Err(e) => {
                    self.message = format!("Failed to load transcript: {}", e);
                    self.show_message = true;
                }
            },
            KeyCode::Char('r') | KeyCode::Char('R') => {
                return Ok(DashboardAction::RecordAndTranscribe);
            }
            KeyCode::Char('t') | KeyCode::Char('T') => {
                return Ok(DashboardAction::TranscribeSelected);
            }
            KeyCode::Char('p') | KeyCode::Char('P') => match self.play_selected_recording().await {
                Ok(()) => {}
                Err(e) => {
                    self.message = format!("Failed to play recording: {}", e);
                    self.show_message = true;
                }
            },
            KeyCode::Char('d') | KeyCode::Delete => {
                self.show_delete_confirmation();
            }
            KeyCode::Char('/') => {
                self.search_mode = true;
                self.search_query.clear();
            }
            _ => {}
        }
        Ok(DashboardAction::Continue)
    }

    pub(super) async fn handle_search_input(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Esc => {
                self.search_mode = false;
                self.search_query.clear();
                self.load_recordings()?;
            }
            KeyCode::Enter => {
                self.search_mode = false;
                self.load_recordings()?;
            }
            KeyCode::Backspace => {
                self.search_query.pop();
            }
            KeyCode::Char(c) => {
                self.search_query.push(c);
            }
            _ => {}
        }
        Ok(DashboardAction::Continue)
    }

    pub(super) async fn handle_delete_confirmation(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.delete_confirm_selection = 0;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.delete_confirm_selection = 1;
            }
            KeyCode::Enter => {
                if self.delete_confirm_selection == 0 {
                    if let Some(recording) = self.delete_candidate.take() {
                        match self.perform_delete_recording(recording).await {
                            Ok(()) => {}
                            Err(e) => {
                                self.message = format!("Failed to delete recording: {}", e);
                                self.show_message = true;
                            }
                        }
                    }
                }
                self.show_delete_confirm = false;
                self.delete_candidate = None;
            }
            KeyCode::Esc => {
                self.show_delete_confirm = false;
                self.delete_candidate = None;
            }
            _ => {}
        }
        Ok(DashboardAction::Continue)
    }

    pub(super) async fn handle_file_dialog_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Esc => {
                // Cancel file import
                self.show_file_dialog = false;
                self.file_path_input.clear();
                self.file_name_input.clear();
                self.file_dialog_stage = FileDialogStage::FilePath;
                Ok(DashboardAction::Continue)
            }
            KeyCode::Enter => {
                match self.file_dialog_stage {
                    FileDialogStage::FilePath => {
                        // Validate file path
                        if self.file_path_input.trim().is_empty() {
                            self.notification_message = Some(("Please enter a file path".to_string(), 40));
                            return Ok(DashboardAction::Continue);
                        }

                        // Check if file exists
                        let file_path = PathBuf::from(self.file_path_input.trim());
                        if !file_path.exists() {
                            self.notification_message = Some(("File not found. Please check the path.".to_string(), 40));
                            return Ok(DashboardAction::Continue);
                        }

                        // Move to name input stage
                        self.file_dialog_stage = FileDialogStage::FileName;
                        // Pre-fill with file stem as default name
                        if let Some(stem) = file_path.file_stem().and_then(|s| s.to_str()) {
                            self.file_name_input = stem.to_string();
                        }
                    }
                    FileDialogStage::FileName => {
                        // Use file name or default to file stem
                        let display_name = if self.file_name_input.trim().is_empty() {
                            let file_path = PathBuf::from(self.file_path_input.trim());
                            file_path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("imported_audio")
                                .to_string()
                        } else {
                            self.file_name_input.trim().to_string()
                        };

                        // Start import process
                        self.show_file_dialog = false;
                        self.start_file_import(self.file_path_input.clone(), display_name)
                            .await?;
                    }
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Char(c) => {
                // Add character to current input
                match self.file_dialog_stage {
                    FileDialogStage::FilePath => {
                        self.file_path_input.push(c);
                    }
                    FileDialogStage::FileName => {
                        self.file_name_input.push(c);
                    }
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Backspace => {
                // Remove character from current input
                match self.file_dialog_stage {
                    FileDialogStage::FilePath => {
                        self.file_path_input.pop();
                    }
                    FileDialogStage::FileName => {
                        self.file_name_input.pop();
                    }
                }
                Ok(DashboardAction::Continue)
            }
            _ => Ok(DashboardAction::Continue),
        }
    }

    pub(super) fn next_recording(&mut self) {
        if self.recordings.is_empty() {
            return;
        }
        let i = match self.table_state.selected() {
            Some(i) => {
                if i >= self.recordings.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    pub(super) fn previous_recording(&mut self) {
        if self.recordings.is_empty() {
            return;
        }
        let i = match self.table_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.recordings.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    pub(super) async fn next_page(&mut self) -> Result<()> {
        // Try to load next page - if it has recordings, advance
        let old_page = self.current_page;
        self.current_page += 1;
        self.load_recordings()?;

        // If no recordings found on next page, go back to previous page
        if self.recordings.is_empty() {
            self.current_page = old_page;
            self.load_recordings()?;
        }

        Ok(())
    }

    pub(super) async fn previous_page(&mut self) -> Result<()> {
        if self.current_page > 0 {
            self.current_page -= 1;
            self.load_recordings()?;
        }
        Ok(())
    }

    pub(super) async fn play_selected_recording(&mut self) -> Result<()> {
        use anyhow::anyhow;
        if let Some(selected) = self.table_state.selected() {
            if let Some(recording) = self.recordings.get(selected) {
                // Locate the audio file in ~/scriba_recordings/<directory_name>/
                let audio_path = self
                    .find_audio_file(recording)
                    .ok_or_else(|| anyhow!("Could not find an audio file for this recording"))?;

                // Determine file extension to choose optimal players
                let is_mp3 = audio_path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.to_lowercase() == "mp3")
                    .unwrap_or(false);

                // Candidate players differ by platform. For MP3 files, prioritize mpv/ffplay over afplay
                #[cfg(target_os = "macos")]
                let candidates: Vec<(&str, &[&str])> = if is_mp3 {
                    vec![
                        ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                        (
                            "ffplay",
                            &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"],
                        ),
                        ("afplay", &[]), // Last resort for MP3
                    ]
                } else {
                    vec![
                        ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                        (
                            "ffplay",
                            &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"],
                        ),
                        ("afplay", &[]), // Works well with WAV
                    ]
                };

                #[cfg(all(unix, not(target_os = "macos")))]
                let candidates: Vec<(&str, &[&str])> = vec![
                    ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                    (
                        "ffplay",
                        &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"],
                    ),
                    ("aplay", &["-c", "2"]), // Force stereo output
                ];

                #[cfg(target_os = "windows")]
                let candidates: Vec<(&str, &[&str])> = vec![
                    ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                    (
                        "ffplay",
                        &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"],
                    ),
                    (
                        "powershell",
                        &["-NoProfile", "-Command", "(New-Object Media.SoundPlayer '"],
                    ), // will be handled specially
                ];

                // Try each candidate until one spawns successfully
                let mut launched_with: Option<String> = None;

                #[cfg(not(target_os = "windows"))]
                for (prog, base_args) in candidates {
                    let mut cmd = TokioCommand::new(prog);
                    // Detach from TTY so player doesn't consume keyboard (Esc) input
                    cmd.stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null());

                    // For afplay on macOS, check if this is a mono WAV file and needs special handling
                    if prog == "afplay" && recording.channels == 1 && !is_mp3 {
                        // Create a temporary stereo version of the mono WAV file
                        if let Ok(stereo_path) = self.create_stereo_temp_file(&audio_path).await {
                            cmd.arg(stereo_path);
                        } else {
                            // Fallback to original mono file
                            cmd.arg(&audio_path);
                        }
                    } else {
                        for a in base_args {
                            cmd.arg(a);
                        }
                        cmd.arg(&audio_path);
                    }

                    match cmd.spawn() {
                        Ok(mut child) => {
                            launched_with = Some(prog.to_string());

                            // Store child process for potential termination - ensure we have a valid PID
                            if let Some(child_id) = child.id() {
                                // Store the process ID for killing on key press immediately
                                self.current_playback_pid = Some(child_id);

                                // Create channel for playback completion notification
                                let (finished_tx, finished_rx) = mpsc::channel(1);
                                self.playback_finished_rx = Some(finished_rx);

                                tokio::spawn(async move {
                                    let _ = child.wait().await;
                                    let _ = finished_tx.send(()).await;
                                });
                                break;
                            } else {
                                // If we can't get PID, we can't control the process
                                launched_with = None;
                                continue;
                            }
                        }
                        Err(e) => {
                            // Store error for debugging if no player works
                            if prog == "mpv" && is_mp3 {
                                self.message = format!("mpv failed to play MP3: {}", e);
                            }
                        }
                    }
                }

                #[cfg(target_os = "windows")]
                {
                    // Try standard players first (mpv, ffplay), then fallback to PowerShell
                    for (prog, base_args) in &candidates[..candidates.len() - 1] {
                        // All except powershell
                        let mut cmd = TokioCommand::new(prog);
                        // Detach from TTY so player doesn't consume keyboard (Esc) input
                        cmd.stdin(Stdio::null())
                            .stdout(Stdio::null())
                            .stderr(Stdio::null());
                        for a in base_args {
                            cmd.arg(a);
                        }
                        cmd.arg(&audio_path);
                        match cmd.spawn() {
                            Ok(mut child) => {
                                if let Some(child_id) = child.id() {
                                    launched_with = Some(prog.to_string());
                                    self.current_playback_pid = Some(child_id);

                                    // Create channel for playback completion notification
                                    let (finished_tx, finished_rx) = mpsc::channel(1);
                                    self.playback_finished_rx = Some(finished_rx);

                                    tokio::spawn(async move {
                                        let _ = child.wait().await;
                                        let _ = finished_tx.send(()).await;
                                    });
                                    break;
                                }
                            }
                            Err(_e) => continue,
                        }
                    }

                    // PowerShell SoundPlayer fallback if no other player worked
                    if launched_with.is_none() {
                        let escaped = audio_path.to_string_lossy().replace("'", "''");
                        let ps =
                            format!("$p=New-Object Media.SoundPlayer '{}';$p.Play();", escaped);
                        let mut pscmd = TokioCommand::new("powershell");
                        // Detach from TTY so player doesn't consume keyboard (Esc) input
                        pscmd
                            .stdin(Stdio::null())
                            .stdout(Stdio::null())
                            .stderr(Stdio::null());
                        match pscmd.arg("-NoProfile").arg("-Command").arg(ps).spawn() {
                            Ok(mut child) => {
                                if let Some(child_id) = child.id() {
                                    launched_with = Some("powershell".to_string());
                                    self.current_playback_pid = Some(child_id);

                                    // Create channel for playback completion notification
                                    let (finished_tx, finished_rx) = mpsc::channel(1);
                                    self.playback_finished_rx = Some(finished_rx);

                                    tokio::spawn(async move {
                                        let _ = child.wait().await;
                                        let _ = finished_tx.send(()).await;
                                    });
                                }
                            }
                            Err(_e) => {}
                        }
                    }
                }

                if let Some(player) = launched_with {
                    let name = recording
                        .display_name
                        .as_ref()
                        .unwrap_or(&recording.directory_name);
                    self.message = format!(
                        "\u{25B6} Playing: {}\nUsing player: {}\n\nPress ESC to stop playback",
                        name, player
                    );
                    self.show_message = true;
                    return Ok(());
                }

                // If we reach here, no player succeeded
                #[cfg(target_os = "macos")]
                let hint = "Install `mpv` (brew install mpv) or ensure `afplay` is available.";
                #[cfg(all(unix, not(target_os = "macos")))]
                let hint = "Install `mpv` or `ffmpeg` (ffplay).";
                #[cfg(target_os = "windows")]
                let hint = "Ensure PowerShell is available or install a player like mpv.";

                Err(anyhow!("No audio player found on PATH. {}", hint))
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    pub(super) fn find_audio_file(&self, recording: &Recording) -> Option<PathBuf> {
        let base_path = home_dir()?
            .join("scriba_recordings")
            .join(&recording.directory_name);
        if !base_path.exists() {
            return None;
        }
        let exts = [
            "wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "aiff", "aif", "caf",
        ];
        if let Ok(read_dir) = std::fs::read_dir(base_path) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if exts.iter().any(|x| x.eq_ignore_ascii_case(ext)) {
                        return Some(path);
                    }
                }
            }
        }
        None
    }

    pub(super) fn get_current_recording(&self) -> Option<Recording> {
        let selected_index = self.table_state.selected()?;
        self.recordings.get(selected_index).cloned()
    }

    pub(super) fn show_delete_confirmation(&mut self) {
        if let Some(selected) = self.table_state.selected() {
            if let Some(recording) = self.recordings.get(selected).cloned() {
                self.delete_candidate = Some(recording);
                self.delete_confirm_selection = 1; // default to No
                self.show_delete_confirm = true;
            }
        }
    }

    pub(super) async fn perform_delete_recording(&mut self, recording: Recording) -> Result<()> {
        if let Some(id) = recording.id {
            match self.db.delete_recording(id) {
                Ok(()) => {
                    let base_path = home_dir()
                        .context("Could not find home directory")?
                        .join("scriba_recordings");
                    let recording_dir = base_path.join(&recording.directory_name);

                    if recording_dir.exists() {
                        if let Err(e) = std::fs::remove_dir_all(&recording_dir) {
                            self.notification_message = Some((format!("DB row deleted but files not removed: {}", e), 60));
                        }
                    }

                    self.load_recordings()?;
                    self.load_stats()?;
                    self.message = "Recording deleted.".to_string();
                    self.show_message = true;
                }
                Err(_e) => {
                    return Err(anyhow::anyhow!(
                        "Could not delete recording (ID: {}).\nHint: This often happens when there are related rows (e.g., transcripts) without ON DELETE CASCADE. Delete dependents first or enable cascading, then retry.",
                        id
                    ));
                }
            }
        } else {
            return Err(anyhow::anyhow!(
                "Selected recording has no database ID; cannot delete."
            ));
        }
        Ok(())
    }

    pub(super) async fn execute_add_external_file(&mut self) -> Result<()> {
        // Check if already recording (transcription can run concurrently)
        if self.recording_task.is_some() {
            self.message = "Recording already in progress".to_string();
            self.show_message = true;
            return Ok(());
        }

        // Show file dialog for importing audio file
        self.show_file_dialog = true;
        self.file_dialog_stage = FileDialogStage::FilePath;
        self.file_path_input.clear();
        self.file_name_input.clear();

        Ok(())
    }

    pub(super) fn load_recordings(&mut self) -> Result<()> {
        let offset = (self.current_page * self.page_size) as i64;

        self.recordings = if self.search_query.is_empty() {
            self.db
                .list_recordings(Some(self.page_size as i64), Some(offset))?
        } else {
            let search_results = self.db.search_transcripts(&self.search_query, None)?;
            search_results
                .into_iter()
                .map(|(recording, _)| recording)
                .collect()
        };

        if !self.recordings.is_empty() {
            self.table_state.select(Some(0));
        } else {
            self.table_state.select(None);
        }

        // Refresh greeting with updated recording data
        if !self.owner_name.is_empty() {
            self.generate_greeting();
        }

        Ok(())
    }

    pub(super) fn load_stats(&mut self) -> Result<()> {
        self.stats = Some(self.db.get_stats()?);
        Ok(())
    }

    pub(super) fn render_browse_view(&mut self, f: &mut Frame) {
        let size = f.size();

        if self.recording_task.is_some() {
            self.render_recording_view(f, size);
            return;
        }

        if self.show_file_dialog {
            self.render_file_dialog_popup(f, size);
            return;
        }

        if self.show_message {
            self.render_message_popup(f, size);
            return;
        }

        if self.show_transcript {
            self.render_transcript_popup(f, size);
            return;
        }

        if self.show_delete_confirm {
            self.render_delete_confirmation_popup(f, size);
            return;
        }

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(6),   // Recording list
                Constraint::Length(2), // Footer (separator + line)
            ])
            .split(size);

        self.render_recording_list(f, chunks[0]);
        self.render_browse_footer(f, chunks[1]);

        if self.search_mode {
            self.render_search_input(f, size);
        }
    }

    pub(super) fn render_browse_footer(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        // Notification override
        if let Some((ref msg, _)) = self.notification_message {
            let is_error = msg.contains("failed") || msg.contains("Failed");
            let style = if is_error {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            };
            let para = Paragraph::new(msg.as_str())
                .style(style)
                .alignment(Alignment::Center);
            f.render_widget(para, area);
            return;
        }

        // Separator line
        let sep = "\u{2500}".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: area.x, y: area.y, width: area.width, height: 1 },
        );

        // Footer content area (below separator)
        let footer_area = if area.height > 1 {
            Rect { x: area.x, y: area.y + 1, width: area.width, height: area.height - 1 }
        } else {
            area
        };

        // Left side: ▸ scriba · v0.21.2 · N recordings · Xh Ym
        let version = env!("CARGO_PKG_VERSION");
        let rec_count = self.stats.as_ref().map(|s| s.total_recordings).unwrap_or(0);
        let total_secs = self.stats.as_ref().map(|s| s.total_duration_seconds).unwrap_or(0);
        let total_h = total_secs / 3600;
        let total_m = (total_secs % 3600) / 60;
        let duration_str = if total_h > 0 {
            format!("{}h {}m", total_h, total_m)
        } else {
            format!("{}m", total_m)
        };

        let mut left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(
                format!("scriba \u{00B7} v{} \u{00B7} {} recordings \u{00B7} {}", version, rec_count, duration_str),
                Style::default().fg(Color::DarkGray),
            ),
        ];

        // Transcribing indicator
        if self.active_transcription.is_some() {
            left_spans.push(Span::styled(
                " \u{00B7} Transcribing...",
                Style::default().fg(Color::Yellow),
            ));
        }

        // Right side: shortcuts
        let right_spans = vec![
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Tab", Style::default().fg(Color::White)),
            Span::styled("] Home  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("Ctrl+R", Style::default().fg(Color::White)),
            Span::styled("] Record  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled("/", Style::default().fg(Color::White)),
            Span::styled("] Search ", Style::default().fg(Color::DarkGray)),
        ];

        // Compute widths to fill gap
        let left_width: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_width: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_area.width as usize).saturating_sub(left_width + right_width);

        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);

        let line = Line::from(spans);
        f.render_widget(Paragraph::new(line), footer_area);
    }


    pub(super) fn render_recording_list(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        if self.recordings.is_empty() {
            let msg = if self.search_query.is_empty() {
                "No recordings yet. Press Ctrl+R to start recording."
            } else {
                "No recordings match your search."
            };
            let para = Paragraph::new(msg)
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            f.render_widget(para, area);
            return;
        }

        let selected_idx = self.table_state.selected().unwrap_or(0);
        let today = chrono::Local::now().date_naive();
        let yesterday = today - chrono::Duration::days(1);

        // Build lines and track which line maps to which recording index
        let mut lines: Vec<Line> = Vec::new();
        let mut line_to_recording: Vec<Option<usize>> = Vec::new();
        let mut current_group: Option<String> = None;

        let name_col_width = (area.width as usize).saturating_sub(16); // leave room for prefix + duration + time

        for (i, recording) in self.recordings.iter().enumerate() {
            let rec_date = recording.created_at.with_timezone(&chrono::Local).date_naive();
            let group_label = if rec_date == today {
                "Today".to_string()
            } else if rec_date == yesterday {
                "Yesterday".to_string()
            } else {
                rec_date.format("%b %-d").to_string()
            };

            if current_group.as_ref() != Some(&group_label) {
                // Blank line before group (except first)
                if current_group.is_some() {
                    lines.push(Line::from(""));
                    line_to_recording.push(None);
                }
                // Date header
                lines.push(Line::from(Span::styled(
                    format!("  {}", group_label),
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
                )));
                line_to_recording.push(None);
                current_group = Some(group_label);
            }

            let is_selected = i == selected_idx;
            let is_active = self.active_transcription.as_ref()
                .is_some_and(|a| a.recording_name == recording.directory_name);
            let is_queued = !is_active
                && self.transcription_queue.iter().any(|p| p.recording_name() == recording.directory_name);

            // Selection marker
            let marker = if is_selected { "\u{25B8} " } else { "  " };
            let marker_style = Style::default().fg(Color::White);

            // Status dot
            let (dot, dot_style) = if is_active {
                let spinner = match self.progress_frame % 4 {
                    0 => "\u{25D0}",
                    1 => "\u{25D3}",
                    2 => "\u{25D1}",
                    _ => "\u{25D2}",
                };
                (spinner, Style::default().fg(Color::Yellow))
            } else if is_queued {
                ("\u{25CB}", Style::default().fg(Color::Yellow))
            } else if recording.has_transcript {
                ("\u{25CF}", Style::default().fg(ACCENT))
            } else {
                ("\u{25CB}", Style::default().fg(Color::DarkGray))
            };

            // Name
            let display_name = recording.display_name.as_ref()
                .unwrap_or(&recording.directory_name);
            let name_style = if is_selected {
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Indexed(249))
            };

            // Duration (short format)
            let duration = recording.duration_seconds
                .map(|d| {
                    let mins = d / 60;
                    if mins >= 60 {
                        format!("{}h{}m", mins / 60, mins % 60)
                    } else {
                        format!("{}m", mins)
                    }
                })
                .unwrap_or_default();

            // Time
            let time = recording.created_at
                .with_timezone(&chrono::Local)
                .format("%H:%M")
                .to_string();

            // Truncate name to fit
            let right_part = format!("{:>6}  {}", duration, time);
            let max_name = name_col_width.saturating_sub(right_part.len() + 6); // 6 = "  ▸ ● "
            let truncated_name: String = if display_name.chars().count() > max_name {
                display_name.chars().take(max_name.saturating_sub(1)).collect::<String>() + "\u{2026}"
            } else {
                display_name.clone()
            };
            let name_padding = max_name.saturating_sub(truncated_name.chars().count());

            let spans = vec![
                Span::styled(format!("  {}", marker), marker_style),
                Span::styled(format!("{} ", dot), dot_style),
                Span::styled(truncated_name, name_style),
                Span::raw(" ".repeat(name_padding)),
                Span::styled(right_part, Style::default().fg(Color::DarkGray)),
            ];

            lines.push(Line::from(spans));
            line_to_recording.push(Some(i));
        }

        // Find the line index for the selected recording to keep it in view
        let selected_line = line_to_recording.iter()
            .position(|r| *r == Some(selected_idx))
            .unwrap_or(0);

        // Calculate scroll offset
        let visible_height = area.height as usize;
        let scroll = if selected_line >= visible_height {
            selected_line - visible_height + 2 // keep 1 line below selected visible
        } else {
            0
        };

        let para = Paragraph::new(lines).scroll((scroll as u16, 0));
        f.render_widget(para, area);
    }

    pub(super) fn render_search_input(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(40),
                Constraint::Length(3),
                Constraint::Percentage(40),
            ])
            .split(area)[1];

        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(20),
                Constraint::Percentage(60),
                Constraint::Percentage(20),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let search_input = Paragraph::new(format!("Search: {}", self.search_query))
            .style(Style::default().fg(Color::Yellow))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(ACCENT))
                    .title("Search Recordings"),
            );

        f.render_widget(search_input, popup_area);
    }

    pub(super) fn render_delete_confirmation_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let recording_name = if let Some(recording) = &self.delete_candidate {
            recording
                .display_name
                .as_ref()
                .unwrap_or(&recording.directory_name)
                .clone()
        } else {
            "?".to_string()
        };

        let sel = self.delete_confirm_selection;
        let sel_bg = Color::Indexed(236);
        let labels = ["Yes, delete", "No, keep it"];
        let label_width = labels.iter().map(|l| l.chars().count() + 2).max().unwrap_or(0);

        let mut lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled("Delete recording: ", Style::default().fg(Color::White)),
                Span::styled(&recording_name, Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::styled("?", Style::default().fg(Color::White)),
            ]),
            Line::from(""),
        ];
        for (i, label) in labels.iter().enumerate() {
            if sel == i {
                let pad = " ".repeat(label_width.saturating_sub(label.chars().count() + 2));
                lines.push(Line::from(vec![
                    Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                    Span::styled(format!("{}{}", label, pad), Style::default().fg(Color::White).bg(sel_bg)),
                ]));
            } else {
                lines.push(Line::from(Span::styled(format!("  {}", label), Style::default().fg(Color::DarkGray))));
            }
        }

        let max_w = lines.iter().map(|l| l.width() as u16).max().unwrap_or(0).max(30);
        let content_width = max_w.min(area.width);
        let left_pad = (area.width.saturating_sub(content_width)) / 2;
        let line_count = lines.len();
        let top_pad = (area.height as usize).saturating_sub(line_count) / 2;
        let mut centered: Vec<Line> = Vec::with_capacity(top_pad + line_count);
        for _ in 0..top_pad { centered.push(Line::from("")); }
        centered.extend(lines);

        f.render_widget(Clear, area);
        let body = Rect { x: area.x + left_pad, width: content_width, ..area };
        f.render_widget(Paragraph::new(centered).alignment(Alignment::Left), body);
    }

    pub(super) fn render_file_dialog_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Length(12),
                Constraint::Percentage(65),
            ])
            .split(area)[1];

        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let (title, prompt, current_input, hint) = match self.file_dialog_stage {
            FileDialogStage::FilePath => (
                "Import Audio File - Step 1/2",
                "Enter the full path to the audio file:",
                &self.file_path_input,
                "Example: /path/to/your/audio.mp3 or ~/Downloads/recording.wav",
            ),
            FileDialogStage::FileName => (
                "Import Audio File - Step 2/2",
                "Enter a display name for this recording:",
                &self.file_name_input,
                "This name will be shown in your recordings list",
            ),
        };

        let content = vec![
            Line::from(vec![Span::styled(
                prompt,
                Style::default().fg(Color::White),
            )]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Input: ", Style::default().fg(Color::Green)),
                Span::styled(
                    format!("{}_", current_input),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(hint, Style::default().fg(Color::Gray))]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Press ENTER to continue, ESC to cancel",
                Style::default().fg(Color::Blue),
            )]),
        ];

        let dialog_paragraph = Paragraph::new(content)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title)
                    .title_style(
                        Style::default()
                            .fg(ACCENT)
                            .add_modifier(Modifier::BOLD),
                    )
                    .border_style(Style::default().fg(ACCENT)),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(dialog_paragraph, popup_area);
    }

    pub(super) fn render_message_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(30),
                Constraint::Length(5),
                Constraint::Percentage(65),
            ])
            .split(area)[1];

        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(20),
                Constraint::Percentage(60),
                Constraint::Percentage(20),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let para = Paragraph::new(self.message.clone())
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Red))
                    .title("Message (press Esc)"),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(para, popup_area);
    }
}
